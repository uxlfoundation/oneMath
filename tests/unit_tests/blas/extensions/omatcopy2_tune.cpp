/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

// Geometry sweep for the omatcopy2 transpose kernel. This file is not part of
// the unit-test binary; compile it by hand when retuning tile extents:
//
//   clang++ -O3 -fsycl omatcopy2_tune.cpp -o omatcopy2_tune
//
// Optional first argument: float, double, cfloat, or cdouble. Emits CSV:
// type,rows,cols,block,m,n,stridea,strideb,gbps
//
// The kernel body is a verbatim copy of launch_trans() from
// src/blas/backends/omatcopy2_kernels.hpp, with the tile extents lifted into
// template parameters so a range of geometries can be measured. Keep the two
// in sync if the kernel changes.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline T conj_if(const T& value, bool do_conj) {
    if constexpr (is_complex<T>::value) {
        return do_conj ? T(value.real(), -value.imag()) : value;
    }
    else {
        (void)do_conj;
        return value;
    }
}

template <int ROWS, int COLS, int BLOCK>
struct geometry {
    static constexpr int rows = ROWS;
    static constexpr int cols = COLS;
    static constexpr int block = BLOCK;
};

template <typename T, typename Geom>
sycl::event launch_trans(sycl::queue& queue, int64_t logical_m, int64_t logical_n, T alpha,
                         bool do_conj, const T* a, int64_t lda, int64_t stridea, T* b, int64_t ldb,
                         int64_t strideb) {
    constexpr int pitch = Geom::rows + 1;

    static_assert(Geom::cols <= Geom::rows, "the group lane count comes from the row extent");
    static_assert(Geom::rows % Geom::block == 0 && Geom::cols % Geom::block == 0,
                  "both tile extents must be covered by whole steps of block");

    return queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> tile(sycl::range<1>(Geom::cols * pitch), cgh);

        const int64_t tiles_r = (logical_m + Geom::rows - 1) / Geom::rows;
        const int64_t tiles_c = (logical_n + Geom::cols - 1) / Geom::cols;

        const sycl::range<2> global(static_cast<size_t>(tiles_c) * Geom::block,
                                    static_cast<size_t>(tiles_r) * Geom::rows);
        const sycl::range<2> local(Geom::block, Geom::rows);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            const int ly = static_cast<int>(item.get_local_id(0));
            const int lx = static_cast<int>(item.get_local_id(1));
            const int64_t tile_r = static_cast<int64_t>(item.get_group(1)) * Geom::rows;
            const int64_t tile_c = static_cast<int64_t>(item.get_group(0)) * Geom::cols;

            const int64_t load_r = tile_r + lx;
            if (load_r < logical_m) {
                for (int k = 0; k < Geom::cols; k += Geom::block) {
                    const int cl = ly + k;
                    const int64_t c = tile_c + cl;
                    if (c < logical_n) {
                        tile[cl * pitch + lx] = a[load_r * stridea + c * lda];
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            const int64_t store_c = tile_c + lx;
            if (lx < Geom::cols && store_c < logical_n) {
                for (int k = 0; k < Geom::rows; k += Geom::block) {
                    const int rl = ly + k;
                    const int64_t r = tile_r + rl;
                    if (r < logical_m) {
                        b[store_c * strideb + r * ldb] =
                            alpha * conj_if(tile[lx * pitch + rl], do_conj);
                    }
                }
            }
        });
    });
}

template <typename T>
T make_value(int64_t i) {
    if constexpr (is_complex<T>::value) {
        using R = typename T::value_type;
        return T(static_cast<R>(i % 97), static_cast<R>(i % 31));
    }
    else {
        return static_cast<T>(i % 97);
    }
}

template <typename T>
struct type_name;
template <>
struct type_name<float> {
    static constexpr const char* value = "float";
};
template <>
struct type_name<double> {
    static constexpr const char* value = "double";
};
template <>
struct type_name<std::complex<float>> {
    static constexpr const char* value = "complex<float>";
};
template <>
struct type_name<std::complex<double>> {
    static constexpr const char* value = "complex<double>";
};

struct problem {
    int64_t m, n, lda, stridea, ldb, strideb;
    int64_t size_a, size_b;
};

problem make_problem(int64_t m, int64_t n, int64_t stridea, int64_t strideb) {
    problem p;
    p.m = m;
    p.n = n;
    p.stridea = stridea;
    p.strideb = strideb;
    p.lda = m * stridea;
    p.ldb = n * strideb;
    p.size_a = (n - 1) * p.lda + (m - 1) * stridea + 1;
    p.size_b = (m - 1) * p.ldb + (n - 1) * strideb + 1;
    return p;
}

template <typename T, typename Geom>
bool verify(sycl::queue& queue) {
    const problem p = make_problem(Geom::rows + 5, Geom::cols + 3, 2, 3);
    std::vector<T> ha(p.size_a), hb(p.size_b), ref;
    for (int64_t i = 0; i < p.size_a; ++i) {
        ha[i] = make_value<T>(i);
    }
    for (int64_t i = 0; i < p.size_b; ++i) {
        hb[i] = make_value<T>(i + 7);
    }
    ref = hb;

    const T alpha = static_cast<T>(2);
    for (int64_t c = 0; c < p.n; ++c) {
        for (int64_t r = 0; r < p.m; ++r) {
            ref[c * p.strideb + r * p.ldb] = alpha * conj_if(ha[r * p.stridea + c * p.lda], true);
        }
    }

    T* da = sycl::malloc_device<T>(p.size_a, queue);
    T* db = sycl::malloc_device<T>(p.size_b, queue);
    queue.copy(ha.data(), da, p.size_a).wait();
    queue.copy(hb.data(), db, p.size_b).wait();
    launch_trans<T, Geom>(queue, p.m, p.n, alpha, true, da, p.lda, p.stridea, db, p.ldb, p.strideb)
        .wait();
    queue.copy(db, hb.data(), p.size_b).wait();
    sycl::free(da, queue);
    sycl::free(db, queue);

    for (int64_t i = 0; i < p.size_b; ++i) {
        if (std::abs(hb[i] - ref[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

constexpr double kBlockSeconds = 0.02;
constexpr int kBlocks = 3;

template <typename T, typename Geom>
double bench(sycl::queue& queue, const problem& p, T* da, T* db) {
    const T alpha = static_cast<T>(1);
    auto timed = [&](int iters) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            launch_trans<T, Geom>(queue, p.m, p.n, alpha, false, da, p.lda, p.stridea, db, p.ldb,
                                  p.strideb);
        }
        queue.wait();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    };

    const double probe = timed(3) / 3.0;
    const int iters = std::max(5, std::min(2000, static_cast<int>(kBlockSeconds / probe)));

    std::vector<double> samples;
    for (int b = 0; b < kBlocks; ++b) {
        samples.push_back(timed(iters) / iters);
    }
    std::sort(samples.begin(), samples.end());
    const double best_seconds = samples[samples.size() / 2];

    const double bytes = 2.0 * static_cast<double>(p.m) * static_cast<double>(p.n) * sizeof(T);
    return bytes / best_seconds * 1e-9;
}

template <typename T, typename Geom>
void run_one(sycl::queue& queue, const problem& p, T* da, T* db, bool& verified, bool& ok) {
    constexpr size_t tile_bytes = static_cast<size_t>(Geom::cols) * (Geom::rows + 1) * sizeof(T);
    constexpr size_t group_size = static_cast<size_t>(Geom::rows) * Geom::block;
    const auto local_mem = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    const auto max_wg = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    if constexpr (Geom::cols <= Geom::rows && Geom::rows % Geom::block == 0 &&
                  Geom::cols % Geom::block == 0) {
        if (tile_bytes > local_mem || group_size > max_wg) {
            return;
        }
        if (!verified) {
            verified = true;
            ok = verify<T, Geom>(queue);
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL %s %dx%dx%d\n", type_name<T>::value, Geom::rows, Geom::cols,
                         Geom::block);
            return;
        }
        const double gbps = bench<T, Geom>(queue, p, da, db);
        std::printf("%s,%d,%d,%d,%lld,%lld,%lld,%lld,%.2f\n", type_name<T>::value, Geom::rows,
                    Geom::cols, Geom::block, (long long)p.m, (long long)p.n, (long long)p.stridea,
                    (long long)p.strideb, gbps);
        std::fflush(stdout);
    }
}

#define FOR_EACH_GEOMETRY(F)                                                                    \
    F(16, 16, 4)                                                                                \
    F(16, 16, 8)                                                                                \
    F(16, 16, 16) F(32, 8, 8) F(32, 16, 4) F(32, 16, 8) F(32, 16, 16) F(32, 32, 4) F(32, 32, 8) \
        F(32, 32, 16) F(32, 32, 32) F(64, 16, 4) F(64, 16, 8) F(64, 16, 16) F(64, 32, 4)        \
            F(64, 32, 8) F(64, 32, 16) F(64, 64, 4) F(64, 64, 8) F(64, 64, 16) F(128, 16, 4)    \
                F(128, 16, 8) F(128, 32, 4) F(128, 32, 8) F(128, 64, 4) F(128, 64, 8)           \
                    F(128, 128, 4) F(128, 128, 8)

template <typename T>
void sweep_type(sycl::queue& queue, const std::vector<problem>& problems) {
    for (const problem& p : problems) {
        T* da = sycl::malloc_device<T>(p.size_a, queue);
        T* db = sycl::malloc_device<T>(p.size_b, queue);
        if (da == nullptr || db == nullptr) {
            std::fprintf(stderr, "alloc failed for %s %lldx%lld stride %lld/%lld\n",
                         type_name<T>::value, (long long)p.m, (long long)p.n, (long long)p.stridea,
                         (long long)p.strideb);
            sycl::free(da, queue);
            sycl::free(db, queue);
            continue;
        }
        queue.fill(da, static_cast<T>(1), p.size_a).wait();
        queue.fill(db, static_cast<T>(0), p.size_b).wait();

#define RUN(R, C, B)                                                   \
    {                                                                  \
        static bool verified = false;                                  \
        static bool ok = true;                                         \
        run_one<T, geometry<R, C, B>>(queue, p, da, db, verified, ok); \
    }
        FOR_EACH_GEOMETRY(RUN)
#undef RUN

        sycl::free(da, queue);
        sycl::free(db, queue);
    }
}

int main(int argc, char** argv) {
    sycl::queue queue(sycl::gpu_selector_v);
    std::fprintf(stderr, "device: %s\n",
                 queue.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(
        stderr, "local mem: %llu bytes, max wg: %llu\n",
        (unsigned long long)queue.get_device().get_info<sycl::info::device::local_mem_size>(),
        (unsigned long long)queue.get_device().get_info<sycl::info::device::max_work_group_size>());

    const std::vector<std::pair<int64_t, int64_t>> strides = {
        { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 1, 2 }
    };
    const std::vector<int64_t> sizes = { 1000, 1024, 3000, 4096, 8192 };

    auto build = [&](size_t bytes_per_elem, size_t budget) {
        std::vector<problem> ps;
        for (const auto& s : strides) {
            for (int64_t size : sizes) {
                const problem p = make_problem(size, size, s.first, s.second);
                const size_t footprint =
                    (static_cast<size_t>(p.size_a) + static_cast<size_t>(p.size_b)) *
                    bytes_per_elem;
                if (footprint > budget)
                    continue;
                ps.push_back(p);
            }
        }
        return ps;
    };

    constexpr size_t budget = 24ull << 30;

    std::printf("type,rows,cols,block,m,n,stridea,strideb,gbps\n");
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() || only == "float")
        sweep_type<float>(queue, build(sizeof(float), budget));
    if (only.empty() || only == "double")
        sweep_type<double>(queue, build(sizeof(double), budget));
    if (only.empty() || only == "cfloat")
        sweep_type<std::complex<float>>(queue, build(sizeof(std::complex<float>), budget));
    if (only.empty() || only == "cdouble")
        sweep_type<std::complex<double>>(queue, build(sizeof(std::complex<double>), budget));
    return 0;
}
