/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

/**
 * @file omatcopy2_kernels.hpp : portable SYCL kernels implementing the
 * omatcopy2 element strides, which vendor geam entry points cannot express.
 */
#ifndef _ONEMATH_BLAS_OMATCOPY2_KERNELS_HPP_
#define _ONEMATH_BLAS_OMATCOPY2_KERNELS_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "oneapi/math/types.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace omatcopy2_kernels {

// Geometry of the local-memory tile that stages the transpose. Without the
// tile one of the two global accesses would step by the leading dimension;
// with it both step by the (small) element stride instead.
//
// The tile is `rows` x `cols` elements and the work-group is `block` x `rows`
// items: the load phase puts the lane index on the row extent and the store
// phase puts it on the column extent, so `cols` sets how wide the stores are.
//
// Every variant below keeps the tile under 17 KB except the gfx90a
// complex<double> strided tile, which uses 32.5 KB. CDNA1-CDNA3 have 64 KB of
// local memory per compute unit; CDNA4 has 160 KB and A100 has 164 KB.

/// The two backends that include this header are each built against a single
/// vendor's runtime. gfx90a and gfx942 use architecture-specific tuning selected
/// at run time within the AMD backend; other AMD architectures use gfx950
/// tuning.
enum class target { nvidia, amd, amd_gfx90a, amd_gfx942 };

inline target get_amd_target(const sycl::device& device) {
    static thread_local std::vector<std::pair<sycl::device, target>> cache;
    for (const auto& cached : cache) {
        if (cached.first == device) {
            return cached.second;
        }
    }

    target result = target::amd;
#ifdef SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE
    // This experimental query is non-const in older DPC++ releases.
    auto query_device = device;
    using architecture = sycl::ext::oneapi::experimental::architecture;
    if (query_device.ext_oneapi_architecture_is(architecture::amd_gpu_gfx90a)) {
        result = target::amd_gfx90a;
    }
    else if (query_device.ext_oneapi_architecture_is(architecture::amd_gpu_gfx942)) {
        result = target::amd_gfx942;
    }
#endif
    if (result == target::amd) {
        // The DPC++ HIP adapter currently reports an unknown architecture
        // through the extension above, and AdaptiveCpp does not expose it.
        const auto name = device.get_info<sycl::info::device::name>();
        if (name.find("MI210") != std::string::npos || name.find("MI250") != std::string::npos ||
            name.find("gfx90a") != std::string::npos) {
            result = target::amd_gfx90a;
        }
        else if (name.find("MI300") != std::string::npos ||
                 name.find("MI308") != std::string::npos ||
                 name.find("MI325") != std::string::npos ||
                 name.find("gfx942") != std::string::npos) {
            result = target::amd_gfx942;
        }
    }
    cache.emplace_back(device, result);
    return result;
}

/// Unit element strides: both global accesses are contiguous, so the two
/// phases want the same width and a square tile is best.
///
/// Measured on gfx950 and on an A100, which agree to within 0.4% on every
/// element type. gfx942 retains those choices except for 16-byte elements,
/// where halving the column extent gains about 2.4%.
template <typename T, target Target>
struct unit_stride_geometry {
    static constexpr int rows = sizeof(T) <= 4 ? 64 : 32;
    static constexpr int cols = Target == target::amd_gfx942 && sizeof(T) == 16 ? 16 : rows;
    static constexpr int block = 8;
};

/// Real element strides: neither access is contiguous, so widening the stores
/// buys nothing and a taller tile amortises the per-tile overhead instead.
///
/// The vendors disagree, and gfx90a and gfx942 also require targeted
/// architecture-specific choices, so these tables are separate.
template <typename T, target Target>
struct strided_geometry;

/// Measured on gfx950.
template <typename T>
struct strided_geometry<T, target::amd> {
    static constexpr int rows = sizeof(T) <= 8 ? 64 : 32;
    static constexpr int cols = 32;
    static constexpr int block = sizeof(T) == 8 ? 16 : 8;
};

/// Measured on two MI210 devices. Sixteen-byte elements prefer a 64 x 32 tile
/// and a 256-item work-group, gaining about 8.5% over the gfx950 choice. Other
/// element types retain the gfx950 geometry.
template <typename T>
struct strided_geometry<T, target::amd_gfx90a> {
    static constexpr int rows = 64;
    static constexpr int cols = 32;
    static constexpr int block = sizeof(T) == 8 ? 16 : (sizeof(T) == 16 ? 4 : 8);
};

/// Measured on gfx942. Eight-byte elements prefer a 32 x 8 tile and a
/// 256-item work-group; the gfx950 choice uses 1024 items. Float and 16-byte
/// elements retain the gfx950 geometry.
template <typename T>
struct strided_geometry<T, target::amd_gfx942> {
    static constexpr int rows = sizeof(T) <= 4 ? 64 : 32;
    static constexpr int cols = sizeof(T) == 8 ? 8 : 32;
    static constexpr int block = 8;
};

/// Measured on an A100. The wave64 table wants 1024-item groups for the 8-byte
/// types, which is the whole of an SM's thread budget on NVIDIA and leaves only
/// two groups resident; 256 items instead is worth 4% on `double` and 2% on
/// `complex<float>` overall, and 7% and 5% once the matrices fit in L2, where
/// the copy is not already bounded by main memory. The narrower tile for
/// 16-byte types costs 0.5% and keeps every variant inside the same 17 KB.
template <typename T>
struct strided_geometry<T, target::nvidia> {
    static constexpr int rows = 64;
    static constexpr int cols = sizeof(T) <= 8 ? 32 : 16;
    static constexpr int block = 4;
};

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

/// b[r * strideb + c * ldb] = alpha * a[r * stridea + c * lda]
///
/// The fastest-moving index is r, so both accesses run along the element
/// stride and stay as coalesced as the strides allow.
template <typename T, typename AccessorA, typename AccessorB>
void launch_nontrans(sycl::handler& cgh, int64_t logical_m, int64_t logical_n, T alpha, AccessorA a,
                     int64_t lda, int64_t stridea, AccessorB b, int64_t ldb, int64_t strideb) {
    cgh.parallel_for(sycl::range<2>(static_cast<size_t>(logical_n), static_cast<size_t>(logical_m)),
                     [=](sycl::id<2> id) {
                         const int64_t c = static_cast<int64_t>(id[0]);
                         const int64_t r = static_cast<int64_t>(id[1]);
                         b[r * strideb + c * ldb] = alpha * a[r * stridea + c * lda];
                     });
}

/// b[c * strideb + r * ldb] = alpha * op(a[r * stridea + c * lda])
///
/// Loads run along r and stores run along c, both with the lane index as the
/// fastest dimension, so neither global access steps by a leading dimension.
template <typename T, typename Geom, typename AccessorA, typename AccessorB>
void launch_trans(sycl::handler& cgh, int64_t logical_m, int64_t logical_n, T alpha, bool do_conj,
                  AccessorA a, int64_t lda, int64_t stridea, AccessorB b, int64_t ldb,
                  int64_t strideb) {
    // Odd padding keeps the transposed local reads off a single bank.
    constexpr int pitch = Geom::rows + 1;

    static_assert(Geom::cols <= Geom::rows, "the group lane count comes from the row extent");
    static_assert(Geom::rows % Geom::block == 0 && Geom::cols % Geom::block == 0,
                  "both tile extents must be covered by whole steps of block");

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

        // Stores use only the first `cols` lanes; the rest of the group idles.
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
}

/// Picks the tile shape from the strides, which decide whether the global
/// accesses are contiguous.
template <typename T, target Target, typename AccessorA, typename AccessorB>
void launch_trans_dispatch(sycl::handler& cgh, int64_t logical_m, int64_t logical_n, T alpha,
                           bool do_conj, AccessorA a, int64_t lda, int64_t stridea, AccessorB b,
                           int64_t ldb, int64_t strideb) {
    if (stridea == 1 && strideb == 1) {
        launch_trans<T, unit_stride_geometry<T, Target>>(cgh, logical_m, logical_n, alpha, do_conj,
                                                         a, lda, stridea, b, ldb, strideb);
    }
    else {
        launch_trans<T, strided_geometry<T, Target>>(cgh, logical_m, logical_n, alpha, do_conj, a,
                                                     lda, stridea, b, ldb, strideb);
    }
}

template <typename T, target Target, typename AccessorA, typename AccessorB>
void launch_trans_for_device(sycl::handler& cgh, target device_target, int64_t logical_m,
                             int64_t logical_n, T alpha, bool do_conj, AccessorA a, int64_t lda,
                             int64_t stridea, AccessorB b, int64_t ldb, int64_t strideb) {
    if constexpr (Target == target::amd) {
        if (device_target == target::amd_gfx90a) {
            launch_trans_dispatch<T, target::amd_gfx90a>(cgh, logical_m, logical_n, alpha, do_conj,
                                                         a, lda, stridea, b, ldb, strideb);
            return;
        }
        if (device_target == target::amd_gfx942) {
            launch_trans_dispatch<T, target::amd_gfx942>(cgh, logical_m, logical_n, alpha, do_conj,
                                                         a, lda, stridea, b, ldb, strideb);
            return;
        }
    }
    launch_trans_dispatch<T, Target>(cgh, logical_m, logical_n, alpha, do_conj, a, lda, stridea, b,
                                     ldb, strideb);
}

template <target Target, typename T>
sycl::event omatcopy2_usm(sycl::queue& queue, oneapi::math::layout layout,
                          oneapi::math::transpose trans, int64_t m, int64_t n, T alpha, const T* a,
                          int64_t lda, int64_t stridea, T* b, int64_t ldb, int64_t strideb,
                          const std::vector<sycl::event>& dependencies) {
    const bool col_major = (layout == oneapi::math::layout::col_major);
    const int64_t logical_m = col_major ? m : n;
    const int64_t logical_n = col_major ? n : m;
    const bool do_trans = (trans != oneapi::math::transpose::nontrans);
    const bool do_conj = (trans == oneapi::math::transpose::conjtrans);

    if (logical_m <= 0 || logical_n <= 0) {
        return queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(dependencies);
            cgh.single_task([=]() {});
        });
    }

    const target device_target =
        do_trans && Target == target::amd ? get_amd_target(queue.get_device()) : Target;
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        if (do_trans) {
            launch_trans_for_device<T, Target>(cgh, device_target, logical_m, logical_n, alpha,
                                               do_conj, a, lda, stridea, b, ldb, strideb);
        }
        else {
            launch_nontrans<T>(cgh, logical_m, logical_n, alpha, a, lda, stridea, b, ldb, strideb);
        }
    });
}

template <target Target, typename T>
void omatcopy2_buffer(sycl::queue& queue, oneapi::math::layout layout,
                      oneapi::math::transpose trans, int64_t m, int64_t n, T alpha,
                      sycl::buffer<T, 1>& a, int64_t lda, int64_t stridea, sycl::buffer<T, 1>& b,
                      int64_t ldb, int64_t strideb) {
    const bool col_major = (layout == oneapi::math::layout::col_major);
    const int64_t logical_m = col_major ? m : n;
    const int64_t logical_n = col_major ? n : m;
    const bool do_trans = (trans != oneapi::math::transpose::nontrans);
    const bool do_conj = (trans == oneapi::math::transpose::conjtrans);

    if (logical_m <= 0 || logical_n <= 0) {
        return;
    }

    const target device_target =
        do_trans && Target == target::amd ? get_amd_target(queue.get_device()) : Target;
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        // Strided writes skip elements, so the untouched ones must be preserved.
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        if (do_trans) {
            launch_trans_for_device<T, Target>(cgh, device_target, logical_m, logical_n, alpha,
                                               do_conj, a_acc, lda, stridea, b_acc, ldb, strideb);
        }
        else {
            launch_nontrans<T>(cgh, logical_m, logical_n, alpha, a_acc, lda, stridea, b_acc, ldb,
                               strideb);
        }
    });
}

} // namespace omatcopy2_kernels
} // namespace blas
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_BLAS_OMATCOPY2_KERNELS_HPP_
