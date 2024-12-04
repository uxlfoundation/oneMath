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

#ifndef ONEMATH_TEST_COMMON_HPP
#define ONEMATH_TEST_COMMON_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

template <typename T>
struct complex_info {
    using real_type = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type = T;
    static const bool is_complex = true;
};

template <typename T>
constexpr bool is_complex() {
    return complex_info<T>::is_complex;
}

inline std::size_t cast_unsigned(std::int64_t i) {
    if (i < 0) {
        throw std::runtime_error("Unexpected negative value");
    }
    return static_cast<std::size_t>(i);
}

template <typename fp>
bool check_equal(fp x, fp x_ref, double abs_error_mag, double rel_error_mag, std::ostream& out) {
    using fp_real = typename complex_info<fp>::real_type;
    static_assert(std::is_floating_point_v<fp_real>,
                  "Expected floating-point real or complex type.");

    const fp_real epsilon = []() {
        if constexpr (sizeof(double) == sizeof(long double) && std::is_same_v<fp_real, double>) {
            // The reference DFT uses long double to maintain accuracy
            // when this isn't possible, lower the accuracy requirements
            return 1e-12;
        }
        else {
            return std::numeric_limits<fp_real>::epsilon();
        }
    }();
    const auto abs_bound = static_cast<fp_real>(abs_error_mag) * epsilon;
    const auto rel_bound = static_cast<fp_real>(rel_error_mag) * epsilon;

    const auto aerr = std::abs(x - x_ref);
    const auto rerr = aerr / std::abs(x_ref);
    const bool ok = (rerr <= rel_bound) || (aerr <= abs_bound);
    if (!ok) {
        out << "Mismatching results: actual = " << x << " vs. reference = " << x_ref << "\n";
        out << " relative error = " << rerr << " absolute error = " << aerr
            << " relative bound = " << rel_bound << " absolute bound = " << abs_bound << "\n";
    }
    return ok;
}

template <typename vec1, typename vec2>
bool check_equal_vector(vec1&& v, vec2&& v_ref, std::size_t n, double abs_error_mag,
                        double rel_error_mag, std::ostream& out) {
    constexpr int max_print = 20;
    int count = 0;
    bool good = true;

    for (std::size_t i = 0; i < n; ++i) {
        // Allow to convert the unsigned index `i` to a signed one to keep this function generic and allow for `v` and `v_ref` to be a vector, a pointer or a random access iterator.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
        auto res = v[i];
        auto ref = v_ref[i];
#pragma clang diagnostic pop
        if (!check_equal(res, ref, abs_error_mag, rel_error_mag, out)) {
            out << " at index i =" << i << "\n";
            good = false;
            ++count;
            if (count > max_print) {
                return good;
            }
        }
    }

    return good;
}

// Random initialization.
template <typename t>
inline t rand_scalar() {
    if constexpr (std::is_same_v<t, int32_t>) {
        return std::rand() % 256 - 128;
    }
    else if constexpr (std::is_floating_point_v<t>) {
        return t(std::rand()) / t(RAND_MAX) - t(0.5);
    }
    else {
        static_assert(complex_info<t>::is_complex, "unexpect type in rand_scalar");
        using fp = typename complex_info<t>::real_type;
        return t(rand_scalar<fp>(), rand_scalar<fp>());
    }
}

template <typename vec>
void rand_vector(vec& v, std::size_t n) {
    using fp = typename vec::value_type;
    v.resize(n);
    for (std::size_t i = 0; i < n; i++) {
        v[i] = rand_scalar<fp>();
    }
}

// Catch asynchronous exceptions.
auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
            print_error_code(e);
        }
    }
};

template <oneapi::math::dft::precision precision, oneapi::math::dft::domain domain>
void commit_descriptor(oneapi::math::dft::descriptor<precision, domain>& descriptor,
                       sycl::queue queue) {
#ifdef CALL_RT_API
    descriptor.commit(queue);
#else
    TEST_RUN_CT_SELECT_NO_ARGS(queue, descriptor.commit);
#endif
}

// is it assumed that the unused elements of the array are ignored
inline std::array<std::int64_t, 4> get_conjugate_even_complex_strides(
    const std::vector<std::int64_t>& sizes) {
    switch (sizes.size()) {
        case 1: return { 0, 1 };
        case 2: return { 0, sizes[1] / 2 + 1, 1 };
        case 3: return { 0, sizes[1] * (sizes[2] / 2 + 1), (sizes[2] / 2 + 1), 1 };
        default:
            throw oneapi::math::unimplemented(
                "dft/test_common", __FUNCTION__,
                "not implemented for " + std::to_string(sizes.size()) + " dimensions");
            return {};
    }
}

// is it assumed that the unused elements of the array are ignored
inline std::array<std::int64_t, 4> get_default_strides(const std::vector<std::int64_t>& sizes) {
    if (sizes.size() > 3) {
        throw oneapi::math::unimplemented(
            "dft/test_common", __FUNCTION__,
            "not implemented for " + std::to_string(sizes.size()) + " dimensions");
    }

    switch (sizes.size()) {
        case 1: return { 0, 1 };
        case 2: return { 0, sizes[1], 1 };
        case 3: return { 0, sizes[1] * sizes[2], sizes[2], 1 };
        default:
            throw oneapi::math::unimplemented(
                "dft/test_common", __FUNCTION__,
                "not implemented for " + std::to_string(sizes.size()) + " dimensions");
            return {};
    }
}

template <typename T>
T get_default(const std::vector<T> vec, std::size_t idx, T default_) {
    if (idx >= vec.size()) {
        return default_;
    }
    return vec[idx];
}

template <oneapi::math::dft::domain domain, bool in_place = false>
std::pair<std::int64_t, std::int64_t> get_default_distances(
    const std::vector<std::int64_t>& sizes, const std::vector<std::int64_t>& strides_fwd,
    const std::vector<std::int64_t>& strides_bwd) {
    std::int64_t size0 = sizes[0];
    std::int64_t size1 = get_default(sizes, 1, 1l);
    std::int64_t size2 = get_default(sizes, 2, 1l);
    std::int64_t size0_real =
        domain == oneapi::math::dft::domain::REAL && sizes.size() == 1 ? size0 / 2 + 1 : size0;
    std::int64_t size1_real =
        domain == oneapi::math::dft::domain::REAL && sizes.size() == 2 ? size1 / 2 + 1 : size1;
    std::int64_t size2_real =
        domain == oneapi::math::dft::domain::REAL && sizes.size() == 3 ? size2 / 2 + 1 : size2;
    std::int64_t backward_distance = size0_real * size1_real * size2_real;
    std::int64_t forward_distance = size0 * size1 * size2;
    if (strides_fwd.size() > 1) {
        forward_distance =
            std::max({ size0 * strides_fwd[1], size1 * get_default(strides_fwd, 2, 0l),
                       size2 * get_default(strides_fwd, 3, 0l) });
    }
    if (strides_bwd.size() > 1) {
        backward_distance =
            std::max({ size0 * strides_bwd[1], size1 * get_default(strides_bwd, 2, 0l),
                       size2 * get_default(strides_bwd, 3, 0l) });
    }
    if (in_place) {
        forward_distance =
            std::max(forward_distance,
                     backward_distance * (domain == oneapi::math::dft::domain::REAL ? 2L : 1L));
    }
    return { forward_distance, backward_distance };
}

//up to 3 dimensions, empty strides = default
template <typename T_vec, typename Allocator = std::allocator<typename T_vec::value_type>>
std::vector<typename T_vec::value_type, Allocator> strided_copy(
    const T_vec& contiguous, const std::vector<std::int64_t>& sizes,
    const std::vector<std::int64_t>& strides, std::int64_t batches, std::int64_t distance,
    Allocator alloc = {}) {
    if (strides.size() == 0) {
        return { contiguous.begin(), contiguous.end(), alloc };
    }
    using T = typename T_vec::value_type;
    std::int64_t size0 = sizes[0];
    std::int64_t size1 = get_default(sizes, 1, 1l);
    std::int64_t size2 = get_default(sizes, 2, 1l);

    std::int64_t stride0 = strides[0];
    std::int64_t stride1 = strides[1];
    std::int64_t stride2 = get_default(strides, 2, 0l);
    std::int64_t stride3 = get_default(strides, 3, 0l);
    std::vector<T, Allocator> res(cast_unsigned(distance * batches + stride0), alloc);
    for (std::int64_t b = 0; b < batches; b++) {
        for (std::int64_t i = 0; i < size0; i++) {
            for (std::int64_t j = 0; j < size1; j++) {
                for (std::int64_t k = 0; k < size2; k++) {
                    res[cast_unsigned(b * distance + i * stride1 + j * stride2 + k * stride3 +
                                      stride0)] =
                        contiguous[cast_unsigned(((b * size0 + i) * size1 + j) * size2 + k)];
                }
            }
        }
    }
    return res;
}

//up to 3 dimensions, empty strides = default
template <bool ConjugateEvenStrides, typename vec1, typename vec2>
bool check_equal_strided(const vec1& v, const vec2& v_ref, std::vector<int64_t> sizes,
                         std::vector<int64_t> strides, double abs_error_mag, double rel_error_mag,
                         std::ostream& out) {
    if (strides.size() == 0) {
        std::array<std::int64_t, 4> strides_arr;
        if constexpr (ConjugateEvenStrides) {
            strides_arr = get_conjugate_even_complex_strides(sizes);
        }
        else {
            strides_arr = get_default_strides(sizes);
        }
        strides = { &strides_arr[0], &strides_arr[sizes.size() + 1] };
    }
    using T = std::decay_t<decltype(v[0])>;
    std::int64_t size0 = sizes[0];
    std::int64_t size1 = get_default(sizes, 1, 1l);
    std::int64_t size2 = get_default(sizes, 2, 1l);
    std::int64_t size0_real = ConjugateEvenStrides && sizes.size() == 1 ? size0 / 2 + 1 : size0;
    std::int64_t size1_real = ConjugateEvenStrides && sizes.size() == 2 ? size1 / 2 + 1 : size1;
    std::int64_t size2_real = ConjugateEvenStrides && sizes.size() == 3 ? size2 / 2 + 1 : size2;

    std::int64_t stride0 = strides[0];
    std::int64_t stride1 = strides[1];
    std::int64_t stride2 = get_default(strides, 2, 0l);
    std::int64_t stride3 = get_default(strides, 3, 0l);

    constexpr int max_print = 20;
    int count = 0;
    bool good = true;

    for (std::int64_t i = 0; i < size0_real; i++) {
        for (std::int64_t j = 0; j < size1_real; j++) {
            for (std::int64_t k = 0; k < size2_real; k++) {
                T res = v[cast_unsigned(i * stride1 + j * stride2 + k * stride3 + stride0)];
                T ref = v_ref[cast_unsigned((i * size1 + j) * size2 + k)];
                if (!check_equal(res, ref, abs_error_mag, rel_error_mag, out)) {
                    out << " at position " << i << ", " << j << ", " << k << "\n";
                    out << " at indices " << i * stride1 + j * stride2 + k * stride3 + stride0
                        << ", " << (i * size1 + j) * size2 + k << "\n";
                    good = false;
                    ++count;
                    if (count > max_print) {
                        return good;
                    }
                }
            }
        }
    }
    return good;
}

struct DFTParams {
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> strides_fwd;
    std::vector<std::int64_t> strides_bwd;
    std::int64_t batches;
    DFTParams(std::vector<std::int64_t> sizes, std::int64_t batches)
            : sizes(sizes),
              strides_fwd({}),
              strides_bwd({}),
              batches(batches) {}
    DFTParams(std::vector<std::int64_t> sizes, std::vector<std::int64_t> strides_fwd,
              std::vector<std::int64_t> strides_bwd, std::int64_t batches)
            : sizes(sizes),
              strides_fwd(strides_fwd),
              strides_bwd(strides_bwd),
              batches(batches) {}
};

class DFTParamsPrint {
public:
    std::string operator()(testing::TestParamInfo<std::tuple<sycl::device*, DFTParams>> dev) const {
        auto [device, params] = dev.param;
        std::string info_name;

        assert(params.sizes.size() > 0);
        info_name.append("sizes_");

        // intersperse dimensions with "x"
        std::for_each(params.sizes.begin(), params.sizes.end() - 1,
                      [&info_name](auto s) { info_name.append(std::to_string(s)).append("x"); });
        info_name.append(std::to_string(params.sizes.back()));

        if (params.strides_fwd.size() != 0) {
            info_name.append("_fwd_strides_");
            // intersperse strides with "_"
            std::for_each(
                params.strides_fwd.begin(), params.strides_fwd.end() - 1,
                [&info_name](auto s) { info_name.append(std::to_string(s)).append("_"); });
            info_name.append(std::to_string(params.strides_fwd.back()));
        }
        if (params.strides_bwd.size() != 0) {
            info_name.append("_bwd_strides_");
            // intersperse strides with "_"
            std::for_each(
                params.strides_bwd.begin(), params.strides_bwd.end() - 1,
                [&info_name](auto s) { info_name.append(std::to_string(s)).append("_"); });
            info_name.append(std::to_string(params.strides_bwd.back()));
        }

        info_name.append("_batches_").append(std::to_string(params.batches));

        std::string dev_name = device->get_info<sycl::info::device::name>();
        std::for_each(dev_name.begin(), dev_name.end(), [](auto& c) {
            if (!isalnum(c))
                c = '_';
        });

        info_name.append("_").append(dev_name);

        return info_name;
    }
};

#endif //ONEMATH_TEST_COMMON_HPP
