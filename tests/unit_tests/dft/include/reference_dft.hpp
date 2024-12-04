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

#ifndef ONEMATH_REFERENCE_DFT_HPP
#define ONEMATH_REFERENCE_DFT_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>

#include <oneapi/math/exceptions.hpp>
#include "test_common.hpp"

namespace detail {
using ref_t = long double; /* Do the calculations using long double */
template <typename TypeIn, typename TypeOut>
void reference_forward_dft_impl(const TypeIn* in, TypeOut* out, std::size_t N, std::size_t stride) {
    static_assert(is_complex<TypeOut>(), "Output type of DFT must be complex");

    constexpr ref_t TWOPI = 2.0L * 3.141592653589793238462643383279502884197L;

    for (std::size_t k = 0; k < N; ++k) {
        std::complex<ref_t> out_temp = 0;
        const auto partial_expo = (static_cast<ref_t>(k) * TWOPI) / static_cast<ref_t>(N);
        for (std::size_t n = 0; n < N; ++n) {
            const auto expo = static_cast<ref_t>(n) * partial_expo;
            out_temp += static_cast<std::complex<ref_t>>(in[n * stride]) *
                        std::complex<ref_t>{ std::cos(expo), -std::sin(expo) };
        }
        out[k * stride] = static_cast<TypeOut>(out_temp);
    }
}

template <typename TypeIn, typename TypeOut, int dims>
struct reference {};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 1> {
    static void forward_dft(const std::vector<std::size_t>& sizes, const TypeIn* in, TypeOut* out) {
        reference_forward_dft_impl(in, out, sizes[0], 1);
    }
};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 2> {
    static void forward_dft(const std::vector<std::size_t>& sizes, const TypeIn* in, TypeOut* out) {
        const auto elements = std::accumulate(sizes.begin(), sizes.end(), 1U, std::multiplies<>{});
        std::vector<std::complex<ref_t>> tmp(elements);
        for (std::size_t i = 0; i < elements; i += sizes[1]) {
            reference_forward_dft_impl(in + i, tmp.data() + i, sizes[1], 1);
        }
        for (std::size_t i = 0; i < sizes[1]; i++) {
            reference_forward_dft_impl(tmp.data() + i, out + i, sizes[0], sizes[1]);
        }
    }
};

template <typename TypeIn, typename TypeOut>
struct reference<TypeIn, TypeOut, 3> {
    static void forward_dft(const std::vector<std::size_t>& sizes, const TypeIn* in, TypeOut* out) {
        const auto elements = std::accumulate(sizes.begin(), sizes.end(), 1U, std::multiplies<>{});
        std::vector<std::complex<ref_t>> tmp1(elements);
        std::vector<std::complex<ref_t>> tmp2(elements);
        for (std::size_t i = 0; i < elements; i += sizes[2]) {
            reference_forward_dft_impl(in + i, tmp1.data() + i, sizes[2], 1);
        }
        for (std::size_t j = 0; j < elements; j += sizes[1] * sizes[2]) {
            for (std::size_t i = 0; i < sizes[2]; i++) {
                reference_forward_dft_impl(tmp1.data() + i + j, tmp2.data() + i + j, sizes[1],
                                           sizes[2]);
            }
        }
        for (std::size_t i = 0; i < sizes[1] * sizes[2]; i++) {
            reference_forward_dft_impl(tmp2.data() + i, out + i, sizes[0], sizes[1] * sizes[2]);
        }
    }
};
} // namespace detail

/** Naive DFT implementation for reference.
 *  
 * Directly compute a single 1D forward DFT of the form:
 * for k in range(0, N):
 *   out[k] = sum( exp(2 pi k n im / N) * in[n] for n in range(0, N) )
 * where N is the size of the input / output arrays. The input may be
 * real or complex, but the output must be complex.
 *  
 * @tparam TypeIn The forward data type. Must be complex or real.
 * @tparam TypeOut The transformed (backward) data type. Written to. Must be 
 * complex.
 * @param in The input forward data.
 * @param out Where to write the output data.
 * @param N The number of elements in the input data set.
 * @param stride the stride between elements in the data set, measured in elements.
**/
template <typename TypeIn, typename TypeOut>
void reference_forward_dft(const std::vector<std::int64_t>& sizes, const TypeIn* in, TypeOut* out) {
    std::vector<std::size_t> unsigned_sizes(sizes.size());
    std::transform(sizes.begin(), sizes.end(), unsigned_sizes.begin(),
                   [](std::int64_t size) { return cast_unsigned(size); });
    switch (unsigned_sizes.size()) {
        case 1: detail::reference<TypeIn, TypeOut, 1>::forward_dft(unsigned_sizes, in, out); break;
        case 2: detail::reference<TypeIn, TypeOut, 2>::forward_dft(unsigned_sizes, in, out); break;
        case 3: detail::reference<TypeIn, TypeOut, 3>::forward_dft(unsigned_sizes, in, out); break;
        default:
            throw oneapi::math::unimplemented(
                "reference_dft", "forward_dft",
                "dft with size " + std::to_string(unsigned_sizes.size()));
    }
}

#endif //ONEMATH_REFERENCE_DFT_HPP
