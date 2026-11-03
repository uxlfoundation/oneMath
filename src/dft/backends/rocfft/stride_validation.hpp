/*******************************************************************************
* Copyright contributors to the oneMath project
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

#ifndef _ONEMATH_DFT_SRC_ROCFFT_STRIDE_VALIDATION_HPP_
#define _ONEMATH_DFT_SRC_ROCFFT_STRIDE_VALIDATION_HPP_

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

// Hardware-independent stride validation used by the rocFFT backend's commit.
// It only depends on the standard library so it can be unit tested without rocFFT.
namespace oneapi::math::dft::rocfft::detail {

// Checks that the data described by strides and lengths doesn't overlap: ordered by stride,
// each dimension must end before the next one starts. Strides and lengths are in rocFFT order
// (innermost dimension first) and strides exclude the offset.
template <typename StridesT, typename LengthsT>
bool strides_fit_lengths(std::size_t dimensions, const StridesT& strides, const LengthsT& lengths) {
    std::vector<std::size_t> order(dimensions);
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return strides[a] < strides[b]; });
    for (std::size_t i = 1; i < dimensions; ++i) {
        if (strides[order[i - 1]] * lengths[order[i - 1]] > strides[order[i]]) {
            return false;
        }
    }
    return true;
}

struct stride_validity {
    bool forward;
    bool backward;
};

// Determines which transform directions have valid strides. vec_a and vec_b are the stride
// vectors in rocFFT order; fwd_lengths and bwd_lengths are the forward and backward domain
// lengths in rocFFT order (they differ only for real transforms). fb_strides is true when
// FWD/BWD_STRIDES are used and false for INPUT/OUTPUT_STRIDES.
template <typename StridesT, typename LengthsT>
stride_validity get_stride_validity(std::size_t dimensions, const StridesT& vec_a,
                                    const StridesT& vec_b, const LengthsT& fwd_lengths,
                                    const LengthsT& bwd_lengths, bool fb_strides) {
    // The forward direction reads forward-domain data through vec_a (fwd_in) and writes
    // backward-domain data through vec_b (fwd_out).
    const bool forward = strides_fit_lengths(dimensions, vec_a, fwd_lengths) &&
                         strides_fit_lengths(dimensions, vec_b, bwd_lengths);
    // With FWD/BWD_STRIDES each vector describes one domain, so the backward direction has
    // the same requirements. With INPUT/OUTPUT_STRIDES the domains swap: vec_a (bwd_in)
    // describes backward-domain data and vec_b (bwd_out) forward-domain data.
    const bool backward = fb_strides ? forward
                                     : strides_fit_lengths(dimensions, vec_a, bwd_lengths) &&
                                           strides_fit_lengths(dimensions, vec_b, fwd_lengths);
    return { forward, backward };
}

} // namespace oneapi::math::dft::rocfft::detail

#endif // _ONEMATH_DFT_SRC_ROCFFT_STRIDE_VALIDATION_HPP_
