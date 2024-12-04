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

#ifndef ONEMATH_PARSEVAL_CHECK_HPP
#define ONEMATH_PARSEVAL_CHECK_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <numeric>

#include "test_common.hpp"

/** Use Parseval's theorem to verify the output of DFT. This does not guarantee that the output
 * of the DFT is correct, and is only a sanity check.
 * 
 * Check Sum(|in[i]|^2) == Sum(|out[i]|^2).
 * 
 * @tparam TypeFwd Forward domain type
 * @tparam TypeBwd Backward domain type
 * @param dft_len DFT size
 * @param in forward domain data
 * @param out bwd domain data
 * @param rescale_forward A value to multiply the in data by.
*/
template <typename TypeFwd, typename TypeBwd>
bool parseval_check(std::size_t dft_len, const TypeFwd* in, TypeBwd* out,
                    TypeFwd rescale_forward = 1) {
    static_assert(is_complex<TypeBwd>());
    bool complex_forward = is_complex<TypeFwd>();
    auto bwd_len = complex_forward ? dft_len : dft_len / 2 + 1;

    float in_sum{ 0 };
    float out_sum{ 0 };
    for (std::size_t i{ 0 }; i < dft_len; ++i) {
        in_sum += static_cast<float>(std::abs(in[i] * rescale_forward) *
                                     std::abs(in[i] * rescale_forward));
    }
    if (complex_forward) {
        for (std::size_t i{ 0 }; i < bwd_len; ++i) {
            out_sum += static_cast<float>(std::abs(out[i]) * std::abs(out[i]));
        }
    }
    else {
        for (std::size_t i{ 0 }; i < bwd_len - 1; ++i) {
            out_sum += static_cast<float>(std::abs(out[i]) * std::abs(out[i]));
        }
        out_sum *= 2;
        out_sum += static_cast<float>(std::abs(out[bwd_len - 1]) * std::abs(out[bwd_len - 1]));
    }
    out_sum /= static_cast<float>(dft_len);
    auto max_norm_ref = *std::max_element(
        in, in + dft_len, [](const auto& a, const auto& b) { return std::abs(a) < std::abs(b); });
    // Heuristic for the average-case error margins
    auto abs_error_margin = 10 * std::abs(max_norm_ref) * std::log2(static_cast<float>(dft_len));
    if (std::abs(in_sum - out_sum) > abs_error_margin) {
        std::cout << "Failed check with Parseval's theorem: Fwd sum = " << in_sum
                  << ", Bwd sum = " << out_sum << " (tol = " << abs_error_margin << ")"
                  << std::endl;
        return false;
    }
    return true;
}
#endif // ONEMATH_PARSEVAL_CHECK_HPP
