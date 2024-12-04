/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef _DFT_DETAIL_STRIDE_HELPER_HPP_
#define _DFT_DETAIL_STRIDE_HELPER_HPP_

namespace oneapi::math::dft::detail {

enum class stride_api {
    INVALID, // Cannot choose: no valid choice
    FB_STRIDES, // Use FWD_STRIDES and BWD_STRIDES
    IO_STRIDES // Use INPUT_STRIDES and OUTPUT_STRIDES
};

/** Throw invalid_argument for stride_api::INVALID
 *  @param function Function name to include in exception.
 *  @param stride_choice The stride_api to check if INVALID. Default is INVALID.
 * 
 *  @throws invalid_argument on stride_api::INVALID.
 */
inline void throw_on_invalid_stride_api(const char* function,
                                        stride_api stride_choice = stride_api::INVALID) {
    if (stride_choice == stride_api::INVALID) {
        throw math::invalid_argument(
            "DFT", function,
            "Invalid INPUT/OUTPUT or FWD/BACKWARD strides. API usage may have been mixed.");
    }
}

// Helper class for mapping input / output strides for backend DFTs to config values.
// Intended to be abused as required for each backend.
template <typename StrideElemT>
struct stride_vectors {
    using stride_elem_t = StrideElemT;
    using stride_vec_t = std::vector<StrideElemT>;

    // The stride API being used.
    const stride_api stride_choice;

    // The storage for strides. vec_a is forward or input.
    stride_vec_t vec_a, vec_b;

    // Input and output strides for forward and backward DFTs.
    stride_vec_t &fwd_in, &fwd_out, &bwd_in, &bwd_out;

    // Input and output offsets for forward and backward DFTs.
    StrideElemT offset_fwd_in, offset_fwd_out, offset_bwd_in, offset_bwd_out;

    /** Initialize the forward / backwards input and output strides for this object.
    *  @tparam ConfigT The config values type.
    *  @param config_values The DFT config values.
    *  @param stride_api The stride API choice. Must not be INVALID.
    **/
    template <typename ConfigT>
    stride_vectors(const ConfigT& config_values, stride_api stride_choice)
            : stride_choice(stride_choice),
              fwd_in(vec_a),
              fwd_out(vec_b),
              bwd_in(stride_choice == stride_api::FB_STRIDES ? vec_b : vec_a),
              bwd_out(stride_choice == stride_api::FB_STRIDES ? vec_a : vec_b) {
        if (stride_choice == stride_api::INVALID) {
            throw math::exception("DFT", "detail::stride_vector constructor",
                                  "Internal error: invalid stride API");
        }
        auto& v1 = stride_choice == stride_api::FB_STRIDES ? config_values.fwd_strides
                                                           : config_values.input_strides;
        auto& v2 = stride_choice == stride_api::FB_STRIDES ? config_values.bwd_strides
                                                           : config_values.output_strides;

        vec_a.resize(v1.size());
        vec_b.resize(v2.size());
        for (std::size_t i{ 0 }; i < v1.size(); ++i) { // v1.size() == v2.size()
            if constexpr (std::is_unsigned_v<StrideElemT>) {
                if (v1[i] < 0 || v2[i] < 0) {
                    throw math::unimplemented("DFT", "commit",
                                              "Backend does not support negative strides.");
                }
            }
            vec_a[i] = static_cast<StrideElemT>(v1[i]);
            vec_b[i] = static_cast<StrideElemT>(v2[i]);
        }
        offset_fwd_in = fwd_in[0];
        offset_fwd_out = fwd_out[0];
        offset_bwd_in = bwd_in[0];
        offset_bwd_out = bwd_out[0];
    }
};

/** Determines whether INPUT/OUTPUT strides, or FWD/BWD strides API is used.
 *  @tparam ConfigT The config values type.
 *  @param config_values The DFT config values.
 *  @returns Stride choice. INVALID if the choice could not be determined.
 * 
 *  @note This does not attempt to determine that the set strides are valid.
 */
template <typename ConfigT>
inline stride_api get_stride_api(const ConfigT& config_values) {
    auto n = config_values.dimensions.size();
    // Test if FWD/BWD strides look like they should be used. If yes, use them.
    if (config_values.fwd_strides.size() == n + 1 && config_values.bwd_strides.size() == n + 1) {
        auto all_zero_fwd = true;
        auto all_zero_bwd = true;
        // If INPUT or OUTPUT have been set, these will be zeroed.
        for (auto v : config_values.fwd_strides) {
            all_zero_fwd = v == 0 && all_zero_fwd;
        }
        for (auto v : config_values.bwd_strides) {
            all_zero_bwd = v == 0 && all_zero_bwd;
        }
        if (!all_zero_fwd && !all_zero_bwd) { // Both must be non-zero.
            return stride_api::FB_STRIDES;
        }
    }
    // FWD/BWD invalid. Test INPUT/OUTPUT for validity.
    if (config_values.input_strides.size() == n + 1 &&
        config_values.output_strides.size() == n + 1) {
        auto all_zero_in = true;
        auto all_zero_out = true;
        // If FWD or BWD have been set, these will be zeroed.
        for (auto v : config_values.input_strides) {
            all_zero_in = v == 0 && all_zero_in;
        }
        for (auto v : config_values.output_strides) {
            all_zero_out = v == 0 && all_zero_out;
        }
        if (!all_zero_in && !all_zero_out) { // Both must be non-zero.
            return stride_api::IO_STRIDES;
        }
    }
    return stride_api::INVALID;
}

} // namespace oneapi::math::dft::detail

#endif //_DFT_DETAIL_STRIDE_HELPER_HPP_
