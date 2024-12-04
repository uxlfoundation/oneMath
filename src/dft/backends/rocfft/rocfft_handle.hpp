/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#ifndef _ONEMATH_DFT_SRC_ROCFFT_ROCFFT_HANDLE_HPP_
#define _ONEMATH_DFT_SRC_ROCFFT_ROCFFT_HANDLE_HPP_

#include <optional>

struct rocfft_plan_t;
struct rocfft_execution_info_t;

struct rocfft_handle {
    std::optional<rocfft_plan_t*> plan = std::nullopt;
    std::optional<rocfft_execution_info_t*> info = std::nullopt;
    std::optional<void*> buffer = std::nullopt;
};

#endif
