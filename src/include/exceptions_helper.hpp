/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef __EXCEPTIONS_HELPER_HPP
#define __EXCEPTIONS_HELPER_HPP

#include <stdexcept>

namespace oneapi {
namespace math {

class backend_unsupported_exception : public std::runtime_error {
public:
    backend_unsupported_exception() : std::runtime_error("Not yet supported for this backend") {}
};

} // namespace math
} // namespace oneapi

#endif // __EXCEPTIONS_HELPER_HPP
