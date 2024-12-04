/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef _ONEMATH_DFT_TYPES_HPP_
#define _ONEMATH_DFT_TYPES_HPP_

#include "detail/types_impl.hpp"

namespace oneapi {
namespace math {
namespace dft {

/** The detail namespace is required since the MKLGPU backend uses identical 
names and function signatures in many places. **/

using precision = detail::precision;
using domain = detail::domain;
using config_param = detail::config_param;
using config_value = detail::config_value;
using DFT_ERROR = detail::DFT_ERROR;

} // namespace dft
} // namespace math
} // namespace oneapi

#endif //_ONEMATH_TYPES_HPP_
