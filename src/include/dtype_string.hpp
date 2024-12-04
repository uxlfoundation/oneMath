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

#ifndef _ONEMATH_ERROR_HELPER_HPP_
#define _ONEMATH_ERROR_HELPER_HPP_

#include <string>

template <typename T>
inline const std::string dtype_string();
template <>
inline const std::string dtype_string<float>() {
    return "float";
}
template <>
inline const std::string dtype_string<double>() {
    return "double";
}
template <>
inline const std::string dtype_string<sycl::half>() {
    return "half";
}
template <>
inline const std::string dtype_string<std::complex<float>>() {
    return "complex<float>";
}
template <>
inline const std::string dtype_string<std::complex<double>>() {
    return "complex<double>";
}
template <>
inline const std::string dtype_string<std::int32_t>() {
    return "int32";
}
template <>
inline const std::string dtype_string<std::int8_t>() {
    return "int8";
}

#endif //_ONEMATH_ERROR_HELPER_HPP_
