/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef _RNG_EXAMPLE_HELPER_HPP__
#define _RNG_EXAMPLE_HELPER_HPP__

template <typename T, typename = void>
struct has_member_code_meta : std::false_type {};

template <typename T>
struct has_member_code_meta<T, std::void_t<decltype(std::declval<T>().get_multi_ptr())>>
        : std::true_type {};

template <typename T, typename std::enable_if<has_member_code_meta<T>::value>::type* = nullptr>
auto get_multi_ptr(T acc) {
// Workaround for AdaptiveCPP, as they do not yet support the get_multi_ptr function
#ifndef __HIPSYCL__
    return acc.get_multi_ptr();
#else
    return acc.get_pointer();
#endif
}

template <typename T, typename std::enable_if<!has_member_code_meta<T>::value>::type* = nullptr>
auto get_multi_ptr(T acc) {
// Workaround for AdaptiveCPP, as they do not yet support the get_multi_ptr function
#ifndef __HIPSYCL__
    return acc.template get_multi_ptr<sycl::access::decorated::yes>();
#else
    return acc.get_pointer();
#endif
}

#endif // _RNG_EXAMPLE_HELPER_HPP__
