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

// These are oneAPI Math Kernel Library (oneMKL) Interfaces specific exceptions

#ifndef _ONEMKL_DETAIL_EXCEPTIONS_HPP_
#define _ONEMKL_DETAIL_EXCEPTIONS_HPP_

#include <exception>
#include <string>
#include "oneapi/mkl/exceptions.hpp"

namespace oneapi {
namespace mkl {

class backend_not_found : public oneapi::mkl::exception {
public:
    backend_not_found(const std::string& info = "")
            : oneapi::mkl::exception(
                  "", "", ((info.length() != 0) ? info : "Couldn't load selected backend")) {}
};

class function_not_found : public oneapi::mkl::exception {
public:
    function_not_found(const std::string& info = "")
            : oneapi::mkl::exception(
                  "", "",
                  ((info.length() != 0) ? info : "Couldn't load functions from selected backend")) {
    }
};

class specification_mismatch : public oneapi::mkl::exception {
public:
    specification_mismatch(const std::string& info = "")
            : oneapi::mkl::exception(
                  "", "",
                  ((info.length() != 0) ? info : "Loaded oneMKL specification version mismatch")) {}
};

} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_DETAIL_EXCEPTIONS_HPP_
