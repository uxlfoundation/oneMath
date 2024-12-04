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

#ifndef _ONEMATH_EXCEPTIONS_HPP_
#define _ONEMATH_EXCEPTIONS_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <exception>
#include <string>

#include "oneapi/math/types.hpp"

// These are oneAPI oneMath Specification exceptions

namespace oneapi {
namespace math {
class exception : public std::exception {
    std::string msg_;

public:
    exception(const std::string& message) : std::exception(), msg_(message) {}
    exception(const std::string& domain, const std::string& function, const std::string& info = "")
            : std::exception() {
        msg_ = std::string("oneMath: ") + domain +
               ((domain.length() != 0 && function.length() != 0) ? "/" : "") + function +
               ((info.length() != 0)
                    ? (((domain.length() + function.length() != 0) ? ": " : "") + info)
                    : "");
    }

    const char* what() const noexcept override {
        return msg_.c_str();
    }
};

class unsupported_device : public oneapi::math::exception {
public:
    unsupported_device(const std::string& message) : exception(message) {}
    unsupported_device(const std::string& domain, const std::string& function,
                       const sycl::device& device)
            : oneapi::math::exception(
                  domain, function,
                  device.get_info<sycl::info::device::name>() + " is not supported") {}
};

class host_bad_alloc : public oneapi::math::exception {
public:
    host_bad_alloc(const std::string& message) : exception(message) {}
    host_bad_alloc(const std::string& domain, const std::string& function)
            : oneapi::math::exception(domain, function, "cannot allocate memory on host") {}
};

class device_bad_alloc : public oneapi::math::exception {
public:
    device_bad_alloc(const std::string& message) : exception(message) {}
    device_bad_alloc(const std::string& domain, const std::string& function,
                     const sycl::device& device)
            : oneapi::math::exception(
                  domain, function,
                  "cannot allocate memory on " + device.get_info<sycl::info::device::name>()) {}
};

class unimplemented : public oneapi::math::exception {
public:
    unimplemented(const std::string& message) : exception(message) {}
    unimplemented(const std::string& domain, const std::string& function,
                  const std::string& info = "")
            : oneapi::math::exception(domain, function, "function is not implemented " + info) {}
};

class invalid_argument : public oneapi::math::exception {
public:
    invalid_argument(const std::string& message) : exception(message) {}
    invalid_argument(const std::string& domain, const std::string& function,
                     const std::string& info = "")
            : oneapi::math::exception(domain, function, "invalid argument " + info) {}
};

class uninitialized : public oneapi::math::exception {
public:
    uninitialized(const std::string& message) : exception(message) {}
    uninitialized(const std::string& domain, const std::string& function,
                  const std::string& info = "")
            : oneapi::math::exception(domain, function,
                                      "handle/descriptor is not initialized " + info) {}
};

class computation_error : public oneapi::math::exception {
public:
    computation_error(const std::string& message) : exception(message) {}
    computation_error(const std::string& domain, const std::string& function,
                      const std::string& info = "")
            : oneapi::math::exception(
                  domain, function,
                  "computation error" + ((info.length() != 0) ? (": " + info) : "")) {}
};

class batch_error : public oneapi::math::exception {
public:
    batch_error(const std::string& message) : exception(message) {}
    batch_error(const std::string& domain, const std::string& function,
                const std::string& info = "")
            : oneapi::math::exception(
                  domain, function, "batch error" + ((info.length() != 0) ? (": " + info) : "")) {}
};

} // namespace math
} // namespace oneapi

#endif // _ONEMATH_EXCEPTIONS_HPP_
