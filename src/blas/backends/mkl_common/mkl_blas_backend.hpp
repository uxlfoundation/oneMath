/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#pragma once

#include <complex>

// Intel(R) oneMKL headers
#include <mkl_version.h>
#include <mkl/blas.hpp>

#include "common_onemkl_conversion.hpp"
#include "oneapi/math/types.hpp"

namespace oneapi {
namespace math {

template <typename T>
class value_or_pointer {
    T value_;
    const T* ptr_;

public:
    // Constructor from value. Accepts not only type T but anything convertible to T.
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
    value_or_pointer(U value) : value_(value),
                                ptr_(nullptr) {}

    // Constructor from pointer, assumed to be device-accessible.
    value_or_pointer(const T* ptr) : value_(T(0)), ptr_(ptr) {}

    bool fixed() const {
        return ptr_ == nullptr;
    }

    T get_fixed_value() const {
        return value_;
    }

    const T* get_pointer() const {
        return ptr_;
    }

    T get() const {
        return ptr_ ? *ptr_ : value_;
    }

    void make_device_accessible(sycl::queue& queue) {
        if (!fixed() &&
            sycl::get_pointer_type(ptr_, queue.get_context()) == sycl::usm::alloc::unknown) {
            *this = *ptr_;
        }
    }
};

} // namespace math
} // namespace oneapi
