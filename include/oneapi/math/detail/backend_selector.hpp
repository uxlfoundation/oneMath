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

#ifndef _ONEMATH_BACKEND_SELECTOR_HPP_
#define _ONEMATH_BACKEND_SELECTOR_HPP_

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/detail/backends.hpp"
#include "oneapi/math/detail/backend_selector_predicates.hpp"

namespace oneapi {
namespace math {

template <backend Backend>
class backend_selector {
public:
    explicit backend_selector(sycl::queue queue) : queue_(queue) {
        backend_selector_precondition<Backend>(queue_);
    }
    sycl::queue& get_queue() {
        return queue_;
    }

private:
    sycl::queue queue_;
};

} // namespace math
} // namespace oneapi

#endif //_ONEMATH_BACKEND_SELECTOR_HPP_
