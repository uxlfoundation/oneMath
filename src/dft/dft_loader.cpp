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

#include "oneapi/mkl/dft/detail/dft_loader.hpp"
#include "oneapi/mkl/dft/forward.hpp"
#include "oneapi/mkl/dft/backward.hpp"

#include "function_table_initializer.hpp"
#include "dft/function_table.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi::mkl::dft::detail {

static oneapi::mkl::detail::table_initializer<mkl::domain::dft, dft_function_table_t>
    function_tables;

template <>
commit_impl<precision::SINGLE, domain::COMPLEX>* create_commit<precision::SINGLE, domain::COMPLEX>(
    const descriptor<precision::SINGLE, domain::COMPLEX>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[{ libkey, sycl_queue }].create_commit_sycl_fz(desc, sycl_queue);
}

template <>
commit_impl<precision::DOUBLE, domain::COMPLEX>* create_commit<precision::DOUBLE, domain::COMPLEX>(
    const descriptor<precision::DOUBLE, domain::COMPLEX>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[{ libkey, sycl_queue }].create_commit_sycl_dz(desc, sycl_queue);
}

template <>
commit_impl<precision::SINGLE, domain::REAL>* create_commit<precision::SINGLE, domain::REAL>(
    const descriptor<precision::SINGLE, domain::REAL>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[{ libkey, sycl_queue }].create_commit_sycl_fr(desc, sycl_queue);
}

template <>
commit_impl<precision::DOUBLE, domain::REAL>* create_commit<precision::DOUBLE, domain::REAL>(
    const descriptor<precision::DOUBLE, domain::REAL>& desc, sycl::queue& sycl_queue) {
    auto libkey = get_device_id(sycl_queue);
    return function_tables[{ libkey, sycl_queue }].create_commit_sycl_dr(desc, sycl_queue);
}

template <precision prec, domain dom>
inline oneapi::mkl::device get_device(descriptor<prec, dom>& desc, const char* func_name) {
    config_value is_committed{ config_value::UNCOMMITTED };
    desc.get_value(config_param::COMMIT_STATUS, &is_committed);
    if (is_committed != config_value::COMMITTED) {
        throw mkl::invalid_argument("DFT", func_name, "Descriptor not committed.");
    }
    // Committed means that the commit pointer is not null.
    return get_device_id(get_commit(desc)->get_queue());
}

} // namespace oneapi::mkl::dft::detail
