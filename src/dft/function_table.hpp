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

#ifndef _DFT_FUNCTION_TABLE_HPP_
#define _DFT_FUNCTION_TABLE_HPP_

#include <complex>
#include <cstdint>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/types.hpp"
#include "oneapi/math/dft/types.hpp"
#include "oneapi/math/dft/descriptor.hpp"

typedef struct {
    int version;
    oneapi::math::dft::detail::commit_impl<oneapi::math::dft::precision::SINGLE,
                                           oneapi::math::dft::domain::COMPLEX>* (
        *create_commit_sycl_fz)(
        const oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                            oneapi::math::dft::domain::COMPLEX>& desc,
        sycl::queue& sycl_queue);
    oneapi::math::dft::detail::commit_impl<oneapi::math::dft::precision::DOUBLE,
                                           oneapi::math::dft::domain::COMPLEX>* (
        *create_commit_sycl_dz)(
        const oneapi::math::dft::descriptor<oneapi::math::dft::precision::DOUBLE,
                                            oneapi::math::dft::domain::COMPLEX>& desc,
        sycl::queue& sycl_queue);
    oneapi::math::dft::detail::commit_impl<oneapi::math::dft::precision::SINGLE,
                                           oneapi::math::dft::domain::REAL>* (
        *create_commit_sycl_fr)(
        const oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                            oneapi::math::dft::domain::REAL>& desc,
        sycl::queue& sycl_queue);
    oneapi::math::dft::detail::commit_impl<oneapi::math::dft::precision::DOUBLE,
                                           oneapi::math::dft::domain::REAL>* (
        *create_commit_sycl_dr)(
        const oneapi::math::dft::descriptor<oneapi::math::dft::precision::DOUBLE,
                                            oneapi::math::dft::domain::REAL>& desc,
        sycl::queue& sycl_queue);
} dft_function_table_t;

#endif //_DFT_FUNCTION_TABLE_HPP_
