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

#ifndef _ONEMATH_GET_DEVICE_ID_HPP_
#define _ONEMATH_GET_DEVICE_ID_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/backends_table.hpp"
#include "oneapi/math/exceptions.hpp"

#define INTEL_ID  32902
#define NVIDIA_ID 4318
#ifndef __HIPSYCL__
#define AMD_ID 4098
#else
#define AMD_ID 1022
#endif

namespace oneapi {
namespace math {

inline oneapi::math::device get_device_id(sycl::queue& queue) {
    oneapi::math::device device_id;
    if (queue.get_device().is_cpu())
        device_id = device::x86cpu;
#ifdef __HIPSYCL__
    else if (queue.is_host())
        device_id = device::x86cpu;
#endif
    else if (queue.get_device().is_gpu()) {
        unsigned int vendor_id =
            static_cast<unsigned int>(queue.get_device().get_info<sycl::info::device::vendor_id>());

        if (vendor_id == INTEL_ID)
            device_id = device::intelgpu;
        else if (vendor_id == NVIDIA_ID)
            device_id = device::nvidiagpu;
        else if (vendor_id == AMD_ID)
            device_id = device::amdgpu;
        else
            device_id = device::generic_device;
    }
    else {
        device_id = device::generic_device;
    }
    return device_id;
}

} //namespace math
} //namespace oneapi

#endif //_ONEMATH_GET_DEVICE_ID_HPP_
