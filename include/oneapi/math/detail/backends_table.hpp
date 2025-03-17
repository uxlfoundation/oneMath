/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef _ONEMATH_BACKENDS_TABLE_HPP_
#define _ONEMATH_BACKENDS_TABLE_HPP_

#include <string>
#include <vector>
#include <map>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/config.hpp"

#ifdef __linux__
#define LIB_NAME(a) "libonemath_" a ".so"
#elif defined(_WIN64)
#define LIB_NAME(a) "onemath_" a ".dll"
#endif

namespace oneapi {
namespace math {

enum class device : uint16_t { x86cpu, aarch64cpu, intelgpu, nvidiagpu, amdgpu, generic_device };
enum class domain : uint16_t { blas, dft, lapack, rng, sparse_blas };

static std::map<domain, std::map<device, std::vector<const char*>>> libraries = {
    { domain::blas,
      { { device::x86cpu,
          {
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
              LIB_NAME("blas_mklcpu"),
#endif
#ifdef ONEMATH_ENABLE_NETLIB_BACKEND
              LIB_NAME("blas_netlib"),
#endif
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_CPU
              LIB_NAME("blas_generic"),
#endif
          } },
        { device::aarch64cpu,
          {
#ifdef ONEMATH_ENABLE_ARMPL_BACKEND
              LIB_NAME("blas_armpl"),
#endif
#ifdef ONEMATH_ENABLE_NETLIB_BACKEND
              LIB_NAME("blas_netlib")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
              LIB_NAME("blas_mklgpu"),
#endif
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_INTEL_GPU
              LIB_NAME("blas_generic"),
#endif
          } },
        { device::amdgpu,
          {
#ifdef ONEMATH_ENABLE_ROCBLAS_BACKEND
              LIB_NAME("blas_rocblas"),
#endif
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_AMD_GPU
              LIB_NAME("blas_generic"),
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ONEMATH_ENABLE_CUBLAS_BACKEND
              LIB_NAME("blas_cublas"),
#endif
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND_NVIDIA_GPU
              LIB_NAME("blas_generic"),
#endif
          } },
        { device::generic_device,
          {
#ifdef ONEMATH_ENABLE_GENERIC_BLAS_BACKEND
              LIB_NAME("blas_generic"),
#endif
          } } } },

    { domain::dft,
      { { device::x86cpu,
          {
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
              LIB_NAME("dft_mklcpu")
#endif
#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::aarch64cpu,
          {
#ifdef ONEMATH_ENABLE_ARMPL_BACKEND
              LIB_NAME("dft_armpl"),
#endif
          } },
        { device::intelgpu,
          {
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
              LIB_NAME("dft_mklgpu")
#endif
#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ONEMATH_ENABLE_ROCFFT_BACKEND
              LIB_NAME("dft_rocfft")
#endif
#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ONEMATH_ENABLE_CUFFT_BACKEND
              LIB_NAME("dft_cufft")
#endif
#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
                  LIB_NAME("dft_portfft")
#endif
          } },
        { device::generic_device,
          {
#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
              LIB_NAME("dft_portfft"),
#endif
          } } } },

    { domain::lapack,
      { { device::x86cpu,
          {
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
              LIB_NAME("lapack_mklcpu")
#endif
          } },
        { device::aarch64cpu,
          {
#ifdef ONEMATH_ENABLE_ARMPL_BACKEND
              LIB_NAME("lapack_armpl"),
#endif
          } },

        { device::intelgpu,
          {
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
              LIB_NAME("lapack_mklgpu")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ONEMATH_ENABLE_ROCSOLVER_BACKEND
              LIB_NAME("lapack_rocsolver")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ONEMATH_ENABLE_CUSOLVER_BACKEND
              LIB_NAME("lapack_cusolver")
#endif
          } } } },

    { domain::rng,
      { { device::x86cpu,
          {
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
              LIB_NAME("rng_mklcpu"),
#endif
#ifdef ONEMATH_ENABLE_ARMPL_BACKEND
              LIB_NAME("rng_armpl")
#endif
          } },
        { device::aarch64cpu,
          {
#ifdef ONEMATH_ENABLE_ARMPL_BACKEND
              LIB_NAME("rng_armpl")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
              LIB_NAME("rng_mklgpu")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ONEMATH_ENABLE_ROCRAND_BACKEND
              LIB_NAME("rng_rocrand")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ONEMATH_ENABLE_CURAND_BACKEND
              LIB_NAME("rng_curand")
#endif
          } } } },

    { domain::sparse_blas,
      { { device::x86cpu,
          {
#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
              LIB_NAME("sparse_blas_mklcpu")
#endif
          } },
        { device::intelgpu,
          {
#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
              LIB_NAME("sparse_blas_mklgpu")
#endif
          } },
        { device::nvidiagpu,
          {
#ifdef ONEMATH_ENABLE_CUSPARSE_BACKEND
              LIB_NAME("sparse_blas_cusparse")
#endif
          } },
        { device::amdgpu,
          {
#ifdef ONEMATH_ENABLE_ROCSPARSE_BACKEND
              LIB_NAME("sparse_blas_rocsparse")
#endif
          } } } },
};

static std::map<domain, const char*> table_names = { { domain::blas, "onemath_blas_table" },
                                                     { domain::lapack, "onemath_lapack_table" },
                                                     { domain::dft, "onemath_dft_table" },
                                                     { domain::rng, "onemath_rng_table" },
                                                     { domain::sparse_blas,
                                                       "onemath_sparse_blas_table" } };

} //namespace math
} //namespace oneapi

#endif //_ONEMATH_BACKENDS_TABLE_HPP_
