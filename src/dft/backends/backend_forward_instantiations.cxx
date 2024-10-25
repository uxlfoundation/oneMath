/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
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

using desc_rf_t =
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>;
using desc_cf_t =
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>;
using desc_rd_t =
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>;
using desc_cd_t =
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>;
using depends_vec_t = const std::vector<sycl::event>&;

#define ONEMKL_DFT_FORWARD_INSTANTIATIONS(DESCRIPTOR_T, SCALAR_T, FORWARD_T, BACKWARD_T)          \
    /* Buffer API */                                                                              \
    template ONEMKL_EXPORT void compute_forward<DESCRIPTOR_T>(DESCRIPTOR_T&,                      \
                                                              sycl::buffer<FORWARD_T>&);          \
    template ONEMKL_EXPORT void compute_forward<DESCRIPTOR_T>(                                    \
        DESCRIPTOR_T&, sycl::buffer<SCALAR_T>&, sycl::buffer<SCALAR_T>&);                         \
    template ONEMKL_EXPORT void compute_forward<DESCRIPTOR_T>(                                    \
        DESCRIPTOR_T&, sycl::buffer<FORWARD_T>&, sycl::buffer<BACKWARD_T>&);                      \
    template ONEMKL_EXPORT void compute_forward<DESCRIPTOR_T>(                                    \
        DESCRIPTOR_T&, sycl::buffer<SCALAR_T>&, sycl::buffer<SCALAR_T>&, sycl::buffer<SCALAR_T>&, \
        sycl::buffer<SCALAR_T>&);                                                                 \
                                                                                                  \
    /* USM API */                                                                                 \
    template ONEMKL_EXPORT sycl::event compute_forward<DESCRIPTOR_T>(DESCRIPTOR_T&, FORWARD_T*,   \
                                                                     depends_vec_t);              \
    template ONEMKL_EXPORT sycl::event compute_forward<DESCRIPTOR_T>(DESCRIPTOR_T&, SCALAR_T*,    \
                                                                     SCALAR_T*, depends_vec_t);   \
    template ONEMKL_EXPORT sycl::event compute_forward<DESCRIPTOR_T>(DESCRIPTOR_T&, FORWARD_T*,   \
                                                                     BACKWARD_T*, depends_vec_t); \
    template ONEMKL_EXPORT sycl::event compute_forward<DESCRIPTOR_T>(                             \
        DESCRIPTOR_T&, SCALAR_T*, SCALAR_T*, SCALAR_T*, SCALAR_T*, depends_vec_t);

ONEMKL_DFT_FORWARD_INSTANTIATIONS(desc_rf_t, float, float, std::complex<float>)
ONEMKL_DFT_FORWARD_INSTANTIATIONS(desc_cf_t, float, std::complex<float>, std::complex<float>)
ONEMKL_DFT_FORWARD_INSTANTIATIONS(desc_rd_t, double, double, std::complex<double>)
ONEMKL_DFT_FORWARD_INSTANTIATIONS(desc_cd_t, double, std::complex<double>, std::complex<double>)

#undef ONEMKL_DFT_FORWARD_INSTANTIATIONS
#undef ONEMKL_DFT_FORWARD_INSTANTIATIONS_REAL_ONLY
