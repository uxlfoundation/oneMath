/*******************************************************************************
* Copyright 2023 Codeplay Software
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

// Commit

template <dft::detail::precision prec, dft::detail::domain dom>
ONEMKL_EXPORT dft::detail::commit_impl<prec, dom>* create_commit(
    const dft::detail::descriptor<prec, dom>& desc, sycl::queue& sycl_queue);

// BUFFER version

template <typename descriptor_type>
using scalar = typename detail::descriptor_info<descriptor_type>::scalar_type;
template <typename descriptor_type>
using fwd = typename detail::descriptor_info<descriptor_type>::forward_type;
template <typename descriptor_type>
using bwd = typename detail::descriptor_info<descriptor_type>::backward_type;

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc,
                                   sycl::buffer<fwd<descriptor_type>, 1>& inout);

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc,
                                   sycl::buffer<scalar<descriptor_type>, 1>& inout_re,
                                   sycl::buffer<scalar<descriptor_type>, 1>& inout_im);

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc, sycl::buffer<fwd<descriptor_type>, 1>& in,
                                   sycl::buffer<bwd<descriptor_type>, 1>& out);

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc,
                                   sycl::buffer<scalar<descriptor_type>, 1>& in_re,
                                   sycl::buffer<scalar<descriptor_type>, 1>& in_im,
                                   sycl::buffer<scalar<descriptor_type>, 1>& out_re,
                                   sycl::buffer<scalar<descriptor_type>, 1>& out_im);

//USM version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                          const std::vector<sycl::event>& dependencies);

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, scalar<descriptor_type>* inout_re,
                                          scalar<descriptor_type>* inout_im,
                                          const std::vector<sycl::event>& dependencies);

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, fwd<descriptor_type>* in,
                                          bwd<descriptor_type>* out,
                                          const std::vector<sycl::event>& dependencies);

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, scalar<descriptor_type>* in_re,
                                          scalar<descriptor_type>* in_im,
                                          scalar<descriptor_type>* out_re,
                                          scalar<descriptor_type>* out_im,
                                          const std::vector<sycl::event>& dependencies);

// BUFFER version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& desc,
                                    sycl::buffer<fwd<descriptor_type>, 1>& inout);

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& desc,
                                    sycl::buffer<scalar<descriptor_type>, 1>& inout_re,
                                    sycl::buffer<scalar<descriptor_type>, 1>& inout_im);

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& desc,
                                    sycl::buffer<bwd<descriptor_type>, 1>& in,
                                    sycl::buffer<fwd<descriptor_type>, 1>& out);

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_backward(descriptor_type& desc,
                                    sycl::buffer<scalar<descriptor_type>, 1>& in_re,
                                    sycl::buffer<scalar<descriptor_type>, 1>& in_im,
                                    sycl::buffer<scalar<descriptor_type>, 1>& out_re,
                                    sycl::buffer<scalar<descriptor_type>, 1>& out_im);

//USM version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                           const std::vector<sycl::event>& dependencies);

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& desc, scalar<descriptor_type>* inout_re,
                                           scalar<descriptor_type>* inout_im,
                                           const std::vector<sycl::event>& dependencies);

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& desc, bwd<descriptor_type>* in,
                                           fwd<descriptor_type>* out,
                                           const std::vector<sycl::event>& dependencies);

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_backward(descriptor_type& desc, scalar<descriptor_type>* in_re,
                                           scalar<descriptor_type>* in_im,
                                           scalar<descriptor_type>* out_re,
                                           scalar<descriptor_type>* out_im,
                                           const std::vector<sycl::event>& dependencies);
