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

#ifndef _ONEMKL_DFT_FORWARD_HPP_
#define _ONEMKL_DFT_FORWARD_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "detail/types_impl.hpp"

namespace oneapi::mkl::dft {

//Buffer version

//In-place transform
template <typename descriptor_type, typename data_type>
void compute_forward(descriptor_type& desc, sycl::buffer<data_type, 1>& inout) {
    static_assert(detail::valid_compute_arg<descriptor_type, data_type>::value,
                  "unexpected type for data_type");

    using fwd_type = typename detail::descriptor_info<descriptor_type>::forward_type;
    auto type_corrected_inout = inout.template reinterpret<fwd_type, 1>(
        detail::reinterpret_range<data_type, fwd_type>(inout.size()));
    get_commit(desc)->forward_ip_cc(desc, type_corrected_inout);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type,
          std::enable_if_t<detail::valid_ip_realreal_impl<descriptor_type, data_type>, bool> = true>
void compute_forward(descriptor_type& desc, sycl::buffer<data_type, 1>& inout_re,
                     sycl::buffer<data_type, 1>& inout_im) {
    static_assert(detail::valid_compute_arg<descriptor_type, data_type>::value,
                  "unexpected type for data_type");

    using scalar_type = typename detail::descriptor_info<descriptor_type>::scalar_type;
    auto type_corrected_inout_re = inout_re.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<data_type, scalar_type>(inout_re.size()));
    auto type_corrected_inout_im = inout_im.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<data_type, scalar_type>(inout_im.size()));
    get_commit(desc)->forward_ip_rr(desc, type_corrected_inout_re, type_corrected_inout_im);
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
void compute_forward(descriptor_type& desc, sycl::buffer<input_type, 1>& in,
                     sycl::buffer<output_type, 1>& out) {
    static_assert(detail::valid_compute_arg<descriptor_type, input_type>::value,
                  "unexpected type for input_type");
    static_assert(detail::valid_compute_arg<descriptor_type, output_type>::value,
                  "unexpected type for output_type");

    using fwd_type = typename detail::descriptor_info<descriptor_type>::forward_type;
    using bwd_type = typename detail::descriptor_info<descriptor_type>::backward_type;
    auto type_corrected_in = in.template reinterpret<fwd_type, 1>(
        detail::reinterpret_range<input_type, fwd_type>(in.size()));
    auto type_corrected_out = out.template reinterpret<bwd_type, 1>(
        detail::reinterpret_range<output_type, bwd_type>(out.size()));
    get_commit(desc)->forward_op_cc(desc, type_corrected_in, type_corrected_out);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
void compute_forward(descriptor_type& desc, sycl::buffer<input_type, 1>& in_re,
                     sycl::buffer<input_type, 1>& in_im, sycl::buffer<output_type, 1>& out_re,
                     sycl::buffer<output_type, 1>& out_im) {
    static_assert(detail::valid_compute_arg<descriptor_type, input_type>::value,
                  "unexpected type for input_type");
    static_assert(detail::valid_compute_arg<descriptor_type, output_type>::value,
                  "unexpected type for output_type");

    using scalar_type = typename detail::descriptor_info<descriptor_type>::scalar_type;
    auto type_corrected_in_re = in_re.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<input_type, scalar_type>(in_re.size()));
    auto type_corrected_in_im = in_im.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<input_type, scalar_type>(in_im.size()));
    auto type_corrected_out_re = out_re.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<output_type, scalar_type>(out_re.size()));
    auto type_corrected_out_im = out_im.template reinterpret<scalar_type, 1>(
        detail::reinterpret_range<output_type, scalar_type>(out_im.size()));
    get_commit(desc)->forward_op_rr(desc, type_corrected_in_re, type_corrected_in_im,
                                    type_corrected_out_re, type_corrected_out_im);
}

//USM version

//In-place transform
template <typename descriptor_type, typename data_type>
sycl::event compute_forward(descriptor_type& desc, data_type* inout,
                            const std::vector<sycl::event>& dependencies = {}) {
    static_assert(detail::valid_compute_arg<descriptor_type, data_type>::value,
                  "unexpected type for data_type");

    using fwd_type = typename detail::descriptor_info<descriptor_type>::forward_type;
    return get_commit(desc)->forward_ip_cc(desc, reinterpret_cast<fwd_type*>(inout), dependencies);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename data_type,
          std::enable_if_t<detail::valid_ip_realreal_impl<descriptor_type, data_type>, bool> = true>
sycl::event compute_forward(descriptor_type& desc, data_type* inout_re, data_type* inout_im,
                            const std::vector<sycl::event>& dependencies = {}) {
    static_assert(detail::valid_compute_arg<descriptor_type, data_type>::value,
                  "unexpected type for data_type");
    using scalar_type = typename detail::descriptor_info<descriptor_type>::scalar_type;
    return get_commit(desc)->forward_ip_rr(desc, reinterpret_cast<scalar_type*>(inout_re),
                                           reinterpret_cast<scalar_type*>(inout_im), dependencies);
}

//Out-of-place transform
template <typename descriptor_type, typename input_type, typename output_type>
sycl::event compute_forward(descriptor_type& desc, input_type* in, output_type* out,
                            const std::vector<sycl::event>& dependencies = {}) {
    static_assert(detail::valid_compute_arg<descriptor_type, input_type>::value,
                  "unexpected type for input_type");
    static_assert(detail::valid_compute_arg<descriptor_type, output_type>::value,
                  "unexpected type for output_type");
    using fwd_type = typename detail::descriptor_info<descriptor_type>::forward_type;
    using bwd_type = typename detail::descriptor_info<descriptor_type>::backward_type;
    return get_commit(desc)->forward_op_cc(desc, reinterpret_cast<fwd_type*>(in),
                                           reinterpret_cast<bwd_type*>(out), dependencies);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type, typename input_type, typename output_type>
sycl::event compute_forward(descriptor_type& desc, input_type* in_re, input_type* in_im,
                            output_type* out_re, output_type* out_im,
                            const std::vector<sycl::event>& dependencies = {}) {
    static_assert(detail::valid_compute_arg<descriptor_type, input_type>::value,
                  "unexpected type for input_type");
    static_assert(detail::valid_compute_arg<descriptor_type, output_type>::value,
                  "unexpected type for output_type");

    using scalar_type = typename detail::descriptor_info<descriptor_type>::scalar_type;
    return get_commit(desc)->forward_op_rr(desc, reinterpret_cast<scalar_type*>(in_re),
                                           reinterpret_cast<scalar_type*>(in_im),
                                           reinterpret_cast<scalar_type*>(out_re),
                                           reinterpret_cast<scalar_type*>(out_im), dependencies);
}
} // namespace oneapi::mkl::dft

#endif // _ONEMKL_DFT_FORWARD_HPP_
