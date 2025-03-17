/*******************************************************************************
* Copyright 2025 SiPearl
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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"

#include "oneapi/math/dft/descriptor.hpp"
#include "oneapi/math/dft/detail/armpl/onemath_dft_armpl.hpp"
#include "oneapi/math/dft/types.hpp"

#include "dft/backends/armpl/armpl_helpers.hpp"

#include <fftw3.h>

namespace oneapi::math::dft::armpl {
namespace detail {
//forward declaration
template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_bwd(dft::detail::commit_impl<prec, dom>* commit);

// BUFFER version
// backward a ARMPL DFT call to the backend, checking that the commit impl is valid.
template <dft::precision prec, dft::domain dom>
inline void check_bwd_commit(dft::descriptor<prec, dom>& desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::armpl) {
        throw math::invalid_argument("DFT", "computer_backward",
                                     "DFT descriptor has not been commited for ARMPL");
    }
}

// Throw an math::invalid_argument if the runtime param in the descriptor does not match
// the expected value.
template <dft::detail::config_param Param, dft::detail::config_value Expected, typename DescT>
inline auto expect_config(DescT& desc, const char* message) {
    dft::detail::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw math::invalid_argument("DFT", "compute_backward", message);
    }
}

template <dft::precision prec, dft::domain dom>
auto get_bwd_plan(dft::detail::commit_impl<prec, dom>* commit) {
    using planType = typename std::conditional_t<prec == oneapi::math::dft::precision::SINGLE,
                                                 fftwf_plan, fftw_plan>;
    return static_cast<std::optional<planType>*>(commit->get_handle())[1].value();
}

} // namespace detail

template <typename in_data_type, typename out_data_type, typename planType>
void ArmplExecuteBackward(const planType plan, in_data_type* in, out_data_type* out) {
    constexpr bool is_real = std::is_floating_point_v<in_data_type>;
    using single_type = std::conditional_t<is_real, float, std::complex<float>>;
    constexpr bool is_single = std::is_same_v<in_data_type, single_type>;
    if constexpr (is_single) {
        fftwf_execute_dft(plan, reinterpret_cast<fftwf_complex*>(in),
                          reinterpret_cast<fftwf_complex*>(out));
    }
    else {
        fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(in),
                         reinterpret_cast<fftw_complex*>(out));
    }
}

template <typename in_data_type, typename out_data_type, typename planType>
void ArmplExecuteBackwardSplit(const planType plan, in_data_type* r_in, in_data_type* i_in,
                               out_data_type* r_out, out_data_type* i_out) {
    constexpr bool is_real = std::is_floating_point_v<in_data_type>;
    using single_type = std::conditional_t<is_real, float, std::complex<float>>;
    constexpr bool is_single = std::is_same_v<in_data_type, single_type>;

    if constexpr (is_single) {
        fftwf_execute_split_dft(plan, i_in, r_in, i_out, r_out);
    }
    else {
        fftw_execute_split_dft(plan, i_in, r_in, i_out, r_out);
    }
}

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<fwd<descriptor_type>, 1>& inout) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");

    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    cpu_queue.submit([&](sycl::handler& cgh) {
        auto inout_acc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        detail::host_task<class host_kernel_inplace>(cgh, [=]() {
            ArmplExecuteBackward(plan, detail::acc_to_ptr(inout_acc) + offsets[0],
                                 detail::acc_to_ptr(inout_acc) + offsets[1]);
        });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<scalar<descriptor_type>, 1>& inout_re,
                                     sycl::buffer<scalar<descriptor_type>, 1>& inout_im) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    cpu_queue.submit([&](sycl::handler& cgh) {
        auto re_acc = inout_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto im_acc = inout_im.template get_access<sycl::access::mode::read_write>(cgh);

        detail::host_task<class host_kernel_split_inplace>(cgh, [=]() {
            ArmplExecuteBackwardSplit(plan, detail::acc_to_ptr(re_acc) + offsets[0],
                                      detail::acc_to_ptr(im_acc) + offsets[0],
                                      detail::acc_to_ptr(re_acc) + offsets[1],
                                      detail::acc_to_ptr(im_acc) + offsets[1]);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<bwd<descriptor_type>, 1>& in,
                                     sycl::buffer<fwd<descriptor_type>, 1>& out) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    cpu_queue.submit([&](sycl::handler& cgh) {
        auto in_acc = in.template get_access<sycl::access::mode::read>(cgh);
        auto out_acc = out.template get_access<sycl::access::mode::write>(cgh);

        detail::host_task<class host_kernel_outofplace>(cgh, [=]() {
            auto in_ptr = const_cast<bwd<descriptor_type>*>(detail::acc_to_ptr(in_acc));
            ArmplExecuteBackward(plan, in_ptr + offsets[0],
                                 detail::acc_to_ptr(out_acc) + offsets[1]);
        });
    });
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<scalar<descriptor_type>, 1>& in_re,
                                     sycl::buffer<scalar<descriptor_type>, 1>& in_im,
                                     sycl::buffer<scalar<descriptor_type>, 1>& out_re,
                                     sycl::buffer<scalar<descriptor_type>, 1>& out_im) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");

    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    cpu_queue.submit([&](sycl::handler& cgh) {
        auto inre_acc = in_re.template get_access<sycl::access::mode::read>(cgh);
        auto inim_acc = in_im.template get_access<sycl::access::mode::read>(cgh);
        auto outre_acc = out_re.template get_access<sycl::access::mode::write>(cgh);
        auto outim_acc = out_im.template get_access<sycl::access::mode::write>(cgh);

        detail::host_task<class host_kernel_split_outofplace>(cgh, [=]() {
            auto inre_ptr = const_cast<scalar<descriptor_type>*>(detail::acc_to_ptr(inre_acc));
            auto inim_ptr = const_cast<scalar<descriptor_type>*>(detail::acc_to_ptr(inim_acc));
            ArmplExecuteBackwardSplit(plan, inre_ptr + offsets[0], inim_ptr + offsets[0],
                                      detail::acc_to_ptr(outre_acc) + offsets[1],
                                      detail::acc_to_ptr(outim_acc) + offsets[1]);
        });
    });
}

//USM version

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                            const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    return cpu_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        detail::host_task<class host_usm_kernel_inplace>(
            cgh, [=]() { ArmplExecuteBackward(plan, inout + offsets[0], inout + offsets[1]); });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc,
                                            scalar<descriptor_type>* inout_re,
                                            scalar<descriptor_type>* inout_im,
                                            const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    return cpu_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        detail::host_task<class host_usm_kernel_split_inplace>(cgh, [=]() {
            ArmplExecuteBackwardSplit(plan, inout_re + offsets[0], inout_im + offsets[0],
                                      inout_re + offsets[1], inout_im + offsets[1]);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, bwd<descriptor_type>* in,
                                            fwd<descriptor_type>* out,
                                            const std::vector<sycl::event>& dependencies) {
    // Check: inplace, complex storage
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    return cpu_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        detail::host_task<class host_usm_kernel_outofplace>(
            cgh, [=]() { ArmplExecuteBackward(plan, in + offsets[0], out + offsets[1]); });
    });
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, scalar<descriptor_type>* in_re,
                                            scalar<descriptor_type>* in_im,
                                            scalar<descriptor_type>* out_re,
                                            scalar<descriptor_type>* out_im,
                                            const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");

    auto commit_handle = dft::detail::get_commit(desc);
    detail::check_bwd_commit(desc);
    sycl::queue& cpu_queue{ commit_handle->get_queue() };
    auto plan = detail::get_bwd_plan(commit_handle);
    auto offsets = detail::get_offsets_bwd(commit_handle);

    return cpu_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        detail::host_task<class host_usm_kernel_split_outofplace>(cgh, [=]() {
            ArmplExecuteBackwardSplit(plan, in_re + offsets[0], in_im + offsets[0],
                                      out_re + offsets[1], out_im + offsets[1]);
        });
    });
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace oneapi::math::dft::armpl
