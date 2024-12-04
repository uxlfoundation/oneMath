/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

#include "oneapi/math/dft/detail/rocfft/onemath_dft_rocfft.hpp"
#include "oneapi/math/dft/descriptor.hpp"

#include "execute_helper.hpp"
#include "../../execute_helper_generic.hpp"
#include "rocfft_handle.hpp"

#include <rocfft.h>
#include <hip/hip_runtime_api.h>

namespace oneapi::math::dft::rocfft {
namespace detail {
//forward declaration
template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_bwd(dft::detail::commit_impl<prec, dom>* commit);

template <dft::precision prec, dft::domain dom>
rocfft_plan get_bwd_plan(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<rocfft_handle*>(commit->get_handle())[1].plan.value();
}

template <dft::precision prec, dft::domain dom>
rocfft_execution_info get_bwd_info(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<rocfft_handle*>(commit->get_handle())[1].info.value();
}
} // namespace detail
// BUFFER version

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<fwd<descriptor_type>, 1>& inout) {
    const std::string func_name = "compute_backward(desc, inout)";
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    if constexpr (std::is_floating_point_v<fwd<descriptor_type>>) {
        offsets[0] *= 2; // offset is supplied in complex but we offset scalar pointer
    }
    if (offsets[0] != offsets[1]) {
        throw oneapi::math::unimplemented(
            "DFT", func_name,
            "rocFFT requires input and output offsets (first value in strides) to be equal for in-place transforms!");
    }

    queue.submit([&](sycl::handler& cgh) {
        auto inout_acc = inout.template get_access<sycl::access::mode::read_write>(cgh);
        commit->add_buffer_workspace_dependency_if_rqd("compute_backward", cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            auto stream = detail::setup_stream(func_name, ih, info);

            auto inout_native = reinterpret_cast<void*>(
                reinterpret_cast<fwd<descriptor_type>*>(detail::native_mem(ih, inout_acc)) +
                offsets[0]);
            detail::execute_checked(func_name, stream, plan, &inout_native, nullptr, info);
        });
    });
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<scalar<descriptor_type>, 1>& inout_re,
                                     sycl::buffer<scalar<descriptor_type>, 1>& inout_im) {
    const std::string func_name = "compute_backward(desc, inout_re, inout_im)";
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    if (offsets[0] != offsets[1]) {
        throw oneapi::math::unimplemented(
            "DFT", func_name,
            "rocFFT requires input and output offsets (first value in strides) to be equal for in-place transforms!");
    }

    queue.submit([&](sycl::handler& cgh) {
        auto inout_re_acc = inout_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto inout_im_acc = inout_im.template get_access<sycl::access::mode::read_write>(cgh);
        commit->add_buffer_workspace_dependency_if_rqd("compute_backward", cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void*, 2> inout_native{
                reinterpret_cast<void*>(reinterpret_cast<scalar<descriptor_type>*>(
                                            detail::native_mem(ih, inout_re_acc)) +
                                        offsets[0]),
                reinterpret_cast<void*>(reinterpret_cast<scalar<descriptor_type>*>(
                                            detail::native_mem(ih, inout_im_acc)) +
                                        offsets[0])
            };
            detail::execute_checked(func_name, stream, plan, inout_native.data(), nullptr, info);
        });
    });
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT void compute_backward(descriptor_type& desc,
                                     sycl::buffer<bwd<descriptor_type>, 1>& in,
                                     sycl::buffer<fwd<descriptor_type>, 1>& out) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    queue.submit([&](sycl::handler& cgh) {
        auto in_acc = in.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_acc = out.template get_access<sycl::access::mode::read_write>(cgh);
        commit->add_buffer_workspace_dependency_if_rqd("compute_backward", cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in, out)";
            auto stream = detail::setup_stream(func_name, ih, info);

            auto in_native = reinterpret_cast<void*>(
                reinterpret_cast<bwd<descriptor_type>*>(detail::native_mem(ih, in_acc)) +
                offsets[0]);
            auto out_native = reinterpret_cast<void*>(
                reinterpret_cast<fwd<descriptor_type>*>(detail::native_mem(ih, out_acc)) +
                offsets[1]);
            detail::execute_checked(func_name, stream, plan, &in_native, &out_native, info);
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
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    queue.submit([&](sycl::handler& cgh) {
        auto in_re_acc = in_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto in_im_acc = in_im.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_re_acc = out_re.template get_access<sycl::access::mode::read_write>(cgh);
        auto out_im_acc = out_im.template get_access<sycl::access::mode::read_write>(cgh);
        commit->add_buffer_workspace_dependency_if_rqd("compute_backward", cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in_re, in_im, out_re, out_im)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void*, 2> in_native{
                reinterpret_cast<void*>(
                    reinterpret_cast<scalar<descriptor_type>*>(detail::native_mem(ih, in_re_acc)) +
                    offsets[0]),
                reinterpret_cast<void*>(
                    reinterpret_cast<scalar<descriptor_type>*>(detail::native_mem(ih, in_im_acc)) +
                    offsets[0])
            };
            std::array<void*, 2> out_native{
                reinterpret_cast<void*>(
                    reinterpret_cast<scalar<descriptor_type>*>(detail::native_mem(ih, out_re_acc)) +
                    offsets[1]),
                reinterpret_cast<void*>(
                    reinterpret_cast<scalar<descriptor_type>*>(detail::native_mem(ih, out_im_acc)) +
                    offsets[1])
            };
            detail::execute_checked(func_name, stream, plan, in_native.data(), out_native.data(),
                                    info);
        });
    });
}

//USM version

//In-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                            const std::vector<sycl::event>& deps) {
    const std::string func_name = "compute_backward(desc, inout, deps)";
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    if constexpr (std::is_floating_point_v<fwd<descriptor_type>>) {
        offsets[0] *= 2; // offset is supplied in complex but we offset scalar pointer
    }
    if (offsets[0] != offsets[1]) {
        throw oneapi::math::unimplemented(
            "DFT", func_name,
            "rocFFT requires input and output offsets (first value in strides) to be equal for in-place transforms!");
    }
    inout += offsets[0];

    sycl::event sycl_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        commit->depend_on_last_usm_workspace_event_if_rqd(cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            auto stream = detail::setup_stream(func_name, ih, info);

            void* inout_ptr = inout;
            detail::execute_checked(func_name, stream, plan, &inout_ptr, nullptr, info);
        });
    });
    commit->set_last_usm_workspace_event_if_rqd(sycl_event);
    return sycl_event;
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc,
                                            scalar<descriptor_type>* inout_re,
                                            scalar<descriptor_type>* inout_im,
                                            const std::vector<sycl::event>& deps) {
    const std::string func_name = "compute_backward(desc, inout_re, inout_im, deps)";
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    if (offsets[0] != offsets[1]) {
        throw oneapi::math::unimplemented(
            "DFT", func_name,
            "rocFFT requires input and output offsets (first value in strides) to be equal for in-place transforms!");
    }

    sycl::event sycl_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        commit->depend_on_last_usm_workspace_event_if_rqd(cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void*, 2> inout_native{ inout_re + offsets[0], inout_im + offsets[0] };
            detail::execute_checked(func_name, stream, plan, inout_native.data(), nullptr, info);
        });
    });
    commit->set_last_usm_workspace_event_if_rqd(sycl_event);
    return sycl_event;
}

//Out-of-place transform
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, bwd<descriptor_type>* in,
                                            fwd<descriptor_type>* out,
                                            const std::vector<sycl::event>& deps) {
    detail::expect_config<dft::config_param::PLACEMENT, dft::config_value::NOT_INPLACE>(
        desc, "Unexpected value for placement");
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    in += offsets[0];
    out += offsets[1];

    sycl::event sycl_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        commit->depend_on_last_usm_workspace_event_if_rqd(cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            const std::string func_name = "compute_backward(desc, in, out, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            void* in_ptr = in;
            void* out_ptr = out;
            detail::execute_checked(func_name, stream, plan, &in_ptr, &out_ptr, info);
        });
    });
    commit->set_last_usm_workspace_event_if_rqd(sycl_event);
    return sycl_event;
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMATH_EXPORT sycl::event compute_backward(descriptor_type& desc, scalar<descriptor_type>* in_re,
                                            scalar<descriptor_type>* in_im,
                                            scalar<descriptor_type>* out_re,
                                            scalar<descriptor_type>* out_im,
                                            const std::vector<sycl::event>& deps) {
    auto commit = detail::checked_get_commit(desc);
    auto queue = commit->get_queue();
    auto plan = detail::get_bwd_plan(commit);
    auto info = detail::get_bwd_info(commit);
    auto offsets = detail::get_offsets_bwd(commit);

    sycl::event sycl_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        commit->depend_on_last_usm_workspace_event_if_rqd(cgh);

        dft::detail::fft_enqueue_task(cgh, [=](sycl::interop_handle ih) {
            const std::string func_name =
                "compute_backward(desc, in_re, in_im, out_re, out_im, deps)";
            auto stream = detail::setup_stream(func_name, ih, info);

            std::array<void*, 2> in_native{ in_re + offsets[0], in_im + offsets[0] };
            std::array<void*, 2> out_native{ out_re + offsets[1], out_im + offsets[1] };
            detail::execute_checked(func_name, stream, plan, in_native.data(), out_native.data(),
                                    info);
        });
    });
    commit->set_last_usm_workspace_event_if_rqd(sycl_event);
    return sycl_event;
}

// Template function instantiations
#include "dft/backends/backend_backward_instantiations.cxx"

} // namespace oneapi::math::dft::rocfft
