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

#ifndef _ONEMATH_DFT_COMMIT_IMPL_HPP_
#define _ONEMATH_DFT_COMMIT_IMPL_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "descriptor_impl.hpp"
#include "external_workspace_helper.hpp"

namespace oneapi::math {
enum class backend;
}

namespace oneapi::math::dft::detail {

template <precision prec, domain dom>
class dft_values;

template <precision prec, domain dom>
class commit_impl {
    sycl::queue queue_;
    math::backend backend_;

public:
    using descriptor_type = typename oneapi::math::dft::detail::descriptor<prec, dom>;
    using fwd_type = typename descriptor_info<descriptor_type>::forward_type;
    using bwd_type = typename descriptor_info<descriptor_type>::backward_type;
    using scalar_type = typename descriptor_info<descriptor_type>::scalar_type;

protected:
    external_workspace_helper<prec, dom> external_workspace_helper_;

public:
    commit_impl(sycl::queue queue, math::backend backend,
                const dft::detail::dft_values<prec, dom>& config_values)
            : queue_(queue),
              backend_(backend),
              external_workspace_helper_(config_values.workspace_placement ==
                                         dft::detail::config_value::WORKSPACE_EXTERNAL) {}

    // rule of three
    commit_impl(const commit_impl& other) = delete;
    commit_impl& operator=(const commit_impl& other) = delete;
    virtual ~commit_impl() = default;

    sycl::queue& get_queue() noexcept {
        return queue_;
    }

    math::backend get_backend() const noexcept {
        return backend_;
    }

    virtual void* get_handle() noexcept = 0;

    virtual void commit(const dft_values<prec, dom>&) = 0;

    inline std::int64_t get_workspace_external_bytes() {
        return external_workspace_helper_.get_rqd_workspace_bytes(*this);
    }

    // set_workspace should be overridden for any backend that enables external workspaces.
    // If these are overridden, get_workspace_external_bytes_impl must also be overridden.
    // For backends that do not support external workspaces, these functions do not need to be overridden.
    // When not overridden, external workspace support is faked: an external workspace can be set,
    // and errors will be generated according to the specificiation,
    // but the required workspace size will always be zero, and any given workspace will not actually be used.
    virtual void set_workspace(scalar_type* usm_workspace) {
        external_workspace_helper_.set_workspace_throw(*this, usm_workspace);
    }
    virtual void set_workspace(sycl::buffer<scalar_type>& buffer_workspace) {
        external_workspace_helper_.set_workspace_throw(*this, buffer_workspace);
    }

    virtual void forward_ip_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& inout) = 0;
    virtual void forward_ip_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>& inout_re,
                               sycl::buffer<scalar_type, 1>& inout_im) = 0;
    virtual void forward_op_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& in,
                               sycl::buffer<bwd_type, 1>& out) = 0;
    virtual void forward_op_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>& in_re,
                               sycl::buffer<scalar_type, 1>& in_im,
                               sycl::buffer<scalar_type, 1>& out_re,
                               sycl::buffer<scalar_type, 1>& out_im) = 0;

    virtual sycl::event forward_ip_cc(descriptor_type& desc, fwd_type* inout,
                                      const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event forward_ip_rr(descriptor_type& desc, scalar_type* inout_re,
                                      scalar_type* inout_im,
                                      const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event forward_op_cc(descriptor_type& desc, fwd_type* in, bwd_type* out,
                                      const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event forward_op_rr(descriptor_type& desc, scalar_type* in_re, scalar_type* in_im,
                                      scalar_type* out_re, scalar_type* out_im,
                                      const std::vector<sycl::event>& dependencies) = 0;

    virtual void backward_ip_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& inout) = 0;
    virtual void backward_ip_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>& inout_re,
                                sycl::buffer<scalar_type, 1>& inout_im) = 0;
    virtual void backward_op_cc(descriptor_type& desc, sycl::buffer<bwd_type, 1>& in,
                                sycl::buffer<fwd_type, 1>& out) = 0;
    virtual void backward_op_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>& in_re,
                                sycl::buffer<scalar_type, 1>& in_im,
                                sycl::buffer<scalar_type, 1>& out_re,
                                sycl::buffer<scalar_type, 1>& out_im) = 0;

    virtual sycl::event backward_ip_cc(descriptor_type& desc, fwd_type* inout,
                                       const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event backward_ip_rr(descriptor_type& desc, scalar_type* inout_re,
                                       scalar_type* inout_im,
                                       const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event backward_op_cc(descriptor_type& desc, bwd_type* in, fwd_type* out,
                                       const std::vector<sycl::event>& dependencies) = 0;
    virtual sycl::event backward_op_rr(descriptor_type& desc, scalar_type* in_re,
                                       scalar_type* in_im, scalar_type* out_re, scalar_type* out_im,
                                       const std::vector<sycl::event>& dependencies) = 0;

    /** For compute calls, throw errors for the external workspace as required.
     * @tparam ArgTs The non-descriptor arg(s) for the compute call. First one is used to check
     * buffer or USM call.
     * @param function_name The function name to user in generated exceptions.
    */
    template <typename... ArgTs>
    void compute_call_throw(const char* function_name) {
        external_workspace_helper_.template compute_call_throw<ArgTs...>(function_name);
    }

    /** Create an accessor out of the workspace buffer when required, to ensure correct dependency
     *  management for the buffer. To be used by backends that don't natively support sycl::buffers.
     * @param function_name The function name to user in generated exceptions.
     * @param cgh The command group handler to associate the accessor with.
    */
    void add_buffer_workspace_dependency_if_rqd(const char* function_name, sycl::handler& cgh) {
        external_workspace_helper_.add_buffer_dependency_if_rqd(function_name, cgh);
    }

    /** If WORKSPACE_EXTERNAL is set, depend on the last USM workspace event added via set_last_usm_workspace_event.
     * @param cgh The command group handler to associate the accessor with.
    */
    void depend_on_last_usm_workspace_event_if_rqd(sycl::handler& cgh) {
        external_workspace_helper_.depend_on_last_usm_workspace_event_if_rqd(cgh);
    }

    /** If WORKSPACE_EXTERNAL is set, store the given event internally to allow it to be depended upon by
     * subsequent calls to depend_on_last_usm_workspace_event.
     * @param sycl_event The last usage of the USM workspace.
    */
    void set_last_usm_workspace_event_if_rqd(sycl::event& sycl_event) {
        external_workspace_helper_.set_last_usm_workspace_event_if_rqd(sycl_event);
    }

protected:
    friend class external_workspace_helper<prec, dom>;

    // This must be reimplemented for backends that support external workspaces.
    virtual std::int64_t get_workspace_external_bytes_impl() {
        return 0;
    }
};

} // namespace oneapi::math::dft::detail

#endif //_ONEMATH_DFT_COMMIT_IMPL_HPP_
