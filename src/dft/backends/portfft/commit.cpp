/*******************************************************************************
* Copyright Codeplay Software Ltd
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

#include <array>
#include <optional>

#include <portfft/portfft.hpp>

#include "oneapi/math/exceptions.hpp"

#include "oneapi/math/dft/detail/commit_impl.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"
#include "oneapi/math/dft/detail/portfft/onemath_dft_portfft.hpp"
#include "oneapi/math/dft/types.hpp"

#include "../stride_helper.hpp"

#include "portfft_helper.hpp"

// alias to avoid ambiguity
namespace pfft = portfft;

namespace oneapi::math::dft::portfft {
namespace detail {

template <dft::precision prec, dft::domain dom>
class portfft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;
    using fwd_type = typename dft::detail::commit_impl<prec, dom>::fwd_type;
    using bwd_type = typename dft::detail::commit_impl<prec, dom>::bwd_type;
    using descriptor_type = typename dft::detail::descriptor<prec, dom>;

    static constexpr pfft::domain domain =
        dom == dft::domain::REAL ? pfft::domain::REAL : pfft::domain::COMPLEX;
    // since only complex-to-complex transforms are supported, we expect both directions to be valid or neither.
    std::array<storage_type<descriptor_type>, 2> committed_descriptors = { std::nullopt,
                                                                           std::nullopt };

public:
    portfft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::portfft,
                                                                config_values) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw math::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // not available in portFFT:
        this->external_workspace_helper_ =
            oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);
        if (config_values.workspace != config_value::ALLOW) {
            throw math::unimplemented("dft/backends/portfft", __FUNCTION__,
                                      "portFFT only supports ALLOW for the WORKSPACE parameter");
        }
        if (config_values.ordering != config_value::ORDERED) {
            throw math::unimplemented("dft/backends/portfft", __FUNCTION__,
                                      "portFFT only supports ORDERED for the ORDERING parameter");
        }
        if (config_values.transpose) {
            throw math::unimplemented("dft/backends/portfft", __FUNCTION__,
                                      "portFFT does not supported transposed output");
        }

        auto stride_api_choice = dft::detail::get_stride_api(config_values);
        dft::detail::throw_on_invalid_stride_api("portFFT commit", stride_api_choice);
        dft::detail::stride_vectors<std::int64_t> stride_vecs(config_values, stride_api_choice);

        // forward descriptor
        pfft::descriptor<scalar_type, domain> fwd_desc(
            { config_values.dimensions.cbegin(), config_values.dimensions.cend() });
        fwd_desc.forward_scale = config_values.fwd_scale;
        fwd_desc.backward_scale = config_values.bwd_scale;
        fwd_desc.number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);
        fwd_desc.complex_storage = config_values.complex_storage == config_value::COMPLEX_COMPLEX
                                       ? pfft::complex_storage::INTERLEAVED_COMPLEX
                                       : pfft::complex_storage::SPLIT_COMPLEX;
        fwd_desc.placement = config_values.placement == config_value::INPLACE
                                 ? pfft::placement::IN_PLACE
                                 : pfft::placement::OUT_OF_PLACE;
        fwd_desc.forward_offset = static_cast<std::size_t>(stride_vecs.offset_fwd_in);
        fwd_desc.backward_offset = static_cast<std::size_t>(stride_vecs.offset_fwd_out);
        fwd_desc.forward_strides = { stride_vecs.fwd_in.cbegin() + 1, stride_vecs.fwd_in.cend() };
        fwd_desc.backward_strides = { stride_vecs.fwd_out.cbegin() + 1,
                                      stride_vecs.fwd_out.cend() };
        fwd_desc.forward_distance = static_cast<std::size_t>(config_values.fwd_dist);
        fwd_desc.backward_distance = static_cast<std::size_t>(config_values.bwd_dist);

        // backward descriptor
        pfft::descriptor<scalar_type, domain> bwd_desc(
            { config_values.dimensions.cbegin(), config_values.dimensions.cend() });
        bwd_desc.forward_scale = config_values.fwd_scale;
        bwd_desc.backward_scale = config_values.bwd_scale;
        bwd_desc.number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);
        bwd_desc.complex_storage = config_values.complex_storage == config_value::COMPLEX_COMPLEX
                                       ? pfft::complex_storage::INTERLEAVED_COMPLEX
                                       : pfft::complex_storage::SPLIT_COMPLEX;
        bwd_desc.placement = config_values.placement == config_value::INPLACE
                                 ? pfft::placement::IN_PLACE
                                 : pfft::placement::OUT_OF_PLACE;
        bwd_desc.forward_offset = static_cast<std::size_t>(stride_vecs.offset_bwd_out);
        bwd_desc.backward_offset = static_cast<std::size_t>(stride_vecs.offset_bwd_in);
        bwd_desc.forward_strides = { stride_vecs.bwd_out.cbegin() + 1, stride_vecs.bwd_out.cend() };
        bwd_desc.backward_strides = { stride_vecs.bwd_in.cbegin() + 1, stride_vecs.bwd_in.cend() };
        bwd_desc.forward_distance = static_cast<std::size_t>(config_values.fwd_dist);
        bwd_desc.backward_distance = static_cast<std::size_t>(config_values.bwd_dist);

        try {
            auto q = this->get_queue();
            committed_descriptors[0] = fwd_desc.commit(q);
            committed_descriptors[1] = bwd_desc.commit(q);
        }
        catch (const pfft::unsupported_configuration& e) {
            throw oneapi::math::unimplemented("dft/backends/portfft", __FUNCTION__, e.what());
        }
    }

    ~portfft_commit() override = default;

    void* get_handle() noexcept override {
        return committed_descriptors.data();
    }

    // All the compute functions are implementated here so they are in the same translation unit as the commit function.
    // If the use of the kernel bundle is in a seperate translation unit from the one it was translated in, the runtime can fail to find it.

    // forward inplace COMPLEX_COMPLEX
    void forward_ip_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& inout) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<fwd_type, 1>>(
            "compute_forward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            detail::get_descriptors(desc)[0]->compute_forward(inout);
        }
    }
    sycl::event forward_ip_cc(descriptor_type& desc, fwd_type* inout,
                              const std::vector<sycl::event>& dependencies) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<fwd_type*>("compute_forward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            return detail::get_descriptors(desc)[0]->compute_forward(inout, dependencies);
        }
        else {
            return {};
        }
    }

    // forward inplace REAL_REAL
    void forward_ip_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>&,
                       sycl::buffer<scalar_type, 1>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<scalar_type, 1>>(
            "compute_forward");
        throw oneapi::math::unimplemented("DFT", "compute_forward(desc, inout_re, inout_im)",
                                          "portFFT does not support real-real complex storage.");
    }
    sycl::event forward_ip_rr(descriptor_type& desc, scalar_type*, scalar_type*,
                              const std::vector<sycl::event>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<scalar_type*>("compute_forward");
        throw oneapi::math::unimplemented("DFT",
                                          "compute_forward(desc, inout_re, inout_im, dependencies)",
                                          "portFFT does not support real-real complex storage.");
    }

    // forward out-of-place COMPLEX_COMPLEX
    void forward_op_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& in,
                       sycl::buffer<bwd_type, 1>& out) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<fwd_type, 1>>(
            "compute_forward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            detail::get_descriptors(desc)[0]->compute_forward(in, out);
        }
    }
    sycl::event forward_op_cc(descriptor_type& desc, fwd_type* in, bwd_type* out,
                              const std::vector<sycl::event>& dependencies) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<fwd_type*>("compute_forward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            return detail::get_descriptors(desc)[0]->compute_forward(in, out, dependencies);
        }
        else {
            return {};
        }
    }

    // forward out-of-place REAL_REAL
    void forward_op_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>&,
                       sycl::buffer<scalar_type, 1>&, sycl::buffer<scalar_type, 1>&,
                       sycl::buffer<scalar_type, 1>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<scalar_type, 1>>(
            "compute_forward");
        throw oneapi::math::unimplemented("DFT",
                                          "compute_forward(desc, in_re, in_im, out_re, out_im)",
                                          "portFFT does not support real-real complex storage.");
    }
    sycl::event forward_op_rr(descriptor_type& desc, scalar_type*, scalar_type*, scalar_type*,
                              scalar_type*, const std::vector<sycl::event>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<scalar_type*>("compute_forward");
        throw oneapi::math::unimplemented(
            "DFT", "compute_forward(desc, in_re, in_im, out_re, out_im, dependencies)",
            "portFFT does not support real-real complex storage.");
    }

    // backward inplace COMPLEX_COMPLEX
    void backward_ip_cc(descriptor_type& desc, sycl::buffer<fwd_type, 1>& inout) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<fwd_type, 1>>(
            "compute_backward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            detail::get_descriptors(desc)[1]->compute_backward(inout);
        }
    }
    sycl::event backward_ip_cc(descriptor_type& desc, fwd_type* inout,
                               const std::vector<sycl::event>& dependencies) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<fwd_type*>("compute_backward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            return detail::get_descriptors(desc)[1]->compute_backward(inout, dependencies);
        }
        else {
            return {};
        }
    }

    // backward inplace REAL_REAL
    void backward_ip_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>&,
                        sycl::buffer<scalar_type, 1>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<scalar_type, 1>>(
            "compute_backward");
        throw oneapi::math::unimplemented("DFT", "compute_backward(desc, inout_re, inout_im)",
                                          "portFFT does not support real-real complex storage.");
    }
    sycl::event backward_ip_rr(descriptor_type& desc, scalar_type*, scalar_type*,
                               const std::vector<sycl::event>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<scalar_type*>(
            "compute_backward");
        throw oneapi::math::unimplemented(
            "DFT", "compute_backward(desc, inout_re, inout_im, dependencies)",
            "portFFT does not support real-real complex storage.");
    }

    // backward out-of-place COMPLEX_COMPLEX
    void backward_op_cc(descriptor_type& desc, sycl::buffer<bwd_type, 1>& in,
                        sycl::buffer<fwd_type, 1>& out) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<bwd_type, 1>>(
            "compute_backward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            detail::get_descriptors(desc)[1]->compute_backward(in, out);
        }
    }
    sycl::event backward_op_cc(descriptor_type& desc, bwd_type* in, fwd_type* out,
                               const std::vector<sycl::event>& dependencies) override {
        constexpr auto pfft_domain = detail::to_pfft_domain<descriptor_type>::type::value;
        dft::detail::get_commit(desc)->template compute_call_throw<bwd_type*>("compute_backward");

        if constexpr (pfft_domain == pfft::domain::COMPLEX) {
            return detail::get_descriptors(desc)[1]->compute_backward(in, out, dependencies);
        }
        else {
            return {};
        }
    }

    // backward out-of-place REAL_REAL
    void backward_op_rr(descriptor_type& desc, sycl::buffer<scalar_type, 1>&,
                        sycl::buffer<scalar_type, 1>&, sycl::buffer<scalar_type, 1>&,
                        sycl::buffer<scalar_type, 1>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<sycl::buffer<scalar_type, 1>>(
            "compute_backward");
        throw oneapi::math::unimplemented("DFT",
                                          "compute_backward(desc, in_re, in_im, out_re, out_im)",
                                          "portFFT does not support real-real complex storage.");
    }
    sycl::event backward_op_rr(descriptor_type& desc, scalar_type*, scalar_type*, scalar_type*,
                               scalar_type*, const std::vector<sycl::event>&) override {
        dft::detail::get_commit(desc)->template compute_call_throw<scalar_type*>(
            "compute_backward");
        throw oneapi::math::unimplemented(
            "DFT", "compute_backward(desc, in_re, in_im, out_re, out_im, deps)",
            "portFFT does not support real-real complex storage.");
    }
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::portfft_commit<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

} // namespace oneapi::math::dft::portfft
