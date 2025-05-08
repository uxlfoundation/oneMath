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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/types.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/detail/backends.hpp"

#include "oneapi/math/dft/detail/commit_impl.hpp"
#include "oneapi/math/dft/detail/types_impl.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"
#include "oneapi/math/dft/detail/mklgpu/onemath_dft_mklgpu.hpp"

#include "common_onemkl_conversion.hpp"
#include "dft/backends/mklgpu/mklgpu_helpers.hpp"
#include "../stride_helper.hpp"

// Intel(R) oneMKL headers
#include <mkl_version.h>
#if INTEL_MKL_VERSION < 20250000
#include <mkl/dfti.hpp>
namespace oneapi::math::dft::mklgpu::detail {
template <typename D, typename C, typename V>
void set_vector_value(D& desc, C p, V const& vec) {
    desc.set_value(p, vec.data());
}
} // namespace oneapi::math::dft::mklgpu::detail
#else
#include <mkl/dft.hpp>
namespace oneapi::math::dft::mklgpu::detail {
template <typename D, typename C, typename V>
void set_vector_value(D& desc, C p, V const& vec) {
    desc.set_value(p, vec);
}
} // namespace oneapi::math::dft::mklgpu::detail
#endif

// Intel oneMKL 2024.1 deprecates input/output strides.
#include <mkl_version.h>
#if INTEL_MKL_VERSION < 20240001
#error MKLGPU requires oneMath 2024.1 or later
#endif

/**
Note that in this file, the Intel oneMKL-GPU library's interface mirrors the
interface of this oneMath library. Consequently, the types under dft::TYPE are
Intel oneMKL types, and types under dft::detail::TYPE are from this library.
**/

namespace oneapi::math::dft::mklgpu {
namespace detail {

/// Commit impl class specialization for MKLGPU.
template <dft::detail::precision prec, dft::detail::domain dom>
class mklgpu_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    // Equivalent MKLGPU precision and domain from oneMath's precision / domain.
    static constexpr oneapi::mkl::dft::precision mklgpu_prec = to_mklgpu(prec);
    static constexpr oneapi::mkl::dft::domain mklgpu_dom = to_mklgpu(dom);

    // A pair of descriptors are needed because of the [[deprecated]]IN/OUTPUT_STRIDES vs F/BWD_STRIDES API.
    // Of the pair [0] is fwd DFT, [1] is backward DFT. If possible, the pointers refer to the same descriptor.
    // Both pointers must be valid.
    using mklgpu_descriptor_t = oneapi::mkl::dft::descriptor<mklgpu_prec, mklgpu_dom>;
    using descriptor_shptr_t = std::shared_ptr<mklgpu_descriptor_t>;
    using handle_t = std::pair<descriptor_shptr_t, descriptor_shptr_t>;

    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;

public:
    mklgpu_commit(sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::mklgpu,
                                                                config_values),
              handle(std::make_shared<mklgpu_descriptor_t>(config_values.dimensions), nullptr) {
        handle.second = handle.first; // Make sure the bwd pointer is valid.
        // MKLGPU does not throw an informative exception for the following:
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw math::exception("dft/backends/mklgpu", "commit",
                                      "Device does not support double precision.");
            }
        }
    }

    virtual void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        this->external_workspace_helper_ =
            oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);

        auto stride_choice = dft::detail::get_stride_api(config_values);
        throw_on_invalid_stride_api("MKLGPU commit", stride_choice);
        // A separate descriptor for each direction may not be required.
        bool one_descriptor = (stride_choice == dft::detail::stride_api::FB_STRIDES) ||
                              (config_values.input_strides == config_values.output_strides);
        // Make sure that second is always pointing to something new if this is a recommit.
        handle.second = handle.first;

        // Generate forward DFT descriptor. If using FWD/BWD_STRIDES API, only
        // one descriptor is needed.
        set_value(*handle.first, config_values, true, stride_choice);
        RETHROW_ONEMKL_EXCEPTIONS(handle.first->commit(this->get_queue()));

        // Generate backward DFT descriptor only if required.
        if (!one_descriptor) {
            handle.second = std::make_shared<mklgpu_descriptor_t>(config_values.dimensions);
            set_value(*handle.second, config_values, false, stride_choice);
            RETHROW_ONEMKL_EXCEPTIONS(handle.second->commit(this->get_queue()));
        }
    }

    void* get_handle() noexcept override {
        return &handle;
    }

    ~mklgpu_commit() override = default;

    virtual void set_workspace(scalar_type* usm_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, usm_workspace);
        RETHROW_ONEMKL_EXCEPTIONS(handle.first->set_workspace(usm_workspace));
        if (handle.first != handle.second) {
            RETHROW_ONEMKL_EXCEPTIONS(handle.second->set_workspace(usm_workspace));
        }
    }

    virtual void set_workspace(sycl::buffer<scalar_type>& buffer_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, buffer_workspace);
        RETHROW_ONEMKL_EXCEPTIONS(handle.first->set_workspace(buffer_workspace));
        if (handle.first != handle.second) {
            RETHROW_ONEMKL_EXCEPTIONS(handle.second->set_workspace(buffer_workspace));
        }
    }

#define BACKEND mklgpu
#include "../backend_compute_signature.cxx"
#undef BACKEND

private:
    // The native MKLGPU class.
    handle_t handle;

    void set_value(mklgpu_descriptor_t& desc, const dft::detail::dft_values<prec, dom>& config,
                   bool assume_fwd_dft, dft::detail::stride_api stride_choice) {
        using onemath_param = dft::detail::config_param;
        using backend_param = oneapi::mkl::dft::config_param;

        // The following are read-only:
        // Dimension, forward domain, precision, commit status.
        // Lengths are supplied at descriptor construction time.
        desc.set_value(backend_param::FORWARD_SCALE, config.fwd_scale);
        desc.set_value(backend_param::BACKWARD_SCALE, config.bwd_scale);
        desc.set_value(backend_param::NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        if (config.complex_storage != dft::detail::config_value::COMPLEX_COMPLEX) {
            throw math::unimplemented("dft/backends/mklgpu", "commit",
                                      "MKLGPU only supports complex-complex complex storage.");
        }
        if (config.real_storage != dft::detail::config_value::REAL_REAL) {
            throw math::unimplemented("dft/backends/mklgpu", "commit",
                                      "MKLGPU only supports real-real real storage.");
        }
        if (config.conj_even_storage != dft::detail::config_value::COMPLEX_COMPLEX) {
            throw math::unimplemented(
                "dft/backends/mklgpu", "commit",
                "MKLGPU only supports complex-complex conjugate even storage.");
        }
        desc.set_value(backend_param::PLACEMENT,
                       to_mklgpu<onemath_param::PLACEMENT>(config.placement));

        if (stride_choice == dft::detail::stride_api::FB_STRIDES) {
            if (config.fwd_strides[0] != 0 || config.fwd_strides[0] != 0) {
                throw math::unimplemented("dft/backends/mklgpu", "commit",
                                          "MKLGPU does not support nonzero offsets.");
            }
            set_vector_value(desc, backend_param::FWD_STRIDES, config.fwd_strides);
            set_vector_value(desc, backend_param::BWD_STRIDES, config.bwd_strides);
        }
        else {
            if (config.input_strides[0] != 0 || config.output_strides[0] != 0) {
                throw math::unimplemented("dft/backends/mklgpu", "commit",
                                          "MKLGPU does not support nonzero offsets.");
            }
            if (assume_fwd_dft) {
                set_vector_value(desc, backend_param::FWD_STRIDES, config.input_strides);
                set_vector_value(desc, backend_param::BWD_STRIDES, config.output_strides);
            }
            else {
                set_vector_value(desc, backend_param::FWD_STRIDES, config.output_strides);
                set_vector_value(desc, backend_param::BWD_STRIDES, config.input_strides);
            }
        }
        desc.set_value(backend_param::FWD_DISTANCE, config.fwd_dist);
        desc.set_value(backend_param::BWD_DISTANCE, config.bwd_dist);
        if (config.workspace_placement == dft::detail::config_value::WORKSPACE_EXTERNAL) {
            // Setting WORKSPACE_INTERNAL (default) causes FFT_INVALID_DESCRIPTOR.
            desc.set_value(backend_param::WORKSPACE, to_mklgpu<onemath_param::WORKSPACE_PLACEMENT>(
                                                         config.workspace_placement));
        }
        // Setting the ordering causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.ordering != dft::detail::config_value::ORDERED) {
            throw math::invalid_argument("dft/backends/mklgpu", "commit",
                                         "MKLGPU only supports ordered ordering.");
        }
        // Setting the transpose causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.transpose != false) {
            throw math::invalid_argument("dft/backends/mklgpu", "commit",
                                         "MKLGPU only supports non-transposed.");
        }
    }

    // This is called by the workspace_helper, and is not part of the user API.
    virtual std::int64_t get_workspace_external_bytes_impl() override {
        std::int64_t workspaceSizeFwd = 0, workspaceSizeBwd = 0;
        using backend_param = oneapi::mkl::dft::config_param;
        handle.first->get_value(backend_param::WORKSPACE_BYTES, &workspaceSizeFwd);
        handle.second->get_value(backend_param::WORKSPACE_BYTES, &workspaceSizeBwd);
        return std::max(workspaceSizeFwd, workspaceSizeFwd);
    }
};
} // namespace detail

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::mklgpu_commit<prec, dom>(sycl_queue, desc.get_values());
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

} // namespace oneapi::math::dft::mklgpu
