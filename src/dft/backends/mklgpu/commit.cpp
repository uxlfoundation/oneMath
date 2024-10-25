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

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/backends.hpp"

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/types_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"
#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"

#include "dft/backends/mklgpu/mklgpu_helpers.hpp"
#include "../stride_helper.hpp"

#include "mkl_version.h"
// MKLGPU header
#if INTEL_MKL_VERSION < 20250000
#include <oneapi/mkl/dfti.hpp>
#else
#include <oneapi/mkl/dft.hpp>
#endif

// MKL 2024.1 deprecates input/output strides.
#if INTEL_MKL_VERSION < 20240001
#error MKLGPU requires oneMKL 2024.1 or later
#endif

/**
Note that in this file, the Intel oneMKL closed-source library's interface mirrors the interface
of this OneMKL open-source library. Consequently, the types under dft::TYPE are closed-source oneMKL types,
and types under dft::detail::TYPE are from this library.
**/

namespace oneapi::mkl::dft::mklgpu {
namespace detail {

/// Commit impl class specialization for MKLGPU.
template <dft::detail::precision prec, dft::detail::domain dom>
class mklgpu_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    // Equivalent MKLGPU precision and domain from OneMKL's precision / domain.
    static constexpr dft::precision mklgpu_prec = to_mklgpu(prec);
    static constexpr dft::domain mklgpu_dom = to_mklgpu(dom);

    // A pair of descriptors are needed because of the [[deprecated]]IN/OUTPUT_STRIDES vs F/BWD_STRIDES API.
    // Of the pair [0] is fwd DFT, [1] is backward DFT. If possible, the pointers refer to the same desciptor.
    // Both pointers must be valid.
    using mklgpu_descriptor_t = dft::descriptor<mklgpu_prec, mklgpu_dom>;
    using descriptor_shptr_t = std::shared_ptr<mklgpu_descriptor_t>;
    using handle_t = std::pair<descriptor_shptr_t, descriptor_shptr_t>;

    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;

public:
    mklgpu_commit(sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::mkl::dft::detail::commit_impl<prec, dom>(queue, backend::mklgpu,
                                                               config_values),
              handle(std::make_shared<mklgpu_descriptor_t>(config_values.dimensions), nullptr) {
        handle.second = handle.first; // Make sure the bwd pointer is valid.
        // MKLGPU does not throw an informative exception for the following:
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw mkl::exception("dft/backends/mklgpu", "commit",
                                     "Device does not support double precision.");
            }
        }
    }

    virtual void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        this->external_workspace_helper_ =
            oneapi::mkl::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::mkl::dft::detail::config_value::WORKSPACE_EXTERNAL);

        auto stride_choice = dft::detail::get_stride_api(config_values);
        throw_on_invalid_stride_api("MKLGPU commit", stride_choice);
        // A separate descriptor for each direction may not be required.
        bool one_descriptor = (stride_choice == dft::detail::stride_api::FB_STRIDES) ||
                              (config_values.input_strides == config_values.output_strides);
        bool forward_good = true;
        // Make sure that second is always pointing to something new if this is a recommit.
        handle.second = handle.first;

        // Generate forward DFT descriptor. If using FWD/BWD_STRIDES API, only
        // one descriptor is needed.
        set_value(*handle.first, config_values, true, stride_choice);
        try {
            handle.first->commit(this->get_queue());
        }
        catch (const std::exception& mkl_exception) {
            // Catching the real Intel oneMKL exception causes headaches with naming
            forward_good = false;
            if (one_descriptor) {
                throw mkl::exception("dft/backends/mklgpu"
                                     "commit",
                                     mkl_exception.what());
            }
        }

        // Generate backward DFT descriptor only if required.
        if (!one_descriptor) {
            handle.second = std::make_shared<mklgpu_descriptor_t>(config_values.dimensions);
            set_value(*handle.second, config_values, false, stride_choice);
            try {
                handle.second->commit(this->get_queue());
            }
            catch (const std::exception& mkl_exception) {
                // Catching the real Intel oneMKL exception causes headaches with naming.
                if (!forward_good) {
                    throw mkl::exception("dft/backends/mklgpu"
                                         "commit",
                                         mkl_exception.what());
                }
            }
        }
    }

    void* get_handle() noexcept override {
        return &handle;
    }

    ~mklgpu_commit() override = default;

    virtual void set_workspace(scalar_type* usm_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, usm_workspace);
        handle.first->set_workspace(usm_workspace);
        if (handle.first != handle.second) {
            handle.second->set_workspace(usm_workspace);
        }
    }

    virtual void set_workspace(sycl::buffer<scalar_type>& buffer_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, buffer_workspace);
        handle.first->set_workspace(buffer_workspace);
        if (handle.first != handle.second) {
            handle.second->set_workspace(buffer_workspace);
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
        using onemkl_param = dft::detail::config_param;
        using backend_param = dft::config_param;

        // The following are read-only:
        // Dimension, forward domain, precision, commit status.
        // Lengths are supplied at descriptor construction time.
        desc.set_value(backend_param::FORWARD_SCALE, config.fwd_scale);
        desc.set_value(backend_param::BACKWARD_SCALE, config.bwd_scale);
        desc.set_value(backend_param::NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        desc.set_value(backend_param::COMPLEX_STORAGE,
                       to_mklgpu<onemkl_param::COMPLEX_STORAGE>(config.complex_storage));
        if (config.real_storage != dft::detail::config_value::REAL_REAL) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports real-real real storage.");
        }
        desc.set_value(backend_param::CONJUGATE_EVEN_STORAGE,
                       to_mklgpu<onemkl_param::CONJUGATE_EVEN_STORAGE>(config.conj_even_storage));
        desc.set_value(backend_param::PLACEMENT,
                       to_mklgpu<onemkl_param::PLACEMENT>(config.placement));

        if (stride_choice == dft::detail::stride_api::FB_STRIDES) {
            if (config.fwd_strides[0] != 0 || config.fwd_strides[0] != 0) {
                throw mkl::unimplemented("dft/backends/mklgpu", "commit",
                                         "MKLGPU does not support nonzero offsets.");
            }
            desc.set_value(backend_param::FWD_STRIDES, config.fwd_strides.data());
            desc.set_value(backend_param::BWD_STRIDES, config.bwd_strides.data());
        }
        else {
            if (config.input_strides[0] != 0 || config.output_strides[0] != 0) {
                throw mkl::unimplemented("dft/backends/mklgpu", "commit",
                                         "MKLGPU does not support nonzero offsets.");
            }
            if (assume_fwd_dft) {
                desc.set_value(backend_param::FWD_STRIDES, config.input_strides.data());
                desc.set_value(backend_param::BWD_STRIDES, config.output_strides.data());
            }
            else {
                desc.set_value(backend_param::FWD_STRIDES, config.output_strides.data());
                desc.set_value(backend_param::BWD_STRIDES, config.input_strides.data());
            }
        }
        desc.set_value(backend_param::FWD_DISTANCE, config.fwd_dist);
        desc.set_value(backend_param::BWD_DISTANCE, config.bwd_dist);
        if (config.workspace_placement == dft::detail::config_value::WORKSPACE_EXTERNAL) {
            // Setting WORKSPACE_INTERNAL (default) causes FFT_INVALID_DESCRIPTOR.
            desc.set_value(backend_param::WORKSPACE,
                           to_mklgpu_config_value<onemkl_param::WORKSPACE_PLACEMENT>(
                               config.workspace_placement));
        }
        // Setting the ordering causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.ordering != dft::detail::config_value::ORDERED) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports ordered ordering.");
        }
        // Setting the transpose causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.transpose != false) {
            throw mkl::invalid_argument("dft/backends/mklgpu", "commit",
                                        "MKLGPU only supports non-transposed.");
        }
    }

    // This is called by the workspace_helper, and is not part of the user API.
    virtual std::int64_t get_workspace_external_bytes_impl() override {
        std::size_t workspaceSizeFwd = 0, workspaceSizeBwd = 0;
        handle.first->get_value(dft::config_param::WORKSPACE_BYTES, &workspaceSizeFwd);
        handle.second->get_value(dft::config_param::WORKSPACE_BYTES, &workspaceSizeBwd);
        return static_cast<std::int64_t>(std::max(workspaceSizeFwd, workspaceSizeFwd));
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

} // namespace oneapi::mkl::dft::mklgpu
