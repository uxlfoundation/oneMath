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
#include "oneapi/math/detail/backends.hpp"
#include "oneapi/math/dft/types.hpp"
#include "oneapi/math/dft/descriptor.hpp"

#include "oneapi/math/dft/detail/mklcpu/onemath_dft_mklcpu.hpp"

#include "oneapi/math/dft/detail/commit_impl.hpp"

#include "dft/backends/mklcpu/commit_derived_impl.hpp"
#include "../stride_helper.hpp"

// Intel(R) oneMKL header
#include <mkl_dfti.h>
#include <mkl_service.h>

namespace oneapi {
namespace math {
namespace dft {
namespace mklcpu {
namespace detail {

template <dft::detail::precision prec, dft::detail::domain dom>
commit_derived_impl<prec, dom>::commit_derived_impl(
    sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values)
        : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::mklcpu, config_values) {
    // create the descriptor once for the lifetime of the descriptor class
    DFT_ERROR status[2] = { DFTI_BAD_DESCRIPTOR, DFTI_BAD_DESCRIPTOR };

    for (auto dir : { DIR::fwd, DIR::bwd }) {
        const auto rank = static_cast<std::int64_t>(config_values.dimensions.size());
        if (config_values.dimensions.size() == 1) {
            status[dir] = DftiCreateDescriptor(&bidirection_handle[dir], mklcpu_prec, mklcpu_dom, 1,
                                               config_values.dimensions[0]);
        }
        else {
            status[dir] = DftiCreateDescriptor(&bidirection_handle[dir], mklcpu_prec, mklcpu_dom,
                                               rank, config_values.dimensions.data());
        }
    }

    if (status[0] != DFTI_NO_ERROR || status[1] != DFTI_NO_ERROR) {
        std::string err = std::string("DftiCreateDescriptor failed with status : ") +
                          DftiErrorMessage(status[0]) + std::string(", ") +
                          DftiErrorMessage(status[1]);
        throw oneapi::math::exception("dft/backends/mklcpu", "create_descriptor", err);
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
commit_derived_impl<prec, dom>::~commit_derived_impl() {
    for (auto dir : { DIR::fwd, DIR::bwd }) {
        DftiFreeDescriptor(&bidirection_handle[dir]);
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
void commit_derived_impl<prec, dom>::commit(
    const dft::detail::dft_values<prec, dom>& config_values) {
    this->external_workspace_helper_ =
        oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
            config_values.workspace_placement ==
            oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);
    set_value(bidirection_handle.data(), config_values);

    this->get_queue()
        .submit([&](sycl::handler& cgh) {
            auto bidir_handle_obj =
                bidirection_buffer.get_access<sycl::access::mode::read_write>(cgh);

            host_task<detail::kernel_name<mklcpu_desc_t>>(cgh, [=]() {
                DFT_ERROR status[2] = { DFTI_BAD_DESCRIPTOR, DFTI_BAD_DESCRIPTOR };

                for (auto dir : { DIR::fwd, DIR::bwd })
                    status[dir] = DftiCommitDescriptor(bidir_handle_obj[dir]);

                // this is important for real-batched transforms, as the backward transform would
                // be inconsistent based on the stride setup, but once recommited before backward
                // it should work just fine. so we error out only if there is a issue with both.
                if (status[0] != DFTI_NO_ERROR && status[1] != DFTI_NO_ERROR) {
                    std::string err = std::string("DftiCommitDescriptor failed with status : ") +
                                      DftiErrorMessage(status[0]) + std::string(", ") +
                                      DftiErrorMessage(status[1]);
                    throw oneapi::math::exception("dft/backends/mklcpu", "commit", err);
                }
            });
        })
        .wait();
}

template <dft::detail::precision prec, dft::detail::domain dom>
void* commit_derived_impl<prec, dom>::get_handle() noexcept {
    return reinterpret_cast<void*>(bidirection_handle.data());
}

template <dft::detail::precision prec, dft::detail::domain dom>
template <typename... Args>
void commit_derived_impl<prec, dom>::set_value_item(mklcpu_desc_t hand, enum DFTI_CONFIG_PARAM name,
                                                    Args... args) {
    DFT_ERROR value_err = DftiSetValue(hand, name, args...);
    if (value_err != DFTI_NO_ERROR) {
        throw oneapi::math::exception("dft/backends/mklcpu", "set_value_item",
                                      DftiErrorMessage(value_err));
    }
}

template <dft::detail::precision prec, dft::detail::domain dom>
void commit_derived_impl<prec, dom>::set_value(mklcpu_desc_t* descHandle,
                                               const dft::detail::dft_values<prec, dom>& config) {
    auto stride_choice = dft::detail::get_stride_api(config);
    dft::detail::throw_on_invalid_stride_api("MKLCPU commit", stride_choice);
    for (auto dir : { DIR::fwd, DIR::bwd }) {
        if (stride_choice == dft::detail::stride_api::IO_STRIDES) {
            set_value_item(descHandle[dir], DFTI_INPUT_STRIDES, config.input_strides.data());
            set_value_item(descHandle[dir], DFTI_OUTPUT_STRIDES, config.output_strides.data());
        }
        else { // Forward / backward strides
            if (dir == DIR::fwd) {
                set_value_item(descHandle[dir], DFTI_INPUT_STRIDES, config.fwd_strides.data());
                set_value_item(descHandle[dir], DFTI_OUTPUT_STRIDES, config.bwd_strides.data());
            }
            else {
                set_value_item(descHandle[dir], DFTI_INPUT_STRIDES, config.bwd_strides.data());
                set_value_item(descHandle[dir], DFTI_OUTPUT_STRIDES, config.fwd_strides.data());
            }
        }
        set_value_item(descHandle[dir], DFTI_BACKWARD_SCALE, config.bwd_scale);
        set_value_item(descHandle[dir], DFTI_FORWARD_SCALE, config.fwd_scale);
        set_value_item(descHandle[dir], DFTI_NUMBER_OF_TRANSFORMS, config.number_of_transforms);
        set_value_item(descHandle[dir], DFTI_INPUT_DISTANCE,
                       (dir == detail::DIR::fwd) ? config.fwd_dist : config.bwd_dist);
        set_value_item(descHandle[dir], DFTI_OUTPUT_DISTANCE,
                       (dir == detail::DIR::fwd) ? config.bwd_dist : config.fwd_dist);
        set_value_item(descHandle[dir], DFTI_COMPLEX_STORAGE,
                       to_mklcpu<config_param::COMPLEX_STORAGE>(config.complex_storage));
        set_value_item(descHandle[dir], DFTI_REAL_STORAGE,
                       to_mklcpu<config_param::REAL_STORAGE>(config.real_storage));
        set_value_item(descHandle[dir], DFTI_CONJUGATE_EVEN_STORAGE,
                       to_mklcpu<config_param::CONJUGATE_EVEN_STORAGE>(config.conj_even_storage));
        set_value_item(descHandle[dir], DFTI_PLACEMENT,
                       to_mklcpu<config_param::PLACEMENT>(config.placement));
        set_value_item(descHandle[dir], DFTI_PACKED_FORMAT,
                       to_mklcpu<config_param::PACKED_FORMAT>(config.packed_format));
        // Setting the workspace causes an FFT_INVALID_DESCRIPTOR.
        if (config.workspace != config_value::ALLOW) {
            throw math::invalid_argument("dft/backends/mklcpu", "commit",
                                         "MKLCPU only supports workspace set to allow");
        }
        // Setting the ordering causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.ordering != dft::detail::config_value::ORDERED) {
            throw math::invalid_argument("dft/backends/mklcpu", "commit",
                                         "MKLCPU only supports ordered ordering.");
        }
        // Setting the transpose causes an FFT_INVALID_DESCRIPTOR. Check that default is used:
        if (config.transpose != false) {
            throw math::invalid_argument("dft/backends/mklcpu", "commit",
                                         "MKLCPU only supports non-transposed.");
        }
    }
}
} // namespace detail

template <dft::detail::precision prec, dft::detail::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::commit_derived_impl<prec, dom>(sycl_queue, desc.get_values());
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

} // namespace mklcpu
} // namespace dft
} // namespace math
} // namespace oneapi
