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
#include <cstdarg>

#include "oneapi/math/detail/exceptions.hpp"
#include "oneapi/math/dft/descriptor.hpp"

#include "dft/descriptor_config_helper.hpp"

namespace oneapi {
namespace math {
namespace dft {
namespace detail {

// Compute the default strides. Modifies real_strides and complex_strides arguments
template <domain dom>
inline void compute_default_strides(const std::vector<std::int64_t>& dimensions,
                                    std::vector<std::int64_t>& fwd_strides,
                                    std::vector<std::int64_t>& bwd_strides) {
    const auto rank = dimensions.size();
    fwd_strides = std::vector<std::int64_t>(rank + 1, 1);
    fwd_strides[0] = 0;
    bwd_strides = fwd_strides;
    if (rank == 1) {
        return;
    }

    bwd_strides[rank - 1] =
        dom == domain::COMPLEX ? dimensions[rank - 1] : (dimensions[rank - 1] / 2) + 1;
    fwd_strides[rank - 1] =
        dom == domain::COMPLEX ? dimensions[rank - 1] : 2 * bwd_strides[rank - 1];
    for (auto i = rank - 1; i > 1; --i) {
        // Can't start at rank - 2 with unsigned type and minimum value of rank being 1.
        bwd_strides[i - 1] = bwd_strides[i] * dimensions[i - 1];
        fwd_strides[i - 1] = fwd_strides[i] * dimensions[i - 1];
    }
}

template <precision prec, domain dom>
void descriptor<prec, dom>::set_value(config_param param, ...) {
    va_list vl;
    va_start(vl, param);
    switch (param) {
        case config_param::FORWARD_DOMAIN:
            throw math::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::DIMENSION:
            throw math::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::LENGTHS: {
            if (values_.dimensions.size() == 1) {
                std::int64_t length = va_arg(vl, std::int64_t);
                detail::set_value<config_param::LENGTHS>(values_, &length);
            }
            else {
                detail::set_value<config_param::LENGTHS>(values_, va_arg(vl, std::int64_t*));
            }
            break;
        }
        case config_param::PRECISION:
            throw math::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        case config_param::INPUT_STRIDES:
            detail::set_value<config_param::INPUT_STRIDES>(values_, va_arg(vl, std::int64_t*));
            break;
        case config_param::OUTPUT_STRIDES:
            detail::set_value<config_param::OUTPUT_STRIDES>(values_, va_arg(vl, std::int64_t*));
            break;
#pragma clang diagnostic pop
        case config_param::FWD_STRIDES:
            detail::set_value<config_param::FWD_STRIDES>(values_, va_arg(vl, std::int64_t*));
            break;
        case config_param::BWD_STRIDES:
            detail::set_value<config_param::BWD_STRIDES>(values_, va_arg(vl, std::int64_t*));
            break;
        // VA arg promotes float args to double, so the following is always double:
        case config_param::FORWARD_SCALE:
            detail::set_value<config_param::FORWARD_SCALE>(values_,
                                                           static_cast<real_t>(va_arg(vl, double)));
            break;
        case config_param::BACKWARD_SCALE:
            detail::set_value<config_param::BACKWARD_SCALE>(
                values_, static_cast<real_t>(va_arg(vl, double)));
            break;
        case config_param::NUMBER_OF_TRANSFORMS:
            detail::set_value<config_param::NUMBER_OF_TRANSFORMS>(values_,
                                                                  va_arg(vl, std::int64_t));
            break;
        case config_param::FWD_DISTANCE:
            detail::set_value<config_param::FWD_DISTANCE>(values_, va_arg(vl, std::int64_t));
            break;
        case config_param::BWD_DISTANCE:
            detail::set_value<config_param::BWD_DISTANCE>(values_, va_arg(vl, std::int64_t));
            break;
        case config_param::PLACEMENT:
            detail::set_value<config_param::PLACEMENT>(values_, va_arg(vl, config_value));
            break;
        case config_param::COMPLEX_STORAGE:
            detail::set_value<config_param::COMPLEX_STORAGE>(values_, va_arg(vl, config_value));
            break;
        case config_param::REAL_STORAGE:
            detail::set_value<config_param::REAL_STORAGE>(values_, va_arg(vl, config_value));
            break;
        case config_param::CONJUGATE_EVEN_STORAGE:
            detail::set_value<config_param::CONJUGATE_EVEN_STORAGE>(values_,
                                                                    va_arg(vl, config_value));
            break;
        case config_param::ORDERING:
            detail::set_value<config_param::ORDERING>(values_, va_arg(vl, config_value));
            break;
        case config_param::TRANSPOSE:
            detail::set_value<config_param::TRANSPOSE>(values_, va_arg(vl, int));
            break;
        case config_param::WORKSPACE:
            detail::set_value<config_param::WORKSPACE>(values_, va_arg(vl, config_value));
            break;
        case config_param::WORKSPACE_PLACEMENT:
            detail::set_value<config_param::WORKSPACE_PLACEMENT>(values_, va_arg(vl, config_value));
            break;
        case config_param::WORKSPACE_EXTERNAL_BYTES:
            throw math::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        case config_param::PACKED_FORMAT:
            detail::set_value<config_param::PACKED_FORMAT>(values_, va_arg(vl, config_value));
            break;
        case config_param::COMMIT_STATUS:
            throw math::invalid_argument("DFT", "set_value", "Read-only parameter.");
            break;
        default: throw math::invalid_argument("DFT", "set_value", "Invalid config_param argument.");
    }
    va_end(vl);
}

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::vector<std::int64_t> dimensions) {
    if (dimensions.size() == 0) {
        throw math::invalid_argument("DFT", "descriptor", "Cannot have 0 dimensional DFT.");
    }
    for (const auto& dim : dimensions) {
        if (dim <= 0) {
            throw math::invalid_argument("DFT", "descriptor",
                                         "Invalid dimension value (negative or 0).");
        }
    }
    compute_default_strides<dom>(dimensions, values_.fwd_strides, values_.bwd_strides);
    values_.input_strides = std::vector<std::int64_t>(dimensions.size() + 1, 0);
    values_.output_strides = std::vector<std::int64_t>(dimensions.size() + 1, 0);
    values_.bwd_scale = real_t(1.0);
    values_.fwd_scale = real_t(1.0);
    values_.number_of_transforms = 1;
    values_.fwd_dist = 1;
    values_.bwd_dist = 1;
    values_.placement = config_value::INPLACE;
    values_.complex_storage = config_value::COMPLEX_COMPLEX;
    values_.real_storage = config_value::REAL_REAL;
    values_.conj_even_storage = config_value::COMPLEX_COMPLEX;
    values_.workspace = config_value::ALLOW;
    values_.workspace_placement = config_value::WORKSPACE_AUTOMATIC;
    values_.ordering = config_value::ORDERED;
    values_.transpose = false;
    values_.packed_format = config_value::CCE_FORMAT;
    values_.dimensions = std::move(dimensions);
}

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(std::int64_t length)
        : descriptor<prec, dom>(std::vector<std::int64_t>{ length }) {}

template <precision prec, domain dom>
descriptor<prec, dom>::descriptor(descriptor<prec, dom>&& other) = default;

template <precision prec, domain dom>
descriptor<prec, dom>& descriptor<prec, dom>::operator=(descriptor<prec, dom>&&) = default;

template <precision prec, domain dom>
descriptor<prec, dom>::~descriptor() = default;

template <precision prec, domain dom>
void descriptor<prec, dom>::get_value(config_param param, ...) const {
    va_list vl;
    va_start(vl, param);
    if (va_arg(vl, void*) == nullptr) {
        throw math::invalid_argument("DFT", "get_value", "config_param is nullptr.");
    }
    va_end(vl);
    va_start(vl, param);
    switch (param) {
        case config_param::FORWARD_DOMAIN: *va_arg(vl, dft::domain*) = dom; break;
        case config_param::DIMENSION:
            *va_arg(vl, std::int64_t*) = static_cast<std::int64_t>(values_.dimensions.size());
            break;
        case config_param::LENGTHS:
            std::copy(values_.dimensions.begin(), values_.dimensions.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::PRECISION: *va_arg(vl, dft::precision*) = prec; break;
        case config_param::FORWARD_SCALE:
            *va_arg(vl, real_t*) = static_cast<real_t>(values_.fwd_scale);
            break;
        case config_param::BACKWARD_SCALE:
            *va_arg(vl, real_t*) = static_cast<real_t>(values_.bwd_scale);
            break;
        case config_param::NUMBER_OF_TRANSFORMS:
            *va_arg(vl, std::int64_t*) = values_.number_of_transforms;
            break;
        case config_param::COMPLEX_STORAGE:
            *va_arg(vl, config_value*) = values_.complex_storage;
            break;
        case config_param::REAL_STORAGE: *va_arg(vl, config_value*) = values_.real_storage; break;
        case config_param::CONJUGATE_EVEN_STORAGE:
            *va_arg(vl, config_value*) = values_.conj_even_storage;
            break;
        case config_param::PLACEMENT: *va_arg(vl, config_value*) = values_.placement; break;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        case config_param::INPUT_STRIDES:
            std::copy(values_.input_strides.begin(), values_.input_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::OUTPUT_STRIDES:
            std::copy(values_.output_strides.begin(), values_.output_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
#pragma clang diagnostic pop
        case config_param::FWD_STRIDES:
            std::copy(values_.fwd_strides.begin(), values_.fwd_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::BWD_STRIDES:
            std::copy(values_.bwd_strides.begin(), values_.bwd_strides.end(),
                      va_arg(vl, std::int64_t*));
            break;
        case config_param::FWD_DISTANCE: *va_arg(vl, std::int64_t*) = values_.fwd_dist; break;
        case config_param::BWD_DISTANCE: *va_arg(vl, std::int64_t*) = values_.bwd_dist; break;
        case config_param::WORKSPACE: *va_arg(vl, config_value*) = values_.workspace; break;
        case config_param::WORKSPACE_PLACEMENT:
            *va_arg(vl, config_value*) = values_.workspace_placement;
            break;
        case config_param::WORKSPACE_EXTERNAL_BYTES:
            if (!pimpl_) {
                throw math::invalid_argument(
                    "DFT", "get_value",
                    "Cannot query WORKSPACE_EXTERNAL_BYTES on uncommitted descriptor.");
            }
            else {
                *va_arg(vl, std::int64_t*) = pimpl_->get_workspace_external_bytes();
            }
            break;
        case config_param::ORDERING: *va_arg(vl, config_value*) = values_.ordering; break;
        case config_param::TRANSPOSE: *va_arg(vl, int*) = values_.transpose; break;
        case config_param::PACKED_FORMAT: *va_arg(vl, config_value*) = values_.packed_format; break;
        case config_param::COMMIT_STATUS:
            *va_arg(vl, config_value*) =
                pimpl_ ? config_value::COMMITTED : config_value::UNCOMMITTED;
            break;
        default: throw math::invalid_argument("DFT", "get_value", "Invalid config_param argument.");
    }
    va_end(vl);
}

template <precision prec, domain dom>
void descriptor<prec, dom>::set_workspace(scalar_type* usm_workspace) {
    if (pimpl_) {
        return pimpl_->set_workspace(usm_workspace);
    }
    else {
        throw math::uninitialized("DFT", "set_workspace",
                                  "Can only set workspace on committed descriptor.");
    }
}

template <precision prec, domain dom>
void descriptor<prec, dom>::set_workspace(sycl::buffer<scalar_type>& buffer_workspace) {
    if (pimpl_) {
        return pimpl_->set_workspace(buffer_workspace);
    }
    else {
        throw math::uninitialized("DFT", "set_workspace",
                                  "Can only set workspace on committed descriptor.");
    }
}

template class descriptor<precision::SINGLE, domain::COMPLEX>;
template class descriptor<precision::SINGLE, domain::REAL>;
template class descriptor<precision::DOUBLE, domain::COMPLEX>;
template class descriptor<precision::DOUBLE, domain::REAL>;

} //namespace detail
} //namespace dft
} //namespace math
} //namespace oneapi
