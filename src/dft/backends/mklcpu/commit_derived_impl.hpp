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

#ifndef _ONEMATH_DFT_COMMIT_DERIVED_IMPL_HPP_
#define _ONEMATH_DFT_COMMIT_DERIVED_IMPL_HPP_

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/dft/detail/types_impl.hpp"
#include "dft/backends/mklcpu/mklcpu_helpers.hpp"

// Intel(R) oneMKL header
#include <mkl_dfti.h>

namespace oneapi {
namespace math {
namespace dft {
namespace mklcpu {
namespace detail {

// this is used for indexing bidirectional_handle
enum DIR { fwd = 0, bwd = 1 };

template <dft::detail::precision prec, dft::detail::domain dom>
class commit_derived_impl final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;
    static constexpr DFTI_CONFIG_VALUE mklcpu_prec = to_mklcpu(prec);
    static constexpr DFTI_CONFIG_VALUE mklcpu_dom = to_mklcpu(dom);
    using mklcpu_desc_t = DFTI_DESCRIPTOR_HANDLE;

public:
    commit_derived_impl(sycl::queue queue, const dft::detail::dft_values<prec, dom>& config_values);

    virtual void commit(const dft::detail::dft_values<prec, dom>& config_values) override;

    virtual void* get_handle() noexcept override;

    virtual ~commit_derived_impl() override;

    sycl::buffer<mklcpu_desc_t, 1> get_handle_buffer() noexcept {
        return bidirection_buffer;
    };

#define BACKEND mklcpu
#include "../backend_compute_signature.cxx"
#undef BACKEND

private:
    // bidirectional_handle[0] is the forward handle, bidirectional_handle[1] is the backward handle
    std::array<mklcpu_desc_t, 2> bidirection_handle{ nullptr, nullptr };
    sycl::buffer<mklcpu_desc_t, 1> bidirection_buffer{ bidirection_handle.data(),
                                                       sycl::range<1>{ 2 } };

    template <typename... Args>
    void set_value_item(mklcpu_desc_t hand, enum DFTI_CONFIG_PARAM name, Args... args);

    void set_value(mklcpu_desc_t* descHandle, const dft::detail::dft_values<prec, dom>& config);
};

template <dft::detail::precision prec, dft::detail::domain dom>
using commit_t = dft::detail::commit_impl<prec, dom>;

template <dft::detail::precision prec, dft::detail::domain dom>
using commit_derived_t = detail::commit_derived_impl<prec, dom>;

} // namespace detail
} // namespace mklcpu
} // namespace dft
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_DFT_COMMIT_DERIVED_IMPL_HPP_
