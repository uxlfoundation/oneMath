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

#ifndef _ONEMATH_DFT_DETAIL_DESCRIPTOR_IMPL_HPP_
#define _ONEMATH_DFT_DETAIL_DESCRIPTOR_IMPL_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/detail/backend_selector.hpp"
#include "oneapi/math/detail/export.hpp"

#include "oneapi/math/dft/detail/types_impl.hpp"
#include "oneapi/math/dft/detail/commit_impl.hpp"

namespace oneapi {
namespace math {
namespace dft {
namespace detail {
// Forward declaration:
template <precision prec, domain dom>
class descriptor;

template <precision prec, domain dom>
inline commit_impl<prec, dom>* get_commit(descriptor<prec, dom>& desc);

template <precision prec, domain dom>
class descriptor {
private:
    using scalar_type = typename descriptor_info<descriptor>::scalar_type;

public:
    // Syntax for 1-dimensional DFT
    descriptor(std::int64_t length);

    // Syntax for d-dimensional DFT
    descriptor(std::vector<std::int64_t> dimensions);

    // Copy operations are included in the oneAPI oneMath specification, but not yet
    // implemented here. If you need copies, please open an issue at
    // https://github.com/uxlfoundation/oneMath/issues

    descriptor(descriptor&&);

    descriptor& operator=(descriptor&&);

    ~descriptor();

    void set_value(config_param param, ...);

    void get_value(config_param param, ...) const;

    void commit(sycl::queue& queue);

#ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
    void commit(backend_selector<backend::mklcpu> selector);
#endif

#ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
    void commit(backend_selector<backend::mklgpu> selector);
#endif

#ifdef ONEMATH_ENABLE_CUFFT_BACKEND
    void commit(backend_selector<backend::cufft> selector);
#endif

#ifdef ONEMATH_ENABLE_ROCFFT_BACKEND
    void commit(backend_selector<backend::rocfft> selector);
#endif

#ifdef ONEMATH_ENABLE_PORTFFT_BACKEND
    void commit(backend_selector<backend::portfft> selector);
#endif

    const dft_values<prec, dom>& get_values() const noexcept {
        return values_;
    }

    void set_workspace(scalar_type* usm_workspace);

    void set_workspace(sycl::buffer<scalar_type>& buffer_workspace);

private:
    // Has a value when the descriptor is committed.
    std::unique_ptr<commit_impl<prec, dom>> pimpl_;

    // descriptor configuration values_ and structs
    dft_values<prec, dom> values_;

    friend commit_impl<prec, dom>* get_commit<prec, dom>(descriptor<prec, dom>&);

    using real_t = typename precision_t<prec>::real_t;
};

template <precision prec, domain dom>
inline commit_impl<prec, dom>* get_commit(descriptor<prec, dom>& desc) {
    return desc.pimpl_.get();
}

} // namespace detail
} // namespace dft
} // namespace math
} // namespace oneapi

#endif // _ONEMATH_DFT_DETAIL_DESCRIPTOR_IMPL_HPP_
