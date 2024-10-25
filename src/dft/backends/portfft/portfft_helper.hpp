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

#ifndef _ONEMKL_DFT_SRC_PORTFFT_HELPERS_HPP_
#define _ONEMKL_DFT_SRC_PORTFFT_HELPERS_HPP_

#include <type_traits>

#include <portfft/portfft.hpp>

#include "oneapi/mkl/dft/detail/commit_impl.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"

namespace pfft = portfft;

namespace oneapi::mkl::dft::portfft::detail {
template <dft::precision prec, dft::domain dom>
inline dft::detail::commit_impl<prec, dom>* checked_get_commit(
    dft::detail::descriptor<prec, dom>& desc) {
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::portfft) {
        throw mkl::invalid_argument("dft/backends/portfft", "get_commit",
                                    "DFT descriptor has not been commited for portFFT");
    }
    return commit_handle;
}

template <typename descriptor_type>
using to_pfft_domain =
    std::conditional<std::is_floating_point_v<fwd<descriptor_type>>,
                     std::integral_constant<pfft::domain, pfft::domain::REAL>,
                     std::integral_constant<pfft::domain, pfft::domain::COMPLEX>>;

template <typename descriptor_type>
using storage_type =
    std::optional<pfft::committed_descriptor<scalar<descriptor_type>,
                                             detail::to_pfft_domain<descriptor_type>::type::value>>;

template <typename descriptor_type>
auto get_descriptors(descriptor_type& desc) {
    auto commit = detail::checked_get_commit(desc);
    return reinterpret_cast<storage_type<descriptor_type>*>(commit->get_handle());
}
} // namespace oneapi::mkl::dft::portfft::detail

#endif
