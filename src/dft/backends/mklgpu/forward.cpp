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

#include <type_traits>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/exceptions.hpp"

#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"
#include "oneapi/mkl/dft/detail/descriptor_impl.hpp"

#include "mklgpu_helpers.hpp"

#include "mkl_version.h"
// MKLGPU header
#if INTEL_MKL_VERSION < 20250000
#include <oneapi/mkl/dfti.hpp>
#else
#include <oneapi/mkl/dft.hpp>
#endif

/**
Note that in this file, the Intel oneMKL-GPU library's interface mirrors the
interface of this OneMKL library. Consequently, the types under dft::TYPE are
closed-source Intel oneMKL types, and types under dft::detail::TYPE are from
this library.
**/

namespace oneapi::mkl::dft::mklgpu {
namespace detail {
/// Forward a MKLGPU DFT call to the backend, checking that the commit impl is valid.
/// Assumes backend descriptor values match those of the frontend.
template <dft::detail::precision prec, dft::detail::domain dom, typename... ArgTs>
inline auto compute_forward(dft::detail::descriptor<prec, dom>& desc, ArgTs&&... args) {
    using mklgpu_desc_t = dft::descriptor<to_mklgpu(prec), to_mklgpu(dom)>;
    using desc_shptr_t = std::shared_ptr<mklgpu_desc_t>;
    using handle_t = std::pair<desc_shptr_t, desc_shptr_t>;
    auto commit_handle = dft::detail::get_commit(desc);
    if (commit_handle == nullptr || commit_handle->get_backend() != backend::mklgpu) {
        throw mkl::invalid_argument("DFT", "compute_forward",
                                    "DFT descriptor has not been commited for MKLGPU");
    }
    auto handle = reinterpret_cast<handle_t*>(commit_handle->get_handle());
    auto mklgpu_desc = handle->first; // First because forward DFT.
    int commit_status{ DFTI_UNCOMMITTED };
    mklgpu_desc->get_value(dft::config_param::COMMIT_STATUS, &commit_status);
    if (commit_status != DFTI_COMMITTED) {
        throw mkl::invalid_argument("DFT", "compute_forward",
                                    "MKLGPU DFT descriptor was not successfully committed.");
    }
    // The MKLGPU backend's iterface contains fewer function signatures than in this
    // open-source library. Consequently, it is not required to forward template arguments
    // to resolve to the correct function.
    return dft::compute_forward(*mklgpu_desc, std::forward<ArgTs>(args)...);
}

/// Throw an mkl::invalid_argument if the runtime param in the descriptor does not match
/// the expected value.
template <dft::detail::config_param Param, dft::detail::config_value Expected, typename DescT>
inline auto expect_config(DescT& desc, const char* message) {
    dft::detail::config_value actual{ 0 };
    desc.get_value(Param, &actual);
    if (actual != Expected) {
        throw mkl::invalid_argument("DFT", "compute_forward", message);
    }
}
} // namespace detail

// BUFFER version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc,
                                   sycl::buffer<fwd<descriptor_type>, 1>& inout) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_forward(desc, inout);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& /*desc*/,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*inout_re*/,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*inout_im*/) {
    throw mkl::unimplemented("DFT", "compute_forward",
                             "MKLGPU does not support compute_forward(desc, inout_re, inout_im).");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc, sycl::buffer<fwd<descriptor_type>, 1>& in,
                                   sycl::buffer<bwd<descriptor_type>, 1>& out) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    return detail::compute_forward(desc, in, out);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT void compute_forward(descriptor_type& desc,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*in_re*/,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*in_im*/,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*out_re*/,
                                   sycl::buffer<scalar<descriptor_type>, 1>& /*out_im*/) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::mkl::unimplemented(
        "DFT", "compute_forward(desc, in_re, in_im, out_re, out_im)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

//USM version

//In-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, fwd<descriptor_type>* inout,
                                          const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT, dft::detail::config_value::INPLACE>(
        desc, "Unexpected value for placement");
    return detail::compute_forward(desc, inout, dependencies);
}

//In-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& /*desc*/,
                                          scalar<descriptor_type>* /*inout_re*/,
                                          scalar<descriptor_type>* /*inout_im*/,
                                          const std::vector<sycl::event>& /*dependencies*/) {
    throw mkl::unimplemented(
        "DFT", "compute_forward",
        "MKLGPU does not support compute_forward(desc, inout_re, inout_im, dependencies).");
}

//Out-of-place transform
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, fwd<descriptor_type>* in,
                                          bwd<descriptor_type>* out,
                                          const std::vector<sycl::event>& dependencies) {
    detail::expect_config<dft::detail::config_param::PLACEMENT,
                          dft::detail::config_value::NOT_INPLACE>(desc,
                                                                  "Unexpected value for placement");
    return detail::compute_forward(desc, in, out, dependencies);
}

//Out-of-place transform, using config_param::COMPLEX_STORAGE=config_value::REAL_REAL data format
template <typename descriptor_type>
ONEMKL_EXPORT sycl::event compute_forward(descriptor_type& desc, scalar<descriptor_type>* /*in_re*/,
                                          scalar<descriptor_type>* /*in_im*/,
                                          scalar<descriptor_type>* /*out_re*/,
                                          scalar<descriptor_type>* /*out_im*/,
                                          const std::vector<sycl::event>& /*dependencies*/) {
    detail::expect_config<dft::detail::config_param::COMPLEX_STORAGE,
                          dft::detail::config_value::REAL_REAL>(
        desc, "Unexpected value for complex storage");
    throw oneapi::mkl::unimplemented(
        "DFT", "compute_forward(desc, in_re, in_im, out_re, out_im, dependencies)",
        "MKLGPU does not support out-of-place FFT with real-real complex storage.");
}

// Template function instantiations
#include "dft/backends/backend_forward_instantiations.cxx"

} // namespace oneapi::mkl::dft::mklgpu
