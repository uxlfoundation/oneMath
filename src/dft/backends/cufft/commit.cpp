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
#include <algorithm>
#include <optional>

#include "oneapi/math/exceptions.hpp"

#include "oneapi/math/dft/detail/commit_impl.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"
#include "oneapi/math/dft/detail/cufft/onemath_dft_cufft.hpp"
#include "oneapi/math/dft/types.hpp"

#include "../stride_helper.hpp"

#include <cufft.h>
#include <cuda.h>

namespace oneapi::math::dft::cufft {
namespace detail {

/// Commit impl class specialization for cuFFT.
template <dft::precision prec, dft::domain dom>
class cufft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;

    // For real to complex transforms, the "type" arg also encodes the direction (e.g. CUFFT_R2C vs CUFFT_C2R) in the plan so we must have one for each direction.
    // We also need this because oneMath uses a directionless "FWD_DISTANCE" and "BWD_DISTANCE" while cuFFT uses a directional "idist" and "odist".
    // plans[0] is forward, plans[1] is backward
    std::array<std::optional<cufftHandle>, 2> plans = { std::nullopt, std::nullopt };
    std::int64_t offset_fwd_in, offset_fwd_out, offset_bwd_in, offset_bwd_out;

public:
    cufft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::cufft,
                                                                config_values) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw math::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
    }

    void clean_plans() {
        auto fix_context = plans[0].has_value() || plans[1].has_value();
        if (plans[0]) {
            if (cufftDestroy(plans[0].value()) != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__,
                                      "Failed to destroy forward cuFFT plan.");
            }
            plans[0] = std::nullopt;
        }
        if (plans[1]) {
            if (cufftDestroy(plans[1].value()) != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__,
                                      "Failed to destroy backward cuFFT plan.");
            }
            plans[1] = std::nullopt;
        }
        if (fix_context) {
            // cufftDestroy changes the context so change it back.
            CUdevice interopDevice =
                sycl::get_native<sycl::backend::ext_oneapi_cuda>(this->get_queue().get_device());
            CUcontext interopContext;
            if (cuDevicePrimaryCtxRetain(&interopContext, interopDevice) != CUDA_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__,
                                      "Failed to change cuda context.");
            }
        }
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        this->external_workspace_helper_ =
            oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);
        clean_plans();

        if (config_values.fwd_scale != 1.0 || config_values.bwd_scale != 1.0) {
            throw math::unimplemented(
                "dft/backends/cufft", __FUNCTION__,
                "cuFFT does not support values other than 1 for FORWARD/BACKWARD_SCALE");
        }

        // The cudaStream for the plan is set at execution time so the interop handler can pick the stream.
        constexpr cufftType fwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2C;
                }
                else {
                    return CUFFT_Z2Z;
                }
            }
            else {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_R2C;
                }
                else {
                    return CUFFT_D2Z;
                }
            }
        }();
        constexpr cufftType bwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2C;
                }
                else {
                    return CUFFT_Z2Z;
                }
            }
            else {
                if constexpr (prec == dft::precision::SINGLE) {
                    return CUFFT_C2R;
                }
                else {
                    return CUFFT_Z2D;
                }
            }
        }();

        constexpr std::size_t max_supported_dims = 3;
        std::array<int, max_supported_dims> n_copy;
        std::copy(config_values.dimensions.begin(), config_values.dimensions.end(), n_copy.data());
        const int rank = static_cast<int>(config_values.dimensions.size());

        auto stride_api_choice = dft::detail::get_stride_api(config_values);
        dft::detail::throw_on_invalid_stride_api("CUFFT commit", stride_api_choice);
        dft::detail::stride_vectors<int> stride_vecs(config_values, stride_api_choice);
        offset_fwd_in = stride_vecs.offset_fwd_in;
        offset_fwd_out = stride_vecs.offset_fwd_out;
        offset_bwd_in = stride_vecs.offset_bwd_in;
        offset_bwd_out = stride_vecs.offset_bwd_out;

        // cufft ignores the first value in inembed and onembed, so there is no harm in putting offset there
        auto a_min = std::min_element(stride_vecs.vec_a.begin() + 1, stride_vecs.vec_a.end());
        auto b_min = std::min_element(stride_vecs.vec_b.begin() + 1, stride_vecs.vec_b.end());
        if constexpr (dom == dft::domain::REAL) {
            if ((a_min != stride_vecs.vec_a.begin() + rank) ||
                (b_min != stride_vecs.vec_b.begin() + rank)) {
                throw math::unimplemented(
                    "dft/backends/cufft", __FUNCTION__,
                    "cufft requires the last stride to be the the smallest one for real transforms!");
            }
        }
        else {
            if (a_min - stride_vecs.vec_a.begin() != b_min - stride_vecs.vec_b.begin()) {
                throw math::unimplemented(
                    "dft/backends/cufft", __FUNCTION__,
                    "cufft requires that if ordered by stride length, the order of strides is the same for input/output or fwd/bwd strides!");
            }
        }
        const int a_stride = static_cast<int>(*a_min);
        const int b_stride = static_cast<int>(*b_min);
        stride_vecs.vec_a.erase(a_min);
        stride_vecs.vec_b.erase(b_min);
        int fwd_istride = a_stride;
        int fwd_ostride = b_stride;
        int bwd_istride =
            stride_api_choice == dft::detail::stride_api::FB_STRIDES ? b_stride : a_stride;
        int bwd_ostride =
            stride_api_choice == dft::detail::stride_api::FB_STRIDES ? a_stride : b_stride;
        if (a_min - stride_vecs.vec_a.begin() != rank) {
            // swap dimensions to have the last one have the smallest stride
            std::swap(n_copy[a_min - stride_vecs.vec_a.begin() - 1], n_copy[rank - 1]);
        }
        for (int i = 1; i < rank; i++) {
            if ((stride_vecs.vec_a[i] % a_stride != 0) || (stride_vecs.vec_b[i] % b_stride != 0)) {
                throw math::unimplemented(
                    "dft/backends/cufft", __FUNCTION__,
                    "cufft requires a stride to be divisible by all smaller strides!");
            }
            stride_vecs.vec_a[i] /= a_stride;
            stride_vecs.vec_b[i] /= b_stride;
        }
        if (rank > 2) {
            if (stride_vecs.vec_a[1] > stride_vecs.vec_a[2] &&
                stride_vecs.vec_b[1] < stride_vecs.vec_b[2]) {
                throw math::unimplemented(
                    "dft/backends/cufft", __FUNCTION__,
                    "cufft requires that if ordered by stride length, the order of strides is the same for input and output strides!");
            }
            else if (stride_vecs.vec_a[1] < stride_vecs.vec_a[2] &&
                     stride_vecs.vec_b[1] < stride_vecs.vec_b[2]) {
                // swap dimensions to have the first one have the biggest stride
                std::swap(stride_vecs.vec_a[1], stride_vecs.vec_a[2]);
                std::swap(stride_vecs.vec_b[1], stride_vecs.vec_b[2]);
                std::swap(n_copy[0], n_copy[1]);
            }
            if ((stride_vecs.vec_a[1] % stride_vecs.vec_a[2] != 0) ||
                (stride_vecs.vec_b[1] % stride_vecs.vec_b[2] != 0)) {
                throw math::unimplemented(
                    "dft/backends/cufft", __FUNCTION__,
                    "cufft requires a stride to be divisible by all smaller strides!");
            }
            stride_vecs.vec_a[1] /= stride_vecs.vec_a[2];
            stride_vecs.vec_b[1] /= stride_vecs.vec_b[2];
        }
        const int batch = static_cast<int>(config_values.number_of_transforms);
        const int fwd_dist = static_cast<int>(config_values.fwd_dist);
        const int bwd_dist = static_cast<int>(config_values.bwd_dist);

        // When creating real-complex descriptions, the strides will always be wrong for one of the directions.
        // This is because the least significant dimension is symmetric.
        // If the strides are invalid (too small to fit) then just don't bother creating the plan
        auto check_stride_validity = [&](auto strides_fwd, auto strides_bwd) {
            int inner_nfwd = n_copy[rank - 1]; // inner dimensions of DFT
            // Complex data is stored conjugate even for real domains
            int inner_nbwd = dom == dft::domain::REAL ? inner_nfwd / 2 + 1 : inner_nfwd;
            int inner_sfwd = strides_fwd.back(); // inner strides of DFT
            int inner_sbwd = strides_bwd.back();
            bool valid = true;
            for (int r = 1; r < rank; ++r) {
                valid = valid && (inner_nfwd <= inner_sfwd) && (inner_nbwd <= inner_sbwd);
                inner_nfwd *= n_copy[rank - r - 1];
                inner_nbwd *= n_copy[rank - r - 1];
                inner_sfwd *= strides_fwd[rank - r - 1];
                inner_sbwd *= strides_bwd[rank - r - 1];
            }
            return valid;
        };

        bool valid_forward = check_stride_validity(stride_vecs.fwd_in, stride_vecs.fwd_out);
        bool valid_backward = stride_api_choice == dft::detail::stride_api::FB_STRIDES
                                  ? valid_forward
                                  : check_stride_validity(stride_vecs.bwd_out, stride_vecs.bwd_in);

        if (!valid_forward && !valid_backward) {
            throw math::exception("dft/backends/cufft", __FUNCTION__, "Invalid strides.");
        }

        if (valid_forward) {
            cufftHandle fwd_plan;
            auto res = cufftCreate(&fwd_plan);
            if (res != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__, "cufftCreate failed.");
            }
            apply_external_workspace_setting(fwd_plan, config_values.workspace_placement);
            res = cufftPlanMany(&fwd_plan, // plan
                                rank, // rank
                                n_copy.data(), // n
                                stride_vecs.fwd_in.data(), // inembed
                                fwd_istride, // istride
                                fwd_dist, // idist
                                stride_vecs.fwd_out.data(), // onembed
                                fwd_ostride, // ostride
                                bwd_dist, // odist
                                fwd_type, // type
                                batch // batch
            );

            if (res != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__,
                                      "Failed to create forward cuFFT plan.");
            }

            plans[0] = fwd_plan;
        }

        if (valid_backward) {
            cufftHandle bwd_plan;
            auto res = cufftCreate(&bwd_plan);
            if (res != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__, "cufftCreate failed.");
            }
            apply_external_workspace_setting(bwd_plan, config_values.workspace_placement);
            // flip fwd_distance and bwd_distance because cuFFt uses input distance and output distance.
            res = cufftPlanMany(&bwd_plan, // plan
                                rank, // rank
                                n_copy.data(), // n
                                stride_vecs.bwd_in.data(), // inembed
                                bwd_istride, // istride
                                bwd_dist, // idist
                                stride_vecs.bwd_out.data(), // onembed
                                bwd_ostride, // ostride
                                fwd_dist, // odist
                                bwd_type, // type
                                batch // batch
            );
            if (res != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", __FUNCTION__,
                                      "Failed to create backward cuFFT plan.");
            }
            plans[1] = bwd_plan;
        }
    }

    ~cufft_commit() override {
        clean_plans();
    }

    static void apply_external_workspace_setting(cufftHandle handle,
                                                 config_value workspace_setting) {
        if (workspace_setting == config_value::WORKSPACE_EXTERNAL) {
            auto res = cufftSetAutoAllocation(handle, 0);
            if (res != CUFFT_SUCCESS) {
                throw math::exception("dft/backends/cufft", "commit",
                                      "cufftSetAutoAllocation(plan, 0) failed.");
            }
        }
    }

    void* get_handle() noexcept override {
        return plans.data();
    }

    std::array<std::int64_t, 2> get_offsets_fwd() noexcept {
        return { offset_fwd_in, offset_fwd_out };
    }

    std::array<std::int64_t, 2> get_offsets_bwd() noexcept {
        return { offset_bwd_in, offset_bwd_out };
    }

    virtual void set_workspace(scalar_type* usm_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, usm_workspace);
        if (plans[0]) {
            cufftSetWorkArea(*plans[0], usm_workspace);
        }
        if (plans[1]) {
            cufftSetWorkArea(*plans[1], usm_workspace);
        }
    }

    void set_buffer_workspace(cufftHandle plan, sycl::buffer<scalar_type>& buffer_workspace) {
        this->get_queue()
            .submit([&](sycl::handler& cgh) {
                auto workspace_acc =
                    buffer_workspace.template get_access<sycl::access::mode::read_write>(cgh);
                cgh.host_task([=](sycl::interop_handle ih) {
                    auto stream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
                    auto result = cufftSetStream(plan, stream);
                    if (result != CUFFT_SUCCESS) {
                        throw oneapi::math::exception(
                            "dft/backends/cufft", "set_workspace",
                            "cufftSetStream returned " + std::to_string(result));
                    }
                    auto workspace_native = reinterpret_cast<scalar_type*>(
                        ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(workspace_acc));
                    cufftSetWorkArea(plan, workspace_native);
                });
            })
            .wait_and_throw();
    }

    virtual void set_workspace(sycl::buffer<scalar_type>& buffer_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, buffer_workspace);
        if (plans[0]) {
            set_buffer_workspace(*plans[0], buffer_workspace);
        }
        if (plans[1]) {
            set_buffer_workspace(*plans[1], buffer_workspace);
        }
    }

    std::int64_t get_plan_workspace_size_bytes(cufftHandle handle) {
        std::size_t size = 0;
        cufftGetSize(handle, &size);
        std::int64_t padded_size = static_cast<int64_t>(size);
        return padded_size;
    }

    virtual std::int64_t get_workspace_external_bytes_impl() override {
        std::int64_t size0 = plans[0] ? get_plan_workspace_size_bytes(*plans[0]) : 0;
        std::int64_t size1 = plans[1] ? get_plan_workspace_size_bytes(*plans[1]) : 0;
        return std::max(size0, size1);
    };

#define BACKEND cufft
#include "../backend_compute_signature.cxx"
#undef BACKEND
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::cufft_commit<prec, dom>(sycl_queue, desc.get_values());
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

namespace detail {
template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_fwd(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<cufft_commit<prec, dom>*>(commit)->get_offsets_fwd();
}

template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_bwd(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<cufft_commit<prec, dom>*>(commit)->get_offsets_bwd();
}

template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::SINGLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*);

template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::SINGLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*);
} //namespace detail

} // namespace oneapi::math::dft::cufft
