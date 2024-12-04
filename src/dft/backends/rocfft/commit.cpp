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
#include "oneapi/math/dft/detail/rocfft/onemath_dft_rocfft.hpp"
#include "oneapi/math/dft/types.hpp"

#include "../stride_helper.hpp"

#include "rocfft_handle.hpp"

#include <rocfft.h>
#include <rocfft-version.h>
#include <hip/hip_runtime_api.h>

namespace oneapi::math::dft::rocfft {
namespace detail {

// rocfft has global setup and cleanup functions which use some global state internally.
// Each can be called multiple times in an application, but due to the global nature, they always need to alternate.
// I don't believe its possible to avoid the user calling rocfft_cleanup in their own code,
// breaking our code, but we can try avoid it for them.
// rocfft_cleanup internally uses some singletons, so it is very difficult to decide if this is safe due to
// the static initialisation order fiasco.
class rocfft_singleton {
    rocfft_singleton() {
        const auto result = rocfft_setup();
        if (result != rocfft_status_success) {
            throw math::exception(
                "DFT", "rocfft",
                "Failed to setup rocfft. returned status " + std::to_string(result));
        }
    }

    ~rocfft_singleton() {
        (void)rocfft_cleanup();
    }

    // no copies or moves allowed
    rocfft_singleton(const rocfft_singleton& other) = delete;
    rocfft_singleton(rocfft_singleton&& other) noexcept = delete;
    rocfft_singleton& operator=(const rocfft_singleton& other) = delete;
    rocfft_singleton& operator=(rocfft_singleton&& other) noexcept = delete;

public:
    static void init() {
        static rocfft_singleton instance;
        (void)instance;
    }
};

/// Commit impl class specialization for rocFFT.
template <dft::precision prec, dft::domain dom>
class rocfft_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;
    // For real to complex transforms, the "transform_type" arg also encodes the direction (e.g. rocfft_transform_type_*_forward vs rocfft_transform_type_*_backward)
    // in the plan so we must have one for each direction.
    // We also need this because oneMath uses a directionless "FWD_DISTANCE" and "BWD_DISTANCE" while rocFFT uses a directional "in_distance" and "out_distance".
    // The same is also true for "FORWARD_SCALE" and "BACKWARD_SCALE".
    // handles[0] is forward, handles[1] is backward
    std::array<rocfft_handle, 2> handles{};
    std::int64_t offset_fwd_in, offset_fwd_out, offset_bwd_in, offset_bwd_out;

public:
    rocfft_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::rocfft,
                                                                config_values) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw math::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
        // initialise the rocFFT global state
        rocfft_singleton::init();
    }

    void clean_plans() {
        if (handles[0].plan) {
            if (rocfft_plan_destroy(handles[0].plan.value()) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to destroy forward plan.");
            }
            handles[0].plan = std::nullopt;
        }
        if (handles[1].plan) {
            if (rocfft_plan_destroy(handles[1].plan.value()) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to destroy backward plan.");
            }
            handles[1].plan = std::nullopt;
        }

        if (handles[0].info) {
            if (rocfft_execution_info_destroy(handles[0].info.value()) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to destroy forward execution info .");
            }
            handles[0].info = std::nullopt;
        }
        if (handles[1].info) {
            if (rocfft_execution_info_destroy(handles[1].info.value()) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to destroy backward execution info .");
            }
            handles[1].info = std::nullopt;
        }
        free_internal_workspace_if_rqd(handles[0], "clear_plans");
        free_internal_workspace_if_rqd(handles[1], "clear_plans");
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        this->external_workspace_helper_ =
            oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);
        clean_plans();

        const rocfft_result_placement placement =
            (config_values.placement == dft::config_value::INPLACE) ? rocfft_placement_inplace
                                                                    : rocfft_placement_notinplace;

        constexpr rocfft_transform_type fwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                return rocfft_transform_type_complex_forward;
            }
            else {
                return rocfft_transform_type_real_forward;
            }
        }();
        constexpr rocfft_transform_type bwd_type = [] {
            if constexpr (dom == dft::domain::COMPLEX) {
                return rocfft_transform_type_complex_inverse;
            }
            else {
                return rocfft_transform_type_real_inverse;
            }
        }();

        constexpr rocfft_precision precision = [] {
            if constexpr (prec == dft::precision::SINGLE) {
                return rocfft_precision_single;
            }
            else {
                return rocfft_precision_double;
            }
        }();

        const std::size_t dimensions = config_values.dimensions.size();

        constexpr std::size_t max_supported_dims = 3;
        std::array<std::size_t, max_supported_dims> lengths;
        // rocfft does dimensions in the reverse order to oneMath
        std::copy(config_values.dimensions.crbegin(), config_values.dimensions.crend(),
                  lengths.data());

        const std::size_t number_of_transforms =
            static_cast<std::size_t>(config_values.number_of_transforms);

        const std::size_t fwd_distance = static_cast<std::size_t>(config_values.fwd_dist);
        const std::size_t bwd_distance = static_cast<std::size_t>(config_values.bwd_dist);

        const rocfft_array_type fwd_array_ty = [&config_values]() {
            if constexpr (dom == dft::domain::COMPLEX) {
                if (config_values.complex_storage == dft::config_value::COMPLEX_COMPLEX) {
                    return rocfft_array_type_complex_interleaved;
                }
                else {
                    return rocfft_array_type_complex_planar;
                }
            }
            else {
                return rocfft_array_type_real;
            }
        }();
        const rocfft_array_type bwd_array_ty = [&config_values]() {
            if constexpr (dom == dft::domain::COMPLEX) {
                if (config_values.complex_storage == dft::config_value::COMPLEX_COMPLEX) {
                    return rocfft_array_type_complex_interleaved;
                }
                else {
                    return rocfft_array_type_complex_planar;
                }
            }
            else {
                if (config_values.conj_even_storage != dft::config_value::COMPLEX_COMPLEX) {
                    throw math::exception(
                        "dft/backends/rocfft", __FUNCTION__,
                        "only COMPLEX_COMPLEX conjugate_even_storage is supported");
                }
                return rocfft_array_type_hermitian_interleaved;
            }
        }();

        auto stride_api_choice = dft::detail::get_stride_api(config_values);
        dft::detail::throw_on_invalid_stride_api("ROCFFT commit", stride_api_choice);
        dft::detail::stride_vectors<size_t> stride_vecs(config_values, stride_api_choice);

        // while rocfft interface accepts offsets, it does not actually handle them
        offset_fwd_in = stride_vecs.offset_fwd_in;
        offset_fwd_out = stride_vecs.offset_fwd_out;
        offset_bwd_in = stride_vecs.offset_bwd_in;
        offset_bwd_out = stride_vecs.offset_bwd_out;

        auto func = __FUNCTION__;
        auto check_strides = [&](const auto& strides) {
            for (int i = 1; i <= dimensions; i++) {
                for (int j = 1; j <= dimensions; j++) {
                    std::int64_t cplx_dim = config_values.dimensions[j - 1];
                    std::int64_t real_dim = (dom == dft::domain::REAL && j == dimensions)
                                                ? (cplx_dim / 2 + 1)
                                                : cplx_dim;
                    if (strides[i] > strides[j] && strides[i] % cplx_dim != 0 &&
                        strides[i] % real_dim != 0) {
                        // rocfft does not throw, it just produces wrong results
                        throw oneapi::math::unimplemented(
                            "DFT", func,
                            "rocfft requires a stride to be divisible by all dimensions associated with smaller strides!");
                    }
                }
            }
        };
        // bwd_in/out alias fwd_in/out, so no need to check everything.
        check_strides(stride_vecs.vec_a);
        check_strides(stride_vecs.vec_b);

        // Reformat slides to conform to rocFFT API.
        std::reverse(stride_vecs.vec_a.begin(), stride_vecs.vec_a.end());
        stride_vecs.vec_a.pop_back(); // Offset is not included.
        std::reverse(stride_vecs.vec_b.begin(), stride_vecs.vec_b.end());
        stride_vecs.vec_b.pop_back(); // Offset is not included.

        // This workaround is needed due to a confirmed issue in rocFFT from version
        // 1.0.23 to 1.0.30. Those rocFFT version correspond to rocm version from
        // 5.6.0 to 6.3.0.
        // Link to rocFFT issue: https://github.com/ROCm/rocFFT/issues/507
        if constexpr (rocfft_version_major == 1 && rocfft_version_minor == 0 &&
                      (rocfft_version_patch > 22 && rocfft_version_patch < 31)) {
            // rocFFT's functional status for problems like cfoA:B:1xB:1:A is unknown as
            // of 4ed3e97bb7c11531684168665d5a980fde0284c9 (due to project's implementation preventing testing thereof)
            if (dom == dft::domain::COMPLEX &&
                config_values.placement == dft::config_value::NOT_INPLACE && dimensions > 2) {
                if (stride_vecs.vec_a != stride_vecs.vec_b)
                    throw oneapi::math::unimplemented(
                        "DFT", func,
                        "due to a bug in rocfft version in use, it requires fwd and bwd stride to be the same for COMPLEX out_of_place computations");
            }
        }

        rocfft_plan_description plan_desc_fwd, plan_desc_bwd; // Can't reuse with ROCm 6 due to bug.
        if (rocfft_plan_description_create(&plan_desc_fwd) != rocfft_status_success) {
            throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                  "Failed to create plan description.");
        }
        if (rocfft_plan_description_create(&plan_desc_bwd) != rocfft_status_success) {
            throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                  "Failed to create plan description.");
        }
        // plan_description can be destroyed afted plan_create
        auto description_destroy = [](rocfft_plan_description p) {
            if (rocfft_plan_description_destroy(p) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to destroy plan description.");
            }
        };
        std::unique_ptr<rocfft_plan_description_t, decltype(description_destroy)>
            description_destroyer_fwd(plan_desc_fwd, description_destroy);
        std::unique_ptr<rocfft_plan_description_t, decltype(description_destroy)>
            description_destroyer_bwd(plan_desc_bwd, description_destroy);

        std::array<std::size_t, 3> stride_a_indices{ 0, 1, 2 };
        std::sort(&stride_a_indices[0], &stride_a_indices[dimensions],
                  [&](std::size_t a, std::size_t b) {
                      return stride_vecs.vec_a[a] < stride_vecs.vec_a[b];
                  });
        std::array<std::size_t, 3> stride_b_indices{ 0, 1, 2 };
        std::sort(&stride_b_indices[0], &stride_b_indices[dimensions],
                  [&](std::size_t a, std::size_t b) {
                      return stride_vecs.vec_b[a] < stride_vecs.vec_b[b];
                  });
        std::array<std::size_t, max_supported_dims> lengths_cplx = lengths;
        if (dom == dft::domain::REAL) {
            lengths_cplx[0] = lengths_cplx[0] / 2 + 1;
        }
        // When creating real-complex descriptions with INPUT/OUTPUT_STRIDES,
        // the strides will always be wrong for one of the directions.
        // This is because the least significant dimension is symmetric.
        // If the strides are invalid (too small to fit) then just don't bother creating the plan.
        auto are_strides_smaller_than_lengths = [=](auto& svec, auto& sindices,
                                                    auto& domain_lengths) {
            return dimensions == 1 ||
                   (domain_lengths[sindices[0]] <= svec[sindices[1]] &&
                    (dimensions == 2 ||
                     svec[sindices[1]] * domain_lengths[sindices[1]] <= svec[sindices[2]]));
        };

        const bool vec_a_valid_as_fwd_domain =
            are_strides_smaller_than_lengths(stride_vecs.vec_a, stride_a_indices, lengths);
        const bool vec_b_valid_as_fwd_domain =
            are_strides_smaller_than_lengths(stride_vecs.vec_b, stride_b_indices, lengths);
        const bool vec_a_valid_as_bwd_domain =
            are_strides_smaller_than_lengths(stride_vecs.vec_a, stride_a_indices, lengths_cplx);
        const bool vec_b_valid_as_bwd_domain =
            are_strides_smaller_than_lengths(stride_vecs.vec_b, stride_b_indices, lengths_cplx);

        // Test if the stride vector being used as the fwd/bwd domain for each direction has valid strides for that use.
        bool valid_forward = (stride_vecs.fwd_in == stride_vecs.vec_a &&
                              vec_a_valid_as_fwd_domain && vec_b_valid_as_bwd_domain) ||
                             (vec_b_valid_as_fwd_domain && vec_a_valid_as_bwd_domain);
        bool valid_backward = (stride_vecs.bwd_in == stride_vecs.vec_a &&
                               vec_a_valid_as_bwd_domain && vec_b_valid_as_fwd_domain) ||
                              (vec_b_valid_as_bwd_domain && vec_a_valid_as_fwd_domain);

        if (!valid_forward && !valid_backward) {
            throw math::exception("dft/backends/cufft", __FUNCTION__, "Invalid strides.");
        }

        if (valid_forward) {
            auto res =
                rocfft_plan_description_set_data_layout(plan_desc_fwd, fwd_array_ty, bwd_array_ty,
                                                        nullptr, // in offsets
                                                        nullptr, // out offsets
                                                        dimensions,
                                                        stride_vecs.fwd_in.data(), //in strides
                                                        fwd_distance, // in distance
                                                        dimensions,
                                                        stride_vecs.fwd_out.data(), // out strides
                                                        bwd_distance // out distance
                );
            if (res != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to set forward data layout.");
            }

            if (rocfft_plan_description_set_scale_factor(plan_desc_fwd, config_values.fwd_scale) !=
                rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to set forward scale factor.");
            }

            rocfft_plan fwd_plan;
            res = rocfft_plan_create(&fwd_plan, placement, fwd_type, precision, dimensions,
                                     lengths.data(), number_of_transforms, plan_desc_fwd);

            if (res != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to create forward plan.");
            }

            handles[0].plan = fwd_plan;

            rocfft_execution_info fwd_info;
            if (rocfft_execution_info_create(&fwd_info) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to create forward execution info.");
            }
            handles[0].info = fwd_info;

            if (config_values.workspace_placement == config_value::WORKSPACE_AUTOMATIC) {
                std::int64_t work_buf_size = get_rocfft_workspace_bytes(handles[0], "commit");
                if (work_buf_size != 0) {
                    void* work_buf;
                    if (hipMalloc(&work_buf, work_buf_size) != hipSuccess) {
                        throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                              "Failed to get allocate forward work buffer.");
                    }
                    set_workspace_impl(handles[0], reinterpret_cast<scalar_type*>(work_buf),
                                       work_buf_size, "commit");
                    handles[0].buffer = work_buf;
                }
            }
        }

        if (valid_backward) {
            auto res =
                rocfft_plan_description_set_data_layout(plan_desc_bwd, bwd_array_ty, fwd_array_ty,
                                                        nullptr, // in offsets
                                                        nullptr, // out offsets
                                                        dimensions,
                                                        stride_vecs.bwd_in.data(), //in strides
                                                        bwd_distance, // in distance
                                                        dimensions,
                                                        stride_vecs.bwd_out.data(), // out strides
                                                        fwd_distance // out distance
                );
            if (res != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to set backward data layout.");
            }

            if (rocfft_plan_description_set_scale_factor(plan_desc_bwd, config_values.bwd_scale) !=
                rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to set backward scale factor.");
            }

            rocfft_plan bwd_plan;
            res = rocfft_plan_create(&bwd_plan, placement, bwd_type, precision, dimensions,
                                     lengths.data(), number_of_transforms, plan_desc_bwd);
            if (res != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to create backward rocFFT plan.");
            }
            handles[1].plan = bwd_plan;

            rocfft_execution_info bwd_info;
            if (rocfft_execution_info_create(&bwd_info) != rocfft_status_success) {
                throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                      "Failed to create backward execution info.");
            }
            handles[1].info = bwd_info;

            if (config_values.workspace_placement == config_value::WORKSPACE_AUTOMATIC) {
                std::int64_t work_buf_size = get_rocfft_workspace_bytes(handles[1], "commit");
                if (work_buf_size != 0) {
                    void* work_buf;
                    if (hipMalloc(&work_buf, work_buf_size) != hipSuccess) {
                        throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                              "Failed to get allocate backward work buffer.");
                    }
                    set_workspace_impl(handles[1], reinterpret_cast<scalar_type*>(work_buf),
                                       work_buf_size, "commit");
                    handles[1].buffer = work_buf;
                }
            }
        }
    }

    ~rocfft_commit() override {
        clean_plans();
    }

    // Rule of three. Copying could lead to memory safety issues.
    rocfft_commit(const rocfft_commit& other) = delete;
    rocfft_commit& operator=(const rocfft_commit& other) = delete;

    void* get_handle() noexcept override {
        return handles.data();
    }

    std::array<std::int64_t, 2> get_offsets_fwd() noexcept {
        return { offset_fwd_in, offset_fwd_out };
    }

    std::array<std::int64_t, 2> get_offsets_bwd() noexcept {
        return { offset_bwd_in, offset_bwd_out };
    }

    /** Get the requried worspace size for a rocfft plan. Implementation to be shared by internal and external workspace mechanisms.

     * @param handle rocfft_handle. Expected to have valid rocfft_plan.
     * @param function The name of the function to give when generating exceptions
     * @return Required space in bytes
    **/
    std::int64_t get_rocfft_workspace_bytes(rocfft_handle& handle, const char* function) {
        if (!handle.plan) {
            throw math::exception("dft/backends/rocfft", function, "Missing internal rocfft plan");
        }
        std::size_t size = 0;
        if (rocfft_plan_get_work_buffer_size(*handle.plan, &size) != rocfft_status_success) {
            throw math::exception("dft/backends/rocfft", function,
                                  "Failed to get rocfft work buffer size.");
        }
        return static_cast<std::int64_t>(size);
    }

    /** Set the rocFFT workspace. Implementation to be shared by internal workspace allocation and external workspace
     * mechanisms. Does not set handle.buffer.
     * 
     * @param handle rocfft_handle. Expected to have valid rocfft_plan and rocfft_info, but no associated buffer.
     * @param workspace Pointer to allocation to use as workspace
     * @param workspace_bytes The size (in bytes) of the given workspace
     * @param function The name of the function to give when generating exceptions
    **/
    void set_workspace_impl(const rocfft_handle& handle, scalar_type* workspace,
                            std::int64_t workspace_bytes, const char* function) {
        if (!handle.info) {
            throw math::exception(
                "dft/backends/rocfft", function,
                "Could not set rocFFT workspace - handle has no associated rocfft_info.");
        }
        if (handle.buffer) {
            throw math::exception(
                "dft/backends/rocfft", function,
                "Could not set rocFFT workspace - an internal buffer is already set.");
        }
        if (workspace_bytes && workspace == nullptr) {
            throw math::exception("dft/backends/rocfft", function, "Trying to nullptr workspace.");
        }
        auto info = *handle.info;
        if (workspace_bytes &&
            rocfft_execution_info_set_work_buffer(info, static_cast<void*>(workspace),
                                                  static_cast<std::size_t>(workspace_bytes)) !=
                rocfft_status_success) {
            throw math::exception("dft/backends/rocfft", function, "Failed to set work buffer.");
        }
    }

    void free_internal_workspace_if_rqd(rocfft_handle& handle, const char* function) {
        if (handle.buffer) {
            if (hipFree(*handle.buffer) != hipSuccess) {
                throw math::exception("dft/backends/rocfft", function,
                                      "Failed to free internal buffer.");
            }
            handle.buffer = std::nullopt;
        }
    }

    virtual void set_workspace(scalar_type* usm_workspace) override {
        std::int64_t total_workspace_bytes{ this->get_workspace_external_bytes() };
        this->external_workspace_helper_.set_workspace_throw(*this, usm_workspace);
        if (handles[0].plan) {
            free_internal_workspace_if_rqd(handles[0], "set_workspace");
            set_workspace_impl(handles[0], usm_workspace, total_workspace_bytes, "set_workspace");
        }
        if (handles[1].plan) {
            free_internal_workspace_if_rqd(handles[1], "set_workspace");
            set_workspace_impl(handles[1], usm_workspace, total_workspace_bytes, "set_workspace");
        }
    }

    void set_buffer_workspace(rocfft_handle& handle, sycl::buffer<scalar_type>& buffer_workspace) {
        auto workspace_bytes = buffer_workspace.size() * sizeof(scalar_type);
        if (buffer_workspace.size() == 0) {
            return; // Nothing to do.
        }
        this->get_queue().submit([&](sycl::handler& cgh) {
            auto workspace_acc =
                buffer_workspace.template get_access<sycl::access::mode::read_write>(cgh);
            cgh.host_task([=](sycl::interop_handle ih) {
                auto workspace_native = reinterpret_cast<scalar_type*>(
                    ih.get_native_mem<sycl::backend::ext_oneapi_hip>(workspace_acc));
                set_workspace_impl(handle, workspace_native, workspace_bytes, "set_workspace");
            });
        });
        this->get_queue().wait_and_throw();
    }

    virtual void set_workspace(sycl::buffer<scalar_type>& buffer_workspace) override {
        this->external_workspace_helper_.set_workspace_throw(*this, buffer_workspace);
        std::size_t total_workspace_count =
            static_cast<std::size_t>(this->get_workspace_external_bytes()) / sizeof(scalar_type);
        if (handles[0].plan) {
            free_internal_workspace_if_rqd(handles[0], "set_workspace");
            set_buffer_workspace(handles[0], buffer_workspace);
        }
        if (handles[1].plan) {
            free_internal_workspace_if_rqd(handles[1], "set_workspace");
            set_buffer_workspace(handles[1], buffer_workspace);
        }
    }

    std::int64_t get_plan_workspace_size_bytes(rocfft_plan_t* plan) {
        // plan work buffer
        if (plan == nullptr) {
            throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                  "Missing internal rocFFT plan.");
        }
        std::size_t work_buf_size;
        if (rocfft_plan_get_work_buffer_size(plan, &work_buf_size) != rocfft_status_success) {
            throw math::exception("dft/backends/rocfft", __FUNCTION__,
                                  "Failed to get work buffer size.");
        }
        return static_cast<std::int64_t>(work_buf_size);
    }

    virtual std::int64_t get_workspace_external_bytes_impl() override {
        std::int64_t size0 = handles[0].plan ? get_plan_workspace_size_bytes(*handles[0].plan) : 0;
        std::int64_t size1 = handles[1].plan ? get_plan_workspace_size_bytes(*handles[1].plan) : 0;
        return std::max(size0, size1);
    };

#define BACKEND rocfft
#include "../backend_compute_signature.cxx"
#undef BACKEND
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::rocfft_commit<prec, dom>(sycl_queue, desc.get_values());
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
    return static_cast<rocfft_commit<prec, dom>*>(commit)->get_offsets_fwd();
}

template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_bwd(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<rocfft_commit<prec, dom>*>(commit)->get_offsets_bwd();
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

} // namespace oneapi::math::dft::rocfft
