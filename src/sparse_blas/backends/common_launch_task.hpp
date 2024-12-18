/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMATH_SPARSE_BLAS_BACKENDS_COMMON_LAUNCH_TASK_HPP_
#define _ONEMATH_SPARSE_BLAS_BACKENDS_COMMON_LAUNCH_TASK_HPP_

/// This file provide a helper function to submit host_task using buffers or USM seamlessly

namespace oneapi::math::sparse::detail {

template <typename T, typename Container>
auto get_value_accessor(sycl::handler& cgh, Container container) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1>*>(container->value_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T, typename... Ts>
auto get_fp_accessors(sycl::handler& cgh, Ts... containers) {
    return std::array<sycl::accessor<T, 1>, sizeof...(containers)>{ get_value_accessor<T>(
        cgh, containers)... };
}

template <typename T>
auto get_row_accessor(sycl::handler& cgh, matrix_handle_t smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1>*>(smhandle->row_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_col_accessor(sycl::handler& cgh, matrix_handle_t smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1>*>(smhandle->col_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_int_accessors(sycl::handler& cgh, matrix_handle_t smhandle) {
    return std::array<sycl::accessor<T, 1>, 2>{ get_row_accessor<T>(cgh, smhandle),
                                                get_col_accessor<T>(cgh, smhandle) };
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_host_task(sycl::handler& cgh, sycl::queue& queue, Functor functor,
                      CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly
    // handled. The accessors's pointer have already been set to the native
    // container types in previous functions. This assumes the underlying
    // pointer of the buffer does not change. This is not guaranteed by the SYCL
    // specification but should be true for all the implementations. This
    // assumption avoids the overhead of resetting the pointer of all data
    // handles for each enqueued command.
    cgh.host_task([functor, queue, capture_only_accessors...](sycl::interop_handle ih) {
        auto unused = std::make_tuple(capture_only_accessors...);
        (void)unused;
        functor(ih);
    });
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_host_task_with_acc(sycl::handler& cgh, sycl::queue& queue, Functor functor,
                               sycl::accessor<std::uint8_t> workspace_acc,
                               CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly
    // handled. The accessors's pointer have already been set to the native
    // container types in previous functions. This assumes the underlying
    // pointer of the buffer does not change. This is not guaranteed by the SYCL
    // specification but should be true for all the implementations. This
    // assumption avoids the overhead of resetting the pointer of all data
    // handles for each enqueued command.
    cgh.host_task(
        [functor, queue, workspace_acc, capture_only_accessors...](sycl::interop_handle ih) {
            auto unused = std::make_tuple(capture_only_accessors...);
            (void)unused;
            functor(ih, workspace_acc);
        });
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_native_command_ext(sycl::handler& cgh, sycl::queue& queue, Functor functor,
                               const std::vector<sycl::event>& dependencies,
                               CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly
    // handled. The accessors's pointer have already been set to the native
    // container types in previous functions. This assumes the underlying
    // pointer of the buffer does not change. This is not guaranteed by the SYCL
    // specification but should be true for all the implementations. This
    // assumption avoids the overhead of resetting the pointer of all data
    // handles for each enqueued command.
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command(
        [functor, queue, dependencies, capture_only_accessors...](sycl::interop_handle ih) {
            auto unused = std::make_tuple(capture_only_accessors...);
            (void)unused;
            // The functor using ext_codeplay_enqueue_native_command need to
            // explicitly wait on the events for the SPARSE domain. The
            // extension ext_codeplay_enqueue_native_command is used to launch
            // the compute operation which depends on the previous optimize
            // step. In cuSPARSE the optimize step is synchronous but it is
            // asynchronous in oneMath. The optimize step may not use the CUDA
            // stream which would make it impossible for
            // ext_codeplay_enqueue_native_command to automatically ensure it
            // has completed before the compute function starts. These waits are
            // used to ensure the optimize step has completed before starting
            // the computation.
            for (auto event : dependencies) {
                event.wait();
            }
            functor(ih);
        });
#else
    (void)dependencies;
    submit_host_task(cgh, queue, functor, capture_only_accessors...);
#endif
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_native_command_ext_with_acc(sycl::handler& cgh, sycl::queue& queue, Functor functor,
                                        const std::vector<sycl::event>& dependencies,
                                        sycl::accessor<std::uint8_t> workspace_acc,
                                        CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly
    // handled. The accessors's pointer have already been set to the native
    // container types in previous functions. This assumes the underlying
    // pointer of the buffer does not change. This is not guaranteed by the SYCL
    // specification but should be true for all the implementations. This
    // assumption avoids the overhead of resetting the pointer of all data
    // handles for each enqueued command.
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    cgh.ext_codeplay_enqueue_native_command([functor, queue, dependencies, workspace_acc,
                                             capture_only_accessors...](sycl::interop_handle ih) {
        auto unused = std::make_tuple(capture_only_accessors...);
        (void)unused;
        // The functor using ext_codeplay_enqueue_native_command need to
        // explicitly wait on the events for the SPARSE domain. The extension
        // ext_codeplay_enqueue_native_command is used to launch the compute
        // operation which depends on the previous optimize step. In cuSPARSE
        // the optimize step is synchronous but it is asynchronous in oneMath.
        // The optimize step may not use the CUDA stream which would make it
        // impossible for ext_codeplay_enqueue_native_command to automatically
        // ensure it has completed before the compute function starts. These
        // waits are used to ensure the optimize step has completed before
        // starting the computation.
        for (auto event : dependencies) {
            event.wait();
        }
        functor(ih, workspace_acc);
    });
#else
    (void)dependencies;
    submit_host_task_with_acc(cgh, queue, functor, workspace_acc, capture_only_accessors...);
#endif
}

/// Helper submit functions to capture all accessors from the generic containers
/// \p other_containers and ensure the dependencies of buffers are respected.
/// The accessors are not directly used as the underlying data pointer has
/// already been captured in previous functions.
/// \p workspace_buffer is an optional buffer. Its accessor will be given to the
/// functor as a last argument if \p UseWorkspace is true.
/// \p UseWorkspace must be true to use the given \p workspace_buffer.
/// \p UseEnqueueNativeCommandExt controls whether host_task are used or the
/// extension ext_codeplay_enqueue_native_command is used to launch tasks. The
/// extension should only be used for asynchronous functions using native
/// backend's functions. The extension can only be used for in-order queues as
/// the same cuStream needs to be used for the 3 steps to run an operation:
/// querying the buffer size, optimizing and running the computation. This means
/// a different cuStream can be used inside the native_command than the native
/// cuStream used by the extension.
template <bool UseWorkspace, bool UseEnqueueNativeCommandExt, typename Functor, typename... Ts>
sycl::event dispatch_submit_impl_fp_int(const std::string& function_name, sycl::queue queue,
                                        const std::vector<sycl::event>& dependencies,
                                        Functor functor, matrix_handle_t sm_handle,
                                        sycl::buffer<std::uint8_t> workspace_buffer,
                                        Ts... other_containers) {
    bool is_in_order_queue = queue.is_in_order();
    if (sm_handle->all_use_buffer()) {
        data_type value_type = sm_handle->get_value_type();
        data_type int_type = sm_handle->get_int_type();

#define ONEMATH_SUBMIT(FP_TYPE, INT_TYPE)                                                         \
    return queue.submit([&](sycl::handler& cgh) {                                                 \
        cgh.depends_on(dependencies);                                                             \
        auto fp_accs = get_fp_accessors<FP_TYPE>(cgh, sm_handle, other_containers...);            \
        auto int_accs = get_int_accessors<INT_TYPE>(cgh, sm_handle);                              \
        auto workspace_acc = workspace_buffer.get_access<sycl::access::mode::read_write>(cgh);    \
        if constexpr (UseWorkspace) {                                                             \
            if constexpr (UseEnqueueNativeCommandExt) {                                           \
                if (is_in_order_queue) {                                                          \
                    submit_native_command_ext_with_acc(cgh, queue, functor, dependencies,         \
                                                       workspace_acc, fp_accs, int_accs);         \
                }                                                                                 \
                else {                                                                            \
                    submit_host_task_with_acc(cgh, queue, functor, workspace_acc, fp_accs,        \
                                              int_accs);                                          \
                }                                                                                 \
            }                                                                                     \
            else {                                                                                \
                submit_host_task_with_acc(cgh, queue, functor, workspace_acc, fp_accs, int_accs); \
            }                                                                                     \
        }                                                                                         \
        else {                                                                                    \
            (void)workspace_buffer;                                                               \
            if constexpr (UseEnqueueNativeCommandExt) {                                           \
                if (is_in_order_queue) {                                                          \
                    submit_native_command_ext(cgh, queue, functor, dependencies, fp_accs,         \
                                              int_accs);                                          \
                }                                                                                 \
                else {                                                                            \
                    submit_host_task(cgh, queue, functor, fp_accs, int_accs);                     \
                }                                                                                 \
            }                                                                                     \
            else {                                                                                \
                submit_host_task(cgh, queue, functor, fp_accs, int_accs);                         \
            }                                                                                     \
        }                                                                                         \
    })
#define ONEMATH_SUBMIT_INT(FP_TYPE)            \
    if (int_type == data_type::int32) {        \
        ONEMATH_SUBMIT(FP_TYPE, std::int32_t); \
    }                                          \
    else if (int_type == data_type::int64) {   \
        ONEMATH_SUBMIT(FP_TYPE, std::int64_t); \
    }

        if (value_type == data_type::real_fp32) {
            ONEMATH_SUBMIT_INT(float)
        }
        else if (value_type == data_type::real_fp64) {
            ONEMATH_SUBMIT_INT(double)
        }
        else if (value_type == data_type::complex_fp32) {
            ONEMATH_SUBMIT_INT(std::complex<float>)
        }
        else if (value_type == data_type::complex_fp64) {
            ONEMATH_SUBMIT_INT(std::complex<double>)
        }

#undef ONEMATH_SUBMIT_INT
#undef ONEMATH_SUBMIT

        throw oneapi::math::exception("sparse_blas", function_name,
                                      "Could not dispatch buffer kernel to a supported type");
    }
    else {
        // USM submit does not need to capture accessors
        if constexpr (!UseWorkspace) {
            return queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(dependencies);
                if constexpr (UseEnqueueNativeCommandExt) {
                    if (is_in_order_queue) {
                        submit_native_command_ext(cgh, queue, functor, dependencies);
                    }
                    else {
                        submit_host_task(cgh, queue, functor);
                    }
                }
                else {
                    submit_host_task(cgh, queue, functor);
                }
            });
        }
        else {
            throw oneapi::math::exception("sparse_blas", function_name,
                                          "Internal error: Cannot use accessor workspace with USM");
        }
    }
}

/// Similar to dispatch_submit_impl_fp_int but only dispatches the host_task based on the floating point value type.
template <typename Functor, typename ContainerT>
sycl::event dispatch_submit_impl_fp(const std::string& function_name, sycl::queue queue,
                                    const std::vector<sycl::event>& dependencies, Functor functor,
                                    ContainerT container_handle) {
    if (container_handle->all_use_buffer()) {
        data_type value_type = container_handle->get_value_type();

#define ONEMATH_SUBMIT(FP_TYPE)                                          \
    return queue.submit([&](sycl::handler& cgh) {                        \
        cgh.depends_on(dependencies);                                    \
        auto fp_accs = get_fp_accessors<FP_TYPE>(cgh, container_handle); \
        submit_host_task(cgh, queue, functor, fp_accs);                  \
    })

        if (value_type == data_type::real_fp32) {
            ONEMATH_SUBMIT(float);
        }
        else if (value_type == data_type::real_fp64) {
            ONEMATH_SUBMIT(double);
        }
        else if (value_type == data_type::complex_fp32) {
            ONEMATH_SUBMIT(std::complex<float>);
        }
        else if (value_type == data_type::complex_fp64) {
            ONEMATH_SUBMIT(std::complex<double>);
        }

#undef ONEMATH_SUBMIT

        throw oneapi::math::exception("sparse_blas", function_name,
                                      "Could not dispatch buffer kernel to a supported type");
    }
    else {
        return queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(dependencies);
            submit_host_task(cgh, queue, functor);
        });
    }
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string& function_name, sycl::queue queue, Functor functor,
                            matrix_handle_t sm_handle, sycl::buffer<std::uint8_t> workspace_buffer,
                            Ts... other_containers) {
    constexpr bool UseWorkspace = true;
    constexpr bool UseEnqueueNativeCommandExt = false;
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, {}, functor, sm_handle, workspace_buffer, other_containers...);
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string& function_name, sycl::queue queue,
                            const std::vector<sycl::event>& dependencies, Functor functor,
                            matrix_handle_t sm_handle, Ts... other_containers) {
    constexpr bool UseWorkspace = false;
    constexpr bool UseEnqueueNativeCommandExt = false;
    sycl::buffer<std::uint8_t> no_workspace(sycl::range<1>(0));
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, dependencies, functor, sm_handle, no_workspace, other_containers...);
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string& function_name, sycl::queue queue, Functor functor,
                            matrix_handle_t sm_handle, Ts... other_containers) {
    constexpr bool UseWorkspace = false;
    constexpr bool UseEnqueueNativeCommandExt = false;
    sycl::buffer<std::uint8_t> no_workspace(sycl::range<1>(0));
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, {}, functor, sm_handle, no_workspace, other_containers...);
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit_native_ext(const std::string& function_name, sycl::queue queue,
                                       Functor functor, matrix_handle_t sm_handle,
                                       sycl::buffer<std::uint8_t> workspace_buffer,
                                       Ts... other_containers) {
    constexpr bool UseWorkspace = true;
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    constexpr bool UseEnqueueNativeCommandExt = true;
#else
    constexpr bool UseEnqueueNativeCommandExt = false;
#endif
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, {}, functor, sm_handle, workspace_buffer, other_containers...);
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit_native_ext(const std::string& function_name, sycl::queue queue,
                                       const std::vector<sycl::event>& dependencies,
                                       Functor functor, matrix_handle_t sm_handle,
                                       Ts... other_containers) {
    constexpr bool UseWorkspace = false;
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    constexpr bool UseEnqueueNativeCommandExt = true;
#else
    constexpr bool UseEnqueueNativeCommandExt = false;
#endif
    sycl::buffer<std::uint8_t> no_workspace(sycl::range<1>(0));
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, dependencies, functor, sm_handle, no_workspace, other_containers...);
}

/// Helper function for dispatch_submit_impl_fp_int
template <typename Functor, typename... Ts>
sycl::event dispatch_submit_native_ext(const std::string& function_name, sycl::queue queue,
                                       Functor functor, matrix_handle_t sm_handle,
                                       Ts... other_containers) {
    constexpr bool UseWorkspace = false;
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    constexpr bool UseEnqueueNativeCommandExt = true;
#else
    constexpr bool UseEnqueueNativeCommandExt = false;
#endif
    sycl::buffer<std::uint8_t> no_workspace(sycl::range<1>(0));
    return dispatch_submit_impl_fp_int<UseWorkspace, UseEnqueueNativeCommandExt>(
        function_name, queue, {}, functor, sm_handle, no_workspace, other_containers...);
}

} // namespace oneapi::math::sparse::detail

#endif // _ONEMATH_SPARSE_BLAS_BACKENDS_COMMON_LAUNCH_TASK_HPP_
