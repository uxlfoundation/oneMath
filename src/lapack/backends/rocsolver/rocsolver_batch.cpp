/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright 2022 Intel Corporation
*
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
#include "rocsolver_helper.hpp"
#include "rocsolver_task.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/detail/rocsolver/onemath_lapack_rocsolver.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace rocsolver {

// BATCH BUFFER API

template <typename Func, typename T>
inline void geqrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<T>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                        sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, batch_size, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, stride_a, tau_,
                                        stride_tau, batch_size);
        });
    });
}

#define GEQRF_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                   \
    void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
                     std::int64_t lda, std::int64_t stride_a, sycl::buffer<TYPE>& tau,          \
                     std::int64_t stride_tau, std::int64_t batch_size,                          \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {            \
        geqrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, stride_a, tau,  \
                    stride_tau, batch_size, scratchpad, scratchpad_size);                       \
    }

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
GEQRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgeqrf_strided_batched_64)
#else
GEQRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgeqrf_strided_batched)
#endif

#undef GEQRF_STRIDED_BATCH_LAUNCHER
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getri_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv, sycl::buffer<float>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                 std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrs_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "getrf_batch");
}
// rocsolver has no batched orgqr/ungqr, so the batch is walked one matrix at a
// time inside a single host task.
template <typename Func, typename T>
inline void orgqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                        std::int64_t stride_a, sycl::buffer<T>& tau, std::int64_t stride_tau,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, batch_size, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto tau_ = sc.get_mem<rocmDataType*>(tau_acc);
            rocblas_status err;
            for (std::int64_t i = 0; i < batch_size; ++i) {
                rocsolver_native_named_func(func_name, func, err, handle, m, n, k,
                                            a_ + i * stride_a, lda, tau_ + i * stride_tau);
            }
        });
    });
}

#define ORGQR_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                     \
    void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     sycl::buffer<TYPE>& tau, std::int64_t stride_tau, std::int64_t batch_size,   \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, tau, \
                    stride_tau, batch_size, scratchpad, scratchpad_size);                         \
    }

ORGQR_STRIDED_BATCH_LAUNCHER(float, rocsolver_sorgqr)
ORGQR_STRIDED_BATCH_LAUNCHER(double, rocsolver_dorgqr)

#undef ORGQR_STRIDED_BATCH_LAUNCHER

#define UNGQR_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                     \
    void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     sycl::buffer<TYPE>& tau, std::int64_t stride_tau, std::int64_t batch_size,   \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, tau, \
                    stride_tau, batch_size, scratchpad, scratchpad_size);                         \
    }

UNGQR_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cungqr)
UNGQR_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zungqr)

#undef UNGQR_STRIDED_BATCH_LAUNCHER
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrf_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}
void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "potrs_batch");
}

// BATCH USM API

template <typename Func, typename T>
inline sycl::event geqrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, T* a, std::int64_t lda, std::int64_t stride_a,
                               T* tau, std::int64_t stride_tau, std::int64_t batch_size,
                               T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, batch_size, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, stride_a, tau_,
                                        stride_tau, batch_size);
        });
    });
    return done;
}

#define GEQRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                \
    sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,         \
                            std::int64_t lda, std::int64_t stride_a, TYPE* tau,                  \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad,  \
                            std::int64_t scratchpad_size,                                        \
                            const std::vector<sycl::event>& dependencies) {                      \
        return geqrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, stride_a, \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size,             \
                           dependencies);                                                        \
    }

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
GEQRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_strided_batched_64)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_strided_batched_64)
#else
GEQRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_strided_batched)
#endif

#undef GEQRF_STRIDED_BATCH_LAUNCHER_USM

// The group api is served by the rocsolver _batched entry points, which take a
// device resident array of matrix pointers.
template <typename Func, typename T>
inline sycl::event geqrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, T** a, std::int64_t* lda, T** tau,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (std::int64_t i = 0; i < group_count; ++i) {
        overflow_check(m[i], n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    // The batched entry point writes the Householder scalars of a whole group
    // into one contiguous array, so they are staged here and scattered back to
    // the caller supplied per-matrix arrays afterwards.
    std::vector<std::int64_t> tau_len(group_count);
    std::int64_t tau_stage_size = 0;
    for (std::int64_t i = 0; i < group_count; ++i) {
        tau_len[i] = std::min(m[i], n[i]);
        tau_stage_size += tau_len[i] * group_sizes[i];
    }

    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    T* tau_stage = (T*)malloc_device(sizeof(T) * tau_stage_size, queue);
    auto done_cpy = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.memcpy(a_dev, a, batch_size * sizeof(T*));
    });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done_cpy);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
            auto* tau_ = reinterpret_cast<rocmDataType*>(tau_stage);
            std::int64_t offset = 0;
            std::int64_t tau_offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; ++i) {
                const std::int64_t len = std::min(m[i], n[i]);
#if ONEMATH_ROCSOLVER_HAS_64BIT_API
                rocsolver_native_named_func(func_name, func, err, handle, m[i], n[i], a_ + offset,
                                            lda[i], tau_ + tau_offset, len, group_sizes[i]);
#else
                rocsolver_native_named_func(func_name, func, err, handle, (int)m[i], (int)n[i],
                                            a_ + offset, (int)lda[i], tau_ + tau_offset, len,
                                            (int)group_sizes[i]);
#endif
                offset += group_sizes[i];
                tau_offset += len * group_sizes[i];
            }
        });
    });

    std::vector<sycl::event> scatter_dependencies;
    scatter_dependencies.reserve(batch_size);
    for (std::int64_t i = 0, global_id = 0, tau_offset = 0; i < group_count; ++i) {
        const std::int64_t len = tau_len[i];
        for (std::int64_t j = 0; j < group_sizes[i]; ++j, ++global_id, tau_offset += len) {
            scatter_dependencies.push_back(queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(done);
                cgh.memcpy(tau[global_id], tau_stage + tau_offset, len * sizeof(T));
            }));
        }
    }

    auto done_scatter = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(scatter_dependencies);
        cgh.host_task([]() {});
    });

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done_scatter);
        cgh.host_task([=](sycl::interop_handle) {
            sycl::free(a_dev, queue);
            sycl::free(tau_stage, queue);
        });
    });
}

#define GEQRF_GROUP_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event geqrf_batch(                                                                     \
        sycl::queue& queue, std::int64_t* m, std::int64_t* n, TYPE** a, std::int64_t* lda,       \
        TYPE** tau, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,       \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return geqrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, tau,      \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
GEQRF_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_batched_64)
GEQRF_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_batched_64)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_batched_64)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_batched_64)
#else
GEQRF_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_batched)
#endif

#undef GEQRF_GROUP_BATCH_LAUNCHER_USM
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrf_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, float** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, double** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<float>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getri_batch(sycl::queue& queue, std::int64_t* n, std::complex<double>** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getri_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, float* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, float** a, std::int64_t* lda, std::int64_t** ipiv,
                        float** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, double** a, std::int64_t* lda, std::int64_t** ipiv,
                        double** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<float>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n,
                        std::int64_t* nrhs, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::complex<double>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "getrs_batch");
}
template <typename Func, typename T>
inline sycl::event orgqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, std::int64_t k, T* a, std::int64_t lda,
                               std::int64_t stride_a, T* tau, std::int64_t stride_tau,
                               std::int64_t batch_size, T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, batch_size, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto tau_ = reinterpret_cast<rocmDataType*>(tau);
            rocblas_status err;
            for (std::int64_t i = 0; i < batch_size; ++i) {
                rocsolver_native_named_func(func_name, func, err, handle, m, n, k,
                                            a_ + i * stride_a, lda, tau_ + i * stride_tau);
            }
        });
    });
    return done;
}

template <typename Func, typename T>
inline sycl::event orgqr_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, std::int64_t* k, T** a,
                               std::int64_t* lda, T** tau, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    overflow_check(group_count, scratchpad_size);
    for (std::int64_t i = 0; i < group_count; ++i)
        overflow_check(m[i], n[i], k[i], lda[i], group_sizes[i]);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            std::int64_t global_id = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; ++i) {
                for (std::int64_t j = 0; j < group_sizes[i]; ++j, ++global_id) {
                    auto a_ = reinterpret_cast<rocmDataType*>(a[global_id]);
                    auto tau_ = reinterpret_cast<rocmDataType*>(tau[global_id]);
                    rocsolver_native_named_func(func_name, func, err, handle, (int)m[i], (int)n[i],
                                                (int)k[i], a_, (int)lda[i], tau_);
                }
            }
        });
    });
    return done;
}

#define ORGQR_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                          \
    sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,    \
                            TYPE* a, std::int64_t lda, std::int64_t stride_a, TYPE* tau,           \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad,    \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda,          \
                           stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,     \
                           dependencies);                                                          \
    }                                                                                              \
    sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, \
                            TYPE** a, std::int64_t* lda, TYPE** tau, std::int64_t group_count,     \
                            std::int64_t* group_sizes, TYPE* scratchpad,                           \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,     \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);   \
    }

ORGQR_BATCH_LAUNCHER_USM(float, rocsolver_sorgqr)
ORGQR_BATCH_LAUNCHER_USM(double, rocsolver_dorgqr)

#undef ORGQR_BATCH_LAUNCHER_USM

#define UNGQR_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                          \
    sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,    \
                            TYPE* a, std::int64_t lda, std::int64_t stride_a, TYPE* tau,           \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad,    \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda,          \
                           stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,     \
                           dependencies);                                                          \
    }                                                                                              \
    sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, \
                            TYPE** a, std::int64_t* lda, TYPE** tau, std::int64_t group_count,     \
                            std::int64_t* group_sizes, TYPE* scratchpad,                           \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return orgqr_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,     \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);   \
    }

UNGQR_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cungqr)
UNGQR_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zungqr)

#undef UNGQR_BATCH_LAUNCHER_USM
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}
sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrf_batch");
}

template <typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, T** a, std::int64_t* lda,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    int* info = (int*)malloc_device(sizeof(int) * batch_size, queue);
    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    auto done_cpy =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_cpy);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
                auto* info_ = reinterpret_cast<rocblas_int*>(info);
                rocsolver_native_named_func(func_name, func, err, handle,
                                            get_rocblas_fill_mode(uplo[i]), (int)n[i], a_ + offset,
                                            (int)lda[i], info_ + offset, (int)group_sizes[i]);
                offset += group_sizes[i];
            }
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                         \
    sycl::event potrf_batch(                                                                      \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, TYPE** a,                  \
        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad, \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {             \
        return potrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda,         \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);  \
    }

POTRF_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_batched)
POTRF_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_batched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_batched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_batched)

#undef POTRF_BATCH_LAUNCHER_USM

sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t stride_a,
                        float* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, double* a, std::int64_t lda, std::int64_t stride_a,
                        double* b, std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}
sycl::event potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "potrs_batch");
}

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs, T** a,
                               std::int64_t* lda, T** b, std::int64_t* ldb,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];

        // rocsolver function only supports nrhs = 1
        if (nrhs[i] != 1)
            throw unimplemented("lapack", "potrs_batch",
                                "rocsolver potrs_batch only supports nrhs = 1");
    }

    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    T** b_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    auto done_cpy_a =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done_cpy_b =
        queue.submit([&](sycl::handler& h) { h.memcpy(b_dev, b, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_cpy_a);
        cgh.depends_on(done_cpy_b);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            rocblas_status err;
            for (int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
                auto** b_ = reinterpret_cast<rocmDataType**>(b_dev);
                rocsolver_native_named_func(func_name, func, err, handle,
                                            get_rocblas_fill_mode(uplo[i]), (int)n[i], (int)nrhs[i],
                                            a_ + offset, (int)lda[i], b_ + offset, (int)ldb[i],
                                            (int)group_sizes[i]);
                offset += group_sizes[i];
            }
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRS_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                          \
    sycl::event potrs_batch(                                                                       \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,         \
        TYPE** a, std::int64_t* lda, TYPE** b, std::int64_t* ldb, std::int64_t group_count,        \
        std::int64_t* group_sizes, TYPE* scratchpad, std::int64_t scratchpad_size,                 \
        const std::vector<sycl::event>& dependencies) {                                            \
        return potrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, \
                           ldb, group_count, group_sizes, scratchpad, scratchpad_size,             \
                           dependencies);                                                          \
    }

POTRS_BATCH_LAUNCHER_USM(float, rocsolver_spotrs_batched)
POTRS_BATCH_LAUNCHER_USM(double, rocsolver_dpotrs_batched)
POTRS_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrs_batched)
POTRS_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrs_batched)

#undef POTRS_BATCH_LAUNCHER_USM

// BATCH SCRATCHPAD API

template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_ipiv, std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t n,
                                                              std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_ipiv,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t n,
                                                               std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_ipiv,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t ldb, std::int64_t stride_b,
                                                std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::transpose trans,
                                                 std::int64_t n, std::int64_t nrhs,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_ipiv, std::int64_t ldb,
                                                 std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
#define GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                 \
    template <>                                                                    \
    std::int64_t geqrf_batch_scratchpad_size<TYPE>(                                \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t lda,     \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) { \
        return 0;                                                                  \
    }

GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH
template <>
std::int64_t potrf_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                 std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue,
                                                              oneapi::math::uplo uplo,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue,
                                                               oneapi::math::uplo uplo,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "potrf_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t ldb,
                                                std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::uplo uplo,
                                                 std::int64_t n, std::int64_t nrhs,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t ldb, std::int64_t stride_b,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b,
    std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b,
    std::int64_t batch_size) {
    throw unimplemented("lapack", "potrs_batch_scratchpad_size");
}
#define ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                             \
    template <>                                                                                \
    std::int64_t orgqr_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {             \
        return 0;                                                                              \
    }

ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(double)

#undef ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH

#define UNGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                             \
    template <>                                                                                \
    std::int64_t ungqr_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {             \
        return 0;                                                                              \
    }

UNGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
UNGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNGQR_STRIDED_BATCH_LAUNCHER_SCRATCH
template <>
std::int64_t getrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrf_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* n,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* n,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* n,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getri_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(sycl::queue& queue, oneapi::math::transpose* trans,
                                                std::int64_t* n, std::int64_t* nrhs,
                                                std::int64_t* lda, std::int64_t* ldb,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(sycl::queue& queue, oneapi::math::transpose* trans,
                                                 std::int64_t* n, std::int64_t* nrhs,
                                                 std::int64_t* lda, std::int64_t* ldb,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
    std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
    std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes) {
    throw unimplemented("lapack", "getrs_batch_scratchpad_size");
}
#define GEQRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                        \
    template <>                                                                   \
    std::int64_t geqrf_batch_scratchpad_size<TYPE>(                               \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                    \
        return 0;                                                                 \
    }

GEQRF_GROUP_LAUNCHER_SCRATCH(float)
GEQRF_GROUP_LAUNCHER_SCRATCH(double)
GEQRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GEQRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GEQRF_GROUP_LAUNCHER_SCRATCH
#define ORGQR_GROUP_LAUNCHER_SCRATCH(TYPE)                                                         \
    template <>                                                                                    \
    std::int64_t orgqr_batch_scratchpad_size<TYPE>(                                                \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                                     \
        return 0;                                                                                  \
    }

ORGQR_GROUP_LAUNCHER_SCRATCH(float)
ORGQR_GROUP_LAUNCHER_SCRATCH(double)

#undef ORGQR_GROUP_LAUNCHER_SCRATCH

#define UNGQR_GROUP_LAUNCHER_SCRATCH(TYPE)                                                         \
    template <>                                                                                    \
    std::int64_t ungqr_batch_scratchpad_size<TYPE>(                                                \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                                     \
        return 0;                                                                                  \
    }

UNGQR_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
UNGQR_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef UNGQR_GROUP_LAUNCHER_SCRATCH
// rocsolverDnXpotrfBatched does not use scratchpad memory
#define POTRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                                  \
    template <>                                                                             \
    std::int64_t potrf_batch_scratchpad_size<TYPE>(                                         \
        sycl::queue & queue, oneapi::math::uplo * uplo, std::int64_t* n, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                              \
        return 0;                                                                           \
    }

POTRF_GROUP_LAUNCHER_SCRATCH(float)
POTRF_GROUP_LAUNCHER_SCRATCH(double)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_GROUP_LAUNCHER_SCRATCH

// rocsolverDnXpotrsBatched does not use scratchpad memory
#define POTRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                   \
    template <>                                                                              \
    std::int64_t potrs_batch_scratchpad_size<TYPE>(                                          \
        sycl::queue & queue, oneapi::math::uplo * uplo, std::int64_t* n, std::int64_t* nrhs, \
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,                      \
        std::int64_t* group_sizes) {                                                         \
        return 0;                                                                            \
    }

POTRS_GROUP_LAUNCHER_SCRATCH(float)
POTRS_GROUP_LAUNCHER_SCRATCH(double)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_GROUP_LAUNCHER_SCRATCH

} // namespace rocsolver
} // namespace lapack
} // namespace math
} // namespace oneapi
