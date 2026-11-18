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

void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<float>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<double>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "geqrf_batch");
}
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
void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}
void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "orgqr_batch");
}
template <typename Info, typename Func, typename T>
inline void potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T>& a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
    overflow_check(n, lda, batch_size, scratchpad_size);
#endif

    sycl::buffer<Info> devInfo{ sycl::range<1>{ static_cast<std::size_t>(batch_size) } };

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<Info*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, stride_a, devInfo_, batch_size);
        });
    });
    lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
#define POTRF_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                    \
    void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,             \
                     std::int64_t batch_size, sycl::buffer<TYPE>& scratchpad,                    \
                     std::int64_t scratchpad_size) {                                             \
        potrf_batch<std::int64_t>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, \
                                  stride_a, batch_size, scratchpad, scratchpad_size);            \
    }

POTRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_spotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dpotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cpotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zpotrf_strided_batched_64)
#else
#define POTRF_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                     \
    void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                 \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     std::int64_t batch_size, sycl::buffer<TYPE>& scratchpad,                     \
                     std::int64_t scratchpad_size) {                                              \
        potrf_batch<int>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                         batch_size, scratchpad, scratchpad_size);                                \
    }

POTRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_spotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zpotrf_strided_batched)
#endif

#undef POTRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<T>& b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);
#endif

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto b_ = sc.get_mem<rocmDataType*>(b_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, nrhs, a_, lda, stride_a, b_, ldb, stride_b, batch_size);
        });
    });
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
#define POTRS_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                      \
    void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                  \
                     std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                   \
                     std::int64_t stride_a, sycl::buffer<TYPE>& b, std::int64_t ldb,               \
                     std::int64_t stride_b, std::int64_t batch_size,                               \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {               \
        potrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, stride_a, \
                    b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);                    \
    }

POTRS_STRIDED_BATCH_LAUNCHER(float, rocsolver_spotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER(double, rocsolver_dpotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cpotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zpotrs_strided_batched_64)
#else
#define POTRS_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                      \
    void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                  \
                     std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                   \
                     std::int64_t stride_a, sycl::buffer<TYPE>& b, std::int64_t ldb,               \
                     std::int64_t stride_b, std::int64_t batch_size,                               \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {               \
        potrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, stride_a, \
                    b, ldb, stride_b, batch_size, scratchpad, scratchpad_size);                    \
    }

POTRS_STRIDED_BATCH_LAUNCHER(float, rocsolver_spotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER(double, rocsolver_dpotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cpotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zpotrs_strided_batched)
#endif

#undef POTRS_STRIDED_BATCH_LAUNCHER
void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}
void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ungqr_batch");
}

// BATCH USM API

sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<float>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double>* tau,
                        std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, float** a,
                        std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, double** a,
                        std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
sycl::event geqrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "geqrf_batch");
}
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
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        float* a, std::int64_t lda, std::int64_t stride_a, float* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        double* a, std::int64_t lda, std::int64_t stride_a, double* tau,
                        std::int64_t stride_tau, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        float** a, std::int64_t* lda, float** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        double** a, std::int64_t* lda, double** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "orgqr_batch");
}
template <typename Info, typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
    overflow_check(n, lda, batch_size, scratchpad_size);
#endif

    Info* devInfo = static_cast<Info*>(malloc_device(sizeof(Info) * batch_size, queue));
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, stride_a, devInfo, batch_size);
        });
    });

    try {
        done.wait_and_throw();
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(devInfo, queue);
    return done;
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
#define POTRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a,  \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,      \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                        \
                            const std::vector<sycl::event>& dependencies) {                        \
        return potrf_batch<std::int64_t>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, \
                                         lda, stride_a, batch_size, scratchpad, scratchpad_size,   \
                                         dependencies);                                            \
    }

POTRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_strided_batched_64)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_strided_batched_64)
#else
#define POTRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                 \
    sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,     \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event>& dependencies) {                       \
        return potrf_batch<int>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda,    \
                                stride_a, batch_size, scratchpad, scratchpad_size, dependencies); \
    }

POTRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_strided_batched)
#endif

#undef POTRF_STRIDED_BATCH_LAUNCHER_USM

template <typename Info, typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, T** a, std::int64_t* lda,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
        overflow_check(n[i], lda[i], group_sizes[i]);
#endif
        batch_size += group_sizes[i];
    }

    Info* info = static_cast<Info*>(malloc_device(sizeof(Info) * batch_size, queue));
    T** a_dev = static_cast<T**>(malloc_device(sizeof(T*) * batch_size, queue));
    auto done_cpy =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            std::int64_t offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
#if ONEMATH_ROCSOLVER_HAS_64BIT_API
                rocsolver_native_named_func(func_name, func, err, handle,
                                            get_rocblas_fill_mode(uplo[i]), n[i], a_ + offset,
                                            lda[i], info + offset, group_sizes[i]);
#else
                rocsolver_native_named_func(
                    func_name, func, err, handle, get_rocblas_fill_mode(uplo[i]),
                    static_cast<int>(n[i]), a_ + offset, static_cast<int>(lda[i]), info + offset,
                    static_cast<int>(group_sizes[i]));
#endif
                offset += group_sizes[i];
            }
        });
    });

    try {
        done.wait_and_throw();
        lapack_info_check_batch(queue, info, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(a_dev, queue);
        sycl::free(info, queue);
        throw;
    }
    sycl::free(a_dev, queue);
    sycl::free(info, queue);
    return done;
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
#define POTRF_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                          \
    sycl::event potrf_batch(                                                                       \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, TYPE** a,                   \
        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,  \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {              \
        return potrf_batch<std::int64_t>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, \
                                         lda, group_count, group_sizes, scratchpad,                \
                                         scratchpad_size, dependencies);                           \
    }

POTRF_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_batched_64)
POTRF_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_batched_64)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_batched_64)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_batched_64)
#else
#define POTRF_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                         \
    sycl::event potrf_batch(                                                                      \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, TYPE** a,                  \
        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad, \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {             \
        return potrf_batch<int>(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda,    \
                                group_count, group_sizes, scratchpad, scratchpad_size,            \
                                dependencies);                                                    \
    }

POTRF_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_batched)
POTRF_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_batched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_batched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_batched)
#endif

#undef POTRF_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                               std::int64_t lda, std::int64_t stride_a, T* b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);
#endif

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto b_ = reinterpret_cast<rocmDataType*>(b);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, nrhs, a_, lda, stride_a, b_, ldb, stride_b, batch_size);
        });
    });
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
#define POTRS_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event potrs_batch(                                                                       \
        sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, TYPE* a,   \
        std::int64_t lda, std::int64_t stride_a, TYPE* b, std::int64_t ldb, std::int64_t stride_b, \
        std::int64_t batch_size, TYPE* scratchpad, std::int64_t scratchpad_size,                   \
        const std::vector<sycl::event>& dependencies) {                                            \
        return potrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda,    \
                           stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size,    \
                           dependencies);                                                          \
    }

POTRS_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_spotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dpotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrs_strided_batched_64)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrs_strided_batched_64)
#else
#define POTRS_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event potrs_batch(                                                                       \
        sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, TYPE* a,   \
        std::int64_t lda, std::int64_t stride_a, TYPE* b, std::int64_t ldb, std::int64_t stride_b, \
        std::int64_t batch_size, TYPE* scratchpad, std::int64_t scratchpad_size,                   \
        const std::vector<sycl::event>& dependencies) {                                            \
        return potrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda,    \
                           stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size,    \
                           dependencies);                                                          \
    }

POTRS_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_spotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dpotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrs_strided_batched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrs_strided_batched)
#endif

#undef POTRS_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs, T** a,
                               std::int64_t* lda, T** b, std::int64_t* ldb,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
#if !ONEMATH_ROCSOLVER_HAS_64BIT_API
        overflow_check(n[i], nrhs[i], lda[i], ldb[i], group_sizes[i]);
#endif
        batch_size += group_sizes[i];
    }

    T** a_dev = static_cast<T**>(malloc_device(sizeof(T*) * batch_size, queue));
    T** b_dev = static_cast<T**>(malloc_device(sizeof(T*) * batch_size, queue));
    auto done_cpy_a =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });
    auto done_cpy_b =
        queue.submit([&](sycl::handler& h) { h.memcpy(b_dev, b, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy_a);
        cgh.depends_on(done_cpy_b);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            std::int64_t offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
                auto** b_ = reinterpret_cast<rocmDataType**>(b_dev);
#if ONEMATH_ROCSOLVER_HAS_64BIT_API
                rocsolver_native_named_func(
                    func_name, func, err, handle, get_rocblas_fill_mode(uplo[i]), n[i], nrhs[i],
                    a_ + offset, lda[i], b_ + offset, ldb[i], group_sizes[i]);
#else
                rocsolver_native_named_func(
                    func_name, func, err, handle, get_rocblas_fill_mode(uplo[i]),
                    static_cast<int>(n[i]), static_cast<int>(nrhs[i]), a_ + offset,
                    static_cast<int>(lda[i]), b_ + offset, static_cast<int>(ldb[i]),
                    static_cast<int>(group_sizes[i]));
#endif
                offset += group_sizes[i];
            }
        });
    });

    try {
        done.wait_and_throw();
    }
    catch (...) {
        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        throw;
    }
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    return done;
}

#if ONEMATH_ROCSOLVER_HAS_64BIT_API
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

POTRS_BATCH_LAUNCHER_USM(float, rocsolver_spotrs_batched_64)
POTRS_BATCH_LAUNCHER_USM(double, rocsolver_dpotrs_batched_64)
POTRS_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrs_batched_64)
POTRS_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrs_batched_64)
#else
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
#endif

#undef POTRS_BATCH_LAUNCHER_USM

sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<float>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::complex<double>* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<float>** a, std::int64_t* lda, std::complex<float>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}
sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k,
                        std::complex<double>** a, std::int64_t* lda, std::complex<double>** tau,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ungqr_batch");
}

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
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                              std::int64_t n, std::int64_t lda,
                                                              std::int64_t stride_a,
                                                              std::int64_t stride_tau,
                                                              std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                               std::int64_t n, std::int64_t lda,
                                                               std::int64_t stride_a,
                                                               std::int64_t stride_tau,
                                                               std::int64_t batch_size) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}

#define POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                      \
    template <>                                                                         \
    std::int64_t potrf_batch_scratchpad_size<TYPE>(                                     \
        sycl::queue & queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t batch_size) {                               \
        return 0;                                                                       \
    }

POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRF_STRIDED_BATCH_LAUNCHER_SCRATCH

#define POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                        \
    template <>                                                                           \
    std::int64_t potrs_batch_scratchpad_size<TYPE>(                                       \
        sycl::queue & queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,  \
        std::int64_t lda, std::int64_t stride_a, std::int64_t ldb, std::int64_t stride_b, \
        std::int64_t batch_size) {                                                        \
        return 0;                                                                         \
    }

POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_STRIDED_BATCH_LAUNCHER_SCRATCH
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_tau,
                                                std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                 std::int64_t k, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_tau,
                                                 std::int64_t batch_size) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
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
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "geqrf_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(sycl::queue& queue, std::int64_t* m,
                                                std::int64_t* n, std::int64_t* k, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(sycl::queue& queue, std::int64_t* m,
                                                 std::int64_t* n, std::int64_t* k,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    throw unimplemented("lapack", "orgqr_batch_scratchpad_size");
}

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

template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* k,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* k,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    throw unimplemented("lapack", "ungqr_batch_scratchpad_size");
}

} // namespace rocsolver
} // namespace lapack
} // namespace math
} // namespace oneapi
