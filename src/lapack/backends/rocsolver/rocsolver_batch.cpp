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

// rocsolver does not use scratchpad memory for any of the batched routines
// below: workspace is managed internally by the rocblas handle.

namespace {

// The rocsolver legacy api indexes pivots with 32-bit ints while oneMath uses
// 64-bit ones, so a strided batch of pivots has to be converted in either
// direction around the native call. Only the leading min(m, n) entries of each
// matrix in the batch are meaningful, so the padding between two consecutive
// pivot arrays is left untouched.

template <typename IpivAcc, typename Ipiv32Acc>
inline sycl::event copy_ipiv_to_32(sycl::queue& queue, IpivAcc ipiv, Ipiv32Acc ipiv32,
                                   std::int64_t ipiv_len, std::int64_t stride_ipiv,
                                   std::int64_t batch_size,
                                   const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(sycl::range<2>{ static_cast<std::size_t>(batch_size),
                                         static_cast<std::size_t>(ipiv_len) },
                         [=](sycl::id<2> index) {
                             const std::int64_t offset = index[0] * stride_ipiv + index[1];
                             ipiv32[offset] = static_cast<std::int32_t>(ipiv[offset]);
                         });
    });
}

template <typename Ipiv32Acc, typename IpivAcc>
inline sycl::event copy_ipiv_to_64(sycl::queue& queue, Ipiv32Acc ipiv32, IpivAcc ipiv,
                                   std::int64_t ipiv_len, std::int64_t stride_ipiv,
                                   std::int64_t batch_size,
                                   const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.parallel_for(sycl::range<2>{ static_cast<std::size_t>(batch_size),
                                         static_cast<std::size_t>(ipiv_len) },
                         [=](sycl::id<2> index) {
                             const std::int64_t offset = index[0] * stride_ipiv + index[1];
                             ipiv[offset] = static_cast<std::int64_t>(ipiv32[offset]);
                         });
    });
}

} // namespace

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

GEQRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgeqrf_strided_batched)

#undef GEQRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, batch_size, scratchpad_size);

    const std::int64_t ipiv_len = std::min(m, n);
    sycl::buffer<int, 1> ipiv32(
        sycl::range<1>{ static_cast<std::size_t>(stride_ipiv * batch_size) });
    sycl::buffer<int> devInfo{ sycl::range<1>{ static_cast<std::size_t>(batch_size) } };

    auto done = queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, stride_a,
                                        ipiv32_, stride_ipiv, devInfo_, batch_size);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<2>{ static_cast<std::size_t>(batch_size),
                                         static_cast<std::size_t>(ipiv_len) },
                         [=](sycl::id<2> index) {
                             const std::int64_t offset = index[0] * stride_ipiv + index[1];
                             ipiv_acc[offset] = static_cast<std::int64_t>(ipiv32_acc[offset]);
                         });
    });
    lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
}

#define GETRF_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                   \
    void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
                     std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, \
                     std::int64_t stride_ipiv, std::int64_t batch_size,                         \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {            \
        getrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, stride_a, ipiv, \
                    stride_ipiv, batch_size, scratchpad, scratchpad_size);                      \
    }

GETRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgetrf_strided_batched)

#undef GETRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getri_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, batch_size, scratchpad_size);

    sycl::buffer<int, 1> ipiv32(
        sycl::range<1>{ static_cast<std::size_t>(stride_ipiv * batch_size) });
    sycl::buffer<int> devInfo{ sycl::range<1>{ static_cast<std::size_t>(batch_size) } };

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(
            sycl::range<2>{ static_cast<std::size_t>(batch_size), static_cast<std::size_t>(n) },
            [=](sycl::id<2> index) {
                const std::int64_t offset = index[0] * stride_ipiv + index[1];
                ipiv32_acc[offset] = static_cast<std::int32_t>(ipiv_acc[offset]);
            });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, n, a_, lda, stride_a, ipiv32_,
                                        stride_ipiv, devInfo_, batch_size);
        });
    });
    lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
}

#define GETRI_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                     \
    void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, \
                     std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,                     \
                     std::int64_t stride_ipiv, std::int64_t batch_size,                           \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        getri_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, n, a, lda, stride_a, ipiv,      \
                    stride_ipiv, batch_size, scratchpad, scratchpad_size);                        \
    }

GETRI_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgetri_strided_batched)

#undef GETRI_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        sycl::buffer<T>& b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);

    sycl::buffer<int, 1> ipiv32(
        sycl::range<1>{ static_cast<std::size_t>(stride_ipiv * batch_size) });

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(
            sycl::range<2>{ static_cast<std::size_t>(batch_size), static_cast<std::size_t>(n) },
            [=](sycl::id<2> index) {
                const std::int64_t offset = index[0] * stride_ipiv + index[1];
                ipiv32_acc[offset] = static_cast<std::int32_t>(ipiv_acc[offset]);
            });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto b_ = sc.get_mem<rocmDataType*>(b_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_operation(trans),
                                        n, nrhs, a_, lda, stride_a, ipiv32_, stride_ipiv, b_, ldb,
                                        stride_b, batch_size);
        });
    });
}

#define GETRS_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                              \
    void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,    \
                     std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,           \
                     std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,              \
                     std::int64_t stride_ipiv, sycl::buffer<TYPE>& b, std::int64_t ldb,    \
                     std::int64_t stride_b, std::int64_t batch_size,                       \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {       \
        getrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,  \
                    stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                    scratchpad_size);                                                      \
    }

GETRS_STRIDED_BATCH_LAUNCHER(float, rocsolver_sgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER(double, rocsolver_dgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zgetrs_strided_batched)

#undef GETRS_STRIDED_BATCH_LAUNCHER

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

template <typename Func, typename T>
inline void potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T>& a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, batch_size, scratchpad_size);

    sycl::buffer<int> devInfo{ sycl::range<1>{ static_cast<std::size_t>(batch_size) } };

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<rocmDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_fill_mode(uplo),
                                        n, a_, lda, stride_a, devInfo_, batch_size);
        });
    });
    lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
}

#define POTRF_STRIDED_BATCH_LAUNCHER(TYPE, ROCSOLVER_ROUTINE)                                \
    void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,            \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,         \
                     std::int64_t batch_size, sycl::buffer<TYPE>& scratchpad,                \
                     std::int64_t scratchpad_size) {                                         \
        potrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                    batch_size, scratchpad, scratchpad_size);                                \
    }

POTRF_STRIDED_BATCH_LAUNCHER(float, rocsolver_spotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(double, rocsolver_dpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, rocsolver_cpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, rocsolver_zpotrf_strided_batched)

#undef POTRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<T>& b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);

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

#undef POTRS_STRIDED_BATCH_LAUNCHER

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

GEQRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_strided_batched)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_strided_batched)

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
    auto done_cpy =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
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
                rocsolver_native_named_func(func_name, func, err, handle, (int)m[i], (int)n[i],
                                            a_ + offset, (int)lda[i], tau_ + tau_offset, len,
                                            (int)group_sizes[i]);
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

    queue.wait();
    sycl::free(a_dev, queue);
    sycl::free(tau_stage, queue);
    return done_scatter;
}

#define GEQRF_GROUP_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event geqrf_batch(                                                                     \
        sycl::queue& queue, std::int64_t* m, std::int64_t* n, TYPE** a, std::int64_t* lda,       \
        TYPE** tau, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,       \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return geqrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, tau,      \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

GEQRF_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgeqrf_batched)
GEQRF_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgeqrf_batched)

#undef GEQRF_GROUP_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, T* a, std::int64_t lda, std::int64_t stride_a,
                               std::int64_t* ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(m, n, lda, batch_size, scratchpad_size);

    const std::int64_t ipiv_len = std::min(m, n);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * stride_ipiv * batch_size, queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, stride_a,
                                        ipiv32, stride_ipiv, devInfo, batch_size);
        });
    });

    auto done_casting =
        copy_ipiv_to_64(queue, ipiv32, ipiv, ipiv_len, stride_ipiv, batch_size, { done });

    try {
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(ipiv32, queue);
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(ipiv32, queue);
    sycl::free(devInfo, queue);
    return done_casting;
}

#define GETRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                \
    sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,         \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,         \
                            std::int64_t stride_ipiv, std::int64_t batch_size, TYPE* scratchpad, \
                            std::int64_t scratchpad_size,                                        \
                            const std::vector<sycl::event>& dependencies) {                      \
        return getrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, stride_a, \
                           ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,           \
                           dependencies);                                                        \
    }

GETRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrf_strided_batched)
GETRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrf_strided_batched)

#undef GETRF_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, T** a, std::int64_t* lda,
                               std::int64_t** ipiv, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (std::int64_t i = 0; i < group_count; ++i) {
        overflow_check(m[i], n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    // Pivots of a group are contiguous in the 32-bit staging buffer so that the
    // native call can address them with a fixed stride.
    std::vector<std::int64_t> ipiv_len(group_count);
    std::int64_t ipiv32_size = 0;
    for (std::int64_t i = 0; i < group_count; ++i) {
        ipiv_len[i] = std::min(m[i], n[i]);
        ipiv32_size += ipiv_len[i] * group_sizes[i];
    }

    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv32_size, queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);
    auto done_cpy =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
            std::int64_t offset = 0;
            std::int64_t ipiv_offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; ++i) {
                const std::int64_t len = std::min(m[i], n[i]);
                rocsolver_native_named_func(func_name, func, err, handle, (int)m[i], (int)n[i],
                                            a_ + offset, (int)lda[i], ipiv32 + ipiv_offset, len,
                                            devInfo + offset, (int)group_sizes[i]);
                offset += group_sizes[i];
                ipiv_offset += len * group_sizes[i];
            }
        });
    });

    std::vector<sycl::event> casting_dependencies;
    casting_dependencies.reserve(batch_size);
    for (std::int64_t i = 0, global_id = 0, ipiv_offset = 0; i < group_count; ++i) {
        const std::int64_t len = ipiv_len[i];
        for (std::int64_t j = 0; j < group_sizes[i]; ++j, ++global_id, ipiv_offset += len) {
            std::int64_t* d_ipiv = ipiv[global_id];
            const int* d_ipiv32 = ipiv32 + ipiv_offset;
            casting_dependencies.push_back(queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(done);
                cgh.parallel_for(sycl::range<1>{ static_cast<std::size_t>(len) },
                                 [=](sycl::id<1> index) {
                                     d_ipiv[index] = static_cast<std::int64_t>(d_ipiv32[index]);
                                 });
            }));
        }
    }

    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(casting_dependencies);
        cgh.host_task([]() {});
    });

    try {
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(a_dev, queue);
        sycl::free(ipiv32, queue);
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(a_dev, queue);
    sycl::free(ipiv32, queue);
    sycl::free(devInfo, queue);
    return done_casting;
}

#define GETRF_GROUP_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, TYPE** a,      \
                            std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,    \
                            std::int64_t* group_sizes, TYPE* scratchpad,                         \
                            std::int64_t scratchpad_size,                                        \
                            const std::vector<sycl::event>& dependencies) {                      \
        return getrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, m, n, a, lda, ipiv,     \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

GETRF_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgetrf_batched)
GETRF_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgetrf_batched)
GETRF_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrf_batched)
GETRF_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrf_batched)

#undef GETRF_GROUP_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getri_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t n,
                               T* a, std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, batch_size, scratchpad_size);

    int* ipiv32 = (int*)malloc_device(sizeof(int) * stride_ipiv * batch_size, queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);

    auto done_casting =
        copy_ipiv_to_32(queue, ipiv, ipiv32, n, stride_ipiv, batch_size, dependencies);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_casting);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, n, a_, lda, stride_a, ipiv32,
                                        stride_ipiv, devInfo, batch_size);
        });
    });

    try {
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(ipiv32, queue);
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(ipiv32, queue);
    sycl::free(devInfo, queue);
    return done;
}

#define GETRI_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                \
    sycl::event getri_batch(                                                                     \
        sycl::queue& queue, std::int64_t n, TYPE* a, std::int64_t lda, std::int64_t stride_a,    \
        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, TYPE* scratchpad, \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return getri_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, n, a, lda, stride_a,    \
                           ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,           \
                           dependencies);                                                        \
    }

GETRI_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetri_strided_batched)
GETRI_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetri_strided_batched)

#undef GETRI_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getri_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* n, T** a, std::int64_t* lda, std::int64_t** ipiv,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (std::int64_t i = 0; i < group_count; ++i) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    std::int64_t ipiv32_size = 0;
    for (std::int64_t i = 0; i < group_count; ++i)
        ipiv32_size += n[i] * group_sizes[i];

    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv32_size, queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);
    auto done_cpy =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    std::vector<sycl::event> casting_dependencies;
    casting_dependencies.reserve(batch_size);
    for (std::int64_t i = 0, global_id = 0, ipiv_offset = 0; i < group_count; ++i) {
        const std::int64_t len = n[i];
        for (std::int64_t j = 0; j < group_sizes[i]; ++j, ++global_id, ipiv_offset += len) {
            const std::int64_t* d_ipiv = ipiv[global_id];
            int* d_ipiv32 = ipiv32 + ipiv_offset;
            casting_dependencies.push_back(queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(dependencies);
                cgh.parallel_for(sycl::range<1>{ static_cast<std::size_t>(len) },
                                 [=](sycl::id<1> index) {
                                     d_ipiv32[index] = static_cast<std::int32_t>(d_ipiv[index]);
                                 });
            }));
        }
    }

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy);
        cgh.depends_on(casting_dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
            std::int64_t offset = 0;
            std::int64_t ipiv_offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; ++i) {
                rocsolver_native_named_func(func_name, func, err, handle, (int)n[i], a_ + offset,
                                            (int)lda[i], ipiv32 + ipiv_offset, n[i],
                                            devInfo + offset, (int)group_sizes[i]);
                offset += group_sizes[i];
                ipiv_offset += n[i] * group_sizes[i];
            }
        });
    });

    try {
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(a_dev, queue);
        sycl::free(ipiv32, queue);
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(a_dev, queue);
    sycl::free(ipiv32, queue);
    sycl::free(devInfo, queue);
    return done;
}

#define GETRI_GROUP_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event getri_batch(                                                                     \
        sycl::queue& queue, std::int64_t* n, TYPE** a, std::int64_t* lda, std::int64_t** ipiv,   \
        std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,                   \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return getri_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, n, a, lda, ipiv,        \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

GETRI_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgetri_batched)
GETRI_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgetri_batched)
GETRI_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetri_batched)
GETRI_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetri_batched)

#undef GETRI_GROUP_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                               T* a, std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                               std::int64_t stride_ipiv, T* b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);

    int* ipiv32 = (int*)malloc_device(sizeof(int) * stride_ipiv * batch_size, queue);

    auto done_casting =
        copy_ipiv_to_32(queue, ipiv, ipiv32, n, stride_ipiv, batch_size, dependencies);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_casting);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<rocmDataType*>(a);
            auto b_ = reinterpret_cast<rocmDataType*>(b);
            rocblas_status err;
            rocsolver_native_named_func(func_name, func, err, handle, get_rocblas_operation(trans),
                                        n, nrhs, a_, lda, stride_a, ipiv32, stride_ipiv, b_, ldb,
                                        stride_b, batch_size);
        });
    });

    queue.wait();
    sycl::free(ipiv32, queue);
    return done;
}

#define GETRS_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                 \
    sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,    \
                            std::int64_t nrhs, TYPE* a, std::int64_t lda, std::int64_t stride_a,  \
                            std::int64_t* ipiv, std::int64_t stride_ipiv, TYPE* b,                \
                            std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,     \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event>& dependencies) {                       \
        return getrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,  \
                           stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                           scratchpad_size, dependencies);                                        \
    }

GETRS_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_sgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrs_strided_batched)
GETRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrs_strided_batched)

#undef GETRS_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                               T** a, std::int64_t* lda, std::int64_t** ipiv, T** b,
                               std::int64_t* ldb, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;

    std::int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (std::int64_t i = 0; i < group_count; ++i) {
        overflow_check(n[i], nrhs[i], lda[i], ldb[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    std::int64_t ipiv32_size = 0;
    for (std::int64_t i = 0; i < group_count; ++i)
        ipiv32_size += n[i] * group_sizes[i];

    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    T** b_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv32_size, queue);
    auto done_cpy_a =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });
    auto done_cpy_b =
        queue.submit([&](sycl::handler& h) { h.memcpy(b_dev, b, batch_size * sizeof(T*)); });

    std::vector<sycl::event> casting_dependencies;
    casting_dependencies.reserve(batch_size);
    for (std::int64_t i = 0, global_id = 0, ipiv_offset = 0; i < group_count; ++i) {
        const std::int64_t len = n[i];
        for (std::int64_t j = 0; j < group_sizes[i]; ++j, ++global_id, ipiv_offset += len) {
            const std::int64_t* d_ipiv = ipiv[global_id];
            int* d_ipiv32 = ipiv32 + ipiv_offset;
            casting_dependencies.push_back(queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(dependencies);
                cgh.parallel_for(sycl::range<1>{ static_cast<std::size_t>(len) },
                                 [=](sycl::id<1> index) {
                                     d_ipiv32[index] = static_cast<std::int32_t>(d_ipiv[index]);
                                 });
            }));
        }
    }

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy_a);
        cgh.depends_on(done_cpy_b);
        cgh.depends_on(casting_dependencies);
        onemath_rocsolver_host_task(cgh, queue, [=](RocsolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto** a_ = reinterpret_cast<rocmDataType**>(a_dev);
            auto** b_ = reinterpret_cast<rocmDataType**>(b_dev);
            std::int64_t offset = 0;
            std::int64_t ipiv_offset = 0;
            rocblas_status err;
            for (std::int64_t i = 0; i < group_count; ++i) {
                rocsolver_native_named_func(
                    func_name, func, err, handle, get_rocblas_operation(trans[i]), (int)n[i],
                    (int)nrhs[i], a_ + offset, (int)lda[i], ipiv32 + ipiv_offset, n[i], b_ + offset,
                    (int)ldb[i], (int)group_sizes[i]);
                offset += group_sizes[i];
                ipiv_offset += n[i] * group_sizes[i];
            }
        });
    });

    queue.wait();
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(ipiv32, queue);
    return done;
}

#define GETRS_GROUP_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                  \
    sycl::event getrs_batch(                                                                     \
        sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs, \
        TYPE** a, std::int64_t* lda, std::int64_t** ipiv, TYPE** b, std::int64_t* ldb,           \
        std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,                   \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return getrs_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, \
                           ipiv, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,  \
                           dependencies);                                                        \
    }

GETRS_GROUP_BATCH_LAUNCHER_USM(float, rocsolver_sgetrs_batched)
GETRS_GROUP_BATCH_LAUNCHER_USM(double, rocsolver_dgetrs_batched)
GETRS_GROUP_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cgetrs_batched)
GETRS_GROUP_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zgetrs_batched)

#undef GETRS_GROUP_BATCH_LAUNCHER_USM

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

template <typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, lda, batch_size, scratchpad_size);

    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);

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
        lapack_info_check_batch(queue, devInfo, __func__, func_name, batch_size);
    }
    catch (...) {
        sycl::free(devInfo, queue);
        throw;
    }
    sycl::free(devInfo, queue);
    return done;
}

#define POTRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, ROCSOLVER_ROUTINE)                                 \
    sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,     \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event>& dependencies) {                       \
        return potrf_batch(#ROCSOLVER_ROUTINE, ROCSOLVER_ROUTINE, queue, uplo, n, a, lda,         \
                           stride_a, batch_size, scratchpad, scratchpad_size, dependencies);      \
    }

POTRF_STRIDED_BATCH_LAUNCHER_USM(float, rocsolver_spotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(double, rocsolver_dpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, rocsolver_cpotrf_strided_batched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, rocsolver_zpotrf_strided_batched)

#undef POTRF_STRIDED_BATCH_LAUNCHER_USM

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

    try {
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

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                               std::int64_t lda, std::int64_t stride_a, T* b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using rocmDataType = typename RocmEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, batch_size, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
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
    return done;
}

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

#undef POTRS_STRIDED_BATCH_LAUNCHER_USM

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
        overflow_check(n[i], nrhs[i], lda[i], ldb[i], group_sizes[i]);
        batch_size += group_sizes[i];
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

    queue.wait();
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
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
//
// rocsolver allocates any workspace it needs from the rocblas handle, so every
// scratchpad query below reports that no user provided memory is required.

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

#define GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                  \
    template <>                                                                     \
    std::int64_t getrf_batch_scratchpad_size<TYPE>(                                 \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t lda,      \
        std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) { \
        return 0;                                                                   \
    }

GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH

#define GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                    \
    template <>                                                                       \
    std::int64_t getri_batch_scratchpad_size<TYPE>(                                   \
        sycl::queue & queue, std::int64_t n, std::int64_t lda, std::int64_t stride_a, \
        std::int64_t stride_ipiv, std::int64_t batch_size) {                          \
        return 0;                                                                     \
    }

GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH

#define GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                             \
    template <>                                                                                \
    std::int64_t getrs_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, \
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,   \
        std::int64_t stride_b, std::int64_t batch_size) {                                      \
        return 0;                                                                              \
    }

GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_STRIDED_BATCH_LAUNCHER_SCRATCH

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

#define GETRF_GROUP_LAUNCHER_SCRATCH(TYPE)                                        \
    template <>                                                                   \
    std::int64_t getrf_batch_scratchpad_size<TYPE>(                               \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                    \
        return 0;                                                                 \
    }

GETRF_GROUP_LAUNCHER_SCRATCH(float)
GETRF_GROUP_LAUNCHER_SCRATCH(double)
GETRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GETRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRF_GROUP_LAUNCHER_SCRATCH

#define GETRI_GROUP_LAUNCHER_SCRATCH(TYPE)                                                      \
    template <>                                                                                 \
    std::int64_t getri_batch_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t* n,        \
                                                   std::int64_t* lda, std::int64_t group_count, \
                                                   std::int64_t* group_sizes) {                 \
        return 0;                                                                               \
    }

GETRI_GROUP_LAUNCHER_SCRATCH(float)
GETRI_GROUP_LAUNCHER_SCRATCH(double)
GETRI_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GETRI_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRI_GROUP_LAUNCHER_SCRATCH

#define GETRS_GROUP_LAUNCHER_SCRATCH(TYPE)                                                         \
    template <>                                                                                    \
    std::int64_t getrs_batch_scratchpad_size<TYPE>(                                                \
        sycl::queue & queue, oneapi::math::transpose * trans, std::int64_t* n, std::int64_t* nrhs, \
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,                            \
        std::int64_t* group_sizes) {                                                               \
        return 0;                                                                                  \
    }

GETRS_GROUP_LAUNCHER_SCRATCH(float)
GETRS_GROUP_LAUNCHER_SCRATCH(double)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_GROUP_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_GROUP_LAUNCHER_SCRATCH

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
