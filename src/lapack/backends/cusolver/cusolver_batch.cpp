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
#include "cublas_helper.hpp"
#include "cusolver_helper.hpp"
#include "cusolver_task.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/detail/cusolver/onemath_lapack_cusolver.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace cusolver {

// BATCH BUFFER API

template <typename Func, typename T>
inline void geqrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<T>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                        sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });
}

#define GEQRF_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                    \
    void geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
                     std::int64_t lda, std::int64_t stride_a, sycl::buffer<TYPE>& tau,          \
                     std::int64_t stride_tau, std::int64_t batch_size,                          \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {            \
        return geqrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, stride_a,  \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size);           \
    }

GEQRF_STRIDED_BATCH_LAUNCHER(float, cusolverDnSgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER(double, cusolverDnDgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getri_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t n,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, stride_ipiv, batch_size, scratchpad_size);

    std::uint64_t ipiv32_size = n * batch_size;
    sycl::buffer<int> ipiv32(sycl::range<1>{ ipiv32_size });
    sycl::buffer<int> devInfo{ batch_size };

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv_acc = sycl::accessor{ ipiv, cgh, sycl::read_only };
        auto ipiv32_acc = sycl::accessor{ ipiv32, cgh, sycl::write_only };
        cgh.parallel_for(sycl::range<1>{ ipiv32_size }, [=](sycl::id<1> index) {
            ipiv32_acc[index] = static_cast<int>(ipiv_acc[(index / n) * stride_ipiv + index % n]);
        });
    });

    // getri_batched is contained within cublas, not cusolver. For this reason
    // we need to use cublas types instead of cusolver types (as is needed for
    // other lapack routines)
    queue.submit([&](sycl::handler& cgh) {
        using blas::cublas::cublas_error;

        sycl::accessor a_acc{ a, cgh, sycl::read_only };
        sycl::accessor scratch_acc{ scratchpad, cgh, sycl::write_only };
        sycl::accessor ipiv32_acc{ ipiv32, cgh };
        sycl::accessor devInfo_acc{ devInfo, cgh, sycl::write_only };

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            cublasStatus_t err;
            CUresult cuda_result;
            cublasHandle_t cublas_handle;
            CUBLAS_ERROR_FUNC(cublasCreate, err, &cublas_handle);
            CUstream cu_stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
            CUBLAS_ERROR_FUNC(cublasSetStream, err, cublas_handle, cu_stream);

            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto info_ = sc.get_mem<int*>(devInfo_acc);

            CUdeviceptr a_dev;
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);
            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);

            CUdeviceptr scratch_dev;
            cuDataType** scratch_batched =
                create_ptr_list_from_stride(scratch_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &scratch_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, scratch_dev, scratch_batched,
                            sizeof(T*) * batch_size);
            auto** scratch_dev_ = reinterpret_cast<cuDataType**>(scratch_dev);

            blas::cublas::cublas_native_named_func(func_name, func, err, cublas_handle, n, a_dev_,
                                                   lda, ipiv32_, scratch_dev_, lda, info_,
                                                   batch_size);

            free(a_batched);
            free(scratch_batched);
            cuMemFree(a_dev);
            cuMemFree(scratch_dev);
        });
    });

    // The inverted matrices stored in scratch_ need to be stored in a_
    queue.submit([&](sycl::handler& cgh) {
        sycl::accessor a_acc{ a, cgh, sycl::write_only };
        sycl::accessor scratch_acc{ scratchpad, cgh, sycl::read_only };
        cgh.parallel_for(sycl::range<1>{ static_cast<size_t>(
                             sycl::max(stride_a * batch_size, lda * n * batch_size)) },
                         [=](sycl::id<1> index) { a_acc[index] = scratch_acc[index]; });
    });

    queue.submit([&](sycl::handler& cgh) {
        sycl::accessor ipiv32_acc{ ipiv32, cgh, sycl::read_only };
        sycl::accessor ipiv_acc{ ipiv, cgh, sycl::write_only };
        cgh.parallel_for(sycl::range<1>{ static_cast<size_t>(ipiv32_size) },
                         [=](sycl::id<1> index) {
                             ipiv_acc[(index / n) * stride_ipiv + index % n] =
                                 static_cast<int64_t>(ipiv32_acc[index]);
                         });
    });
}

#define GETRI_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                      \
    void getri_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda, \
                     std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,                     \
                     std::int64_t stride_ipiv, std::int64_t batch_size,                           \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        return getri_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, n, a, lda, stride_a, ipiv, \
                           stride_ipiv, batch_size, scratchpad, scratchpad_size);                 \
    }

GETRI_STRIDED_BATCH_LAUNCHER(float, cublasSgetriBatched)
GETRI_STRIDED_BATCH_LAUNCHER(double, cublasDgetriBatched)
GETRI_STRIDED_BATCH_LAUNCHER(std::complex<float>, cublasCgetriBatched)
GETRI_STRIDED_BATCH_LAUNCHER(std::complex<double>, cublasZgetriBatched)

#undef GETRI_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        sycl::buffer<T>& b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_ipiv, stride_b, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    sycl::buffer<int> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32_acc[index] = static_cast<std::int32_t>(ipiv_acc[index]);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::write>(cgh);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto ipiv_ = sc.get_mem<std::int32_t*>(ipiv_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            cusolverStatus_t err;

            // Does not use scratch so call cuSolver asynchronously and sync at end
            for (int64_t i = 0; i < batch_size; ++i) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_operation(trans), n,
                                      nrhs, a_ + stride_a * i, lda, ipiv_ + stride_ipiv * i,
                                      b_ + stride_b * i, ldb, nullptr);
            }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUSOLVER_SYNC(err, handle)
#endif
        });
    });
}

#define GETRS_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                      \
    void getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,           \
                     std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                  \
                     std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,                     \
                     std::int64_t stride_ipiv, sycl::buffer<TYPE>& b, std::int64_t ldb,           \
                     std::int64_t stride_b, std::int64_t batch_size,                              \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,    \
                           stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                           scratchpad_size);                                                      \
    }

GETRS_STRIDED_BATCH_LAUNCHER(float, cusolverDnSgetrs)
GETRS_STRIDED_BATCH_LAUNCHER(double, cusolverDnDgetrs)
GETRS_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCgetrs)
GETRS_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void getrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, lda, stride_a, stride_ipiv, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer with 32-bit ints then copy over results
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    sycl::buffer<int> ipiv32(sycl::range<1>{ ipiv_size });
    sycl::buffer<int> devInfo{ batch_size };

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::write>(cgh);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto ipiv_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (std::int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, a_ + stride_a * i,
                                           lda, scratch_, ipiv_ + stride_ipiv * i, devInfo_ + i);
            }
        });
    });

    // Copy from 32-bit USM to 64-bit
    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size },
                         [=](sycl::id<1> index) { ipiv_acc[index] = ipiv32_acc[index]; });
    });

    lapack_info_check(queue, devInfo, __func__, func_name, batch_size);
}

#define GETRF_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                    \
    void getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
                     std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, \
                     std::int64_t stride_ipiv, std::int64_t batch_size,                         \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {            \
        return getrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, stride_a,  \
                           ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);         \
    }

GETRF_STRIDED_BATCH_LAUNCHER(float, cusolverDnSgetrf)
GETRF_STRIDED_BATCH_LAUNCHER(double, cusolverDnDgetrf)
GETRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCgetrf)
GETRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void orgqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                        std::int64_t stride_a, sycl::buffer<T>& tau, std::int64_t stride_tau,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, k, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });
}

#define ORGQR_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                      \
    void orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     sycl::buffer<TYPE>& tau, std::int64_t stride_tau, std::int64_t batch_size,   \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        return orgqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size);             \
    }

ORGQR_STRIDED_BATCH_LAUNCHER(float, cusolverDnSorgqr)
ORGQR_STRIDED_BATCH_LAUNCHER(double, cusolverDnDorgqr)

#undef ORGQR_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<T>& a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                        sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, batch_size, scratchpad_size);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            CUdeviceptr a_dev;
            CUresult cuda_result;
            cusolverStatus_t err;

            auto a_ = sc.get_mem<cuDataType*>(a_acc);

            // Transform ptr and stride to list of ptr's
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);

            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);

            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                       (int)n, a_dev_, (int)lda, nullptr, (int)batch_size);

            free(a_batched);
            cuMemFree(a_dev);
        });
    });
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                      \
    void potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                 \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     std::int64_t batch_size, sycl::buffer<TYPE>& scratchpad,                     \
                     std::int64_t scratchpad_size) {                                              \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                           batch_size, scratchpad, scratchpad_size);                              \
    }

POTRF_STRIDED_BATCH_LAUNCHER(float, cusolverDnSpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER(double, cusolverDnDpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                        oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<T>& a, std::int64_t lda, std::int64_t stride_a,
                        sycl::buffer<T>& b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_a, stride_b, batch_size, scratchpad_size);

    // cuSolver function only supports nrhs = 1
    if (nrhs != 1)
        throw unimplemented("lapack", "potrs_batch", "cusolver potrs_batch only supports nrhs = 1");

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            CUdeviceptr a_dev, b_dev;
            cusolverStatus_t err;
            CUresult cuda_result;

            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);

            // Transform ptr and stride to list of ptr's
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            cuDataType** b_batched = create_ptr_list_from_stride(b_, stride_b, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &b_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, b_dev, b_batched, sizeof(T*) * batch_size);

            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);
            auto** b_dev_ = reinterpret_cast<cuDataType**>(b_dev);

            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                       (int)n, (int)nrhs, a_dev_, (int)lda, b_dev_, ldb, nullptr,
                                       (int)batch_size);

            free(a_batched);
            free(b_batched);
            cuMemFree(a_dev);
            cuMemFree(b_dev);
        });
    });
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRS_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                     \
    void potrs_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,                \
                     std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                 \
                     std::int64_t stride_a, sycl::buffer<TYPE>& b, std::int64_t ldb,             \
                     std::int64_t stride_b, std::int64_t batch_size,                             \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {             \
        return potrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda,    \
                           stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size); \
    }

POTRS_STRIDED_BATCH_LAUNCHER(float, cusolverDnSpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER(double, cusolverDnDpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZpotrsBatched)

#undef POTRS_STRIDED_BATCH_LAUNCHER

template <typename Func, typename T>
inline void ungqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                        std::int64_t stride_a, sycl::buffer<T>& tau, std::int64_t stride_tau,
                        std::int64_t batch_size, sycl::buffer<T>& scratchpad,
                        std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, k, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });
}

#define UNGQR_STRIDED_BATCH_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                      \
    void ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
                     sycl::buffer<TYPE>& a, std::int64_t lda, std::int64_t stride_a,              \
                     sycl::buffer<TYPE>& tau, std::int64_t stride_tau, std::int64_t batch_size,   \
                     sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        return ungqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size);             \
    }

UNGQR_STRIDED_BATCH_LAUNCHER(std::complex<float>, cusolverDnCungqr)
UNGQR_STRIDED_BATCH_LAUNCHER(std::complex<double>, cusolverDnZungqr)

#undef UNGQR_STRIDED_BATCH_LAUNCHER

// BATCH USM API

template <typename Func, typename T>
inline sycl::event geqrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, T* a, std::int64_t lda, std::int64_t stride_a,
                               T* tau, std::int64_t stride_tau, std::int64_t batch_size,
                               T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });

    return done;
}

#define GEQRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                \
    sycl::event geqrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,        \
                            std::int64_t lda, std::int64_t stride_a, TYPE* tau,                 \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad, \
                            std::int64_t scratchpad_size,                                       \
                            const std::vector<sycl::event>& dependencies) {                     \
        return geqrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, stride_a,  \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size,            \
                           dependencies);                                                       \
    }

GEQRF_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgeqrf)
GEQRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event geqrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, T** a, std::int64_t* lda, T** tau,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i)
        overflow_check(m[i], n[i], lda[i], group_sizes[i]);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType**>(a);
            auto tau_ = reinterpret_cast<cuDataType**>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            int64_t global_id = 0;
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    cusolver_native_named_func(func_name, func, err, handle, m[group_id],
                                               n[group_id], a_[global_id], lda[group_id],
                                               tau_[global_id], scratch_, scratchpad_size, nullptr);
                }
            }
        });
    });

    return done;
}

#define GEQRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event geqrf_batch(                                                                     \
        sycl::queue& queue, std::int64_t* m, std::int64_t* n, TYPE** a, std::int64_t* lda,       \
        TYPE** tau, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,       \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return geqrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, tau,        \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

GEQRF_BATCH_LAUNCHER_USM(float, cusolverDnSgeqrf)
GEQRF_BATCH_LAUNCHER_USM(double, cusolverDnDgeqrf)
GEQRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgeqrf)
GEQRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, T* a, std::int64_t lda, std::int64_t stride_a,
                               std::int64_t* ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, lda, stride_a, stride_ipiv, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratchpad_ = reinterpret_cast<cuDataType*>(scratchpad);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, a_ + stride_a * i,
                                           lda, scratchpad_, ipiv_ + stride_ipiv * i, devInfo_ + i);
            }
        });
    });

    // Copy from 32-bit USM to 64-bit
    sycl::event done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(sycl::range<1>{ ipiv_size },
                         [=](sycl::id<1> index) { ipiv[index] = ipiv32[index]; });
    });

    // Enqueue free memory, don't return event as not-neccessary for user to wait for ipiv32 being released
    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done_casting);
        cgh.host_task([=](sycl::interop_handle ih) { sycl::free(ipiv32, queue); });
    });

    // lapack_info_check calls queue.wait()
    lapack_info_check(queue, devInfo, __func__, func_name, batch_size);
    sycl::free(devInfo, queue);

    return done_casting;
}

#define GETRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                 \
    sycl::event getrf_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,         \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,         \
                            std::int64_t stride_ipiv, std::int64_t batch_size, TYPE* scratchpad, \
                            std::int64_t scratchpad_size,                                        \
                            const std::vector<sycl::event>& dependencies) {                      \
        return getrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, stride_a,   \
                           ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,           \
                           dependencies);                                                        \
    }

GETRF_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSgetrf)
GETRF_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDgetrf)
GETRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrf)
GETRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, T** a, std::int64_t* lda,
                               std::int64_t** ipiv, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i) {
        overflow_check(m[i], n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    int** ipiv32 = (int**)malloc(sizeof(int*) * batch_size);
    int64_t global_id = 0;
    for (int64_t group_id = 0; group_id < group_count; ++group_id)
        for (int64_t local_id = 0; local_id < group_sizes[group_id]; ++local_id, ++global_id)
            ipiv32[global_id] = (int*)malloc_device(sizeof(int) * n[group_id], queue);
    int* devInfo = (int*)malloc_device(sizeof(int) * batch_size, queue);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType**>(a);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            int64_t global_id = 0;
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    cusolver_native_named_func(func_name, func, err, handle, m[group_id],
                                               n[group_id], a_[global_id], lda[group_id], scratch_,
                                               ipiv32[global_id], devInfo + global_id);
                }
            }
        });
    });

    // Copy from 32-bit USM to 64-bit
    std::vector<sycl::event> casting_dependencies(group_count);
    for (int64_t group_id = 0, global_id = 0; group_id < group_count; ++group_id) {
        uint64_t ipiv_size = n[group_id];
        for (int64_t local_id = 0; local_id < group_sizes[group_id]; ++local_id, ++global_id) {
            int64_t* d_ipiv = ipiv[global_id];
            int* d_ipiv32 = ipiv32[global_id];

            sycl::event e = queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(done);
                cgh.parallel_for(sycl::range<1>{ ipiv_size },
                                 [=](sycl::id<1> index) { d_ipiv[index] = d_ipiv32[index]; });
            });
            casting_dependencies[group_id] = e;
        }
    }

    // Enqueue free memory
    sycl::event done_freeing = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(casting_dependencies);
        cgh.host_task([=](sycl::interop_handle ih) {
            for (int64_t global_id = 0; global_id < batch_size; ++global_id)
                sycl::free(ipiv32[global_id], queue);
            free(ipiv32);
        });
    });

    // lapack_info_check calls queue.wait()
    lapack_info_check(queue, devInfo, __func__, func_name, batch_size);
    sycl::free(devInfo, queue);

    return done_freeing;
}

#define GETRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event getrf_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, TYPE** a,      \
                            std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,    \
                            std::int64_t* group_sizes, TYPE* scratchpad,                         \
                            std::int64_t scratchpad_size,                                        \
                            const std::vector<sycl::event>& dependencies) {                      \
        return getrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, ipiv,       \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies); \
    }

GETRF_BATCH_LAUNCHER_USM(float, cusolverDnSgetrf)
GETRF_BATCH_LAUNCHER_USM(double, cusolverDnDgetrf)
GETRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrf)
GETRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrf)

#undef GETRS_BATCH_LAUNCHER_USM

template <typename Func, typename T>
sycl::event getri_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t n, T* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, T* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, stride_ipiv, batch_size, scratchpad_size);

    std::uint64_t ipiv32_size = n * batch_size;
    int* ipiv32 = sycl::malloc_device<int>(ipiv32_size, queue);
    int* devInfo = sycl::malloc_device<int>(batch_size, queue);

    sycl::event done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>{ static_cast<size_t>(ipiv32_size) }, [=](sycl::id<1> index) {
                ipiv32[index] = static_cast<int>(ipiv[(index / n) * stride_ipiv + index % n]);
            });
    });

    // getri_batched is contained within cublas, not cusolver. For this reason
    // we need to use cublas types instead of cusolver types (as is needed for
    // other lapack routines)
    auto done = queue.submit([&](sycl::handler& cgh) {
        using blas::cublas::cublas_error;

        cgh.depends_on(done_casting);
        cgh.depends_on(dependencies);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            cublasStatus_t err;
            CUresult cuda_result;
            cublasHandle_t cublas_handle;
            CUBLAS_ERROR_FUNC(cublasCreate, err, &cublas_handle);
            CUstream cu_stream = sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
            CUBLAS_ERROR_FUNC(cublasSetStream, err, cublas_handle, cu_stream);

            CUdeviceptr a_dev;
            auto* a_ = reinterpret_cast<cuDataType*>(a);
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);
            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);

            CUdeviceptr scratch_dev;
            auto* scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cuDataType** scratch_batched =
                create_ptr_list_from_stride(scratch_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &scratch_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, scratch_dev, scratch_batched,
                            sizeof(T*) * batch_size);
            auto** scratch_dev_ = reinterpret_cast<cuDataType**>(scratch_dev);

            blas::cublas::cublas_native_named_func(func_name, func, err, cublas_handle, n, a_dev_,
                                                   lda, ipiv32, scratch_dev_, lda, devInfo,
                                                   batch_size);

            free(a_batched);
            free(scratch_batched);
            cuMemFree(a_dev);
            cuMemFree(scratch_dev);
        });
    });

    // The inverted matrices stored in scratch_ need to be stored in a_
    auto copy1 = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(
            sycl::range<1>{ static_cast<size_t>(stride_a * (batch_size - 1) + lda * n) },
            [=](sycl::id<1> index) { a[index] = scratchpad[index]; });
    });

    auto copy2 = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(
            sycl::range<1>{ static_cast<size_t>(ipiv32_size) }, [=](sycl::id<1> index) {
                ipiv[(index / n) * stride_ipiv + index % n] = static_cast<int64_t>(ipiv32[index]);
            });
    });
    copy1.wait();
    copy2.wait();
    sycl::free(ipiv32, queue);
    sycl::free(devInfo, queue);
    return done;
}

#define GETRI_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event getri_batch(                                                                      \
        sycl::queue& queue, std::int64_t n, TYPE* a, std::int64_t lda, std::int64_t stride_a,     \
        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size, TYPE* scratchpad,  \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {             \
        return getri_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, n, a, lda, stride_a, ipiv, \
                           stride_ipiv, batch_size, scratchpad, scratchpad_size, dependencies);   \
    }

GETRI_BATCH_LAUNCHER_USM(float, cublasSgetriBatched)
GETRI_BATCH_LAUNCHER_USM(double, cublasDgetriBatched)
GETRI_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgetriBatched)
GETRI_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgetriBatched)

#undef GETRI_BATCH_LAUNCHER_USM

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

template <typename Func, typename T>
inline sycl::event getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                               T* a, std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                               std::int64_t stride_ipiv, T* b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_ipiv, stride_b, batch_size, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new memory and convert 64-bit values.
    std::uint64_t ipiv_size = stride_ipiv * batch_size;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32[index] = static_cast<std::int32_t>(ipiv[index]);
        });
    });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_casting);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            cusolverStatus_t err;

            // Does not use scratch so call cuSolver asynchronously and sync at end
            for (int64_t i = 0; i < batch_size; ++i) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_operation(trans), n,
                                      nrhs, a_ + stride_a * i, lda, ipiv_ + stride_ipiv * i,
                                      b_ + stride_b * i, ldb, nullptr);
            }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUSOLVER_SYNC(err, handle)
#endif

            sycl::free(ipiv32, queue);
        });
    });

    return done;
}

#define GETRS_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                  \
    sycl::event getrs_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,    \
                            std::int64_t nrhs, TYPE* a, std::int64_t lda, std::int64_t stride_a,  \
                            std::int64_t* ipiv, std::int64_t stride_ipiv, TYPE* b,                \
                            std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,     \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event>& dependencies) {                       \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,    \
                           stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size, scratchpad, \
                           scratchpad_size, dependencies);                                        \
    }

GETRS_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                               T** a, std::int64_t* lda, std::int64_t** ipiv, T** b,
                               std::int64_t* ldb, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i) {
        overflow_check(n[i], nrhs[i], lda[i], ldb[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    // cuSolver legacy api does not accept 64-bit ints.
    // ipiv is an array of pointers in host memory, pointing to
    // an array of 64-bit ints in device memory. Each vec of ipiv
    // values need to be converted from 64-bit to 32-bit. The list
    // must stay on host.
    int** ipiv32 = (int**)malloc(sizeof(int*) * batch_size);
    std::vector<sycl::event> casting_dependencies(batch_size);
    int64_t global_id = 0;
    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        for (int64_t local_id = 0; local_id < group_sizes[group_id]; ++local_id, ++global_id) {
            uint64_t ipiv_size = n[group_id];
            int* d_group_ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);
            ipiv32[global_id] = d_group_ipiv32;
            int64_t* d_group_ipiv = ipiv[global_id];

            auto e = queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
                    d_group_ipiv32[index] = static_cast<std::int32_t>(d_group_ipiv[index]);
                });
            });
            casting_dependencies[global_id] = e;
        }
    }

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(casting_dependencies);

        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType**>(a);
            auto b_ = reinterpret_cast<cuDataType**>(b);
            cusolverStatus_t err;
            int64_t global_id = 0;

            // Does not use scratch so call cuSolver asynchronously and sync at end
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle,
                                          get_cublas_operation(trans[group_id]), n[group_id],
                                          nrhs[group_id], a_[global_id], lda[group_id],
                                          ipiv32[global_id], b_[global_id], ldb[group_id], nullptr);
                }
            }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUSOLVER_SYNC(err, handle)
#endif

            for (int64_t i = 0; i < batch_size; ++i)
                sycl::free(ipiv32[i], queue);
            free(ipiv32);
        });
    });

    return done;
}

#define GETRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event getrs_batch(                                                                     \
        sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs, \
        TYPE** a, std::int64_t* lda, std::int64_t** ipiv, TYPE** b, std::int64_t* ldb,           \
        std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad,                   \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {            \
        return getrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda,   \
                           ipiv, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,  \
                           dependencies);                                                        \
    }

GETRS_BATCH_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_BATCH_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, std::int64_t k, T* a, std::int64_t lda,
                               std::int64_t stride_a, T* tau, std::int64_t stride_tau,
                               std::int64_t batch_size, T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, k, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });

    return done;
}

#define ORGQR_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                  \
    sycl::event orgqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,   \
                            TYPE* a, std::int64_t lda, std::int64_t stride_a, TYPE* tau,          \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad,   \
                            std::int64_t scratchpad_size,                                         \
                            const std::vector<sycl::event>& dependencies) {                       \
        return orgqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size,              \
                           dependencies);                                                         \
    }

ORGQR_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSorgqr)
ORGQR_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDorgqr)

#undef ORGQR_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgqr_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, std::int64_t* k, T** a,
                               std::int64_t* lda, T** tau, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i)
        overflow_check(m[i], n[i], k[i], lda[i], group_sizes[i]);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType**>(a);
            auto tau_ = reinterpret_cast<cuDataType**>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            int64_t global_id = 0;
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    cusolver_native_named_func(func_name, func, err, handle, m[group_id],
                                               n[group_id], k[group_id], a_[global_id],
                                               lda[group_id], tau_[global_id], scratch_,
                                               scratchpad_size, nullptr);
                }
            }
        });
    });

    return done;
}

#define ORGQR_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                           \
    sycl::event orgqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, \
                            TYPE** a, std::int64_t* lda, TYPE** tau, std::int64_t group_count,     \
                            std::int64_t* group_sizes, TYPE* scratchpad,                           \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return orgqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,       \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);   \
    }

ORGQR_BATCH_LAUNCHER_USM(float, cusolverDnSorgqr)
ORGQR_BATCH_LAUNCHER_USM(double, cusolverDnDorgqr)

#undef ORGQR_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, lda, stride_a, batch_size, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            CUdeviceptr a_dev;
            cusolverStatus_t err;
            CUresult cuda_result;

            auto* a_ = reinterpret_cast<cuDataType*>(a);

            // Transform ptr and stride to list of ptr's
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);

            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);

            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                       (int)n, a_dev_, (int)lda, nullptr, (int)batch_size);

            free(a_batched);
            cuMemFree(a_dev);
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                  \
    sycl::event potrf_batch(sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, TYPE* a, \
                            std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,     \
                            TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                            const std::vector<sycl::event>& dependencies) {                       \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, stride_a, \
                           batch_size, scratchpad, scratchpad_size, dependencies);                \
    }

POTRF_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, T** a, std::int64_t* lda,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];
    }

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            CUdeviceptr a_dev;
            CUresult cuda_result;
            cusolverStatus_t err;

            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a, sizeof(T*) * batch_size);

            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);

            // Does not use scratch so call cuSolver asynchronously and sync at end
            for (int64_t i = 0; i < group_count; i++) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo[i]),
                                      (int)n[i], a_dev_ + offset, (int)lda[i], nullptr,
                                      (int)group_sizes[i]);
                offset += group_sizes[i];
            }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUSOLVER_SYNC(err, handle)
#endif

            cuMemFree(a_dev);
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRF_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event potrf_batch(                                                                      \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, TYPE** a,                  \
        std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes, TYPE* scratchpad, \
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {             \
        return potrf_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda,           \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);  \
    }

POTRF_BATCH_LAUNCHER_USM(float, cusolverDnSpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(double, cusolverDnDpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrfBatched)
POTRF_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrfBatched)

#undef POTRF_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                               std::int64_t lda, std::int64_t stride_a, T* b, std::int64_t ldb,
                               std::int64_t stride_b, std::int64_t batch_size, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(n, nrhs, lda, ldb, stride_a, stride_b, batch_size, scratchpad_size);

    // cuSolver function only supports nrhs = 1
    if (nrhs != 1)
        throw unimplemented("lapack", "potrs_batch", "cusolver potrs_batch only supports nrhs = 1");

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            CUresult cuda_result;
            CUdeviceptr a_dev, b_dev;
            auto* a_ = reinterpret_cast<cuDataType*>(a);
            auto* b_ = reinterpret_cast<cuDataType*>(b);
            cusolverStatus_t err;

            // Transform ptr and stride to list of ptr's
            cuDataType** a_batched = create_ptr_list_from_stride(a_, stride_a, batch_size);
            cuDataType** b_batched = create_ptr_list_from_stride(b_, stride_b, batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &a_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemAlloc, cuda_result, &b_dev, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, a_dev, a_batched, sizeof(T*) * batch_size);
            CUDA_ERROR_FUNC(cuMemcpyHtoD, cuda_result, b_dev, b_batched, sizeof(T*) * batch_size);

            auto** a_dev_ = reinterpret_cast<cuDataType**>(a_dev);
            auto** b_dev_ = reinterpret_cast<cuDataType**>(b_dev);

            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                       (int)n, (int)nrhs, a_dev_, (int)lda, b_dev_, ldb, nullptr,
                                       (int)batch_size);

            free(a_batched);
            free(b_batched);
            cuMemFree(a_dev);
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRS_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                   \
    sycl::event potrs_batch(                                                                       \
        sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n, std::int64_t nrhs, TYPE* a,   \
        std::int64_t lda, std::int64_t stride_a, TYPE* b, std::int64_t ldb, std::int64_t stride_b, \
        std::int64_t batch_size, TYPE* scratchpad, std::int64_t scratchpad_size,                   \
        const std::vector<sycl::event>& dependencies) {                                            \
        return potrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda,      \
                           stride_a, b, ldb, stride_b, batch_size, scratchpad, scratchpad_size,    \
                           dependencies);                                                          \
    }

POTRS_STRIDED_BATCH_LAUNCHER_USM(float, cusolverDnSpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(double, cusolverDnDpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrsBatched)
POTRS_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrsBatched)

#undef POTRS_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrs_batch(const char* func_name, Func func, sycl::queue& queue,
                               oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs, T** a,
                               std::int64_t* lda, T** b, std::int64_t* ldb,
                               std::int64_t group_count, std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    int64_t batch_size = 0;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(n[i], lda[i], group_sizes[i]);
        batch_size += group_sizes[i];

        // cuSolver function only supports nrhs = 1
        if (nrhs[i] != 1)
            throw unimplemented("lapack", "potrs_batch",
                                "cusolver potrs_batch only supports nrhs = 1");
    }

    int* info = (int*)malloc_device(sizeof(int*) * batch_size, queue);
    T** a_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    T** b_dev = (T**)malloc_device(sizeof(T*) * batch_size, queue);
    auto done_cpy_a =
        queue.submit([&](sycl::handler& h) { h.memcpy(a_dev, a, batch_size * sizeof(T*)); });

    auto done_cpy_b =
        queue.submit([&](sycl::handler& h) { h.memcpy(b_dev, b, batch_size * sizeof(T*)); });

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.depends_on(done_cpy_a);
        cgh.depends_on(done_cpy_b);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cusolverStatus_t err;

            // Does not use scratch so call cuSolver asynchronously and sync at end
            for (int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<cuDataType**>(a_dev);
                auto** b_ = reinterpret_cast<cuDataType**>(b_dev);
                auto info_ = reinterpret_cast<int*>(info);
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, get_cublas_fill_mode(uplo[i]),
                                      (int)n[i], (int)nrhs[i], a_ + offset, (int)lda[i],
                                      b_ + offset, (int)ldb[i], info_, (int)group_sizes[i]);
                offset += group_sizes[i];
            }
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUSOLVER_SYNC(err, handle)
#endif
        });
    });
    return done;
}

// Scratchpad memory not needed as parts of buffer a is used as workspace memory
#define POTRS_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event potrs_batch(                                                                     \
        sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,       \
        TYPE** a, std::int64_t* lda, TYPE** b, std::int64_t* ldb, std::int64_t group_count,      \
        std::int64_t* group_sizes, TYPE* scratchpad, std::int64_t scratchpad_size,               \
        const std::vector<sycl::event>& dependencies) {                                          \
        return potrs_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, \
                           ldb, group_count, group_sizes, scratchpad, scratchpad_size,           \
                           dependencies);                                                        \
    }

POTRS_BATCH_LAUNCHER_USM(float, cusolverDnSpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(double, cusolverDnDpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrsBatched)
POTRS_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrsBatched)

#undef POTRS_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungqr_batch(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                               std::int64_t n, std::int64_t k, T* a, std::int64_t lda,
                               std::int64_t stride_a, T* tau, std::int64_t stride_tau,
                               std::int64_t batch_size, T* scratchpad, std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(m, n, k, lda, stride_a, stride_tau, batch_size, scratchpad_size);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t i = 0; i < batch_size; ++i) {
                cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_ + stride_a * i,
                                           lda, tau_ + stride_tau * i, scratch_, scratchpad_size,
                                           nullptr);
            }
        });
    });

    return done;
}

#define UNGQR_STRIDED_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                  \
    sycl::event ungqr_batch(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,   \
                            TYPE* a, std::int64_t lda, std::int64_t stride_a, TYPE* tau,          \
                            std::int64_t stride_tau, std::int64_t batch_size, TYPE* scratchpad,   \
                            std::int64_t scratchpad_size,                                         \
                            const std::vector<sycl::event>& dependencies) {                       \
        return ungqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, stride_a, \
                           tau, stride_tau, batch_size, scratchpad, scratchpad_size,              \
                           dependencies);                                                         \
    }

UNGQR_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCungqr)
UNGQR_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZungqr)

#undef UNGQR_STRIDED_BATCH_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungqr_batch(const char* func_name, Func func, sycl::queue& queue,
                               std::int64_t* m, std::int64_t* n, std::int64_t* k, T** a,
                               std::int64_t* lda, T** tau, std::int64_t group_count,
                               std::int64_t* group_sizes, T* scratchpad,
                               std::int64_t scratchpad_size,
                               const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;

    overflow_check(group_count, scratchpad_size);
    for (int64_t i = 0; i < group_count; ++i)
        overflow_check(m[i], n[i], k[i], lda[i], group_sizes[i]);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType**>(a);
            auto tau_ = reinterpret_cast<cuDataType**>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            int64_t global_id = 0;
            cusolverStatus_t err;

            // Uses scratch so sync between each cuSolver call
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                for (int64_t local_id = 0; local_id < group_sizes[group_id];
                     ++local_id, ++global_id) {
                    cusolver_native_named_func(func_name, func, err, handle, m[group_id],
                                               n[group_id], k[group_id], a_[global_id],
                                               lda[group_id], tau_[global_id], scratch_,
                                               scratchpad_size, nullptr);
                }
            }
        });
    });

    return done;
}

#define UNGQR_BATCH_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                           \
    sycl::event ungqr_batch(sycl::queue& queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, \
                            TYPE** a, std::int64_t* lda, TYPE** tau, std::int64_t group_count,     \
                            std::int64_t* group_sizes, TYPE* scratchpad,                           \
                            std::int64_t scratchpad_size,                                          \
                            const std::vector<sycl::event>& dependencies) {                        \
        return ungqr_batch(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau,       \
                           group_count, group_sizes, scratchpad, scratchpad_size, dependencies);   \
    }

UNGQR_BATCH_LAUNCHER_USM(std::complex<float>, cusolverDnCungqr)
UNGQR_BATCH_LAUNCHER_USM(std::complex<double>, cusolverDnZungqr)

#undef UNGQR_BATCH_LAUNCHER_USM

// BATCH SCRATCHPAD API

template <typename Func>
inline void getrf_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t m, std::int64_t n, std::int64_t lda,
                                        std::int64_t stride_a, std::int64_t stride_ipiv,
                                        std::int64_t batch_size, int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m, n, nullptr, lda, scratch_size);
        });
    });
    e.wait();
}

#define GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                       \
    template <>                                                                            \
    std::int64_t getrf_batch_scratchpad_size<TYPE>(                                        \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t lda,             \
        std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {        \
        int scratch_size;                                                                  \
        getrf_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda, \
                                    stride_a, stride_ipiv, batch_size, &scratch_size);     \
        return scratch_size;                                                               \
    }

GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(float, cusolverDnSgetrf_bufferSize)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(double, cusolverDnDgetrf_bufferSize)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgetrf_bufferSize)
GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgetrf_bufferSize)

#undef GETRF_STRIDED_BATCH_LAUNCHER_SCRATCH

// Scratch memory needs to be the same size as a
#define GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE)                                    \
    template <>                                                                       \
    std::int64_t getri_batch_scratchpad_size<TYPE>(                                   \
        sycl::queue & queue, std::int64_t n, std::int64_t lda, std::int64_t stride_a, \
        std::int64_t stride_ipiv, std::int64_t batch_size) {                          \
        assert(stride_a >= lda * n && "A matrices must not overlap");                 \
        return stride_a * (batch_size - 1) + lda * n;                                 \
    }

GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(float)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(double)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>)
GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRI_STRIDED_BATCH_LAUNCHER_SCRATCH

// cusolverDnXgetrs does not use scratchpad memory
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

template <typename Func>
inline void geqrf_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t m, std::int64_t n, std::int64_t lda,
                                        std::int64_t stride_a, std::int64_t stride_tau,
                                        std::int64_t batch_size, int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m, n, nullptr, lda, scratch_size);
        });
    });
    e.wait();
}

#define GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                       \
    template <>                                                                            \
    std::int64_t geqrf_batch_scratchpad_size<TYPE>(                                        \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t lda,             \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {         \
        int scratch_size;                                                                  \
        geqrf_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda, \
                                    stride_a, stride_tau, batch_size, &scratch_size);      \
        return scratch_size;                                                               \
    }

GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(float, cusolverDnSgeqrf_bufferSize)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(double, cusolverDnDgeqrf_bufferSize)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgeqrf_bufferSize)
GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgeqrf_bufferSize)

#undef GEQRF_STRIDED_BATCH_LAUNCHER_SCRATCH

// cusolverDnXpotrfBatched does not use scratchpad memory
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

// cusolverDnXpotrsBatched does not use scratchpad memory
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

template <typename Func>
inline void orgqr_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t m, std::int64_t n, std::int64_t k,
                                        std::int64_t lda, std::int64_t stride_a,
                                        std::int64_t stride_tau, std::int64_t batch_size,
                                        int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m, n, k, nullptr, lda, nullptr,
                                  scratch_size);
        });
    });
    e.wait();
}

#define ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                           \
    template <>                                                                                \
    std::int64_t orgqr_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {             \
        int scratch_size;                                                                      \
        orgqr_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,  \
                                    stride_a, stride_tau, batch_size, &scratch_size);          \
        return scratch_size;                                                                   \
    }

ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(float, cusolverDnSorgqr_bufferSize)
ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(double, cusolverDnDorgqr_bufferSize)

#undef ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH

template <typename Func>
inline void ungqr_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t m, std::int64_t n, std::int64_t k,
                                        std::int64_t lda, std::int64_t stride_a,
                                        std::int64_t stride_tau, std::int64_t batch_size,
                                        int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            cusolverStatus_t err;

            CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m, n, k, nullptr, lda, nullptr,
                                  scratch_size);
        });
    });
    e.wait();
}

#define ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                           \
    template <>                                                                                \
    std::int64_t ungqr_batch_scratchpad_size<TYPE>(                                            \
        sycl::queue & queue, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda, \
        std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {             \
        int scratch_size;                                                                      \
        ungqr_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,  \
                                    stride_a, stride_tau, batch_size, &scratch_size);          \
        return scratch_size;                                                                   \
    }

ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCungqr_bufferSize)
ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZungqr_bufferSize)

#undef ORGQR_STRIDED_BATCH_LAUNCHER_SCRATCH

template <typename Func>
inline void getrf_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                        std::int64_t group_count, std::int64_t* group_sizes,
                                        int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int group_scratch_size = 0;
            *scratch_size = 0;
            cusolverStatus_t err;

            // Get the maximum scratch_size across the groups
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m[group_id], n[group_id],
                                      nullptr, lda[group_id], &group_scratch_size);
                *scratch_size =
                    group_scratch_size > *scratch_size ? group_scratch_size : *scratch_size;
            }
        });
    });
    e.wait();
}

#define GETRF_GROUP_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                               \
    template <>                                                                            \
    std::int64_t getrf_batch_scratchpad_size<TYPE>(                                        \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* lda,          \
        std::int64_t group_count, std::int64_t* group_sizes) {                             \
        int scratch_size;                                                                  \
        getrf_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda, \
                                    group_count, group_sizes, &scratch_size);              \
        return scratch_size;                                                               \
    }

GETRF_GROUP_LAUNCHER_SCRATCH(float, cusolverDnSgetrf_bufferSize)
GETRF_GROUP_LAUNCHER_SCRATCH(double, cusolverDnDgetrf_bufferSize)
GETRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgetrf_bufferSize)
GETRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgetrf_bufferSize)

#undef GETRF_GROUP_LAUNCHER_SCRATCH

#define GETRI_GROUP_LAUNCHER_SCRATCH(TYPE)                                                      \
    template <>                                                                                 \
    std::int64_t getri_batch_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t* n,        \
                                                   std::int64_t* lda, std::int64_t group_count, \
                                                   std::int64_t* group_sizes) {                 \
        std::int64_t max_scratch_sz = 0;                                                        \
        for (auto group_id = 0; group_id < group_count; ++group_id) {                           \
            auto scratch_sz = lda[group_id] * n[group_id];                                      \
            if (scratch_sz > max_scratch_sz)                                                    \
                max_scratch_sz = scratch_sz;                                                    \
        }                                                                                       \
        return max_scratch_sz;                                                                  \
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

template <typename Func>
inline void geqrf_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                        std::int64_t group_count, std::int64_t* group_sizes,
                                        int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int group_scratch_size = 0;
            *scratch_size = 0;
            cusolverStatus_t err;

            // Get the maximum scratch_size across the groups
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m[group_id], n[group_id],
                                      nullptr, lda[group_id], &group_scratch_size);
                *scratch_size =
                    group_scratch_size > *scratch_size ? group_scratch_size : *scratch_size;
            }
        });
    });
    e.wait();
}

#define GEQRF_GROUP_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                               \
    template <>                                                                            \
    std::int64_t geqrf_batch_scratchpad_size<TYPE>(                                        \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* lda,          \
        std::int64_t group_count, std::int64_t* group_sizes) {                             \
        int scratch_size;                                                                  \
        geqrf_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda, \
                                    group_count, group_sizes, &scratch_size);              \
        return scratch_size;                                                               \
    }

GEQRF_GROUP_LAUNCHER_SCRATCH(float, cusolverDnSgeqrf_bufferSize)
GEQRF_GROUP_LAUNCHER_SCRATCH(double, cusolverDnDgeqrf_bufferSize)
GEQRF_GROUP_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgeqrf_bufferSize)
GEQRF_GROUP_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgeqrf_bufferSize)

#undef GEQRF_GROUP_LAUNCHER_SCRATCH

template <typename Func>
inline void orgqr_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                        std::int64_t* lda, std::int64_t group_count,
                                        std::int64_t* group_sizes, int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int group_scratch_size = 0;
            *scratch_size = 0;
            cusolverStatus_t err;

            // Get the maximum scratch_size across the groups
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m[group_id], n[group_id],
                                      k[group_id], nullptr, lda[group_id], nullptr,
                                      &group_scratch_size);
                *scratch_size =
                    group_scratch_size > *scratch_size ? group_scratch_size : *scratch_size;
            }
        });
    });
    e.wait();
}

#define ORGQR_GROUP_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                       \
    template <>                                                                                    \
    std::int64_t orgqr_batch_scratchpad_size<TYPE>(                                                \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                                     \
        int scratch_size;                                                                          \
        orgqr_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,      \
                                    group_count, group_sizes, &scratch_size);                      \
        return scratch_size;                                                                       \
    }

ORGQR_GROUP_LAUNCHER_SCRATCH(float, cusolverDnSorgqr_bufferSize)
ORGQR_GROUP_LAUNCHER_SCRATCH(double, cusolverDnDorgqr_bufferSize)

#undef ORGQR_GROUP_LAUNCHER_SCRATCH

// cusolverDnXpotrfBatched does not use scratchpad memory
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

// cusolverDnXpotrsBatched does not use scratchpad memory
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

template <typename Func>
inline void ungqr_batch_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                        std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                        std::int64_t* lda, std::int64_t group_count,
                                        std::int64_t* group_sizes, int* scratch_size) {
    auto e = queue.submit([&](sycl::handler& cgh) {
        onemath_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int group_scratch_size = 0;
            *scratch_size = 0;
            cusolverStatus_t err;

            // Get the maximum scratch_size across the groups
            for (int64_t group_id = 0; group_id < group_count; ++group_id) {
                CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, m[group_id], n[group_id],
                                      k[group_id], nullptr, lda[group_id], nullptr,
                                      &group_scratch_size);
                *scratch_size =
                    group_scratch_size > *scratch_size ? group_scratch_size : *scratch_size;
            }
        });
    });
    e.wait();
}

#define UNGQR_GROUP_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                       \
    template <>                                                                                    \
    std::int64_t ungqr_batch_scratchpad_size<TYPE>(                                                \
        sycl::queue & queue, std::int64_t* m, std::int64_t* n, std::int64_t* k, std::int64_t* lda, \
        std::int64_t group_count, std::int64_t* group_sizes) {                                     \
        int scratch_size;                                                                          \
        ungqr_batch_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,      \
                                    group_count, group_sizes, &scratch_size);                      \
        return scratch_size;                                                                       \
    }

UNGQR_GROUP_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCungqr_bufferSize)
UNGQR_GROUP_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZungqr_bufferSize)

#undef UNGQR_GROUP_LAUNCHER_SCRATCH

} // namespace cusolver
} // namespace lapack
} // namespace math
} // namespace oneapi
