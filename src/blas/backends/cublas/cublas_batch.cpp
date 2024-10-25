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
#include "cublas_task.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {
namespace column_major {

// Buffer APIs

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1>& y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1>& y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1>& c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

template <typename Ta, typename Tb, typename Tc, typename Ts>
inline void gemm_batch_impl(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                            int64_t n, int64_t k, Ts alpha, sycl::buffer<Ta, 1>& a, int64_t lda,
                            int64_t stride_a, sycl::buffer<Tb, 1>& b, int64_t ldb, int64_t stride_b,
                            Ts beta, sycl::buffer<Tc, 1>& c, int64_t ldc, int64_t stride_c,
                            int64_t batch_size) {
    using cuTypeA = typename CudaEquivalentType<Ta>::Type;
    using cuTypeB = typename CudaEquivalentType<Tb>::Type;
    using cuTypeC = typename CudaEquivalentType<Tc>::Type;
    using cuTypeS = typename CudaEquivalentType<Ts>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);

    cublasGemmAlgo_t cublas_gemm_algo = CUBLAS_GEMM_DEFAULT;
    queue.submit([&](sycl::handler& cgh) {
        if (!verify_support<sycl::half, Ta, Tb, Tc, Ts>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuTypeA*>(a_acc);
            auto b_ = sc.get_mem<cuTypeB*>(b_acc);
            auto c_ = sc.get_mem<cuTypeC*>(c_acc);
            cublasStatus_t err;
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUBLAS_ERROR_FUNC_T("cublasGemmStridedBatchedEx", cublasGemmStridedBatchedEx, err,
                                handle, get_cublas_operation(transa), get_cublas_operation(transb),
                                m, n, k, &alpha, a_, get_cublas_datatype<cuTypeA>(), lda, stride_a,
                                b_, get_cublas_datatype<cuTypeB>(), ldb, stride_b, &beta, c_,
                                get_cublas_datatype<cuTypeC>(), ldc, stride_c, batch_size,
                                get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#else
            CUBLAS_ERROR_FUNC_T_SYNC(
                "cublasGemmStridedBatchedEx", cublasGemmStridedBatchedEx, err, handle,
                get_cublas_operation(transa), get_cublas_operation(transb), m, n, k, &alpha, a_,
                get_cublas_datatype<cuTypeA>(), lda, stride_a, b_, get_cublas_datatype<cuTypeB>(),
                ldb, stride_b, &beta, c_, get_cublas_datatype<cuTypeC>(), ldc, stride_c, batch_size,
                get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#endif
        });
    });
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                               \
    void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                    int64_t k, TYPE_S alpha, sycl::buffer<TYPE_A, 1>& a, int64_t lda,             \
                    int64_t stride_a, sycl::buffer<TYPE_B, 1>& b, int64_t ldb, int64_t stride_b,  \
                    TYPE_S beta, sycl::buffer<TYPE_C, 1>& c, int64_t ldc, int64_t stride_c,       \
                    int64_t batch_size) {                                                         \
        gemm_batch_impl<TYPE_A, TYPE_B, TYPE_C, TYPE_S>(queue, transa, transb, m, n, k, alpha, a, \
                                                        lda, stride_a, b, ldb, stride_b, beta, c, \
                                                        ldc, stride_c, batch_size);               \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, sycl::half, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(float, float, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(double, double, double, double)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, std::complex<float>, std::complex<float>,
                            std::complex<float>)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, std::complex<double>, std::complex<double>,
                            std::complex<double>)

#undef GEMM_STRIDED_BATCH_LAUNCHER

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                               \
    void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                    int64_t k, TYPE_S alpha, sycl::buffer<TYPE_A, 1>& a, int64_t lda,             \
                    int64_t stride_a, sycl::buffer<TYPE_B, 1>& b, int64_t ldb, int64_t stride_b,  \
                    TYPE_S beta, sycl::buffer<TYPE_C, 1>& c, int64_t ldc, int64_t stride_c,       \
                    int64_t batch_size) {                                                         \
        throw unimplemented("blas", "gemm_batch",                                                 \
                            std::string("for dtype unimplemented dtype combination <") +          \
                                dtype_string<TYPE_A>() + "," + dtype_string<TYPE_B>() + "," +     \
                                dtype_string<TYPE_C>() + "," + dtype_string<TYPE_S>() + ">");     \
    }

GEMM_STRIDED_BATCH_LAUNCHER(std::int8_t, std::int8_t, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(std::int8_t, std::int8_t, std::int32_t, float)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

// USM APIs

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const float** x, int64_t* incx, float** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const double** x, int64_t* incx, double** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<float>** x, int64_t* incx,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const float* x, int64_t incx,
                       std::int64_t stridex, float* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const double* x, int64_t incx,
                       std::int64_t stridex, double* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                       std::int64_t stridex, std::complex<float>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                       std::int64_t stridex, std::complex<double>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, float* alpha, const float** x, int64_t* incx,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, double* alpha, const double** x,
                       int64_t* incx, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                       int64_t stridex, float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float* a, int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float beta, float* y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double* a, int64_t lda, int64_t stride_a, const double* x,
                       int64_t incx, int64_t stride_x, double beta, double* y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, const std::complex<float>* x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, const std::complex<double>* x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event gemv_batch(const char* func_name, Func func, sycl::queue& queue,
                              transpose* trans, int64_t* m, int64_t* n, T* alpha, const T** a,
                              int64_t* lda, const T** x, int64_t* incx, T* beta, T** y,
                              int64_t* incy, int64_t group_count, int64_t* group_size,
                              const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], incx[i], incy[i], group_size[i]);
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            auto** a_ = reinterpret_cast<const cuDataType**>(a);
            auto** x_ = reinterpret_cast<const cuDataType**>(x);
            auto** y_ = reinterpret_cast<cuDataType**>(y);
            for (int64_t i = 0; i < group_count; i++) {
                cublas_native_named_func(func_name, func, err, handle,
                                         get_cublas_operation(trans[i]), (int)m[i], (int)n[i],
                                         (cuDataType*)&alpha[i], a_ + offset, (int)lda[i],
                                         x_ + offset, (int)incx[i], (cuDataType*)&beta[i],
                                         y_ + offset, (int)incy[i], (int)group_size[i]);
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define GEMV_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event gemv_batch(                                                                        \
        sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n, TYPE* alpha, const TYPE** a, \
        int64_t* lda, const TYPE** x, int64_t* incx, TYPE* beta, TYPE** y, int64_t* incy,          \
        int64_t group_count, int64_t* group_size, const std::vector<sycl::event>& dependencies) {  \
        return gemv_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, trans, m, n, alpha, a, lda, x,   \
                          incx, beta, y, incy, group_count, group_size, dependencies);             \
    }

GEMV_BATCH_LAUNCHER_USM(float, cublasSgemvBatched)
GEMV_BATCH_LAUNCHER_USM(double, cublasDgemvBatched)
GEMV_BATCH_LAUNCHER_USM(std::complex<float>, cublasCgemvBatched)
GEMV_BATCH_LAUNCHER_USM(std::complex<double>, cublasZgemvBatched)

#undef GEMV_BATCH_LAUNCHER_USM

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const float* a,
                       int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const double* a,
                       int64_t lda, int64_t stride_a, const double* x, int64_t incx,
                       int64_t stride_x, double* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       const std::complex<float>* x, int64_t incx, int64_t stride_x,
                       std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       const std::complex<double>* x, int64_t incx, int64_t stride_x,
                       std::complex<double>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const double** a, int64_t* lda, const double** x, int64_t* incx, double** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** x,
                       int64_t* incx, std::complex<float>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
}

template <typename Ta, typename Tb, typename Tc, typename Ts>
inline sycl::event gemm_batch_strided_usm_impl(sycl::queue& queue, transpose transa,
                                               transpose transb, int64_t m, int64_t n, int64_t k,
                                               Ts alpha, const Ta* a, int64_t lda, int64_t stride_a,
                                               const Tb* b, int64_t ldb, int64_t stride_b, Ts beta,
                                               Tc* c, int64_t ldc, int64_t stride_c,
                                               int64_t batch_size,
                                               const std::vector<sycl::event>& dependencies) {
    using cuTypeA = typename CudaEquivalentType<Ta>::Type;
    using cuTypeB = typename CudaEquivalentType<Tb>::Type;
    using cuTypeC = typename CudaEquivalentType<Tc>::Type;
    using cuTypeS = typename CudaEquivalentType<Ts>::Type;
    overflow_check(m, n, k, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_size);

    cublasGemmAlgo_t cublas_gemm_algo = CUBLAS_GEMM_DEFAULT;
    auto done = queue.submit([&](sycl::handler& cgh) {
        if (!verify_support<sycl::half, Ta, Tb, Tc, Ts>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            cublasStatus_t err;
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
            CUBLAS_ERROR_FUNC_T("cublasGemmStridedBatchedEx", cublasGemmStridedBatchedEx, err,
                                handle, get_cublas_operation(transa), get_cublas_operation(transb),
                                m, n, k, &alpha, a, get_cublas_datatype<cuTypeA>(), lda, stride_a,
                                b, get_cublas_datatype<cuTypeB>(), ldb, stride_b, &beta, c,
                                get_cublas_datatype<cuTypeC>(), ldc, stride_c, batch_size,
                                get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#else
            CUBLAS_ERROR_FUNC_T_SYNC(
                "cublasGemmStridedBatchedEx", cublasGemmStridedBatchedEx, err, handle,
                get_cublas_operation(transa), get_cublas_operation(transb), m, n, k, &alpha, a,
                get_cublas_datatype<cuTypeA>(), lda, stride_a, b, get_cublas_datatype<cuTypeB>(),
                ldb, stride_b, &beta, c, get_cublas_datatype<cuTypeC>(), ldc, stride_c, batch_size,
                get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#endif
        });
    });
    return done;
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                        \
    sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,  \
                           int64_t n, int64_t k, TYPE_S alpha, const TYPE_A* a, int64_t lda,   \
                           int64_t stride_a, const TYPE_B* b, int64_t ldb, int64_t stride_b,   \
                           TYPE_S beta, TYPE_C* c, int64_t ldc, int64_t stride_c,              \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) { \
        return gemm_batch_strided_usm_impl(queue, transa, transb, m, n, k, alpha, a, lda,      \
                                           stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, \
                                           batch_size, dependencies);                          \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, sycl::half, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, float, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, double, double, double)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, std::complex<float>, std::complex<float>,
                                std::complex<float>)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, std::complex<double>, std::complex<double>,
                                std::complex<double>)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                        \
    sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,  \
                           int64_t n, int64_t k, TYPE_S alpha, const TYPE_A* a, int64_t lda,   \
                           int64_t stride_a, const TYPE_B* b, int64_t ldb, int64_t stride_b,   \
                           TYPE_S beta, TYPE_C* c, int64_t ldc, int64_t stride_c,              \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) { \
        throw unimplemented("blas", "gemm_batch",                                              \
                            std::string("for dtype unimplemented dtype combination <") +       \
                                dtype_string<TYPE_A>() + "," + dtype_string<TYPE_B>() + "," +  \
                                dtype_string<TYPE_C>() + "," + dtype_string<TYPE_S>() + ">");  \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, std::int32_t, float)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

template <typename Ta, typename Tb, typename Tc, typename Ts>
inline sycl::event gemm_batch_usm_impl(sycl::queue& queue, transpose* transa, transpose* transb,
                                       int64_t* m, int64_t* n, int64_t* k, Ts* alpha, const Ta** a,
                                       int64_t* lda, const Tb** b, int64_t* ldb, Ts* beta, Tc** c,
                                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                                       const std::vector<sycl::event>& dependencies) {
    using cuTypeA = typename CudaEquivalentType<Ta>::Type;
    using cuTypeB = typename CudaEquivalentType<Tb>::Type;
    using cuTypeC = typename CudaEquivalentType<Tc>::Type;
    using cuTypeS = typename CudaEquivalentType<Ts>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], k[i], lda[i], ldb[i], ldc[i], group_size[i]);
    }

    cublasGemmAlgo_t cublas_gemm_algo = CUBLAS_GEMM_DEFAULT;
    auto done = queue.submit([&](sycl::handler& cgh) {
        if (!verify_support<sycl::half, Ta, Tb, Tc, Ts>(queue, sycl::aspect::fp16)) {
            throw oneapi::mkl::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
                CUBLAS_ERROR_FUNC_T(
                    "cublasGemmBatchedEx", cublasGemmBatchedEx, err, handle,
                    get_cublas_operation(transa[i]), get_cublas_operation(transb[i]), (int)m[i],
                    (int)n[i], (int)k[i], &alpha[i], (const void* const*)(a + offset),
                    get_cublas_datatype<cuTypeA>(), (int)lda[i], (const void* const*)(b + offset),
                    get_cublas_datatype<cuTypeB>(), (int)ldb[i], &beta[i],
                    (void* const*)(c + offset), get_cublas_datatype<cuTypeC>(), (int)ldc[i],
                    (int)group_size[i], get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#else
                CUBLAS_ERROR_FUNC_T_SYNC(
                    "cublasGemmBatchedEx", cublasGemmBatchedEx, err, handle,
                    get_cublas_operation(transa[i]), get_cublas_operation(transb[i]), (int)m[i],
                    (int)n[i], (int)k[i], &alpha[i], (const void *const *)(a + offset),
                    get_cublas_datatype<cuTypeA>(), (int)lda[i], (const void *const *)(b + offset),
                    get_cublas_datatype<cuTypeB>(), (int)ldb[i], &beta[i],
                    (void *const *)(c + offset), get_cublas_datatype<cuTypeC>(), (int)ldc[i],
                    (int)group_size[i], get_cublas_datatype<cuTypeS>(), cublas_gemm_algo);
#endif
                offset += group_size[i];
            }
        });
    });
    return done;
}

#define GEMM_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                                    \
    sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,   \
                           int64_t* n, int64_t* k, TYPE_S* alpha, const TYPE_A** a, int64_t* lda,  \
                           const TYPE_B** b, int64_t* ldb, TYPE_S* beta, TYPE_C** c, int64_t* ldc, \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        return gemm_batch_usm_impl(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, \
                                   ldc, group_count, group_size, dependencies);                    \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_BATCH_LAUNCHER_USM(sycl::half, sycl::half, float, float)
GEMM_BATCH_LAUNCHER_USM(float, float, float, float)
GEMM_BATCH_LAUNCHER_USM(double, double, double, double)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, std::complex<float>, std::complex<float>,
                        std::complex<float>)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, std::complex<double>, std::complex<double>,
                        std::complex<double>)

#undef GEMM_BATCH_LAUNCHER_USM

#define GEMM_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                                    \
    sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,   \
                           int64_t* n, int64_t* k, TYPE_S* alpha, const TYPE_A** a, int64_t* lda,  \
                           const TYPE_B** b, int64_t* ldb, TYPE_S* beta, TYPE_C** c, int64_t* ldc, \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        throw unimplemented("blas", "gemm_batch",                                                  \
                            std::string("for dtype unimplemented dtype combination <") +           \
                                dtype_string<TYPE_A>() + "," + dtype_string<TYPE_B>() + "," +      \
                                dtype_string<TYPE_C>() + "," + dtype_string<TYPE_S>() + ">");      \
    }

GEMM_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, float, float)
GEMM_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, std::int32_t, float)

#undef GEMM_BATCH_LAUNCHER_USM

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float* a,
                       int64_t lda, int64_t stride_a, float* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double* a,
                       int64_t lda, int64_t stride_a, double* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       std::complex<float>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       std::complex<double>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
}

template <typename Func, typename T>
inline sycl::event trsm_batch(const char* func_name, Func func, sycl::queue& queue,
                              side* left_right, uplo* upper_lower, transpose* trans,
                              diag* unit_diag, int64_t* m, int64_t* n, T* alpha, const T** a,
                              int64_t* lda, T** b, int64_t* ldb, int64_t group_count,
                              int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    for (int64_t i = 0; i < group_count; i++) {
        overflow_check(m[i], n[i], lda[i], ldb[i], group_size[i]);
    }
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cublas_host_task(cgh, queue, [=](CublasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            int64_t offset = 0;
            cublasStatus_t err;
            for (int64_t i = 0; i < group_count; i++) {
                auto** a_ = reinterpret_cast<const cuDataType**>(a);
                auto** b_ = reinterpret_cast<cuDataType**>(b);
                cublas_native_named_func(
                    func_name, func, err, handle, get_cublas_side_mode(left_right[i]),
                    get_cublas_fill_mode(upper_lower[i]), get_cublas_operation(trans[i]),
                    get_cublas_diag_type(unit_diag[i]), (int)m[i], (int)n[i],
                    (cuDataType*)&alpha[i], a_ + offset, (int)lda[i], b_ + offset, (int)ldb[i],
                    (int)group_size[i]);

                offset += group_size[i];
            }
        });
    });
    return done;
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,                \
                           transpose* trans, diag* unit_diag, int64_t* m, int64_t* n, TYPE* alpha, \
                           const TYPE** a, int64_t* lda, TYPE** b, int64_t* ldb,                   \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        return trsm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, left_right, upper_lower, trans,  \
                          unit_diag, m, n, alpha, a, lda, b, ldb, group_count, group_size,         \
                          dependencies);                                                           \
    }

TRSM_BATCH_LAUNCHER_USM(float, cublasStrsmBatched)
TRSM_BATCH_LAUNCHER_USM(double, cublasDtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZtrsmBatched)

#undef TRSM_BATCH_LAUNCHER_USM

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, float* alpha, const float** a, int64_t* lda, float* beta,
                       float** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, double* alpha, const double** a, int64_t* lda, double* beta,
                       double** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       int64_t* lda, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, float beta,
                       float* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double* a, int64_t lda, int64_t stride_a, double beta,
                       double* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float* a, int64_t lda, int64_t stride_a, float* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double* a, int64_t lda, int64_t stride_a, double* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                           int64_t stride_a, std::complex<float>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                           int64_t stride_a, std::complex<double>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float* a, int64_t lda, int64_t stride_a,
                          float beta, const float* b, int64_t ldb, int64_t stride_b, float* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double* a, int64_t lda, int64_t stride_a,
                          double beta, const double* b, int64_t ldb, int64_t stride_b, double* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, int64_t ldb, int64_t stride_b,
                          std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, int64_t ldb, int64_t stride_b,
                          std::complex<double>* c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, const float** a, int64_t* lda, float** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, const double** a, int64_t* lda, double** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                           std::complex<float>** b, int64_t* ldb, int64_t group_count,
                           int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, const std::complex<double>** a,
                           int64_t* lda, std::complex<double>** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, float** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, double** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, std::complex<float>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, std::complex<double>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
}

} // namespace column_major
namespace row_major {

// Buffer APIs

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1>& y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1>& y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1>& c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                               \
    void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                    int64_t k, TYPE_S alpha, sycl::buffer<TYPE_A, 1>& a, int64_t lda,             \
                    int64_t stride_a, sycl::buffer<TYPE_B, 1>& b, int64_t ldb, int64_t stride_b,  \
                    TYPE_S beta, sycl::buffer<TYPE_C, 1>& c, int64_t ldc, int64_t stride_c,       \
                    int64_t batch_size) {                                                         \
        throw unimplemented("blas", "gemm_batch", "for row_major layout");                        \
    }

GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_STRIDED_BATCH_LAUNCHER(sycl::half, sycl::half, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(std::int8_t, std::int8_t, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(std::int8_t, std::int8_t, std::int32_t, float)
GEMM_STRIDED_BATCH_LAUNCHER(float, float, float, float)
GEMM_STRIDED_BATCH_LAUNCHER(double, double, double, double)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<float>, std::complex<float>, std::complex<float>,
                            std::complex<float>)
GEMM_STRIDED_BATCH_LAUNCHER(std::complex<double>, std::complex<double>, std::complex<double>,
                            std::complex<double>)

#undef GEMM_STRIDED_BATCH_LAUNCHER

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

// USM APIs

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const float** x, int64_t* incx, float** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const double** x, int64_t* incx, double** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<float>** x, int64_t* incx,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const float* x, int64_t incx,
                       std::int64_t stridex, float* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const double* x, int64_t incx,
                       std::int64_t stridex, double* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                       std::int64_t stridex, std::complex<float>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                       std::int64_t stridex, std::complex<double>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, float* alpha, const float** x, int64_t* incx,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, double* alpha, const double** x,
                       int64_t* incx, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                       int64_t stridex, float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float* a, int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float beta, float* y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double* a, int64_t lda, int64_t stride_a, const double* x,
                       int64_t incx, int64_t stride_x, double beta, double* y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, const std::complex<float>* x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, const std::complex<double>* x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, float* alpha,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float* beta,
                       float** y, int64_t* incy, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, double* alpha,
                       const double** a, int64_t* lda, const double** x, int64_t* incx,
                       double* beta, double** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>* beta,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<double>* alpha, const std::complex<double>** a, int64_t* lda,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>* beta,
                       std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const float* a,
                       int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const double* a,
                       int64_t lda, int64_t stride_a, const double* x, int64_t incx,
                       int64_t stride_x, double* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       const std::complex<float>* x, int64_t incx, int64_t stride_x,
                       std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       const std::complex<double>* x, int64_t incx, int64_t stride_x,
                       std::complex<double>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const double** a, int64_t* lda, const double** x, int64_t* incx, double** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** x,
                       int64_t* incx, std::complex<float>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
}

#define GEMM_STRIDED_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                        \
    sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,  \
                           int64_t n, int64_t k, TYPE_S alpha, const TYPE_A* a, int64_t lda,   \
                           int64_t stride_a, const TYPE_B* b, int64_t ldb, int64_t stride_b,   \
                           TYPE_S beta, TYPE_C* c, int64_t ldc, int64_t stride_c,              \
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) { \
        throw unimplemented("blas", "gemm_batch", "for row_major layout");                     \
    }

GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_STRIDED_BATCH_LAUNCHER_USM(sycl::half, sycl::half, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, std::int32_t, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(float, float, float, float)
GEMM_STRIDED_BATCH_LAUNCHER_USM(double, double, double, double)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<float>, std::complex<float>, std::complex<float>,
                                std::complex<float>)
GEMM_STRIDED_BATCH_LAUNCHER_USM(std::complex<double>, std::complex<double>, std::complex<double>,
                                std::complex<double>)

#undef GEMM_STRIDED_BATCH_LAUNCHER_USM

#define GEMM_BATCH_LAUNCHER_USM(TYPE_A, TYPE_B, TYPE_C, TYPE_S)                                    \
    sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,   \
                           int64_t* n, int64_t* k, TYPE_S* alpha, const TYPE_A** a, int64_t* lda,  \
                           const TYPE_B** b, int64_t* ldb, TYPE_S* beta, TYPE_C** c, int64_t* ldc, \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        throw unimplemented("blas", "gemm_batch", "for row_major layout");                         \
    }

GEMM_BATCH_LAUNCHER_USM(sycl::half, sycl::half, sycl::half, sycl::half)
GEMM_BATCH_LAUNCHER_USM(sycl::half, sycl::half, float, float)
GEMM_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, float, float)
GEMM_BATCH_LAUNCHER_USM(std::int8_t, std::int8_t, std::int32_t, float)
GEMM_BATCH_LAUNCHER_USM(float, float, float, float)
GEMM_BATCH_LAUNCHER_USM(double, double, double, double)
GEMM_BATCH_LAUNCHER_USM(std::complex<float>, std::complex<float>, std::complex<float>,
                        std::complex<float>)
GEMM_BATCH_LAUNCHER_USM(std::complex<double>, std::complex<double>, std::complex<double>,
                        std::complex<double>)

#undef GEMM_BATCH_LAUNCHER_USM

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float* a,
                       int64_t lda, int64_t stride_a, float* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double* a,
                       int64_t lda, int64_t stride_a, double* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       std::complex<float>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       std::complex<double>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

template <typename Func, typename T>
inline sycl::event trsm_batch(const char* func_name, Func func, sycl::queue& queue,
                              side* left_right, uplo* upper_lower, transpose* trans,
                              diag* unit_diag, int64_t* m, int64_t* n, T* alpha, const T** a,
                              int64_t* lda, T** b, int64_t* ldb, int64_t group_count,
                              int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
}

#define TRSM_BATCH_LAUNCHER_USM(TYPE, CUBLAS_ROUTINE)                                              \
    sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,                \
                           transpose* trans, diag* unit_diag, int64_t* m, int64_t* n, TYPE* alpha, \
                           const TYPE** a, int64_t* lda, TYPE** b, int64_t* ldb,                   \
                           int64_t group_count, int64_t* group_size,                               \
                           const std::vector<sycl::event>& dependencies) {                         \
        return trsm_batch(#CUBLAS_ROUTINE, CUBLAS_ROUTINE, queue, left_right, upper_lower, trans,  \
                          unit_diag, m, n, alpha, a, lda, b, ldb, group_count, group_size,         \
                          dependencies);                                                           \
    }

TRSM_BATCH_LAUNCHER_USM(float, cublasStrsmBatched)
TRSM_BATCH_LAUNCHER_USM(double, cublasDtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<float>, cublasCtrsmBatched)
TRSM_BATCH_LAUNCHER_USM(std::complex<double>, cublasZtrsmBatched)

#undef TRSM_BATCH_LAUNCHER_USM

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, float* alpha, const float** a, int64_t* lda, float* beta,
                       float** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, double* alpha, const double** a, int64_t* lda, double* beta,
                       double** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       int64_t* lda, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, float beta,
                       float* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double* a, int64_t lda, int64_t stride_a, double beta,
                       double* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float* a, int64_t lda, int64_t stride_a, float* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double* a, int64_t lda, int64_t stride_a, double* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                           int64_t stride_a, std::complex<float>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                           int64_t stride_a, std::complex<double>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float* a, int64_t lda, int64_t stride_a,
                          float beta, const float* b, int64_t ldb, int64_t stride_b, float* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double* a, int64_t lda, int64_t stride_a,
                          double beta, const double* b, int64_t ldb, int64_t stride_b, double* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, int64_t ldb, int64_t stride_b,
                          std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, int64_t ldb, int64_t stride_b,
                          std::complex<double>* c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, const float** a, int64_t* lda, float** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, const double** a, int64_t* lda, double** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                           std::complex<float>** b, int64_t* ldb, int64_t group_count,
                           int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, const std::complex<double>** a,
                           int64_t* lda, std::complex<double>** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, float** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, double** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, std::complex<float>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, std::complex<double>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
}

} // namespace row_major
} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
