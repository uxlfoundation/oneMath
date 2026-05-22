/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

// Buffer APIs

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

void axpy_batch(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1>& y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1>& y, int64_t incy, int64_t stride_y, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1>& c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_sgemm_batch>(cgh, [=]() {
            const float* A0 = a_acc.GET_MULTI_PTR;
            const float* B0 = b_acc.GET_MULTI_PTR;
            float* C0 = c_acc.GET_MULTI_PTR;

            CBLAS_TRANSPOSE tA = (transa == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transa == transpose::trans) ? CblasTrans : CblasConjTrans);
            CBLAS_TRANSPOSE tB = (transb == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transb == transpose::trans) ? CblasTrans : CblasConjTrans);

#ifdef COLUMN_MAJOR
            constexpr CBLAS_ORDER order = CblasColMajor;
#endif
#ifdef ROW_MAJOR
            constexpr CBLAS_ORDER order = CblasRowMajor;
#endif

            for (int64_t i = 0; i < batch_size; ++i) {
                const float* A = A0 + i * stride_a;
                const float* B = B0 + i * stride_b;
                float* C = C0 + i * stride_c;

                ::cblas_sgemm(order, tA, tB, (blasint)m, (blasint)n, (blasint)k, alpha, A,
                              (blasint)lda, B, (blasint)ldb, beta, C, (blasint)ldc);
            }
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, double beta,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_dgemm_batch>(cgh, [=]() {
            const double* A0 = a_acc.GET_MULTI_PTR;
            const double* B0 = b_acc.GET_MULTI_PTR;
            double* C0 = c_acc.GET_MULTI_PTR;

            CBLAS_TRANSPOSE tA = (transa == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transa == transpose::trans) ? CblasTrans : CblasConjTrans);
            CBLAS_TRANSPOSE tB = (transb == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transb == transpose::trans) ? CblasTrans : CblasConjTrans);

#ifdef COLUMN_MAJOR
            constexpr CBLAS_ORDER order = CblasColMajor;
#endif
#ifdef ROW_MAJOR
            constexpr CBLAS_ORDER order = CblasRowMajor;
#endif

            for (int64_t i = 0; i < batch_size; ++i) {
                const double* A = A0 + i * stride_a;
                const double* B = B0 + i * stride_b;
                double* C = C0 + i * stride_c;

                ::cblas_dgemm(order, tA, tB, (blasint)m, (blasint)n, (blasint)k, alpha, A,
                              (blasint)lda, B, (blasint)ldb, beta, C, (blasint)ldc);
            }
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                int64_t stride_b, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_cgemm_batch>(cgh, [=]() {
            const std::complex<float>* A0 = a_acc.GET_MULTI_PTR;
            const std::complex<float>* B0 = b_acc.GET_MULTI_PTR;
            std::complex<float>* C0 = c_acc.GET_MULTI_PTR;

            CBLAS_TRANSPOSE tA = (transa == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transa == transpose::trans) ? CblasTrans : CblasConjTrans);
            CBLAS_TRANSPOSE tB = (transb == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transb == transpose::trans) ? CblasTrans : CblasConjTrans);

#ifdef COLUMN_MAJOR
            constexpr CBLAS_ORDER order = CblasColMajor;
#endif
#ifdef ROW_MAJOR
            constexpr CBLAS_ORDER order = CblasRowMajor;
#endif

            for (int64_t i = 0; i < batch_size; ++i) {
                const std::complex<float>* A = A0 + i * stride_a;
                const std::complex<float>* B = B0 + i * stride_b;
                std::complex<float>* C = C0 + i * stride_c;

                ::cblas_cgemm(order, tA, tB, (blasint)m, (blasint)n, (blasint)k,
                              static_cast<const void*>(&alpha), static_cast<const void*>(A),
                              (blasint)lda, static_cast<const void*>(B), (blasint)ldb,
                              static_cast<const void*>(&beta), static_cast<void*>(C), (blasint)ldc);
            }
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_zgemm_batch>(cgh, [=]() {
            const std::complex<double>* A0 = a_acc.GET_MULTI_PTR;
            const std::complex<double>* B0 = b_acc.GET_MULTI_PTR;
            std::complex<double>* C0 = c_acc.GET_MULTI_PTR;

            CBLAS_TRANSPOSE tA = (transa == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transa == transpose::trans) ? CblasTrans : CblasConjTrans);
            CBLAS_TRANSPOSE tB = (transb == transpose::nontrans)
                                     ? CblasNoTrans
                                     : ((transb == transpose::trans) ? CblasTrans : CblasConjTrans);

#ifdef COLUMN_MAJOR
            constexpr CBLAS_ORDER order = CblasColMajor;
#endif
#ifdef ROW_MAJOR
            constexpr CBLAS_ORDER order = CblasRowMajor;
#endif

            for (int64_t i = 0; i < batch_size; ++i) {
                const std::complex<double>* A = A0 + i * stride_a;
                const std::complex<double>* B = B0 + i * stride_b;
                std::complex<double>* C = C0 + i * stride_c;

                ::cblas_zgemm(order, tA, tB, (blasint)m, (blasint)n, (blasint)k,
                              static_cast<const void*>(&alpha), static_cast<const void*>(A),
                              (blasint)lda, static_cast<const void*>(B), (blasint)ldb,
                              static_cast<const void*>(&beta), static_cast<void*>(C), (blasint)ldc);
            }
        });
    });
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<std::int32_t, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

// USM APIs

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const float** x, int64_t* incx, float** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const double** x, int64_t* incx, double** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<float>** x, int64_t* incx,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const float* x, int64_t incx,
                       std::int64_t stridex, float* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const double* x, int64_t incx,
                       std::int64_t stridex, double* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                       std::int64_t stridex, std::complex<float>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                       std::int64_t stridex, std::complex<double>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "copy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "copy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, float* alpha, const float** x, int64_t* incx,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, double* alpha, const double** x,
                       int64_t* incx, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                       int64_t stridex, float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpy_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float* a, int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float beta, float* y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double* a, int64_t lda, int64_t stride_a, const double* x,
                       int64_t incx, int64_t stride_x, double beta, double* y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, const std::complex<float>* x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, const std::complex<double>* x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, float* alpha,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float* beta,
                       float** y, int64_t* incy, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, double* alpha,
                       const double** a, int64_t* lda, const double** x, int64_t* incx,
                       double* beta, double** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>* beta,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<double>* alpha, const std::complex<double>** a, int64_t* lda,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>* beta,
                       std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemv_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemv_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const float* a,
                       int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const double* a,
                       int64_t lda, int64_t stride_a, const double* x, int64_t incx,
                       int64_t stride_x, double* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       const std::complex<float>* x, int64_t incx, int64_t stride_x,
                       std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       const std::complex<double>* x, int64_t incx, int64_t stride_x,
                       std::complex<double>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const double** a, int64_t* lda, const double** x, int64_t* incx, double** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** x,
                       int64_t* incx, std::complex<float>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "dgmm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const float** a, int64_t* lda,
                       const float** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, double* alpha, const double** a, int64_t* lda,
                       const double** b, int64_t* ldb, double* beta, double** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** b,
                       int64_t* ldb, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** b,
                       int64_t* ldb, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, sycl::half* alpha, const sycl::half** a,
                       int64_t* lda, const sycl::half** b, int64_t* ldb, sycl::half* beta,
                       sycl::half** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const sycl::half** a, int64_t* lda,
                       const sycl::half** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, std::int32_t** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const float* a, int64_t lda, int64_t stride_a,
                       const float* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, double alpha, const double* a, int64_t lda, int64_t stride_a,
                       const double* b, int64_t ldb, int64_t stride_b, double beta, double* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                       int64_t lda, int64_t stride_a, const std::complex<float>* b, int64_t ldb,
                       int64_t stride_b, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                       int64_t lda, int64_t stride_a, const std::complex<double>* b, int64_t ldb,
                       int64_t stride_b, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda,
                       int64_t stride_a, const sycl::half* b, int64_t ldb, int64_t stride_b,
                       sycl::half beta, sycl::half* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const sycl::half* a, int64_t lda, int64_t stride_a,
                       const sycl::half* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta,
                       std::int32_t* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "gemm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "gemm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float* a,
                       int64_t lda, int64_t stride_a, float* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double* a,
                       int64_t lda, int64_t stride_a, double* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       std::complex<float>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       std::complex<double>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, float* alpha, const float** a,
                       int64_t* lda, float** b, int64_t* ldb, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, double* alpha, const double** a,
                       int64_t* lda, double** b, int64_t* ldb, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, std::complex<float>** b,
                       int64_t* ldb, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, std::complex<double>** b,
                       int64_t* ldb, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "trsm_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "trsm_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, float* alpha, const float** a, int64_t* lda, float* beta,
                       float** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, double* alpha, const double** a, int64_t* lda, double* beta,
                       double** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       int64_t* lda, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, float beta,
                       float* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double* a, int64_t lda, int64_t stride_a, double beta,
                       double* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "syrk_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "syrk_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float* a, int64_t lda, int64_t stride_a, float* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double* a, int64_t lda, int64_t stride_a, double* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                           int64_t stride_a, std::complex<float>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                           int64_t stride_a, std::complex<double>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float* a, int64_t lda, int64_t stride_a,
                          float beta, const float* b, int64_t ldb, int64_t stride_b, float* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double* a, int64_t lda, int64_t stride_a,
                          double beta, const double* b, int64_t ldb, int64_t stride_b, double* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, int64_t ldb, int64_t stride_b,
                          std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, int64_t ldb, int64_t stride_b,
                          std::complex<double>* c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatadd_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, const float** a, int64_t* lda, float** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, const double** a, int64_t* lda, double** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                           std::complex<float>** b, int64_t* ldb, int64_t group_count,
                           int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, const std::complex<double>** a,
                           int64_t* lda, std::complex<double>** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "omatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, float** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, double** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, std::complex<float>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, std::complex<double>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "imatcopy_batch", "for row_major layout");
#endif
}
