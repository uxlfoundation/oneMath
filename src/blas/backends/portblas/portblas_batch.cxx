/*******************************************************************************
* Copyright Codeplay Software
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

void syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                std::int64_t lda, std::int64_t stride_a, float beta, sycl::buffer<float, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "");
}

void syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                std::int64_t lda, std::int64_t stride_a, double beta, sycl::buffer<double, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "");
}

void syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "");
}

void syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "syrk_batch", "");
}

void gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                float alpha, sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex, float beta,
                sycl::buffer<float, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "");
}

void gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                double alpha, sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex, double beta,
                sycl::buffer<double, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "");
}

void gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                std::int64_t lda, std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& x,
                std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "");
}

void gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                std::int64_t lda, std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& x,
                std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemv_batch", "");
}

void dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "");
}

void dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "");
}

void dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "");
}

void dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m, std::int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stridea,
                sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc, std::int64_t stridec,
                std::int64_t batch_size) {
    throw unimplemented("blas", "dgmm_batch", "");
}

void axpy_batch(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<float, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_axpy_batch, queue, n, alpha, x, incx, stridex, y, incy, stridey,
                     batch_size);
}

void axpy_batch(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<double, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_axpy_batch, queue, n, alpha, x, incx, stridex, y, incy, stridey,
                     batch_size);
}

void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "");
}

void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    throw unimplemented("blas", "axpy_batch", "");
}
void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<float, 1>& y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "");
}

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
                std::int64_t stridex, sycl::buffer<double, 1>& y, std::int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "");
}

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "");
}

void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                std::int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    throw unimplemented("blas", "copy_batch", "");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_gemm_strided_batched, queue, transa, transb, m, n, k, alpha, a, lda,
                     stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t stride_b, double beta,
                sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_gemm_strided_batched, queue, transa, transb, m, n, k, alpha, a, lda,
                     stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for complex");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for complex");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                sycl::buffer<sycl::half, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1>& c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for complex");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                sycl::buffer<sycl::half, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, std::int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for unsupported dtype");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                sycl::buffer<std::int8_t, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for unsupported dtype");
}

void gemm_batch(sycl::queue& queue, oneapi::math::transpose transa, oneapi::math::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                sycl::buffer<std::int8_t, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::int8_t, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                float beta, sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch", " for unsupported dtype");
}

void trsm_batch(sycl::queue& queue, oneapi::math::side left_right, oneapi::math::uplo upper_lower,
                oneapi::math::transpose trans, oneapi::math::diag unit_diag, std::int64_t m,
                std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                std::int64_t stride_a, sycl::buffer<float, 1>& b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "");
}

void trsm_batch(sycl::queue& queue, oneapi::math::side left_right, oneapi::math::uplo upper_lower,
                oneapi::math::transpose trans, oneapi::math::diag unit_diag, std::int64_t m,
                std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                std::int64_t stride_a, sycl::buffer<double, 1>& b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "");
}

void trsm_batch(sycl::queue& queue, oneapi::math::side left_right, oneapi::math::uplo upper_lower,
                oneapi::math::transpose trans, oneapi::math::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "");
}

void trsm_batch(sycl::queue& queue, oneapi::math::side left_right, oneapi::math::uplo upper_lower,
                oneapi::math::transpose trans, oneapi::math::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda, std::int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    throw unimplemented("blas", "trsm_batch", "");
}

void omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<float, 1>& b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_omatcopy_batch, queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                     stride_b, batch_size);
}

void omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<double, 1>& b, std::int64_t ldb,
                    std::int64_t stride_b, std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_omatcopy_batch, queue, trans, m, n, alpha, a, lda, stride_a, b, ldb,
                     stride_b, batch_size);
}

void omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<float> alpha,
                    sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b,
                    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "");
}

void omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<double> alpha,
                    sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                    std::int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    throw unimplemented("blas", "omatcopy_batch", "");
}

void imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, float alpha, sycl::buffer<float, 1>& ab, std::int64_t lda,
                    std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "");
}

void imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, double alpha, sycl::buffer<double, 1>& ab, std::int64_t lda,
                    std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "");
}

void imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<float> alpha,
                    sycl::buffer<std::complex<float>, 1>& ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "");
}

void imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                    std::int64_t n, std::complex<double> alpha,
                    sycl::buffer<std::complex<double>, 1>& ab, std::int64_t lda, std::int64_t ldb,
                    std::int64_t stride, std::int64_t batch_size) {
    throw unimplemented("blas", "imatcopy_batch", "");
}

void omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                   oneapi::math::transpose transb, std::int64_t m, std::int64_t n, float alpha,
                   sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a, float beta,
                   sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                   sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                   std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_omatadd_batch, queue, transa, transb, m, n, alpha, a, lda, stride_a,
                     beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

void omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                   oneapi::math::transpose transb, std::int64_t m, std::int64_t n, double alpha,
                   sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stride_a, double beta,
                   sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                   sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                   std::int64_t batch_size) {
    CALL_PORTBLAS_FN(::blas::_omatadd_batch, queue, transa, transb, m, n, alpha, a, lda, stride_a,
                     beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

void omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                   oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                   std::int64_t lda, std::int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                   std::int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "");
}

void omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                   oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                   std::int64_t stride_b, sycl::buffer<std::complex<double>, 1>& c,
                   std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    throw unimplemented("blas", "omatadd_batch", "");
}

// USM APIs

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo* upper_lower,
                       oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* k,
                       float* alpha, const float** a, std::int64_t* lda, float* beta, float** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo* upper_lower,
                       oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* k,
                       double* alpha, const double** a, std::int64_t* lda, double* beta, double** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo* upper_lower,
                       oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* k,
                       std::complex<float>* alpha, const std::complex<float>** a, std::int64_t* lda,
                       std::complex<float>* beta, std::complex<float>** c, std::int64_t* ldc,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo* upper_lower,
                       oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* k,
                       std::complex<double>* alpha, const std::complex<double>** a,
                       std::int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                       const float* a, std::int64_t lda, std::int64_t stride_a, float beta,
                       float* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                       const double* a, std::int64_t lda, std::int64_t stride_a, double beta,
                       double* c, std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans, std::int64_t n, std::int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
                       std::int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event syrk_batch(sycl::queue& queue, oneapi::math::uplo upper_lower,
                       oneapi::math::transpose trans, std::int64_t n, std::int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
                       std::int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "syrk_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                       std::int64_t n, float alpha, const float* a, std::int64_t lda,
                       std::int64_t stridea, const float* x, std::int64_t incx,
                       std::int64_t stridex, float beta, float* y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                       std::int64_t n, double alpha, const double* a, std::int64_t lda,
                       std::int64_t stridea, const double* x, std::int64_t incx,
                       std::int64_t stridex, double beta, double* y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                       std::int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                       std::int64_t lda, std::int64_t stridea, const std::complex<float>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
                       std::complex<float>* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                       std::int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                       std::int64_t lda, std::int64_t stridea, const std::complex<double>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<double> beta,
                       std::complex<double>* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* m,
                       std::int64_t* n, float* alpha, const float** a, std::int64_t* lda,
                       const float** x, std::int64_t* incx, float* beta, float** y,
                       std::int64_t* incy, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* m,
                       std::int64_t* n, double* alpha, const double** a, std::int64_t* lda,
                       const double** x, std::int64_t* incx, double* beta, double** y,
                       std::int64_t* incy, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* m,
                       std::int64_t* n, std::complex<float>* alpha, const std::complex<float>** a,
                       std::int64_t* lda, const std::complex<float>** x, std::int64_t* incx,
                       std::complex<float>* beta, std::complex<float>** y, std::int64_t* incy,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event gemv_batch(sycl::queue& queue, oneapi::math::transpose* trans, std::int64_t* m,
                       std::int64_t* n, std::complex<double>* alpha, const std::complex<double>** a,
                       std::int64_t* lda, const std::complex<double>** x, std::int64_t* incx,
                       std::complex<double>* beta, std::complex<double>** y, std::int64_t* incy,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m,
                       std::int64_t n, const float* a, std::int64_t lda, std::int64_t stridea,
                       const float* x, std::int64_t incx, std::int64_t stridex, float* c,
                       std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m,
                       std::int64_t n, const double* a, std::int64_t lda, std::int64_t stridea,
                       const double* x, std::int64_t incx, std::int64_t stridex, double* c,
                       std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m,
                       std::int64_t n, const std::complex<float>* a, std::int64_t lda,
                       std::int64_t stridea, const std::complex<float>* x, std::int64_t incx,
                       std::int64_t stridex, std::complex<float>* c, std::int64_t ldc,
                       std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side left_right, std::int64_t m,
                       std::int64_t n, const std::complex<double>* a, std::int64_t lda,
                       std::int64_t stridea, const std::complex<double>* x, std::int64_t incx,
                       std::int64_t stridex, std::complex<double>* c, std::int64_t ldc,
                       std::int64_t stridec, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side* left_right, std::int64_t* m,
                       std::int64_t* n, const float** a, std::int64_t* lda, const float** x,
                       std::int64_t* incx, float** c, std::int64_t* ldc, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side* left_right, std::int64_t* m,
                       std::int64_t* n, const double** a, std::int64_t* lda, const double** x,
                       std::int64_t* incx, double** c, std::int64_t* ldc, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side* left_right, std::int64_t* m,
                       std::int64_t* n, const std::complex<float>** a, std::int64_t* lda,
                       const std::complex<float>** x, std::int64_t* incx, std::complex<float>** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event dgmm_batch(sycl::queue& queue, oneapi::math::side* left_right, std::int64_t* m,
                       std::int64_t* n, const std::complex<double>** a, std::int64_t* lda,
                       const std::complex<double>** x, std::int64_t* incx, std::complex<double>** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dgmm_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, float* alpha, const float** x,
                       std::int64_t* incx, float** y, std::int64_t* incy, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, double* alpha, const double** x,
                       std::int64_t* incx, double** y, std::int64_t* incy, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, std::int64_t* incx, std::complex<float>** y,
                       std::int64_t* incy, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, std::int64_t* incx, std::complex<double>** y,
                       std::int64_t* incy, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, float alpha, const float* x,
                       std::int64_t incx, std::int64_t stridex, float* y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_axpy_batch, queue, n, alpha, x, incx, stridex, y, incy, stridey,
                         batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                       std::int64_t incx, std::int64_t stridex, double* y, std::int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_axpy_batch, queue, n, alpha, x, incx, stridex, y, incy, stridey,
                         batch_size, dependencies);
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, std::int64_t incx, std::int64_t stridex,
                       std::complex<float>* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, std::int64_t incx, std::int64_t stridex,
                       std::complex<double>* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const float** x, std::int64_t* incx,
                       float** y, std::int64_t* incy, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const double** x, std::int64_t* incx,
                       double** y, std::int64_t* incy, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const std::complex<float>** x,
                       std::int64_t* incx, std::complex<float>** y, std::int64_t* incy,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const std::complex<double>** x,
                       std::int64_t* incx, std::complex<double>** y, std::int64_t* incy,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                       std::int64_t stridex, float* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                       std::int64_t stridex, double* y, std::int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<float>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                       std::int64_t incx, std::int64_t stridex, std::complex<double>* y,
                       std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, float* alpha, const float** a, std::int64_t* lda,
                       const float** b, std::int64_t* ldb, float* beta, float** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, double* alpha, const double** a, std::int64_t* lda,
                       const double** b, std::int64_t* ldb, double* beta, double** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       std::int64_t* lda, const std::complex<float>** b, std::int64_t* ldb,
                       std::complex<float>* beta, std::complex<float>** c, std::int64_t* ldc,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       std::int64_t* lda, const std::complex<double>** b, std::int64_t* ldb,
                       std::complex<double>* beta, std::complex<double>** c, std::int64_t* ldc,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, sycl::half* alpha, const sycl::half** a, std::int64_t* lda,
                       const sycl::half** b, std::int64_t* ldb, sycl::half* beta, sycl::half** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, float* alpha, const sycl::half** a, std::int64_t* lda,
                       const sycl::half** b, std::int64_t* ldb, float* beta, float** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, float* alpha, const std::int8_t** a, std::int64_t* lda,
                       const std::int8_t** b, std::int64_t* ldb, float* beta, float** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose* transa,
                       oneapi::math::transpose* transb, std::int64_t* m, std::int64_t* n,
                       std::int64_t* k, float* alpha, const std::int8_t** a, std::int64_t* lda,
                       const std::int8_t** b, std::int64_t* ldb, float* beta, std::int32_t** c,
                       std::int64_t* ldc, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, float alpha, const float* a, std::int64_t lda,
                       std::int64_t stride_a, const float* b, std::int64_t ldb,
                       std::int64_t stride_b, float beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_gemm_strided_batched, queue, transa, transb, m, n, k, alpha, a,
                         lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
                         dependencies);
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, double alpha, const double* a, std::int64_t lda,
                       std::int64_t stride_a, const double* b, std::int64_t ldb,
                       std::int64_t stride_b, double beta, double* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_gemm_strided_batched, queue, transa, transb, m, n, k, alpha, a,
                         lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
                         dependencies);
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                       std::int64_t lda, std::int64_t stride_a, const std::complex<float>* b,
                       std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                       std::complex<float>* c, std::int64_t ldc, std::int64_t stride_c,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                       std::int64_t lda, std::int64_t stride_a, const std::complex<double>* b,
                       std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                       std::complex<double>* c, std::int64_t ldc, std::int64_t stride_c,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, sycl::half alpha, const sycl::half* a, std::int64_t lda,
                       std::int64_t stride_a, const sycl::half* b, std::int64_t ldb,
                       std::int64_t stride_b, sycl::half beta, sycl::half* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, float alpha, const sycl::half* a, std::int64_t lda,
                       std::int64_t stride_a, const sycl::half* b, std::int64_t ldb,
                       std::int64_t stride_b, float beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, float alpha, const std::int8_t* a, std::int64_t lda,
                       std::int64_t stride_a, const std::int8_t* b, std::int64_t ldb,
                       std::int64_t stride_b, float beta, float* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event gemm_batch(sycl::queue& queue, oneapi::math::transpose transa,
                       oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                       std::int64_t k, float alpha, const std::int8_t* a, std::int64_t lda,
                       std::int64_t stride_a, const std::int8_t* b, std::int64_t ldb,
                       std::int64_t stride_b, float beta, std::int32_t* c, std::int64_t ldc,
                       std::int64_t stride_c, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                       const float* a, std::int64_t lda, std::int64_t stride_a, float* b,
                       std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                       const double* a, std::int64_t lda, std::int64_t stride_a, double* b,
                       std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag, std::int64_t m, std::int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
                       std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                       std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side left_right,
                       oneapi::math::uplo upper_lower, oneapi::math::transpose trans,
                       oneapi::math::diag unit_diag, std::int64_t m, std::int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
                       std::int64_t stride_a, std::complex<double>* b, std::int64_t ldb,
                       std::int64_t stride_b, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side* left_right,
                       oneapi::math::uplo* upper_lower, oneapi::math::transpose* trans,
                       oneapi::math::diag* unit_diag, std::int64_t* m, std::int64_t* n,
                       float* alpha, const float** a, std::int64_t* lda, float** b,
                       std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side* left_right,
                       oneapi::math::uplo* upper_lower, oneapi::math::transpose* trans,
                       oneapi::math::diag* unit_diag, std::int64_t* m, std::int64_t* n,
                       double* alpha, const double** a, std::int64_t* lda, double** b,
                       std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side* left_right,
                       oneapi::math::uplo* upper_lower, oneapi::math::transpose* trans,
                       oneapi::math::diag* unit_diag, std::int64_t* m, std::int64_t* n,
                       std::complex<float>* alpha, const std::complex<float>** a, std::int64_t* lda,
                       std::complex<float>** b, std::int64_t* ldb, std::int64_t group_count,
                       std::int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event trsm_batch(sycl::queue& queue, oneapi::math::side* left_right,
                       oneapi::math::uplo* upper_lower, oneapi::math::transpose* trans,
                       oneapi::math::diag* unit_diag, std::int64_t* m, std::int64_t* n,
                       std::complex<double>* alpha, const std::complex<double>** a,
                       std::int64_t* lda, std::complex<double>** b, std::int64_t* ldb,
                       std::int64_t group_count, std::int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsm_batch", " for USM");
}

sycl::event omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, float alpha, const float* a, std::int64_t lda,
                           std::int64_t stride_a, float* b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_omatcopy_batch, queue, trans, m, n, alpha, a, lda, stride_a, b,
                         ldb, stride_b, batch_size, dependencies);
}

sycl::event omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, double alpha, const double* a, std::int64_t lda,
                           std::int64_t stride_a, double* b, std::int64_t ldb,
                           std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_omatcopy_batch, queue, trans, m, n, alpha, a, lda, stride_a, b,
                         ldb, stride_b, batch_size, dependencies);
}

sycl::event omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                           std::int64_t lda, std::int64_t stride_a, std::complex<float>* b,
                           std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", " for USM");
}

sycl::event omatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, std::complex<double> alpha,
                           const std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                           std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                           std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatcopy_batch", " for USM");
}

sycl::event imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, float alpha, float* ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", " for USM");
}

sycl::event imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, double alpha, double* ab, std::int64_t lda,
                           std::int64_t ldb, std::int64_t stride, std::int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", " for USM");
}

sycl::event imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, std::complex<float> alpha, std::complex<float>* ab,
                           std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                           std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", " for USM");
}

sycl::event imatcopy_batch(sycl::queue& queue, oneapi::math::transpose trans, std::int64_t m,
                           std::int64_t n, std::complex<double> alpha, std::complex<double>* ab,
                           std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                           std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "imatcopy_batch", " for USM");
}

sycl::event omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                          oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                          float alpha, const float* a, std::int64_t lda, std::int64_t stride_a,
                          float beta, const float* b, std::int64_t ldb, std::int64_t stride_b,
                          float* c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_omatadd_batch, queue, transa, transb, m, n, alpha, a, lda,
                         stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
                         dependencies);
}

sycl::event omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                          oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                          double alpha, const double* a, std::int64_t lda, std::int64_t stride_a,
                          double beta, const double* b, std::int64_t ldb, std::int64_t stride_b,
                          double* c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_omatadd_batch, queue, transa, transb, m, n, alpha, a, lda,
                         stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
                         dependencies);
}

sycl::event omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                          oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                          std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
                          std::int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
                          std::complex<float>* c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", " for USM");
}

sycl::event omatadd_batch(sycl::queue& queue, oneapi::math::transpose transa,
                          oneapi::math::transpose transb, std::int64_t m, std::int64_t n,
                          std::complex<double> alpha, const std::complex<double>* a,
                          std::int64_t lda, std::int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
                          std::complex<double>* c, std::int64_t ldc, std::int64_t stride_c,
                          std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "omatadd_batch", " for USM");
}
