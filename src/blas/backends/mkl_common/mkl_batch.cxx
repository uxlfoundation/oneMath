/*******************************************************************************
* Copyright 2022 Intel Corporation
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
                std::int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, std::int64_t stridey,
                std::int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size));
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
                std::int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                std::int64_t stridey, std::int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size));
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size));
}

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                int64_t incx, std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::copy_batch(queue, n, x, incx, stridex, y, incy, stridey, batch_size));
}

void axpy_batch(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size));
}

void axpy_batch(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size));
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size));
}

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy, stridey, batch_size));
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, float beta, sycl::buffer<float, 1>& y, int64_t incy,
                int64_t stride_y, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m,
                                                     n, alpha, a, lda, stride_a, x, incx, stride_x,
                                                     beta, y, incy, stride_y, batch_size));
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x, double beta,
                sycl::buffer<double, 1>& y, int64_t incy, int64_t stride_y, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m,
                                                     n, alpha, a, lda, stride_a, x, incx, stride_x,
                                                     beta, y, incy, stride_y, batch_size));
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stride_y, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m,
                                                     n, alpha, a, lda, stride_a, x, incx, stride_x,
                                                     beta, y, incy, stride_y, batch_size));
}

void gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
                int64_t stride_x, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stride_y,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m,
                                                     n, alpha, a, lda, stride_a, x, incx, stride_x,
                                                     beta, y, incy, stride_y, batch_size));
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stride_x, sycl::buffer<float, 1>& c, int64_t ldc,
                int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m,
                                                     n, a, lda, stride_a, x, incx, stride_x, c, ldc,
                                                     stride_c, batch_size));
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m,
                                                     n, a, lda, stride_a, x, incx, stride_x, c, ldc,
                                                     stride_c, batch_size));
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m,
                                                     n, a, lda, stride_a, x, incx, stride_x, c, ldc,
                                                     stride_c, batch_size));
}

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stride_x,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m,
                                                     n, a, lda, stride_a, x, incx, stride_x, c, ldc,
                                                     stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, double beta,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                int64_t stride_b, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    throw unimplemented("blas", "gemm_batch",
                        "unsupported dtype combination: int8_t, int8_t, float, float");
}

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<std::int32_t, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size));
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size));
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size));
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size));
}

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size));
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size));
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size));
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size));
}

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size));
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans),
                                                         m, n, alpha, a, lda, stride_a, b, ldb,
                                                         stride_b, batch_size));
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans),
                                                         m, n, alpha, a, lda, stride_a, b, ldb,
                                                         stride_b, batch_size));
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans),
                                                         m, n, alpha, a, lda, stride_a, b, ldb,
                                                         stride_b, batch_size));
}

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans),
                                                         m, n, alpha, a, lda, stride_a, b, ldb,
                                                         stride_b, batch_size));
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::imatcopy_batch(
        queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda, ldb, stride, batch_size));
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::imatcopy_batch(
        queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda, ldb, stride, batch_size));
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::imatcopy_batch(
        queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda, ldb, stride, batch_size));
}

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::imatcopy_batch(
        queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda, ldb, stride, batch_size));
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size));
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size));
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size));
}

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size));
}

// USM APIs

sycl::event copy_batch(sycl::queue& queue, int64_t n, const float* x, int64_t incx,
                       std::int64_t stridex, float* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy_batch(queue, n, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const double* x, int64_t incx,
                       std::int64_t stridex, double* y, int64_t incy, std::int64_t stridey,
                       std::int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy_batch(queue, n, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                       std::int64_t stridex, std::complex<float>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy_batch(queue, n, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                       std::int64_t stridex, std::complex<double>* y, int64_t incy,
                       std::int64_t stridey, std::int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy_batch(queue, n, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const float** x, int64_t* incx, float** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::copy_batch(queue, n, x, incx, y, incy, group_count, group_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const double** x, int64_t* incx, double** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::copy_batch(queue, n, x, incx, y, incy, group_count, group_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<float>** x, int64_t* incx,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::copy_batch(queue, n, x, incx, y, incy, group_count, group_size, dependencies));
}

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::copy_batch(queue, n, x, incx, y, incy, group_count, group_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                       int64_t stridex, float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, stridex, y, incy,
                                                         stridey, batch_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, float* alpha, const float** x, int64_t* incx,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, y, incy,
                                                         group_count, group_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, double* alpha, const double** x,
                       int64_t* incx, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, y, incy,
                                                         group_count, group_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, y, incy,
                                                         group_count, group_size, dependencies));
}

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::axpy_batch(queue, n, alpha, x, incx, y, incy,
                                                         group_count, group_size, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, float alpha,
                       const float* a, int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float beta, float* y, int64_t incy, int64_t stride_y,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv_batch(
        queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, stride_a, x, incx,
        stride_x, beta, y, incy, stride_y, batch_size, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n, double alpha,
                       const double* a, int64_t lda, int64_t stride_a, const double* x,
                       int64_t incx, int64_t stride_x, double beta, double* y, int64_t incy,
                       int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv_batch(
        queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, stride_a, x, incx,
        stride_x, beta, y, incy, stride_y, batch_size, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, const std::complex<float>* x, int64_t incx,
                       int64_t stride_x, std::complex<float> beta, std::complex<float>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv_batch(
        queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, stride_a, x, incx,
        stride_x, beta, y, incy, stride_y, batch_size, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose transa, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, const std::complex<double>* x, int64_t incx,
                       int64_t stride_x, std::complex<double> beta, std::complex<double>* y,
                       int64_t incy, int64_t stride_y, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv_batch(
        queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, stride_a, x, incx,
        stride_x, beta, y, incy, stride_y, batch_size, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, float* alpha,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float* beta,
                       float** y, int64_t* incy, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, x,
                               incx, beta, y, incy, group_count, groupsize, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n, double* alpha,
                       const double** a, int64_t* lda, const double** x, int64_t* incx,
                       double* beta, double** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, x,
                               incx, beta, y, incy, group_count, groupsize, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>* beta,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, x,
                               incx, beta, y, incy, group_count, groupsize, dependencies));
}

sycl::event gemv_batch(sycl::queue& queue, transpose* transa, int64_t* m, int64_t* n,
                       std::complex<double>* alpha, const std::complex<double>** a, int64_t* lda,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>* beta,
                       std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gemv_batch(queue, detail::get_onemkl_transpose(transa), m, n, alpha, a, lda, x,
                               incx, beta, y, incy, group_count, groupsize, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const float* a,
                       int64_t lda, int64_t stride_a, const float* x, int64_t incx,
                       int64_t stride_x, float* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m, n, a, lda, stride_a,
                               x, incx, stride_x, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const double* a,
                       int64_t lda, int64_t stride_a, const double* x, int64_t incx,
                       int64_t stride_x, double* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m, n, a, lda, stride_a,
                               x, incx, stride_x, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       const std::complex<float>* x, int64_t incx, int64_t stride_x,
                       std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m, n, a, lda, stride_a,
                               x, incx, stride_x, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       const std::complex<double>* x, int64_t incx, int64_t stride_x,
                       std::complex<double>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right), m, n, a, lda, stride_a,
                               x, incx, stride_x, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right),
                                                         m, n, a, lda, x, incx, c, ldc, group_count,
                                                         groupsize, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const double** a, int64_t* lda, const double** x, int64_t* incx, double** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right),
                                                         m, n, a, lda, x, incx, c, ldc, group_count,
                                                         groupsize, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** x,
                       int64_t* incx, std::complex<float>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right),
                                                         m, n, a, lda, x, incx, c, ldc, group_count,
                                                         groupsize, dependencies));
}

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** c, int64_t* ldc, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::dgmm_batch(queue, detail::get_onemkl_side(left_right),
                                                         m, n, a, lda, x, incx, c, ldc, group_count,
                                                         groupsize, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const float* a, int64_t lda, int64_t stride_a,
                       const float* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, double alpha, const double* a, int64_t lda, int64_t stride_a,
                       const double* b, int64_t ldb, int64_t stride_b, double beta, double* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                       int64_t lda, int64_t stride_a, const std::complex<float>* b, int64_t ldb,
                       int64_t stride_b, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                       int64_t lda, int64_t stride_a, const std::complex<double>* b, int64_t ldb,
                       int64_t stride_b, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda,
                       int64_t stride_a, const sycl::half* b, int64_t ldb, int64_t stride_b,
                       sycl::half beta, sycl::half* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const sycl::half* a, int64_t lda, int64_t stride_a,
                       const sycl::half* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch",
                        "unsupported dtype combination: int8_t, int8_t, float, float");
}

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta,
                       std::int32_t* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const float** a, int64_t* lda,
                       const float** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, double* alpha, const double** a, int64_t* lda,
                       const double** b, int64_t* ldb, double* beta, double** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** b,
                       int64_t* ldb, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** b,
                       int64_t* ldb, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, group_size, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, sycl::half* alpha, const sycl::half** a,
                       int64_t* lda, const sycl::half** b, int64_t* ldb, sycl::half* beta,
                       sycl::half** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const sycl::half** a, int64_t* lda,
                       const sycl::half** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemm_batch",
                        "unsupported dtype combination: int8_t, int8_t, float, float");
}

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, std::int32_t** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemm_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n, k,
        alpha, a, lda, b, ldb, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float* a,
                       int64_t lda, int64_t stride_a, float* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double* a,
                       int64_t lda, int64_t stride_a, double* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       std::complex<float>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       std::complex<double>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, float* alpha, const float** a,
                       int64_t* lda, float** b, int64_t* ldb, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, b, ldb, group_count, groupsize, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, double* alpha, const double** a,
                       int64_t* lda, double** b, int64_t* ldb, int64_t group_count,
                       int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, b, ldb, group_count, groupsize, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, std::complex<float>** b,
                       int64_t* ldb, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, b, ldb, group_count, groupsize, dependencies));
}

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, std::complex<double>** b,
                       int64_t* ldb, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::trsm_batch(
        queue, detail::get_onemkl_side(left_right), detail::get_onemkl_uplo(upper_lower),
        detail::get_onemkl_transpose(trans), detail::get_onemkl_diag(unit_diag), m, n, alpha, a,
        lda, b, ldb, group_count, groupsize, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, float beta,
                       float* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double* a, int64_t lda, int64_t stride_a, double beta,
                       double* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, float* alpha, const float** a, int64_t* lda, float* beta,
                       float** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, double* alpha, const double** a, int64_t* lda, double* beta,
                       double** c, int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       int64_t* lda, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* groupsize,
                       const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syrk_batch(
        queue, detail::get_onemkl_uplo(upper_lower), detail::get_onemkl_transpose(trans), n, k,
        alpha, a, lda, beta, c, ldc, group_count, groupsize, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float* a, int64_t lda, int64_t stride_a, float* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double* a, int64_t lda, int64_t stride_a, double* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                           int64_t stride_a, std::complex<float>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                           int64_t stride_a, std::complex<double>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   stride_a, b, ldb, stride_b, batch_size, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, stride, batch_size, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, stride, batch_size, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, stride, batch_size, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, stride, batch_size, dependencies));
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float* a, int64_t lda, int64_t stride_a,
                          float beta, const float* b, int64_t ldb, int64_t stride_b, float* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double* a, int64_t lda, int64_t stride_a,
                          double beta, const double* b, int64_t ldb, int64_t stride_b, double* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, int64_t ldb, int64_t stride_b,
                          std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, int64_t ldb, int64_t stride_b,
                          std::complex<double>* c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::omatadd_batch(
        queue, detail::get_onemkl_transpose(transa), detail::get_onemkl_transpose(transb), m, n,
        alpha, a, lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size,
        dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, const float** a, int64_t* lda, float** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   b, ldb, group_count, groupsize, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, const double** a, int64_t* lda, double** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   b, ldb, group_count, groupsize, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                           std::complex<float>** b, int64_t* ldb, int64_t group_count,
                           int64_t* groupsize, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   b, ldb, group_count, groupsize, dependencies));
}

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, const std::complex<double>** a,
                           int64_t* lda, std::complex<double>** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::omatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, a, lda,
                                   b, ldb, group_count, groupsize, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, float** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, group_count, groupsize, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, double** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, group_count, groupsize, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, std::complex<float>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, group_count, groupsize, dependencies));
}

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, std::complex<double>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::imatcopy_batch(queue, detail::get_onemkl_transpose(trans), m, n, alpha, ab, lda,
                                   ldb, group_count, groupsize, dependencies));
}
