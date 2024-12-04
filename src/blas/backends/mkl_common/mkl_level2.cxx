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

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx,
          float beta, sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                               alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                               alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                               alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                               alpha, a, lda, x, incx, beta, y, incy));
}

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
          sycl::buffer<float, 1>& x, std::int64_t incx, float beta, sycl::buffer<float, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n, kl,
                                               ku, alpha, a, lda, x, incx, beta, y, incy));
}

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
          sycl::buffer<double, 1>& x, std::int64_t incx, double beta, sycl::buffer<double, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n, kl,
                                               ku, alpha, a, lda, x, incx, beta, y, incy));
}

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n, kl,
                                               ku, alpha, a, lda, x, incx, beta, y, incy));
}

void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
          std::int64_t ku, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n, kl,
                                               ku, alpha, a, lda, x, incx, beta, y, incy));
}

void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
         std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& a,
         std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, double alpha,
         sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
         std::int64_t incy, sycl::buffer<double, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda));
}

void hbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hbmv(queue, detail::get_onemkl_uplo(uplo), n, k, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void hbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hbmv(queue, detail::get_onemkl_uplo(uplo), n, k, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void hemv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hemv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void hemv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hemv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void her(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::her(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, lda));
}

void her(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::her(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, lda));
}

void her2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::her2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a, lda));
}

void her2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::her2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a, lda));
}

void hpmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hpmv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a, x,
                                               incx, beta, y, incy));
}

void hpmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::hpmv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a, x,
                                               incx, beta, y, incy));
}

void hpr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::hpr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a));
}

void hpr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::hpr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a));
}

void hpr2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::hpr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a));
}

void hpr2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::hpr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a));
}

void sbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, float alpha,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx,
          float beta, sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::sbmv(queue, detail::get_onemkl_uplo(uplo), n, k, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void sbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, double alpha,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::sbmv(queue, detail::get_onemkl_uplo(uplo), n, k, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void spmv(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
          sycl::buffer<float, 1>& x, std::int64_t incx, float beta, sycl::buffer<float, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::spmv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a, x,
                                               incx, beta, y, incy));
}

void spmv(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
          sycl::buffer<double, 1>& x, std::int64_t incx, double beta, sycl::buffer<double, 1>& y,
          std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::spmv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a, x,
                                               incx, beta, y, incy));
}

void spr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
         std::int64_t incx, sycl::buffer<float, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::spr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a));
}

void spr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
         std::int64_t incx, sycl::buffer<double, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::spr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a));
}

void spr2(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
          sycl::buffer<float, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::spr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a));
}

void spr2(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
          sycl::buffer<double, 1>& a) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::spr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a));
}

void symv(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
          std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
          sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::symv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void symv(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
          std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
          sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::symv(queue, detail::get_onemkl_uplo(uplo), n, alpha, a,
                                               lda, x, incx, beta, y, incy));
}

void syr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
         std::int64_t incx, sycl::buffer<float, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::syr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, lda));
}

void syr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
         std::int64_t incx, sycl::buffer<double, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::syr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, lda));
}

void syr2(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
          sycl::buffer<float, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::syr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a, lda));
}

void syr2(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
          sycl::buffer<double, 1>& a, std::int64_t lda) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::syr2(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, y, incy, a, lda));
}

void tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx));
}

void tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, x, incx));
}

void trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trmv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

void trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::trsv(queue, detail::get_onemkl_uplo(uplo),
                                               detail::get_onemkl_transpose(trans),
                                               detail::get_onemkl_diag(diag), n, a, lda, x, incx));
}

// USM APIs

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
                 const float* a, std::int64_t lda, const float* x, std::int64_t incx, float beta,
                 float* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
                 const double* a, std::int64_t lda, const double* x, std::int64_t incx, double beta,
                 double* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
                 const std::complex<float>* x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
                 const std::complex<double>* x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gemv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, float alpha, const float* a, std::int64_t lda,
                 const float* x, std::int64_t incx, float beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, double alpha, const double* a, std::int64_t lda,
                 const double* x, std::int64_t incx, double beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* x,
                 std::int64_t incx, std::complex<float> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* x,
                 std::int64_t incx, std::complex<double> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::gbmv(queue, detail::get_onemkl_transpose(trans), m, n,
                                                   kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, float alpha, const float* x,
                std::int64_t incx, const float* y, std::int64_t incy, float* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, double alpha, const double* x,
                std::int64_t incx, const double* y, std::int64_t incy, double* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, const std::complex<float>* y,
                 std::int64_t incy, std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, const std::complex<double>* y,
                 std::int64_t incy, std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, const std::complex<float>* y,
                 std::int64_t incy, std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, const std::complex<double>* y,
                 std::int64_t incy, std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies));
}

sycl::event hbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k,
                 std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
                 const std::complex<float>* x, std::int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hbmv(queue, detail::get_onemkl_uplo(uplo), n, k,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event hbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k,
                 std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
                 const std::complex<double>* x, std::int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hbmv(queue, detail::get_onemkl_uplo(uplo), n, k,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event hemv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, std::int64_t lda, const std::complex<float>* x,
                 std::int64_t incx, std::complex<float> beta, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hemv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, lda, x, incx, beta, y, incy, dependencies));
}

sycl::event hemv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, std::int64_t lda, const std::complex<double>* x,
                 std::int64_t incx, std::complex<double> beta, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hemv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, lda, x, incx, beta, y, incy, dependencies));
}

sycl::event her(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha,
                const std::complex<float>* x, std::int64_t incx, std::complex<float>* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::her(queue, detail::get_onemkl_uplo(uplo), n, alpha, x,
                                                  incx, a, lda, dependencies));
}

sycl::event her(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha,
                const std::complex<double>* x, std::int64_t incx, std::complex<double>* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::her(queue, detail::get_onemkl_uplo(uplo), n, alpha, x,
                                                  incx, a, lda, dependencies));
}

sycl::event her2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, const std::complex<float>* y,
                 std::int64_t incy, std::complex<float>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::her2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, lda, dependencies));
}

sycl::event her2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, const std::complex<double>* y,
                 std::int64_t incy, std::complex<double>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::her2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, lda, dependencies));
}

sycl::event hpmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, const std::complex<float>* x, std::int64_t incx,
                 std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hpmv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, x, incx, beta, y, incy, dependencies));
}

sycl::event hpmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, const std::complex<double>* x, std::int64_t incx,
                 std::complex<double> beta, std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hpmv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, x, incx, beta, y, incy, dependencies));
}

sycl::event hpr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha,
                const std::complex<float>* x, std::int64_t incx, std::complex<float>* a,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::hpr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, dependencies));
}

sycl::event hpr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha,
                const std::complex<double>* x, std::int64_t incx, std::complex<double>* a,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::hpr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, dependencies));
}

sycl::event hpr2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, const std::complex<float>* y,
                 std::int64_t incy, std::complex<float>* a,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hpr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, dependencies));
}

sycl::event hpr2(sycl::queue& queue, uplo uplo, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, const std::complex<double>* y,
                 std::int64_t incy, std::complex<double>* a,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::hpr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, dependencies));
}

sycl::event sbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, float alpha,
                 const float* a, std::int64_t lda, const float* x, std::int64_t incx, float beta,
                 float* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::sbmv(queue, detail::get_onemkl_uplo(uplo), n, k,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event sbmv(sycl::queue& queue, uplo uplo, std::int64_t n, std::int64_t k, double alpha,
                 const double* a, std::int64_t lda, const double* x, std::int64_t incx, double beta,
                 double* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::sbmv(queue, detail::get_onemkl_uplo(uplo), n, k,
                                                   alpha, a, lda, x, incx, beta, y, incy,
                                                   dependencies));
}

sycl::event spmv(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* a,
                 const float* x, std::int64_t incx, float beta, float* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::spmv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, x, incx, beta, y, incy, dependencies));
}

sycl::event spmv(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* a,
                 const double* x, std::int64_t incx, double beta, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::spmv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, x, incx, beta, y, incy, dependencies));
}

sycl::event spr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* x,
                std::int64_t incx, float* a, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::spr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, dependencies));
}

sycl::event spr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* x,
                std::int64_t incx, double* a, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::spr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x, incx, a, dependencies));
}

sycl::event spr2(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* x,
                 std::int64_t incx, const float* y, std::int64_t incy, float* a,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::spr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, dependencies));
}

sycl::event spr2(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* x,
                 std::int64_t incx, const double* y, std::int64_t incy, double* a,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::spr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, dependencies));
}

sycl::event symv(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* a,
                 std::int64_t lda, const float* x, std::int64_t incx, float beta, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::symv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, lda, x, incx, beta, y, incy, dependencies));
}

sycl::event symv(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* a,
                 std::int64_t lda, const double* x, std::int64_t incx, double beta, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::symv(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   a, lda, x, incx, beta, y, incy, dependencies));
}

sycl::event syr(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* x,
                std::int64_t incx, float* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x,
                                                  incx, a, lda, dependencies));
}

sycl::event syr(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* x,
                std::int64_t incx, double* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syr(queue, detail::get_onemkl_uplo(uplo), n, alpha, x,
                                                  incx, a, lda, dependencies));
}

sycl::event syr2(sycl::queue& queue, uplo uplo, std::int64_t n, float alpha, const float* x,
                 std::int64_t incx, const float* y, std::int64_t incy, float* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, lda, dependencies));
}

sycl::event syr2(sycl::queue& queue, uplo uplo, std::int64_t n, double alpha, const double* x,
                 std::int64_t incx, const double* y, std::int64_t incy, double* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::syr2(queue, detail::get_onemkl_uplo(uplo), n, alpha,
                                                   x, incx, y, incy, a, lda, dependencies));
}

sycl::event tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const std::complex<float>* a, std::int64_t lda,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tbsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 std::int64_t k, const std::complex<double>* a, std::int64_t lda,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tbsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, k, a, lda, x, incx, dependencies));
}

sycl::event tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const float* a, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const double* a, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<float>* a, std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<double>* a, std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const float* a, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const double* a, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<float>* a, std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event tpsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<double>* a, std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::tpsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, x, incx, dependencies));
}

sycl::event trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<float>* a, std::int64_t lda, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trmv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<double>* a, std::int64_t lda, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trmv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const float* a, std::int64_t lda, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const double* a, std::int64_t lda, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<float>* a, std::int64_t lda, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}

sycl::event trsv(sycl::queue& queue, uplo uplo, transpose trans, diag diag, std::int64_t n,
                 const std::complex<double>* a, std::int64_t lda, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::trsv(queue, detail::get_onemkl_uplo(uplo), detail::get_onemkl_transpose(trans),
                         detail::get_onemkl_diag(diag), n, a, lda, x, incx, dependencies));
}
