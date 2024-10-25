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

void gemv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          real_t alpha, sycl::buffer<real_t, 1>& a, std::int64_t lda, sycl::buffer<real_t, 1>& x,
          std::int64_t incx, real_t beta, sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_gemv, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "gemv", " for complex");
}

void gbmv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, real_t alpha, sycl::buffer<real_t, 1>& a,
          std::int64_t lda, sycl::buffer<real_t, 1>& x, std::int64_t incx, real_t beta,
          sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_gbmv, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                     incy);
}

void gbmv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "gbmv", " for complex");
}

void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, real_t alpha,
         sycl::buffer<real_t, 1>& x, std::int64_t incx, sycl::buffer<real_t, 1>& y,
         std::int64_t incy, sycl::buffer<real_t, 1>& a, std::int64_t lda) {
    CALL_PORTBLAS_FN(::blas::_ger, queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda) {
    throw unimplemented("blas", "gerc", "");
}

void geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda) {
    throw unimplemented("blas", "geru", "");
}

void hbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "hbmv", "");
}

void hemv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "hemv", "");
}

void her(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
         sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda) {
    throw unimplemented("blas", "her", "");
}

void her2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda) {
    throw unimplemented("blas", "her2", "");
}

void hpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& a,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "hpmv", "");
}

void hpr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
         sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
         sycl::buffer<std::complex<real_t>, 1>& a) {
    throw unimplemented("blas", "hpr", "");
}

void hpr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& a) {
    throw unimplemented("blas", "hpr2", "");
}

void sbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          real_t alpha, sycl::buffer<real_t, 1>& a, std::int64_t lda, sycl::buffer<real_t, 1>& x,
          std::int64_t incx, real_t beta, sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_sbmv, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                     incy);
}

void symv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
          sycl::buffer<real_t, 1>& a, std::int64_t lda, sycl::buffer<real_t, 1>& x,
          std::int64_t incx, real_t beta, sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_symv, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void syr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
         sycl::buffer<real_t, 1>& x, std::int64_t incx, sycl::buffer<real_t, 1>& a,
         std::int64_t lda) {
    CALL_PORTBLAS_FN(::blas::_syr, queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
          sycl::buffer<real_t, 1>& x, std::int64_t incx, sycl::buffer<real_t, 1>& y,
          std::int64_t incy, sycl::buffer<real_t, 1>& a, std::int64_t lda) {
    CALL_PORTBLAS_FN(::blas::_syr2, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void spmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
          sycl::buffer<real_t, 1>& a, sycl::buffer<real_t, 1>& x, std::int64_t incx, real_t beta,
          sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_spmv, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
         sycl::buffer<real_t, 1>& x, std::int64_t incx, sycl::buffer<real_t, 1>& a) {
    CALL_PORTBLAS_FN(::blas::_spr, queue, upper_lower, n, alpha, x, incx, a);
}

void spr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
          sycl::buffer<real_t, 1>& x, std::int64_t incx, sycl::buffer<real_t, 1>& y,
          std::int64_t incy, sycl::buffer<real_t, 1>& a) {
    CALL_PORTBLAS_FN(::blas::_spr2, queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void tbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<real_t, 1>& a,
          std::int64_t lda, sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_tbmv, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "tbmv", "");
}

void tbsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, sycl::buffer<real_t, 1>& a,
          std::int64_t lda, sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_tbsv, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
          sycl::buffer<std::complex<real_t>, 1>& a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "tbsv", "");
}

void tpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<real_t, 1>& a,
          sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_tpmv, queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& a,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "tpmv", "");
}

void tpsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<real_t, 1>& a,
          sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_tpsv, queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& a,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "tpsv", "");
}

void trmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<real_t, 1>& a, std::int64_t lda,
          sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_trmv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "trmv", " for complex");
}

void trsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<real_t, 1>& a, std::int64_t lda,
          sycl::buffer<real_t, 1>& x, std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_trsv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          oneapi::mkl::diag unit_diag, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& a,
          std::int64_t lda, sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "trsv", "");
}

// USM APIs

sycl::event gemv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 real_t alpha, const real_t* a, std::int64_t lda, const real_t* x,
                 std::int64_t incx, real_t beta, real_t* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_gemv, queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                         dependencies);
}

sycl::event gemv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::complex<real_t> alpha, const std::complex<real_t>* a, std::int64_t lda,
                 const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t> beta,
                 std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gemv", " for USM");
}

sycl::event gbmv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, real_t alpha, const real_t* a, std::int64_t lda,
                 const real_t* x, std::int64_t incx, real_t beta, real_t* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_gbmv, queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                         incy, dependencies);
}

sycl::event gbmv(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                 std::int64_t kl, std::int64_t ku, std::complex<real_t> alpha,
                 const std::complex<real_t>* a, std::int64_t lda, const std::complex<real_t>* x,
                 std::int64_t incx, std::complex<real_t> beta, std::complex<real_t>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gbmv", " for USM");
}

sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, real_t alpha, const real_t* x,
                std::int64_t incx, const real_t* y, std::int64_t incy, real_t* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_ger, queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
                 const std::complex<real_t>* x, std::int64_t incx, const std::complex<real_t>* y,
                 std::int64_t incy, std::complex<real_t>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "gerc", " for USM");
}

sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
                 const std::complex<real_t>* x, std::int64_t incx, const std::complex<real_t>* y,
                 std::int64_t incy, std::complex<real_t>* a, std::int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "geru", " for USM");
}

sycl::event hbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 std::complex<real_t> alpha, const std::complex<real_t>* a, std::int64_t lda,
                 const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t> beta,
                 std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hbmv", " for USM");
}

sycl::event hemv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<real_t> alpha, const std::complex<real_t>* a, std::int64_t lda,
                 const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t> beta,
                 std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hemv", " for USM");
}

sycl::event her(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t>* a,
                std::int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "her", " for USM");
}

sycl::event her2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<real_t> alpha, const std::complex<real_t>* x, std::int64_t incx,
                 const std::complex<real_t>* y, std::int64_t incy, std::complex<real_t>* a,
                 std::int64_t lda, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "her2", " for USM");
}

sycl::event hpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<real_t> alpha, const std::complex<real_t>* a,
                 const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t> beta,
                 std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpmv", " for USM");
}

sycl::event hpr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t>* a,
                const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpr", " for USM");
}

sycl::event hpr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                 std::complex<real_t> alpha, const std::complex<real_t>* x, std::int64_t incx,
                 const std::complex<real_t>* y, std::int64_t incy, std::complex<real_t>* a,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "hpr2", " for USM");
}

sycl::event sbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, std::int64_t k,
                 real_t alpha, const real_t* a, std::int64_t lda, const real_t* x,
                 std::int64_t incx, real_t beta, real_t* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_sbmv, queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                         incy, dependencies);
}

sycl::event symv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                 const real_t* a, std::int64_t lda, const real_t* x, std::int64_t incx, real_t beta,
                 real_t* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_symv, queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                         incy, dependencies);
}

sycl::event syr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                const real_t* x, std::int64_t incx, real_t* a, std::int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_syr, queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
}

sycl::event syr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                 const real_t* x, std::int64_t incx, const real_t* y, std::int64_t incy, real_t* a,
                 std::int64_t lda, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_syr2, queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,
                         dependencies);
}

sycl::event spmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                 const real_t* a, const real_t* x, std::int64_t incx, real_t beta, real_t* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_spmv, queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,
                         dependencies);
}

sycl::event spr(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                const real_t* x, std::int64_t incx, real_t* a,
                const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_spr, queue, upper_lower, n, alpha, x, incx, a, dependencies);
}

sycl::event spr2(sycl::queue& queue, oneapi::mkl::uplo upper_lower, std::int64_t n, real_t alpha,
                 const real_t* x, std::int64_t incx, const real_t* y, std::int64_t incy, real_t* a,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_spr2, queue, upper_lower, n, alpha, x, incx, y, incy, a,
                         dependencies);
}

sycl::event tbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const real_t* a,
                 std::int64_t lda, real_t* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_tbmv, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                         dependencies);
}

sycl::event tbmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<real_t>* a, std::int64_t lda, std::complex<real_t>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tbmv", " for USM");
}

sycl::event tbsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k, const real_t* a,
                 std::int64_t lda, real_t* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_tbsv, queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx,
                         dependencies);
}

sycl::event tbsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, std::int64_t k,
                 const std::complex<real_t>* a, std::int64_t lda, std::complex<real_t>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tbsv", " for USM");
}

sycl::event tpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const real_t* a, real_t* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_tpmv, queue, upper_lower, trans, unit_diag, n, a, x, incx,
                         dependencies);
}

sycl::event tpmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<real_t>* a,
                 std::complex<real_t>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tpmv", " for USM");
}

sycl::event tpsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const real_t* a, real_t* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_tpsv, queue, upper_lower, trans, unit_diag, n, a, x, incx,
                         dependencies);
}

sycl::event tpsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<real_t>* a,
                 std::complex<real_t>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "tpsv", " for USM");
}

sycl::event trmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const real_t* a, std::int64_t lda,
                 real_t* x, std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_trmv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                         dependencies);
}

sycl::event trmv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<real_t>* a,
                 std::int64_t lda, std::complex<real_t>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trmv", " for USM");
}

sycl::event trsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const real_t* a, std::int64_t lda,
                 real_t* x, std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_trsv, queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                         dependencies);
}

sycl::event trsv(sycl::queue& queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 oneapi::mkl::diag unit_diag, std::int64_t n, const std::complex<real_t>* a,
                 std::int64_t lda, std::complex<real_t>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "trsv", " for USM");
}
