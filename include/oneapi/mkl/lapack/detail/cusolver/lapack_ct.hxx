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

// Buffer APIs

static inline void gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e,
                         sycl::buffer<std::complex<float>>& tauq,
                         sycl::buffer<std::complex<float>>& taup,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq, taup,
                                         scratchpad, scratchpad_size);
}
static inline void gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<double>& tauq, sycl::buffer<double>& taup,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq, taup,
                                         scratchpad, scratchpad_size);
}
static inline void gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e, sycl::buffer<float>& tauq,
                         sycl::buffer<float>& taup, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq, taup,
                                         scratchpad, scratchpad_size);
}
static inline void gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<std::complex<double>>& tauq,
                         sycl::buffer<std::complex<double>>& taup,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq, taup,
                                         scratchpad, scratchpad_size);
}
static inline void gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void getrs(backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void getrs(backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<double>& b, std::int64_t ldb,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void getrs(backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& b,
                         std::int64_t ldb, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void getrs(backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void gesvd(backend_selector<backend::cusolver> selector, oneapi::mkl::jobsvd jobu,
                         oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& s,
                         sycl::buffer<double>& u, std::int64_t ldu, sycl::buffer<double>& vt,
                         std::int64_t ldvt, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s, u, ldu,
                                         vt, ldvt, scratchpad, scratchpad_size);
}
static inline void gesvd(backend_selector<backend::cusolver> selector, oneapi::mkl::jobsvd jobu,
                         oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& s,
                         sycl::buffer<float>& u, std::int64_t ldu, sycl::buffer<float>& vt,
                         std::int64_t ldvt, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s, u, ldu,
                                         vt, ldvt, scratchpad, scratchpad_size);
}
static inline void gesvd(backend_selector<backend::cusolver> selector, oneapi::mkl::jobsvd jobu,
                         oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& s, sycl::buffer<std::complex<float>>& u,
                         std::int64_t ldu, sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s, u, ldu,
                                         vt, ldvt, scratchpad, scratchpad_size);
}
static inline void gesvd(backend_selector<backend::cusolver> selector, oneapi::mkl::jobsvd jobu,
                         oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& s, sycl::buffer<std::complex<double>>& u,
                         std::int64_t ldu, sycl::buffer<std::complex<double>>& vt,
                         std::int64_t ldvt, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s, u, ldu,
                                         vt, ldvt, scratchpad, scratchpad_size);
}
static inline void heevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                         oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::heevd(selector.get_queue(), jobz, uplo, n, a, lda, w, scratchpad,
                                         scratchpad_size);
}
static inline void heevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                         oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::heevd(selector.get_queue(), jobz, uplo, n, a, lda, w, scratchpad,
                                         scratchpad_size);
}
static inline void hegvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hegvd(selector.get_queue(), itype, jobz, uplo, n, a, lda, b, ldb,
                                         w, scratchpad, scratchpad_size);
}
static inline void hegvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hegvd(selector.get_queue(), itype, jobz, uplo, n, a, lda, b, ldb,
                                         w, scratchpad, scratchpad_size);
}
static inline void hetrd(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hetrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                         scratchpad, scratchpad_size);
}
static inline void hetrd(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hetrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                         scratchpad, scratchpad_size);
}
static inline void hetrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hetrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void hetrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::hetrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void orgbr(backend_selector<backend::cusolver> selector, oneapi::mkl::generate vec,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                         scratchpad, scratchpad_size);
}
static inline void orgbr(backend_selector<backend::cusolver> selector, oneapi::mkl::generate vec,
                         std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& tau,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                         scratchpad, scratchpad_size);
}
static inline void orgqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgqr(selector.get_queue(), m, n, k, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void orgqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgqr(selector.get_queue(), m, n, k, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void orgtr(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgtr(selector.get_queue(), uplo, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void orgtr(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgtr(selector.get_queue(), uplo, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void ormtr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormtr(selector.get_queue(), side, uplo, trans, m, n, a, lda, tau,
                                         c, ldc, scratchpad, scratchpad_size);
}
static inline void ormtr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormtr(selector.get_queue(), side, uplo, trans, m, n, a, lda, tau,
                                         c, ldc, scratchpad, scratchpad_size);
}
static inline void ormrq(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormrq(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void ormrq(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormrq(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void ormqr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormqr(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void ormqr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ormqr(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void potrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potri(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potri(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potri(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potri(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                         scratchpad_size);
}
static inline void potrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void potrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void potrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void potrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                         scratchpad, scratchpad_size);
}
static inline void syevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                         oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
                         std::int64_t lda, sycl::buffer<double>& w,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::syevd(selector.get_queue(), jobz, uplo, n, a, lda, w, scratchpad,
                                         scratchpad_size);
}
static inline void syevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                         oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
                         std::int64_t lda, sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::syevd(selector.get_queue(), jobz, uplo, n, a, lda, w, scratchpad,
                                         scratchpad_size);
}
static inline void sygvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
                         std::int64_t ldb, sycl::buffer<double>& w,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sygvd(selector.get_queue(), itype, jobz, uplo, n, a, lda, b, ldb,
                                         w, scratchpad, scratchpad_size);
}
static inline void sygvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                         sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
                         std::int64_t ldb, sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sygvd(selector.get_queue(), itype, jobz, uplo, n, a, lda, b, ldb,
                                         w, scratchpad, scratchpad_size);
}
static inline void sytrd(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& d, sycl::buffer<double>& e,
                         sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                         scratchpad, scratchpad_size);
}
static inline void sytrd(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& d, sycl::buffer<float>& e, sycl::buffer<float>& tau,
                         sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                         scratchpad, scratchpad_size);
}
static inline void sytrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void sytrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void sytrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void sytrf(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::int64_t>& ipiv,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv, scratchpad,
                                         scratchpad_size);
}
static inline void trtrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a, lda,
                                         b, ldb, scratchpad, scratchpad_size);
}
static inline void trtrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                         sycl::buffer<double>& b, std::int64_t ldb,
                         sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a, lda,
                                         b, ldb, scratchpad, scratchpad_size);
}
static inline void trtrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                         sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a, lda,
                                         b, ldb, scratchpad, scratchpad_size);
}
static inline void trtrs(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                         std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a, lda,
                                         b, ldb, scratchpad, scratchpad_size);
}
static inline void ungbr(backend_selector<backend::cusolver> selector, oneapi::mkl::generate vec,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                         scratchpad, scratchpad_size);
}
static inline void ungbr(backend_selector<backend::cusolver> selector, oneapi::mkl::generate vec,
                         std::int64_t m, std::int64_t n, std::int64_t k,
                         sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                         scratchpad, scratchpad_size);
}
static inline void ungqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungqr(selector.get_queue(), m, n, k, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void ungqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                         std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
                         std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungqr(selector.get_queue(), m, n, k, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void ungtr(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungtr(selector.get_queue(), uplo, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void ungtr(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungtr(selector.get_queue(), uplo, n, a, lda, tau, scratchpad,
                                         scratchpad_size);
}
static inline void unmrq(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmrq(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void unmrq(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmrq(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void unmqr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmqr(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void unmqr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                         std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmqr(selector.get_queue(), side, trans, m, n, k, a, lda, tau, c,
                                         ldc, scratchpad, scratchpad_size);
}
static inline void unmtr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>>& tau,
                         sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<float>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmtr(selector.get_queue(), side, uplo, trans, m, n, a, lda, tau,
                                         c, ldc, scratchpad, scratchpad_size);
}
static inline void unmtr(backend_selector<backend::cusolver> selector, oneapi::mkl::side side,
                         oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>>& tau,
                         sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                         sycl::buffer<std::complex<double>>& scratchpad,
                         std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::unmtr(selector.get_queue(), side, uplo, trans, m, n, a, lda, tau,
                                         c, ldc, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<float>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<double>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<std::complex<double>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getrs_batch(backend_selector<backend::cusolver> selector,
                               oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                               std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                               stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(backend_selector<backend::cusolver> selector,
                               oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                               std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                               stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(backend_selector<backend::cusolver> selector,
                               oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, sycl::buffer<std::complex<float>>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                               stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void getrs_batch(backend_selector<backend::cusolver> selector,
                               oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, sycl::buffer<std::complex<double>>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                               stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                               std::int64_t stride_ipiv, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, sycl::buffer<std::complex<double>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                               std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a, ipiv,
                                               stride_ipiv, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<float>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<double>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void potrf_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda, stride_a,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void potrf_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                               std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda, stride_a,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void potrf_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda, stride_a,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void potrf_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, sycl::buffer<std::complex<double>>& a,
                               std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda, stride_a,
                                               batch_size, scratchpad, scratchpad_size);
}
static inline void potrs_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<float>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                               stride_a, b, ldb, stride_b, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void potrs_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
                               std::int64_t lda, std::int64_t stride_a, sycl::buffer<double>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                               stride_a, b, ldb, stride_b, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void potrs_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<float>>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                               stride_a, b, ldb, stride_b, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void potrs_batch(backend_selector<backend::cusolver> selector, oneapi::mkl::uplo uplo,
                               std::int64_t n, std::int64_t nrhs,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<double>>& b,
                               std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                               stride_a, b, ldb, stride_b, batch_size, scratchpad,
                                               scratchpad_size);
}
static inline void ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
                               std::int64_t lda, std::int64_t stride_a,
                               sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                               std::int64_t batch_size,
                               sycl::buffer<std::complex<float>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}
static inline void ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                               std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                               std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                               std::int64_t stride_tau, std::int64_t batch_size,
                               sycl::buffer<std::complex<double>>& scratchpad,
                               std::int64_t scratchpad_size) {
    oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda, stride_a, tau,
                                               stride_tau, batch_size, scratchpad, scratchpad_size);
}

// USM APIs

static inline sycl::event gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda, float* d,
                                float* e, std::complex<float>* tauq, std::complex<float>* taup,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq,
                                                taup, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, double* a, std::int64_t lda, double* d, double* e,
                                double* tauq, double* taup, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq,
                                                taup, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, float* a, std::int64_t lda, float* d, float* e,
                                float* tauq, float* taup, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq,
                                                taup, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gebrd(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                double* d, double* e, std::complex<double>* tauq,
                                std::complex<double>* taup, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gebrd(selector.get_queue(), m, n, a, lda, d, e, tauq,
                                                taup, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, float* a, std::int64_t lda, float* tau,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, double* a, std::int64_t lda, double* tau,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* tau, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event gerqf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gerqf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* tau, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, double* a, std::int64_t lda, double* tau,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, float* a, std::int64_t lda, float* tau,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event geqrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf(selector.get_queue(), m, n, a, lda, tau, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda,
                                std::int64_t* ipiv, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, double* a, std::int64_t lda, std::int64_t* ipiv,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, float* a, std::int64_t lda, std::int64_t* ipiv,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrf(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                std::int64_t* ipiv, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf(selector.get_queue(), m, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                                double* a, std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                                float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event getri(backend_selector<backend::cusolver> selector, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri(selector.get_queue(), n, a, lda, ipiv, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event getrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                                std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* b, std::int64_t ldb,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv,
                                                b, ldb, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                                double* a, std::int64_t lda, std::int64_t* ipiv, double* b,
                                std::int64_t ldb, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv,
                                                b, ldb, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                                float* a, std::int64_t lda, std::int64_t* ipiv, float* b,
                                std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv,
                                                b, ldb, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                                std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* b, std::int64_t ldb,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs(selector.get_queue(), trans, n, nrhs, a, lda, ipiv,
                                                b, ldb, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event gesvd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                std::int64_t n, double* a, std::int64_t lda, double* s, double* u,
                                std::int64_t ldu, double* vt, std::int64_t ldvt, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s,
                                                u, ldu, vt, ldvt, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event gesvd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                std::int64_t n, float* a, std::int64_t lda, float* s, float* u,
                                std::int64_t ldu, float* vt, std::int64_t ldvt, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s,
                                                u, ldu, vt, ldvt, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event gesvd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                std::int64_t n, std::complex<float>* a, std::int64_t lda, float* s,
                                std::complex<float>* u, std::int64_t ldu, std::complex<float>* vt,
                                std::int64_t ldvt, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s,
                                                u, ldu, vt, ldvt, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event gesvd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                double* s, std::complex<double>* u, std::int64_t ldu,
                                std::complex<double>* vt, std::int64_t ldvt,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::gesvd(selector.get_queue(), jobu, jobvt, m, n, a, lda, s,
                                                u, ldu, vt, ldvt, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event heevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, float* w, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::heevd(selector.get_queue(), jobz, uplo, n, a, lda, w,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event heevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, double* w, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::heevd(selector.get_queue(), jobz, uplo, n, a, lda, w,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hegvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                                std::int64_t ldb, float* w, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hegvd(selector.get_queue(), itype, jobz, uplo, n, a, lda,
                                                b, ldb, w, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event hegvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                                std::int64_t ldb, double* w, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hegvd(selector.get_queue(), itype, jobz, uplo, n, a, lda,
                                                b, ldb, w, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event hetrd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, float* d, float* e, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hetrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hetrd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, double* d, double* e, std::complex<double>* tau,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hetrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hetrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hetrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event hetrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::hetrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgbr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                std::int64_t k, float* a, std::int64_t lda, float* tau,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgbr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                std::int64_t k, double* a, std::int64_t lda, double* tau,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                                double* tau, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr(selector.get_queue(), m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                                float* tau, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr(selector.get_queue(), m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* tau, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgtr(selector.get_queue(), uplo, n, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                double* tau, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgtr(selector.get_queue(), uplo, n, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ormtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                float* a, std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormtr(selector.get_queue(), side, uplo, trans, m, n, a,
                                                lda, tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ormtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                double* a, std::int64_t lda, double* tau, double* c,
                                std::int64_t ldc, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormtr(selector.get_queue(), side, uplo, trans, m, n, a,
                                                lda, tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ormrq(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k, float* a,
                                std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormrq(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ormrq(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k, double* a,
                                std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormrq(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ormqr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k, double* a,
                                std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormqr(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ormqr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k, float* a,
                                std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ormqr(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event potrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potri(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potri(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potri(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potri(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potri(selector.get_queue(), uplo, n, a, lda, scratchpad,
                                                scratchpad_size, dependencies);
}
static inline sycl::event potrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, float* a,
                                std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                double* a, std::int64_t lda, double* b, std::int64_t ldb,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                                std::int64_t ldb, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                                std::int64_t ldb, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs(selector.get_queue(), uplo, n, nrhs, a, lda, b, ldb,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event syevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                double* w, double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::syevd(selector.get_queue(), jobz, uplo, n, a, lda, w,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event syevd(backend_selector<backend::cusolver> selector, oneapi::mkl::job jobz,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* w, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::syevd(selector.get_queue(), jobz, uplo, n, a, lda, w,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sygvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                double* a, std::int64_t lda, double* b, std::int64_t ldb, double* w,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sygvd(selector.get_queue(), itype, jobz, uplo, n, a, lda,
                                                b, ldb, w, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event sygvd(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                float* a, std::int64_t lda, float* b, std::int64_t ldb, float* w,
                                float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sygvd(selector.get_queue(), itype, jobz, uplo, n, a, lda,
                                                b, ldb, w, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event sytrd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                double* d, double* e, double* tau, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrd(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                float* d, float* e, float* tau, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrd(selector.get_queue(), uplo, n, a, lda, d, e, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda,
                                std::int64_t* ipiv, float* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda,
                                std::int64_t* ipiv, double* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event sytrf(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::int64_t* ipiv,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::sytrf(selector.get_queue(), uplo, n, a, lda, ipiv,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event trtrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                                std::int64_t ldb, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a,
                                                lda, b, ldb, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event trtrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                                double* a, std::int64_t lda, double* b, std::int64_t ldb,
                                double* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a,
                                                lda, b, ldb, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event trtrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, float* a,
                                std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a,
                                                lda, b, ldb, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event trtrs(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                                std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                                std::int64_t ldb, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::trtrs(selector.get_queue(), uplo, trans, diag, n, nrhs, a,
                                                lda, b, ldb, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event ungbr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                std::complex<float>* tau, std::complex<float>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungbr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungbr(selector.get_queue(), vec, m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::int64_t k, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr(selector.get_queue(), m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr(backend_selector<backend::cusolver> selector, std::int64_t m,
                                std::int64_t n, std::int64_t k, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* tau,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr(selector.get_queue(), m, n, k, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungtr(selector.get_queue(), uplo, n, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                std::int64_t lda, std::complex<double>* tau,
                                std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungtr(selector.get_queue(), uplo, n, a, lda, tau,
                                                scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event unmrq(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* c, std::int64_t ldc,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmrq(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event unmrq(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* c,
                                std::int64_t ldc, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmrq(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event unmqr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* c, std::int64_t ldc,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmqr(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event unmqr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                std::int64_t m, std::int64_t n, std::int64_t k,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* c,
                                std::int64_t ldc, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmqr(selector.get_queue(), side, trans, m, n, k, a, lda,
                                                tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event unmtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                                std::complex<float>* c, std::int64_t ldc,
                                std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmtr(selector.get_queue(), side, uplo, trans, m, n, a,
                                                lda, tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event unmtr(backend_selector<backend::cusolver> selector,
                                oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                std::complex<double>* a, std::int64_t lda,
                                std::complex<double>* tau, std::complex<double>* c,
                                std::int64_t ldc, std::complex<double>* scratchpad,
                                std::int64_t scratchpad_size,
                                const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::unmtr(selector.get_queue(), side, uplo, trans, m, n, a,
                                                lda, tau, c, ldc, scratchpad, scratchpad_size,
                                                dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, float* a, std::int64_t lda,
                                      std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      tau, stride_tau, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, double* a, std::int64_t lda,
                                      std::int64_t stride_a, double* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      tau, stride_tau, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      tau, stride_tau, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double>* tau,
                                      std::int64_t stride_tau, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      tau, stride_tau, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, float** a, std::int64_t* lda, float** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, double** a, std::int64_t* lda, double** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                                      std::complex<float>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event geqrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                                      std::complex<double>** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch(selector.get_queue(), m, n, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, float* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, double* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, float** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, double** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrf_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrf_batch(selector.get_queue(), m, n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                                      float* a, std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t* ipiv, std::int64_t stride_ipiv,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                                      double* a, std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t* ipiv, std::int64_t stride_ipiv,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t n,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, stride_a,
                                                      ipiv, stride_ipiv, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t* n,
                                      float** a, std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t* n,
                                      double** a, std::int64_t* lda, std::int64_t** ipiv,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t* n,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getri_batch(backend_selector<backend::cusolver> selector, std::int64_t* n,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::int64_t** ipiv, std::int64_t group_count,
                                      std::int64_t* group_sizes, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getri_batch(selector.get_queue(), n, a, lda, ipiv,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::transpose trans, std::int64_t n,
                                      std::int64_t nrhs, float* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, float* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(
        selector.get_queue(), trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
        batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::transpose trans, std::int64_t n,
                                      std::int64_t nrhs, double* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t* ipiv,
                                      std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                                      std::int64_t stride_b, std::int64_t batch_size,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(
        selector.get_queue(), trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
        batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(
    backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
    std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<float>* b, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size, std::complex<float>* scratchpad,
    std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(
        selector.get_queue(), trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
        batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(
    backend_selector<backend::cusolver> selector, oneapi::mkl::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
    std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<double>* b, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size, std::complex<double>* scratchpad,
    std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(
        selector.get_queue(), trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b,
        batch_size, scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::transpose* trans, std::int64_t* n,
                                      std::int64_t* nrhs, float** a, std::int64_t* lda,
                                      std::int64_t** ipiv, float** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                                      ipiv, b, ldb, group_count, group_sizes,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::transpose* trans, std::int64_t* n,
                                      std::int64_t* nrhs, double** a, std::int64_t* lda,
                                      std::int64_t** ipiv, double** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                                      ipiv, b, ldb, group_count, group_sizes,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::transpose* trans, std::int64_t* n,
                                      std::int64_t* nrhs, std::complex<float>** a,
                                      std::int64_t* lda, std::int64_t** ipiv,
                                      std::complex<float>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                                      ipiv, b, ldb, group_count, group_sizes,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event getrs_batch(
    backend_selector<backend::cusolver> selector, oneapi::mkl::transpose* trans, std::int64_t* n,
    std::int64_t* nrhs, std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
    std::complex<double>** b, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
    const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::getrs_batch(selector.get_queue(), trans, n, nrhs, a, lda,
                                                      ipiv, b, ldb, group_count, group_sizes,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                                      std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda,
                                                      stride_a, tau, stride_tau, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                                      std::int64_t stride_a, double* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda,
                                                      stride_a, tau, stride_tau, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, float** a,
                                      std::int64_t* lda, float** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event orgqr_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, double** a,
                                      std::int64_t* lda, double** tau, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch(selector.get_queue(), m, n, k, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, float* a,
                                      std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      stride_a, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, double* a,
                                      std::int64_t lda, std::int64_t stride_a,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      stride_a, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t batch_size,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      stride_a, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::int64_t batch_size,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      stride_a, batch_size, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, float** a,
                                      std::int64_t* lda, std::int64_t group_count,
                                      std::int64_t* group_sizes, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, double** a,
                                      std::int64_t* lda, std::int64_t group_count,
                                      std::int64_t* group_sizes, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrf_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrf_batch(selector.get_queue(), uplo, n, a, lda,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                      float* a, std::int64_t lda, std::int64_t stride_a, float* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, float* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      stride_a, b, ldb, stride_b, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                      double* a, std::int64_t lda, std::int64_t stride_a, double* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, double* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      stride_a, b, ldb, stride_b, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                      std::complex<float>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<float>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      stride_a, b, ldb, stride_b, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                      std::complex<double>* a, std::int64_t lda,
                                      std::int64_t stride_a, std::complex<double>* b,
                                      std::int64_t ldb, std::int64_t stride_b,
                                      std::int64_t batch_size, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      stride_a, b, ldb, stride_b, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
                                      float** a, std::int64_t* lda, float** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      float* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      b, ldb, group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
                                      double** a, std::int64_t* lda, double** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      double* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      b, ldb, group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
                                      std::complex<float>** a, std::int64_t* lda,
                                      std::complex<float>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      b, ldb, group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event potrs_batch(backend_selector<backend::cusolver> selector,
                                      oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
                                      std::complex<double>** a, std::int64_t* lda,
                                      std::complex<double>** b, std::int64_t* ldb,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::potrs_batch(selector.get_queue(), uplo, n, nrhs, a, lda,
                                                      b, ldb, group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::int64_t k, std::complex<float>* a,
                                      std::int64_t lda, std::int64_t stride_a,
                                      std::complex<float>* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, std::complex<float>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda,
                                                      stride_a, tau, stride_tau, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t m,
                                      std::int64_t n, std::int64_t k, std::complex<double>* a,
                                      std::int64_t lda, std::int64_t stride_a,
                                      std::complex<double>* tau, std::int64_t stride_tau,
                                      std::int64_t batch_size, std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda,
                                                      stride_a, tau, stride_tau, batch_size,
                                                      scratchpad, scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, std::complex<float>** a,
                                      std::int64_t* lda, std::complex<float>** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}
static inline sycl::event ungqr_batch(backend_selector<backend::cusolver> selector, std::int64_t* m,
                                      std::int64_t* n, std::int64_t* k, std::complex<double>** a,
                                      std::int64_t* lda, std::complex<double>** tau,
                                      std::int64_t group_count, std::int64_t* group_sizes,
                                      std::complex<double>* scratchpad,
                                      std::int64_t scratchpad_size,
                                      const std::vector<sycl::event>& dependencies = {}) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch(selector.get_queue(), m, n, k, a, lda, tau,
                                                      group_count, group_sizes, scratchpad,
                                                      scratchpad_size, dependencies);
}

// SCRATCHPAD APIs
template <typename fp_type>
std::int64_t gebrd_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::gebrd_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         lda);
}
template <typename fp_type>
std::int64_t gerqf_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::gerqf_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         lda);
}
template <typename fp_type>
std::int64_t geqrf_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::geqrf_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         lda);
}
template <typename fp_type>
std::int64_t gesvd_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                                   std::int64_t m, std::int64_t n, std::int64_t lda,
                                   std::int64_t ldu, std::int64_t ldvt) {
    return oneapi::mkl::lapack::cusolver::gesvd_scratchpad_size<fp_type>(
        selector.get_queue(), jobu, jobvt, m, n, lda, ldu, ldvt);
}
template <typename fp_type>
std::int64_t getrf_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::getrf_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         lda);
}
template <typename fp_type>
std::int64_t getri_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t n,
                                   std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::getri_scratchpad_size<fp_type>(selector.get_queue(), n,
                                                                         lda);
}
template <typename fp_type>
std::int64_t getrs_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                                   std::int64_t lda, std::int64_t ldb) {
    return oneapi::mkl::lapack::cusolver::getrs_scratchpad_size<fp_type>(selector.get_queue(),
                                                                         trans, n, nrhs, lda, ldb);
}
template <typename fp_type>
std::int64_t heevd_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::heevd_scratchpad_size<fp_type>(selector.get_queue(), jobz,
                                                                         uplo, n, lda);
}
template <typename fp_type>
std::int64_t hegvd_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                   oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldb) {
    return oneapi::mkl::lapack::cusolver::hegvd_scratchpad_size<fp_type>(
        selector.get_queue(), itype, jobz, uplo, n, lda, ldb);
}
template <typename fp_type>
std::int64_t hetrd_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::hetrd_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t hetrf_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::hetrf_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t orgbr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::generate vect, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::orgbr_scratchpad_size<fp_type>(selector.get_queue(), vect,
                                                                         m, n, k, lda);
}
template <typename fp_type>
std::int64_t orgtr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::orgtr_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t orgqr_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::orgqr_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         k, lda);
}
template <typename fp_type>
std::int64_t ormrq_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                   std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::ormrq_scratchpad_size<fp_type>(selector.get_queue(), side,
                                                                         trans, m, n, k, lda, ldc);
}
template <typename fp_type>
std::int64_t ormqr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                   std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::ormqr_scratchpad_size<fp_type>(selector.get_queue(), side,
                                                                         trans, m, n, k, lda, ldc);
}
template <typename fp_type>
std::int64_t ormtr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::ormtr_scratchpad_size<fp_type>(
        selector.get_queue(), side, uplo, trans, m, n, lda, ldc);
}
template <typename fp_type>
std::int64_t potrf_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::potrf_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t potrs_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                   std::int64_t lda, std::int64_t ldb) {
    return oneapi::mkl::lapack::cusolver::potrs_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, nrhs, lda, ldb);
}
template <typename fp_type>
std::int64_t potri_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::potri_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t sytrf_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::sytrf_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t syevd_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::syevd_scratchpad_size<fp_type>(selector.get_queue(), jobz,
                                                                         uplo, n, lda);
}
template <typename fp_type>
std::int64_t sygvd_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t itype,
                                   oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldb) {
    return oneapi::mkl::lapack::cusolver::sygvd_scratchpad_size<fp_type>(
        selector.get_queue(), itype, jobz, uplo, n, lda, ldb);
}
template <typename fp_type>
std::int64_t sytrd_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::sytrd_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t trtrs_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                   oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                                   std::int64_t lda, std::int64_t ldb) {
    return oneapi::mkl::lapack::cusolver::trtrs_scratchpad_size<fp_type>(
        selector.get_queue(), uplo, trans, diag, n, nrhs, lda, ldb);
}
template <typename fp_type>
std::int64_t ungbr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::generate vect, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::ungbr_scratchpad_size<fp_type>(selector.get_queue(), vect,
                                                                         m, n, k, lda);
}
template <typename fp_type>
std::int64_t ungqr_scratchpad_size(backend_selector<backend::cusolver> selector, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::ungqr_scratchpad_size<fp_type>(selector.get_queue(), m, n,
                                                                         k, lda);
}
template <typename fp_type>
std::int64_t ungtr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda) {
    return oneapi::mkl::lapack::cusolver::ungtr_scratchpad_size<fp_type>(selector.get_queue(), uplo,
                                                                         n, lda);
}
template <typename fp_type>
std::int64_t unmrq_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                   std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::unmrq_scratchpad_size<fp_type>(selector.get_queue(), side,
                                                                         trans, m, n, k, lda, ldc);
}
template <typename fp_type>
std::int64_t unmqr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                   std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::unmqr_scratchpad_size<fp_type>(selector.get_queue(), side,
                                                                         trans, m, n, k, lda, ldc);
}
template <typename fp_type>
std::int64_t unmtr_scratchpad_size(backend_selector<backend::cusolver> selector,
                                   oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t lda, std::int64_t ldc) {
    return oneapi::mkl::lapack::cusolver::unmtr_scratchpad_size<fp_type>(
        selector.get_queue(), side, uplo, trans, m, n, lda, ldc);
}
template <typename fp_type>
std::int64_t getrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t m, std::int64_t n, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_ipiv,
                                         std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::getrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, lda, stride_a, stride_ipiv, batch_size);
}
template <typename fp_type>
std::int64_t getri_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t n, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_ipiv, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::getri_batch_scratchpad_size<fp_type>(
        selector.get_queue(), n, lda, stride_a, stride_ipiv, batch_size);
}
template <typename fp_type>
std::int64_t getrs_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::transpose trans, std::int64_t n,
                                         std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_ipiv, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::getrs_batch_scratchpad_size<fp_type>(
        selector.get_queue(), trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b,
        batch_size);
}
template <typename fp_type>
std::int64_t geqrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t m, std::int64_t n, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t stride_tau,
                                         std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, lda, stride_a, stride_tau, batch_size);
}
template <typename fp_type>
std::int64_t potrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::potrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), uplo, n, lda, stride_a, batch_size);
}
template <typename fp_type>
std::int64_t potrs_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                         std::int64_t lda, std::int64_t stride_a, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::potrs_batch_scratchpad_size<fp_type>(
        selector.get_queue(), uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <typename fp_type>
std::int64_t orgqr_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t m, std::int64_t n, std::int64_t k,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <typename fp_type>
std::int64_t ungqr_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t m, std::int64_t n, std::int64_t k,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t stride_tau, std::int64_t batch_size) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <typename fp_type>
std::int64_t getrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::getrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, lda, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t getri_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::getri_batch_scratchpad_size<fp_type>(
        selector.get_queue(), n, lda, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t getrs_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::transpose* trans, std::int64_t* n,
                                         std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::getrs_batch_scratchpad_size<fp_type>(
        selector.get_queue(), trans, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t geqrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::geqrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, lda, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t orgqr_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::orgqr_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, k, lda, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t potrf_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::uplo* uplo, std::int64_t* n,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::potrf_batch_scratchpad_size<fp_type>(
        selector.get_queue(), uplo, n, lda, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t potrs_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         oneapi::mkl::uplo* uplo, std::int64_t* n,
                                         std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::potrs_batch_scratchpad_size<fp_type>(
        selector.get_queue(), uplo, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <typename fp_type>
std::int64_t ungqr_batch_scratchpad_size(backend_selector<backend::cusolver> selector,
                                         std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes) {
    return oneapi::mkl::lapack::cusolver::ungqr_batch_scratchpad_size<fp_type>(
        selector.get_queue(), m, n, k, lda, group_count, group_sizes);
}
