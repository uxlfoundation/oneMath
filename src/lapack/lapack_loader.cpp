/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/math/lapack/detail/lapack_loader.hpp"

#include "function_table_initializer.hpp"
#include "lapack/function_table.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace detail {

static oneapi::math::detail::table_initializer<domain::lapack, lapack_function_table_t>
    function_tables;

void gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& d,
           sycl::buffer<float>& e, sycl::buffer<std::complex<float>>& tauq,
           sycl::buffer<std::complex<float>>& taup, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size);
}
void gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<double>& tauq, sycl::buffer<double>& taup,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size);
}
void gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tauq, sycl::buffer<float>& taup, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size);
}
void gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tauq,
           sycl::buffer<std::complex<double>>& taup, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size);
}
void gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgerqf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgerqf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgerqf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgerqf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetri_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetri_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetri_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetri_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<float>>& b,
           std::int64_t ldb, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size);
}
void getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size);
}
void getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size);
}
void getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size);
}
void gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
           oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& s, sycl::buffer<double>& u, std::int64_t ldu,
           sycl::buffer<double>& vt, std::int64_t ldvt, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size);
}
void gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
           oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& s, sycl::buffer<float>& u, std::int64_t ldu,
           sycl::buffer<float>& vt, std::int64_t ldvt, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size);
}
void gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
           oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& s,
           sycl::buffer<std::complex<float>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size);
}
void gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
           oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& s,
           sycl::buffer<std::complex<double>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<double>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size);
}
void heevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
           oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cheevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size);
}
void heevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
           oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zheevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size);
}
void hegvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb, sycl::buffer<float>& w,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].chegvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size);
}
void hegvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zhegvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size);
}
void hetrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& d,
           sycl::buffer<float>& e, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].chetrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size);
}
void hetrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zhetrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size);
}
void hetrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].chetrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void hetrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zhetrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void orgbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sorgbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void orgbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dorgbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void orgqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dorgqr_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void orgqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sorgqr_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void orgtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sorgtr_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void orgtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dorgtr_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ormtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sormtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size);
}
void ormtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dormtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size);
}
void ormrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sormrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void ormrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dormrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void ormqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dormqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void ormqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sormqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].spotrf_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dpotrf_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cpotrf_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zpotrf_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].spotri_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dpotri_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cpotri_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zpotri_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size);
}
void potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
           std::int64_t ldb, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].spotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size);
}
void potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
           std::int64_t ldb, sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size);
}
void potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size);
}
void potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size);
}
void syevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
           oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dsyevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size);
}
void syevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
           oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& w, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ssyevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size);
}
void sygvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb, sycl::buffer<double>& w,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dsygvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size);
}
void sygvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& w,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ssygvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size);
}
void sytrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dsytrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size);
}
void sytrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ssytrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size);
}
void sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ssytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dsytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].csytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zsytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size);
}
void trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ctrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size);
}
void trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
           std::int64_t ldb, sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dtrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size);
}
void trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
           std::int64_t ldb, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].strtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size);
}
void trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
           oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].ztrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size);
}
void ungbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cungbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ungbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zungbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ungqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cungqr_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ungqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zungqr_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ungtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cungtr_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void ungtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zungtr_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size);
}
void unmrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cunmrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void unmrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zunmrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void unmqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cunmqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void unmqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zunmqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size);
}
void unmtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cunmtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size);
}
void unmtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
           oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zunmtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size);
}
sycl::event gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, float* d, float* e,
                  std::complex<float>* tauq, std::complex<float>* taup,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgebrd_usm_sycl(
        queue, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* d, double* e, double* tauq, double* taup,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgebrd_usm_sycl(
        queue, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* d, float* e, float* tauq, float* taup,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgebrd_usm_sycl(
        queue, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, double* d, double* e,
                  std::complex<double>* tauq, std::complex<double>* taup,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgebrd_usm_sycl(
        queue, m, n, a, lda, d, e, tauq, taup, scratchpad, scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, double* a,
                  std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, float* a,
                  std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getri(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                  std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<float>* b, std::int64_t ldb,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrs_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                  std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                  std::int64_t* ipiv, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrs_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                  std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t* ipiv,
                  float* b, std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrs_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                  std::int64_t n, std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<double>* b, std::int64_t ldb,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrs_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, double* a,
                  std::int64_t lda, double* s, double* u, std::int64_t ldu, double* vt,
                  std::int64_t ldvt, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s,
                                                              u, ldu, vt, ldvt, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n, float* a,
                  std::int64_t lda, float* s, float* u, std::int64_t ldu, float* vt,
                  std::int64_t ldvt, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s,
                                                              u, ldu, vt, ldvt, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, float* s, std::complex<float>* u,
                  std::int64_t ldu, std::complex<float>* vt, std::int64_t ldvt,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s,
                                                              u, ldu, vt, ldvt, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::jobsvd jobu,
                  oneapi::math::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, double* s, std::complex<double>* u,
                  std::int64_t ldu, std::complex<double>* vt, std::int64_t ldvt,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s,
                                                              u, ldu, vt, ldvt, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event heevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  float* w, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cheevd_usm_sycl(
        queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event heevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, double* w, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zheevd_usm_sycl(
        queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event hegvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                  std::int64_t ldb, float* w, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].chegvd_usm_sycl(
        queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event hegvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, double* w, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zhegvd_usm_sycl(
        queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event hetrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, float* d, float* e,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].chetrd_usm_sycl(
        queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event hetrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, double* d, double* e,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zhetrd_usm_sycl(
        queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event hetrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].chetrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event hetrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zhetrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sorgbr_usm_sycl(
        queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dorgbr_usm_sycl(
        queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dorgqr_usm_sycl(
        queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sorgqr_usm_sycl(
        queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sorgtr_usm_sycl(
        queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event orgtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dorgtr_usm_sycl(
        queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ormtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, float* a, std::int64_t lda, float* tau, float* c,
                  std::int64_t ldc, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sormtr_usm_sycl(queue, side, uplo, trans, m, n, a,
                                                              lda, tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ormtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, double* a, std::int64_t lda, double* tau, double* c,
                  std::int64_t ldc, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dormtr_usm_sycl(queue, side, uplo, trans, m, n, a,
                                                              lda, tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ormrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  float* a, std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sormrq_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ormrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  double* a, std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dormrq_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ormqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  double* a, std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dormqr_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ormqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  float* a, std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sormqr_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potri(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda, float* b,
                  std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrs_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda, double* b,
                  std::int64_t ldb, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrs_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* b, std::int64_t ldb, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrs_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* b, std::int64_t ldb, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrs_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event syevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, double* a, std::int64_t lda, double* w,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dsyevd_usm_sycl(
        queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event syevd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::job jobz,
                  oneapi::math::uplo uplo, std::int64_t n, float* a, std::int64_t lda, float* w,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ssyevd_usm_sycl(
        queue, jobz, uplo, n, a, lda, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event sygvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* b, std::int64_t ldb, double* w, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dsygvd_usm_sycl(
        queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event sygvd(oneapi::math::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::math::job jobz, oneapi::math::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* b, std::int64_t ldb, float* w, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ssygvd_usm_sycl(
        queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* d, double* e, double* tau,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dsytrd_usm_sycl(
        queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrd(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* d, float* e, float* tau,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ssytrd_usm_sycl(
        queue, uplo, n, a, lda, d, e, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ssytrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, std::int64_t* ipiv,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dsytrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].csytrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zsytrf_usm_sycl(
        queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
                  std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* b, std::int64_t ldb, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ctrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a,
                                                              lda, b, ldb, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
                  std::int64_t nrhs, double* a, std::int64_t lda, double* b, std::int64_t ldb,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dtrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a,
                                                              lda, b, ldb, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
                  std::int64_t nrhs, float* a, std::int64_t lda, float* b, std::int64_t ldb,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].strtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a,
                                                              lda, b, ldb, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n,
                  std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* b, std::int64_t ldb, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].ztrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a,
                                                              lda, b, ldb, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event ungbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cungbr_usm_sycl(
        queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungbr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zungbr_usm_sycl(
        queue, vec, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cungqr_usm_sycl(
        queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungqr(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zungqr_usm_sycl(
        queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cungtr_usm_sycl(
        queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zungtr_usm_sycl(
        queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size, dependencies);
}
sycl::event unmrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* c, std::int64_t ldc, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cunmrq_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event unmrq(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* c, std::int64_t ldc, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zunmrq_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event unmqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* c, std::int64_t ldc, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cunmqr_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event unmqr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* c, std::int64_t ldc, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zunmqr_usm_sycl(queue, side, trans, m, n, k, a, lda,
                                                              tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event unmtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* c, std::int64_t ldc,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cunmtr_usm_sycl(queue, side, uplo, trans, m, n, a,
                                                              lda, tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
sycl::event unmtr(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
                  oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* c, std::int64_t ldc,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zunmtr_usm_sycl(queue, side, uplo, trans, m, n, a,
                                                              lda, tau, c, ldc, scratchpad,
                                                              scratchpad_size, dependencies);
}
void geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgeqrf_batch_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgeqrf_batch_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgeqrf_batch_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgeqrf_batch_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetri_batch_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetri_batch_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetri_batch_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetri_batch_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a,
                                                         ipiv, stride_ipiv, b, ldb, stride_b,
                                                         batch_size, scratchpad, scratchpad_size);
}
void getrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a,
                                                         ipiv, stride_ipiv, b, ldb, stride_b,
                                                         batch_size, scratchpad, scratchpad_size);
}
void getrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a,
                                                         ipiv, stride_ipiv, b, ldb, stride_b,
                                                         batch_size, scratchpad, scratchpad_size);
}
void getrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a,
                                                         ipiv, stride_ipiv, b, ldb, stride_b,
                                                         batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sgetrf_batch_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dgetrf_batch_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cgetrf_batch_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zgetrf_batch_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size);
}
void orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].sorgqr_batch_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dorgqr_batch_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].spotrf_batch_sycl(queue, uplo, n, a, lda, stride_a,
                                                         batch_size, scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a,
                                                         batch_size, scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a,
                                                         batch_size, scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a,
                                                         batch_size, scratchpad, scratchpad_size);
}
void potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<float>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].spotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size);
}
void potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<double>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].dpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size);
}
void potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::complex<float>>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size);
}
void potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::complex<double>>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size);
}
void ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].cungqr_batch_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
void ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[{ libkey, queue }].zungqr_batch_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        float* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgeqrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        double* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgeqrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgeqrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgeqrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrf_batch_usm_sycl(
        queue, m, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetri_batch_usm_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetri_batch_usm_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetri_batch_usm_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t n,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetri_batch_usm_sycl(
        queue, n, a, lda, stride_a, ipiv, stride_ipiv, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, float* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<float>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose trans, std::int64_t n, std::int64_t nrhs,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<double>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                        std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sorgqr_batch_usm_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                        std::int64_t stride_a, double* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dorgqr_batch_usm_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda,
                        std::int64_t stride_a, float* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrs_batch_usm_sycl(
        queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                        std::int64_t stride_a, double* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrs_batch_usm_sycl(
        queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrs_batch_usm_sycl(
        queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrs_batch_usm_sycl(
        queue, uplo, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cungqr_batch_usm_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zungqr_batch_usm_sycl(
        queue, m, n, k, a, lda, stride_a, tau, stride_tau, batch_size, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, float** a, std::int64_t* lda, float** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgeqrf_group_usm_sycl(
        queue, m, n, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, double** a, std::int64_t* lda, double** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgeqrf_group_usm_sycl(
        queue, m, n, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::complex<float>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgeqrf_group_usm_sycl(
        queue, m, n, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event geqrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::complex<double>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgeqrf_group_usm_sycl(
        queue, m, n, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, float** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrf_group_usm_sycl(
        queue, m, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, double** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrf_group_usm_sycl(
        queue, m, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrf_group_usm_sycl(
        queue, m, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrf_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrf_group_usm_sycl(
        queue, m, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetri_group_usm_sycl(
        queue, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* n,
                        double** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetri_group_usm_sycl(
        queue, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetri_group_usm_sycl(
        queue, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getri_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetri_group_usm_sycl(
        queue, n, a, lda, ipiv, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        float** a, std::int64_t* lda, std::int64_t** ipiv, float** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sgetrs_group_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, group_count, group_sizes, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        double** a, std::int64_t* lda, std::int64_t** ipiv, double** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dgetrs_group_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, group_count, group_sizes, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::complex<float>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cgetrs_group_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, group_count, group_sizes, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::math::device libkey, sycl::queue& queue,
                        oneapi::math::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::complex<double>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zgetrs_group_usm_sycl(
        queue, trans, n, nrhs, a, lda, ipiv, b, ldb, group_count, group_sizes, scratchpad,
        scratchpad_size, dependencies);
}
sycl::event orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, float** a, std::int64_t* lda, float** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].sorgqr_group_usm_sycl(
        queue, m, n, k, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event orgqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, double** a, std::int64_t* lda,
                        double** tau, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dorgqr_group_usm_sycl(
        queue, m, n, k, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, float** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrf_group_usm_sycl(
        queue, uplo, n, a, lda, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, double** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrf_group_usm_sycl(
        queue, uplo, n, a, lda, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrf_group_usm_sycl(
        queue, uplo, n, a, lda, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrf_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrf_group_usm_sycl(
        queue, uplo, n, a, lda, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, float** a, std::int64_t* lda,
                        float** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].spotrs_group_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, double** a, std::int64_t* lda,
                        double** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].dpotrs_group_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, std::complex<float>** a,
                        std::int64_t* lda, std::complex<float>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cpotrs_group_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event potrs_batch(oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, std::complex<double>** a,
                        std::int64_t* lda, std::complex<double>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zpotrs_group_usm_sycl(
        queue, uplo, n, nrhs, a, lda, b, ldb, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, std::complex<float>** a,
                        std::int64_t* lda, std::complex<float>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].cungqr_group_usm_sycl(
        queue, m, n, k, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}
sycl::event ungqr_batch(oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, std::complex<double>** a,
                        std::int64_t* lda, std::complex<double>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[{ libkey, queue }].zungqr_group_usm_sycl(
        queue, m, n, k, a, lda, tau, group_count, group_sizes, scratchpad, scratchpad_size,
        dependencies);
}

template <>
std::int64_t gebrd_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].zgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].zgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].zgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gesvd_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                                          std::int64_t m, std::int64_t n, std::int64_t lda,
                                          std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[{ libkey, queue }].sgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n,
                                                                          lda, ldu, ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::jobsvd jobu, oneapi::math::jobsvd jobvt,
                                           std::int64_t m, std::int64_t n, std::int64_t lda,
                                           std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[{ libkey, queue }].dgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n,
                                                                          lda, ldu, ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::math::jobsvd jobu,
                                                        oneapi::math::jobsvd jobvt, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda,
                                                        std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[{ libkey, queue }].cgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n,
                                                                          lda, ldu, ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::jobsvd jobu,
                                                         oneapi::math::jobsvd jobvt, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda,
                                                         std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[{ libkey, queue }].zgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n,
                                                                          lda, ldu, ldvt);
}
template <>
std::int64_t getrf_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].zgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t n,
                                                        std::int64_t lda) {
    return function_tables[{ libkey, queue }].cgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getrs_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::transpose trans, std::int64_t n,
                                          std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].sgetrs_scratchpad_size_sycl(queue, trans, n, nrhs,
                                                                          lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::transpose trans, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].dgetrs_scratchpad_size_sycl(queue, trans, n, nrhs,
                                                                          lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::math::transpose trans,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].cgetrs_scratchpad_size_sycl(queue, trans, n, nrhs,
                                                                          lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::transpose trans,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].zgetrs_scratchpad_size_sycl(queue, trans, n, nrhs,
                                                                          lda, ldb);
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::job jobz,
                                                        oneapi::math::uplo uplo, std::int64_t n,
                                                        std::int64_t lda) {
    return function_tables[{ libkey, queue }].cheevd_scratchpad_size_sycl(queue, jobz, uplo, n,
                                                                          lda);
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, oneapi::math::job jobz,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zheevd_scratchpad_size_sycl(queue, jobz, uplo, n,
                                                                          lda);
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t itype,
                                                        oneapi::math::job jobz,
                                                        oneapi::math::uplo uplo, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].chegvd_scratchpad_size_sycl(queue, itype, jobz, uplo,
                                                                          n, lda, ldb);
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t itype,
                                                         oneapi::math::job jobz,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].zhegvd_scratchpad_size_sycl(queue, itype, jobz, uplo,
                                                                          n, lda, ldb);
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].chetrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zhetrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].chetrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zhetrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgbr_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::generate vect, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda) {
    return function_tables[{ libkey, queue }].sorgbr_scratchpad_size_sycl(queue, vect, m, n, k,
                                                                          lda);
}
template <>
std::int64_t orgbr_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::generate vect, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dorgbr_scratchpad_size_sycl(queue, vect, m, n, k,
                                                                          lda);
}
template <>
std::int64_t orgtr_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].sorgtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgtr_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dorgtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgqr_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].sorgqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t orgqr_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dorgqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ormrq_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::side side, oneapi::math::transpose trans,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].sormrq_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t ormrq_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::side side, oneapi::math::transpose trans,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].dormrq_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t ormqr_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::side side, oneapi::math::transpose trans,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].sormqr_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t ormqr_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::side side, oneapi::math::transpose trans,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].dormqr_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t ormtr_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::side side, oneapi::math::uplo uplo,
                                          oneapi::math::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].sormtr_scratchpad_size_sycl(queue, side, uplo, trans,
                                                                          m, n, lda, ldc);
}
template <>
std::int64_t ormtr_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::side side, oneapi::math::uplo uplo,
                                           oneapi::math::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].dormtr_scratchpad_size_sycl(queue, side, uplo, trans,
                                                                          m, n, lda, ldc);
}
template <>
std::int64_t potrf_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].spotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrs_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].spotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda,
                                                                          ldb);
}
template <>
std::int64_t potrs_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].dpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda,
                                                                          ldb);
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].cpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda,
                                                                          ldb);
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t nrhs, std::int64_t lda,
                                                         std::int64_t ldb) {
    return function_tables[{ libkey, queue }].zpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda,
                                                                          ldb);
}
template <>
std::int64_t potri_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].spotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].ssytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dsytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].csytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zsytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t syevd_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::job jobz, oneapi::math::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].ssyevd_scratchpad_size_sycl(queue, jobz, uplo, n,
                                                                          lda);
}
template <>
std::int64_t syevd_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::job jobz, oneapi::math::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].dsyevd_scratchpad_size_sycl(queue, jobz, uplo, n,
                                                                          lda);
}
template <>
std::int64_t sygvd_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          std::int64_t itype, oneapi::math::job jobz,
                                          oneapi::math::uplo uplo, std::int64_t n, std::int64_t lda,
                                          std::int64_t ldb) {
    return function_tables[{ libkey, queue }].ssygvd_scratchpad_size_sycl(queue, itype, jobz, uplo,
                                                                          n, lda, ldb);
}
template <>
std::int64_t sygvd_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           std::int64_t itype, oneapi::math::job jobz,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].dsygvd_scratchpad_size_sycl(queue, itype, jobz, uplo,
                                                                          n, lda, ldb);
}
template <>
std::int64_t sytrd_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[{ libkey, queue }].ssytrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrd_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[{ libkey, queue }].dsytrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t trtrs_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                          oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                          oneapi::math::diag diag, std::int64_t n,
                                          std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].strtrs_scratchpad_size_sycl(queue, uplo, trans, diag,
                                                                          n, nrhs, lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                           oneapi::math::uplo uplo, oneapi::math::transpose trans,
                                           oneapi::math::diag diag, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].dtrtrs_scratchpad_size_sycl(queue, uplo, trans, diag,
                                                                          n, nrhs, lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        oneapi::math::transpose trans,
                                                        oneapi::math::diag diag, std::int64_t n,
                                                        std::int64_t nrhs, std::int64_t lda,
                                                        std::int64_t ldb) {
    return function_tables[{ libkey, queue }].ctrtrs_scratchpad_size_sycl(queue, uplo, trans, diag,
                                                                          n, nrhs, lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo,
    oneapi::math::transpose trans, oneapi::math::diag diag, std::int64_t n, std::int64_t nrhs,
    std::int64_t lda, std::int64_t ldb) {
    return function_tables[{ libkey, queue }].ztrtrs_scratchpad_size_sycl(queue, uplo, trans, diag,
                                                                          n, nrhs, lda, ldb);
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::math::generate vect, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    return function_tables[{ libkey, queue }].cungbr_scratchpad_size_sycl(queue, vect, m, n, k,
                                                                          lda);
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::generate vect,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda) {
    return function_tables[{ libkey, queue }].zungbr_scratchpad_size_sycl(queue, vect, m, n, k,
                                                                          lda);
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    return function_tables[{ libkey, queue }].cungqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t k,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zungqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[{ libkey, queue }].cungtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::math::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[{ libkey, queue }].zungtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::side side,
                                                        oneapi::math::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    return function_tables[{ libkey, queue }].cunmrq_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
    oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t ldc) {
    return function_tables[{ libkey, queue }].zunmrq_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::side side,
                                                        oneapi::math::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    return function_tables[{ libkey, queue }].cunmqr_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
    oneapi::math::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
    std::int64_t ldc) {
    return function_tables[{ libkey, queue }].zunmqr_scratchpad_size_sycl(queue, side, trans, m, n,
                                                                          k, lda, ldc);
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                        sycl::queue& queue, oneapi::math::side side,
                                                        oneapi::math::uplo uplo,
                                                        oneapi::math::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].cunmtr_scratchpad_size_sycl(queue, side, uplo, trans,
                                                                          m, n, lda, ldc);
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::side side,
    oneapi::math::uplo uplo, oneapi::math::transpose trans, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t ldc) {
    return function_tables[{ libkey, queue }].zunmtr_scratchpad_size_sycl(queue, side, uplo, trans,
                                                                          m, n, lda, ldc);
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].sgetrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dgetrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cgetrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zgetrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].sgetri_batch_scratchpad_size_sycl(
        queue, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dgetri_batch_scratchpad_size_sycl(
        queue, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cgetri_batch_scratchpad_size_sycl(
        queue, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t n, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zgetri_batch_scratchpad_size_sycl(
        queue, n, lda, stride_a, stride_ipiv, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::transpose trans, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t ldb, std::int64_t stride_b,
                                                std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].sgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::transpose trans, std::int64_t n,
                                                 std::int64_t nrhs, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t ldb, std::int64_t stride_b,
                                                 std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv,
    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv,
    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_tau,
                                                std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].sgeqrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_tau,
                                                 std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dgeqrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cgeqrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zgeqrf_batch_scratchpad_size_sycl(
        queue, m, n, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::uplo uplo, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].spotrf_batch_scratchpad_size_sycl(
        queue, uplo, n, lda, stride_a, batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::uplo uplo, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dpotrf_batch_scratchpad_size_sycl(
        queue, uplo, n, lda, stride_a, batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cpotrf_batch_scratchpad_size_sycl(
        queue, uplo, n, lda, stride_a, batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zpotrf_batch_scratchpad_size_sycl(
        queue, uplo, n, lda, stride_a, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::uplo uplo, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t ldb,
                                                std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].spotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::uplo uplo, std::int64_t n,
                                                 std::int64_t nrhs, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t ldb,
                                                 std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo uplo, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].sorgqr_batch_scratchpad_size_sycl(
        queue, m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].dorgqr_batch_scratchpad_size_sycl(
        queue, m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].cungqr_batch_scratchpad_size_sycl(
        queue, m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[{ libkey, queue }].zungqr_batch_scratchpad_size_sycl(
        queue, m, n, k, lda, stride_a, stride_tau, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].sgetrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dgetrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                              sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cgetrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                               sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zgetrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].sgetri_group_scratchpad_size_sycl(
        queue, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dgetri_group_scratchpad_size_sycl(
        queue, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                              sycl::queue& queue, std::int64_t* n,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cgetri_group_scratchpad_size_sycl(
        queue, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                               sycl::queue& queue, std::int64_t* n,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zgetri_group_scratchpad_size_sycl(
        queue, n, lda, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::transpose* trans, std::int64_t* n,
                                                std::int64_t* nrhs, std::int64_t* lda,
                                                std::int64_t* ldb, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].sgetrs_group_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::transpose* trans, std::int64_t* n,
                                                 std::int64_t* nrhs, std::int64_t* lda,
                                                 std::int64_t* ldb, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dgetrs_group_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose* trans,
    std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb,
    std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cgetrs_group_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::transpose* trans,
    std::int64_t* n, std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb,
    std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zgetrs_group_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].sgeqrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dgeqrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(oneapi::math::device libkey,
                                                              sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cgeqrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(oneapi::math::device libkey,
                                                               sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zgeqrf_group_scratchpad_size_sycl(
        queue, m, n, lda, group_count, group_sizes);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].sorgqr_group_scratchpad_size_sycl(
        queue, m, n, k, lda, group_count, group_sizes);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dorgqr_group_scratchpad_size_sycl(
        queue, m, n, k, lda, group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::uplo* uplo, std::int64_t* n,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].spotrf_group_scratchpad_size_sycl(
        queue, uplo, n, lda, group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::uplo* uplo, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dpotrf_group_scratchpad_size_sycl(
        queue, uplo, n, lda, group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
    std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cpotrf_group_scratchpad_size_sycl(
        queue, uplo, n, lda, group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
    std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zpotrf_group_scratchpad_size_sycl(
        queue, uplo, n, lda, group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(oneapi::math::device libkey, sycl::queue& queue,
                                                oneapi::math::uplo* uplo, std::int64_t* n,
                                                std::int64_t* nrhs, std::int64_t* lda,
                                                std::int64_t* ldb, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].spotrs_group_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(oneapi::math::device libkey, sycl::queue& queue,
                                                 oneapi::math::uplo* uplo, std::int64_t* n,
                                                 std::int64_t* nrhs, std::int64_t* lda,
                                                 std::int64_t* ldb, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].dpotrs_group_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cpotrs_group_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, oneapi::math::uplo* uplo, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zpotrs_group_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, ldb, group_count, group_sizes);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m, std::int64_t* n,
    std::int64_t* k, std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].cungqr_group_scratchpad_size_sycl(
        queue, m, n, k, lda, group_count, group_sizes);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    oneapi::math::device libkey, sycl::queue& queue, std::int64_t* m, std::int64_t* n,
    std::int64_t* k, std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[{ libkey, queue }].zungqr_group_scratchpad_size_sycl(
        queue, m, n, k, lda, group_count, group_sizes);
}

} //namespace detail
} //namespace lapack
} //namespace math
} //namespace oneapi
