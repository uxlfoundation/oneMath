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

#include "oneapi/mkl/lapack/detail/lapack_loader.hpp"

#include "function_table_initializer.hpp"
#include "lapack/function_table.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace detail {

static oneapi::mkl::detail::table_initializer<domain::lapack, lapack_function_table_t>
    function_tables;

void gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& d,
           sycl::buffer<float>& e, sycl::buffer<std::complex<float>>& tauq,
           sycl::buffer<std::complex<float>>& taup, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                                        scratchpad_size);
}
void gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<double>& tauq, sycl::buffer<double>& taup,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                                        scratchpad_size);
}
void gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tauq, sycl::buffer<float>& taup, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                                        scratchpad_size);
}
void gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tauq,
           sycl::buffer<std::complex<double>>& taup, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zgebrd_sycl(queue, m, n, a, lda, d, e, tauq, taup, scratchpad,
                                        scratchpad_size);
}
void gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sgerqf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgerqf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cgerqf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zgerqf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zgeqrf_sycl(queue, m, n, a, lda, tau, scratchpad, scratchpad_size);
}
void getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zgetrf_sycl(queue, m, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cgetri_sycl(queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dgetri_sycl(queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sgetri_sycl(queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zgetri_sycl(queue, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<float>>& b,
           std::int64_t ldb, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                                        scratchpad_size);
}
void getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                                        scratchpad_size);
}
void getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                                        scratchpad_size);
}
void getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zgetrs_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb, scratchpad,
                                        scratchpad_size);
}
void gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
           oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& s, sycl::buffer<double>& u, std::int64_t ldu,
           sycl::buffer<double>& vt, std::int64_t ldvt, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                        scratchpad, scratchpad_size);
}
void gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
           oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& s, sycl::buffer<float>& u, std::int64_t ldu,
           sycl::buffer<float>& vt, std::int64_t ldvt, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                        scratchpad, scratchpad_size);
}
void gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
           oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& s,
           sycl::buffer<std::complex<float>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                        scratchpad, scratchpad_size);
}
void gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
           oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& s,
           sycl::buffer<std::complex<double>>& u, std::int64_t ldu,
           sycl::buffer<std::complex<double>>& vt, std::int64_t ldvt,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zgesvd_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                        scratchpad, scratchpad_size);
}
void heevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
           oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cheevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                        scratchpad_size);
}
void heevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
           oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zheevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                        scratchpad_size);
}
void hegvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb, sycl::buffer<float>& w,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].chegvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                                        scratchpad_size);
}
void hegvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb, sycl::buffer<double>& w,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zhegvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                                        scratchpad_size);
}
void hetrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<float>& d,
           sycl::buffer<float>& e, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].chetrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                        scratchpad_size);
}
void hetrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zhetrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                        scratchpad_size);
}
void hetrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].chetrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void hetrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zhetrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void orgbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sorgbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                        scratchpad_size);
}
void orgbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dorgbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                        scratchpad_size);
}
void orgqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dorgqr_sycl(queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
void orgqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sorgqr_sycl(queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
void orgtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sorgtr_sycl(queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
void orgtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dorgtr_sycl(queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
void ormtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sormtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void ormtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dormtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void ormrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sormrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void ormrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dormrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void ormqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
           sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dormqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void ormqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
           sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].sormqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].spotrf_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dpotrf_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cpotrf_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zpotrf_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].spotri_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dpotri_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cpotri_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zpotri_sycl(queue, uplo, n, a, lda, scratchpad, scratchpad_size);
}
void potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
           std::int64_t ldb, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].spotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                        scratchpad_size);
}
void potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
           std::int64_t ldb, sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                        scratchpad_size);
}
void potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                        scratchpad_size);
}
void potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zpotrs_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                        scratchpad_size);
}
void syevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
           oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dsyevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                        scratchpad_size);
}
void syevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
           oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& w, sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].ssyevd_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                        scratchpad_size);
}
void sygvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb, sycl::buffer<double>& w,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dsygvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                                        scratchpad_size);
}
void sygvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& w,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].ssygvd_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w, scratchpad,
                                        scratchpad_size);
}
void sytrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
           sycl::buffer<double>& e, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].dsytrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                        scratchpad_size);
}
void sytrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
           sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].ssytrd_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                        scratchpad_size);
}
void sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].ssytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dsytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].csytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zsytrf_sycl(queue, uplo, n, a, lda, ipiv, scratchpad, scratchpad_size);
}
void trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].ctrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                                        scratchpad, scratchpad_size);
}
void trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dtrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                                        scratchpad, scratchpad_size);
}
void trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].strtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                                        scratchpad, scratchpad_size);
}
void trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
           oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].ztrtrs_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b, ldb,
                                        scratchpad, scratchpad_size);
}
void ungbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cungbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                        scratchpad_size);
}
void ungbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
           std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
           std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zungbr_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                        scratchpad_size);
}
void ungqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cungqr_sycl(queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
void ungqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
           std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zungqr_sycl(queue, m, n, k, a, lda, tau, scratchpad, scratchpad_size);
}
void ungtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cungtr_sycl(queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
void ungtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zungtr_sycl(queue, uplo, n, a, lda, tau, scratchpad, scratchpad_size);
}
void unmrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cunmrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void unmrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zunmrq_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void unmqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cunmqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void unmqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zunmqr_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void unmtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].cunmtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
void unmtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    function_tables[libkey].zunmtr_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c, ldc,
                                        scratchpad, scratchpad_size);
}
sycl::event gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, float* d, float* e,
                  std::complex<float>* tauq, std::complex<float>* taup,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgebrd_usm_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* d, double* e, double* tauq, double* taup,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgebrd_usm_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* d, float* e, float* tauq, float* taup,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgebrd_usm_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event gebrd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, double* d, double* e,
                  std::complex<double>* tauq, std::complex<double>* taup,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgebrd_usm_sycl(queue, m, n, a, lda, d, e, tauq, taup,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event gerqf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgerqf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event geqrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgeqrf_usm_sycl(queue, m, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  double* a, std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getrf(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrf_usm_sycl(queue, m, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, double* a,
                  std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, float* a,
                  std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getri(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetri_usm_sycl(queue, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<float>* b, std::int64_t ldb,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrs_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                  std::int64_t* ipiv, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrs_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda, std::int64_t* ipiv,
                  float* b, std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrs_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::int64_t* ipiv, std::complex<double>* b, std::int64_t ldb,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrs_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b, ldb,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, double* a,
                  std::int64_t lda, double* s, double* u, std::int64_t ldu, double* vt,
                  std::int64_t ldvt, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, float* a,
                  std::int64_t lda, float* s, float* u, std::int64_t ldu, float* vt,
                  std::int64_t ldvt, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, float* s, std::complex<float>* u, std::int64_t ldu,
                  std::complex<float>* vt, std::int64_t ldvt, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size, dependencies);
}
sycl::event gesvd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, double* s, std::complex<double>* u,
                  std::int64_t ldu, std::complex<double>* vt, std::int64_t ldvt,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgesvd_usm_sycl(queue, jobu, jobvt, m, n, a, lda, s, u, ldu, vt,
                                                   ldvt, scratchpad, scratchpad_size, dependencies);
}
sycl::event heevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  float* w, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cheevd_usm_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event heevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  double* w, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zheevd_usm_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event hegvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* b,
                  std::int64_t ldb, float* w, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].chegvd_usm_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event hegvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, double* w, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zhegvd_usm_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event hetrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, float* d, float* e,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].chetrd_usm_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event hetrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, double* d, double* e,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zhetrd_usm_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event hetrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].chetrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event hetrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zhetrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sorgbr_usm_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dorgbr_usm_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dorgqr_usm_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sorgqr_usm_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* tau, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sorgtr_usm_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event orgtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* tau, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dorgtr_usm_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ormtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, float* a, std::int64_t lda, float* tau, float* c,
                  std::int64_t ldc, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sormtr_usm_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size, dependencies);
}
sycl::event ormtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, double* a, std::int64_t lda, double* tau, double* c,
                  std::int64_t ldc, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dormtr_usm_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size, dependencies);
}
sycl::event ormrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  float* a, std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sormrq_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event ormrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  double* a, std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dormrq_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event ormqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  double* a, std::int64_t lda, double* tau, double* c, std::int64_t ldc,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dormqr_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event ormqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  float* a, std::int64_t lda, float* tau, float* c, std::int64_t ldc,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sormqr_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrf_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potri(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotri_usm_sycl(queue, uplo, n, a, lda, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda, float* b,
                  std::int64_t ldb, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrs_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda, double* b,
                  std::int64_t ldb, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrs_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* b, std::int64_t ldb, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrs_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event potrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* b, std::int64_t ldb, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrs_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event syevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, double* a, std::int64_t lda, double* w,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dsyevd_usm_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event syevd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, float* a, std::int64_t lda, float* w,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ssyevd_usm_sycl(queue, jobz, uplo, n, a, lda, w, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sygvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, double* a,
                  std::int64_t lda, double* b, std::int64_t ldb, double* w, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dsygvd_usm_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event sygvd(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, float* a,
                  std::int64_t lda, float* b, std::int64_t ldb, float* w, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ssygvd_usm_sycl(queue, itype, jobz, uplo, n, a, lda, b, ldb, w,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event sytrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, double* d, double* e, double* tau,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dsytrd_usm_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sytrd(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, float* d, float* e, float* tau,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ssytrd_usm_sycl(queue, uplo, n, a, lda, d, e, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, float* a, std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ssytrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, double* a, std::int64_t lda, std::int64_t* ipiv,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dsytrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].csytrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event sytrf(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zsytrf_usm_sycl(queue, uplo, n, a, lda, ipiv, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                  std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* b, std::int64_t ldb, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ctrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                  std::int64_t nrhs, double* a, std::int64_t lda, double* b, std::int64_t ldb,
                  double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dtrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                  std::int64_t nrhs, float* a, std::int64_t lda, float* b, std::int64_t ldb,
                  float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].strtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event trtrs(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  oneapi::mkl::transpose trans, oneapi::mkl::diag diag, std::int64_t n,
                  std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* b, std::int64_t ldb, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].ztrtrs_usm_sycl(queue, uplo, trans, diag, n, nrhs, a, lda, b,
                                                   ldb, scratchpad, scratchpad_size, dependencies);
}
sycl::event ungbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cungbr_usm_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ungbr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zungbr_usm_sycl(queue, vec, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ungqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cungqr_usm_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ungqr(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                  std::int64_t k, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zungqr_usm_sycl(queue, m, n, k, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ungtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cungtr_usm_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event ungtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zungtr_usm_sycl(queue, uplo, n, a, lda, tau, scratchpad,
                                                   scratchpad_size, dependencies);
}
sycl::event unmrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* c, std::int64_t ldc, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cunmrq_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event unmrq(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* c, std::int64_t ldc, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zunmrq_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event unmqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<float>* a, std::int64_t lda, std::complex<float>* tau,
                  std::complex<float>* c, std::int64_t ldc, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cunmqr_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event unmqr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* tau,
                  std::complex<double>* c, std::int64_t ldc, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zunmqr_usm_sycl(queue, side, trans, m, n, k, a, lda, tau, c, ldc,
                                                   scratchpad, scratchpad_size, dependencies);
}
sycl::event unmtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, std::complex<float>* a, std::int64_t lda,
                  std::complex<float>* tau, std::complex<float>* c, std::int64_t ldc,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cunmtr_usm_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size, dependencies);
}
sycl::event unmtr(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, std::complex<double>* a, std::int64_t lda,
                  std::complex<double>* tau, std::complex<double>* c, std::int64_t ldc,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zunmtr_usm_sycl(queue, side, uplo, trans, m, n, a, lda, tau, c,
                                                   ldc, scratchpad, scratchpad_size, dependencies);
}
void geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sgeqrf_batch_sycl(queue, m, n, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dgeqrf_batch_sycl(queue, m, n, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<float>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].cgeqrf_batch_sycl(queue, m, n, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::complex<double>>& tau, std::int64_t stride_tau,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].zgeqrf_batch_sycl(queue, m, n, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].sgetri_batch_sycl(queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].dgetri_batch_sycl(queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].cgetri_batch_sycl(queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].zgetri_batch_sycl(queue, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<float>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].sgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                                              stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                                              scratchpad_size);
}
void getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 sycl::buffer<double>& b, std::int64_t ldb, std::int64_t stride_b,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].dgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                                              stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                                              scratchpad_size);
}
void getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                                              stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                                              scratchpad_size);
}
void getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                 std::int64_t stride_ipiv, sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zgetrs_batch_sycl(queue, trans, n, nrhs, a, lda, stride_a, ipiv,
                                              stride_ipiv, b, ldb, stride_b, batch_size, scratchpad,
                                              scratchpad_size);
}
void getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].sgetrf_batch_sycl(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].dgetrf_batch_sycl(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<float>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<float>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].cgetrf_batch_sycl(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 sycl::buffer<std::complex<double>>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                 std::int64_t batch_size, sycl::buffer<std::complex<double>>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].zgetrf_batch_sycl(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                                              batch_size, scratchpad, scratchpad_size);
}
void orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<float>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].sorgqr_batch_sycl(queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 sycl::buffer<double>& tau, std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].dorgqr_batch_sycl(queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].spotrf_batch_sycl(queue, uplo, n, a, lda, stride_a, batch_size,
                                              scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                 std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].dpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a, batch_size,
                                              scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a, batch_size,
                                              scratchpad, scratchpad_size);
}
void potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zpotrf_batch_sycl(queue, uplo, n, a, lda, stride_a, batch_size,
                                              scratchpad, scratchpad_size);
}
void potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<float>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].spotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                                              stride_b, batch_size, scratchpad, scratchpad_size);
}
void potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<double>& b, std::int64_t ldb,
                 std::int64_t stride_b, std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                 std::int64_t scratchpad_size) {
    function_tables[libkey].dpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                                              stride_b, batch_size, scratchpad, scratchpad_size);
}
void potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::complex<float>>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                                              stride_b, batch_size, scratchpad, scratchpad_size);
}
void potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                 std::int64_t n, std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                 std::int64_t lda, std::int64_t stride_a, sycl::buffer<std::complex<double>>& b,
                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zpotrs_batch_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b, ldb,
                                              stride_b, batch_size, scratchpad, scratchpad_size);
}
void ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].cungqr_batch_sycl(queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
void ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
                 std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                 std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                 std::int64_t stride_tau, std::int64_t batch_size,
                 sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    function_tables[libkey].zungqr_batch_sycl(queue, m, n, k, a, lda, stride_a, tau, stride_tau,
                                              batch_size, scratchpad, scratchpad_size);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        float* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgeqrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        double* tau, std::int64_t stride_tau, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgeqrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgeqrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgeqrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t* ipiv, std::int64_t stride_ipiv,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrf_batch_usm_sycl(queue, m, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetri_batch_usm_sycl(queue, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetri_batch_usm_sycl(queue, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetri_batch_usm_sycl(queue, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetri_batch_usm_sycl(queue, n, a, lda, stride_a, ipiv,
                                                         stride_ipiv, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, float* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, float* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, double* a,
                        std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
                        std::int64_t stride_ipiv, double* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<float>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t* ipiv, std::int64_t stride_ipiv, std::complex<double>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrs_batch_usm_sycl(
        queue, trans, n, nrhs, a, lda, stride_a, ipiv, stride_ipiv, b, ldb, stride_b, batch_size,
        scratchpad, scratchpad_size, dependencies);
}
sycl::event orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                        std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sorgqr_batch_usm_sycl(queue, m, n, k, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                        std::int64_t stride_a, double* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dorgqr_batch_usm_sycl(queue, m, n, k, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, float* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, double* a, std::int64_t lda, std::int64_t stride_a,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrf_batch_usm_sycl(
        queue, uplo, n, a, lda, stride_a, batch_size, scratchpad, scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda,
                        std::int64_t stride_a, float* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrs_batch_usm_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                        std::int64_t stride_a, double* b, std::int64_t ldb, std::int64_t stride_b,
                        std::int64_t batch_size, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrs_batch_usm_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* b, std::int64_t ldb,
                        std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrs_batch_usm_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo,
                        std::int64_t n, std::int64_t nrhs, std::complex<double>* a,
                        std::int64_t lda, std::int64_t stride_a, std::complex<double>* b,
                        std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrs_batch_usm_sycl(queue, uplo, n, nrhs, a, lda, stride_a, b,
                                                         ldb, stride_b, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<float>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cungqr_batch_usm_sycl(queue, m, n, k, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double>* a, std::int64_t lda,
                        std::int64_t stride_a, std::complex<double>* tau, std::int64_t stride_tau,
                        std::int64_t batch_size, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zungqr_batch_usm_sycl(queue, m, n, k, a, lda, stride_a, tau,
                                                         stride_tau, batch_size, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, float** a, std::int64_t* lda, float** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgeqrf_group_usm_sycl(queue, m, n, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, double** a, std::int64_t* lda, double** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgeqrf_group_usm_sycl(queue, m, n, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::complex<float>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgeqrf_group_usm_sycl(queue, m, n, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event geqrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::complex<double>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgeqrf_group_usm_sycl(queue, m, n, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, float** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrf_group_usm_sycl(queue, m, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, double** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes, double* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrf_group_usm_sycl(queue, m, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrf_group_usm_sycl(queue, m, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t** ipiv, std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrf_group_usm_sycl(queue, m, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* n, float** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetri_group_usm_sycl(queue, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* n, double** a,
                        std::int64_t* lda, std::int64_t** ipiv, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetri_group_usm_sycl(queue, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* n,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetri_group_usm_sycl(queue, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getri_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* n,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetri_group_usm_sycl(queue, n, a, lda, ipiv, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        float** a, std::int64_t* lda, std::int64_t** ipiv, float** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sgetrs_group_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b,
                                                         ldb, group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        double** a, std::int64_t* lda, std::int64_t** ipiv, double** b,
                        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dgetrs_group_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b,
                                                         ldb, group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        std::complex<float>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::complex<float>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cgetrs_group_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b,
                                                         ldb, group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event getrs_batch(oneapi::mkl::device libkey, sycl::queue& queue,
                        oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
                        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv,
                        std::complex<double>** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zgetrs_group_usm_sycl(queue, trans, n, nrhs, a, lda, ipiv, b,
                                                         ldb, group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, float** a, std::int64_t* lda, float** tau,
                        std::int64_t group_count, std::int64_t* group_sizes, float* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].sorgqr_group_usm_sycl(queue, m, n, k, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event orgqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, double** a, std::int64_t* lda,
                        double** tau, std::int64_t group_count, std::int64_t* group_sizes,
                        double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dorgqr_group_usm_sycl(queue, m, n, k, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, float** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrf_group_usm_sycl(queue, uplo, n, a, lda, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, double** a, std::int64_t* lda, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrf_group_usm_sycl(queue, uplo, n, a, lda, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::complex<float>** a, std::int64_t* lda,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrf_group_usm_sycl(queue, uplo, n, a, lda, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event potrf_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::complex<double>** a, std::int64_t* lda,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrf_group_usm_sycl(queue, uplo, n, a, lda, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, float** a, std::int64_t* lda,
                        float** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, float* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].spotrs_group_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb,
                                                         group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, double** a, std::int64_t* lda,
                        double** b, std::int64_t* ldb, std::int64_t group_count,
                        std::int64_t* group_sizes, double* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].dpotrs_group_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb,
                                                         group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, std::complex<float>** a,
                        std::int64_t* lda, std::complex<float>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cpotrs_group_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb,
                                                         group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event potrs_batch(oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo,
                        std::int64_t* n, std::int64_t* nrhs, std::complex<double>** a,
                        std::int64_t* lda, std::complex<double>** b, std::int64_t* ldb,
                        std::int64_t group_count, std::int64_t* group_sizes,
                        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zpotrs_group_usm_sycl(queue, uplo, n, nrhs, a, lda, b, ldb,
                                                         group_count, group_sizes, scratchpad,
                                                         scratchpad_size, dependencies);
}
sycl::event ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, std::complex<float>** a,
                        std::int64_t* lda, std::complex<float>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<float>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].cungqr_group_usm_sycl(queue, m, n, k, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}
sycl::event ungqr_batch(oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m,
                        std::int64_t* n, std::int64_t* k, std::complex<double>** a,
                        std::int64_t* lda, std::complex<double>** tau, std::int64_t group_count,
                        std::int64_t* group_sizes, std::complex<double>* scratchpad,
                        std::int64_t scratchpad_size,
                        const std::vector<sycl::event>& dependencies) {
    return function_tables[libkey].zungqr_group_usm_sycl(queue, m, n, k, a, lda, tau, group_count,
                                                         group_sizes, scratchpad, scratchpad_size,
                                                         dependencies);
}

template <>
std::int64_t gebrd_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].sgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gebrd_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zgebrd_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].sgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zgerqf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].sgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t geqrf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zgeqrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t gesvd_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                                          std::int64_t m, std::int64_t n, std::int64_t lda,
                                          std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[libkey].sgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n, lda, ldu,
                                                               ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                                           std::int64_t m, std::int64_t n, std::int64_t lda,
                                           std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[libkey].dgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n, lda, ldu,
                                                               ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::mkl::jobsvd jobu,
                                                        oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda,
                                                        std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[libkey].cgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n, lda, ldu,
                                                               ldvt);
}
template <>
std::int64_t gesvd_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::mkl::jobsvd jobu,
                                                         oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda,
                                                         std::int64_t ldu, std::int64_t ldvt) {
    return function_tables[libkey].zgesvd_scratchpad_size_sycl(queue, jobu, jobvt, m, n, lda, ldu,
                                                               ldvt);
}
template <>
std::int64_t getrf_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].sgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getrf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zgetrf_scratchpad_size_sycl(queue, m, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].sgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t n,
                                                        std::int64_t lda) {
    return function_tables[libkey].cgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getri_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[libkey].zgetri_scratchpad_size_sycl(queue, n, lda);
}
template <>
std::int64_t getrs_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::transpose trans, std::int64_t n,
                                          std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].sgetrs_scratchpad_size_sycl(queue, trans, n, nrhs, lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::transpose trans, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].dgetrs_scratchpad_size_sycl(queue, trans, n, nrhs, lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].cgetrs_scratchpad_size_sycl(queue, trans, n, nrhs, lda, ldb);
}
template <>
std::int64_t getrs_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].zgetrs_scratchpad_size_sycl(queue, trans, n, nrhs, lda, ldb);
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::job jobz,
                                                        oneapi::mkl::uplo uplo, std::int64_t n,
                                                        std::int64_t lda) {
    return function_tables[libkey].cheevd_scratchpad_size_sycl(queue, jobz, uplo, n, lda);
}
template <>
std::int64_t heevd_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::job jobz,
                                                         oneapi::mkl::uplo uplo, std::int64_t n,
                                                         std::int64_t lda) {
    return function_tables[libkey].zheevd_scratchpad_size_sycl(queue, jobz, uplo, n, lda);
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t itype,
                                                        oneapi::mkl::job jobz,
                                                        oneapi::mkl::uplo uplo, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].chegvd_scratchpad_size_sycl(queue, itype, jobz, uplo, n, lda,
                                                               ldb);
}
template <>
std::int64_t hegvd_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t itype,
                                                         oneapi::mkl::job jobz,
                                                         oneapi::mkl::uplo uplo, std::int64_t n,
                                                         std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].zhegvd_scratchpad_size_sycl(queue, itype, jobz, uplo, n, lda,
                                                               ldb);
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].chetrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrd_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zhetrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].chetrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zhetrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgbr_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::generate vect, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda) {
    return function_tables[libkey].sorgbr_scratchpad_size_sycl(queue, vect, m, n, k, lda);
}
template <>
std::int64_t orgbr_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::generate vect, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda) {
    return function_tables[libkey].dorgbr_scratchpad_size_sycl(queue, vect, m, n, k, lda);
}
template <>
std::int64_t orgtr_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[libkey].sorgtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgtr_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[libkey].dorgtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t orgqr_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda) {
    return function_tables[libkey].sorgqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t orgqr_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda) {
    return function_tables[libkey].dorgqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ormrq_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].sormrq_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t ormrq_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].dormrq_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t ormqr_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                          std::int64_t m, std::int64_t n, std::int64_t k,
                                          std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].sormqr_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t ormqr_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                           std::int64_t m, std::int64_t n, std::int64_t k,
                                           std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].dormqr_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t ormtr_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                          oneapi::mkl::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].sormtr_scratchpad_size_sycl(queue, side, uplo, trans, m, n, lda,
                                                               ldc);
}
template <>
std::int64_t ormtr_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                           oneapi::mkl::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].dormtr_scratchpad_size_sycl(queue, side, uplo, trans, m, n, lda,
                                                               ldc);
}
template <>
std::int64_t potrf_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[libkey].spotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[libkey].dpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zpotrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potrs_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,
                                          std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].spotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb);
}
template <>
std::int64_t potrs_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].dpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb);
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t nrhs,
                                                        std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].cpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb);
}
template <>
std::int64_t potrs_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t nrhs,
                                                         std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].zpotrs_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb);
}
template <>
std::int64_t potri_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[libkey].spotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[libkey].dpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t potri_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zpotri_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[libkey].ssytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[libkey].dsytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].csytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrf_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zsytrf_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t syevd_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                          std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].ssyevd_scratchpad_size_sycl(queue, jobz, uplo, n, lda);
}
template <>
std::int64_t syevd_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                           std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].dsyevd_scratchpad_size_sycl(queue, jobz, uplo, n, lda);
}
template <>
std::int64_t sygvd_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          std::int64_t itype, oneapi::mkl::job jobz,
                                          oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                          std::int64_t ldb) {
    return function_tables[libkey].ssygvd_scratchpad_size_sycl(queue, itype, jobz, uplo, n, lda,
                                                               ldb);
}
template <>
std::int64_t sygvd_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           std::int64_t itype, oneapi::mkl::job jobz,
                                           oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                           std::int64_t ldb) {
    return function_tables[libkey].dsygvd_scratchpad_size_sycl(queue, itype, jobz, uplo, n, lda,
                                                               ldb);
}
template <>
std::int64_t sytrd_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, std::int64_t n,
                                          std::int64_t lda) {
    return function_tables[libkey].ssytrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t sytrd_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, std::int64_t n,
                                           std::int64_t lda) {
    return function_tables[libkey].dsytrd_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t trtrs_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                          oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                          oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                                          std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].strtrs_scratchpad_size_sycl(queue, uplo, trans, diag, n, nrhs,
                                                               lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                           oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                           oneapi::mkl::diag diag, std::int64_t n,
                                           std::int64_t nrhs, std::int64_t lda, std::int64_t ldb) {
    return function_tables[libkey].dtrtrs_scratchpad_size_sycl(queue, uplo, trans, diag, n, nrhs,
                                                               lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        oneapi::mkl::transpose trans,
                                                        oneapi::mkl::diag diag, std::int64_t n,
                                                        std::int64_t nrhs, std::int64_t lda,
                                                        std::int64_t ldb) {
    return function_tables[libkey].ctrtrs_scratchpad_size_sycl(queue, uplo, trans, diag, n, nrhs,
                                                               lda, ldb);
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         oneapi::mkl::transpose trans,
                                                         oneapi::mkl::diag diag, std::int64_t n,
                                                         std::int64_t nrhs, std::int64_t lda,
                                                         std::int64_t ldb) {
    return function_tables[libkey].ztrtrs_scratchpad_size_sycl(queue, uplo, trans, diag, n, nrhs,
                                                               lda, ldb);
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue,
                                                        oneapi::mkl::generate vect, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    return function_tables[libkey].cungbr_scratchpad_size_sycl(queue, vect, m, n, k, lda);
}
template <>
std::int64_t ungbr_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue,
                                                         oneapi::mkl::generate vect, std::int64_t m,
                                                         std::int64_t n, std::int64_t k,
                                                         std::int64_t lda) {
    return function_tables[libkey].zungbr_scratchpad_size_sycl(queue, vect, m, n, k, lda);
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t k,
                                                        std::int64_t lda) {
    return function_tables[libkey].cungqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ungqr_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t k,
                                                         std::int64_t lda) {
    return function_tables[libkey].zungqr_scratchpad_size_sycl(queue, m, n, k, lda);
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].cungtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t ungtr_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    return function_tables[libkey].zungtr_scratchpad_size_sycl(queue, uplo, n, lda);
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::side side,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    return function_tables[libkey].cunmrq_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::side side,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda,
                                                         std::int64_t ldc) {
    return function_tables[libkey].zunmrq_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::side side,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    return function_tables[libkey].cunmqr_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t unmqr_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::side side,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda,
                                                         std::int64_t ldc) {
    return function_tables[libkey].zunmqr_scratchpad_size_sycl(queue, side, trans, m, n, k, lda,
                                                               ldc);
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                        sycl::queue& queue, oneapi::mkl::side side,
                                                        oneapi::mkl::uplo uplo,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].cunmtr_scratchpad_size_sycl(queue, side, uplo, trans, m, n, lda,
                                                               ldc);
}
template <>
std::int64_t unmtr_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                         sycl::queue& queue, oneapi::mkl::side side,
                                                         oneapi::mkl::uplo uplo,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t lda, std::int64_t ldc) {
    return function_tables[libkey].zunmtr_scratchpad_size_sycl(queue, side, uplo, trans, m, n, lda,
                                                               ldc);
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t batch_size) {
    return function_tables[libkey].sgetrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    return function_tables[libkey].dgetrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[libkey].cgetrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[libkey].zgetrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t batch_size) {
    return function_tables[libkey].sgetri_batch_scratchpad_size_sycl(queue, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t batch_size) {
    return function_tables[libkey].dgetri_batch_scratchpad_size_sycl(queue, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[libkey].cgetri_batch_scratchpad_size_sycl(queue, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t n, std::int64_t lda,
    std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t batch_size) {
    return function_tables[libkey].zgetri_batch_scratchpad_size_sycl(queue, n, lda, stride_a,
                                                                     stride_ipiv, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::transpose trans, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_ipiv,
                                                std::int64_t ldb, std::int64_t stride_b,
                                                std::int64_t batch_size) {
    return function_tables[libkey].sgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::transpose trans, std::int64_t n,
                                                 std::int64_t nrhs, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_ipiv,
                                                 std::int64_t ldb, std::int64_t stride_b,
                                                 std::int64_t batch_size) {
    return function_tables[libkey].dgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv,
    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].cgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv,
    std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].zgetrs_batch_scratchpad_size_sycl(
        queue, trans, n, nrhs, lda, stride_a, stride_ipiv, ldb, stride_b, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t stride_tau,
                                                std::int64_t batch_size) {
    return function_tables[libkey].sgeqrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t stride_tau,
                                                 std::int64_t batch_size) {
    return function_tables[libkey].dgeqrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].cgeqrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].zgeqrf_batch_scratchpad_size_sycl(queue, m, n, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t batch_size) {
    return function_tables[libkey].spotrf_batch_scratchpad_size_sycl(queue, uplo, n, lda, stride_a,
                                                                     batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::uplo uplo, std::int64_t n,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t batch_size) {
    return function_tables[libkey].dpotrf_batch_scratchpad_size_sycl(queue, uplo, n, lda, stride_a,
                                                                     batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size) {
    return function_tables[libkey].cpotrf_batch_scratchpad_size_sycl(queue, uplo, n, lda, stride_a,
                                                                     batch_size);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
    std::int64_t lda, std::int64_t stride_a, std::int64_t batch_size) {
    return function_tables[libkey].zpotrf_batch_scratchpad_size_sycl(queue, uplo, n, lda, stride_a,
                                                                     batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t stride_a, std::int64_t ldb,
                                                std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].spotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::uplo uplo, std::int64_t n,
                                                 std::int64_t nrhs, std::int64_t lda,
                                                 std::int64_t stride_a, std::int64_t ldb,
                                                 std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].dpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].cpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
    std::int64_t nrhs, std::int64_t lda, std::int64_t stride_a, std::int64_t ldb,
    std::int64_t stride_b, std::int64_t batch_size) {
    return function_tables[libkey].zpotrs_batch_scratchpad_size_sycl(
        queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda, std::int64_t stride_a,
                                                std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].sorgqr_batch_scratchpad_size_sycl(queue, m, n, k, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                                 std::int64_t lda, std::int64_t stride_a,
                                                 std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].dorgqr_batch_scratchpad_size_sycl(queue, m, n, k, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].cungqr_batch_scratchpad_size_sycl(queue, m, n, k, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
    std::int64_t lda, std::int64_t stride_a, std::int64_t stride_tau, std::int64_t batch_size) {
    return function_tables[libkey].zungqr_batch_scratchpad_size_sycl(queue, m, n, k, lda, stride_a,
                                                                     stride_tau, batch_size);
}
template <>
std::int64_t getrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].sgetrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dgetrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                              sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[libkey].cgetrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getrf_batch_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                               sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[libkey].zgetrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].sgetri_group_scratchpad_size_sycl(queue, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t* n, std::int64_t* lda,
                                                 std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dgetri_group_scratchpad_size_sycl(queue, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                              sycl::queue& queue, std::int64_t* n,
                                                              std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[libkey].cgetri_group_scratchpad_size_sycl(queue, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getri_batch_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                               sycl::queue& queue, std::int64_t* n,
                                                               std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[libkey].zgetri_group_scratchpad_size_sycl(queue, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::transpose* trans, std::int64_t* n,
                                                std::int64_t* nrhs, std::int64_t* lda,
                                                std::int64_t* ldb, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].sgetrs_group_scratchpad_size_sycl(queue, trans, n, nrhs, lda,
                                                                     ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::transpose* trans, std::int64_t* n,
                                                 std::int64_t* nrhs, std::int64_t* lda,
                                                 std::int64_t* ldb, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dgetrs_group_scratchpad_size_sycl(queue, trans, n, nrhs, lda,
                                                                     ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[libkey].cgetrs_group_scratchpad_size_sycl(queue, trans, n, nrhs, lda,
                                                                     ldb, group_count, group_sizes);
}
template <>
std::int64_t getrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[libkey].zgetrs_group_scratchpad_size_sycl(queue, trans, n, nrhs, lda,
                                                                     ldb, group_count, group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* lda,
                                                std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].sgeqrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dgeqrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<float>>(oneapi::mkl::device libkey,
                                                              sycl::queue& queue, std::int64_t* m,
                                                              std::int64_t* n, std::int64_t* lda,
                                                              std::int64_t group_count,
                                                              std::int64_t* group_sizes) {
    return function_tables[libkey].cgeqrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t geqrf_batch_scratchpad_size<std::complex<double>>(oneapi::mkl::device libkey,
                                                               sycl::queue& queue, std::int64_t* m,
                                                               std::int64_t* n, std::int64_t* lda,
                                                               std::int64_t group_count,
                                                               std::int64_t* group_sizes) {
    return function_tables[libkey].zgeqrf_group_scratchpad_size_sycl(queue, m, n, lda, group_count,
                                                                     group_sizes);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].sorgqr_group_scratchpad_size_sycl(queue, m, n, k, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t orgqr_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dorgqr_group_scratchpad_size_sycl(queue, m, n, k, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::uplo* uplo, std::int64_t* n,
                                                std::int64_t* lda, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].spotrf_group_scratchpad_size_sycl(queue, uplo, n, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::uplo* uplo, std::int64_t* n,
                                                 std::int64_t* lda, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dpotrf_group_scratchpad_size_sycl(queue, uplo, n, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n,
    std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[libkey].cpotrf_group_scratchpad_size_sycl(queue, uplo, n, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrf_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n,
    std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[libkey].zpotrf_group_scratchpad_size_sycl(queue, uplo, n, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<float>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                oneapi::mkl::uplo* uplo, std::int64_t* n,
                                                std::int64_t* nrhs, std::int64_t* lda,
                                                std::int64_t* ldb, std::int64_t group_count,
                                                std::int64_t* group_sizes) {
    return function_tables[libkey].spotrs_group_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<double>(oneapi::mkl::device libkey, sycl::queue& queue,
                                                 oneapi::mkl::uplo* uplo, std::int64_t* n,
                                                 std::int64_t* nrhs, std::int64_t* lda,
                                                 std::int64_t* ldb, std::int64_t group_count,
                                                 std::int64_t* group_sizes) {
    return function_tables[libkey].dpotrs_group_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[libkey].cpotrs_group_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t potrs_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n,
    std::int64_t* nrhs, std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count,
    std::int64_t* group_sizes) {
    return function_tables[libkey].zpotrs_group_scratchpad_size_sycl(queue, uplo, n, nrhs, lda, ldb,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<float>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m, std::int64_t* n,
    std::int64_t* k, std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[libkey].cungqr_group_scratchpad_size_sycl(queue, m, n, k, lda,
                                                                     group_count, group_sizes);
}
template <>
std::int64_t ungqr_batch_scratchpad_size<std::complex<double>>(
    oneapi::mkl::device libkey, sycl::queue& queue, std::int64_t* m, std::int64_t* n,
    std::int64_t* k, std::int64_t* lda, std::int64_t group_count, std::int64_t* group_sizes) {
    return function_tables[libkey].zungqr_group_scratchpad_size_sycl(queue, m, n, k, lda,
                                                                     group_count, group_sizes);
}

} //namespace detail
} //namespace lapack
} //namespace mkl
} //namespace oneapi
