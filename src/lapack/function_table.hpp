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

#pragma once

#include <complex>
#include <cstdint>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"

typedef struct {
    int version;
    void (*cgebrd_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<float>& d, sycl::buffer<float>& e,
                        sycl::buffer<std::complex<float>>& tauq,
                        sycl::buffer<std::complex<float>>& taup,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgebrd_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& d, sycl::buffer<double>& e,
                        sycl::buffer<double>& tauq, sycl::buffer<double>& taup,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgebrd_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& d, sycl::buffer<float>& e,
                        sycl::buffer<float>& tauq, sycl::buffer<float>& taup,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*zgebrd_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<double>& d, sycl::buffer<double>& e,
                        sycl::buffer<std::complex<double>>& tauq,
                        sycl::buffer<std::complex<double>>& taup,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*sgerqf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgerqf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& tau,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*cgerqf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zgerqf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cgeqrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgeqrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& tau,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgeqrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zgeqrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cgetrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgetrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgetrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*zgetrf_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cgetri_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgetri_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgetri_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*zgetri_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                        std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cgetrs_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<float>>& b,
                        std::int64_t ldb, sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgetrs_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv, sycl::buffer<double>& b, std::int64_t ldb,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgetrs_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv, sycl::buffer<float>& b, std::int64_t ldb,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*zgetrs_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& b,
                        std::int64_t ldb, sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dgesvd_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                        std::int64_t m, std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                        sycl::buffer<double>& s, sycl::buffer<double>& u, std::int64_t ldu,
                        sycl::buffer<double>& vt, std::int64_t ldvt,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sgesvd_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                        std::int64_t m, std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                        sycl::buffer<float>& s, sycl::buffer<float>& u, std::int64_t ldu,
                        sycl::buffer<float>& vt, std::int64_t ldvt, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cgesvd_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                        std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
                        std::int64_t lda, sycl::buffer<float>& s,
                        sycl::buffer<std::complex<float>>& u, std::int64_t ldu,
                        sycl::buffer<std::complex<float>>& vt, std::int64_t ldvt,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zgesvd_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                        std::int64_t m, std::int64_t n, sycl::buffer<std::complex<double>>& a,
                        std::int64_t lda, sycl::buffer<double>& s,
                        sycl::buffer<std::complex<double>>& u, std::int64_t ldu,
                        sycl::buffer<std::complex<double>>& vt, std::int64_t ldvt,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cheevd_sycl)(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                        std::int64_t n, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zheevd_sycl)(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                        std::int64_t n, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*chegvd_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                        oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                        sycl::buffer<float>& w, sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zhegvd_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                        oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                        sycl::buffer<double>& w, sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*chetrd_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<float>& d, sycl::buffer<float>& e,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zhetrd_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<double>& d, sycl::buffer<double>& e,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*chetrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zhetrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*sorgbr_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
                        sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dorgbr_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
                        sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dorgqr_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sorgqr_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*sorgtr_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*dorgtr_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*sormtr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                        oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& tau,
                        sycl::buffer<float>& c, std::int64_t ldc, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dormtr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                        oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& tau,
                        sycl::buffer<double>& c, std::int64_t ldc, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*sormrq_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& c,
                        std::int64_t ldc, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dormrq_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c,
                        std::int64_t ldc, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dormqr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& c,
                        std::int64_t ldc, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*sormqr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& c,
                        std::int64_t ldc, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*spotrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dpotrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cpotrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zpotrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*spotri_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dpotri_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cpotri_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zpotri_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*spotrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                        sycl::buffer<float>& b, std::int64_t ldb, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dpotrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                        sycl::buffer<double>& b, std::int64_t ldb, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cpotrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zpotrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        std::int64_t nrhs, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dsyevd_sycl)(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                        std::int64_t n, sycl::buffer<double>& a, std::int64_t lda,
                        sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*ssyevd_sycl)(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                        std::int64_t n, sycl::buffer<float>& a, std::int64_t lda,
                        sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dsygvd_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                        oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<double>& a,
                        std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
                        sycl::buffer<double>& w, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*ssygvd_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                        oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<float>& a,
                        std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
                        sycl::buffer<float>& w, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dsytrd_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& d,
                        sycl::buffer<double>& e, sycl::buffer<double>& tau,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*ssytrd_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& d,
                        sycl::buffer<float>& e, sycl::buffer<float>& tau,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*ssytrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*dsytrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*csytrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zsytrf_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::int64_t>& ipiv,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*ctrtrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                        oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*dtrtrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                        oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<double>& a, std::int64_t lda, sycl::buffer<double>& b,
                        std::int64_t ldb, sycl::buffer<double>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*strtrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                        oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<float>& a, std::int64_t lda, sycl::buffer<float>& b,
                        std::int64_t ldb, sycl::buffer<float>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*ztrtrs_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                        oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cungbr_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zungbr_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cungqr_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zungqr_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cungtr_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zungtr_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cunmrq_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zunmrq_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cunmqr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zunmqr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                        std::int64_t m, std::int64_t n, std::int64_t k,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*cunmtr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                        oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>>& tau,
                        sycl::buffer<std::complex<float>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<float>>& scratchpad,
                        std::int64_t scratchpad_size);
    void (*zunmtr_sycl)(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                        oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                        sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>>& tau,
                        sycl::buffer<std::complex<double>>& c, std::int64_t ldc,
                        sycl::buffer<std::complex<double>>& scratchpad,
                        std::int64_t scratchpad_size);
    sycl::event (*cgebrd_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, float* d, float* e,
                                   std::complex<float>* tauq, std::complex<float>* taup,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgebrd_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                   std::int64_t lda, double* d, double* e, double* tauq,
                                   double* taup, double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgebrd_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                   std::int64_t lda, float* d, float* e, float* tauq, float* taup,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgebrd_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, double* d, double* e,
                                   std::complex<double>* tauq, std::complex<double>* taup,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgerqf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                   std::int64_t lda, float* tau, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgerqf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                   std::int64_t lda, double* tau, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgerqf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgerqf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgeqrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgeqrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                   std::int64_t lda, double* tau, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgeqrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                   std::int64_t lda, float* tau, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgeqrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a,
                                   std::int64_t lda, std::int64_t* ipiv, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a,
                                   std::int64_t lda, std::int64_t* ipiv, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrf_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetri_usm_sycl)(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                                   std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetri_usm_sycl)(sycl::queue& queue, std::int64_t n, double* a, std::int64_t lda,
                                   std::int64_t* ipiv, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetri_usm_sycl)(sycl::queue& queue, std::int64_t n, float* a, std::int64_t lda,
                                   std::int64_t* ipiv, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetri_usm_sycl)(sycl::queue& queue, std::int64_t n, std::complex<double>* a,
                                   std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                                   std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                                   std::int64_t* ipiv, std::complex<float>* b, std::int64_t ldb,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                                   std::int64_t nrhs, double* a, std::int64_t lda,
                                   std::int64_t* ipiv, double* b, std::int64_t ldb,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                                   std::int64_t nrhs, float* a, std::int64_t lda,
                                   std::int64_t* ipiv, float* b, std::int64_t ldb,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                                   std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                                   std::int64_t* ipiv, std::complex<double>* b, std::int64_t ldb,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dgesvd_usm_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                   oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   double* a, std::int64_t lda, double* s, double* u,
                                   std::int64_t ldu, double* vt, std::int64_t ldvt,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sgesvd_usm_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                   oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   float* a, std::int64_t lda, float* s, float* u, std::int64_t ldu,
                                   float* vt, std::int64_t ldvt, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cgesvd_usm_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                   oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, float* s,
                                   std::complex<float>* u, std::int64_t ldu,
                                   std::complex<float>* vt, std::int64_t ldvt,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zgesvd_usm_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                   oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, double* s,
                                   std::complex<double>* u, std::int64_t ldu,
                                   std::complex<double>* vt, std::int64_t ldvt,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cheevd_usm_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                   std::int64_t lda, float* w, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zheevd_usm_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                   std::int64_t lda, double* w, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*chegvd_usm_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float>* a,
                                   std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                                   float* w, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zhegvd_usm_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, std::complex<double>* a,
                                   std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                                   double* w, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*chetrd_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, float* d, float* e,
                                   std::complex<float>* tau, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zhetrd_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, double* d, double* e,
                                   std::complex<double>* tau, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*chetrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zhetrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sorgbr_usm_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                                   std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                                   float* tau, float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dorgbr_usm_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                                   std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                                   double* tau, double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dorgqr_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, double* a, std::int64_t lda, double* tau,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sorgqr_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, float* a, std::int64_t lda, float* tau,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sorgtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   float* a, std::int64_t lda, float* tau, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dorgtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   double* a, std::int64_t lda, double* tau, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sormtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                                   float* tau, float* c, std::int64_t ldc, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dormtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                                   double* tau, double* c, std::int64_t ldc, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sormrq_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, float* a, std::int64_t lda, float* tau, float* c,
                                   std::int64_t ldc, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dormrq_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, double* a, std::int64_t lda, double* tau,
                                   double* c, std::int64_t ldc, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dormqr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, double* a, std::int64_t lda, double* tau,
                                   double* c, std::int64_t ldc, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*sormqr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, float* a, std::int64_t lda, float* tau, float* c,
                                   std::int64_t ldc, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   float* a, std::int64_t lda, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   double* a, std::int64_t lda, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*spotri_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   float* a, std::int64_t lda, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotri_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   double* a, std::int64_t lda, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotri_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotri_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, float* a, std::int64_t lda, float* b,
                                   std::int64_t ldb, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, double* a, std::int64_t lda, double* b,
                                   std::int64_t ldb, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* b, std::int64_t ldb,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::int64_t nrhs, std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* b, std::int64_t ldb,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dsyevd_usm_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, double* a,
                                   std::int64_t lda, double* w, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ssyevd_usm_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, float* a,
                                   std::int64_t lda, float* w, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dsygvd_usm_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, double* a,
                                   std::int64_t lda, double* b, std::int64_t ldb, double* w,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ssygvd_usm_sycl)(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,
                                   oneapi::mkl::uplo uplo, std::int64_t n, float* a,
                                   std::int64_t lda, float* b, std::int64_t ldb, float* w,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dsytrd_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   double* a, std::int64_t lda, double* d, double* e, double* tau,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ssytrd_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   float* a, std::int64_t lda, float* d, float* e, float* tau,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ssytrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   float* a, std::int64_t lda, std::int64_t* ipiv,
                                   float* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dsytrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   double* a, std::int64_t lda, std::int64_t* ipiv,
                                   double* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*csytrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zsytrf_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ctrtrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                   std::int64_t n, std::int64_t nrhs, std::complex<float>* a,
                                   std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*dtrtrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                   std::int64_t n, std::int64_t nrhs, double* a, std::int64_t lda,
                                   double* b, std::int64_t ldb, double* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*strtrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                   std::int64_t n, std::int64_t nrhs, float* a, std::int64_t lda,
                                   float* b, std::int64_t ldb, float* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*ztrtrs_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                   oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                   std::int64_t n, std::int64_t nrhs, std::complex<double>* a,
                                   std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cungbr_usm_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::complex<float>* a,
                                   std::int64_t lda, std::complex<float>* tau,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zungbr_usm_sycl)(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,
                                   std::int64_t n, std::int64_t k, std::complex<double>* a,
                                   std::int64_t lda, std::complex<double>* tau,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cungqr_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zungqr_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cungtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zungtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                   std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cunmrq_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* c,
                                   std::int64_t ldc, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zunmrq_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* c,
                                   std::int64_t ldc, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cunmqr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* tau, std::complex<float>* c,
                                   std::int64_t ldc, std::complex<float>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zunmqr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                   std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* tau, std::complex<double>* c,
                                   std::int64_t ldc, std::complex<double>* scratchpad,
                                   std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*cunmtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<float>* a,
                                   std::int64_t lda, std::complex<float>* tau,
                                   std::complex<float>* c, std::int64_t ldc,
                                   std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    sycl::event (*zunmtr_usm_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                   oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                                   std::int64_t m, std::int64_t n, std::complex<double>* a,
                                   std::int64_t lda, std::complex<double>* tau,
                                   std::complex<double>* c, std::int64_t ldc,
                                   std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                                   const std::vector<sycl::event>& dependencies);
    void (*sgeqrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<float>& tau, std::int64_t stride_tau,
                              std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*dgeqrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<double>& tau, std::int64_t stride_tau,
                              std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*cgeqrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                              std::int64_t stride_tau, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zgeqrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                              std::int64_t stride_tau, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*sgetri_batch_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<float>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*dgetri_batch_sycl)(sycl::queue& queue, std::int64_t n, sycl::buffer<double>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*cgetri_batch_sycl)(sycl::queue& queue, std::int64_t n,
                              sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zgetri_batch_sycl)(sycl::queue& queue, std::int64_t n,
                              sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*sgetrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, sycl::buffer<float>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*dgetrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, sycl::buffer<double>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*cgetrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zgetrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*sgetrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*dgetrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::int64_t>& ipiv, std::int64_t stride_ipiv,
                              std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*cgetrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zgetrf_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int64_t>& ipiv,
                              std::int64_t stride_ipiv, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*sorgqr_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                              sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<float>& tau, std::int64_t stride_tau,
                              std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*dorgqr_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                              sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<double>& tau, std::int64_t stride_tau,
                              std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*spotrf_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              sycl::buffer<float>& a, std::int64_t lda, std::int64_t stride_a,
                              std::int64_t batch_size, sycl::buffer<float>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*dpotrf_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              sycl::buffer<double>& a, std::int64_t lda, std::int64_t stride_a,
                              std::int64_t batch_size, sycl::buffer<double>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*cpotrf_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                              std::int64_t stride_a, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zpotrf_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                              std::int64_t stride_a, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*spotrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<float>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<float>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size);
    void (*dpotrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<double>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<double>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size);
    void (*cpotrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<std::complex<float>>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zpotrs_batch_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                              std::int64_t nrhs, sycl::buffer<std::complex<double>>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*cungqr_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<float>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::complex<float>>& tau,
                              std::int64_t stride_tau, std::int64_t batch_size,
                              sycl::buffer<std::complex<float>>& scratchpad,
                              std::int64_t scratchpad_size);
    void (*zungqr_batch_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<double>>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::complex<double>>& tau,
                              std::int64_t stride_tau, std::int64_t batch_size,
                              sycl::buffer<std::complex<double>>& scratchpad,
                              std::int64_t scratchpad_size);
    sycl::event (*sgeqrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         float* a, std::int64_t lda, std::int64_t stride_a,
                                         float* tau, std::int64_t stride_tau,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgeqrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         double* a, std::int64_t lda, std::int64_t stride_a,
                                         double* tau, std::int64_t stride_tau,
                                         std::int64_t batch_size, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgeqrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::complex<float>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<float>* tau,
                                         std::int64_t stride_tau, std::int64_t batch_size,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgeqrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<double>* tau,
                                         std::int64_t stride_tau, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         float* a, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         double* a, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv,
                                         std::int64_t batch_size, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::complex<float>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t* ipiv,
                                         std::int64_t stride_ipiv, std::int64_t batch_size,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrf_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t* ipiv,
                                         std::int64_t stride_ipiv, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetri_batch_usm_sycl)(sycl::queue& queue, std::int64_t n, float* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetri_batch_usm_sycl)(sycl::queue& queue, std::int64_t n, double* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv,
                                         std::int64_t batch_size, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetri_batch_usm_sycl)(sycl::queue& queue, std::int64_t n, std::complex<float>* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv,
                                         std::int64_t batch_size, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetri_batch_usm_sycl)(sycl::queue& queue, std::int64_t n,
                                         std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t* ipiv,
                                         std::int64_t stride_ipiv, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                         std::int64_t n, std::int64_t nrhs, float* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv, float* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                         std::int64_t n, std::int64_t nrhs, double* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t* ipiv, std::int64_t stride_ipiv, double* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrs_batch_usm_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::complex<float>* a, std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
        std::int64_t stride_ipiv, std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
        const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrs_batch_usm_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::complex<double>* a, std::int64_t lda, std::int64_t stride_a, std::int64_t* ipiv,
        std::int64_t stride_ipiv, std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
        std::int64_t batch_size, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
        const std::vector<sycl::event>& dependencies);
    sycl::event (*sorgqr_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, float* a, std::int64_t lda,
                                         std::int64_t stride_a, float* tau, std::int64_t stride_tau,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dorgqr_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, double* a, std::int64_t lda,
                                         std::int64_t stride_a, double* tau,
                                         std::int64_t stride_tau, std::int64_t batch_size,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrf_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         float* a, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t batch_size, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrf_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         double* a, std::int64_t lda, std::int64_t stride_a,
                                         std::int64_t batch_size, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrf_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::complex<float>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t batch_size,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrf_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::int64_t nrhs, float* a, std::int64_t lda,
                                         std::int64_t stride_a, float* b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::int64_t nrhs, double* a, std::int64_t lda,
                                         std::int64_t stride_a, double* b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::int64_t nrhs, std::complex<float>* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::complex<float>* b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrs_batch_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                                         std::int64_t nrhs, std::complex<double>* a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         std::complex<double>* b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cungqr_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::complex<float>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<float>* tau,
                                         std::int64_t stride_tau, std::int64_t batch_size,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zungqr_batch_usm_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                         std::int64_t k, std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<double>* tau,
                                         std::int64_t stride_tau, std::int64_t batch_size,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgeqrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         float** a, std::int64_t* lda, float** tau,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgeqrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         double** a, std::int64_t* lda, double** tau,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgeqrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::complex<float>** a, std::int64_t* lda,
                                         std::complex<float>** tau, std::int64_t group_count,
                                         std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgeqrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::complex<double>** a, std::int64_t* lda,
                                         std::complex<double>** tau, std::int64_t group_count,
                                         std::int64_t* group_sizes,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         float** a, std::int64_t* lda, std::int64_t** ipiv,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         double** a, std::int64_t* lda, std::int64_t** ipiv,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::complex<float>** a, std::int64_t* lda,
                                         std::int64_t** ipiv, std::int64_t group_count,
                                         std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrf_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::complex<double>** a, std::int64_t* lda,
                                         std::int64_t** ipiv, std::int64_t group_count,
                                         std::int64_t* group_sizes,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetri_group_usm_sycl)(sycl::queue& queue, std::int64_t* n, float** a,
                                         std::int64_t* lda, std::int64_t** ipiv,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetri_group_usm_sycl)(sycl::queue& queue, std::int64_t* n, double** a,
                                         std::int64_t* lda, std::int64_t** ipiv,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetri_group_usm_sycl)(sycl::queue& queue, std::int64_t* n,
                                         std::complex<float>** a, std::int64_t* lda,
                                         std::int64_t** ipiv, std::int64_t group_count,
                                         std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetri_group_usm_sycl)(sycl::queue& queue, std::int64_t* n,
                                         std::complex<double>** a, std::int64_t* lda,
                                         std::int64_t** ipiv, std::int64_t group_count,
                                         std::int64_t* group_sizes,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*sgetrs_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose* trans,
                                         std::int64_t* n, std::int64_t* nrhs, float** a,
                                         std::int64_t* lda, std::int64_t** ipiv, float** b,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes, float* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dgetrs_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose* trans,
                                         std::int64_t* n, std::int64_t* nrhs, double** a,
                                         std::int64_t* lda, std::int64_t** ipiv, double** b,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cgetrs_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::transpose* trans,
                                         std::int64_t* n, std::int64_t* nrhs,
                                         std::complex<float>** a, std::int64_t* lda,
                                         std::int64_t** ipiv, std::complex<float>** b,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zgetrs_group_usm_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
        std::complex<double>** a, std::int64_t* lda, std::int64_t** ipiv, std::complex<double>** b,
        std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes,
        std::complex<double>* scratchpad, std::int64_t scratchpad_size,
        const std::vector<sycl::event>& dependencies);
    sycl::event (*sorgqr_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, float** a, std::int64_t* lda, float** tau,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dorgqr_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, double** a, std::int64_t* lda,
                                         double** tau, std::int64_t group_count,
                                         std::int64_t* group_sizes, double* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrf_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, float** a, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrf_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, double** a, std::int64_t* lda,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrf_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, std::complex<float>** a,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes, std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrf_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, std::complex<double>** a,
                                         std::int64_t* lda, std::int64_t group_count,
                                         std::int64_t* group_sizes,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*spotrs_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, std::int64_t* nrhs, float** a,
                                         std::int64_t* lda, float** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         float* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*dpotrs_group_usm_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                         std::int64_t* n, std::int64_t* nrhs, double** a,
                                         std::int64_t* lda, double** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         double* scratchpad, std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*cpotrs_group_usm_sycl)(
        sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
        std::complex<float>** a, std::int64_t* lda, std::complex<float>** b, std::int64_t* ldb,
        std::int64_t group_count, std::int64_t* group_sizes, std::complex<float>* scratchpad,
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies);
    sycl::event (*zpotrs_group_usm_sycl)(
        sycl::queue& queue, oneapi::mkl::uplo* uplo, std::int64_t* n, std::int64_t* nrhs,
        std::complex<double>** a, std::int64_t* lda, std::complex<double>** b, std::int64_t* ldb,
        std::int64_t group_count, std::int64_t* group_sizes, std::complex<double>* scratchpad,
        std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies);
    sycl::event (*cungqr_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::complex<float>** a,
                                         std::int64_t* lda, std::complex<float>** tau,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         std::complex<float>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);
    sycl::event (*zungqr_group_usm_sycl)(sycl::queue& queue, std::int64_t* m, std::int64_t* n,
                                         std::int64_t* k, std::complex<double>** a,
                                         std::int64_t* lda, std::complex<double>** tau,
                                         std::int64_t group_count, std::int64_t* group_sizes,
                                         std::complex<double>* scratchpad,
                                         std::int64_t scratchpad_size,
                                         const std::vector<sycl::event>& dependencies);

    std::int64_t (*sgebrd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dgebrd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*cgebrd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zgebrd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*sgerqf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dgerqf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*cgerqf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zgerqf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*sgeqrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dgeqrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*cgeqrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zgeqrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*sgesvd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                                oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldu,
                                                std::int64_t ldvt);
    std::int64_t (*dgesvd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                                oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldu,
                                                std::int64_t ldvt);
    std::int64_t (*cgesvd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                                oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldu,
                                                std::int64_t ldvt);
    std::int64_t (*zgesvd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                                                oneapi::mkl::jobsvd jobvt, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldu,
                                                std::int64_t ldvt);
    std::int64_t (*sgetrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dgetrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*cgetrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zgetrf_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*sgetri_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dgetri_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*cgetri_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zgetri_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*sgetrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*dgetrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*cgetrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*zgetrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::transpose trans,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*cheevd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*zheevd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*chegvd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t itype,
                                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldb);
    std::int64_t (*zhegvd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t itype,
                                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldb);
    std::int64_t (*chetrd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zhetrd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*chetrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zhetrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*sorgbr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::generate vect,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda);
    std::int64_t (*dorgbr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::generate vect,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda);
    std::int64_t (*sorgtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*dorgtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*sorgqr_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda);
    std::int64_t (*dorgqr_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda);
    std::int64_t (*sormrq_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*dormrq_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*sormqr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*dormqr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*sormtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldc);
    std::int64_t (*dormtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldc);
    std::int64_t (*spotrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*dpotrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*cpotrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zpotrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*spotrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*dpotrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*cpotrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*zpotrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*spotri_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*dpotri_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*cpotri_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zpotri_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*ssytrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*dsytrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*csytrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zsytrf_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*ssyevd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*dsyevd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::job jobz,
                                                oneapi::mkl::uplo uplo, std::int64_t n,
                                                std::int64_t lda);
    std::int64_t (*ssygvd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t itype,
                                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldb);
    std::int64_t (*dsygvd_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t itype,
                                                oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldb);
    std::int64_t (*ssytrd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*dsytrd_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*strtrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans,
                                                oneapi::mkl::diag diag, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*dtrtrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans,
                                                oneapi::mkl::diag diag, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*ctrtrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans,
                                                oneapi::mkl::diag diag, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*ztrtrs_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans,
                                                oneapi::mkl::diag diag, std::int64_t n,
                                                std::int64_t nrhs, std::int64_t lda,
                                                std::int64_t ldb);
    std::int64_t (*cungbr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::generate vect,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda);
    std::int64_t (*zungbr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::generate vect,
                                                std::int64_t m, std::int64_t n, std::int64_t k,
                                                std::int64_t lda);
    std::int64_t (*cungqr_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda);
    std::int64_t (*zungqr_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                                std::int64_t k, std::int64_t lda);
    std::int64_t (*cungtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*zungtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                std::int64_t n, std::int64_t lda);
    std::int64_t (*cunmrq_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*zunmrq_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*cunmqr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*zunmqr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t k, std::int64_t lda,
                                                std::int64_t ldc);
    std::int64_t (*cunmtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldc);
    std::int64_t (*zunmtr_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::side side,
                                                oneapi::mkl::uplo uplo,
                                                oneapi::mkl::transpose trans, std::int64_t m,
                                                std::int64_t n, std::int64_t lda, std::int64_t ldc);
    std::int64_t (*sgetrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*dgetrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*cgetrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*zgetrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*sgetri_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*dgetri_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*cgetri_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*zgetri_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t n,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_ipiv,
                                                      std::int64_t batch_size);
    std::int64_t (*sgetrs_batch_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    std::int64_t (*dgetrs_batch_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    std::int64_t (*cgetrs_batch_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    std::int64_t (*zgetrs_batch_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
        std::int64_t lda, std::int64_t stride_a, std::int64_t stride_ipiv, std::int64_t ldb,
        std::int64_t stride_b, std::int64_t batch_size);
    std::int64_t (*sgeqrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*dgeqrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*cgeqrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*zgeqrf_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*spotrf_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t batch_size);
    std::int64_t (*dpotrf_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t batch_size);
    std::int64_t (*cpotrf_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t batch_size);
    std::int64_t (*zpotrf_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t lda,
                                                      std::int64_t stride_a,
                                                      std::int64_t batch_size);
    std::int64_t (*spotrs_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t nrhs,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t ldb, std::int64_t stride_b,
                                                      std::int64_t batch_size);
    std::int64_t (*dpotrs_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t nrhs,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t ldb, std::int64_t stride_b,
                                                      std::int64_t batch_size);
    std::int64_t (*cpotrs_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t nrhs,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t ldb, std::int64_t stride_b,
                                                      std::int64_t batch_size);
    std::int64_t (*zpotrs_batch_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                      std::int64_t n, std::int64_t nrhs,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t ldb, std::int64_t stride_b,
                                                      std::int64_t batch_size);
    std::int64_t (*sorgqr_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t k,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*dorgqr_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t k,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*cungqr_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t k,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*zungqr_batch_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t m,
                                                      std::int64_t n, std::int64_t k,
                                                      std::int64_t lda, std::int64_t stride_a,
                                                      std::int64_t stride_tau,
                                                      std::int64_t batch_size);
    std::int64_t (*sgetrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dgetrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cgetrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zgetrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*sgetri_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* n,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dgetri_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* n,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cgetri_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* n,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zgetri_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* n,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*sgetrs_group_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes);
    std::int64_t (*dgetrs_group_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes);
    std::int64_t (*cgetrs_group_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes);
    std::int64_t (*zgetrs_group_scratchpad_size_sycl)(
        sycl::queue& queue, oneapi::mkl::transpose* trans, std::int64_t* n, std::int64_t* nrhs,
        std::int64_t* lda, std::int64_t* ldb, std::int64_t group_count, std::int64_t* group_sizes);
    std::int64_t (*sgeqrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dgeqrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cgeqrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zgeqrf_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*sorgqr_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* k,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dorgqr_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* k,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*spotrf_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dpotrf_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cpotrf_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zpotrf_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* lda,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*spotrs_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* nrhs,
                                                      std::int64_t* lda, std::int64_t* ldb,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*dpotrs_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* nrhs,
                                                      std::int64_t* lda, std::int64_t* ldb,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cpotrs_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* nrhs,
                                                      std::int64_t* lda, std::int64_t* ldb,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zpotrs_group_scratchpad_size_sycl)(sycl::queue& queue, oneapi::mkl::uplo* uplo,
                                                      std::int64_t* n, std::int64_t* nrhs,
                                                      std::int64_t* lda, std::int64_t* ldb,
                                                      std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*cungqr_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* k,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);
    std::int64_t (*zungqr_group_scratchpad_size_sycl)(sycl::queue& queue, std::int64_t* m,
                                                      std::int64_t* n, std::int64_t* k,
                                                      std::int64_t* lda, std::int64_t group_count,
                                                      std::int64_t* group_sizes);

} lapack_function_table_t;
