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

#ifndef lapack_int
#define lapack_int int64_t
#endif
#ifndef CBLAS_INT
#define CBLAS_INT int64_t
#endif
#define WeirdNEC
extern "C" {
#ifdef USE_MKLREF
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif
}
static_assert(sizeof(lapack_int) == 8);
static_assert(sizeof(CBLAS_INT) == 8);

namespace reference {
inline CBLAS_TRANSPOSE cblas_trans(oneapi::mkl::transpose t) {
    if (t == oneapi::mkl::transpose::nontrans)
        return CblasNoTrans;
    if (t == oneapi::mkl::transpose::trans)
        return CblasTrans;
    if (t == oneapi::mkl::transpose::conjtrans)
        return CblasConjTrans;
    return CblasNoTrans;
}
inline CBLAS_UPLO cblas_uplo(oneapi::mkl::uplo u) {
    if (u == oneapi::mkl::uplo::upper)
        return CblasUpper;
    if (u == oneapi::mkl::uplo::lower)
        return CblasLower;
    return CblasUpper;
}
inline CBLAS_DIAG cblas_diag(oneapi::mkl::diag d) {
    if (d == oneapi::mkl::diag::nonunit)
        return CblasNonUnit;
    if (d == oneapi::mkl::diag::unit)
        return CblasUnit;
    return CblasNonUnit;
}
inline CBLAS_SIDE cblas_side(const char* c) {
    return *c == 'R' || *c == 'r' ? CblasRight : CblasLeft;
}
inline CBLAS_SIDE cblas_side(oneapi::mkl::side s) {
    if (s == oneapi::mkl::side::left)
        return CblasLeft;
    if (s == oneapi::mkl::side::right)
        return CblasRight;
    return CblasLeft;
}
inline char to_char(oneapi::mkl::transpose t) {
    if (t == oneapi::mkl::transpose::nontrans)
        return 'N';
    if (t == oneapi::mkl::transpose::trans)
        return 'T';
    if (t == oneapi::mkl::transpose::conjtrans)
        return 'C';
    return 'N';
}
inline char to_char(oneapi::mkl::offset t) {
    if (t == oneapi::mkl::offset::fix)
        return 'F';
    if (t == oneapi::mkl::offset::row)
        return 'R';
    if (t == oneapi::mkl::offset::column)
        return 'C';
    return 'N';
}

inline char to_char(oneapi::mkl::uplo u) {
    if (u == oneapi::mkl::uplo::upper)
        return 'U';
    if (u == oneapi::mkl::uplo::lower)
        return 'L';
    return 'U';
}

inline char to_char(oneapi::mkl::diag d) {
    if (d == oneapi::mkl::diag::nonunit)
        return 'N';
    if (d == oneapi::mkl::diag::unit)
        return 'U';
    return 'N';
}

inline char to_char(oneapi::mkl::side s) {
    if (s == oneapi::mkl::side::left)
        return 'L';
    if (s == oneapi::mkl::side::right)
        return 'R';
    return 'L';
}

inline char to_char(oneapi::mkl::job j) {
    if (j == oneapi::mkl::job::novec)
        return 'N';
    if (j == oneapi::mkl::job::vec)
        return 'V';
    if (j == oneapi::mkl::job::updatevec)
        return 'U';
    if (j == oneapi::mkl::job::allvec)
        return 'A';
    if (j == oneapi::mkl::job::somevec)
        return 'S';
    if (j == oneapi::mkl::job::overwritevec)
        return 'O';
    return 'N';
}
inline char to_char(oneapi::mkl::jobsvd j) {
    if (j == oneapi::mkl::jobsvd::novec)
        return 'N';
    if (j == oneapi::mkl::jobsvd::vectors)
        return 'A';
    if (j == oneapi::mkl::jobsvd::vectorsina)
        return 'O';
    if (j == oneapi::mkl::jobsvd::somevec)
        return 'S';
    return 'N';
}
inline char to_char(oneapi::mkl::generate v) {
    if (v == oneapi::mkl::generate::Q)
        return 'Q';
    if (v == oneapi::mkl::generate::P)
        return 'P';
    return 'Q';
}

inline void gemm(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
                 float beta, float* c, int64_t ldc) {
    cblas_sgemm(CblasColMajor, cblas_trans(transa), cblas_trans(transb), m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
}
inline void gemm(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, int64_t m, int64_t n,
                 int64_t k, double alpha, const double* a, int64_t lda, const double* b,
                 int64_t ldb, double beta, double* c, int64_t ldc) {
    cblas_dgemm(CblasColMajor, cblas_trans(transa), cblas_trans(transb), m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
}
inline void gemm(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, int64_t m, int64_t n,
                 int64_t k, std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                 std::complex<float>* c, int64_t ldc) {
    cblas_cgemm(CblasColMajor, cblas_trans(transa), cblas_trans(transb), m, n, k, (void*)&alpha,
                (void*)a, lda, (void*)(b), ldb, (void*)&beta, (void*)c, ldc);
}
inline void gemm(oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, int64_t m, int64_t n,
                 int64_t k, std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* b, int64_t ldb, std::complex<double> beta,
                 std::complex<double>* c, int64_t ldc) {
    cblas_zgemm(CblasColMajor, cblas_trans(transa), cblas_trans(transb), m, n, k, (void*)&alpha,
                (void*)a, lda, (void*)(b), ldb, (void*)&beta, (void*)c, ldc);
}

inline int64_t syevd(oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, float* a, int64_t lda,
                     float* w) {
    return LAPACKE_ssyevd(LAPACK_COL_MAJOR, to_char(j), to_char(u), n, a, lda, w);
}
inline int64_t syevd(oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, double* a, int64_t lda,
                     double* w) {
    return LAPACKE_dsyevd(LAPACK_COL_MAJOR, to_char(j), to_char(u), n, a, lda, w);
}

inline int64_t sygvd(int64_t itype, oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, float* a,
                     int64_t lda, float* b, int64_t ldb, float* w) {
    return LAPACKE_ssygvd(LAPACK_COL_MAJOR, itype, to_char(j), to_char(u), n, a, lda, b, ldb, w);
}
inline int64_t sygvd(int64_t itype, oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, double* a,
                     int64_t lda, double* b, int64_t ldb, double* w) {
    return LAPACKE_dsygvd(LAPACK_COL_MAJOR, itype, to_char(j), to_char(u), n, a, lda, b, ldb, w);
}

inline void syrk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t lda, float beta, float* c, int64_t ldc) {
    cblas_ssyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, c, ldc);
}
inline void syrk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k, double alpha,
                 const double* a, int64_t lda, double beta, double* c, int64_t ldc) {
    cblas_dsyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, c, ldc);
}
inline void syrk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 std::complex<float> beta, std::complex<float>* c, int64_t ldc) {
    cblas_csyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, (void*)&alpha, a, lda,
                (void*)&beta, (void*)c, ldc);
}
inline void syrk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 std::complex<double> beta, std::complex<double>* c, int64_t ldc) {
    cblas_zsyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, (void*)&alpha, a, lda,
                (void*)&beta, (void*)c, ldc);
}
inline void herk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k, float alpha,
                 const std::complex<float>* a, int64_t lda, float beta, std::complex<float>* c,
                 int64_t ldc) {
    cblas_cherk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, (void*)c,
                ldc);
}
inline void herk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k, double alpha,
                 const std::complex<double>* a, int64_t lda, double beta, std::complex<double>* c,
                 int64_t ldc) {
    cblas_zherk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, (void*)c,
                ldc);
}
inline void sy_he_rk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                     float alpha, const float* a, int64_t lda, float beta, float* c, int64_t ldc) {
    cblas_ssyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, c, ldc);
}
inline void sy_he_rk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                     double alpha, const double* a, int64_t lda, double beta, double* c,
                     int64_t ldc) {
    cblas_dsyrk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, c, ldc);
}
inline void sy_he_rk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                     float alpha, const std::complex<float>* a, int64_t lda, float beta,
                     std::complex<float>* c, int64_t ldc) {
    cblas_cherk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, (void*)c,
                ldc);
}
inline void sy_he_rk(oneapi::mkl::uplo u, oneapi::mkl::transpose t, int64_t n, int64_t k,
                     double alpha, const std::complex<double>* a, int64_t lda, double beta,
                     std::complex<double>* c, int64_t ldc) {
    cblas_zherk(CblasColMajor, cblas_uplo(u), cblas_trans(t), n, k, alpha, a, lda, beta, (void*)c,
                ldc);
}

inline void trmm(oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose transa,
                 oneapi::mkl::diag diag, int64_t m, int64_t n, float alpha, const float* a,
                 int64_t lda, float* b, int64_t ldb) {
    cblas_strmm(CblasColMajor, cblas_side(side), cblas_uplo(uplo), cblas_trans(transa),
                cblas_diag(diag), m, n, alpha, a, lda, b, ldb);
}
inline void trmm(oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose transa,
                 oneapi::mkl::diag diag, int64_t m, int64_t n, double alpha, const double* a,
                 int64_t lda, double* b, int64_t ldb) {
    cblas_dtrmm(CblasColMajor, cblas_side(side), cblas_uplo(uplo), cblas_trans(transa),
                cblas_diag(diag), m, n, alpha, a, lda, b, ldb);
}
inline void trmm(oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose transa,
                 oneapi::mkl::diag diag, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* b, int64_t ldb) {
    cblas_ctrmm(CblasColMajor, cblas_side(side), cblas_uplo(uplo), cblas_trans(transa),
                cblas_diag(diag), m, n, (void*)&alpha, (void*)(a), lda, (void*)(b), ldb);
}
inline void trmm(oneapi::mkl::side side, oneapi::mkl::uplo uplo, oneapi::mkl::transpose transa,
                 oneapi::mkl::diag diag, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* b, int64_t ldb) {
    cblas_ztrmm(CblasColMajor, cblas_side(side), cblas_uplo(uplo), cblas_trans(transa),
                cblas_diag(diag), m, n, (void*)&alpha, (void*)(a), lda, (void*)(b), ldb);
}

inline void swap(int64_t n, float* X, int64_t incX, float* Y, int64_t incY) {
    cblas_sswap(n, X, incX, Y, incY);
}
inline void swap(int64_t n, double* X, int64_t incX, double* Y, int64_t incY) {
    cblas_dswap(n, X, incX, Y, incY);
}
inline void swap(int64_t n, std::complex<float>* X, int64_t incX, std::complex<float>* Y,
                 int64_t incY) {
    cblas_cswap(n, (void*)X, incX, (void*)Y, incY);
}
inline void swap(int64_t n, std::complex<double>* X, int64_t incX, std::complex<double>* Y,
                 int64_t incY) {
    cblas_zswap(n, (void*)X, incX, (void*)Y, incY);
}

template <typename fp_real>
fp_real lamch(char cmach);
template <>
inline float lamch(char cmach) {
    return LAPACKE_slamch(cmach);
}
template <>
inline double lamch(char cmach) {
    return LAPACKE_dlamch(cmach);
}

inline float lange(char norm, int64_t m, int64_t n, const std::complex<float>* a, int64_t lda) {
    return LAPACKE_clange(LAPACK_COL_MAJOR, norm, m, n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda);
}
inline double lange(char norm, int64_t m, int64_t n, const double* a, int64_t lda) {
    return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, m, n, a, lda);
}
inline float lange(char norm, int64_t m, int64_t n, const float* a, int64_t lda) {
    return LAPACKE_slange(LAPACK_COL_MAJOR, norm, m, n, a, lda);
}
inline double lange(char norm, int64_t m, int64_t n, const std::complex<double>* a, int64_t lda) {
    return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, m, n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda);
}

inline float lanhe(char norm, oneapi::mkl::uplo u, int64_t n, const std::complex<float>* a,
                   int64_t lda) {
    return LAPACKE_clanhe(LAPACK_COL_MAJOR, norm, to_char(u), n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda);
}
inline double lanhe(char norm, oneapi::mkl::uplo u, int64_t n, const std::complex<double>* a,
                    int64_t lda) {
    return LAPACKE_zlanhe(LAPACK_COL_MAJOR, norm, to_char(u), n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda);
}

inline float lansy(char norm, oneapi::mkl::uplo u, int64_t n, const std::complex<float>* a,
                   int64_t lda) {
    return LAPACKE_clansy(LAPACK_COL_MAJOR, norm, to_char(u), n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda);
}
inline double lansy(char norm, oneapi::mkl::uplo u, int64_t n, const double* a, int64_t lda) {
    return LAPACKE_dlansy(LAPACK_COL_MAJOR, norm, to_char(u), n, a, lda);
}
inline float lansy(char norm, oneapi::mkl::uplo u, int64_t n, const float* a, int64_t lda) {
    return LAPACKE_slansy(LAPACK_COL_MAJOR, norm, to_char(u), n, a, lda);
}
inline double lansy(char norm, oneapi::mkl::uplo u, int64_t n, const std::complex<double>* a,
                    int64_t lda) {
    return LAPACKE_zlansy(LAPACK_COL_MAJOR, norm, to_char(u), n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda);
}

inline int64_t lacpy(char u, int64_t m, int64_t n, const std::complex<float>* a, int64_t lda,
                     std::complex<float>* b, int64_t ldb) {
    return LAPACKE_clacpy(LAPACK_COL_MAJOR, u, m, n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(b), ldb);
}
inline int64_t lacpy(char u, int64_t m, int64_t n, const double* a, int64_t lda, double* b,
                     int64_t ldb) {
    return LAPACKE_dlacpy(LAPACK_COL_MAJOR, u, m, n, a, lda, b, ldb);
}
inline int64_t lacpy(char u, int64_t m, int64_t n, const float* a, int64_t lda, float* b,
                     int64_t ldb) {
    return LAPACKE_slacpy(LAPACK_COL_MAJOR, u, m, n, a, lda, b, ldb);
}
inline int64_t lacpy(char u, int64_t m, int64_t n, const std::complex<double>* a, int64_t lda,
                     std::complex<double>* b, int64_t ldb) {
    return LAPACKE_zlacpy(LAPACK_COL_MAJOR, u, m, n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(b), ldb);
}
inline int64_t lacpy(oneapi::mkl::uplo u, int64_t m, int64_t n, const std::complex<float>* a,
                     int64_t lda, std::complex<float>* b, int64_t ldb) {
    return LAPACKE_clacpy(LAPACK_COL_MAJOR, to_char(u), m, n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(b), ldb);
}
inline int64_t lacpy(oneapi::mkl::uplo u, int64_t m, int64_t n, const double* a, int64_t lda,
                     double* b, int64_t ldb) {
    return LAPACKE_dlacpy(LAPACK_COL_MAJOR, to_char(u), m, n, a, lda, b, ldb);
}
inline int64_t lacpy(oneapi::mkl::uplo u, int64_t m, int64_t n, const float* a, int64_t lda,
                     float* b, int64_t ldb) {
    return LAPACKE_slacpy(LAPACK_COL_MAJOR, to_char(u), m, n, a, lda, b, ldb);
}
inline int64_t lacpy(oneapi::mkl::uplo u, int64_t m, int64_t n, const std::complex<double>* a,
                     int64_t lda, std::complex<double>* b, int64_t ldb) {
    return LAPACKE_zlacpy(LAPACK_COL_MAJOR, to_char(u), m, n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(b), ldb);
}

inline int64_t laset(oneapi::mkl::uplo u, int64_t m, int64_t n, std::complex<float> alpha,
                     std::complex<float> beta, std::complex<float>* a, int64_t lda) {
    return LAPACKE_claset(LAPACK_COL_MAJOR, to_char(u), m, n,
                          reinterpret_cast<lapack_complex_float&>(alpha),
                          reinterpret_cast<lapack_complex_float&>(beta),
                          reinterpret_cast<lapack_complex_float*>(a), lda);
}
inline int64_t laset(oneapi::mkl::uplo u, int64_t m, int64_t n, double alpha, double beta,
                     double* a, int64_t lda) {
    return LAPACKE_dlaset(LAPACK_COL_MAJOR, to_char(u), m, n, alpha, beta, a, lda);
}
inline int64_t laset(oneapi::mkl::uplo u, int64_t m, int64_t n, float alpha, float beta, float* a,
                     int64_t lda) {
    return LAPACKE_slaset(LAPACK_COL_MAJOR, to_char(u), m, n, alpha, beta, a, lda);
}
inline int64_t laset(oneapi::mkl::uplo u, int64_t m, int64_t n, std::complex<double> alpha,
                     std::complex<double> beta, std::complex<double>* a, int64_t lda) {
    return LAPACKE_zlaset(LAPACK_COL_MAJOR, to_char(u), m, n,
                          reinterpret_cast<lapack_complex_double&>(alpha),
                          reinterpret_cast<lapack_complex_double&>(beta),
                          reinterpret_cast<lapack_complex_double*>(a), lda);
}
inline int64_t laset(char u, int64_t m, int64_t n, std::complex<float> alpha,
                     std::complex<float> beta, std::complex<float>* a, int64_t lda) {
    return LAPACKE_claset(LAPACK_COL_MAJOR, u, m, n, reinterpret_cast<lapack_complex_float&>(alpha),
                          reinterpret_cast<lapack_complex_float&>(beta),
                          reinterpret_cast<lapack_complex_float*>(a), lda);
}
inline int64_t laset(char u, int64_t m, int64_t n, double alpha, double beta, double* a,
                     int64_t lda) {
    return LAPACKE_dlaset(LAPACK_COL_MAJOR, u, m, n, alpha, beta, a, lda);
}
inline int64_t laset(char u, int64_t m, int64_t n, float alpha, float beta, float* a, int64_t lda) {
    return LAPACKE_slaset(LAPACK_COL_MAJOR, u, m, n, alpha, beta, a, lda);
}
inline int64_t laset(char u, int64_t m, int64_t n, std::complex<double> alpha,
                     std::complex<double> beta, std::complex<double>* a, int64_t lda) {
    return LAPACKE_zlaset(LAPACK_COL_MAJOR, u, m, n,
                          reinterpret_cast<lapack_complex_double&>(alpha),
                          reinterpret_cast<lapack_complex_double&>(beta),
                          reinterpret_cast<lapack_complex_double*>(a), lda);
}

inline int64_t gebrd(int64_t m, int64_t n, std::complex<float>* a, int64_t lda, float* d, float* e,
                     std::complex<float>* tauq, std::complex<float>* taup) {
    return LAPACKE_cgebrd(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda,
                          d, e, reinterpret_cast<lapack_complex_float*>(tauq),
                          reinterpret_cast<lapack_complex_float*>(taup));
}
inline int64_t gebrd(int64_t m, int64_t n, double* a, int64_t lda, double* d, double* e,
                     double* tauq, double* taup) {
    return LAPACKE_dgebrd(LAPACK_COL_MAJOR, m, n, a, lda, d, e, tauq, taup);
}
inline int64_t gebrd(int64_t m, int64_t n, float* a, int64_t lda, float* d, float* e, float* tauq,
                     float* taup) {
    return LAPACKE_sgebrd(LAPACK_COL_MAJOR, m, n, a, lda, d, e, tauq, taup);
}
inline int64_t gebrd(int64_t m, int64_t n, std::complex<double>* a, int64_t lda, double* d,
                     double* e, std::complex<double>* tauq, std::complex<double>* taup) {
    return LAPACKE_zgebrd(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda,
                          d, e, reinterpret_cast<lapack_complex_double*>(tauq),
                          reinterpret_cast<lapack_complex_double*>(taup));
}

inline int64_t geqrf(int64_t m, int64_t n, std::complex<float>* a, int64_t lda,
                     std::complex<float>* tau) {
    return LAPACKE_cgeqrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(tau));
}
inline int64_t geqrf(int64_t m, int64_t n, double* a, int64_t lda, double* tau) {
    return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
inline int64_t geqrf(int64_t m, int64_t n, float* a, int64_t lda, float* tau) {
    return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
inline int64_t geqrf(int64_t m, int64_t n, std::complex<double>* a, int64_t lda,
                     std::complex<double>* tau) {
    return LAPACKE_zgeqrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(tau));
}

inline int64_t gerqf(int64_t m, int64_t n, std::complex<float>* a, int64_t lda,
                     std::complex<float>* tau) {
    return LAPACKE_cgerqf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(tau));
}
inline int64_t gerqf(int64_t m, int64_t n, double* a, int64_t lda, double* tau) {
    return LAPACKE_dgerqf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
inline int64_t gerqf(int64_t m, int64_t n, float* a, int64_t lda, float* tau) {
    return LAPACKE_sgerqf(LAPACK_COL_MAJOR, m, n, a, lda, tau);
}
inline int64_t gerqf(int64_t m, int64_t n, std::complex<double>* a, int64_t lda,
                     std::complex<double>* tau) {
    return LAPACKE_zgerqf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(tau));
}

inline int64_t gesvd(oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                     std::complex<float>* a, int64_t lda, float* s, std::complex<float>* u,
                     int64_t ldu, std::complex<float>* vt, int64_t ldvt, float* superb) {
    return LAPACKE_cgesvd(LAPACK_COL_MAJOR, to_char(jobu), to_char(jobvt), m, n,
                          reinterpret_cast<lapack_complex_float*>(a), lda, s,
                          reinterpret_cast<lapack_complex_float*>(u), ldu,
                          reinterpret_cast<lapack_complex_float*>(vt), ldvt, superb);
}
inline int64_t gesvd(oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                     double* a, int64_t lda, double* s, double* u, int64_t ldu, double* vt,
                     int64_t ldvt, double* superb) {
    return LAPACKE_dgesvd(LAPACK_COL_MAJOR, to_char(jobu), to_char(jobvt), m, n, a, lda, s, u, ldu,
                          vt, ldvt, superb);
}
inline int64_t gesvd(oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                     float* a, int64_t lda, float* s, float* u, int64_t ldu, float* vt,
                     int64_t ldvt, float* superb) {
    return LAPACKE_sgesvd(LAPACK_COL_MAJOR, to_char(jobu), to_char(jobvt), m, n, a, lda, s, u, ldu,
                          vt, ldvt, superb);
}
inline int64_t gesvd(oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, int64_t m, int64_t n,
                     std::complex<double>* a, int64_t lda, double* s, std::complex<double>* u,
                     int64_t ldu, std::complex<double>* vt, int64_t ldvt, double* superb) {
    return LAPACKE_zgesvd(LAPACK_COL_MAJOR, to_char(jobu), to_char(jobvt), m, n,
                          reinterpret_cast<lapack_complex_double*>(a), lda, s,
                          reinterpret_cast<lapack_complex_double*>(u), ldu,
                          reinterpret_cast<lapack_complex_double*>(vt), ldvt, superb);
}

inline int64_t getrf(int64_t m, int64_t n, std::complex<float>* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t getrf(int64_t m, int64_t n, double* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, lda, reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t getrf(int64_t m, int64_t n, float* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, a, lda, reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t getrf(int64_t m, int64_t n, std::complex<double>* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n, reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}

inline int64_t heevd(oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, std::complex<float>* a,
                     int64_t lda, float* w) {
    return LAPACKE_cheevd(LAPACK_COL_MAJOR, to_char(j), to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda, w);
}
inline int64_t heevd(oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n, std::complex<double>* a,
                     int64_t lda, double* w) {
    return LAPACKE_zheevd(LAPACK_COL_MAJOR, to_char(j), to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda, w);
}

inline int64_t hegvd(int64_t itype, oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n,
                     std::complex<float>* a, int64_t lda, std::complex<float>* b, int64_t ldb,
                     float* w) {
    return LAPACKE_chegvd(LAPACK_COL_MAJOR, itype, to_char(j), to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(b), ldb, w);
}
inline int64_t hegvd(int64_t itype, oneapi::mkl::job j, oneapi::mkl::uplo u, int64_t n,
                     std::complex<double>* a, int64_t lda, std::complex<double>* b, int64_t ldb,
                     double* w) {
    return LAPACKE_zhegvd(LAPACK_COL_MAJOR, itype, to_char(j), to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(b), ldb, w);
}

inline int64_t hetrd(oneapi::mkl::uplo u, int64_t n, std::complex<float>* a, int64_t lda, float* d,
                     float* e, std::complex<float>* tau) {
    return LAPACKE_chetrd(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda, d, e,
                          reinterpret_cast<lapack_complex_float*>(tau));
}
inline int64_t hetrd(oneapi::mkl::uplo u, int64_t n, std::complex<double>* a, int64_t lda,
                     double* d, double* e, std::complex<double>* tau) {
    return LAPACKE_zhetrd(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda, d, e,
                          reinterpret_cast<lapack_complex_double*>(tau));
}

inline int64_t hetrf(oneapi::mkl::uplo u, int64_t n, std::complex<float>* a, int64_t lda,
                     int64_t* ipiv) {
    return LAPACKE_chetrf(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t hetrf(oneapi::mkl::uplo u, int64_t n, std::complex<double>* a, int64_t lda,
                     int64_t* ipiv) {
    return LAPACKE_zhetrf(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}

inline int64_t ungtr(oneapi::mkl::uplo u, int64_t n, std::complex<float>* a, int64_t lda,
                     const std::complex<float>* tau) {
    return LAPACKE_cungtr(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<const lapack_complex_float*>(tau));
}
inline int64_t ungtr(oneapi::mkl::uplo u, int64_t n, std::complex<double>* a, int64_t lda,
                     const std::complex<double>* tau) {
    return LAPACKE_zungtr(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<const lapack_complex_double*>(tau));
}

inline int64_t unmtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                     int64_t m, int64_t n, const std::complex<float>* a, int64_t lda,
                     const std::complex<float>* tau, std::complex<float>* c, int64_t ldc) {
    return LAPACKE_cunmtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<const lapack_complex_float*>(tau),
                          reinterpret_cast<lapack_complex_float*>(c), ldc);
}
inline int64_t unmtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                     int64_t m, int64_t n, const std::complex<double>* a, int64_t lda,
                     const std::complex<double>* tau, std::complex<double>* c, int64_t ldc) {
    return LAPACKE_zunmtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<const lapack_complex_double*>(tau),
                          reinterpret_cast<lapack_complex_double*>(c), ldc);
}

inline int64_t orgtr(oneapi::mkl::uplo u, int64_t n, double* a, int64_t lda, const double* tau) {
    return LAPACKE_dorgtr(LAPACK_COL_MAJOR, to_char(u), n, a, lda, tau);
}
inline int64_t orgtr(oneapi::mkl::uplo u, int64_t n, float* a, int64_t lda, const float* tau) {
    return LAPACKE_sorgtr(LAPACK_COL_MAJOR, to_char(u), n, a, lda, tau);
}

inline int64_t ormtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                     int64_t m, int64_t n, float* a, int64_t lda, const float* tau, float* c,
                     int64_t ldc) {
    return LAPACKE_sormtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n, a, lda,
                          tau, c, ldc);
}
inline int64_t ormtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                     int64_t m, int64_t n, double* a, int64_t lda, const double* tau, double* c,
                     int64_t ldc) {
    return LAPACKE_dormtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n, a, lda,
                          tau, c, ldc);
}

inline int64_t or_un_mtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                         int64_t m, int64_t n, float* a, int64_t lda, const float* tau, float* c,
                         int64_t ldc) {
    return LAPACKE_sormtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n, a, lda,
                          tau, c, ldc);
}
inline int64_t or_un_mtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                         int64_t m, int64_t n, double* a, int64_t lda, const double* tau, double* c,
                         int64_t ldc) {
    return LAPACKE_dormtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n, a, lda,
                          tau, c, ldc);
}
inline int64_t or_un_mtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                         int64_t m, int64_t n, std::complex<float>* a, int64_t lda,
                         std::complex<float>* tau, std::complex<float>* c, int64_t ldc) {
    return LAPACKE_cunmtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n,
                          reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(tau),
                          reinterpret_cast<lapack_complex_float*>(c), ldc);
}
inline int64_t or_un_mtr(oneapi::mkl::side side, oneapi::mkl::uplo u, oneapi::mkl::transpose trans,
                         int64_t m, int64_t n, std::complex<double>* a, int64_t lda,
                         std::complex<double>* tau, std::complex<double>* c, int64_t ldc) {
    return LAPACKE_zunmtr(LAPACK_COL_MAJOR, to_char(side), to_char(u), to_char(trans), m, n,
                          reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(tau),
                          reinterpret_cast<lapack_complex_double*>(c), ldc);
}

inline int64_t sytrd(oneapi::mkl::uplo u, int64_t n, float* a, int64_t lda, float* d, float* e,
                     float* tau) {
    return LAPACKE_ssytrd(LAPACK_COL_MAJOR, to_char(u), n, a, lda, d, e, tau);
}
inline int64_t sytrd(oneapi::mkl::uplo u, int64_t n, double* a, int64_t lda, double* d, double* e,
                     double* tau) {
    return LAPACKE_dsytrd(LAPACK_COL_MAJOR, to_char(u), n, a, lda, d, e, tau);
}

inline int64_t sytrf(oneapi::mkl::uplo u, int64_t n, float* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_ssytrf(LAPACK_COL_MAJOR, to_char(u), n, a, lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t sytrf(oneapi::mkl::uplo u, int64_t n, double* a, int64_t lda, int64_t* ipiv) {
    return LAPACKE_dsytrf(LAPACK_COL_MAJOR, to_char(u), n, a, lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t sytrf(oneapi::mkl::uplo u, int64_t n, std::complex<float>* a, int64_t lda,
                     int64_t* ipiv) {
    return LAPACKE_csytrf(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}
inline int64_t sytrf(oneapi::mkl::uplo u, int64_t n, std::complex<double>* a, int64_t lda,
                     int64_t* ipiv) {
    return LAPACKE_zsytrf(LAPACK_COL_MAJOR, to_char(u), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_int*>(ipiv));
}

inline void orgbr(oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k, double* a,
                  int64_t lda, const double* tau) {
    LAPACKE_dorgbr(LAPACK_COL_MAJOR, to_char(vect), m, n, k, a, lda, tau);
}
inline void orgbr(oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k, float* a,
                  int64_t lda, const float* tau) {
    LAPACKE_sorgbr(LAPACK_COL_MAJOR, to_char(vect), m, n, k, a, lda, tau);
}

inline int64_t or_un_gqr(int64_t m, int64_t n, int64_t k, float* a, int64_t lda, const float* tau) {
    return LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, n, k, a, lda, tau);
}
inline int64_t or_un_gqr(int64_t m, int64_t n, int64_t k, double* a, int64_t lda,
                         const double* tau) {
    return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, k, a, lda, tau);
}
inline int64_t or_un_gqr(int64_t m, int64_t n, int64_t k, std::complex<float>* a, int64_t lda,
                         const std::complex<float>* tau) {
    return LAPACKE_cungqr(LAPACK_COL_MAJOR, m, n, k, reinterpret_cast<lapack_complex_float*>(a),
                          lda, reinterpret_cast<const lapack_complex_float*>(tau));
}
inline int64_t or_un_gqr(int64_t m, int64_t n, int64_t k, std::complex<double>* a, int64_t lda,
                         const std::complex<double>* tau) {
    return LAPACKE_zungqr(LAPACK_COL_MAJOR, m, n, k, reinterpret_cast<lapack_complex_double*>(a),
                          lda, reinterpret_cast<const lapack_complex_double*>(tau));
}

inline int64_t or_un_mqr(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const float* a, int64_t lda, const float* tau, float* c,
                         int64_t ldc) {
    return LAPACKE_sormqr(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k, a, lda, tau, c, ldc);
}
inline int64_t or_un_mqr(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const double* a, int64_t lda, const double* tau, double* c,
                         int64_t ldc) {
    return LAPACKE_dormqr(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k, a, lda, tau, c, ldc);
}
inline int64_t or_un_mqr(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const std::complex<float>* a, int64_t lda,
                         const std::complex<float>* tau, std::complex<float>* c, int64_t ldc) {
    return LAPACKE_cunmqr(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<const lapack_complex_float*>(tau),
                          reinterpret_cast<lapack_complex_float*>(c), ldc);
}
inline int64_t or_un_mqr(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const std::complex<double>* a, int64_t lda,
                         const std::complex<double>* tau, std::complex<double>* c, int64_t ldc) {
    return LAPACKE_zunmqr(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<const lapack_complex_double*>(tau),
                          reinterpret_cast<lapack_complex_double*>(c), ldc);
}

inline int64_t or_un_grq(int64_t m, int64_t n, int64_t k, float* a, int64_t lda, const float* tau) {
    return LAPACKE_sorgrq(LAPACK_COL_MAJOR, m, n, k, a, lda, tau);
}
inline int64_t or_un_grq(int64_t m, int64_t n, int64_t k, double* a, int64_t lda,
                         const double* tau) {
    return LAPACKE_dorgrq(LAPACK_COL_MAJOR, m, n, k, a, lda, tau);
}
inline int64_t or_un_grq(int64_t m, int64_t n, int64_t k, std::complex<float>* a, int64_t lda,
                         const std::complex<float>* tau) {
    return LAPACKE_cungrq(LAPACK_COL_MAJOR, m, n, k, reinterpret_cast<lapack_complex_float*>(a),
                          lda, reinterpret_cast<const lapack_complex_float*>(tau));
}
inline int64_t or_un_grq(int64_t m, int64_t n, int64_t k, std::complex<double>* a, int64_t lda,
                         const std::complex<double>* tau) {
    return LAPACKE_zungrq(LAPACK_COL_MAJOR, m, n, k, reinterpret_cast<lapack_complex_double*>(a),
                          lda, reinterpret_cast<const lapack_complex_double*>(tau));
}

inline int64_t or_un_mrq(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const float* a, int64_t lda, const float* tau, float* c,
                         int64_t ldc) {
    return LAPACKE_sormrq(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k, a, lda, tau, c, ldc);
}
inline int64_t or_un_mrq(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const double* a, int64_t lda, const double* tau, double* c,
                         int64_t ldc) {
    return LAPACKE_dormrq(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k, a, lda, tau, c, ldc);
}
inline int64_t or_un_mrq(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const std::complex<float>* a, int64_t lda,
                         const std::complex<float>* tau, std::complex<float>* c, int64_t ldc) {
    return LAPACKE_cunmrq(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<const lapack_complex_float*>(tau),
                          reinterpret_cast<lapack_complex_float*>(c), ldc);
}
inline int64_t or_un_mrq(oneapi::mkl::side s, oneapi::mkl::transpose t, int64_t m, int64_t n,
                         int64_t k, const std::complex<double>* a, int64_t lda,
                         const std::complex<double>* tau, std::complex<double>* c, int64_t ldc) {
    return LAPACKE_zunmrq(LAPACK_COL_MAJOR, to_char(s), to_char(t), m, n, k,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<const lapack_complex_double*>(tau),
                          reinterpret_cast<lapack_complex_double*>(c), ldc);
}

inline int64_t potrf(oneapi::mkl::uplo upper_lower, int64_t n, std::complex<float>* a,
                     int64_t lda) {
    return LAPACKE_cpotrf(LAPACK_COL_MAJOR, to_char(upper_lower), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda);
}
inline int64_t potrf(oneapi::mkl::uplo upper_lower, int64_t n, double* a, int64_t lda) {
    return LAPACKE_dpotrf(LAPACK_COL_MAJOR, to_char(upper_lower), n, a, lda);
}
inline int64_t potrf(oneapi::mkl::uplo upper_lower, int64_t n, float* a, int64_t lda) {
    return LAPACKE_spotrf(LAPACK_COL_MAJOR, to_char(upper_lower), n, a, lda);
}
inline int64_t potrf(oneapi::mkl::uplo upper_lower, int64_t n, std::complex<double>* a,
                     int64_t lda) {
    return LAPACKE_zpotrf(LAPACK_COL_MAJOR, to_char(upper_lower), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda);
}

inline int64_t potrs(oneapi::mkl::uplo upper_lower, int64_t n, int64_t nrhs,
                     const std::complex<float>* a, int64_t lda, std::complex<float>* b,
                     int64_t ldb) {
    return LAPACKE_cpotrs(LAPACK_COL_MAJOR, to_char(upper_lower), n, nrhs,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(b), ldb);
}
inline int64_t potrs(oneapi::mkl::uplo upper_lower, int64_t n, int64_t nrhs, const double* a,
                     int64_t lda, double* b, int64_t ldb) {
    return LAPACKE_dpotrs(LAPACK_COL_MAJOR, to_char(upper_lower), n, nrhs, a, lda, b, ldb);
}
inline int64_t potrs(oneapi::mkl::uplo upper_lower, int64_t n, int64_t nrhs, const float* a,
                     int64_t lda, float* b, int64_t ldb) {
    return LAPACKE_spotrs(LAPACK_COL_MAJOR, to_char(upper_lower), n, nrhs, a, lda, b, ldb);
}
inline int64_t potrs(oneapi::mkl::uplo upper_lower, int64_t n, int64_t nrhs,
                     const std::complex<double>* a, int64_t lda, std::complex<double>* b,
                     int64_t ldb) {
    return LAPACKE_zpotrs(LAPACK_COL_MAJOR, to_char(upper_lower), n, nrhs,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(b), ldb);
}

inline int64_t potri(oneapi::mkl::uplo upper_lower, int64_t n, std::complex<float>* a,
                     int64_t lda) {
    return LAPACKE_cpotri(LAPACK_COL_MAJOR, to_char(upper_lower), n,
                          reinterpret_cast<lapack_complex_float*>(a), lda);
}
inline int64_t potri(oneapi::mkl::uplo upper_lower, int64_t n, double* a, int64_t lda) {
    return LAPACKE_dpotri(LAPACK_COL_MAJOR, to_char(upper_lower), n, a, lda);
}
inline int64_t potri(oneapi::mkl::uplo upper_lower, int64_t n, float* a, int64_t lda) {
    return LAPACKE_spotri(LAPACK_COL_MAJOR, to_char(upper_lower), n, a, lda);
}
inline int64_t potri(oneapi::mkl::uplo upper_lower, int64_t n, std::complex<double>* a,
                     int64_t lda) {
    return LAPACKE_zpotri(LAPACK_COL_MAJOR, to_char(upper_lower), n,
                          reinterpret_cast<lapack_complex_double*>(a), lda);
}

inline int64_t laswp(int64_t n, std::complex<float>* a, int64_t lda, int64_t k1, int64_t k2,
                     const int64_t* ipiv, int64_t incx) {
    return LAPACKE_claswp(LAPACK_COL_MAJOR, n, reinterpret_cast<lapack_complex_float*>(a), lda, k1,
                          k2, reinterpret_cast<const lapack_int*>(ipiv), incx);
}
inline int64_t laswp(int64_t n, double* a, int64_t lda, int64_t k1, int64_t k2, const int64_t* ipiv,
                     int64_t incx) {
    return LAPACKE_dlaswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2,
                          reinterpret_cast<const lapack_int*>(ipiv), incx);
}
inline int64_t laswp(int64_t n, float* a, int64_t lda, int64_t k1, int64_t k2, const int64_t* ipiv,
                     int64_t incx) {
    return LAPACKE_slaswp(LAPACK_COL_MAJOR, n, a, lda, k1, k2,
                          reinterpret_cast<const lapack_int*>(ipiv), incx);
}
inline int64_t laswp(int64_t n, std::complex<double>* a, int64_t lda, int64_t k1, int64_t k2,
                     const int64_t* ipiv, int64_t incx) {
    return LAPACKE_zlaswp(LAPACK_COL_MAJOR, n, reinterpret_cast<lapack_complex_double*>(a), lda, k1,
                          k2, reinterpret_cast<const lapack_int*>(ipiv), incx);
}

inline void ungbr(oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k,
                  std::complex<float>* a, int64_t lda, const std::complex<float>* tau) {
    LAPACKE_cungbr(LAPACK_COL_MAJOR, to_char(vect), m, n, k,
                   reinterpret_cast<lapack_complex_float*>(a), lda,
                   reinterpret_cast<const lapack_complex_float*>(tau));
}
inline void ungbr(oneapi::mkl::generate vect, int64_t m, int64_t n, int64_t k,
                  std::complex<double>* a, int64_t lda, const std::complex<double>* tau) {
    LAPACKE_zungbr(LAPACK_COL_MAJOR, to_char(vect), m, n, k,
                   reinterpret_cast<lapack_complex_double*>(a), lda,
                   reinterpret_cast<const lapack_complex_double*>(tau));
}

inline int64_t trtrs(oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                     int64_t n, int64_t nrhs, const float* a, int64_t lda, float* b, int64_t ldb) {
    return LAPACKE_strtrs(LAPACK_COL_MAJOR, to_char(uplo), to_char(trans), to_char(diag), n, nrhs,
                          a, lda, b, ldb);
}
inline int64_t trtrs(oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                     int64_t n, int64_t nrhs, const double* a, int64_t lda, double* b,
                     int64_t ldb) {
    return LAPACKE_dtrtrs(LAPACK_COL_MAJOR, to_char(uplo), to_char(trans), to_char(diag), n, nrhs,
                          a, lda, b, ldb);
}
inline int64_t trtrs(oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                     int64_t n, int64_t nrhs, const std::complex<float>* a, int64_t lda,
                     std::complex<float>* b, int64_t ldb) {
    return LAPACKE_ctrtrs(LAPACK_COL_MAJOR, to_char(uplo), to_char(trans), to_char(diag), n, nrhs,
                          reinterpret_cast<const lapack_complex_float*>(a), lda,
                          reinterpret_cast<lapack_complex_float*>(b), ldb);
}
inline int64_t trtrs(oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                     int64_t n, int64_t nrhs, const std::complex<double>* a, int64_t lda,
                     std::complex<double>* b, int64_t ldb) {
    return LAPACKE_ztrtrs(LAPACK_COL_MAJOR, to_char(uplo), to_char(trans), to_char(diag), n, nrhs,
                          reinterpret_cast<const lapack_complex_double*>(a), lda,
                          reinterpret_cast<lapack_complex_double*>(b), ldb);
}

} //namespace reference
