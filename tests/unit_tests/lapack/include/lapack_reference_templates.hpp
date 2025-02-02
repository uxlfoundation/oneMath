/*******************************************************************************
* Copyright 2025 SiPearl
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

#ifndef _REFERENCE_LAPACK_TEMPLATES_HPP__
#define _REFERENCE_LAPACK_TEMPLATES_HPP__

#include "oneapi/math/detail/exceptions.hpp"
#include "oneapi/math/detail/config.hpp"
#include <string>
#include "cblas.h"
#include "lapacke.h"

#ifdef __linux__
#include <dlfcn.h>
#define LIB_TYPE                void*
#define GET_LIB_HANDLE(libname) dlopen((libname), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)
#define GET_FUNC(lib, fn)       dlsym(lib, (fn))
#elif defined(_WIN64)
#include <windows.h>
#define LIB_TYPE                HINSTANCE
#define GET_LIB_HANDLE(libname) LoadLibrary(libname)
#define GET_FUNC(lib, fn)       GetProcAddress((lib), (fn))
#endif

extern "C" {
static LIB_TYPE h_liblapacke = NULL;

static LIB_TYPE lapacke_library() {
    if (h_liblapacke == NULL) {
        h_liblapacke = GET_LIB_HANDLE(ONEMATH_REF_LAPACKE_LIBNAME);
        if (h_liblapacke == NULL) {
            throw oneapi::math::library_not_found(
                "LAPACKE", "lapacke_library()",
                std::string("failed to load LAPACKE library ") + ONEMATH_REF_LAPACKE_LIBNAME);
        }
    }
    return h_liblapacke;
}

static LIB_TYPE h_libcblas = NULL;
static LIB_TYPE cblas_library() {
    if (h_libcblas == NULL) {
        h_libcblas = GET_LIB_HANDLE(ONEMATH_REF_CBLAS_LIBNAME);
        if (h_libcblas == NULL) {
            throw oneapi::math::library_not_found(
                "BLAS", "cblas_library()",
                std::string("failed to load CBLAS library ") + ONEMATH_REF_CBLAS_LIBNAME);
        }
    }
    return h_libcblas;
}

static void (*cblas_sgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const float alpha,
                             const float* a, const int lda, const float* b, const int ldb,
                             const float beta, float* c, const int ldc);
static void (*cblas_dgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const double alpha,
                             const double* a, const int lda, const double* b, const int ldb,
                             const double beta, double* c, const int ldc);
static void (*cblas_cgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const void* alpha,
                             const void* a, const int lda, const void* b, const int ldb,
                             const void* beta, void* c, const int ldc);
static void (*cblas_zgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const void* alpha,
                             const void* a, const int lda, const void* b, const int ldb,
                             const void* beta, void* c, const int ldc);
static void (*cblas_ssyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const float alpha, const float* a,
                             const int lda, const float beta, float* c, const int ldc);
static void (*cblas_dsyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const double alpha, const double* a,
                             const int lda, const double beta, double* c, const int ldc);
static void (*cblas_csyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const void* alpha, const void* a,
                             const int lda, const void* beta, void* c, const int ldc);
static void (*cblas_zsyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const void* alpha, const void* a,
                             const int lda, const void* beta, void* c, const int ldc);
static void (*cblas_cherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const float alpha, const void* a,
                             const int lda, const float beta, void* c, const int ldc);
static void (*cblas_zherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const double alpha, const void* a,
                             const int lda, const double beta, void* c, const int ldc);
static void (*cblas_strmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const float alpha, const float* a, const int lda, float* b,
                             const int ldb);
static void (*cblas_dtrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const double alpha, const double* a, const int lda, double* b,
                             const int ldb);
static void (*cblas_ctrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void* alpha, const void* a, const int lda, void* b,
                             const int ldb);
static void (*cblas_ztrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void* alpha, const void* a, const int lda, void* b,
                             const int ldb);
static void (*cblas_sswap_p)(const int n, float* x, const int incx, float* y, const int incy);
static void (*cblas_dswap_p)(const int n, double* x, const int incx, double* y, const int incy);
static void (*cblas_cswap_p)(const int n, void* x, const int incx, void* y, const int incy);
static void (*cblas_zswap_p)(const int n, void* x, const int incx, void* y, const int incy);

static void cblas_sgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const float alpha,
                                const float* a, const int lda, const float* b, const int ldb,
                                const float beta, float* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_sgemm_p == NULL)
            cblas_sgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const float alpha, const float* a,
                          const int lda, const float* b, const int ldb, const float beta, float* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_sgemm");
        if (cblas_sgemm_p != NULL)
            cblas_sgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const double alpha,
                                const double* a, const int lda, const double* b, const int ldb,
                                const double beta, double* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dgemm_p == NULL)
            cblas_dgemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa,
                                      CBLAS_TRANSPOSE transb, const int m, const int n, const int k,
                                      const double alpha, const double* a, const int lda,
                                      const double* b, const int ldb, const double beta, double* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_dgemm");
        if (cblas_dgemm_p != NULL)
            cblas_dgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_cgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const void* alpha,
                                const void* a, const int lda, const void* b, const int ldb,
                                const void* beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cgemm_p == NULL)
            cblas_cgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const void* alpha, const void* a,
                          const int lda, const void* b, const int ldb, const void* beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_cgemm");
        if (cblas_cgemm_p != NULL)
            cblas_cgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const void* alpha,
                                const void* a, const int lda, const void* b, const int ldb,
                                const void* beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zgemm_p == NULL)
            cblas_zgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const void* alpha, const void* a,
                          const int lda, const void* b, const int ldb, const void* beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zgemm");
        if (cblas_zgemm_p != NULL)
            cblas_zgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_ssyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const float alpha, const float* a,
                                const int lda, const float beta, float* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssyrk_p == NULL)
            cblas_ssyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const float alpha, const float* a,
                                      const int lda, const float beta, float* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_ssyrk");
        if (cblas_ssyrk_p != NULL)
            cblas_ssyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_dsyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const double alpha, const double* a,
                                const int lda, const double beta, double* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsyrk_p == NULL)
            cblas_dsyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const double alpha, const double* a,
                                      const int lda, const double beta, double* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_dsyrk");
        if (cblas_dsyrk_p != NULL)
            cblas_dsyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_csyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const void* alpha, const void* a,
                                const int lda, const void* beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csyrk_p == NULL)
            cblas_csyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const void* alpha, const void* a,
                                      const int lda, const void* beta, void* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_csyrk");
        if (cblas_csyrk_p != NULL)
            cblas_csyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_zsyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const void* alpha, const void* a,
                                const int lda, const void* beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsyrk_p == NULL)
            cblas_zsyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const void* alpha, const void* a,
                                      const int lda, const void* beta, void* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_zsyrk");
        if (cblas_zsyrk_p != NULL)
            cblas_zsyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_cherk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const float alpha, const void* a,
                                const int lda, const float beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cherk_p == NULL)
            cblas_cherk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const float alpha, const void* a,
                                      const int lda, const float beta, void* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_cherk");
        if (cblas_cherk_p != NULL)
            cblas_cherk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_zherk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const double alpha, const void* a,
                                const int lda, const double beta, void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zherk_p == NULL)
            cblas_zherk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const double alpha, const void* a,
                                      const int lda, const double beta, void* c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_zherk");
        if (cblas_zherk_p != NULL)
            cblas_zherk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}
static void cblas_strmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const float alpha, const float* a, const int lda, float* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_strmm_p == NULL)
            cblas_strmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const float alpha, const float* a, const int lda,
                                      float* b, const int ldb))GET_FUNC(h_libcblas, "cblas_strmm");
        if (cblas_strmm_p != NULL)
            cblas_strmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_dtrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const double alpha, const double* a, const int lda, double* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_dtrmm_p == NULL)
            cblas_dtrmm_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                CBLAS_DIAG diag, const int m, const int n, const double alpha, const double* a,
                const int lda, double* b, const int ldb))GET_FUNC(h_libcblas, "cblas_dtrmm");
        if (cblas_dtrmm_p != NULL)
            cblas_dtrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ctrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void* alpha, const void* a, const int lda, void* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ctrmm_p == NULL)
            cblas_ctrmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      void* b, const int ldb))GET_FUNC(h_libcblas, "cblas_ctrmm");
        if (cblas_ctrmm_p != NULL)
            cblas_ctrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ztrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void* alpha, const void* a, const int lda, void* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ztrmm_p == NULL)
            cblas_ztrmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      void* b, const int ldb))GET_FUNC(h_libcblas, "cblas_ztrmm");
        if (cblas_ztrmm_p != NULL)
            cblas_ztrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_sswap_wrapper(const int n, float* x, const int incx, float* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sswap_p == NULL)
            cblas_sswap_p = (void (*)(const int n, float* x, const int incx, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sswap");
        if (cblas_sswap_p != NULL)
            cblas_sswap_p(n, x, incx, y, incy);
    }
}

static void cblas_dswap_wrapper(const int n, double* x, const int incx, double* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dswap_p == NULL)
            cblas_dswap_p = (void (*)(const int n, double* x, const int incx, double* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dswap");
        if (cblas_dswap_p != NULL)
            cblas_dswap_p(n, x, incx, y, incy);
    }
}

static void cblas_cswap_wrapper(const int n, void* x, const int incx, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cswap_p == NULL)
            cblas_cswap_p = (void (*)(const int n, void* x, const int incx, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_cswap");
        if (cblas_cswap_p != NULL)
            cblas_cswap_p(n, x, incx, y, incy);
    }
}

static void cblas_zswap_wrapper(const int n, void* x, const int incx, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zswap_p == NULL)
            cblas_zswap_p = (void (*)(const int n, void* x, const int incx, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zswap");
        if (cblas_zswap_p != NULL)
            cblas_zswap_p(n, x, incx, y, incy);
    }
}

static float (*LAPACKE_slamch_p)(char cmach);
static double (*LAPACKE_dlamch_p)(char cmach);

static float (*LAPACKE_clange_p)(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                 const lapack_complex_float* a, lapack_int lda);
static double (*LAPACKE_dlange_p)(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                  const double* a, lapack_int lda);
static float (*LAPACKE_slange_p)(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                 const float* a, lapack_int lda);
static double (*LAPACKE_zlange_p)(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                  const lapack_complex_double* a, lapack_int lda);

static float (*LAPACKE_clanhe_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                 const lapack_complex_float* a, lapack_int lda);

static double (*LAPACKE_zlanhe_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                  const lapack_complex_double* a, lapack_int lda);

static float (*LAPACKE_clansy_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                 const lapack_complex_float* a, lapack_int lda);

static double (*LAPACKE_dlansy_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                  const double* a, lapack_int lda);

static float (*LAPACKE_slansy_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                 const float* a, lapack_int lda);

static double (*LAPACKE_zlansy_p)(int matrix_layout, char norm, char uplo, lapack_int n,
                                  const lapack_complex_double* a, lapack_int lda);

static lapack_int (*LAPACKE_ssygvd_p)(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                      lapack_int n, float* a, lapack_int lda, float* b,
                                      lapack_int ldb, float* w);

static lapack_int (*LAPACKE_dsygvd_p)(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                      lapack_int n, double* a, lapack_int lda, double* b,
                                      lapack_int ldb, double* w);

static lapack_int (*LAPACKE_ssyevd_p)(int matrix_layout, char jobz, char uplo, lapack_int n,
                                      float* a, lapack_int lda, float* w);

static lapack_int (*LAPACKE_dsyevd_p)(int matrix_layout, char jobz, char uplo, lapack_int n,
                                      double* a, lapack_int lda, double* w);

static lapack_int (*LAPACKE_clacpy_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      const lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb);

static lapack_int (*LAPACKE_dlacpy_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      const double* a, lapack_int lda, double* b, lapack_int ldb);

static lapack_int (*LAPACKE_slacpy_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      const float* a, lapack_int lda, float* b, lapack_int ldb);

static lapack_int (*LAPACKE_zlacpy_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      const lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb);

static lapack_int (*LAPACKE_claset_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      lapack_complex_float alpha, lapack_complex_float beta,
                                      lapack_complex_float* a, lapack_int lda);

static lapack_int (*LAPACKE_dlaset_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      double alpha, double beta, double* a, lapack_int lda);

static lapack_int (*LAPACKE_slaset_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      float alpha, float beta, float* a, lapack_int lda);

static lapack_int (*LAPACKE_zlaset_p)(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                      lapack_complex_double alpha, lapack_complex_double beta,
                                      lapack_complex_double* a, lapack_int lda);

static lapack_int (*LAPACKE_cgebrd_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, float* d, float* e,
                                      lapack_complex_float* tauq, lapack_complex_float* taup);

static lapack_int (*LAPACKE_dgebrd_p)(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, double* d, double* e, double* tauq,
                                      double* taup);

static lapack_int (*LAPACKE_sgebrd_p)(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, float* d, float* e, float* tauq, float* taup);

static lapack_int (*LAPACKE_zgebrd_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, double* d,
                                      double* e, lapack_complex_double* tauq,
                                      lapack_complex_double* taup);

static lapack_int (*LAPACKE_cgeqrf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* tau);

static lapack_int (*LAPACKE_dgeqrf_p)(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, double* tau);

static lapack_int (*LAPACKE_sgeqrf_p)(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, float* tau);

static lapack_int (*LAPACKE_zgeqrf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* tau);

static lapack_int (*LAPACKE_cgerqf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* tau);

static lapack_int (*LAPACKE_dgerqf_p)(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, double* tau);

static lapack_int (*LAPACKE_sgerqf_p)(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, float* tau);

static lapack_int (*LAPACKE_cgesvd_p)(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                      lapack_int n, lapack_complex_float* a, lapack_int lda,
                                      float* s, lapack_complex_float* u, lapack_int ldu,
                                      lapack_complex_float* vt, lapack_int ldvt, float* superb);

static lapack_int (*LAPACKE_zgerqf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* tau);

static lapack_int (*LAPACKE_dgesvd_p)(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                      lapack_int n, double* a, lapack_int lda, double* s, double* u,
                                      lapack_int ldu, double* vt, lapack_int ldvt, double* superb);

static lapack_int (*LAPACKE_sgesvd_p)(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                      lapack_int n, float* a, lapack_int lda, float* s, float* u,
                                      lapack_int ldu, float* vt, lapack_int ldvt, float* superb);

static lapack_int (*LAPACKE_zgesvd_p)(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                      lapack_int n, lapack_complex_double* a, lapack_int lda,
                                      double* s, lapack_complex_double* u, lapack_int ldu,
                                      lapack_complex_double* vt, lapack_int ldvt, double* superb);

static lapack_int (*LAPACKE_cgetrf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_dgetrf_p)(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                      lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_sgetrf_p)(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                      lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_zgetrf_p)(int matrix_layout, lapack_int m, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_cheevd_p)(int matrix_layout, char jobz, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, float* w);

static lapack_int (*LAPACKE_zheevd_p)(int matrix_layout, char jobz, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, double* w);

static lapack_int (*LAPACKE_chegvd_p)(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                      lapack_int n, lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb, float* w);

static lapack_int (*LAPACKE_zhegvd_p)(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                      lapack_int n, lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb, double* w);

static lapack_int (*LAPACKE_chetrd_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, float* d, float* e,
                                      lapack_complex_float* tau);

static lapack_int (*LAPACKE_zhetrd_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, double* d,
                                      double* e, lapack_complex_double* tau);

static lapack_int (*LAPACKE_chetrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_zhetrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_cungtr_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);

static lapack_int (*LAPACKE_zungtr_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);

static lapack_int (*LAPACKE_cunmtr_p)(int matrix_layout, char side, char uplo, char trans,
                                      lapack_int m, lapack_int n, const lapack_complex_float* a,
                                      lapack_int lda, const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);

static lapack_int (*LAPACKE_zunmtr_p)(int matrix_layout, char side, char uplo, char trans,
                                      lapack_int m, lapack_int n, const lapack_complex_double* a,
                                      lapack_int lda, const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);

static lapack_int (*LAPACKE_dorgtr_p)(int matrix_layout, char uplo, lapack_int n, double* a,
                                      lapack_int lda, const double* tau);

static lapack_int (*LAPACKE_sorgtr_p)(int matrix_layout, char uplo, lapack_int n, float* a,
                                      lapack_int lda, const float* tau);

static lapack_int (*LAPACKE_sormtr_p)(int matrix_layout, char side, char uplo, char trans,
                                      lapack_int m, lapack_int n, const float* a, lapack_int lda,
                                      const float* tau, float* c, lapack_int ldc);

static lapack_int (*LAPACKE_dormtr_p)(int matrix_layout, char side, char uplo, char trans,
                                      lapack_int m, lapack_int n, const double* a, lapack_int lda,
                                      const double* tau, double* c, lapack_int ldc);

static lapack_int (*LAPACKE_ssytrd_p)(int matrix_layout, char uplo, lapack_int n, float* a,
                                      lapack_int lda, float* d, float* e, float* tau);

static lapack_int (*LAPACKE_dsytrd_p)(int matrix_layout, char uplo, lapack_int n, double* a,
                                      lapack_int lda, double* d, double* e, double* tau);

static lapack_int (*LAPACKE_csytrd_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, float* d, float* e,
                                      lapack_complex_float* tau);

static lapack_int (*LAPACKE_zsytrd_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, double* d,
                                      double* e, lapack_complex_double* tau);

static lapack_int (*LAPACKE_ssytrf_p)(int matrix_layout, char uplo, lapack_int n, float* a,
                                      lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_dsytrf_p)(int matrix_layout, char uplo, lapack_int n, double* a,
                                      lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_csytrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda, lapack_int* ipiv);

static lapack_int (*LAPACKE_zsytrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda, lapack_int* ipiv);

static void (*LAPACKE_dorgbr_p)(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                lapack_int k, double* a, lapack_int lda, const double* tau);

static void (*LAPACKE_sorgbr_p)(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                lapack_int k, float* a, lapack_int lda, const float* tau);

static lapack_int (*LAPACKE_cungqr_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);

static lapack_int (*LAPACKE_zungqr_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);

static lapack_int (*LAPACKE_sormqr_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const float* a, lapack_int lda,
                                      const float* tau, float* c, lapack_int ldc);

static lapack_int (*LAPACKE_dormqr_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const double* a, lapack_int lda,
                                      const double* tau, double* c, lapack_int ldc);

static lapack_int (*LAPACKE_dorgqr_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      double* a, lapack_int lda, const double* tau);

static lapack_int (*LAPACKE_sorgqr_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      float* a, lapack_int lda, const float* tau);

static lapack_int (*LAPACKE_cunmqr_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const lapack_complex_float* a,
                                      lapack_int lda, const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);

static lapack_int (*LAPACKE_zunmqr_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const lapack_complex_double* a,
                                      lapack_int lda, const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);

static lapack_int (*LAPACKE_dorgrq_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      double* a, lapack_int lda, const double* tau);

static lapack_int (*LAPACKE_sorgrq_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      float* a, lapack_int lda, const float* tau);

static lapack_int (*LAPACKE_cungrq_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);

static lapack_int (*LAPACKE_zungrq_p)(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                                      lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);

static lapack_int (*LAPACKE_sormrq_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const float* a, lapack_int lda,
                                      const float* tau, float* c, lapack_int ldc);

static lapack_int (*LAPACKE_dormrq_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const double* a, lapack_int lda,
                                      const double* tau, double* c, lapack_int ldc);

static lapack_int (*LAPACKE_cunmrq_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const lapack_complex_float* a,
                                      lapack_int lda, const lapack_complex_float* tau,
                                      lapack_complex_float* c, lapack_int ldc);

static lapack_int (*LAPACKE_zunmrq_p)(int matrix_layout, char side, char trans, lapack_int m,
                                      lapack_int n, lapack_int k, const lapack_complex_double* a,
                                      lapack_int lda, const lapack_complex_double* tau,
                                      lapack_complex_double* c, lapack_int ldc);

static lapack_int (*LAPACKE_cpotrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda);

static lapack_int (*LAPACKE_dpotrf_p)(int matrix_layout, char uplo, lapack_int n, double* a,
                                      lapack_int lda);

static lapack_int (*LAPACKE_spotrf_p)(int matrix_layout, char uplo, lapack_int n, float* a,
                                      lapack_int lda);

static lapack_int (*LAPACKE_zpotrf_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda);

static lapack_int (*LAPACKE_cpotrs_p)(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_float* a, lapack_int lda,
                                      lapack_complex_float* b, lapack_int ldb);

static lapack_int (*LAPACKE_dpotrs_p)(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                                      const double* a, lapack_int lda, double* b, lapack_int ldb);

static lapack_int (*LAPACKE_spotrs_p)(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                                      const float* a, lapack_int lda, float* b, lapack_int ldb);

static lapack_int (*LAPACKE_zpotrs_p)(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                                      const lapack_complex_double* a, lapack_int lda,
                                      lapack_complex_double* b, lapack_int ldb);

static lapack_int (*LAPACKE_cpotri_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_float* a, lapack_int lda);

static lapack_int (*LAPACKE_dpotri_p)(int matrix_layout, char uplo, lapack_int n, double* a,
                                      lapack_int lda);

static lapack_int (*LAPACKE_spotri_p)(int matrix_layout, char uplo, lapack_int n, float* a,
                                      lapack_int lda);

static lapack_int (*LAPACKE_zpotri_p)(int matrix_layout, char uplo, lapack_int n,
                                      lapack_complex_double* a, lapack_int lda);

static lapack_int (*LAPACKE_claswp_p)(int matrix_layout, lapack_int n, lapack_complex_float* a,
                                      lapack_int lda, lapack_int k1, lapack_int k2,
                                      const lapack_int* ipiv, lapack_int incx);

static lapack_int (*LAPACKE_dlaswp_p)(int matrix_layout, lapack_int n, double* a, lapack_int lda,
                                      lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                                      lapack_int incx);

static lapack_int (*LAPACKE_slaswp_p)(int matrix_layout, lapack_int n, float* a, lapack_int lda,
                                      lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                                      lapack_int incx);

static lapack_int (*LAPACKE_zlaswp_p)(int matrix_layout, lapack_int n, lapack_complex_double* a,
                                      lapack_int lda, lapack_int k1, lapack_int k2,
                                      const lapack_int* ipiv, lapack_int incx);

static lapack_int (*LAPACKE_cungbr_p)(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                      lapack_int k, lapack_complex_float* a, lapack_int lda,
                                      const lapack_complex_float* tau);

static lapack_int (*LAPACKE_zungbr_p)(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                      lapack_int k, lapack_complex_double* a, lapack_int lda,
                                      const lapack_complex_double* tau);

static lapack_int (*LAPACKE_strtrs_p)(int matrix_layout, char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs, const float* a, lapack_int lda,
                                      float* b, lapack_int ldb);

static lapack_int (*LAPACKE_dtrtrs_p)(int matrix_layout, char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs, const double* a,
                                      lapack_int lda, double* b, lapack_int ldb);

static lapack_int (*LAPACKE_ctrtrs_p)(int matrix_layout, char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs, const lapack_complex_float* a,
                                      lapack_int lda, lapack_complex_float* b, lapack_int ldb);

static lapack_int (*LAPACKE_ztrtrs_p)(int matrix_layout, char uplo, char trans, char diag,
                                      lapack_int n, lapack_int nrhs, const lapack_complex_double* a,
                                      lapack_int lda, lapack_complex_double* b, lapack_int ldb);

static float LAPACKE_slamch_wrapper(char cmach) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slamch_p == NULL)
            LAPACKE_slamch_p = (float (*)(char))GET_FUNC(h_liblapacke, "LAPACKE_slamch");
        if (LAPACKE_slamch_p != NULL)
            return LAPACKE_slamch_p(cmach);
    }
    return 0.0f;
}

static double LAPACKE_dlamch_wrapper(char cmach) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlamch_p == NULL)
            LAPACKE_dlamch_p = (double (*)(char))GET_FUNC(h_liblapacke, "LAPACKE_dlamch");
        if (LAPACKE_dlamch_p != NULL)
            return LAPACKE_dlamch_p(cmach);
    }
    return 0.0;
}

static float LAPACKE_clange_wrapper(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                    const lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_clange_p == NULL)
            LAPACKE_clange_p =
                (float (*)(int, char, lapack_int, lapack_int, const lapack_complex_float*,
                           lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_clange");
        if (LAPACKE_clange_p != NULL)
            return LAPACKE_clange_p(matrix_layout, norm, m, n, a, lda);
    }
    return 0.0f;
}

static double LAPACKE_dlange_wrapper(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                     const double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlange_p == NULL)
            LAPACKE_dlange_p = (double (*)(int, char, lapack_int, lapack_int, const double*,
                                           lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dlange");
        if (LAPACKE_dlange_p != NULL)
            return LAPACKE_dlange_p(matrix_layout, norm, m, n, a, lda);
    }
    return 0.0;
}

static float LAPACKE_slange_wrapper(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                    const float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slange_p == NULL)
            LAPACKE_slange_p = (float (*)(int, char, lapack_int, lapack_int, const float*,
                                          lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_slange");
        if (LAPACKE_slange_p != NULL)
            return LAPACKE_slange_p(matrix_layout, norm, m, n, a, lda);
    }
    return 0.0f;
}

static double LAPACKE_zlange_wrapper(int matrix_layout, char norm, lapack_int m, lapack_int n,
                                     const lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlange_p == NULL)
            LAPACKE_zlange_p =
                (double (*)(int, char, lapack_int, lapack_int, const lapack_complex_double*,
                            lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlange");
        if (LAPACKE_zlange_p != NULL)
            return LAPACKE_zlange_p(matrix_layout, norm, m, n, a, lda);
    }
    return 0.0;
}
static float LAPACKE_clanhe_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                    const lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_clanhe_p == NULL)
            LAPACKE_clanhe_p = (float (*)(int, char, char, lapack_int, const lapack_complex_float*,
                                          lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_clanhe");
        if (LAPACKE_clanhe_p != NULL)
            return LAPACKE_clanhe_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0f;
}
static double LAPACKE_zlanhe_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                     const lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlanhe_p == NULL)
            LAPACKE_zlanhe_p =
                (double (*)(int, char, char, lapack_int, const lapack_complex_double*,
                            lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlanhe");
        if (LAPACKE_zlanhe_p != NULL)
            return LAPACKE_zlanhe_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0;
}
static float LAPACKE_clansy_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                    const lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_clansy_p == NULL)
            LAPACKE_clansy_p = (float (*)(int, char, char, lapack_int, const lapack_complex_float*,
                                          lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_clansy");
        if (LAPACKE_clansy_p != NULL)
            return LAPACKE_clansy_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0f;
}
static double LAPACKE_dlansy_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                     const double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlansy_p == NULL)
            LAPACKE_dlansy_p = (double (*)(int, char, char, lapack_int, const double*,
                                           lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dlansy");
        if (LAPACKE_dlansy_p != NULL)
            return LAPACKE_dlansy_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0;
}
static float LAPACKE_slansy_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                    const float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slansy_p == NULL)
            LAPACKE_slansy_p = (float (*)(int, char, char, lapack_int, const float*,
                                          lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_slansy");
        if (LAPACKE_slansy_p != NULL)
            return LAPACKE_slansy_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0f;
}
static double LAPACKE_zlansy_wrapper(int matrix_layout, char norm, char uplo, lapack_int n,
                                     const lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlansy_p == NULL)
            LAPACKE_zlansy_p =
                (double (*)(int, char, char, lapack_int, const lapack_complex_double*,
                            lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlansy");
        if (LAPACKE_zlansy_p != NULL)
            return LAPACKE_zlansy_p(matrix_layout, norm, uplo, n, a, lda);
    }
    return 0.0;
}
static lapack_int LAPACKE_ssygvd_wrapper(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                         lapack_int n, float* a, lapack_int lda, float* b,
                                         lapack_int ldb, float* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ssygvd_p == NULL)
            LAPACKE_ssygvd_p =
                (lapack_int(*)(int, lapack_int, char, char, lapack_int, float*, lapack_int, float*,
                               lapack_int, float*))GET_FUNC(h_liblapacke, "LAPACKE_ssygvd");
        if (LAPACKE_ssygvd_p != NULL)
            return LAPACKE_ssygvd_p(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    }
    return -1;
}
static lapack_int LAPACKE_dsygvd_wrapper(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                         lapack_int n, double* a, lapack_int lda, double* b,
                                         lapack_int ldb, double* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dsygvd_p == NULL)
            LAPACKE_dsygvd_p = (lapack_int(*)(int, lapack_int, char, char, lapack_int, double*,
                                              lapack_int, double*, lapack_int,
                                              double*))GET_FUNC(h_liblapacke, "LAPACKE_dsygvd");
        if (LAPACKE_dsygvd_p != NULL)
            return LAPACKE_dsygvd_p(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    }
    return -1;
}
static lapack_int LAPACKE_ssyevd_wrapper(int matrix_layout, char jobz, char uplo, lapack_int n,
                                         float* a, lapack_int lda, float* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ssyevd_p == NULL)
            LAPACKE_ssyevd_p = (lapack_int(*)(int, char, char, lapack_int, float*, lapack_int,
                                              float*))GET_FUNC(h_liblapacke, "LAPACKE_ssyevd");
        if (LAPACKE_ssyevd_p != NULL)
            return LAPACKE_ssyevd_p(matrix_layout, jobz, uplo, n, a, lda, w);
    }
    return -1;
}
static lapack_int LAPACKE_dsyevd_wrapper(int matrix_layout, char jobz, char uplo, lapack_int n,
                                         double* a, lapack_int lda, double* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dsyevd_p == NULL)
            LAPACKE_dsyevd_p = (lapack_int(*)(int, char, char, lapack_int, double*, lapack_int,
                                              double*))GET_FUNC(h_liblapacke, "LAPACKE_dsyevd");
        if (LAPACKE_dsyevd_p != NULL)
            return LAPACKE_dsyevd_p(matrix_layout, jobz, uplo, n, a, lda, w);
    }
    return -1;
}
static lapack_int LAPACKE_clacpy_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         const lapack_complex_float* a, lapack_int lda,
                                         lapack_complex_float* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_clacpy_p == NULL)
            LAPACKE_clacpy_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, const lapack_complex_float*, lapack_int,
                lapack_complex_float*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_clacpy");
        if (LAPACKE_clacpy_p != NULL)
            return LAPACKE_clacpy_p(matrix_layout, uplo, m, n, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_dlacpy_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         const double* a, lapack_int lda, double* b,
                                         lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlacpy_p == NULL)
            LAPACKE_dlacpy_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, const double*, lapack_int,
                               double*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dlacpy");
        if (LAPACKE_dlacpy_p != NULL)
            return LAPACKE_dlacpy_p(matrix_layout, uplo, m, n, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_slacpy_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         const float* a, lapack_int lda, float* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slacpy_p == NULL)
            LAPACKE_slacpy_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, const float*, lapack_int, float*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_slacpy");
        if (LAPACKE_slacpy_p != NULL)
            return LAPACKE_slacpy_p(matrix_layout, uplo, m, n, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_zlacpy_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         const lapack_complex_double* a, lapack_int lda,
                                         lapack_complex_double* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlacpy_p == NULL)
            LAPACKE_zlacpy_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, const lapack_complex_double*, lapack_int,
                lapack_complex_double*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlacpy");
        if (LAPACKE_zlacpy_p != NULL)
            return LAPACKE_zlacpy_p(matrix_layout, uplo, m, n, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_claset_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         lapack_complex_float alpha, lapack_complex_float beta,
                                         lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_claset_p == NULL)
            LAPACKE_claset_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, lapack_complex_float, lapack_complex_float,
                lapack_complex_float*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_claset");
        if (LAPACKE_claset_p != NULL)
            return LAPACKE_claset_p(matrix_layout, uplo, m, n, alpha, beta, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_dlaset_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         double alpha, double beta, double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlaset_p == NULL)
            LAPACKE_dlaset_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, double, double, double*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dlaset");
        if (LAPACKE_dlaset_p != NULL)
            return LAPACKE_dlaset_p(matrix_layout, uplo, m, n, alpha, beta, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_slaset_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         float alpha, float beta, float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slaset_p == NULL)
            LAPACKE_slaset_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, float, float, float*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_slaset");
        if (LAPACKE_slaset_p != NULL)
            return LAPACKE_slaset_p(matrix_layout, uplo, m, n, alpha, beta, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_zlaset_wrapper(int matrix_layout, char uplo, lapack_int m, lapack_int n,
                                         lapack_complex_double alpha, lapack_complex_double beta,
                                         lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlaset_p == NULL)
            LAPACKE_zlaset_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, lapack_complex_double, lapack_complex_double,
                lapack_complex_double*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlaset");
        if (LAPACKE_zlaset_p != NULL)
            return LAPACKE_zlaset_p(matrix_layout, uplo, m, n, alpha, beta, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_cgebrd_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda, float* d,
                                         float* e, lapack_complex_float* tauq,
                                         lapack_complex_float* taup) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cgebrd_p == NULL)
            LAPACKE_cgebrd_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                               float*, float*, lapack_complex_float*,
                               lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cgebrd");
        if (LAPACKE_cgebrd_p != NULL)
            return LAPACKE_cgebrd_p(matrix_layout, m, n, a, lda, d, e, tauq, taup);
    }
    return -1;
}
static lapack_int LAPACKE_dgebrd_wrapper(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                         lapack_int lda, double* d, double* e, double* tauq,
                                         double* taup) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dgebrd_p == NULL)
            LAPACKE_dgebrd_p =
                (lapack_int(*)(int, lapack_int, lapack_int, double*, lapack_int, double*, double*,
                               double*, double*))GET_FUNC(h_liblapacke, "LAPACKE_dgebrd");
        if (LAPACKE_dgebrd_p != NULL)
            return LAPACKE_dgebrd_p(matrix_layout, m, n, a, lda, d, e, tauq, taup);
    }
    return -1;
}
static lapack_int LAPACKE_sgebrd_wrapper(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                         lapack_int lda, float* d, float* e, float* tauq,
                                         float* taup) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sgebrd_p == NULL)
            LAPACKE_sgebrd_p =
                (lapack_int(*)(int, lapack_int, lapack_int, float*, lapack_int, float*, float*,
                               float*, float*))GET_FUNC(h_liblapacke, "LAPACKE_sgebrd");
        if (LAPACKE_sgebrd_p != NULL)
            return LAPACKE_sgebrd_p(matrix_layout, m, n, a, lda, d, e, tauq, taup);
    }
    return -1;
}
static lapack_int LAPACKE_zgebrd_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda, double* d,
                                         double* e, lapack_complex_double* tauq,
                                         lapack_complex_double* taup) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zgebrd_p == NULL)
            LAPACKE_zgebrd_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                               double*, double*, lapack_complex_double*,
                               lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zgebrd");
        if (LAPACKE_zgebrd_p != NULL)
            return LAPACKE_zgebrd_p(matrix_layout, m, n, a, lda, d, e, tauq, taup);
    }
    return -1;
}
static lapack_int LAPACKE_cgeqrf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cgeqrf_p == NULL)
            LAPACKE_cgeqrf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                               lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cgeqrf");
        if (LAPACKE_cgeqrf_p != NULL)
            return LAPACKE_cgeqrf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_dgeqrf_wrapper(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                         lapack_int lda, double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dgeqrf_p == NULL)
            LAPACKE_dgeqrf_p = (lapack_int(*)(int, lapack_int, lapack_int, double*, lapack_int,
                                              double*))GET_FUNC(h_liblapacke, "LAPACKE_dgeqrf");
        if (LAPACKE_dgeqrf_p != NULL)
            return LAPACKE_dgeqrf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sgeqrf_wrapper(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                         lapack_int lda, float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sgeqrf_p == NULL)
            LAPACKE_sgeqrf_p = (lapack_int(*)(int, lapack_int, lapack_int, float*, lapack_int,
                                              float*))GET_FUNC(h_liblapacke, "LAPACKE_sgeqrf");
        if (LAPACKE_sgeqrf_p != NULL)
            return LAPACKE_sgeqrf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zgeqrf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zgeqrf_p == NULL)
            LAPACKE_zgeqrf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                               lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zgeqrf");
        if (LAPACKE_zgeqrf_p != NULL)
            return LAPACKE_zgeqrf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_cgerqf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cgerqf_p == NULL)
            LAPACKE_cgerqf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                               lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cgerqf");
        if (LAPACKE_cgerqf_p != NULL)
            return LAPACKE_cgerqf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_dgerqf_wrapper(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                         lapack_int lda, double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dgerqf_p == NULL)
            LAPACKE_dgerqf_p = (lapack_int(*)(int, lapack_int, lapack_int, double*, lapack_int,
                                              double*))GET_FUNC(h_liblapacke, "LAPACKE_dgerqf");
        if (LAPACKE_dgerqf_p != NULL)
            return LAPACKE_dgerqf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sgerqf_wrapper(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                         lapack_int lda, float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sgerqf_p == NULL)
            LAPACKE_sgerqf_p = (lapack_int(*)(int, lapack_int, lapack_int, float*, lapack_int,
                                              float*))GET_FUNC(h_liblapacke, "LAPACKE_sgerqf");
        if (LAPACKE_sgerqf_p != NULL)
            return LAPACKE_sgerqf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zgerqf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zgerqf_p == NULL)
            LAPACKE_zgerqf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                               lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zgerqf");
        if (LAPACKE_zgerqf_p != NULL)
            return LAPACKE_zgerqf_p(matrix_layout, m, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_cgesvd_wrapper(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                         lapack_int n, lapack_complex_float* a, lapack_int lda,
                                         float* s, lapack_complex_float* u, lapack_int ldu,
                                         lapack_complex_float* vt, lapack_int ldvt, float* superb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cgesvd_p == NULL)
            LAPACKE_cgesvd_p = (lapack_int(*)(
                int, char, char, lapack_int, lapack_int, lapack_complex_float*, lapack_int, float*,
                lapack_complex_float*, lapack_int, lapack_complex_float*, lapack_int,
                float*))GET_FUNC(h_liblapacke, "LAPACKE_cgesvd");
        if (LAPACKE_cgesvd_p != NULL)
            return LAPACKE_cgesvd_p(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                    superb);
    }
    return -1;
}
static lapack_int LAPACKE_dgesvd_wrapper(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                         lapack_int n, double* a, lapack_int lda, double* s,
                                         double* u, lapack_int ldu, double* vt, lapack_int ldvt,
                                         double* superb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dgesvd_p == NULL)
            LAPACKE_dgesvd_p = (lapack_int(*)(
                int, char, char, lapack_int, lapack_int, double*, lapack_int, double*, double*,
                lapack_int, double*, lapack_int, double*))GET_FUNC(h_liblapacke, "LAPACKE_dgesvd");
        if (LAPACKE_dgesvd_p != NULL)
            return LAPACKE_dgesvd_p(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                    superb);
    }
    return -1;
}
static lapack_int LAPACKE_sgesvd_wrapper(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                         lapack_int n, float* a, lapack_int lda, float* s, float* u,
                                         lapack_int ldu, float* vt, lapack_int ldvt,
                                         float* superb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sgesvd_p == NULL)
            LAPACKE_sgesvd_p = (lapack_int(*)(
                int, char, char, lapack_int, lapack_int, float*, lapack_int, float*, float*,
                lapack_int, float*, lapack_int, float*))GET_FUNC(h_liblapacke, "LAPACKE_sgesvd");
        if (LAPACKE_sgesvd_p != NULL)
            return LAPACKE_sgesvd_p(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                    superb);
    }
    return -1;
}
static lapack_int LAPACKE_zgesvd_wrapper(int matrix_layout, char jobu, char jobvt, lapack_int m,
                                         lapack_int n, lapack_complex_double* a, lapack_int lda,
                                         double* s, lapack_complex_double* u, lapack_int ldu,
                                         lapack_complex_double* vt, lapack_int ldvt,
                                         double* superb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zgesvd_p == NULL)
            LAPACKE_zgesvd_p = (lapack_int(*)(
                int, char, char, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                double*, lapack_complex_double*, lapack_int, lapack_complex_double*, lapack_int,
                double*))GET_FUNC(h_liblapacke, "LAPACKE_zgesvd");
        if (LAPACKE_zgesvd_p != NULL)
            return LAPACKE_zgesvd_p(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                    superb);
    }
    return -1;
}
static lapack_int LAPACKE_cgetrf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cgetrf_p == NULL)
            LAPACKE_cgetrf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_cgetrf");
        if (LAPACKE_cgetrf_p != NULL)
            return LAPACKE_cgetrf_p(matrix_layout, m, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_dgetrf_wrapper(int matrix_layout, lapack_int m, lapack_int n, double* a,
                                         lapack_int lda, lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dgetrf_p == NULL)
            LAPACKE_dgetrf_p = (lapack_int(*)(int, lapack_int, lapack_int, double*, lapack_int,
                                              lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_dgetrf");
        if (LAPACKE_dgetrf_p != NULL)
            return LAPACKE_dgetrf_p(matrix_layout, m, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_sgetrf_wrapper(int matrix_layout, lapack_int m, lapack_int n, float* a,
                                         lapack_int lda, lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sgetrf_p == NULL)
            LAPACKE_sgetrf_p = (lapack_int(*)(int, lapack_int, lapack_int, float*, lapack_int,
                                              lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_sgetrf");
        if (LAPACKE_sgetrf_p != NULL)
            return LAPACKE_sgetrf_p(matrix_layout, m, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_zgetrf_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zgetrf_p == NULL)
            LAPACKE_zgetrf_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_zgetrf");
        if (LAPACKE_zgetrf_p != NULL)
            return LAPACKE_zgetrf_p(matrix_layout, m, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_cheevd_wrapper(int matrix_layout, char jobz, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda, float* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cheevd_p == NULL)
            LAPACKE_cheevd_p =
                (lapack_int(*)(int, char, char, lapack_int, lapack_complex_float*, lapack_int,
                               float*))GET_FUNC(h_liblapacke, "LAPACKE_cheevd");
        if (LAPACKE_cheevd_p != NULL)
            return LAPACKE_cheevd_p(matrix_layout, jobz, uplo, n, a, lda, w);
    }
    return -1;
}
static lapack_int LAPACKE_zheevd_wrapper(int matrix_layout, char jobz, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda, double* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zheevd_p == NULL)
            LAPACKE_zheevd_p =
                (lapack_int(*)(int, char, char, lapack_int, lapack_complex_double*, lapack_int,
                               double*))GET_FUNC(h_liblapacke, "LAPACKE_zheevd");
        if (LAPACKE_zheevd_p != NULL)
            return LAPACKE_zheevd_p(matrix_layout, jobz, uplo, n, a, lda, w);
    }
    return -1;
}
static lapack_int LAPACKE_chegvd_wrapper(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                         lapack_int n, lapack_complex_float* a, lapack_int lda,
                                         lapack_complex_float* b, lapack_int ldb, float* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_chegvd_p == NULL)
            LAPACKE_chegvd_p = (lapack_int(*)(
                int, lapack_int, char, char, lapack_int, lapack_complex_float*, lapack_int,
                lapack_complex_float*, lapack_int, float*))GET_FUNC(h_liblapacke, "LAPACKE_chegvd");
        if (LAPACKE_chegvd_p != NULL)
            return LAPACKE_chegvd_p(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    }
    return -1;
}
static lapack_int LAPACKE_zhegvd_wrapper(int matrix_layout, lapack_int itype, char jobz, char uplo,
                                         lapack_int n, lapack_complex_double* a, lapack_int lda,
                                         lapack_complex_double* b, lapack_int ldb, double* w) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zhegvd_p == NULL)
            LAPACKE_zhegvd_p =
                (lapack_int(*)(int, lapack_int, char, char, lapack_int, lapack_complex_double*,
                               lapack_int, lapack_complex_double*, lapack_int,
                               double*))GET_FUNC(h_liblapacke, "LAPACKE_zhegvd");
        if (LAPACKE_zhegvd_p != NULL)
            return LAPACKE_zhegvd_p(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
    }
    return -1;
}
static lapack_int LAPACKE_chetrd_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda, float* d,
                                         float* e, lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_chetrd_p == NULL)
            LAPACKE_chetrd_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_float*, lapack_int, float*, float*,
                lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_chetrd");
        if (LAPACKE_chetrd_p != NULL)
            return LAPACKE_chetrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zhetrd_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda, double* d,
                                         double* e, lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zhetrd_p == NULL)
            LAPACKE_zhetrd_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_double*, lapack_int, double*, double*,
                lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zhetrd");
        if (LAPACKE_zhetrd_p != NULL)
            return LAPACKE_zhetrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_chetrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_chetrf_p == NULL)
            LAPACKE_chetrf_p =
                (lapack_int(*)(int, char, lapack_int, lapack_complex_float*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_chetrf");
        if (LAPACKE_chetrf_p != NULL)
            return LAPACKE_chetrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_zhetrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zhetrf_p == NULL)
            LAPACKE_zhetrf_p =
                (lapack_int(*)(int, char, lapack_int, lapack_complex_double*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_zhetrf");
        if (LAPACKE_zhetrf_p != NULL)
            return LAPACKE_zhetrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_cungtr_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         const lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cungtr_p == NULL)
            LAPACKE_cungtr_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_float*, lapack_int,
                const lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cungtr");
        if (LAPACKE_cungtr_p != NULL)
            return LAPACKE_cungtr_p(matrix_layout, uplo, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zungtr_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         const lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zungtr_p == NULL)
            LAPACKE_zungtr_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_double*, lapack_int,
                const lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zungtr");
        if (LAPACKE_zungtr_p != NULL)
            return LAPACKE_zungtr_p(matrix_layout, uplo, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_cunmtr_wrapper(int matrix_layout, char side, char uplo, char trans,
                                         lapack_int m, lapack_int n, const lapack_complex_float* a,
                                         lapack_int lda, const lapack_complex_float* tau,
                                         lapack_complex_float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cunmtr_p == NULL)
            LAPACKE_cunmtr_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const lapack_complex_float*, lapack_int,
                                              const lapack_complex_float*, lapack_complex_float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cunmtr");
        if (LAPACKE_cunmtr_p != NULL)
            return LAPACKE_cunmtr_p(matrix_layout, side, uplo, trans, m, n, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_zunmtr_wrapper(int matrix_layout, char side, char uplo, char trans,
                                         lapack_int m, lapack_int n, const lapack_complex_double* a,
                                         lapack_int lda, const lapack_complex_double* tau,
                                         lapack_complex_double* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zunmtr_p == NULL)
            LAPACKE_zunmtr_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const lapack_complex_double*, lapack_int,
                                              const lapack_complex_double*, lapack_complex_double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zunmtr");
        if (LAPACKE_zunmtr_p != NULL)
            return LAPACKE_zunmtr_p(matrix_layout, side, uplo, trans, m, n, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dorgtr_wrapper(int matrix_layout, char uplo, lapack_int n, double* a,
                                         lapack_int lda, const double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dorgtr_p == NULL)
            LAPACKE_dorgtr_p =
                (lapack_int(*)(int, char, lapack_int, double*, lapack_int, const double*))GET_FUNC(
                    h_liblapacke, "LAPACKE_dorgtr");
        if (LAPACKE_dorgtr_p != NULL)
            return LAPACKE_dorgtr_p(matrix_layout, uplo, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sorgtr_wrapper(int matrix_layout, char uplo, lapack_int n, float* a,
                                         lapack_int lda, const float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sorgtr_p == NULL)
            LAPACKE_sorgtr_p =
                (lapack_int(*)(int, char, lapack_int, float*, lapack_int, const float*))GET_FUNC(
                    h_liblapacke, "LAPACKE_sorgtr");
        if (LAPACKE_sorgtr_p != NULL)
            return LAPACKE_sorgtr_p(matrix_layout, uplo, n, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sormtr_wrapper(int matrix_layout, char side, char uplo, char trans,
                                         lapack_int m, lapack_int n, const float* a, lapack_int lda,
                                         const float* tau, float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sormtr_p == NULL)
            LAPACKE_sormtr_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const float*, lapack_int, const float*, float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_sormtr");
        if (LAPACKE_sormtr_p != NULL)
            return LAPACKE_sormtr_p(matrix_layout, side, uplo, trans, m, n, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dormtr_wrapper(int matrix_layout, char side, char uplo, char trans,
                                         lapack_int m, lapack_int n, const double* a,
                                         lapack_int lda, const double* tau, double* c,
                                         lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dormtr_p == NULL)
            LAPACKE_dormtr_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const double*, lapack_int, const double*, double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dormtr");
        if (LAPACKE_dormtr_p != NULL)
            return LAPACKE_dormtr_p(matrix_layout, side, uplo, trans, m, n, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_ssytrd_wrapper(int matrix_layout, char uplo, lapack_int n, float* a,
                                         lapack_int lda, float* d, float* e, float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ssytrd_p == NULL)
            LAPACKE_ssytrd_p =
                (lapack_int(*)(int, char, lapack_int, float*, lapack_int, float*, float*,
                               float*))GET_FUNC(h_liblapacke, "LAPACKE_ssytrd");
        if (LAPACKE_ssytrd_p != NULL)
            return LAPACKE_ssytrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_dsytrd_wrapper(int matrix_layout, char uplo, lapack_int n, double* a,
                                         lapack_int lda, double* d, double* e, double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dsytrd_p == NULL)
            LAPACKE_dsytrd_p =
                (lapack_int(*)(int, char, lapack_int, double*, lapack_int, double*, double*,
                               double*))GET_FUNC(h_liblapacke, "LAPACKE_dsytrd");
        if (LAPACKE_dsytrd_p != NULL)
            return LAPACKE_dsytrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_csytrd_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda, float* d,
                                         float* e, lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_csytrd_p == NULL)
            LAPACKE_csytrd_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_float*, lapack_int, float*, float*,
                lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_csytrd");
        if (LAPACKE_csytrd_p != NULL)
            return LAPACKE_csytrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zsytrd_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda, double* d,
                                         double* e, lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zsytrd_p == NULL)
            LAPACKE_zsytrd_p = (lapack_int(*)(
                int, char, lapack_int, lapack_complex_double*, lapack_int, double*, double*,
                lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zsytrd");
        if (LAPACKE_zsytrd_p != NULL)
            return LAPACKE_zsytrd_p(matrix_layout, uplo, n, a, lda, d, e, tau);
    }
    return -1;
}
static lapack_int LAPACKE_ssytrf_wrapper(int matrix_layout, char uplo, lapack_int n, float* a,
                                         lapack_int lda, lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ssytrf_p == NULL)
            LAPACKE_ssytrf_p = (lapack_int(*)(int, char, lapack_int, float*, lapack_int,
                                              lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_ssytrf");
        if (LAPACKE_ssytrf_p != NULL)
            return LAPACKE_ssytrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_dsytrf_wrapper(int matrix_layout, char uplo, lapack_int n, double* a,
                                         lapack_int lda, lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dsytrf_p == NULL)
            LAPACKE_dsytrf_p = (lapack_int(*)(int, char, lapack_int, double*, lapack_int,
                                              lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_dsytrf");
        if (LAPACKE_dsytrf_p != NULL)
            return LAPACKE_dsytrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_csytrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_csytrf_p == NULL)
            LAPACKE_csytrf_p =
                (lapack_int(*)(int, char, lapack_int, lapack_complex_float*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_csytrf");
        if (LAPACKE_csytrf_p != NULL)
            return LAPACKE_csytrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static lapack_int LAPACKE_zsytrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda,
                                         lapack_int* ipiv) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zsytrf_p == NULL)
            LAPACKE_zsytrf_p =
                (lapack_int(*)(int, char, lapack_int, lapack_complex_double*, lapack_int,
                               lapack_int*))GET_FUNC(h_liblapacke, "LAPACKE_zsytrf");
        if (LAPACKE_zsytrf_p != NULL)
            return LAPACKE_zsytrf_p(matrix_layout, uplo, n, a, lda, ipiv);
    }
    return -1;
}
static void LAPACKE_dorgbr_wrapper(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                   lapack_int k, double* a, lapack_int lda, const double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dorgbr_p == NULL)
            LAPACKE_dorgbr_p =
                (void (*)(int, char, lapack_int, lapack_int, lapack_int, double*, lapack_int,
                          const double*))GET_FUNC(h_liblapacke, "LAPACKE_dorgbr");
        if (LAPACKE_dorgbr_p != NULL)
            LAPACKE_dorgbr_p(matrix_layout, vect, m, n, k, a, lda, tau);
    }
}
static void LAPACKE_sorgbr_wrapper(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                   lapack_int k, float* a, lapack_int lda, const float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sorgbr_p == NULL)
            LAPACKE_sorgbr_p =
                (void (*)(int, char, lapack_int, lapack_int, lapack_int, float*, lapack_int,
                          const float*))GET_FUNC(h_liblapacke, "LAPACKE_sorgbr");
        if (LAPACKE_sorgbr_p != NULL)
            LAPACKE_sorgbr_p(matrix_layout, vect, m, n, k, a, lda, tau);
    }
}
static lapack_int LAPACKE_cungqr_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_float* a, lapack_int lda,
                                         const lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cungqr_p == NULL)
            LAPACKE_cungqr_p = (lapack_int(*)(
                int, lapack_int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                const lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cungqr");
        if (LAPACKE_cungqr_p != NULL)
            return LAPACKE_cungqr_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zungqr_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_double* a, lapack_int lda,
                                         const lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zungqr_p == NULL)
            LAPACKE_zungqr_p = (lapack_int(*)(
                int, lapack_int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                const lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zungqr");
        if (LAPACKE_zungqr_p != NULL)
            return LAPACKE_zungqr_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sormqr_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const float* a, lapack_int lda,
                                         const float* tau, float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sormqr_p == NULL)
            LAPACKE_sormqr_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const float*, lapack_int, const float*, float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_sormqr");
        if (LAPACKE_sormqr_p != NULL)
            return LAPACKE_sormqr_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dormqr_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const double* a,
                                         lapack_int lda, const double* tau, double* c,
                                         lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dormqr_p == NULL)
            LAPACKE_dormqr_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const double*, lapack_int, const double*, double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dormqr");
        if (LAPACKE_dormqr_p != NULL)
            return LAPACKE_dormqr_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dorgqr_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, double* a, lapack_int lda,
                                         const double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dorgqr_p == NULL)
            LAPACKE_dorgqr_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_int, double*, lapack_int,
                               const double*))GET_FUNC(h_liblapacke, "LAPACKE_dorgqr");
        if (LAPACKE_dorgqr_p != NULL)
            return LAPACKE_dorgqr_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sorgqr_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, float* a, lapack_int lda, const float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sorgqr_p == NULL)
            LAPACKE_sorgqr_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_int, float*, lapack_int,
                               const float*))GET_FUNC(h_liblapacke, "LAPACKE_sorgqr");
        if (LAPACKE_sorgqr_p != NULL)
            return LAPACKE_sorgqr_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_cunmqr_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const lapack_complex_float* a,
                                         lapack_int lda, const lapack_complex_float* tau,
                                         lapack_complex_float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cunmqr_p == NULL)
            LAPACKE_cunmqr_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const lapack_complex_float*, lapack_int,
                                              const lapack_complex_float*, lapack_complex_float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cunmqr");
        if (LAPACKE_cunmqr_p != NULL)
            return LAPACKE_cunmqr_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_zunmqr_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const lapack_complex_double* a,
                                         lapack_int lda, const lapack_complex_double* tau,
                                         lapack_complex_double* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zunmqr_p == NULL)
            LAPACKE_zunmqr_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const lapack_complex_double*, lapack_int,
                                              const lapack_complex_double*, lapack_complex_double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zunmqr");
        if (LAPACKE_zunmqr_p != NULL)
            return LAPACKE_zunmqr_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dorgrq_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, double* a, lapack_int lda,
                                         const double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dorgrq_p == NULL)
            LAPACKE_dorgrq_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_int, double*, lapack_int,
                               const double*))GET_FUNC(h_liblapacke, "LAPACKE_dorgrq");
        if (LAPACKE_dorgrq_p != NULL)
            return LAPACKE_dorgrq_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sorgrq_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, float* a, lapack_int lda, const float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sorgrq_p == NULL)
            LAPACKE_sorgrq_p =
                (lapack_int(*)(int, lapack_int, lapack_int, lapack_int, float*, lapack_int,
                               const float*))GET_FUNC(h_liblapacke, "LAPACKE_sorgrq");
        if (LAPACKE_sorgrq_p != NULL)
            return LAPACKE_sorgrq_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_cungrq_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_float* a, lapack_int lda,
                                         const lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cungrq_p == NULL)
            LAPACKE_cungrq_p = (lapack_int(*)(
                int, lapack_int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                const lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cungrq");
        if (LAPACKE_cungrq_p != NULL)
            return LAPACKE_cungrq_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zungrq_wrapper(int matrix_layout, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_double* a, lapack_int lda,
                                         const lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zungrq_p == NULL)
            LAPACKE_zungrq_p = (lapack_int(*)(
                int, lapack_int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                const lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zungrq");
        if (LAPACKE_zungrq_p != NULL)
            return LAPACKE_zungrq_p(matrix_layout, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_sormrq_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const float* a, lapack_int lda,
                                         const float* tau, float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_sormrq_p == NULL)
            LAPACKE_sormrq_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const float*, lapack_int, const float*, float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_sormrq");
        if (LAPACKE_sormrq_p != NULL)
            return LAPACKE_sormrq_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_dormrq_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const double* a,
                                         lapack_int lda, const double* tau, double* c,
                                         lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dormrq_p == NULL)
            LAPACKE_dormrq_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const double*, lapack_int, const double*, double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dormrq");
        if (LAPACKE_dormrq_p != NULL)
            return LAPACKE_dormrq_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_cunmrq_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const lapack_complex_float* a,
                                         lapack_int lda, const lapack_complex_float* tau,
                                         lapack_complex_float* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cunmrq_p == NULL)
            LAPACKE_cunmrq_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const lapack_complex_float*, lapack_int,
                                              const lapack_complex_float*, lapack_complex_float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cunmrq");
        if (LAPACKE_cunmrq_p != NULL)
            return LAPACKE_cunmrq_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_zunmrq_wrapper(int matrix_layout, char side, char trans, lapack_int m,
                                         lapack_int n, lapack_int k, const lapack_complex_double* a,
                                         lapack_int lda, const lapack_complex_double* tau,
                                         lapack_complex_double* c, lapack_int ldc) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zunmrq_p == NULL)
            LAPACKE_zunmrq_p = (lapack_int(*)(int, char, char, lapack_int, lapack_int, lapack_int,
                                              const lapack_complex_double*, lapack_int,
                                              const lapack_complex_double*, lapack_complex_double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zunmrq");
        if (LAPACKE_zunmrq_p != NULL)
            return LAPACKE_zunmrq_p(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    return -1;
}
static lapack_int LAPACKE_cpotrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cpotrf_p == NULL)
            LAPACKE_cpotrf_p = (lapack_int(*)(int, char, lapack_int, lapack_complex_float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cpotrf");
        if (LAPACKE_cpotrf_p != NULL)
            return LAPACKE_cpotrf_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_dpotrf_wrapper(int matrix_layout, char uplo, lapack_int n, double* a,
                                         lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dpotrf_p == NULL)
            LAPACKE_dpotrf_p = (lapack_int(*)(int, char, lapack_int, double*, lapack_int))GET_FUNC(
                h_liblapacke, "LAPACKE_dpotrf");
        if (LAPACKE_dpotrf_p != NULL)
            return LAPACKE_dpotrf_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_spotrf_wrapper(int matrix_layout, char uplo, lapack_int n, float* a,
                                         lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_spotrf_p == NULL)
            LAPACKE_spotrf_p = (lapack_int(*)(int, char, lapack_int, float*, lapack_int))GET_FUNC(
                h_liblapacke, "LAPACKE_spotrf");
        if (LAPACKE_spotrf_p != NULL)
            return LAPACKE_spotrf_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_zpotrf_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zpotrf_p == NULL)
            LAPACKE_zpotrf_p = (lapack_int(*)(int, char, lapack_int, lapack_complex_double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zpotrf");
        if (LAPACKE_zpotrf_p != NULL)
            return LAPACKE_zpotrf_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_cpotrs_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_int nrhs, const lapack_complex_float* a,
                                         lapack_int lda, lapack_complex_float* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cpotrs_p == NULL)
            LAPACKE_cpotrs_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, const lapack_complex_float*, lapack_int,
                lapack_complex_float*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cpotrs");
        if (LAPACKE_cpotrs_p != NULL)
            return LAPACKE_cpotrs_p(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_dpotrs_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_int nrhs, const double* a, lapack_int lda,
                                         double* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dpotrs_p == NULL)
            LAPACKE_dpotrs_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, const double*, lapack_int,
                               double*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dpotrs");
        if (LAPACKE_dpotrs_p != NULL)
            return LAPACKE_dpotrs_p(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_spotrs_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_int nrhs, const float* a, lapack_int lda, float* b,
                                         lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_spotrs_p == NULL)
            LAPACKE_spotrs_p =
                (lapack_int(*)(int, char, lapack_int, lapack_int, const float*, lapack_int, float*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_spotrs");
        if (LAPACKE_spotrs_p != NULL)
            return LAPACKE_spotrs_p(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_zpotrs_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_int nrhs, const lapack_complex_double* a,
                                         lapack_int lda, lapack_complex_double* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zpotrs_p == NULL)
            LAPACKE_zpotrs_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, const lapack_complex_double*, lapack_int,
                lapack_complex_double*, lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zpotrs");
        if (LAPACKE_zpotrs_p != NULL)
            return LAPACKE_zpotrs_p(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_cpotri_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_float* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cpotri_p == NULL)
            LAPACKE_cpotri_p = (lapack_int(*)(int, char, lapack_int, lapack_complex_float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_cpotri");
        if (LAPACKE_cpotri_p != NULL)
            return LAPACKE_cpotri_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_dpotri_wrapper(int matrix_layout, char uplo, lapack_int n, double* a,
                                         lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dpotri_p == NULL)
            LAPACKE_dpotri_p = (lapack_int(*)(int, char, lapack_int, double*, lapack_int))GET_FUNC(
                h_liblapacke, "LAPACKE_dpotri");
        if (LAPACKE_dpotri_p != NULL)
            return LAPACKE_dpotri_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_spotri_wrapper(int matrix_layout, char uplo, lapack_int n, float* a,
                                         lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_spotri_p == NULL)
            LAPACKE_spotri_p = (lapack_int(*)(int, char, lapack_int, float*, lapack_int))GET_FUNC(
                h_liblapacke, "LAPACKE_spotri");
        if (LAPACKE_spotri_p != NULL)
            return LAPACKE_spotri_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_zpotri_wrapper(int matrix_layout, char uplo, lapack_int n,
                                         lapack_complex_double* a, lapack_int lda) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zpotri_p == NULL)
            LAPACKE_zpotri_p = (lapack_int(*)(int, char, lapack_int, lapack_complex_double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zpotri");
        if (LAPACKE_zpotri_p != NULL)
            return LAPACKE_zpotri_p(matrix_layout, uplo, n, a, lda);
    }
    return -1;
}
static lapack_int LAPACKE_claswp_wrapper(int matrix_layout, lapack_int n, lapack_complex_float* a,
                                         lapack_int lda, lapack_int k1, lapack_int k2,
                                         const lapack_int* ipiv, lapack_int incx) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_claswp_p == NULL)
            LAPACKE_claswp_p = (lapack_int(*)(int, lapack_int, lapack_complex_float*, lapack_int,
                                              lapack_int, lapack_int, const lapack_int*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_claswp");
        if (LAPACKE_claswp_p != NULL)
            return LAPACKE_claswp_p(matrix_layout, n, a, lda, k1, k2, ipiv, incx);
    }
    return -1;
}
static lapack_int LAPACKE_dlaswp_wrapper(int matrix_layout, lapack_int n, double* a, lapack_int lda,
                                         lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                                         lapack_int incx) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dlaswp_p == NULL)
            LAPACKE_dlaswp_p = (lapack_int(*)(int, lapack_int, double*, lapack_int, lapack_int,
                                              lapack_int, const lapack_int*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dlaswp");
        if (LAPACKE_dlaswp_p != NULL)
            return LAPACKE_dlaswp_p(matrix_layout, n, a, lda, k1, k2, ipiv, incx);
    }
    return -1;
}
static lapack_int LAPACKE_slaswp_wrapper(int matrix_layout, lapack_int n, float* a, lapack_int lda,
                                         lapack_int k1, lapack_int k2, const lapack_int* ipiv,
                                         lapack_int incx) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_slaswp_p == NULL)
            LAPACKE_slaswp_p = (lapack_int(*)(int, lapack_int, float*, lapack_int, lapack_int,
                                              lapack_int, const lapack_int*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_slaswp");
        if (LAPACKE_slaswp_p != NULL)
            return LAPACKE_slaswp_p(matrix_layout, n, a, lda, k1, k2, ipiv, incx);
    }
    return -1;
}
static lapack_int LAPACKE_zlaswp_wrapper(int matrix_layout, lapack_int n, lapack_complex_double* a,
                                         lapack_int lda, lapack_int k1, lapack_int k2,
                                         const lapack_int* ipiv, lapack_int incx) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zlaswp_p == NULL)
            LAPACKE_zlaswp_p = (lapack_int(*)(int, lapack_int, lapack_complex_double*, lapack_int,
                                              lapack_int, lapack_int, const lapack_int*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_zlaswp");
        if (LAPACKE_zlaswp_p != NULL)
            return LAPACKE_zlaswp_p(matrix_layout, n, a, lda, k1, k2, ipiv, incx);
    }
    return -1;
}
static lapack_int LAPACKE_cungbr_wrapper(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_float* a, lapack_int lda,
                                         const lapack_complex_float* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_cungbr_p == NULL)
            LAPACKE_cungbr_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, lapack_int, lapack_complex_float*, lapack_int,
                const lapack_complex_float*))GET_FUNC(h_liblapacke, "LAPACKE_cungbr");
        if (LAPACKE_cungbr_p != NULL)
            return LAPACKE_cungbr_p(matrix_layout, vect, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_zungbr_wrapper(int matrix_layout, char vect, lapack_int m, lapack_int n,
                                         lapack_int k, lapack_complex_double* a, lapack_int lda,
                                         const lapack_complex_double* tau) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_zungbr_p == NULL)
            LAPACKE_zungbr_p = (lapack_int(*)(
                int, char, lapack_int, lapack_int, lapack_int, lapack_complex_double*, lapack_int,
                const lapack_complex_double*))GET_FUNC(h_liblapacke, "LAPACKE_zungbr");
        if (LAPACKE_zungbr_p != NULL)
            return LAPACKE_zungbr_p(matrix_layout, vect, m, n, k, a, lda, tau);
    }
    return -1;
}
static lapack_int LAPACKE_strtrs_wrapper(int matrix_layout, char uplo, char trans, char diag,
                                         lapack_int n, lapack_int nrhs, const float* a,
                                         lapack_int lda, float* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_strtrs_p == NULL)
            LAPACKE_strtrs_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const float*, lapack_int, float*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_strtrs");
        if (LAPACKE_strtrs_p != NULL)
            return LAPACKE_strtrs_p(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_dtrtrs_wrapper(int matrix_layout, char uplo, char trans, char diag,
                                         lapack_int n, lapack_int nrhs, const double* a,
                                         lapack_int lda, double* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_dtrtrs_p == NULL)
            LAPACKE_dtrtrs_p = (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                                              const double*, lapack_int, double*,
                                              lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_dtrtrs");
        if (LAPACKE_dtrtrs_p != NULL)
            return LAPACKE_dtrtrs_p(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_ctrtrs_wrapper(int matrix_layout, char uplo, char trans, char diag,
                                         lapack_int n, lapack_int nrhs,
                                         const lapack_complex_float* a, lapack_int lda,
                                         lapack_complex_float* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ctrtrs_p == NULL)
            LAPACKE_ctrtrs_p =
                (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                               const lapack_complex_float*, lapack_int, lapack_complex_float*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_ctrtrs");
        if (LAPACKE_ctrtrs_p != NULL)
            return LAPACKE_ctrtrs_p(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
static lapack_int LAPACKE_ztrtrs_wrapper(int matrix_layout, char uplo, char trans, char diag,
                                         lapack_int n, lapack_int nrhs,
                                         const lapack_complex_double* a, lapack_int lda,
                                         lapack_complex_double* b, lapack_int ldb) {
    if (lapacke_library() != NULL) {
        if (LAPACKE_ztrtrs_p == NULL)
            LAPACKE_ztrtrs_p =
                (lapack_int(*)(int, char, char, char, lapack_int, lapack_int,
                               const lapack_complex_double*, lapack_int, lapack_complex_double*,
                               lapack_int))GET_FUNC(h_liblapacke, "LAPACKE_ztrtrs");
        if (LAPACKE_ztrtrs_p != NULL)
            return LAPACKE_ztrtrs_p(matrix_layout, uplo, trans, diag, n, nrhs, a, lda, b, ldb);
    }
    return -1;
}
}

#endif // _REFERENCE_LAPACK_TEMPLATES_HPP__
