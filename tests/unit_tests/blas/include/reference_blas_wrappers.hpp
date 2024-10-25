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

#ifndef _REFERENCE_BLAS_WRAPPERS_HPP__
#define _REFERENCE_BLAS_WRAPPERS_HPP__

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include <string>
#include "cblas.h"

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
static LIB_TYPE h_libblas = NULL;
static LIB_TYPE blas_library() {
    if (h_libblas == NULL) {
        h_libblas = GET_LIB_HANDLE(ONEMKL_REF_BLAS_LIBNAME);
        if (h_libblas == NULL) {
            throw oneapi::mkl::library_not_found(
                "BLAS", "blas_library()",
                std::string("failed to load BLAS library ") + ONEMKL_REF_BLAS_LIBNAME);
        }
    }
    return h_libblas;
}

static LIB_TYPE h_libcblas = NULL;
static LIB_TYPE cblas_library() {
    if (h_libcblas == NULL) {
        h_libcblas = GET_LIB_HANDLE(ONEMKL_REF_CBLAS_LIBNAME);
        if (h_libcblas == NULL) {
            throw oneapi::mkl::library_not_found(
                "BLAS", "cblas_library()",
                std::string("failed to load CBLAS library ") + ONEMKL_REF_CBLAS_LIBNAME);
        }
    }
    return h_libcblas;
}

/* Level 3 */

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
static void (*cblas_ssymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const float alpha, const float* a,
                             const int lda, const float* b, const int ldb, const float beta,
                             float* c, const int ldc);
static void (*cblas_dsymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const double alpha, const double* a,
                             const int lda, const double* b, const int ldb, const double beta,
                             double* c, const int ldc);
static void (*cblas_csymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void* alpha, const void* a,
                             const int lda, const void* b, const int ldb, const void* beta, void* c,
                             const int ldc);
static void (*cblas_zsymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void* alpha, const void* a,
                             const int lda, const void* b, const int ldb, const void* beta, void* c,
                             const int ldc);
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
static void (*cblas_chemm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void* alpha, const void* a,
                             const int lda, const void* b, const int ldb, const void* beta, void* c,
                             const int ldc);
static void (*cblas_zhemm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void* alpha, const void* a,
                             const int lda, const void* b, const int ldb, const void* beta, void* c,
                             const int ldc);
static void (*cblas_cherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const float alpha, const void* a,
                             const int lda, const float beta, void* c, const int ldc);
static void (*cblas_zherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const double alpha, const void* a,
                             const int lda, const double beta, void* c, const int ldc);
static void (*cblas_ssyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const float alpha, const float* a,
                              const int lda, const float* b, const int ldb, const float beta,
                              float* c, const int ldc);
static void (*cblas_dsyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const double alpha, const double* a,
                              const int lda, const double* b, const int ldb, const double beta,
                              double* c, const int ldc);
static void (*cblas_csyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void* alpha, const void* a,
                              const int lda, const void* b, const int ldb, const void* beta,
                              void* c, const int ldc);
static void (*cblas_zsyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void* alpha, const void* a,
                              const int lda, const void* b, const int ldb, const void* beta,
                              void* c, const int ldc);
static void (*cblas_cher2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void* alpha, const void* a,
                              const int lda, const void* b, const int ldb, const float beta,
                              void* c, const int ldc);
static void (*cblas_zher2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void* alpha, const void* a,
                              const int lda, const void* b, const int ldb, const double beta,
                              void* c, const int ldc);
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
static void (*cblas_strsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const float alpha, const float* a, const int lda, float* b,
                             const int ldb);
static void (*cblas_dtrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const double alpha, const double* a, const int lda, double* b,
                             const int ldb);
static void (*cblas_ctrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void* alpha, const void* a, const int lda, void* b,
                             const int ldb);
static void (*cblas_ztrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void* alpha, const void* a, const int lda, void* b,
                             const int ldb);

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

static void cblas_ssymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const float alpha, const float* a,
                                const int lda, const float* b, const int ldb, const float beta,
                                float* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssymm_p == NULL)
            cblas_ssymm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int m,
                          const int n, const float alpha, const float* a, const int lda,
                          const float* b, const int ldb, const float beta, float* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_ssymm");
        if (cblas_ssymm_p != NULL)
            cblas_ssymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dsymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const double alpha, const double* a,
                                const int lda, const double* b, const int ldb, const double beta,
                                double* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsymm_p == NULL)
            cblas_dsymm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int m,
                          const int n, const double alpha, const double* a, const int lda,
                          const double* b, const int ldb, const double beta, double* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_dsymm");
        if (cblas_dsymm_p != NULL)
            cblas_dsymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_csymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void* alpha, const void* a,
                                const int lda, const void* b, const int ldb, const void* beta,
                                void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csymm_p == NULL)
            cblas_csymm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void* alpha, const void* a,
                                      const int lda, const void* b, const int ldb, const void* beta,
                                      void* c, const int ldc))GET_FUNC(h_libcblas, "cblas_csymm");
        if (cblas_csymm_p != NULL)
            cblas_csymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zsymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void* alpha, const void* a,
                                const int lda, const void* b, const int ldb, const void* beta,
                                void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsymm_p == NULL)
            cblas_zsymm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void* alpha, const void* a,
                                      const int lda, const void* b, const int ldb, const void* beta,
                                      void* c, const int ldc))GET_FUNC(h_libcblas, "cblas_zsymm");
        if (cblas_zsymm_p != NULL)
            cblas_zsymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
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

static void cblas_chemm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void* alpha, const void* a,
                                const int lda, const void* b, const int ldb, const void* beta,
                                void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_chemm_p == NULL)
            cblas_chemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void* alpha, const void* a,
                                      const int lda, const void* b, const int ldb, const void* beta,
                                      void* c, const int ldc))GET_FUNC(h_libcblas, "cblas_chemm");
        if (cblas_chemm_p != NULL)
            cblas_chemm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zhemm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void* alpha, const void* a,
                                const int lda, const void* b, const int ldb, const void* beta,
                                void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zhemm_p == NULL)
            cblas_zhemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void* alpha, const void* a,
                                      const int lda, const void* b, const int ldb, const void* beta,
                                      void* c, const int ldc))GET_FUNC(h_libcblas, "cblas_zhemm");
        if (cblas_zhemm_p != NULL)
            cblas_zhemm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
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

static void cblas_ssyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const float alpha, const float* a,
                                 const int lda, const float* b, const int ldb, const float beta,
                                 float* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr2k_p == NULL)
            cblas_ssyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const float alpha, const float* a, const int lda,
                          const float* b, const int ldb, const float beta, float* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_ssyr2k");
        if (cblas_ssyr2k_p != NULL)
            cblas_ssyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dsyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const double alpha, const double* a,
                                 const int lda, const double* b, const int ldb, const double beta,
                                 double* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr2k_p == NULL)
            cblas_dsyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const double alpha, const double* a, const int lda,
                          const double* b, const int ldb, const double beta, double* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_dsyr2k");
        if (cblas_dsyr2k_p != NULL)
            cblas_dsyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_csyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void* alpha, const void* a,
                                 const int lda, const void* b, const int ldb, const void* beta,
                                 void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csyr2k_p == NULL)
            cblas_csyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void* alpha, const void* a, const int lda,
                          const void* b, const int ldb, const void* beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_csyr2k");
        if (cblas_csyr2k_p != NULL)
            cblas_csyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zsyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void* alpha, const void* a,
                                 const int lda, const void* b, const int ldb, const void* beta,
                                 void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsyr2k_p == NULL)
            cblas_zsyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void* alpha, const void* a, const int lda,
                          const void* b, const int ldb, const void* beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zsyr2k");
        if (cblas_zsyr2k_p != NULL)
            cblas_zsyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_cher2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void* alpha, const void* a,
                                 const int lda, const void* b, const int ldb, const float beta,
                                 void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cher2k_p == NULL)
            cblas_cher2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void* alpha, const void* a, const int lda,
                          const void* b, const int ldb, const float beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_cher2k");
        if (cblas_cher2k_p != NULL)
            cblas_cher2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zher2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void* alpha, const void* a,
                                 const int lda, const void* b, const int ldb, const double beta,
                                 void* c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zher2k_p == NULL)
            cblas_zher2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void* alpha, const void* a, const int lda,
                          const void* b, const int ldb, const double beta, void* c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zher2k");
        if (cblas_zher2k_p != NULL)
            cblas_zher2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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

static void cblas_strsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const float alpha, const float* a, const int lda, float* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_strsm_p == NULL)
            cblas_strsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const float alpha, const float* a, const int lda,
                                      float* b, const int ldb))GET_FUNC(h_libcblas, "cblas_strsm");
        if (cblas_strsm_p != NULL)
            cblas_strsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_dtrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const double alpha, const double* a, const int lda, double* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_dtrsm_p == NULL)
            cblas_dtrsm_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                CBLAS_DIAG diag, const int m, const int n, const double alpha, const double* a,
                const int lda, double* b, const int ldb))GET_FUNC(h_libcblas, "cblas_dtrsm");
        if (cblas_dtrsm_p != NULL)
            cblas_dtrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ctrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void* alpha, const void* a, const int lda, void* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ctrsm_p == NULL)
            cblas_ctrsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      void* b, const int ldb))GET_FUNC(h_libcblas, "cblas_ctrsm");
        if (cblas_ctrsm_p != NULL)
            cblas_ctrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ztrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void* alpha, const void* a, const int lda, void* b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ztrsm_p == NULL)
            cblas_ztrsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      void* b, const int ldb))GET_FUNC(h_libcblas, "cblas_ztrsm");
        if (cblas_ztrsm_p != NULL)
            cblas_ztrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

/* Level 2 */

static void (*cblas_sgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const float alpha, const float* a, const int lda, const float* x,
                             const int incx, const float beta, float* y, const int incy);
static void (*cblas_dgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const double alpha, const double* a, const int lda, const double* x,
                             const int incx, const double beta, double* y, const int incy);
static void (*cblas_cgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_zgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_sgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const float alpha, const float* a, const int lda,
                             const float* x, const int incx, const float beta, float* y,
                             const int incy);
static void (*cblas_dgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const double alpha, const double* a, const int lda,
                             const double* x, const int incx, const double beta, double* y,
                             const int incy);
static void (*cblas_cgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const void* alpha, const void* a, const int lda,
                             const void* x, const int incx, const void* beta, void* y,
                             const int incy);
static void (*cblas_zgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const void* alpha, const void* a, const int lda,
                             const void* x, const int incx, const void* beta, void* y,
                             const int incy);
static void (*cblas_sger_p)(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                            const float* x, const int incx, const float* y, const int incy,
                            float* a, const int lda);
static void (*cblas_dger_p)(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                            const double* x, const int incx, const double* y, const int incy,
                            double* a, const int lda);
static void (*cblas_cgerc_p)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                             const void* x, const int incx, const void* y, const int incy, void* a,
                             const int lda);
static void (*cblas_zgerc_p)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                             const void* x, const int incx, const void* y, const int incy, void* a,
                             const int lda);
static void (*cblas_cgeru_p)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                             const void* x, const int incx, const void* y, const int incy, void* a,
                             const int lda);
static void (*cblas_zgeru_p)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                             const void* x, const int incx, const void* y, const int incy, void* a,
                             const int lda);
static void (*cblas_chbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_zhbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_chemv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_zhemv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* a, const int lda, const void* x,
                             const int incx, const void* beta, void* y, const int incy);
static void (*cblas_cher_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const void* x, const int incx, void* a,
                            const int lda);
static void (*cblas_zher_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const void* x, const int incx, void* a,
                            const int lda);
static void (*cblas_cher2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* x, const int incx, const void* y,
                             const int incy, void* a, const int lda);
static void (*cblas_zher2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* x, const int incx, const void* y,
                             const int incy, void* a, const int lda);
static void (*cblas_chpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* a, const void* x, const int incx,
                             const void* beta, void* y, const int incy);
static void (*cblas_zhpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* a, const void* x, const int incx,
                             const void* beta, void* y, const int incy);
static void (*cblas_chpr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const void* x, const int incx, void* a);
static void (*cblas_zhpr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const void* x, const int incx, void* a);
static void (*cblas_chpr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* x, const int incx, const void* y,
                             const int incy, void* a);
static void (*cblas_zhpr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void* alpha, const void* x, const int incx, const void* y,
                             const int incy, void* a);
static void (*cblas_ssbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const float alpha, const float* a, const int lda, const float* x,
                             const int incx, const float beta, float* y, const int incy);
static void (*cblas_dsbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const double alpha, const double* a, const int lda, const double* x,
                             const int incx, const double beta, double* y, const int incy);
static void (*cblas_ssymv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float* a, const int lda, const float* x,
                             const int incx, const float beta, float* y, const int incy);
static void (*cblas_dsymv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double* a, const int lda, const double* x,
                             const int incx, const double beta, double* y, const int incy);
static void (*cblas_ssyr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const float* x, const int incx, float* a,
                            const int lda);
static void (*cblas_dsyr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const double* x, const int incx, double* a,
                            const int lda);
static void (*cblas_ssyr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float* x, const int incx, const float* y,
                             const int incy, float* a, const int lda);
static void (*cblas_dsyr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double* x, const int incx, const double* y,
                             const int incy, double* a, const int lda);
static void (*cblas_sspmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float* a, const float* x, const int incx,
                             const float beta, float* y, const int incy);
static void (*cblas_dspmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double* a, const double* x, const int incx,
                             const double beta, double* y, const int incy);
static void (*cblas_sspr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const float* x, const int incx, float* a);
static void (*cblas_dspr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const double* x, const int incx, double* a);
static void (*cblas_sspr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float* x, const int incx, const float* y,
                             const int incy, float* a);
static void (*cblas_dspr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double* x, const int incx, const double* y,
                             const int incy, double* a);
static void (*cblas_stbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const float* a,
                             const int lda, float* x, const int incx);
static void (*cblas_dtbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const double* a,
                             const int lda, double* x, const int incx);
static void (*cblas_ctbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                             const int lda, void* x, const int incx);
static void (*cblas_ztbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                             const int lda, void* x, const int incx);
static void (*cblas_stbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const float* a,
                             const int lda, float* x, const int incx);
static void (*cblas_dtbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const double* a,
                             const int lda, double* x, const int incx);
static void (*cblas_ctbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                             const int lda, void* x, const int incx);
static void (*cblas_ztbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                             const int lda, void* x, const int incx);
static void (*cblas_stpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                             const int incx);
static void (*cblas_dtpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                             const int incx);
static void (*cblas_ctpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                             const int incx);
static void (*cblas_ztpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                             const int incx);
static void (*cblas_stpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                             const int incx);
static void (*cblas_dtpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                             const int incx);
static void (*cblas_ctpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                             const int incx);
static void (*cblas_ztpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                             const int incx);
static void (*cblas_strmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                             float* x, const int incx);
static void (*cblas_dtrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                             double* x, const int incx);
static void (*cblas_ctrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                             void* x, const int incx);
static void (*cblas_ztrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                             void* x, const int incx);
static void (*cblas_strsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                             float* x, const int incx);
static void (*cblas_dtrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                             double* x, const int incx);
static void (*cblas_ctrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                             void* x, const int incx);
static void (*cblas_ztrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                             void* x, const int incx);

static void cblas_sgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const float alpha, const float* a, const int lda,
                                const float* x, const int incx, const float beta, float* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sgemv_p == NULL)
            cblas_sgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const float alpha, const float* a, const int lda,
                                      const float* x, const int incx, const float beta, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sgemv");
        if (cblas_sgemv_p != NULL)
            cblas_sgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const double alpha, const double* a, const int lda,
                                const double* x, const int incx, const double beta, double* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dgemv_p == NULL)
            cblas_dgemv_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                const double alpha, const double* a, const int lda, const double* x, const int incx,
                const double beta, double* y, const int incy))GET_FUNC(h_libcblas, "cblas_dgemv");
        if (cblas_dgemv_p != NULL)
            cblas_dgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const void* alpha, const void* a, const int lda,
                                const void* x, const int incx, const void* beta, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cgemv_p == NULL)
            cblas_cgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_cgemv");
        if (cblas_cgemv_p != NULL)
            cblas_cgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const void* alpha, const void* a, const int lda,
                                const void* x, const int incx, const void* beta, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zgemv_p == NULL)
            cblas_zgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zgemv");
        if (cblas_zgemv_p != NULL)
            cblas_zgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_sgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const float alpha, const float* a,
                                const int lda, const float* x, const int incx, const float beta,
                                float* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sgbmv_p == NULL)
            cblas_sgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const float alpha, const float* a, const int lda,
                          const float* x, const int incx, const float beta, float* y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_sgbmv");
        if (cblas_sgbmv_p != NULL)
            cblas_sgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const double alpha, const double* a,
                                const int lda, const double* x, const int incx, const double beta,
                                double* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dgbmv_p == NULL)
            cblas_dgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const double alpha, const double* a, const int lda,
                          const double* x, const int incx, const double beta, double* y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_dgbmv");
        if (cblas_dgbmv_p != NULL)
            cblas_dgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const void* alpha, const void* a,
                                const int lda, const void* x, const int incx, const void* beta,
                                void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cgbmv_p == NULL)
            cblas_cgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const void* alpha, const void* a, const int lda,
                          const void* x, const int incx, const void* beta, void* y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_cgbmv");
        if (cblas_cgbmv_p != NULL)
            cblas_cgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const void* alpha, const void* a,
                                const int lda, const void* x, const int incx, const void* beta,
                                void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zgbmv_p == NULL)
            cblas_zgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const void* alpha, const void* a, const int lda,
                          const void* x, const int incx, const void* beta, void* y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_zgbmv");
        if (cblas_zgbmv_p != NULL)
            cblas_zgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_sger_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                               const float* x, const int incx, const float* y, const int incy,
                               float* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_sger_p == NULL)
            cblas_sger_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                          const float* x, const int incx, const float* y, const int incy, float* a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_sger");
        if (cblas_sger_p != NULL)
            cblas_sger_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_dger_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                               const double* x, const int incx, const double* y, const int incy,
                               double* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dger_p == NULL)
            cblas_dger_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                          const double* x, const int incx, const double* y, const int incy,
                          double* a, const int lda))GET_FUNC(h_libcblas, "cblas_dger");
        if (cblas_dger_p != NULL)
            cblas_dger_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_cgerc_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                                const void* x, const int incx, const void* y, const int incy,
                                void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cgerc_p == NULL)
            cblas_cgerc_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                          const void* x, const int incx, const void* y, const int incy, void* a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_cgerc");
        if (cblas_cgerc_p != NULL)
            cblas_cgerc_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zgerc_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                                const void* x, const int incx, const void* y, const int incy,
                                void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zgerc_p == NULL)
            cblas_zgerc_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                          const void* x, const int incx, const void* y, const int incy, void* a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_zgerc");
        if (cblas_zgerc_p != NULL)
            cblas_zgerc_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_cgeru_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                                const void* x, const int incx, const void* y, const int incy,
                                void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cgeru_p == NULL)
            cblas_cgeru_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                          const void* x, const int incx, const void* y, const int incy, void* a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_cgeru");
        if (cblas_cgeru_p != NULL)
            cblas_cgeru_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zgeru_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                                const void* x, const int incx, const void* y, const int incy,
                                void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zgeru_p == NULL)
            cblas_zgeru_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void* alpha,
                          const void* x, const int incx, const void* y, const int incy, void* a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_zgeru");
        if (cblas_zgeru_p != NULL)
            cblas_zgeru_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_chbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const void* alpha, const void* a, const int lda,
                                const void* x, const int incx, const void* beta, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chbmv_p == NULL)
            cblas_chbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chbmv");
        if (cblas_chbmv_p != NULL)
            cblas_chbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zhbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const void* alpha, const void* a, const int lda,
                                const void* x, const int incx, const void* beta, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhbmv_p == NULL)
            cblas_zhbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhbmv");
        if (cblas_zhbmv_p != NULL)
            cblas_zhbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_chemv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* a, const int lda, const void* x,
                                const int incx, const void* beta, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chemv_p == NULL)
            cblas_chemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chemv");
        if (cblas_chemv_p != NULL)
            cblas_chemv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zhemv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* a, const int lda, const void* x,
                                const int incx, const void* beta, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhemv_p == NULL)
            cblas_zhemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* a, const int lda,
                                      const void* x, const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhemv");
        if (cblas_zhemv_p != NULL)
            cblas_zhemv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cher_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const void* x, const int incx, void* a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cher_p == NULL)
            cblas_cher_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const void* x, const int incx, void* a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_cher");
        if (cblas_cher_p != NULL)
            cblas_cher_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_zher_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const void* x, const int incx, void* a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zher_p == NULL)
            cblas_zher_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const void* x, const int incx, void* a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_zher");
        if (cblas_zher_p != NULL)
            cblas_zher_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_cher2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* x, const int incx, const void* y,
                                const int incy, void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cher2_p == NULL)
            cblas_cher2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* x, const int incx,
                                      const void* y, const int incy, void* a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_cher2");
        if (cblas_cher2_p != NULL)
            cblas_cher2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zher2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* x, const int incx, const void* y,
                                const int incy, void* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zher2_p == NULL)
            cblas_zher2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* x, const int incx,
                                      const void* y, const int incy, void* a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_zher2");
        if (cblas_zher2_p != NULL)
            cblas_zher2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_chpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* a, const void* x, const int incx,
                                const void* beta, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chpmv_p == NULL)
            cblas_chpmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* a, const void* x,
                                      const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chpmv");
        if (cblas_chpmv_p != NULL)
            cblas_chpmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_zhpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* a, const void* x, const int incx,
                                const void* beta, void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhpmv_p == NULL)
            cblas_zhpmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void* alpha, const void* a, const void* x,
                                      const int incx, const void* beta, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhpmv");
        if (cblas_zhpmv_p != NULL)
            cblas_zhpmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_chpr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const void* x, const int incx, void* a) {
    if (cblas_library() != NULL) {
        if (cblas_chpr_p == NULL)
            cblas_chpr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const void* x, const int incx,
                                     void* a))GET_FUNC(h_libcblas, "cblas_chpr");
        if (cblas_chpr_p != NULL)
            cblas_chpr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_zhpr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const void* x, const int incx, void* a) {
    if (cblas_library() != NULL) {
        if (cblas_zhpr_p == NULL)
            cblas_zhpr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const void* x, const int incx,
                                     void* a))GET_FUNC(h_libcblas, "cblas_zhpr");
        if (cblas_zhpr_p != NULL)
            cblas_zhpr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_chpr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* x, const int incx, const void* y,
                                const int incy, void* a) {
    if (cblas_library() != NULL) {
        if (cblas_chpr2_p == NULL)
            cblas_chpr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const void* alpha, const void* x, const int incx, const void* y,
                          const int incy, void* a))GET_FUNC(h_libcblas, "cblas_chpr2");
        if (cblas_chpr2_p != NULL)
            cblas_chpr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_zhpr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void* alpha, const void* x, const int incx, const void* y,
                                const int incy, void* a) {
    if (cblas_library() != NULL) {
        if (cblas_zhpr2_p == NULL)
            cblas_zhpr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const void* alpha, const void* x, const int incx, const void* y,
                          const int incy, void* a))GET_FUNC(h_libcblas, "cblas_zhpr2");
        if (cblas_zhpr2_p != NULL)
            cblas_zhpr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_ssbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const float alpha, const float* a, const int lda,
                                const float* x, const int incx, const float beta, float* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ssbmv_p == NULL)
            cblas_ssbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const float alpha, const float* a, const int lda,
                                      const float* x, const int incx, const float beta, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ssbmv");
        if (cblas_ssbmv_p != NULL)
            cblas_ssbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dsbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const double alpha, const double* a, const int lda,
                                const double* x, const int incx, const double beta, double* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dsbmv_p == NULL)
            cblas_dsbmv_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                const double alpha, const double* a, const int lda, const double* x, const int incx,
                const double beta, double* y, const int incy))GET_FUNC(h_libcblas, "cblas_dsbmv");
        if (cblas_dsbmv_p != NULL)
            cblas_dsbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_ssymv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float* a, const int lda, const float* x,
                                const int incx, const float beta, float* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ssymv_p == NULL)
            cblas_ssymv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float* a, const int lda,
                                      const float* x, const int incx, const float beta, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ssymv");
        if (cblas_ssymv_p != NULL)
            cblas_ssymv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dsymv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double* a, const int lda, const double* x,
                                const int incx, const double beta, double* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dsymv_p == NULL)
            cblas_dsymv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double* a, const int lda,
                                      const double* x, const int incx, const double beta, double* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dsymv");
        if (cblas_dsymv_p != NULL)
            cblas_dsymv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_ssyr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const float* x, const int incx, float* a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr_p == NULL)
            cblas_ssyr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const float* x, const int incx, float* a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_ssyr");
        if (cblas_ssyr_p != NULL)
            cblas_ssyr_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_dsyr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const double* x, const int incx, double* a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr_p == NULL)
            cblas_dsyr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const double* x, const int incx, double* a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_dsyr");
        if (cblas_dsyr_p != NULL)
            cblas_dsyr_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_ssyr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float* x, const int incx, const float* y,
                                const int incy, float* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr2_p == NULL)
            cblas_ssyr2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float* x, const int incx,
                                      const float* y, const int incy, float* a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_ssyr2");
        if (cblas_ssyr2_p != NULL)
            cblas_ssyr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_dsyr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double* x, const int incx,
                                const double* y, const int incy, double* a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr2_p == NULL)
            cblas_dsyr2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double* x, const int incx,
                                      const double* y, const int incy, double* a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_dsyr2");
        if (cblas_dsyr2_p != NULL)
            cblas_dsyr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_sspmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float* a, const float* x, const int incx,
                                const float beta, float* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sspmv_p == NULL)
            cblas_sspmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float* a, const float* x,
                                      const int incx, const float beta, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sspmv");
        if (cblas_sspmv_p != NULL)
            cblas_sspmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_dspmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double* a, const double* x,
                                const int incx, const double beta, double* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dspmv_p == NULL)
            cblas_dspmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double* a, const double* x,
                                      const int incx, const double beta, double* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dspmv");
        if (cblas_dspmv_p != NULL)
            cblas_dspmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_sspr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const float* x, const int incx, float* a) {
    if (cblas_library() != NULL) {
        if (cblas_sspr_p == NULL)
            cblas_sspr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const float* x, const int incx,
                                     float* a))GET_FUNC(h_libcblas, "cblas_sspr");
        if (cblas_sspr_p != NULL)
            cblas_sspr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_dspr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const double* x, const int incx, double* a) {
    if (cblas_library() != NULL) {
        if (cblas_dspr_p == NULL)
            cblas_dspr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const double* x, const int incx,
                                     double* a))GET_FUNC(h_libcblas, "cblas_dspr");
        if (cblas_dspr_p != NULL)
            cblas_dspr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_sspr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float* x, const int incx, const float* y,
                                const int incy, float* a) {
    if (cblas_library() != NULL) {
        if (cblas_sspr2_p == NULL)
            cblas_sspr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const float alpha, const float* x, const int incx, const float* y,
                          const int incy, float* a))GET_FUNC(h_libcblas, "cblas_sspr2");
        if (cblas_sspr2_p != NULL)
            cblas_sspr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_dspr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double* x, const int incx,
                                const double* y, const int incy, double* a) {
    if (cblas_library() != NULL) {
        if (cblas_dspr2_p == NULL)
            cblas_dspr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const double alpha, const double* x, const int incx, const double* y,
                          const int incy, double* a))GET_FUNC(h_libcblas, "cblas_dspr2");
        if (cblas_dspr2_p != NULL)
            cblas_dspr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_stbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const float* a,
                                const int lda, float* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stbmv_p == NULL)
            cblas_stbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const float* a, const int lda, float* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_stbmv");
        if (cblas_stbmv_p != NULL)
            cblas_stbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_dtbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const double* a,
                                const int lda, double* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtbmv_p == NULL)
            cblas_dtbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const double* a, const int lda, double* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dtbmv");
        if (cblas_dtbmv_p != NULL)
            cblas_dtbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ctbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                                const int lda, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctbmv_p == NULL)
            cblas_ctbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void* a, const int lda, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ctbmv");
        if (cblas_ctbmv_p != NULL)
            cblas_ctbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ztbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                                const int lda, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztbmv_p == NULL)
            cblas_ztbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void* a, const int lda, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ztbmv");
        if (cblas_ztbmv_p != NULL)
            cblas_ztbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_stbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const float* a,
                                const int lda, float* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stbsv_p == NULL)
            cblas_stbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const float* a, const int lda, float* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_stbsv");
        if (cblas_stbsv_p != NULL)
            cblas_stbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_dtbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const double* a,
                                const int lda, double* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtbsv_p == NULL)
            cblas_dtbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const double* a, const int lda, double* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dtbsv");
        if (cblas_dtbsv_p != NULL)
            cblas_dtbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ctbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                                const int lda, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctbsv_p == NULL)
            cblas_ctbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void* a, const int lda, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ctbsv");
        if (cblas_ctbsv_p != NULL)
            cblas_ctbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ztbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void* a,
                                const int lda, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztbsv_p == NULL)
            cblas_ztbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void* a, const int lda, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ztbsv");
        if (cblas_ztbsv_p != NULL)
            cblas_ztbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_stpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stpmv_p == NULL)
            cblas_stpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_stpmv");
        if (cblas_stpmv_p != NULL)
            cblas_stpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_dtpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtpmv_p == NULL)
            cblas_dtpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_dtpmv");
        if (cblas_dtpmv_p != NULL)
            cblas_dtpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ctpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctpmv_p == NULL)
            cblas_ctpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctpmv");
        if (cblas_ctpmv_p != NULL)
            cblas_ctpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ztpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztpmv_p == NULL)
            cblas_ztpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztpmv");
        if (cblas_ztpmv_p != NULL)
            cblas_ztpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_stpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stpsv_p == NULL)
            cblas_stpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float* a, float* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_stpsv");
        if (cblas_stpsv_p != NULL)
            cblas_stpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_dtpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtpsv_p == NULL)
            cblas_dtpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double* a, double* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_dtpsv");
        if (cblas_dtpsv_p != NULL)
            cblas_dtpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ctpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctpsv_p == NULL)
            cblas_ctpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctpsv");
        if (cblas_ctpsv_p != NULL)
            cblas_ctpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ztpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztpsv_p == NULL)
            cblas_ztpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztpsv");
        if (cblas_ztpsv_p != NULL)
            cblas_ztpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_strmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                                float* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_strmv_p == NULL)
            cblas_strmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                          float* x, const int incx))GET_FUNC(h_libcblas, "cblas_strmv");
        if (cblas_strmv_p != NULL)
            cblas_strmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_dtrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                                double* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtrmv_p == NULL)
            cblas_dtrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                          double* x, const int incx))GET_FUNC(h_libcblas, "cblas_dtrmv");
        if (cblas_dtrmv_p != NULL)
            cblas_dtrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ctrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                                void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctrmv_p == NULL)
            cblas_ctrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, const int lda, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctrmv");
        if (cblas_ctrmv_p != NULL)
            cblas_ctrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ztrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                                void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztrmv_p == NULL)
            cblas_ztrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, const int lda, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztrmv");
        if (cblas_ztrmv_p != NULL)
            cblas_ztrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_strsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                                float* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_strsv_p == NULL)
            cblas_strsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float* a, const int lda,
                          float* x, const int incx))GET_FUNC(h_libcblas, "cblas_strsv");
        if (cblas_strsv_p != NULL)
            cblas_strsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_dtrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                                double* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtrsv_p == NULL)
            cblas_dtrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double* a, const int lda,
                          double* x, const int incx))GET_FUNC(h_libcblas, "cblas_dtrsv");
        if (cblas_dtrsv_p != NULL)
            cblas_dtrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ctrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                                void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctrsv_p == NULL)
            cblas_ctrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, const int lda, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctrsv");
        if (cblas_ctrsv_p != NULL)
            cblas_ctrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ztrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void* a, const int lda,
                                void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztrsv_p == NULL)
            cblas_ztrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void* a, const int lda, void* x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztrsv");
        if (cblas_ztrsv_p != NULL)
            cblas_ztrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

/* Level 1 */

static float (*cblas_sasum_p)(const int n, const float* x, const int incx);
static double (*cblas_dasum_p)(const int n, const double* x, const int incx);
static float (*cblas_scasum_p)(const int n, const void* x, const int incx);
static double (*cblas_dzasum_p)(const int n, const void* x, const int incx);
static void (*cblas_saxpy_p)(const int n, const float alpha, const float* x, const int incx,
                             float* y, const int incy);
static void (*cblas_daxpy_p)(const int n, const double alpha, const double* x, const int incx,
                             double* y, const int incy);
static void (*cblas_caxpy_p)(const int n, const void* alpha, const void* x, const int incx, void* y,
                             const int incy);
static void (*cblas_zaxpy_p)(const int n, const void* alpha, const void* x, const int incx, void* y,
                             const int incy);
static void (*cblas_scopy_p)(const int n, const float* x, const int incx, float* y, const int incy);
static void (*cblas_dcopy_p)(const int n, const double* x, const int incx, double* y,
                             const int incy);
static void (*cblas_ccopy_p)(const int n, const void* x, const int incx, void* y, const int incy);
static void (*cblas_zcopy_p)(const int n, const void* x, const int incx, void* y, const int incy);
static float (*cblas_sdot_p)(const int n, const float* x, const int incx, const float* y,
                             const int incy);
static double (*cblas_ddot_p)(const int n, const double* x, const int incx, const double* y,
                              const int incy);
static double (*cblas_dsdot_p)(const int n, const float* x, const int incx, const float* y,
                               const int incy);
static float (*cblas_sdsdot_p)(const int n, const float sb, const float* x, const int incx,
                               const float* y, const int incy);
static float (*cblas_snrm2_p)(const int n, const float* x, const int incx);
static double (*cblas_dnrm2_p)(const int n, const double* x, const int incx);
static float (*cblas_scnrm2_p)(const int n, const void* x, const int incx);
static double (*cblas_dznrm2_p)(const int n, const void* x, const int incx);
static void (*cblas_srot_p)(const int n, float* x, const int incx, float* y, const int incy,
                            const float c, const float s);
static void (*cblas_drot_p)(const int n, double* x, const int incx, double* y, const int incy,
                            const double c, const double s);
static void (*csrot_p)(const int* n, void* x, const int* incx, void* y, const int* incy,
                       const float* c, const float* s);
static void (*zdrot_p)(const int* n, void* x, const int* incx, void* y, const int* incy,
                       const double* c, const double* s);
static void (*cblas_srotg_p)(float* a, float* b, float* c, float* s);
static void (*cblas_drotg_p)(double* a, double* b, double* c, double* s);
static void (*crotg_p)(void* a, void* b, float* c, void* s);
static void (*zrotg_p)(void* a, void* b, double* c, void* s);
static void (*cblas_srotm_p)(const int n, float* x, const int incx, float* y, const int incy,
                             const float* param);
static void (*cblas_drotm_p)(const int n, double* x, const int incx, double* y, const int incy,
                             const double* param);
static void (*cblas_srotmg_p)(float* d1, float* d2, float* x1, float y1, float* param);
static void (*cblas_drotmg_p)(double* d1, double* d2, double* x1, double y1, double* param);
static void (*cblas_sscal_p)(const int n, const float alpha, float* x, const int incx);
static void (*cblas_dscal_p)(const int n, const double alpha, double* x, const int incx);
static void (*cblas_cscal_p)(const int n, const void* alpha, void* x, const int incx);
static void (*cblas_zscal_p)(const int n, const void* alpha, void* x, const int incx);
static void (*cblas_csscal_p)(const int n, const float alpha, void* x, const int incx);
static void (*cblas_zdscal_p)(const int n, const double alpha, void* x, const int incx);
static void (*cblas_sswap_p)(const int n, float* x, const int incx, float* y, const int incy);
static void (*cblas_dswap_p)(const int n, double* x, const int incx, double* y, const int incy);
static void (*cblas_cswap_p)(const int n, void* x, const int incx, void* y, const int incy);
static void (*cblas_zswap_p)(const int n, void* x, const int incx, void* y, const int incy);
static void (*cblas_cdotc_sub_p)(const int n, const void* x, const int incx, const void* y,
                                 const int incy, void* pres);
static void (*cblas_zdotc_sub_p)(const int n, const void* x, const int incx, const void* y,
                                 const int incy, void* pres);
static void (*cblas_cdotu_sub_p)(const int n, const void* x, const int incx, const void* y,
                                 const int incy, void* pres);
static void (*cblas_zdotu_sub_p)(const int n, const void* x, const int incx, const void* y,
                                 const int incy, void* pres);
static int (*cblas_isamax_p)(const int n, const float* x, const int incx);
static int (*cblas_idamax_p)(const int n, const double* x, const int incx);
static int (*cblas_icamax_p)(const int n, const void* x, const int incx);
static int (*cblas_izamax_p)(const int n, const void* x, const int incx);

static float cblas_sasum_wrapper(const int n, const float* x, const int incx) {
    float sasum_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sasum_p == NULL)
            cblas_sasum_p = (float (*)(const int n, const float* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_sasum");
        if (cblas_sasum_p != NULL)
            sasum_res = cblas_sasum_p(n, x, incx);
    }
    return sasum_res;
}

static double cblas_dasum_wrapper(const int n, const double* x, const int incx) {
    double dasum_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dasum_p == NULL)
            cblas_dasum_p = (double (*)(const int n, const double* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dasum");
        if (cblas_dasum_p != NULL)
            dasum_res = cblas_dasum_p(n, x, incx);
    }
    return dasum_res;
}

static float cblas_scasum_wrapper(const int n, const void* x, const int incx) {
    float scasum_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_scasum_p == NULL)
            cblas_scasum_p = (float (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_scasum");
        if (cblas_scasum_p != NULL)
            scasum_res = cblas_scasum_p(n, x, incx);
    }
    return scasum_res;
}

static double cblas_dzasum_wrapper(const int n, const void* x, const int incx) {
    double dzasum_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dzasum_p == NULL)
            cblas_dzasum_p = (double (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dzasum");
        if (cblas_dzasum_p != NULL)
            dzasum_res = cblas_dzasum_p(n, x, incx);
    }
    return dzasum_res;
}

static void cblas_saxpy_wrapper(const int n, const float alpha, const float* x, const int incx,
                                float* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_saxpy_p == NULL)
            cblas_saxpy_p =
                (void (*)(const int n, const float alpha, const float* x, const int incx, float* y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_saxpy");
        if (cblas_saxpy_p != NULL)
            cblas_saxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_daxpy_wrapper(const int n, const double alpha, const double* x, const int incx,
                                double* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_daxpy_p == NULL)
            cblas_daxpy_p =
                (void (*)(const int n, const double alpha, const double* x, const int incx,
                          double* y, const int incy))GET_FUNC(h_libcblas, "cblas_daxpy");
        if (cblas_daxpy_p != NULL)
            cblas_daxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_caxpy_wrapper(const int n, const void* alpha, const void* x, const int incx,
                                void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_caxpy_p == NULL)
            cblas_caxpy_p = (void (*)(const int n, const void* alpha, const void* x, const int incx,
                                      void* y, const int incy))GET_FUNC(h_libcblas, "cblas_caxpy");
        if (cblas_caxpy_p != NULL)
            cblas_caxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_zaxpy_wrapper(const int n, const void* alpha, const void* x, const int incx,
                                void* y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zaxpy_p == NULL)
            cblas_zaxpy_p = (void (*)(const int n, const void* alpha, const void* x, const int incx,
                                      void* y, const int incy))GET_FUNC(h_libcblas, "cblas_zaxpy");
        if (cblas_zaxpy_p != NULL)
            cblas_zaxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_scopy_wrapper(const int n, const float* x, const int incx, float* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_scopy_p == NULL)
            cblas_scopy_p = (void (*)(const int n, const float* x, const int incx, float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_scopy");
        if (cblas_scopy_p != NULL)
            cblas_scopy_p(n, x, incx, y, incy);
    }
}

static void cblas_dcopy_wrapper(const int n, const double* x, const int incx, double* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dcopy_p == NULL)
            cblas_dcopy_p = (void (*)(const int n, const double* x, const int incx, double* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dcopy");
        if (cblas_dcopy_p != NULL)
            cblas_dcopy_p(n, x, incx, y, incy);
    }
}

static void cblas_ccopy_wrapper(const int n, const void* x, const int incx, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ccopy_p == NULL)
            cblas_ccopy_p = (void (*)(const int n, const void* x, const int incx, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ccopy");
        if (cblas_ccopy_p != NULL)
            cblas_ccopy_p(n, x, incx, y, incy);
    }
}

static void cblas_zcopy_wrapper(const int n, const void* x, const int incx, void* y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zcopy_p == NULL)
            cblas_zcopy_p = (void (*)(const int n, const void* x, const int incx, void* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zcopy");
        if (cblas_zcopy_p != NULL)
            cblas_zcopy_p(n, x, incx, y, incy);
    }
}

static float cblas_sdot_wrapper(const int n, const float* x, const int incx, const float* y,
                                const int incy) {
    float sdot_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sdot_p == NULL)
            cblas_sdot_p = (float (*)(const int n, const float* x, const int incx, const float* y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sdot");
        if (cblas_sdot_p != NULL)
            sdot_res = cblas_sdot_p(n, x, incx, y, incy);
    }
    return sdot_res;
}

static double cblas_ddot_wrapper(const int n, const double* x, const int incx, const double* y,
                                 const int incy) {
    double ddot_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_ddot_p == NULL)
            cblas_ddot_p =
                (double (*)(const int n, const double* x, const int incx, const double* y,
                            const int incy))GET_FUNC(h_libcblas, "cblas_ddot");
        if (cblas_ddot_p != NULL)
            ddot_res = cblas_ddot_p(n, x, incx, y, incy);
    }
    return ddot_res;
}

static double cblas_dsdot_wrapper(const int n, const float* x, const int incx, const float* y,
                                  const int incy) {
    double dsdot_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dsdot_p == NULL)
            cblas_dsdot_p = (double (*)(const int n, const float* x, const int incx, const float* y,
                                        const int incy))GET_FUNC(h_libcblas, "cblas_dsdot");
        if (cblas_dsdot_p != NULL)
            dsdot_res = cblas_dsdot_p(n, x, incx, y, incy);
    }
    return dsdot_res;
}

static float cblas_sdsdot_wrapper(const int n, const float sb, const float* x, const int incx,
                                  const float* y, const int incy) {
    float sdsdot_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sdsdot_p == NULL)
            cblas_sdsdot_p =
                (float (*)(const int n, const float sb, const float* x, const int incx,
                           const float* y, const int incy))GET_FUNC(h_libcblas, "cblas_sdsdot");
        if (cblas_sdsdot_p != NULL)
            sdsdot_res = cblas_sdsdot_p(n, sb, x, incx, y, incy);
    }
    return sdsdot_res;
}

static float cblas_snrm2_wrapper(const int n, const float* x, const int incx) {
    float snrm2_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_snrm2_p == NULL)
            cblas_snrm2_p = (float (*)(const int n, const float* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_snrm2");
        if (cblas_snrm2_p != NULL)
            snrm2_res = cblas_snrm2_p(n, x, incx);
    }
    return snrm2_res;
}

static double cblas_dnrm2_wrapper(const int n, const double* x, const int incx) {
    double dnrm2_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dnrm2_p == NULL)
            cblas_dnrm2_p = (double (*)(const int n, const double* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dnrm2");
        if (cblas_dnrm2_p != NULL)
            dnrm2_res = cblas_dnrm2_p(n, x, incx);
    }
    return dnrm2_res;
}

static float cblas_scnrm2_wrapper(const int n, const void* x, const int incx) {
    float scnrm2_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_scnrm2_p == NULL)
            cblas_scnrm2_p = (float (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_scnrm2");
        if (cblas_scnrm2_p != NULL)
            scnrm2_res = cblas_scnrm2_p(n, x, incx);
    }
    return scnrm2_res;
}

static double cblas_dznrm2_wrapper(const int n, const void* x, const int incx) {
    double dznrm2_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dznrm2_p == NULL)
            cblas_dznrm2_p = (double (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dznrm2");
        if (cblas_dznrm2_p != NULL)
            dznrm2_res = cblas_dznrm2_p(n, x, incx);
    }
    return dznrm2_res;
}

static void cblas_srot_wrapper(const int n, float* x, const int incx, float* y, const int incy,
                               const float c, const float s) {
    if (cblas_library() != NULL) {
        if (cblas_srot_p == NULL)
            cblas_srot_p =
                (void (*)(const int n, float* x, const int incx, float* y, const int incy,
                          const float c, const float s))GET_FUNC(h_libcblas, "cblas_srot");
        if (cblas_srot_p != NULL)
            cblas_srot_p(n, x, incx, y, incy, c, s);
    }
}

static void cblas_drot_wrapper(const int n, double* x, const int incx, double* y, const int incy,
                               const double c, const double s) {
    if (cblas_library() != NULL) {
        if (cblas_drot_p == NULL)
            cblas_drot_p =
                (void (*)(const int n, double* x, const int incx, double* y, const int incy,
                          const double c, const double s))GET_FUNC(h_libcblas, "cblas_drot");
        if (cblas_drot_p != NULL)
            cblas_drot_p(n, x, incx, y, incy, c, s);
    }
}

static void csrot_wrapper(const int* n, void* x, const int* incx, void* y, const int* incy,
                          const float* c, const float* s) {
    if (blas_library() != NULL) {
        if (csrot_p == NULL)
            csrot_p = (void (*)(const int* n, void* x, const int* incx, void* y, const int* incy,
                                const float* c, const float* s))GET_FUNC(h_libblas, "csrot_");
        if (csrot_p == NULL)
            csrot_p = (void (*)(const int* n, void* x, const int* incx, void* y, const int* incy,
                                const float* c, const float* s))GET_FUNC(h_libblas, "CSROT");
        if (csrot_p != NULL)
            csrot_p(n, x, incx, y, incy, c, s);
    }
}

static void zdrot_wrapper(const int* n, void* x, const int* incx, void* y, const int* incy,
                          const double* c, const double* s) {
    if (blas_library() != NULL) {
        if (zdrot_p == NULL)
            zdrot_p = (void (*)(const int* n, void* x, const int* incx, void* y, const int* incy,
                                const double* c, const double* s))GET_FUNC(h_libblas, "zdrot_");
        if (zdrot_p == NULL)
            zdrot_p = (void (*)(const int* n, void* x, const int* incx, void* y, const int* incy,
                                const double* c, const double* s))GET_FUNC(h_libblas, "ZDROT");
        if (zdrot_p != NULL)
            zdrot_p(n, x, incx, y, incy, c, s);
    }
}

static void cblas_srotg_wrapper(float* a, float* b, float* c, float* s) {
    if (cblas_library() != NULL) {
        if (cblas_srotg_p == NULL)
            cblas_srotg_p = (void (*)(float* a, float* b, float* c, float* s))GET_FUNC(
                h_libcblas, "cblas_srotg");
        if (cblas_srotg_p != NULL)
            cblas_srotg_p(a, b, c, s);
    }
}

static void cblas_drotg_wrapper(double* a, double* b, double* c, double* s) {
    if (cblas_library() != NULL) {
        if (cblas_drotg_p == NULL)
            cblas_drotg_p = (void (*)(double* a, double* b, double* c, double* s))GET_FUNC(
                h_libcblas, "cblas_drotg");
        if (cblas_drotg_p != NULL)
            cblas_drotg_p(a, b, c, s);
    }
}

static void crotg_wrapper(void* a, void* b, float* c, void* s) {
    if (blas_library() != NULL) {
        if (crotg_p == NULL)
            crotg_p = (void (*)(void* a, void* b, float* c, void* s))GET_FUNC(h_libblas, "crotg_");
        if (crotg_p == NULL)
            crotg_p = (void (*)(void* a, void* b, float* c, void* s))GET_FUNC(h_libblas, "CROTG");
        if (crotg_p != NULL)
            crotg_p(a, b, c, s);
    }
}

static void zrotg_wrapper(void* a, void* b, double* c, void* s) {
    if (blas_library() != NULL) {
        if (zrotg_p == NULL)
            zrotg_p = (void (*)(void* a, void* b, double* c, void* s))GET_FUNC(h_libblas, "zrotg_");
        if (zrotg_p == NULL)
            zrotg_p = (void (*)(void* a, void* b, double* c, void* s))GET_FUNC(h_libblas, "ZROTG");
        if (zrotg_p != NULL)
            zrotg_p(a, b, c, s);
    }
}

static void cblas_srotm_wrapper(const int n, float* x, const int incx, float* y, const int incy,
                                const float* param) {
    if (cblas_library() != NULL) {
        if (cblas_srotm_p == NULL)
            cblas_srotm_p =
                (void (*)(const int n, float* x, const int incx, float* y, const int incy,
                          const float* param))GET_FUNC(h_libcblas, "cblas_srotm");
        if (cblas_srotm_p != NULL)
            cblas_srotm_p(n, x, incx, y, incy, param);
    }
}

static void cblas_drotm_wrapper(const int n, double* x, const int incx, double* y, const int incy,
                                const double* param) {
    if (cblas_library() != NULL) {
        if (cblas_drotm_p == NULL)
            cblas_drotm_p =
                (void (*)(const int n, double* x, const int incx, double* y, const int incy,
                          const double* param))GET_FUNC(h_libcblas, "cblas_drotm");
        if (cblas_drotm_p != NULL)
            cblas_drotm_p(n, x, incx, y, incy, param);
    }
}

static void cblas_srotmg_wrapper(float* d1, float* d2, float* x1, float y1, float* param) {
    if (cblas_library() != NULL) {
        if (cblas_srotmg_p == NULL)
            cblas_srotmg_p = (void (*)(float* d1, float* d2, float* x1, float y1,
                                       float* param))GET_FUNC(h_libcblas, "cblas_srotmg");
        if (cblas_srotmg_p != NULL)
            cblas_srotmg_p(d1, d2, x1, y1, param);
    }
}

static void cblas_drotmg_wrapper(double* d1, double* d2, double* x1, double y1, double* param) {
    if (cblas_library() != NULL) {
        if (cblas_drotmg_p == NULL)
            cblas_drotmg_p = (void (*)(double* d1, double* d2, double* x1, double y1,
                                       double* param))GET_FUNC(h_libcblas, "cblas_drotmg");
        if (cblas_drotmg_p != NULL)
            cblas_drotmg_p(d1, d2, x1, y1, param);
    }
}

static void cblas_sscal_wrapper(const int n, const float alpha, float* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_sscal_p == NULL)
            cblas_sscal_p = (void (*)(const int n, const float alpha, float* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_sscal");
        if (cblas_sscal_p != NULL)
            cblas_sscal_p(n, alpha, x, incx);
    }
}

static void cblas_dscal_wrapper(const int n, const double alpha, double* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dscal_p == NULL)
            cblas_dscal_p = (void (*)(const int n, const double alpha, double* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dscal");
        if (cblas_dscal_p != NULL)
            cblas_dscal_p(n, alpha, x, incx);
    }
}

static void cblas_cscal_wrapper(const int n, const void* alpha, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_cscal_p == NULL)
            cblas_cscal_p = (void (*)(const int n, const void* alpha, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_cscal");
        if (cblas_cscal_p != NULL)
            cblas_cscal_p(n, alpha, x, incx);
    }
}

static void cblas_zscal_wrapper(const int n, const void* alpha, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_zscal_p == NULL)
            cblas_zscal_p = (void (*)(const int n, const void* alpha, void* x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_zscal");
        if (cblas_zscal_p != NULL)
            cblas_zscal_p(n, alpha, x, incx);
    }
}

static void cblas_csscal_wrapper(const int n, const float alpha, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_csscal_p == NULL)
            cblas_csscal_p = (void (*)(const int n, const float alpha, void* x,
                                       const int incx))GET_FUNC(h_libcblas, "cblas_csscal");
        if (cblas_csscal_p != NULL)
            cblas_csscal_p(n, alpha, x, incx);
    }
}

static void cblas_zdscal_wrapper(const int n, const double alpha, void* x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_zdscal_p == NULL)
            cblas_zdscal_p = (void (*)(const int n, const double alpha, void* x,
                                       const int incx))GET_FUNC(h_libcblas, "cblas_zdscal");
        if (cblas_zdscal_p != NULL)
            cblas_zdscal_p(n, alpha, x, incx);
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

static void cblas_cdotc_sub_wrapper(const int n, const void* x, const int incx, const void* y,
                                    const int incy, void* pres) {
    if (cblas_library() != NULL) {
        if (cblas_cdotc_sub_p == NULL)
            cblas_cdotc_sub_p =
                (void (*)(const int n, const void* x, const int incx, const void* y, const int incy,
                          void* pres))GET_FUNC(h_libcblas, "cblas_cdotc_sub");
        if (cblas_cdotc_sub_p != NULL)
            cblas_cdotc_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_zdotc_sub_wrapper(const int n, const void* x, const int incx, const void* y,
                                    const int incy, void* pres) {
    if (cblas_library() != NULL) {
        if (cblas_zdotc_sub_p == NULL)
            cblas_zdotc_sub_p =
                (void (*)(const int n, const void* x, const int incx, const void* y, const int incy,
                          void* pres))GET_FUNC(h_libcblas, "cblas_zdotc_sub");
        if (cblas_zdotc_sub_p != NULL)
            cblas_zdotc_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_cdotu_sub_wrapper(const int n, const void* x, const int incx, const void* y,
                                    const int incy, void* pres) {
    if (cblas_library() != NULL) {
        if (cblas_cdotu_sub_p == NULL)
            cblas_cdotu_sub_p =
                (void (*)(const int n, const void* x, const int incx, const void* y, const int incy,
                          void* pres))GET_FUNC(h_libcblas, "cblas_cdotu_sub");
        if (cblas_cdotu_sub_p != NULL)
            cblas_cdotu_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_zdotu_sub_wrapper(const int n, const void* x, const int incx, const void* y,
                                    const int incy, void* pres) {
    if (cblas_library() != NULL) {
        if (cblas_zdotu_sub_p == NULL)
            cblas_zdotu_sub_p =
                (void (*)(const int n, const void* x, const int incx, const void* y, const int incy,
                          void* pres))GET_FUNC(h_libcblas, "cblas_zdotu_sub");
        if (cblas_zdotu_sub_p != NULL)
            cblas_zdotu_sub_p(n, x, incx, y, incy, pres);
    }
}

static int cblas_isamax_wrapper(const int n, const float* x, const int incx) {
    int isamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_isamax_p == NULL)
            cblas_isamax_p = (int (*)(const int n, const float* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_isamax");
        if (cblas_isamax_p != NULL)
            isamax_res = cblas_isamax_p(n, x, incx);
    }
    return isamax_res;
}

static int cblas_idamax_wrapper(const int n, const double* x, const int incx) {
    int idamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_idamax_p == NULL)
            cblas_idamax_p = (int (*)(const int n, const double* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_idamax");
        if (cblas_idamax_p != NULL)
            idamax_res = cblas_idamax_p(n, x, incx);
    }
    return idamax_res;
}

static int cblas_icamax_wrapper(const int n, const void* x, const int incx) {
    int icamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_icamax_p == NULL)
            cblas_icamax_p = (int (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_icamax");
        if (cblas_icamax_p != NULL)
            icamax_res = cblas_icamax_p(n, x, incx);
    }
    return icamax_res;
}

static int cblas_izamax_wrapper(const int n, const void* x, const int incx) {
    int izamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_izamax_p == NULL)
            cblas_izamax_p = (int (*)(const int n, const void* x, const int incx))GET_FUNC(
                h_libcblas, "cblas_izamax");
        if (cblas_izamax_p != NULL)
            izamax_res = cblas_izamax_p(n, x, incx);
    }
    return izamax_res;
}
}

#endif /* header guard */
