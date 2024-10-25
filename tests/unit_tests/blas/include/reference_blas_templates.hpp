/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _REFERENCE_BLAS_TEMPLATES_HPP__
#define _REFERENCE_BLAS_TEMPLATES_HPP__

#include <stdlib.h>
#include <complex>
#include <cstdint>
#include "cblas.h"
#include "oneapi/mkl/types.hpp"
#include "test_helper.hpp"
#include "reference_blas_wrappers.hpp"

inline bool isNonTranspose(CBLAS_TRANSPOSE trans) {
    return trans == CblasNoTrans;
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src& src, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int row,
                            int col, int ld, T_dest*& dest) {
    int i, j, Iend, Jend;
    if (layout == CblasColMajor) {
        Jend = isNonTranspose(trans) ? col : row;
        Iend = isNonTranspose(trans) ? row : col;
    }
    else {
        Jend = isNonTranspose(trans) ? row : col;
        Iend = isNonTranspose(trans) ? col : row;
    }

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            dest[i + ld * j] = (T_dest)src[i + ld * j];
        }
    }
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src& src, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int row,
                            int col, int ld, T_dest off, T_dest*& dest) {
    int i, j, Iend, Jend;
    if (layout == CblasColMajor) {
        Jend = isNonTranspose(trans) ? col : row;
        Iend = isNonTranspose(trans) ? row : col;
    }
    else {
        Jend = isNonTranspose(trans) ? row : col;
        Iend = isNonTranspose(trans) ? col : row;
    }

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
        }
    }
}

template <typename T_src, typename T_dest, typename T_off>
static inline void copy_mat(T_src& src, CBLAS_LAYOUT layout, int row, int col, int ld,
                            CBLAS_OFFSET off_kind, T_off off, T_dest& dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int i, j;
    T_data tmp;

    int Jend = (layout == CblasColMajor) ? col : row;
    int Iend = (layout == CblasColMajor) ? row : col;

    if (off_kind == CblasFixOffset) {
        tmp = off[0];
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (((off_kind == CblasColOffset) && (layout == CblasColMajor)) ||
             ((off_kind == CblasRowOffset) && (layout == CblasRowMajor))) {
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                tmp = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (j = 0; j < Jend; j++) {
            tmp = off[j];
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

template <typename T_src, typename T_desc>
static inline void update_c(T_src& src, CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, int row,
                            int col, int ld, T_desc*& dest) {
    int i, j;

    int Jend = (layout == CblasColMajor) ? col : row;
    int Iend = (layout == CblasColMajor) ? row : col;

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            if (((upper_lower == CblasUpper) && (layout == CblasColMajor)) ||
                ((upper_lower == CblasLower) && (layout == CblasRowMajor))) {
                if (j >= i)
                    dest[i + ld * j] = (T_desc)src[i + ld * j];
                else
                    dest[i + ld * j] = (T_desc)0.0;
            }
            else {
                if (j <= i)
                    dest[i + ld * j] = (T_desc)src[i + ld * j];
                else
                    dest[i + ld * j] = (T_desc)0.0;
            }
        }
    }
}

/* Level 3 */

template <typename fp>
static void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
                 const int* n, const int* k, const fp* alpha, const fp* a, const int* lda,
                 const fp* b, const int* ldb, const fp* beta, fp* c, const int* ldc);

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const sycl::half* alpha, const sycl::half* a, const int* lda,
          const sycl::half* b, const int* ldb, const sycl::half* beta, sycl::half* c,
          const int* ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb, sizec;
    const float alphaf = *alpha;
    const float betaf = *beta;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    float* af = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float* bf = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    float* cf = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizec);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, cf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, alphaf, af, *lda, bf, *ldb, betaf, cf,
                        *ldc);
    copy_mat(cf, layout, CblasNoTrans, *m, *n, *ldc, c);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const float* alpha, const float* a, const int* lda,
          const float* b, const int* ldb, const float* beta, float* c, const int* ldc) {
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                        *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const double* alpha, const double* a, const int* lda,
          const double* b, const int* ldb, const double* beta, double* c, const int* ldc) {
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                        *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const std::complex<float>* alpha,
          const std::complex<float>* a, const int* lda, const std::complex<float>* b,
          const int* ldb, const std::complex<float>* beta, std::complex<float>* c, const int* ldc) {
    cblas_cgemm_wrapper(layout, transa, transb, *m, *n, *k, (const void*)alpha, (const void*)a,
                        *lda, (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const std::complex<double>* alpha,
          const std::complex<double>* a, const int* lda, const std::complex<double>* b,
          const int* ldb, const std::complex<double>* beta, std::complex<double>* c,
          const int* ldc) {
    cblas_zgemm_wrapper(layout, transa, transb, *m, *n, *k, (const void*)alpha, (const void*)a,
                        *lda, (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <typename fpa, typename fpc>
static void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
                 const int* n, const int* k, const fpc* alpha, const fpa* a, const int* lda,
                 const fpa* b, const int* ldb, const fpc* beta, fpc* c, const int* ldc);

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const float* alpha, const sycl::half* a, const int* lda,
          const sycl::half* b, const int* ldb, const float* beta, float* c, const int* ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
    }
    float* af = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float* bf = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, af, *lda, bf, *ldb, *beta, c,
                        *ldc);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int* m,
          const int* n, const int* k, const float* alpha, const oneapi::mkl::bfloat16* a,
          const int* lda, const oneapi::mkl::bfloat16* b, const int* ldb, const float* beta,
          float* c, const int* ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
    }
    float* af = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float* bf = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, af, *lda, bf, *ldb, *beta, c,
                        *ldc);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
}

template <typename fp>
static void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m,
                 const int* n, const fp* alpha, const fp* a, const int* lda, const fp* b,
                 const int* ldb, const fp* beta, fp* c, const int* ldc);

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
          const float* beta, float* c, const int* ldc) {
    cblas_ssymm_wrapper(layout, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
          const double* beta, double* c, const int* ldc) {
    cblas_dsymm_wrapper(layout, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* b, const int* ldb, const std::complex<float>* beta,
          std::complex<float>* c, const int* ldc) {
    cblas_csymm_wrapper(layout, left_right, uplo, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* b, const int* ldb, const std::complex<double>* beta,
          std::complex<double>* c, const int* ldc) {
    cblas_zsymm_wrapper(layout, left_right, uplo, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <typename fp>
static void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n,
                 const int* k, const fp* alpha, const fp* a, const int* lda, const fp* beta, fp* c,
                 const int* ldc);

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const float* alpha, const float* a, const int* lda, const float* beta, float* c,
          const int* ldc) {
    cblas_ssyrk_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const double* alpha, const double* a, const int* lda, const double* beta, double* c,
          const int* ldc) {
    cblas_dsyrk_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* beta, std::complex<float>* c, const int* ldc) {
    cblas_csyrk_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                        (const void*)beta, (void*)c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* beta, std::complex<double>* c, const int* ldc) {
    cblas_zsyrk_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                        (const void*)beta, (void*)c, *ldc);
}

template <typename fp>
static void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m,
                 const int* n, const fp* alpha, const fp* a, const int* lda, const fp* b,
                 const int* ldb, const fp* beta, fp* c, const int* ldc);

template <>
void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* b, const int* ldb, const std::complex<float>* beta,
          std::complex<float>* c, const int* ldc) {
    cblas_chemm_wrapper(layout, left_right, uplo, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <>
void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int* m, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* b, const int* ldb, const std::complex<double>* beta,
          std::complex<double>* c, const int* ldc) {
    cblas_zhemm_wrapper(layout, left_right, uplo, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n,
                 const int* k, const fp_scalar* alpha, const fp_data* a, const int* lda,
                 const fp_scalar* beta, fp_data* c, const int* ldc);

template <>
void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const float* alpha, const std::complex<float>* a, const int* lda, const float* beta,
          std::complex<float>* c, const int* ldc) {
    cblas_cherk_wrapper(layout, uplo, trans, *n, *k, *alpha, (const void*)a, *lda, *beta, (void*)c,
                        *ldc);
}

template <>
void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
          const double* alpha, const std::complex<double>* a, const int* lda, const double* beta,
          std::complex<double>* c, const int* ldc) {
    cblas_zherk_wrapper(layout, uplo, trans, *n, *k, *alpha, (const void*)a, *lda, *beta, (void*)c,
                        *ldc);
}

template <typename fp>
static void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n,
                  const int* k, const fp* alpha, const fp* a, const int* lda, const fp* b,
                  const int* ldb, const fp* beta, fp* c, const int* ldc);

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
           const float* beta, float* c, const int* ldc) {
    cblas_ssyr2k_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const double* alpha, const double* a, const int* lda, const double* b, const int* ldb,
           const double* beta, double* c, const int* ldc) {
    cblas_dsyr2k_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
           const std::complex<float>* b, const int* ldb, const std::complex<float>* beta,
           std::complex<float>* c, const int* ldc) {
    cblas_csyr2k_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                         (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
           const std::complex<double>* b, const int* ldb, const std::complex<double>* beta,
           std::complex<double>* c, const int* ldc) {
    cblas_zsyr2k_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                         (const void*)b, *ldb, (const void*)beta, (void*)c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n,
                  const int* k, const fp_data* alpha, const fp_data* a, const int* lda,
                  const fp_data* b, const int* ldb, const fp_scalar* beta, fp_data* c,
                  const int* ldc);

template <>
void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
           const std::complex<float>* b, const int* ldb, const float* beta, std::complex<float>* c,
           const int* ldc) {
    cblas_cher2k_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                         (const void*)b, *ldb, *beta, (void*)c, *ldc);
}

template <>
void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int* n, const int* k,
           const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
           const std::complex<double>* b, const int* ldb, const double* beta,
           std::complex<double>* c, const int* ldc) {
    cblas_zher2k_wrapper(layout, uplo, trans, *n, *k, (const void*)alpha, (const void*)a, *lda,
                         (const void*)b, *ldb, *beta, (void*)c, *ldc);
}

template <typename fp>
static void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                 CBLAS_DIAG diag, const int* m, const int* n, const fp* alpha, const fp* a,
                 const int* lda, fp* b, const int* ldb);

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const float* alpha, const float* a,
          const int* lda, float* b, const int* ldb) {
    cblas_strmm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const double* alpha, const double* a,
          const int* lda, double* b, const int* ldb) {
    cblas_dtrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const std::complex<float>* alpha,
          const std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb) {
    cblas_ctrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void*)alpha,
                        (const void*)a, *lda, (void*)b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const std::complex<double>* alpha,
          const std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb) {
    cblas_ztrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void*)alpha,
                        (const void*)a, *lda, (void*)b, *ldb);
}

template <typename fp>
static void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                 CBLAS_DIAG diag, const int* m, const int* n, const fp* alpha, const fp* a,
                 const int* lda, fp* b, const int* ldb);

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const float* alpha, const float* a,
          const int* lda, float* b, const int* ldb) {
    cblas_strsm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const double* alpha, const double* a,
          const int* lda, double* b, const int* ldb) {
    cblas_dtrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const std::complex<float>* alpha,
          const std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb) {
    cblas_ctrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void*)alpha,
                        (const void*)a, *lda, (void*)b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int* m, const int* n, const std::complex<double>* alpha,
          const std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb) {
    cblas_ztrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void*)alpha,
                        (const void*)a, *lda, (void*)b, *ldb);
}

/* Level 2 */

template <typename fp>
static void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n,
                 const fp* alpha, const fp* a, const int* lda, const fp* x, const int* incx,
                 const fp* beta, fp* y, const int* incy);

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n,
          const float* alpha, const float* a, const int* lda, const float* x, const int* incx,
          const float* beta, float* y, const int* incy) {
    cblas_sgemv_wrapper(layout, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n,
          const double* alpha, const double* a, const int* lda, const double* x, const int* incx,
          const double* beta, double* y, const int* incy) {
    cblas_dgemv_wrapper(layout, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
          std::complex<float>* y, const int* incy) {
    cblas_cgemv_wrapper(layout, trans, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
          std::complex<double>* y, const int* incy) {
    cblas_zgemv_wrapper(layout, trans, *m, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <typename fp>
static void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n, int* kl,
                 int* ku, const fp* alpha, const fp* a, const int* lda, const fp* x,
                 const int* incx, const fp* beta, fp* y, const int* incy);

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n, int* kl, int* ku,
          const float* alpha, const float* a, const int* lda, const float* x, const int* incx,
          const float* beta, float* y, const int* incy) {
    cblas_sgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y,
                        *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n, int* kl, int* ku,
          const double* alpha, const double* a, const int* lda, const double* x, const int* incx,
          const double* beta, double* y, const int* incy) {
    cblas_dgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y,
                        *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n, int* kl, int* ku,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
          std::complex<float>* y, const int* incy) {
    cblas_cgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int* m, const int* n, int* kl, int* ku,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
          std::complex<double>* y, const int* incy) {
    cblas_zgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <typename fp>
static void ger(CBLAS_LAYOUT layout, const int* m, const int* n, const fp* alpha, const fp* x,
                const int* incx, const fp* y, const int* incy, fp* a, const int* lda);

template <>
void ger(CBLAS_LAYOUT layout, const int* m, const int* n, const float* alpha, const float* x,
         const int* incx, const float* y, const int* incy, float* a, const int* lda) {
    cblas_sger_wrapper(layout, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void ger(CBLAS_LAYOUT layout, const int* m, const int* n, const double* alpha, const double* x,
         const int* incx, const double* y, const int* incy, double* a, const int* lda) {
    cblas_dger_wrapper(layout, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void gerc(CBLAS_LAYOUT layout, const int* m, const int* n, const fp* alpha, const fp* x,
                 const int* incx, const fp* y, const int* incy, fp* a, const int* lda);

template <>
void gerc(CBLAS_LAYOUT layout, const int* m, const int* n, const std::complex<float>* alpha,
          const std::complex<float>* x, const int* incx, const std::complex<float>* y,
          const int* incy, std::complex<float>* a, const int* lda) {
    cblas_cgerc_wrapper(layout, *m, *n, (const void*)alpha, (const void*)x, *incx, (const void*)y,
                        *incy, (void*)a, *lda);
}

template <>
void gerc(CBLAS_LAYOUT layout, const int* m, const int* n, const std::complex<double>* alpha,
          const std::complex<double>* x, const int* incx, const std::complex<double>* y,
          const int* incy, std::complex<double>* a, const int* lda) {
    cblas_zgerc_wrapper(layout, *m, *n, (const void*)alpha, (const void*)x, *incx, (const void*)y,
                        *incy, (void*)a, *lda);
}

template <typename fp>
static void geru(CBLAS_LAYOUT layout, const int* m, const int* n, const fp* alpha, const fp* x,
                 const int* incx, const fp* y, const int* incy, fp* a, const int* lda);

template <>
void geru(CBLAS_LAYOUT layout, const int* m, const int* n, const std::complex<float>* alpha,
          const std::complex<float>* x, const int* incx, const std::complex<float>* y,
          const int* incy, std::complex<float>* a, const int* lda) {
    cblas_cgeru_wrapper(layout, *m, *n, (const void*)alpha, (const void*)x, *incx, (const void*)y,
                        *incy, (void*)a, *lda);
}

template <>
void geru(CBLAS_LAYOUT layout, const int* m, const int* n, const std::complex<double>* alpha,
          const std::complex<double>* x, const int* incx, const std::complex<double>* y,
          const int* incy, std::complex<double>* a, const int* lda) {
    cblas_zgeru_wrapper(layout, *m, *n, (const void*)alpha, (const void*)x, *incx, (const void*)y,
                        *incy, (void*)a, *lda);
}

template <typename fp>
static void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
                 const fp* alpha, const fp* a, const int* lda, const fp* x, const int* incx,
                 const fp* beta, fp* y, const int* incy);

template <>
void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
          std::complex<float>* y, const int* incy) {
    cblas_chbmv_wrapper(layout, upper_lower, *n, *k, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <>
void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
          std::complex<double>* y, const int* incy) {
    cblas_zhbmv_wrapper(layout, upper_lower, *n, *k, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <typename fp>
static void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* a, const int* lda, const fp* x, const int* incx, const fp* beta, fp* y,
                 const int* incy);

template <>
void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* a, const int* lda,
          const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
          std::complex<float>* y, const int* incy) {
    cblas_chemv_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <>
void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* a, const int* lda,
          const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
          std::complex<double>* y, const int* incy) {
    cblas_zhemv_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)a, *lda,
                        (const void*)x, *incx, (const void*)beta, (void*)y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp_scalar* alpha,
                const fp_data* x, const int* incx, fp_data* a, const int* lda);

template <>
void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
         const std::complex<float>* x, const int* incx, std::complex<float>* a, const int* lda) {
    cblas_cher_wrapper(layout, upper_lower, *n, *alpha, (const void*)x, *incx, (void*)a, *lda);
}

template <>
void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
         const std::complex<double>* x, const int* incx, std::complex<double>* a, const int* lda) {
    cblas_zher_wrapper(layout, upper_lower, *n, *alpha, (const void*)x, *incx, (void*)a, *lda);
}

template <typename fp>
static void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* x, const int* incx, const fp* y, const int* incy, fp* a, const int* lda);

template <>
void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* x, const int* incx,
          const std::complex<float>* y, const int* incy, std::complex<float>* a, const int* lda) {
    cblas_cher2_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)x, *incx,
                        (const void*)y, *incy, (void*)a, *lda);
}

template <>
void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* x, const int* incx,
          const std::complex<double>* y, const int* incy, std::complex<double>* a, const int* lda) {
    cblas_zher2_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)x, *incx,
                        (const void*)y, *incy, (void*)a, *lda);
}

template <typename fp>
static void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* a, const fp* x, const int* incx, const fp* beta, fp* y, const int* incy);

template <>
void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* a,
          const std::complex<float>* x, const int* incx, const std::complex<float>* beta,
          std::complex<float>* y, const int* incy) {
    cblas_chpmv_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)a, (const void*)x,
                        *incx, (const void*)beta, (void*)y, *incy);
}

template <>
void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* a,
          const std::complex<double>* x, const int* incx, const std::complex<double>* beta,
          std::complex<double>* y, const int* incy) {
    cblas_zhpmv_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)a, (const void*)x,
                        *incx, (const void*)beta, (void*)y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp_scalar* alpha,
                const fp_data* x, const int* incx, fp_data* a);

template <>
void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
         const std::complex<float>* x, const int* incx, std::complex<float>* a) {
    cblas_chpr_wrapper(layout, upper_lower, *n, *alpha, (const void*)x, *incx, (void*)a);
}

template <>
void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
         const std::complex<double>* x, const int* incx, std::complex<double>* a) {
    cblas_zhpr_wrapper(layout, upper_lower, *n, *alpha, (const void*)x, *incx, (void*)a);
}

template <typename fp>
static void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* x, const int* incx, const fp* y, const int* incy, fp* a);

template <>
void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<float>* alpha, const std::complex<float>* x, const int* incx,
          const std::complex<float>* y, const int* incy, std::complex<float>* a) {
    cblas_chpr2_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)x, *incx,
                        (const void*)y, *incy, (void*)a);
}

template <>
void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n,
          const std::complex<double>* alpha, const std::complex<double>* x, const int* incx,
          const std::complex<double>* y, const int* incy, std::complex<double>* a) {
    cblas_zhpr2_wrapper(layout, upper_lower, *n, (const void*)alpha, (const void*)x, *incx,
                        (const void*)y, *incy, (void*)a);
}

template <typename fp>
static void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
                 const fp* alpha, const fp* a, const int* lda, const fp* x, const int* incx,
                 const fp* beta, fp* y, const int* incy);

template <>
void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
          const float* alpha, const float* a, const int* lda, const float* x, const int* incx,
          const float* beta, float* y, const int* incy) {
    cblas_ssbmv_wrapper(layout, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const int* k,
          const double* alpha, const double* a, const int* lda, const double* x, const int* incx,
          const double* beta, double* y, const int* incy) {
    cblas_dsbmv_wrapper(layout, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* a, const int* lda, const fp* x, const int* incx, const fp* beta, fp* y,
                 const int* incy);

template <>
void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
          const float* a, const int* lda, const float* x, const int* incx, const float* beta,
          float* y, const int* incy) {
    cblas_ssymv_wrapper(layout, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
          const double* a, const int* lda, const double* x, const int* incx, const double* beta,
          double* y, const int* incy) {
    cblas_dsymv_wrapper(layout, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                const fp* x, const int* incx, fp* a, const int* lda);

template <>
void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
         const float* x, const int* incx, float* a, const int* lda) {
    cblas_ssyr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <>
void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
         const double* x, const int* incx, double* a, const int* lda) {
    cblas_dsyr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <typename fp>
static void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* x, const int* incx, const fp* y, const int* incy, fp* a, const int* lda);

template <>
void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
          const float* x, const int* incx, const float* y, const int* incy, float* a,
          const int* lda) {
    cblas_ssyr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
          const double* x, const int* incx, const double* y, const int* incy, double* a,
          const int* lda) {
    cblas_dsyr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* a, const fp* x, const int* incx, const fp* beta, fp* y, const int* incy);

template <>
void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
          const float* a, const float* x, const int* incx, const float* beta, float* y,
          const int* incy) {
    cblas_sspmv_wrapper(layout, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <>
void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
          const double* a, const double* x, const int* incx, const double* beta, double* y,
          const int* incy) {
    cblas_dspmv_wrapper(layout, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                const fp* x, const int* incx, fp* a);

template <>
void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
         const float* x, const int* incx, float* a) {
    cblas_sspr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a);
}

template <>
void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
         const double* x, const int* incx, double* a) {
    cblas_dspr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a);
}

template <typename fp>
static void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const fp* alpha,
                 const fp* x, const int* incx, const fp* y, const int* incy, fp* a);

template <>
void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const float* alpha,
          const float* x, const int* incx, const float* y, const int* incy, float* a) {
    cblas_sspr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <>
void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int* n, const double* alpha,
          const double* x, const int* incx, const double* y, const int* incy, double* a) {
    cblas_dspr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <typename fp>
static void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const int* k, const fp* a, const int* lda,
                 fp* x, const int* incx);

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const float* a, const int* lda, float* x, const int* incx) {
    cblas_stbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const double* a, const int* lda, double* x, const int* incx) {
    cblas_dtbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const std::complex<float>* a, const int* lda,
          std::complex<float>* x, const int* incx) {
    cblas_ctbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void*)a, *lda,
                        (void*)x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const std::complex<double>* a, const int* lda,
          std::complex<double>* x, const int* incx) {
    cblas_ztbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void*)a, *lda,
                        (void*)x, *incx);
}

template <typename fp>
static void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const int* k, const fp* a, const int* lda,
                 fp* x, const int* incx);

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const float* a, const int* lda, float* x, const int* incx) {
    cblas_stbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const double* a, const int* lda, double* x, const int* incx) {
    cblas_dtbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const std::complex<float>* a, const int* lda,
          std::complex<float>* x, const int* incx) {
    cblas_ctbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void*)a, *lda,
                        (void*)x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const int* k, const std::complex<double>* a, const int* lda,
          std::complex<double>* x, const int* incx) {
    cblas_ztbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void*)a, *lda,
                        (void*)x, *incx);
}

template <typename fp>
static void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const fp* a, fp* x, const int* incx);

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const float* a, float* x, const int* incx) {
    cblas_stpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const double* a, double* x, const int* incx) {
    cblas_dtpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<float>* a, std::complex<float>* x, const int* incx) {
    cblas_ctpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, (void*)x, *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<double>* a, std::complex<double>* x, const int* incx) {
    cblas_ztpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, (void*)x, *incx);
}

template <typename fp>
static void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const fp* a, fp* x, const int* incx);

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const float* a, float* x, const int* incx) {
    cblas_stpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const double* a, double* x, const int* incx) {
    cblas_dtpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<float>* a, std::complex<float>* x, const int* incx) {
    cblas_ctpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, (void*)x, *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<double>* a, std::complex<double>* x, const int* incx) {
    cblas_ztpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, (void*)x, *incx);
}

template <typename fp>
static void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const fp* a, const int* lda, fp* x,
                 const int* incx);

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const float* a, const int* lda, float* x, const int* incx) {
    cblas_strmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const double* a, const int* lda, double* x, const int* incx) {
    cblas_dtrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<float>* a, const int* lda, std::complex<float>* x,
          const int* incx) {
    cblas_ctrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, *lda, (void*)x,
                        *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<double>* a, const int* lda, std::complex<double>* x,
          const int* incx) {
    cblas_ztrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, *lda, (void*)x,
                        *incx);
}

template <typename fp>
static void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int* n, const fp* a, const int* lda, fp* x,
                 const int* incx);

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const float* a, const int* lda, float* x, const int* incx) {
    cblas_strsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const double* a, const int* lda, double* x, const int* incx) {
    cblas_dtrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<float>* a, const int* lda, std::complex<float>* x,
          const int* incx) {
    cblas_ctrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, *lda, (void*)x,
                        *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int* n, const std::complex<double>* a, const int* lda, std::complex<double>* x,
          const int* incx) {
    cblas_ztrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void*)a, *lda, (void*)x,
                        *incx);
}

/* Level 1 */

template <typename fp_data, typename fp_res>
static fp_res asum(const int* n, const fp_data* x, const int* incx);

template <>
float asum(const int* n, const float* x, const int* incx) {
    return cblas_sasum_wrapper(*n, x, *incx);
}

template <>
double asum(const int* n, const double* x, const int* incx) {
    return cblas_dasum_wrapper(*n, x, *incx);
}

template <>
float asum(const int* n, const std::complex<float>* x, const int* incx) {
    return cblas_scasum_wrapper(*n, (const void*)x, *incx);
}

template <>
double asum(const int* n, const std::complex<double>* x, const int* incx) {
    return cblas_dzasum_wrapper(*n, (const void*)x, *incx);
}

template <typename fp>
static void axpy(const int* n, const fp* alpha, const fp* x, const int* incx, fp* y,
                 const int* incy);

template <>
void axpy(const int* n, const float* alpha, const float* x, const int* incx, float* y,
          const int* incy) {
    cblas_saxpy_wrapper(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int* n, const double* alpha, const double* x, const int* incx, double* y,
          const int* incy) {
    cblas_daxpy_wrapper(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int* n, const std::complex<float>* alpha, const std::complex<float>* x,
          const int* incx, std::complex<float>* y, const int* incy) {
    cblas_caxpy_wrapper(*n, (const void*)alpha, (const void*)x, *incx, (void*)y, *incy);
}

template <>
void axpy(const int* n, const std::complex<double>* alpha, const std::complex<double>* x,
          const int* incx, std::complex<double>* y, const int* incy) {
    cblas_zaxpy_wrapper(*n, (const void*)alpha, (const void*)x, *incx, (void*)y, *incy);
}

template <typename fp>
static void copy(const int* n, const fp* x, const int* incx, fp* y, const int* incy);

template <>
void copy(const int* n, const float* x, const int* incx, float* y, const int* incy) {
    cblas_scopy_wrapper(*n, x, *incx, y, *incy);
}
template <>
void copy(const int* n, const double* x, const int* incx, double* y, const int* incy) {
    cblas_dcopy_wrapper(*n, x, *incx, y, *incy);
}
template <>
void copy(const int* n, const std::complex<float>* x, const int* incx, std::complex<float>* y,
          const int* incy) {
    cblas_ccopy_wrapper(*n, (const void*)x, *incx, (void*)y, *incy);
}
template <>
void copy(const int* n, const std::complex<double>* x, const int* incx, std::complex<double>* y,
          const int* incy) {
    cblas_zcopy_wrapper(*n, (const void*)x, *incx, (void*)y, *incy);
}

template <typename fp, typename fp_res>
static fp_res dot(const int* n, const fp* x, const int* incx, const fp* y, const int* incy);

template <>
float dot(const int* n, const float* x, const int* incx, const float* y, const int* incy) {
    return cblas_sdot_wrapper(*n, x, *incx, y, *incy);
}

template <>
double dot(const int* n, const double* x, const int* incx, const double* y, const int* incy) {
    return cblas_ddot_wrapper(*n, x, *incx, y, *incy);
}

template <>
double dot(const int* n, const float* x, const int* incx, const float* y, const int* incy) {
    return cblas_dsdot_wrapper(*n, x, *incx, y, *incy);
}

static float sdsdot(const int* n, const float* sb, const float* x, const int* incx, const float* y,
                    const int* incy) {
    return cblas_sdsdot_wrapper(*n, *sb, x, *incx, y, *incy);
}

template <typename fp, typename fp_res>
static fp_res nrm2(const int* n, const fp* x, const int* incx);

template <>
float nrm2(const int* n, const float* x, const int* incx) {
    return cblas_snrm2_wrapper(*n, x, *incx);
}

template <>
double nrm2(const int* n, const double* x, const int* incx) {
    return cblas_dnrm2_wrapper(*n, x, *incx);
}

template <>
float nrm2(const int* n, const std::complex<float>* x, const int* incx) {
    return cblas_scnrm2_wrapper(*n, (const void*)x, *incx);
}

template <>
double nrm2(const int* n, const std::complex<double>* x, const int* incx) {
    return cblas_dznrm2_wrapper(*n, (const void*)x, *incx);
}

template <typename fp, typename fp_scalar>
static void rot(const int* n, fp* x, const int* incx, fp* y, const int* incy, const fp_scalar* c,
                const fp_scalar* s);

template <>
void rot(const int* n, float* x, const int* incx, float* y, const int* incy, const float* c,
         const float* s) {
    cblas_srot_wrapper(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int* n, double* x, const int* incx, double* y, const int* incy, const double* c,
         const double* s) {
    cblas_drot_wrapper(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int* n, std::complex<float>* x, const int* incx, std::complex<float>* y,
         const int* incy, const float* c, const float* s) {
    csrot_wrapper(n, (void*)x, incx, (void*)y, incy, c, s);
}

template <>
void rot(const int* n, std::complex<double>* x, const int* incx, std::complex<double>* y,
         const int* incy, const double* c, const double* s) {
    zdrot_wrapper(n, (void*)x, incx, (void*)y, incy, c, s);
}

template <typename fp, typename fp_c>
static void rotg(fp* a, fp* b, fp_c* c, fp* s);

template <>
void rotg(float* a, float* b, float* c, float* s) {
    cblas_srotg_wrapper(a, b, c, s);
}

template <>
void rotg(double* a, double* b, double* c, double* s) {
    cblas_drotg_wrapper(a, b, c, s);
}

template <>
void rotg(std::complex<float>* a, std::complex<float>* b, float* c, std::complex<float>* s) {
    crotg_wrapper((void*)a, (void*)b, c, (void*)s);
}

template <>
void rotg(std::complex<double>* a, std::complex<double>* b, double* c, std::complex<double>* s) {
    zrotg_wrapper((void*)a, (void*)b, c, (void*)s);
}

template <typename fp>
static void rotm(const int* n, fp* x, const int* incx, fp* y, const int* incy, const fp* param);

template <>
void rotm(const int* n, float* x, const int* incx, float* y, const int* incy, const float* param) {
    cblas_srotm_wrapper(*n, x, *incx, y, *incy, param);
}

template <>
void rotm(const int* n, double* x, const int* incx, double* y, const int* incy,
          const double* param) {
    cblas_drotm_wrapper(*n, x, *incx, y, *incy, param);
}

template <typename fp>
static void rotmg(fp* d1, fp* d2, fp* x1, fp* y1, fp* param);

template <>
void rotmg(float* d1, float* d2, float* x1, float* y1, float* param) {
    cblas_srotmg_wrapper(d1, d2, x1, *y1, param);
}

template <>
void rotmg(double* d1, double* d2, double* x1, double* y1, double* param) {
    cblas_drotmg_wrapper(d1, d2, x1, *y1, param);
}

template <typename fp_scalar, typename fp_data>
static void scal(const int* n, const fp_scalar* alpha, fp_data* x, const int* incx);

template <>
void scal(const int* n, const float* alpha, float* x, const int* incx) {
    cblas_sscal_wrapper(*n, *alpha, x, *incx);
}
template <>
void scal(const int* n, const double* alpha, double* x, const int* incx) {
    cblas_dscal_wrapper(*n, *alpha, x, *incx);
}
template <>
void scal(const int* n, const std::complex<float>* alpha, std::complex<float>* x, const int* incx) {
    cblas_cscal_wrapper(*n, (const void*)alpha, (void*)x, *incx);
}
template <>
void scal(const int* n, const std::complex<double>* alpha, std::complex<double>* x,
          const int* incx) {
    cblas_zscal_wrapper(*n, (const void*)alpha, (void*)x, *incx);
}
template <>
void scal(const int* n, const float* alpha, std::complex<float>* x, const int* incx) {
    cblas_csscal_wrapper(*n, *alpha, (void*)x, *incx);
}
template <>
void scal(const int* n, const double* alpha, std::complex<double>* x, const int* incx) {
    cblas_zdscal_wrapper(*n, *alpha, (void*)x, *incx);
}

template <typename fp>
static void swap(const int* n, fp* x, const int* incx, fp* y, const int* incy);

template <>
void swap(const int* n, float* x, const int* incx, float* y, const int* incy) {
    cblas_sswap_wrapper(*n, x, *incx, y, *incy);
}

template <>
void swap(const int* n, double* x, const int* incx, double* y, const int* incy) {
    cblas_dswap_wrapper(*n, x, *incx, y, *incy);
}

template <>
void swap(const int* n, std::complex<float>* x, const int* incx, std::complex<float>* y,
          const int* incy) {
    cblas_cswap_wrapper(*n, (void*)x, *incx, (void*)y, *incy);
}

template <>
void swap(const int* n, std::complex<double>* x, const int* incx, std::complex<double>* y,
          const int* incy) {
    cblas_zswap_wrapper(*n, (void*)x, *incx, (void*)y, *incy);
}

template <typename fp>
static void dotc(fp* pres, const int* n, const fp* x, const int* incx, const fp* y,
                 const int* incy);

template <>
void dotc(std::complex<float>* pres, const int* n, const std::complex<float>* x, const int* incx,
          const std::complex<float>* y, const int* incy) {
    cblas_cdotc_sub_wrapper(*n, (const void*)x, *incx, (const void*)y, *incy, (void*)pres);
}

template <>
void dotc(std::complex<double>* pres, const int* n, const std::complex<double>* x, const int* incx,
          const std::complex<double>* y, const int* incy) {
    cblas_zdotc_sub_wrapper(*n, (const void*)x, *incx, (const void*)y, *incy, (void*)pres);
}

template <typename fp>
static void dotu(fp* pres, const int* n, const fp* x, const int* incx, const fp* y,
                 const int* incy);

template <>
void dotu(std::complex<float>* pres, const int* n, const std::complex<float>* x, const int* incx,
          const std::complex<float>* y, const int* incy) {
    cblas_cdotu_sub_wrapper(*n, (const void*)x, *incx, (const void*)y, *incy, (void*)pres);
}

template <>
void dotu(std::complex<double>* pres, const int* n, const std::complex<double>* x, const int* incx,
          const std::complex<double>* y, const int* incy) {
    cblas_zdotu_sub_wrapper(*n, (const void*)x, *incx, (const void*)y, *incy, (void*)pres);
}

template <typename fp>
static int iamax(const int* n, const fp* x, const int* incx);

template <>
int iamax(const int* n, const float* x, const int* incx) {
    return cblas_isamax_wrapper(*n, x, *incx);
}

template <>
int iamax(const int* n, const double* x, const int* incx) {
    return cblas_idamax_wrapper(*n, x, *incx);
}

template <>
int iamax(const int* n, const std::complex<float>* x, const int* incx) {
    return cblas_icamax_wrapper(*n, (const void*)x, *incx);
}

template <>
int iamax(const int* n, const std::complex<double>* x, const int* incx) {
    return cblas_izamax_wrapper(*n, (const void*)x, *incx);
}

inline float abs_val(float val) {
    return std::abs(val);
}

inline double abs_val(double val) {
    return std::abs(val);
}

inline float abs_val(std::complex<float> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

inline double abs_val(std::complex<double> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

template <typename fp>
static int iamin(const int* n, const fp* x, const int* incx);

template <>
int iamin(const int* n, const float* x, const int* incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int* n, const double* x, const int* incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int* n, const std::complex<float>* x, const int* incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int* n, const std::complex<double>* x, const int* incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

/* Extensions */

template <typename fp>
static void axpby(const int* n, const fp* alpha, const fp* x, const int* incx, const fp* beta,
                  fp* y, const int* incy);

template <>
void axpby(const int* n, const float* alpha, const float* x, const int* incx, const float* beta,
           float* y, const int* incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int* n, const double* alpha, const double* x, const int* incx, const double* beta,
           double* y, const int* incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int* n, const std::complex<float>* alpha, const std::complex<float>* x,
           const int* incx, const std::complex<float>* beta, std::complex<float>* y,
           const int* incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int* n, const std::complex<double>* alpha, const std::complex<double>* x,
           const int* incx, const std::complex<double>* beta, std::complex<double>* y,
           const int* incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <typename fps, typename fpa, typename fpb, typename fpc>
static void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                      CBLAS_OFFSET offsetc, const int* m, const int* n, const int* k,
                      const fps* alpha, const fpa* a, const int* lda, const fpa* ao, const fpb* b,
                      const int* ldb, const fpb* bo, const fps* beta, fpc* c, const int* ldc,
                      const fpc* co);

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int* m, const int* n, const int* k, const float* alpha,
               const int8_t* a, const int* lda, const int8_t* ao, const int8_t* b, const int* ldb,
               const int8_t* bo, const float* beta, int32_t* c, const int* ldc, const int32_t* co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double* ad = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double* bd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double* cd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int* m, const int* n, const int* k, const float* alpha,
               const int8_t* a, const int* lda, const int8_t* ao, const uint8_t* b, const int* ldb,
               const uint8_t* bo, const float* beta, int32_t* c, const int* ldc,
               const int32_t* co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double* ad = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double* bd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double* cd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int* m, const int* n, const int* k, const float* alpha,
               const uint8_t* a, const int* lda, const uint8_t* ao, const int8_t* b, const int* ldb,
               const int8_t* bo, const float* beta, int32_t* c, const int* ldc, const int32_t* co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double* ad = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double* bd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double* cd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int* m, const int* n, const int* k, const float* alpha,
               const uint8_t* a, const int* lda, const uint8_t* ao, const uint8_t* b,
               const int* ldb, const uint8_t* bo, const float* beta, int32_t* c, const int* ldc,
               const int32_t* co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double* ad = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double* bd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double* cd = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <typename fp>
static void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
                  CBLAS_TRANSPOSE transb, const int* n, const int* k, const fp* alpha, const fp* a,
                  const int* lda, const fp* b, const int* ldb, const fp* beta, fp* c,
                  const int* ldc);

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int* n, const int* k, const float* alpha, const float* a,
           const int* lda, const float* b, const int* ldb, const float* beta, float* c,
           const int* ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    float* cf = (float*)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_sgemm_wrapper(layout, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int* n, const int* k, const double* alpha, const double* a,
           const int* lda, const double* b, const int* ldb, const double* beta, double* c,
           const int* ldc) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    double* cf = (double*)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_dgemm_wrapper(layout, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int* n, const int* k, const std::complex<float>* alpha,
           const std::complex<float>* a, const int* lda, const std::complex<float>* b,
           const int* ldb, const std::complex<float>* beta, std::complex<float>* c,
           const int* ldc) {
    // Not supported in NETLIB. CGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<float>* cf =
        (std::complex<float>*)oneapi::mkl::aligned_alloc(64, sizeof(std::complex<float>) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_cgemm_wrapper(layout, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int* n, const int* k, const std::complex<double>* alpha,
           const std::complex<double>* a, const int* lda, const std::complex<double>* b,
           const int* ldb, const std::complex<double>* beta, std::complex<double>* c,
           const int* ldc) {
    // Not supported in NETLIB. ZGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<double>* cf =
        (std::complex<double>*)oneapi::mkl::aligned_alloc(64, sizeof(std::complex<double>) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_zgemm_wrapper(layout, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <typename fp>
static void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int* m, const int* n,
                 const fp* a, const int* lda, const fp* x, const int* incx, fp* c, const int* ldc);

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int* m, const int* n, const float* a,
          const int* lda, const float* x, const int* incx, float* c, const int* ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    float tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor)
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
                else
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor)
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
                else
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int* m, const int* n, const double* a,
          const int* lda, const double* x, const int* incx, double* c, const int* ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    double tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor)
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
                else
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor)
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
                else
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int* m, const int* n,
          const std::complex<float>* a, const int* lda, const std::complex<float>* x,
          const int* incx, std::complex<float>* c, const int* ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    std::complex<float> tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor) {
                    c[j + i * (*ldc)] =
                        std::complex<float>((tmp.real() * a[j + i * (*lda)].real() -
                                             tmp.imag() * a[j + i * (*lda)].imag()),
                                            (tmp.real() * a[j + i * (*lda)].imag() +
                                             tmp.imag() * a[j + i * (*lda)].real()));
                }
                else {
                    c[i + j * (*ldc)] =
                        std::complex<float>((tmp.real() * a[i + j * (*lda)].real() -
                                             tmp.imag() * a[i + j * (*lda)].imag()),
                                            (tmp.real() * a[i + j * (*lda)].imag() +
                                             tmp.imag() * a[i + j * (*lda)].real()));
                }
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor) {
                    c[i + j * (*ldc)] =
                        std::complex<float>((tmp.real() * a[i + j * (*lda)].real() -
                                             tmp.imag() * a[i + j * (*lda)].imag()),
                                            (tmp.real() * a[i + j * (*lda)].imag() +
                                             tmp.imag() * a[i + j * (*lda)].real()));
                }
                else {
                    c[j + i * (*ldc)] =
                        std::complex<float>((tmp.real() * a[j + i * (*lda)].real() -
                                             tmp.imag() * a[j + i * (*lda)].imag()),
                                            (tmp.real() * a[j + i * (*lda)].imag() +
                                             tmp.imag() * a[j + i * (*lda)].real()));
                }
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int* m, const int* n,
          const std::complex<double>* a, const int* lda, const std::complex<double>* x,
          const int* incx, std::complex<double>* c, const int* ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    std::complex<double> tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor) {
                    c[j + i * (*ldc)] =
                        std::complex<double>((tmp.real() * a[j + i * (*lda)].real() -
                                              tmp.imag() * a[j + i * (*lda)].imag()),
                                             (tmp.real() * a[j + i * (*lda)].imag() +
                                              tmp.imag() * a[j + i * (*lda)].real()));
                }
                else {
                    c[i + j * (*ldc)] =
                        std::complex<double>((tmp.real() * a[i + j * (*lda)].real() -
                                              tmp.imag() * a[i + j * (*lda)].imag()),
                                             (tmp.real() * a[i + j * (*lda)].imag() +
                                              tmp.imag() * a[i + j * (*lda)].real()));
                }
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor) {
                    c[i + j * (*ldc)] =
                        std::complex<double>((tmp.real() * a[i + j * (*lda)].real() -
                                              tmp.imag() * a[i + j * (*lda)].imag()),
                                             (tmp.real() * a[i + j * (*lda)].imag() +
                                              tmp.imag() * a[i + j * (*lda)].real()));
                }
                else {
                    c[j + i * (*ldc)] =
                        std::complex<double>((tmp.real() * a[j + i * (*lda)].real() -
                                              tmp.imag() * a[j + i * (*lda)].imag()),
                                             (tmp.real() * a[j + i * (*lda)].imag() +
                                              tmp.imag() * a[j + i * (*lda)].real()));
                }
            }
        }
    }
}

// std::conj can take a real type as input, but still returns a complex type.
// This version always returns the same type it has as input
template <typename fp>
fp sametype_conj(fp x) {
    if constexpr (std::is_same_v<fp, std::complex<float>> ||
                  std::is_same_v<fp, std::complex<double>>) {
        return std::conj(x);
    }
    else {
        return x;
    }
}

template <typename fp>
void omatcopy_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int64_t m, int64_t n,
                  fp alpha, fp* A, int64_t lda, fp* B, int64_t ldb) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::col_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }
    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[j * ldb + i] = alpha * A[j * lda + i];
            }
        }
    }
    else if (trans == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[i * ldb + j] = alpha * A[j * lda + i];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[i * ldb + j] = alpha * sametype_conj(A[j * lda + i]);
            }
        }
    }
}

template <typename fp>
void omatcopy2_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, const int64_t& m,
                   const int64_t& n, const fp& alpha, const fp* in_matrix, const int64_t& ld_in,
                   const int64_t& inc_in, fp* out_matrix, const int64_t& ld_out,
                   const int64_t inc_out) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::col_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }
    if (trans == oneapi::mkl::transpose::trans) {
        for (int64_t i = 0; i < logical_m; ++i) {
            for (int64_t j = 0; j < logical_n; ++j) {
                {
                    out_matrix[j * inc_out + i * ld_out] =
                        alpha * in_matrix[i * inc_in + j * ld_in];
                }
            }
        }
    }
    else if (trans == oneapi::mkl::transpose::nontrans) {
        for (int i = 0; i < logical_n; ++i) {
            for (int j = 0; j < logical_m; ++j) {
                {
                    out_matrix[j * inc_out + i * ld_out] =
                        alpha * in_matrix[j * inc_in + i * ld_in];
                }
            }
        }
    }
    else {
        for (int64_t i = 0; i < logical_m; ++i) {
            for (int64_t j = 0, c = 0; j < logical_n; ++j, ++c) {
                out_matrix[j * inc_out + i * ld_out] =
                    alpha * sametype_conj(in_matrix[i * inc_in + j * ld_in]);
            }
        }
    }

    return;
}

template <typename fp>
void imatcopy_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int64_t m, int64_t n,
                  fp alpha, fp* A, int64_t lda, int64_t ldb) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::col_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }
    std::vector<fp> temp(m * n);
    int64_t ld_temp = (trans == oneapi::mkl::transpose::nontrans ? logical_m : logical_n);

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[j * ld_temp + i] = alpha * A[j * lda + i];
            }
        }
    }
    else if (trans == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[i * ld_temp + j] = alpha * A[j * lda + i];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[i * ld_temp + j] = alpha * sametype_conj(A[j * lda + i]);
            }
        }
    }

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                A[j * ldb + i] = temp[j * ld_temp + i];
            }
        }
    }
    else {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                A[i * ldb + j] = temp[i * ld_temp + j];
            }
        }
    }
}

template <typename fp>
void omatadd_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose transa,
                 oneapi::mkl::transpose transb, int64_t m, int64_t n, fp alpha, fp* A, int64_t lda,
                 fp beta, fp* B, int64_t ldb, fp* C, int64_t ldc) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::col_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }

    for (int64_t j = 0; j < logical_n; j++) {
        for (int64_t i = 0; i < logical_m; i++) {
            C[j * ldc + i] = 0.0;
        }
    }

    if (transa == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * A[j * lda + i];
            }
        }
    }
    else if (transa == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * A[i * lda + j];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * sametype_conj(A[i * lda + j]);
            }
        }
    }

    if (transb == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * B[j * ldb + i];
            }
        }
    }
    else if (transb == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * B[i * ldb + j];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * sametype_conj(B[i * ldb + j]);
            }
        }
    }
}

#endif /* header guard */
