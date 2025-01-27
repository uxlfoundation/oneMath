/*******************************************************************************
* Copyright 2025 SiPearl
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

// Buffer APIs

template <typename T, typename CBLAS_FUNC>
void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b, int64_t ldb, T beta,
          sycl::buffer<T, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gemm>(cgh, [=]() {
            cblas_func(MAJOR, transa_, transb_, m, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb,
                       cast_to_void_if_complex(beta), accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define GEMM_LAUNCHER(TYPE, ROUTINE)                                                        \
    void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
              int64_t k, TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda,                 \
              sycl::buffer<TYPE, 1>& b, int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c,   \
              int64_t ldc) {                                                                \
        gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, ROUTINE); \
    }

GEMM_LAUNCHER(float, ::cblas_sgemm)
GEMM_LAUNCHER(double, ::cblas_dgemm)
GEMM_LAUNCHER(std::complex<float>, ::cblas_cgemm)
GEMM_LAUNCHER(std::complex<double>, ::cblas_zgemm)

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
          sycl::buffer<sycl::half, 1>& b, int64_t ldb, sycl::half beta,
          sycl::buffer<sycl::half, 1>& c, int64_t ldc) {
    queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hgemm>(cgh, [=]() {
            ::cblas_hgemm(MAJOR, transa_, transb_, m, n, k, alpha,
                          (const __fp16*)accessor_a.GET_MULTI_PTR, lda,
                          (const __fp16*)accessor_b.GET_MULTI_PTR, ldb, beta,
                          (__fp16*)accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda, sycl::buffer<sycl::half, 1>& b,
          int64_t ldb, float beta, sycl::buffer<float, 1>& c, int64_t ldc) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
    if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
        throw oneapi::math::unimplemented(
            "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
    }
#endif
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hgemm_float>(cgh, [=]() {
            if (beta == 0.0) {
                //C is not relevant as input, we can compute in half and convert result
                const float alphaf = (float)alpha;
                const float betaf = (float)beta;
#ifdef COLUMN_MAJOR
                auto f16_c = new __fp16[ldc * n]();
#endif
#ifdef ROW_MAJOR
                auto f16_c = new __fp16[ldc * m]();
#endif
                //__fp16 cp = 0.0f;
                //copy_mat(accessor_c, MAJOR, transpose::N, m, n, ldc , cp, f16_c);
                ::cblas_hgemm(MAJOR, transa_, transb_, m, n, k, alphaf,
                              (const __fp16*)accessor_a.GET_MULTI_PTR, lda,
                              (const __fp16*)accessor_b.GET_MULTI_PTR, ldb, betaf, f16_c, ldc);
                float co = 0.0f;
                copy_mat(f16_c, MAJOR, m, n, ldc, offset::F, &co, (float*)accessor_c.GET_MULTI_PTR);
                delete[] f16_c;
            }
            else {
                //need to compute in fp32
                int64_t sizea, sizeb;
#ifdef COLUMN_MAJOR
                sizea = (transa == transpose::N) ? lda * k : lda * m;
                sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
#endif
#ifdef ROW_MAJOR
                sizea = (transa == transpose::N) ? lda * m : lda * k;
                sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
#endif
                // copy A and B to float
                auto f32_a = new float[sizea]();
                auto f32_b = new float[sizeb]();
                copy_mat(accessor_a, MAJOR, transa, m, k, lda, 0.0f, f32_a);
                copy_mat(accessor_b, MAJOR, transb, k, n, ldb, 0.0f, f32_b);
                ::cblas_sgemm(MAJOR, transa_, transb_, m, n, k, alpha, f32_a, lda, f32_b, ldb, beta,
                              accessor_c.GET_MULTI_PTR, ldc);
                delete[] f32_a;
                delete[] f32_b;
            }
        });
    });
}

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          float alpha, sycl::buffer<bfloat16, 1>& a, int64_t lda, sycl::buffer<bfloat16, 1>& b,
          int64_t ldb, float beta, sycl::buffer<float, 1>& c, int64_t ldc) {
    throw unimplemented("blas", __func__, MAJOR_MINOR_TEXT);
}

template <typename T, typename CBLAS_FUNC>
void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<T> alpha, sycl::buffer<std::complex<T>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<T>, 1>& b, int64_t ldb, std::complex<T> beta,
          sycl::buffer<std::complex<T>, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hemm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, m, n, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb,
                       cast_to_void_if_complex(beta), accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define HEMM_LAUNCHER(TYPE, ROUTINE)                                                              \
    void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,        \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,        \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                    \
        hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc, ROUTINE); \
    }

HEMM_LAUNCHER(std::complex<float>, ::cblas_chemm)
HEMM_LAUNCHER(std::complex<double>, ::cblas_zhemm)

template <typename T, typename CBLAS_FUNC>
void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, T alpha,
          sycl::buffer<std::complex<T>, 1>& a, int64_t lda, T beta,
          sycl::buffer<std::complex<T>, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_herk>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, alpha, accessor_a.GET_MULTI_PTR, lda,
                       beta, accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define HERK_LAUNCHER(TYPE, ROUTINE)                                                       \
    void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
              TYPE alpha, sycl::buffer<std::complex<TYPE>, 1>& a, int64_t lda, TYPE beta,  \
              sycl::buffer<std::complex<TYPE>, 1>& c, int64_t ldc) {                       \
        herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, ROUTINE);       \
    }

HERK_LAUNCHER(float, ::cblas_cherk)
HERK_LAUNCHER(double, ::cblas_zherk)

template <typename T, typename CBLAS_FUNC>
void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<T> alpha, sycl::buffer<std::complex<T>, 1>& a, int64_t lda,
           sycl::buffer<std::complex<T>, 1>& b, int64_t ldb, T beta,
           sycl::buffer<std::complex<T>, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_her2k>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb, beta,
                       accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define HER2K_LAUNCHER(TYPE, ROUTINE)                                                         \
    void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
               std::complex<TYPE> alpha, sycl::buffer<std::complex<TYPE>, 1>& a, int64_t lda, \
               sycl::buffer<std::complex<TYPE>, 1>& b, int64_t ldb, TYPE beta,                \
               sycl::buffer<std::complex<TYPE>, 1>& c, int64_t ldc) {                         \
        her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, ROUTINE); \
    }

HER2K_LAUNCHER(float, ::cblas_cher2k)
HER2K_LAUNCHER(double, ::cblas_zher2k)

template <typename T, typename CBLAS_FUNC>
void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, T alpha,
          sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b, int64_t ldb, T beta,
          sycl::buffer<T, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_symm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, m, n, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb,
                       cast_to_void_if_complex(beta), accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define SYMM_LAUNCHER(TYPE, ROUTINE)                                                              \
    void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,        \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,        \
              int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {                    \
        symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc, ROUTINE); \
    }

SYMM_LAUNCHER(float, ::cblas_ssymm)
SYMM_LAUNCHER(double, ::cblas_dsymm)
SYMM_LAUNCHER(std::complex<float>, ::cblas_csymm)
SYMM_LAUNCHER(std::complex<double>, ::cblas_zsymm)

template <typename T, typename CBLAS_FUNC>
void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, T alpha,
          sycl::buffer<T, 1>& a, int64_t lda, T beta, sycl::buffer<T, 1>& c, int64_t ldc,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_syrk>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, cast_to_void_if_complex(beta),
                       accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define SYRK_LAUNCHER(TYPE, ROUTINE)                                                       \
    void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, TYPE beta,                \
              sycl::buffer<TYPE, 1>& c, int64_t ldc) {                                     \
        syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, ROUTINE);       \
    }

SYRK_LAUNCHER(float, ::cblas_ssyrk)
SYRK_LAUNCHER(double, ::cblas_dsyrk)
SYRK_LAUNCHER(std::complex<float>, ::cblas_csyrk)
SYRK_LAUNCHER(std::complex<double>, ::cblas_zsyrk)

template <typename T, typename CBLAS_FUNC>
void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, T alpha,
           sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b, int64_t ldb, T beta,
           sycl::buffer<T, 1>& c, int64_t ldc, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_syr2k>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_b.GET_MULTI_PTR, ldb,
                       cast_to_void_if_complex(beta), accessor_c.GET_MULTI_PTR, ldc);
        });
    });
}

#define SYR2K_LAUNCHER(TYPE, ROUTINE)                                                         \
    void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,   \
               TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b,   \
               int64_t ldb, TYPE beta, sycl::buffer<TYPE, 1>& c, int64_t ldc) {               \
        syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc, ROUTINE); \
    }

SYR2K_LAUNCHER(float, ::cblas_ssyr2k)
SYR2K_LAUNCHER(double, ::cblas_dsyr2k)
SYR2K_LAUNCHER(std::complex<float>, ::cblas_csyr2k)
SYR2K_LAUNCHER(std::complex<double>, ::cblas_zsyr2k)

template <typename T, typename CBLAS_FUNC>
void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa, diag unit_diag,
          int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
          int64_t ldb, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_trmm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                       cast_to_void_if_complex(alpha), accessor_a.GET_MULTI_PTR, lda,
                       accessor_b.GET_MULTI_PTR, ldb);
        });
    });
}

#define TRMM_LAUNCHER(TYPE, ROUTINE)                                                         \
    void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,       \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,    \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                          \
        trmm(queue, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, \
             ROUTINE);                                                                       \
    }

TRMM_LAUNCHER(float, ::cblas_strmm)
TRMM_LAUNCHER(double, ::cblas_dtrmm)
TRMM_LAUNCHER(std::complex<float>, ::cblas_ctrmm)
TRMM_LAUNCHER(std::complex<double>, ::cblas_ztrmm)

template <typename T, typename CBLAS_FUNC>
void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa, diag unit_diag,
          int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b,
          int64_t ldb, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_trsm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                       cast_to_void_if_complex(alpha), accessor_a.GET_MULTI_PTR, lda,
                       accessor_b.GET_MULTI_PTR, ldb);
        });
    });
}

#define TRSM_LAUNCHER(TYPE, ROUTINE)                                                         \
    void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,       \
              diag unit_diag, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& a,    \
              int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t ldb) {                          \
        trsm(queue, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b, ldb, \
             ROUTINE);                                                                       \
    }

TRSM_LAUNCHER(float, ::cblas_strsm)
TRSM_LAUNCHER(double, ::cblas_dtrsm)
TRSM_LAUNCHER(std::complex<float>, ::cblas_ctrsm)
TRSM_LAUNCHER(std::complex<double>, ::cblas_ztrsm)

// USM APIs

template <typename T, typename CBLAS_FUNC>
sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, T alpha, const T* a, int64_t lda, const T* b, int64_t ldb, T beta, T* c,
                 int64_t ldc, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class armpl_kernel_gemm>(cgh, [=]() {
            cblas_func(MAJOR, transa_, transb_, m, n, k, cast_to_void_if_complex(alpha), a, lda, b,
                       ldb, cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define GEMM_USM_LAUNCHER(TYPE, ROUTINE)                                                           \
    sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, \
                     int64_t k, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b,             \
                     int64_t ldb, TYPE beta, TYPE* c, int64_t ldc,                                 \
                     const std::vector<sycl::event>& dependencies) {                               \
        return gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,           \
                    dependencies, ROUTINE);                                                        \
    }

GEMM_USM_LAUNCHER(float, ::cblas_sgemm)
GEMM_USM_LAUNCHER(double, ::cblas_dgemm)
GEMM_USM_LAUNCHER(std::complex<float>, ::cblas_cgemm)
GEMM_USM_LAUNCHER(std::complex<double>, ::cblas_zgemm)

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda, const sycl::half* b,
                 int64_t ldb, sycl::half beta, sycl::half* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        host_task<class armpl_kernel_hgemm_usm>(cgh, [=]() {
            ::cblas_hgemm(MAJOR, transa_, transb_, m, n, k, alpha, (const __fp16*)a, lda,
                          (const __fp16*)b, ldb, beta, (__fp16*)c, ldc);
        });
    });
    return done;
}

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const sycl::half* a, int64_t lda, const sycl::half* b,
                 int64_t ldb, float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
#ifndef __HIPSYCL__ //hip reports aspect as not supported even if it works
        if (!verify_support<sycl::half, sycl::half>(queue, sycl::aspect::fp16)) {
            throw oneapi::math::unimplemented(
                "blas", "sycl::half", "half is not supported by the device or the sycl compiler");
        }
#endif
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_TRANSPOSE transb_ = cblas_convert(transb);
        float f32_alpha = (float)alpha;
        float f32_beta = (float)beta;
        host_task<class armpl_kernel_hgemm_float_usm>(cgh, [=]() {
            if (beta == 0.0) {
                //C is not relevant as input, we can compute in half and convert output to float
                const float alphaf = (float)alpha;
                const float betaf = (float)beta;
#ifdef COLUMN_MAJOR
                auto f16_c = new __fp16[ldc * n]();
#endif
#ifdef ROW_MAJOR
                auto f16_c = new __fp16[ldc * m]();
#endif
                __fp16 cp = 0.0f;
                copy_mat(c, MAJOR, transpose::N, m, n, ldc, cp, f16_c);
                ::cblas_hgemm(MAJOR, transa_, transb_, m, n, k, alphaf, (const __fp16*)a, lda,
                              (const __fp16*)b, ldb, betaf, f16_c, ldc);
                float co = 0.0f;
                copy_mat(f16_c, MAJOR, m, n, ldc, offset::F, &co, c);
                delete[] f16_c;
            }
            else {
                // copy A, B to float
                int64_t sizea, sizeb, sizec;
#ifdef COLUMN_MAJOR
                sizea = (transa == transpose::N) ? lda * k : lda * m;
                sizeb = (transb == transpose::N) ? ldb * n : ldb * k;
#endif
#ifdef ROW_MAJOR
                sizea = (transa == transpose::N) ? lda * m : lda * k;
                sizeb = (transb == transpose::N) ? ldb * k : ldb * n;
#endif
                auto f32_a = new float[sizea]();
                auto f32_b = new float[sizeb]();
                copy_mat(a, MAJOR, transa, m, k, lda, 0.0f, f32_a);
                copy_mat(b, MAJOR, transb, k, n, ldb, 0.0f, f32_b);
                ::cblas_sgemm(MAJOR, transa_, transb_, m, n, k, f32_alpha, f32_a, lda, f32_b, ldb,
                              f32_beta, c, ldc);
                delete[] f32_a;
                delete[] f32_b;
            }
        });
    });
    return done;
}

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const bfloat16* a, int64_t lda, const bfloat16* b,
                 int64_t ldb, float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", __func__, MAJOR_MINOR_TEXT);
}

template <typename T, typename CBLAS_FUNC>
sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 T alpha, const T* a, int64_t lda, const T* b, int64_t ldb, T beta, T* c,
                 int64_t ldc, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hemm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, m, n, cast_to_void_if_complex(alpha), a,
                       lda, b, ldb, cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define HEMM_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     std::complex<TYPE> alpha, const std::complex<TYPE>* a, int64_t lda,          \
                     const std::complex<TYPE>* b, int64_t ldb, std::complex<TYPE> beta,           \
                     std::complex<TYPE>* c, int64_t ldc,                                          \
                     const std::vector<sycl::event>& dependencies) {                              \
        return hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,    \
                    dependencies, ROUTINE);                                                       \
    }

HEMM_USM_LAUNCHER(float, ::cblas_chemm)
HEMM_USM_LAUNCHER(double, ::cblas_zhemm)

template <typename T, typename CBLAS_FUNC>
sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 T alpha, const std::complex<T>* a, int64_t lda, T beta, std::complex<T>* c,
                 int64_t ldc, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_herk>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha), a, lda,
                       cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define HERK_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                     TYPE alpha, const std::complex<TYPE>* a, int64_t lda, TYPE beta,             \
                     std::complex<TYPE>* c, int64_t ldc,                                          \
                     const std::vector<sycl::event>& dependencies) {                              \
        return herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies,   \
                    ROUTINE);                                                                     \
    }

HERK_USM_LAUNCHER(float, ::cblas_cherk)
HERK_USM_LAUNCHER(double, ::cblas_zherk)

template <typename T, typename CBLAS_FUNC>
sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  std::complex<T> alpha, const std::complex<T>* a, int64_t lda,
                  const std::complex<T>* b, int64_t ldb, T beta, std::complex<T>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_her2k>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha), a, lda, b,
                       ldb, beta, c, ldc);
        });
    });
    return done;
}

#define HER2K_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      std::complex<TYPE> alpha, const std::complex<TYPE>* a, int64_t lda,          \
                      const std::complex<TYPE>* b, int64_t ldb, TYPE beta, std::complex<TYPE>* c,  \
                      int64_t ldc, const std::vector<sycl::event>& dependencies) {                 \
        return her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,         \
                     dependencies, ROUTINE);                                                       \
    }

HER2K_USM_LAUNCHER(float, ::cblas_cher2k)
HER2K_USM_LAUNCHER(double, ::cblas_zher2k)

template <typename T, typename CBLAS_FUNC>
sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 T alpha, const T* a, int64_t lda, const T* b, int64_t ldb, T beta, T* c,
                 int64_t ldc, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_symm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, m, n, cast_to_void_if_complex(alpha), a,
                       lda, b, ldb, cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define SYMM_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, \
                     TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                     TYPE beta, TYPE* c, int64_t ldc,                                             \
                     const std::vector<sycl::event>& dependencies) {                              \
        return symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,    \
                    dependencies, ROUTINE);                                                       \
    }

SYMM_USM_LAUNCHER(float, ::cblas_ssymm)
SYMM_USM_LAUNCHER(double, ::cblas_dsymm)
SYMM_USM_LAUNCHER(std::complex<float>, ::cblas_csymm)
SYMM_USM_LAUNCHER(std::complex<double>, ::cblas_zsymm)

template <typename T, typename CBLAS_FUNC>
sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 T alpha, const T* a, int64_t lda, T beta, T* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_syrk>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha), a, lda,
                       cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define SYRK_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                     TYPE alpha, const TYPE* a, int64_t lda, TYPE beta, TYPE* c, int64_t ldc,     \
                     const std::vector<sycl::event>& dependencies) {                              \
        return syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies,   \
                    ROUTINE);                                                                     \
    }

SYRK_USM_LAUNCHER(float, ::cblas_ssyrk)
SYRK_USM_LAUNCHER(double, ::cblas_dsyrk)
SYRK_USM_LAUNCHER(std::complex<float>, ::cblas_csyrk)
SYRK_USM_LAUNCHER(std::complex<double>, ::cblas_zsyrk)

template <typename T, typename CBLAS_FUNC>
sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  T alpha, const T* a, int64_t lda, const T* b, int64_t ldb, T beta, T* c,
                  int64_t ldc, const std::vector<sycl::event>& dependencies,
                  CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_syr2k>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, n, k, cast_to_void_if_complex(alpha), a, lda, b,
                       ldb, cast_to_void_if_complex(beta), c, ldc);
        });
    });
    return done;
}

#define SYR2K_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, \
                      TYPE alpha, const TYPE* a, int64_t lda, const TYPE* b, int64_t ldb,          \
                      TYPE beta, TYPE* c, int64_t ldc,                                             \
                      const std::vector<sycl::event>& dependencies) {                              \
        return syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,         \
                     dependencies, ROUTINE);                                                       \
    }

SYR2K_USM_LAUNCHER(float, ::cblas_ssyr2k)
SYR2K_USM_LAUNCHER(double, ::cblas_dsyr2k)
SYR2K_USM_LAUNCHER(std::complex<float>, ::cblas_csyr2k)
SYR2K_USM_LAUNCHER(std::complex<double>, ::cblas_zsyr2k)

template <typename T, typename CBLAS_FUNC>
sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,
                 diag unit_diag, int64_t m, int64_t n, T alpha, const T* a, int64_t lda, T* b,
                 int64_t ldb, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trmm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                       cast_to_void_if_complex(alpha), a, lda, b, ldb);
        });
    });
    return done;
}

#define TRMM_USM_LAUNCHER(TYPE, ROUTINE)                                                           \
    sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,      \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trmm(queue, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b,     \
                    ldb, dependencies, ROUTINE);                                                   \
    }

TRMM_USM_LAUNCHER(float, ::cblas_strmm)
TRMM_USM_LAUNCHER(double, ::cblas_dtrmm)
TRMM_USM_LAUNCHER(std::complex<float>, ::cblas_ctrmm)
TRMM_USM_LAUNCHER(std::complex<double>, ::cblas_ztrmm)

template <typename T, typename CBLAS_FUNC>
sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,
                 diag unit_diag, int64_t m, int64_t n, T alpha, const T* a, int64_t lda, T* b,
                 int64_t ldb, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_SIDE left_right_ = cblas_convert(left_right);
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trsm>(cgh, [=]() {
            cblas_func(MAJOR, left_right_, upper_lower_, transa_, unit_diag_, m, n,
                       cast_to_void_if_complex(alpha), a, lda, b, ldb);
        });
    });
    return done;
}

#define TRSM_USM_LAUNCHER(TYPE, ROUTINE)                                                           \
    sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose transa,      \
                     diag unit_diag, int64_t m, int64_t n, TYPE alpha, const TYPE* a, int64_t lda, \
                     TYPE* b, int64_t ldb, const std::vector<sycl::event>& dependencies) {         \
        return trsm(queue, left_right, upper_lower, transa, unit_diag, m, n, alpha, a, lda, b,     \
                    ldb, dependencies, ROUTINE);                                                   \
    }

TRSM_USM_LAUNCHER(float, ::cblas_strsm)
TRSM_USM_LAUNCHER(double, ::cblas_dtrsm)
TRSM_USM_LAUNCHER(std::complex<float>, ::cblas_ctrsm)
TRSM_USM_LAUNCHER(std::complex<double>, ::cblas_ztrsm)
