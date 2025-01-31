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
void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          T alpha, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx, T beta,
          sycl::buffer<T, 1>& y, int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gbmv>(cgh, [=]() {
            cblas_func(MAJOR, trans_, m, n, kl, ku, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define GBMV_LAUNCHER(TYPE, ROUTINE)                                                             \
    void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku, \
              TYPE alpha, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,       \
              int64_t incx, TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                 \
        gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);        \
    }

GBMV_LAUNCHER(float, ::cblas_sgbmv)
GBMV_LAUNCHER(double, ::cblas_dgbmv)
GBMV_LAUNCHER(std::complex<float>, ::cblas_cgbmv)
GBMV_LAUNCHER(std::complex<double>, ::cblas_zgbmv)

template <typename T, typename CBLAS_FUNC>
void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& a,
          int64_t lda, sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y,
          int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gemv>(cgh, [=]() {
            cblas_func(MAJOR, trans_, m, n, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define GEMV_LAUNCHER(TYPE, ROUTINE)                                                         \
    void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,         \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                           \
        gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);            \
    }

GEMV_LAUNCHER(float, ::cblas_sgemv)
GEMV_LAUNCHER(double, ::cblas_dgemv)
GEMV_LAUNCHER(std::complex<float>, ::cblas_cgemv)
GEMV_LAUNCHER(std::complex<double>, ::cblas_zgemv)

template <typename T, typename CBLAS_FUNC>
void ger(sycl::queue& queue, int64_t m, int64_t n, T alpha, sycl::buffer<T, 1>& x, int64_t incx,
         sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& a, int64_t lda,
         CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_ger>(cgh, [=]() {
            cblas_func(MAJOR, m, n, alpha, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR,
                       incy, accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define GER_LAUNCHER(TYPE, ROUTINE)                                                          \
    void ger(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, \
             int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& a, \
             int64_t lda) {                                                                  \
        ger(queue, m, n, alpha, x, incx, y, incy, a, lda, ROUTINE);                          \
    }

GER_LAUNCHER(float, ::cblas_sger)
GER_LAUNCHER(double, ::cblas_dger)

template <typename T, typename CBLAS_FUNC>
void gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<T> alpha,
          sycl::buffer<std::complex<T>, 1>& x, int64_t incx, sycl::buffer<std::complex<T>, 1>& y,
          int64_t incy, sycl::buffer<std::complex<T>, 1>& a, int64_t lda, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_gerc>(cgh, [=]() {
            cblas_func(MAJOR, m, n, cast_to_void_if_complex(alpha), accessor_x.GET_MULTI_PTR, incx,
                       accessor_y.GET_MULTI_PTR, incy, accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define GERC_LAUNCHER(TYPE, ROUTINE)                                                          \
    void gerc(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, \
              int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& a, \
              int64_t lda) {                                                                  \
        gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, ROUTINE);                          \
    }

GERC_LAUNCHER(std::complex<float>, ::cblas_cgerc)
GERC_LAUNCHER(std::complex<double>, ::cblas_zgerc)

#define GERU_LAUNCHER(TYPE, ROUTINE)                                                          \
    void geru(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, \
              int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& a, \
              int64_t lda) {                                                                  \
        gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, ROUTINE);                          \
    }

GERU_LAUNCHER(std::complex<float>, ::cblas_cgeru)
GERU_LAUNCHER(std::complex<double>, ::cblas_zgeru)

template <typename T, typename CBLAS_FUNC>
void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, std::complex<T> alpha,
          sycl::buffer<std::complex<T>, 1>& a, int64_t lda, sycl::buffer<std::complex<T>, 1>& x,
          int64_t incx, std::complex<T> beta, sycl::buffer<std::complex<T>, 1>& y, int64_t incy,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hbmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, k, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define HBMV_LAUNCHER(TYPE, ROUTINE)                                                         \
    void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,        \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                           \
        hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);      \
    }

HBMV_LAUNCHER(std::complex<float>, ::cblas_chbmv)
HBMV_LAUNCHER(std::complex<double>, ::cblas_zhbmv)

template <typename T, typename CBLAS_FUNC>
void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<T> alpha,
          sycl::buffer<std::complex<T>, 1>& a, int64_t lda, sycl::buffer<std::complex<T>, 1>& x,
          int64_t incx, std::complex<T> beta, sycl::buffer<std::complex<T>, 1>& y, int64_t incy,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hemv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha),
                       accessor_a.GET_MULTI_PTR, lda, accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define HEMV_LAUNCHER(TYPE, ROUTINE)                                                         \
    void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                           \
        hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);         \
    }

HEMV_LAUNCHER(std::complex<float>, ::cblas_chemv)
HEMV_LAUNCHER(std::complex<double>, ::cblas_zhemv)

template <typename T, typename CBLAS_FUNC>
void her(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha,
         sycl::buffer<std::complex<T>, 1>& x, int64_t incx, sycl::buffer<std::complex<T>, 1>& a,
         int64_t lda, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_her>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define HER_LAUNCHER(TYPE, ROUTINE)                                       \
    void her(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, \
             sycl::buffer<std::complex<TYPE>, 1>& x, int64_t incx,        \
             sycl::buffer<std::complex<TYPE>, 1>& a, int64_t lda) {       \
        her(queue, upper_lower, n, alpha, x, incx, a, lda, ROUTINE);      \
    }

HER_LAUNCHER(float, ::cblas_cher)
HER_LAUNCHER(double, ::cblas_zher)

template <typename T, typename CBLAS_FUNC>
void her2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
          int64_t incx, sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& a, int64_t lda,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_her2>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha),
                       accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy,
                       accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define HER2_LAUNCHER(TYPE, ROUTINE)                                                          \
    void her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                        \
        her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, ROUTINE);                \
    }

HER2_LAUNCHER(std::complex<float>, ::cblas_cher2)
HER2_LAUNCHER(std::complex<double>, ::cblas_zher2)

template <typename T, typename CBLAS_FUNC>
void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& ap,
          sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hpmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha),
                       accessor_ap.GET_MULTI_PTR, accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define HPMV_LAUNCHER(TYPE, ROUTINE)                                                        \
    void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                  \
              sycl::buffer<TYPE, 1>& ap, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                     \
        hpmv(queue, upper_lower, n, alpha, ap, x, incx, beta, y, incy, ROUTINE);            \
    }

HPMV_LAUNCHER(std::complex<float>, ::cblas_chpmv)
HPMV_LAUNCHER(std::complex<double>, ::cblas_zhpmv)

template <typename T, typename CBLAS_FUNC>
void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha,
         sycl::buffer<std::complex<T>, 1>& x, int64_t incx, sycl::buffer<std::complex<T>, 1>& ap,
         CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hpr>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_ap.GET_MULTI_PTR);
        });
    });
}

#define HPR_LAUNCHER(TYPE, ROUTINE)                                       \
    void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, \
             sycl::buffer<std::complex<TYPE>, 1>& x, int64_t incx,        \
             sycl::buffer<std::complex<TYPE>, 1>& ap) {                   \
        hpr(queue, upper_lower, n, alpha, x, incx, ap, ROUTINE);          \
    }

HPR_LAUNCHER(float, ::cblas_chpr)
HPR_LAUNCHER(double, ::cblas_zhpr)

template <typename T, typename CBLAS_FUNC>
void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
          int64_t incx, sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& ap,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_hpr2>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha),
                       accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy,
                       accessor_ap.GET_MULTI_PTR);
        });
    });
}

#define HPR2_LAUNCHER(TYPE, ROUTINE)                                                          \
    void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, \
              sycl::buffer<TYPE, 1>& ap) {                                                    \
        hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, ap, ROUTINE);                    \
    }

HPR2_LAUNCHER(std::complex<float>, ::cblas_chpr2)
HPR2_LAUNCHER(std::complex<double>, ::cblas_zhpr2)

template <typename T, typename CBLAS_FUNC>
void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, T alpha,
          sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx, T beta,
          sycl::buffer<T, 1>& y, int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_sbmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, k, alpha, accessor_a.GET_MULTI_PTR, lda,
                       accessor_x.GET_MULTI_PTR, incx, beta, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define SBMV_LAUNCHER(TYPE, ROUTINE)                                                         \
    void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,        \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                           \
        sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);      \
    }

SBMV_LAUNCHER(float, ::cblas_ssbmv)
SBMV_LAUNCHER(double, ::cblas_dsbmv)

template <typename T, typename CBLAS_FUNC>
void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& ap,
          sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y, int64_t incy,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_spmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_ap.GET_MULTI_PTR,
                       accessor_x.GET_MULTI_PTR, incx, beta, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define SPMV_LAUNCHER(TYPE, ROUTINE)                                                        \
    void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                  \
              sycl::buffer<TYPE, 1>& ap, sycl::buffer<TYPE, 1>& x, int64_t incx, TYPE beta, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                     \
        spmv(queue, upper_lower, n, alpha, ap, x, incx, beta, y, incy, ROUTINE);            \
    }

SPMV_LAUNCHER(float, ::cblas_sspmv)
SPMV_LAUNCHER(double, ::cblas_dspmv)

template <typename T, typename CBLAS_FUNC>
void spr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
         int64_t incx, sycl::buffer<T, 1>& ap, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_spr>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_ap.GET_MULTI_PTR);
        });
    });
}

#define SPR_LAUNCHER(TYPE, ROUTINE)                                               \
    void spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,         \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& ap) { \
        spr(queue, upper_lower, n, alpha, x, incx, ap, ROUTINE);                  \
    }

SPR_LAUNCHER(float, ::cblas_sspr)
SPR_LAUNCHER(double, ::cblas_dspr)

template <typename T, typename CBLAS_FUNC>
void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
          int64_t incx, sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& ap,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_spr2>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_y.GET_MULTI_PTR, incy, accessor_ap.GET_MULTI_PTR);
        });
    });
}

#define SPR2_LAUNCHER(TYPE, ROUTINE)                                                          \
    void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, \
              sycl::buffer<TYPE, 1>& ap) {                                                    \
        spr2(queue, upper_lower, n, alpha, x, incx, y, incy, ap, ROUTINE);                    \
    }

SPR2_LAUNCHER(float, ::cblas_sspr2)
SPR2_LAUNCHER(double, ::cblas_dspr2)

template <typename T, typename CBLAS_FUNC>
void symv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& a,
          int64_t lda, sycl::buffer<T, 1>& x, int64_t incx, T beta, sycl::buffer<T, 1>& y,
          int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_symv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_a.GET_MULTI_PTR, lda,
                       accessor_x.GET_MULTI_PTR, incx, beta, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define SYMV_LAUNCHER(TYPE, ROUTINE)                                                         \
    void symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                           \
        symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, ROUTINE);         \
    }

SYMV_LAUNCHER(float, ::cblas_ssymv)
SYMV_LAUNCHER(double, ::cblas_dsymv)

template <typename T, typename CBLAS_FUNC>
void syr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
         int64_t incx, sycl::buffer<T, 1>& a, int64_t lda, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_syr>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define SYR_LAUNCHER(TYPE, ROUTINE)                                                           \
    void syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                     \
             sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& a, int64_t lda) { \
        syr(queue, upper_lower, n, alpha, x, incx, a, lda, ROUTINE);                          \
    }

SYR_LAUNCHER(float, ::cblas_ssyr)
SYR_LAUNCHER(double, ::cblas_dsyr)

template <typename T, typename CBLAS_FUNC>
void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, sycl::buffer<T, 1>& x,
          int64_t incx, sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& a, int64_t lda,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_syr2>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, accessor_x.GET_MULTI_PTR, incx,
                       accessor_y.GET_MULTI_PTR, incy, accessor_a.GET_MULTI_PTR, lda);
        });
    });
}

#define SYR2_LAUNCHER(TYPE, ROUTINE)                                                          \
    void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                    \
              sycl::buffer<TYPE, 1>& x, int64_t incx, sycl::buffer<TYPE, 1>& y, int64_t incy, \
              sycl::buffer<TYPE, 1>& a, int64_t lda) {                                        \
        syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, ROUTINE);                \
    }

SYR2_LAUNCHER(float, ::cblas_ssyr2)
SYR2_LAUNCHER(double, ::cblas_dsyr2)

template <typename T, typename CBLAS_FUNC>
void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_tbmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, k, accessor_a.GET_MULTI_PTR, lda,
                       accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define TBMV_LAUNCHER(TYPE, ROUTINE)                                                            \
    void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,       \
              int64_t incx) {                                                                   \
        tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, ROUTINE);             \
    }

TBMV_LAUNCHER(float, ::cblas_stbmv)
TBMV_LAUNCHER(double, ::cblas_dtbmv)
TBMV_LAUNCHER(std::complex<float>, ::cblas_ctbmv)
TBMV_LAUNCHER(std::complex<double>, ::cblas_ztbmv)

template <typename T, typename CBLAS_FUNC>
void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_tbsv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, k, accessor_a.GET_MULTI_PTR, lda,
                       accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define TBSV_LAUNCHER(TYPE, ROUTINE)                                                            \
    void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              int64_t k, sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x,       \
              int64_t incx) {                                                                   \
        tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, ROUTINE);             \
    }

TBSV_LAUNCHER(float, ::cblas_stbsv)
TBSV_LAUNCHER(double, ::cblas_dtbsv)
TBSV_LAUNCHER(std::complex<float>, ::cblas_ctbsv)
TBSV_LAUNCHER(std::complex<double>, ::cblas_ztbsv)

template <typename T, typename CBLAS_FUNC>
void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<T, 1>& ap, sycl::buffer<T, 1>& x, int64_t incx, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_tpmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, accessor_ap.GET_MULTI_PTR,
                       accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define TPMV_LAUNCHER(TYPE, ROUTINE)                                                            \
    void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& ap, sycl::buffer<TYPE, 1>& x, int64_t incx) {              \
        tpmv(queue, upper_lower, trans, unit_diag, n, ap, x, incx, ROUTINE);                    \
    }

TPMV_LAUNCHER(float, ::cblas_stpmv)
TPMV_LAUNCHER(double, ::cblas_dtpmv)
TPMV_LAUNCHER(std::complex<float>, ::cblas_ctpmv)
TPMV_LAUNCHER(std::complex<double>, ::cblas_ztpmv)

template <typename T, typename CBLAS_FUNC>
void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<T, 1>& ap, sycl::buffer<T, 1>& x, int64_t incx, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_ap = ap.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_tpsv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, accessor_ap.GET_MULTI_PTR,
                       accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define TPSV_LAUNCHER(TYPE, ROUTINE)                                                            \
    void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& ap, sycl::buffer<TYPE, 1>& x, int64_t incx) {              \
        tpsv(queue, upper_lower, trans, unit_diag, n, ap, x, incx, ROUTINE);                    \
    }

TPSV_LAUNCHER(float, ::cblas_stpsv)
TPSV_LAUNCHER(double, ::cblas_dtpsv)
TPSV_LAUNCHER(std::complex<float>, ::cblas_ctpsv)
TPSV_LAUNCHER(std::complex<double>, ::cblas_ztpsv)

template <typename T, typename CBLAS_FUNC>
void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& b, int64_t incx,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_trmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, transa_, unit_diag_, n, accessor_a.GET_MULTI_PTR, lda,
                       accessor_b.GET_MULTI_PTR, incx);
        });
    });
}

#define TRMV_LAUNCHER(TYPE, ROUTINE)                                                             \
    void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& b, int64_t incx) {   \
        trmv(queue, upper_lower, transa, unit_diag, n, a, lda, b, incx, ROUTINE);                \
    }

TRMV_LAUNCHER(float, ::cblas_strmv)
TRMV_LAUNCHER(double, ::cblas_dtrmv)
TRMV_LAUNCHER(std::complex<float>, ::cblas_ctrmv)
TRMV_LAUNCHER(std::complex<double>, ::cblas_ztrmv)

template <typename T, typename CBLAS_FUNC>
void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<T, 1>& a, int64_t lda, sycl::buffer<T, 1>& x, int64_t incx,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        auto accessor_a = a.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_trsv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, accessor_a.GET_MULTI_PTR, lda,
                       accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define TRSV_LAUNCHER(TYPE, ROUTINE)                                                            \
    void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n, \
              sycl::buffer<TYPE, 1>& a, int64_t lda, sycl::buffer<TYPE, 1>& x, int64_t incx) {  \
        trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, ROUTINE);                \
    }

TRSV_LAUNCHER(float, ::cblas_strsv)
TRSV_LAUNCHER(double, ::cblas_dtrsv)
TRSV_LAUNCHER(std::complex<float>, ::cblas_ctrsv)
TRSV_LAUNCHER(std::complex<double>, ::cblas_ztrsv)

// USM APIs

template <typename T, typename CBLAS_FUNC>
sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 T alpha, const T* a, int64_t lda, const T* x, int64_t incx, T beta, T* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_gbmv_usm>(cgh, [=]() {
            cblas_func(MAJOR, trans_, m, n, kl, ku, cast_to_void_if_complex(alpha), a, lda, x, incx,
                       cast_to_void_if_complex(beta), y, incy);
        });
    });
    return done;
}

#define GBMV_USM_LAUNCHER(TYPE, ROUTINE)                                                    \
    sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, \
                     int64_t ku, TYPE alpha, const TYPE* a, int64_t lda, const TYPE* x,     \
                     int64_t incx, TYPE beta, TYPE* y, int64_t incy,                        \
                     const std::vector<sycl::event>& dependencies) {                        \
        return gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,      \
                    dependencies, ROUTINE);                                                 \
    }

GBMV_USM_LAUNCHER(float, ::cblas_sgbmv)
GBMV_USM_LAUNCHER(double, ::cblas_dgbmv)
GBMV_USM_LAUNCHER(std::complex<float>, ::cblas_cgbmv)
GBMV_USM_LAUNCHER(std::complex<double>, ::cblas_zgbmv)

template <typename T, typename CBLAS_FUNC>
sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, T alpha, const T* a,
                 int64_t lda, const T* x, int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        host_task<class armpl_kernel_gemv_usm>(cgh, [=]() {
            cblas_func(MAJOR, trans_, m, n, cast_to_void_if_complex(alpha), a, lda, x, incx,
                       cast_to_void_if_complex(beta), y, incy);
        });
    });
    return done;
}

#define GEMV_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, TYPE alpha,       \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                \
        return gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies,      \
                    ROUTINE);                                                                     \
    }

GEMV_USM_LAUNCHER(float, ::cblas_sgemv)
GEMV_USM_LAUNCHER(double, ::cblas_dgemv)
GEMV_USM_LAUNCHER(std::complex<float>, ::cblas_cgemv)
GEMV_USM_LAUNCHER(std::complex<double>, ::cblas_zgemv)

template <typename T, typename CBLAS_FUNC>
sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, T alpha, const T* x, int64_t incx,
                const T* y, int64_t incy, T* a, int64_t lda,
                const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_ger_usm>(cgh, [=]() {
            cblas_func(MAJOR, m, n, cast_to_void_if_complex(alpha), x, incx, y, incy, a, lda);
        });
    });
    return done;
}

#define GER_USM_LAUNCHER(TYPE, ROUTINE)                                                  \
    sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, const TYPE* x, \
                    int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,     \
                    const std::vector<sycl::event>& dependencies) {                      \
        return ger(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies, ROUTINE); \
    }

GER_USM_LAUNCHER(float, ::cblas_sger)
GER_USM_LAUNCHER(double, ::cblas_dger)

template <typename T, typename CBLAS_FUNC>
sycl::event gerc_geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<T> alpha,
                      const std::complex<T>* x, int64_t incx, const std::complex<T>* y,
                      int64_t incy, std::complex<T>* a, int64_t lda,
                      const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_gerc_geru_usm>(cgh, [=]() {
            cblas_func(MAJOR, m, n, cast_to_void_if_complex(alpha), x, incx, y, incy, a, lda);
        });
    });
    return done;
}

#define GERC_USM_LAUNCHER(TYPE, ROUTINE)                                                       \
    sycl::event gerc(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, const TYPE* x,      \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,          \
                     const std::vector<sycl::event>& dependencies) {                           \
        return gerc_geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies, ROUTINE); \
    }

GERC_USM_LAUNCHER(std::complex<float>, ::cblas_cgerc)
GERC_USM_LAUNCHER(std::complex<double>, ::cblas_zgerc)

#define GERU_USM_LAUNCHER(TYPE, ROUTINE)                                                       \
    sycl::event geru(sycl::queue& queue, int64_t m, int64_t n, TYPE alpha, const TYPE* x,      \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,          \
                     const std::vector<sycl::event>& dependencies) {                           \
        return gerc_geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies, ROUTINE); \
    }

GERU_USM_LAUNCHER(std::complex<float>, ::cblas_cgeru)
GERU_USM_LAUNCHER(std::complex<double>, ::cblas_zgeru)

template <typename T, typename CBLAS_FUNC>
sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, T alpha, const T* a,
                 int64_t lda, const T* x, int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hbmv_usm>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, k, cast_to_void_if_complex(alpha), a, lda, x, incx,
                       cast_to_void_if_complex(beta), y, incy);
        });
    });
    return done;
}

#define HBMV_USM_LAUNCHER(TYPE, ROUTINE)                                                           \
    sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,       \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y,  \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                 \
        return hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy, dependencies, \
                    ROUTINE);                                                                      \
    }

HBMV_USM_LAUNCHER(std::complex<float>, ::cblas_chbmv)
HBMV_USM_LAUNCHER(std::complex<double>, ::cblas_zhbmv)

template <typename T, typename CBLAS_FUNC>
sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* a, int64_t lda,
                 const T* x, int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hemv_usm>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha), a, lda, x, incx,
                       cast_to_void_if_complex(beta), y, incy);
        });
    });
    return done;
}

#define HEMV_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies,  \
                    ROUTINE);                                                                    \
    }

HEMV_USM_LAUNCHER(std::complex<float>, ::cblas_chemv)
HEMV_USM_LAUNCHER(std::complex<double>, ::cblas_zhemv)

template <typename T, typename CBLAS_FUNC>
sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const std::complex<T>* x,
                int64_t incx, std::complex<T>* a, int64_t lda,
                const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_her_usm>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

#define HER_USM_LAUNCHER(TYPE, ROUTINE)                                                            \
    sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,                   \
                    const std::complex<TYPE>* x, int64_t incx, std::complex<TYPE>* a, int64_t lda, \
                    const std::vector<sycl::event>& dependencies) {                                \
        return her(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies, ROUTINE);          \
    }

HER_USM_LAUNCHER(float, ::cblas_cher)
HER_USM_LAUNCHER(double, ::cblas_zher)

template <typename T, typename CBLAS_FUNC>
sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                 const T* y, int64_t incy, T* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_her2_usm>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha), x, incx, y, incy, a,
                       lda);
        });
    });
    return done;
}

#define HER2_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies,        \
                    ROUTINE);                                                                    \
    }

HER2_USM_LAUNCHER(std::complex<float>, ::cblas_cher2)
HER2_USM_LAUNCHER(std::complex<double>, ::cblas_zher2)

template <typename T, typename CBLAS_FUNC>
sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* ap, const T* x,
                 int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hpmv_usm>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha), ap, x, incx,
                       cast_to_void_if_complex(beta), y, incy);
        });
    });
    return done;
}

#define HPMV_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* ap, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,               \
                     const std::vector<sycl::event>& dependencies) {                              \
        return hpmv(queue, upper_lower, n, alpha, ap, x, incx, beta, y, incy, dependencies,       \
                    ROUTINE);                                                                     \
    }

HPMV_USM_LAUNCHER(std::complex<float>, ::cblas_chpmv)
HPMV_USM_LAUNCHER(std::complex<double>, ::cblas_zhpmv)

template <typename T, typename CBLAS_FUNC>
sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const std::complex<T>* x,
                int64_t incx, std::complex<T>* ap, const std::vector<sycl::event>& dependencies,
                CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hpr_usm>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

#define HPR_USM_LAUNCHER(TYPE, ROUTINE)                                                \
    sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha,       \
                    const std::complex<TYPE>* x, int64_t incx, std::complex<TYPE>* ap, \
                    const std::vector<sycl::event>& dependencies) {                    \
        return hpr(queue, upper_lower, n, alpha, x, incx, ap, dependencies, ROUTINE);  \
    }

HPR_USM_LAUNCHER(float, ::cblas_chpr)
HPR_USM_LAUNCHER(double, ::cblas_zhpr)

template <typename T, typename CBLAS_FUNC>
sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                 const T* y, int64_t incy, T* ap, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_hpr2>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, cast_to_void_if_complex(alpha), x, incx, y, incy,
                       ap);
        });
    });
    return done;
}

#define HPR2_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* ap,                        \
                     const std::vector<sycl::event>& dependencies) {                             \
        return hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, ap, dependencies, ROUTINE);  \
    }

HPR2_USM_LAUNCHER(std::complex<float>, ::cblas_chpr2)
HPR2_USM_LAUNCHER(std::complex<double>, ::cblas_zhpr2)

template <typename T, typename CBLAS_FUNC>
sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, T alpha, const T* a,
                 int64_t lda, const T* x, int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_sbmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, k, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

#define SBMV_USM_LAUNCHER(TYPE, ROUTINE)                                                           \
    sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, TYPE alpha,       \
                     const TYPE* a, int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y,  \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {                 \
        return sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy, dependencies, \
                    ROUTINE);                                                                      \
    }

SBMV_USM_LAUNCHER(float, ::cblas_ssbmv)
SBMV_USM_LAUNCHER(double, ::cblas_dsbmv)

template <typename T, typename CBLAS_FUNC>
sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* ap, const T* x,
                 int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_spmv>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, ap, x, incx, beta, y, incy); });
    });
    return done;
}

#define SPMV_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* ap, \
                     const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy,               \
                     const std::vector<sycl::event>& dependencies) {                              \
        return spmv(queue, upper_lower, n, alpha, ap, x, incx, beta, y, incy, dependencies,       \
                    ROUTINE);                                                                     \
    }

SPMV_USM_LAUNCHER(float, ::cblas_sspmv)
SPMV_USM_LAUNCHER(double, ::cblas_dspmv)

template <typename T, typename CBLAS_FUNC>
sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                T* ap, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_spr>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, ap); });
    });
    return done;
}

#define SPR_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                    int64_t incx, TYPE* ap, const std::vector<sycl::event>& dependencies) {     \
        return spr(queue, upper_lower, n, alpha, x, incx, ap, dependencies, ROUTINE);           \
    }

SPR_USM_LAUNCHER(float, ::cblas_sspr)
SPR_USM_LAUNCHER(double, ::cblas_dspr)

template <typename T, typename CBLAS_FUNC>
sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                 const T* y, int64_t incy, T* ap, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_spr2>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, y, incy, ap); });
    });
    return done;
}

#define SPR2_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* ap,                        \
                     const std::vector<sycl::event>& dependencies) {                             \
        return spr2(queue, upper_lower, n, alpha, x, incx, y, incy, ap, dependencies, ROUTINE);  \
    }

SPR2_USM_LAUNCHER(float, ::cblas_sspr2)
SPR2_USM_LAUNCHER(double, ::cblas_dspr2)

template <typename T, typename CBLAS_FUNC>
sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* a, int64_t lda,
                 const T* x, int64_t incx, T beta, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_symv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, n, alpha, a, lda, x, incx, beta, y, incy);
        });
    });
    return done;
}

#define SYMV_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* a, \
                     int64_t lda, const TYPE* x, int64_t incx, TYPE beta, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                             \
        return symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies,  \
                    ROUTINE);                                                                    \
    }

SYMV_USM_LAUNCHER(float, ::cblas_ssymv)
SYMV_USM_LAUNCHER(double, ::cblas_dsymv)

template <typename T, typename CBLAS_FUNC>
sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                T* a, int64_t lda, const std::vector<sycl::event>& dependencies,
                CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_syr>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, a, lda); });
    });
    return done;
}

#define SYR_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                    int64_t incx, TYPE* a, int64_t lda,                                         \
                    const std::vector<sycl::event>& dependencies) {                             \
        return syr(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies, ROUTINE);       \
    }

SYR_USM_LAUNCHER(float, ::cblas_ssyr)
SYR_USM_LAUNCHER(double, ::cblas_dsyr)

template <typename T, typename CBLAS_FUNC>
sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, T alpha, const T* x, int64_t incx,
                 const T* y, int64_t incy, T* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        host_task<class armpl_kernel_syr2>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, n, alpha, x, incx, y, incy, a, lda); });
    });
    return done;
}

#define SYR2_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, TYPE alpha, const TYPE* x, \
                     int64_t incx, const TYPE* y, int64_t incy, TYPE* a, int64_t lda,            \
                     const std::vector<sycl::event>& dependencies) {                             \
        return syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies,        \
                    ROUTINE);                                                                    \
    }

SYR2_USM_LAUNCHER(float, ::cblas_ssyr2)
SYR2_USM_LAUNCHER(double, ::cblas_dsyr2)

template <typename T, typename CBLAS_FUNC>
sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const T* a, int64_t lda, T* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_tbmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

#define TBMV_USM_LAUNCHER(TYPE, ROUTINE)                                                       \
    sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,  \
                     const std::vector<sycl::event>& dependencies) {                           \
        return tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies, \
                    ROUTINE);                                                                  \
    }

TBMV_USM_LAUNCHER(float, ::cblas_stbmv)
TBMV_USM_LAUNCHER(double, ::cblas_dtbmv)
TBMV_USM_LAUNCHER(std::complex<float>, ::cblas_ctbmv)
TBMV_USM_LAUNCHER(std::complex<double>, ::cblas_ztbmv)

template <typename T, typename CBLAS_FUNC>
sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const T* a, int64_t lda, T* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_tbsv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, k, a, lda, x, incx);
        });
    });
    return done;
}

#define TBSV_USM_LAUNCHER(TYPE, ROUTINE)                                                       \
    sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,    \
                     int64_t n, int64_t k, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,  \
                     const std::vector<sycl::event>& dependencies) {                           \
        return tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies, \
                    ROUTINE);                                                                  \
    }

TBSV_USM_LAUNCHER(float, ::cblas_stbsv)
TBSV_USM_LAUNCHER(double, ::cblas_dtbsv)
TBSV_USM_LAUNCHER(std::complex<float>, ::cblas_ctbsv)
TBSV_USM_LAUNCHER(std::complex<double>, ::cblas_ztbsv)

template <typename T, typename CBLAS_FUNC>
sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const T* ap, T* x, int64_t incx, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_tpmv>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx); });
    });
    return done;
}

#define TPMV_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, const TYPE* ap, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                              \
        return tpmv(queue, upper_lower, trans, unit_diag, n, ap, x, incx, dependencies, ROUTINE); \
    }

TPMV_USM_LAUNCHER(float, ::cblas_stpmv)
TPMV_USM_LAUNCHER(double, ::cblas_dtpmv)
TPMV_USM_LAUNCHER(std::complex<float>, ::cblas_ctpmv)
TPMV_USM_LAUNCHER(std::complex<double>, ::cblas_ztpmv)

template <typename T, typename CBLAS_FUNC>
sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const T* ap, T* x, int64_t incx, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_tpsv>(
            cgh, [=]() { cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, ap, x, incx); });
    });
    return done;
}

#define TPSV_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,       \
                     int64_t n, const TYPE* ap, TYPE* x, int64_t incx,                            \
                     const std::vector<sycl::event>& dependencies) {                              \
        return tpsv(queue, upper_lower, trans, unit_diag, n, ap, x, incx, dependencies, ROUTINE); \
    }

TPSV_USM_LAUNCHER(float, ::cblas_stpsv)
TPSV_USM_LAUNCHER(double, ::cblas_dtpsv)
TPSV_USM_LAUNCHER(std::complex<float>, ::cblas_ctpsv)
TPSV_USM_LAUNCHER(std::complex<double>, ::cblas_ztpsv)

template <typename T, typename CBLAS_FUNC>
sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
                 const T* a, int64_t lda, T* b, int64_t incx,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE transa_ = cblas_convert(transa);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trmv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, transa_, unit_diag_, n, a, lda, b, incx);
        });
    });
    return done;
}

#define TRMV_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* b, int64_t incx,           \
                     const std::vector<sycl::event>& dependencies) {                         \
        return trmv(queue, upper_lower, transa, unit_diag, n, a, lda, b, incx, dependencies, \
                    ROUTINE);                                                                \
    }

TRMV_USM_LAUNCHER(float, ::cblas_strmv)
TRMV_USM_LAUNCHER(double, ::cblas_dtrmv)
TRMV_USM_LAUNCHER(std::complex<float>, ::cblas_ctrmv)
TRMV_USM_LAUNCHER(std::complex<double>, ::cblas_ztrmv)

template <typename T, typename CBLAS_FUNC>
sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const T* a, int64_t lda, T* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        CBLAS_UPLO upper_lower_ = cblas_convert(upper_lower);
        CBLAS_TRANSPOSE trans_ = cblas_convert(trans);
        CBLAS_DIAG unit_diag_ = cblas_convert(unit_diag);
        host_task<class armpl_kernel_trsv>(cgh, [=]() {
            cblas_func(MAJOR, upper_lower_, trans_, unit_diag_, n, a, lda, x, incx);
        });
    });
    return done;
}

#define TRSV_USM_LAUNCHER(TYPE, ROUTINE)                                                    \
    sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, \
                     int64_t n, const TYPE* a, int64_t lda, TYPE* x, int64_t incx,          \
                     const std::vector<sycl::event>& dependencies) {                        \
        return trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies, \
                    ROUTINE);                                                               \
    }

TRSV_USM_LAUNCHER(float, ::cblas_strsv)
TRSV_USM_LAUNCHER(double, ::cblas_dtrsv)
TRSV_USM_LAUNCHER(std::complex<float>, ::cblas_ctrsv)
TRSV_USM_LAUNCHER(std::complex<double>, ::cblas_ztrsv)
