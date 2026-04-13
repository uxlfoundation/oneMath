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

// Buffer APIs

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          float alpha, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx, float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sgbmv>(cgh, [=]() {
            ::cblas_sgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const float)alpha,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const float)beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          double alpha, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx, double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dgbmv>(cgh, [=]() {
            ::cblas_dgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const double)alpha,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const double)beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cgbmv>(cgh, [=]() {
            ::cblas_cgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void*)&alpha,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const void*)&beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zgbmv>(cgh, [=]() {
            ::cblas_zgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void*)&alpha,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const void*)&beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx,
          float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sgemv>(cgh, [=]() {
            ::cblas_sgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const float)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const float)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx,
          double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dgemv>(cgh, [=]() {
            ::cblas_dgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const double)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const double)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cgemv>(cgh, [=]() {
            ::cblas_cgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zgemv>(cgh, [=]() {
            ::cblas_zgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void ger(sycl::queue& queue, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& a,
         int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sger>(cgh, [=]() {
            ::cblas_sger(MAJOR, (const int)m, (const int)n, (const float)alpha,
                         accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                         (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void ger(sycl::queue& queue, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& a,
         int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dger>(cgh, [=]() {
            ::cblas_dger(MAJOR, (const int)m, (const int)n, (const double)alpha,
                         accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                         (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cgerc>(cgh, [=]() {
            ::cblas_cgerc(MAJOR, (const int)m, (const int)n, (const void*)&alpha,
                          accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                          (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zgerc>(cgh, [=]() {
            ::cblas_zgerc(MAJOR, (const int)m, (const int)n, (const void*)&alpha,
                          accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                          (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cgeru>(cgh, [=]() {
            ::cblas_cgeru(MAJOR, (const int)m, (const int)n, (const void*)&alpha,
                          accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                          (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zgeru>(cgh, [=]() {
            ::cblas_zgeru(MAJOR, (const int)m, (const int)n, (const void*)&alpha,
                          accessor_x.GET_MULTI_PTR, (const int)incx, accessor_y.GET_MULTI_PTR,
                          (const int)incy, accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_chbmv>(cgh, [=]() {
            ::cblas_chbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zhbmv>(cgh, [=]() {
            ::cblas_zhbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_chemv>(cgh, [=]() {
            ::cblas_chemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zhemv>(cgh, [=]() {
            ::cblas_zhemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const void*)&beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void her(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cher>(cgh, [=]() {
            ::cblas_cher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void her(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zher>(cgh, [=]() {
            ::cblas_zher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cher2>(cgh, [=]() {
            ::cblas_cher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_a.GET_MULTI_PTR,
                          (const int)lda);
        });
    });
}

void her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zher2>(cgh, [=]() {
            ::cblas_zher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_a.GET_MULTI_PTR,
                          (const int)lda);
        });
    });
}

void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& ap, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
          int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_chpmv>(cgh, [=]() {
            ::cblas_chpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_ap.GET_MULTI_PTR, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const void*)&beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& ap, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx, std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
          int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zhpmv>(cgh, [=]() {
            ::cblas_zhpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_ap.GET_MULTI_PTR, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const void*)&beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_chpr>(cgh, [=]() {
            ::cblas_chpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_ap.GET_MULTI_PTR);
        });
    });
}

void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zhpr>(cgh, [=]() {
            ::cblas_zhpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_ap.GET_MULTI_PTR);
        });
    });
}

void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_chpr2>(cgh, [=]() {
            ::cblas_chpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_ap.GET_MULTI_PTR);
        });
    });
}

void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zhpr2>(cgh, [=]() {
            ::cblas_zhpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_ap.GET_MULTI_PTR);
        });
    });
}

void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx,
          float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ssbmv>(cgh, [=]() {
            ::cblas_ssbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const float)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const float)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx,
          double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dsbmv>(cgh, [=]() {
            ::cblas_dsbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const double)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const double)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& ap,
          sycl::buffer<float, 1>& x, int64_t incx, float beta, sycl::buffer<float, 1>& y,
          int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sspmv>(cgh, [=]() {
            ::cblas_sspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_ap.GET_MULTI_PTR, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const float)beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
          sycl::buffer<double, 1>& ap, sycl::buffer<double, 1>& x, int64_t incx, double beta,
          sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dspmv>(cgh, [=]() {
            ::cblas_dspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_ap.GET_MULTI_PTR, accessor_x.GET_MULTI_PTR,
                          (const int)incx, (const double)beta, accessor_y.GET_MULTI_PTR,
                          (const int)incy);
        });
    });
}

void spr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sspr>(cgh, [=]() {
            ::cblas_sspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_ap.GET_MULTI_PTR);
        });
    });
}

void spr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dspr>(cgh, [=]() {
            ::cblas_dspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_ap.GET_MULTI_PTR);
        });
    });
}

void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
          int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sspr2>(cgh, [=]() {
            ::cblas_sspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_ap.GET_MULTI_PTR);
        });
    });
}

void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
          int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& ap) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_ap = ap.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dspr2>(cgh, [=]() {
            ::cblas_dspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_ap.GET_MULTI_PTR);
        });
    });
}

void symv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& a,
          int64_t lda, sycl::buffer<float, 1>& x, int64_t incx, float beta,
          sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ssymv>(cgh, [=]() {
            ::cblas_ssymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const float)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void symv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& a,
          int64_t lda, sycl::buffer<double, 1>& x, int64_t incx, double beta,
          sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dsymv>(cgh, [=]() {
            ::cblas_dsymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_a.GET_MULTI_PTR, (const int)lda,
                          accessor_x.GET_MULTI_PTR, (const int)incx, (const double)beta,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void syr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ssyr>(cgh, [=]() {
            ::cblas_ssyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void syr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& a, int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dsyr>(cgh, [=]() {
            ::cblas_dsyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_a.GET_MULTI_PTR, (const int)lda);
        });
    });
}

void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
          int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& a,
          int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ssyr2>(cgh, [=]() {
            ::cblas_ssyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_a.GET_MULTI_PTR,
                          (const int)lda);
        });
    });
}

void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
          int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& a,
          int64_t lda) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dsyr2>(cgh, [=]() {
            ::cblas_dsyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_a.GET_MULTI_PTR,
                          (const int)lda);
        });
    });
}

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_stbmv>(cgh, [=]() {
            ::cblas_stbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtbmv>(cgh, [=]() {
            ::cblas_dtbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctbmv>(cgh, [=]() {
            ::cblas_ctbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztbmv>(cgh, [=]() {
            ::cblas_ztbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_stbsv>(cgh, [=]() {
            ::cblas_stbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtbsv>(cgh, [=]() {
            ::cblas_dtbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctbsv>(cgh, [=]() {
            ::cblas_ctbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztbsv>(cgh, [=]() {
            ::cblas_ztbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k,
                          accessor_a.GET_MULTI_PTR, (const int)lda, accessor_x.GET_MULTI_PTR,
                          (const int)incx);
        });
    });
}

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& ap, sycl::buffer<float, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_stpmv>(cgh, [=]() {
            ::cblas_stpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& ap, sycl::buffer<double, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtpmv>(cgh, [=]() {
            ::cblas_dtpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& ap, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctpmv>(cgh, [=]() {
            ::cblas_ctpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& ap, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztpmv>(cgh, [=]() {
            ::cblas_ztpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& ap, sycl::buffer<float, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_stpsv>(cgh, [=]() {
            ::cblas_stpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& ap, sycl::buffer<double, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtpsv>(cgh, [=]() {
            ::cblas_dtpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& ap, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctpsv>(cgh, [=]() {
            ::cblas_ctpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& ap, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_ap = ap.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztpsv>(cgh, [=]() {
            ::cblas_ztpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_ap.GET_MULTI_PTR,
                          accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& b, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_strmv>(cgh, [=]() {
            ::cblas_strmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_b.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& b, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtrmv>(cgh, [=]() {
            ::cblas_dtrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_b.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctrmv>(cgh, [=]() {
            ::cblas_ctrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_b.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztrmv>(cgh, [=]() {
            ::cblas_ztrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_b.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_strsv>(cgh, [=]() {
            ::cblas_strsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dtrsv>(cgh, [=]() {
            ::cblas_dtrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ctrsv>(cgh, [=]() {
            ::cblas_ctrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read>(cgh);
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ztrsv>(cgh, [=]() {
            ::cblas_ztrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, accessor_a.GET_MULTI_PTR,
                          (const int)lda, accessor_x.GET_MULTI_PTR, (const int)incx);
        });
    });
}

// USM APIs

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 float alpha, const float* a, int64_t lda, const float* x, int64_t incx, float beta,
                 float* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sgbmv_usm>(cgh, [=]() {
            ::cblas_sgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const float)alpha, a, (const int)lda, x,
                          (const int)incx, (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 double alpha, const double* a, int64_t lda, const double* x, int64_t incx,
                 double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dgbmv_usm>(cgh, [=]() {
            ::cblas_dgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const double)alpha, a, (const int)lda, x,
                          (const int)incx, (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cgbmv_usm>(cgh, [=]() {
            ::cblas_cgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void*)&alpha, a, (const int)lda, x,
                          (const int)incx, (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zgbmv_usm>(cgh, [=]() {
            ::cblas_zgbmv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const int)kl, (const int)ku, (const void*)&alpha, a, (const int)lda, x,
                          (const int)incx, (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                 const float* a, int64_t lda, const float* x, int64_t incx, float beta, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sgemv_usm>(cgh, [=]() {
            ::cblas_sgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                 const double* a, int64_t lda, const double* x, int64_t incx, double beta,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dgemv_usm>(cgh, [=]() {
            ::cblas_dgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cgemv_usm>(cgh, [=]() {
            ::cblas_cgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zgemv_usm>(cgh, [=]() {
            ::cblas_zgemv(MAJOR, convert_to_cblas_trans(trans), (const int)m, (const int)n,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, float alpha, const float* x, int64_t incx,
                const float* y, int64_t incy, float* a, int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sger_usm>(cgh, [=]() {
            ::cblas_sger(MAJOR, (const int)m, (const int)n, (const float)alpha, x, (const int)incx,
                         y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, double alpha, const double* x,
                int64_t incx, const double* y, int64_t incy, double* a, int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dger_usm>(cgh, [=]() {
            ::cblas_dger(MAJOR, (const int)m, (const int)n, (const double)alpha, x, (const int)incx,
                         y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cgerc_usm>(cgh, [=]() {
            ::cblas_cgerc(MAJOR, (const int)m, (const int)n, (const void*)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zgerc_usm>(cgh, [=]() {
            ::cblas_zgerc(MAJOR, (const int)m, (const int)n, (const void*)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cgeru_usm>(cgh, [=]() {
            ::cblas_cgeru(MAJOR, (const int)m, (const int)n, (const void*)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zgeru_usm>(cgh, [=]() {
            ::cblas_zgeru(MAJOR, (const int)m, (const int)n, (const void*)&alpha, x,
                          (const int)incx, y, (const int)incy, a, (const int)lda);
        });
    });
    return done;
}

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_chbmv_usm>(cgh, [=]() {
            ::cblas_chbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zhbmv_usm>(cgh, [=]() {
            ::cblas_zhbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, int64_t lda, const std::complex<float>* x,
                 int64_t incx, std::complex<float> beta, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_chemv_usm>(cgh, [=]() {
            ::cblas_chemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, int64_t lda, const std::complex<double>* x,
                 int64_t incx, std::complex<double> beta, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zhemv_usm>(cgh, [=]() {
            ::cblas_zhemv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, a, (const int)lda, x, (const int)incx,
                          (const void*)&beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
                const std::complex<float>* x, int64_t incx, std::complex<float>* a, int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cher_usm>(cgh, [=]() {
            ::cblas_cher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
                const std::complex<double>* x, int64_t incx, std::complex<double>* a, int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zher_usm>(cgh, [=]() {
            ::cblas_zher(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cher2_usm>(cgh, [=]() {
            ::cblas_cher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zher2_usm>(cgh, [=]() {
            ::cblas_zher2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* ap, const std::complex<float>* x, int64_t incx,
                 std::complex<float> beta, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_chpmv_usm>(cgh, [=]() {
            ::cblas_chpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, ap, x, (const int)incx, (const void*)&beta, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* ap, const std::complex<double>* x, int64_t incx,
                 std::complex<double> beta, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zhpmv_usm>(cgh, [=]() {
            ::cblas_zhpmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, ap, x, (const int)incx, (const void*)&beta, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
                const std::complex<float>* x, int64_t incx, std::complex<float>* ap,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_chpr_usm>(cgh, [=]() {
            ::cblas_chpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
                const std::complex<double>* x, int64_t incx, std::complex<double>* ap,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zhpr_usm>(cgh, [=]() {
            ::cblas_zhpr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* ap,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_chpr2_usm>(cgh, [=]() {
            ::cblas_chpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* ap,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zhpr2_usm>(cgh, [=]() {
            ::cblas_zhpr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const void*)&alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t lda, const float* x, int64_t incx, float beta, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ssbmv_usm>(cgh, [=]() {
            ::cblas_ssbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
                 const double* a, int64_t lda, const double* x, int64_t incx, double beta,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dsbmv_usm>(cgh, [=]() {
            ::cblas_dsbmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n, (const int)k,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* ap,
                 const float* x, int64_t incx, float beta, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sspmv_usm>(cgh, [=]() {
            ::cblas_sspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, ap, x, (const int)incx, (const float)beta, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* ap,
                 const double* x, int64_t incx, double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dspmv_usm>(cgh, [=]() {
            ::cblas_dspmv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, ap, x, (const int)incx, (const double)beta, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                int64_t incx, float* ap, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sspr_usm>(cgh, [=]() {
            ::cblas_sspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                int64_t incx, double* ap, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dspr_usm>(cgh, [=]() {
            ::cblas_dspr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, ap);
        });
    });
    return done;
}

sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                 int64_t incx, const float* y, int64_t incy, float* ap,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sspr2_usm>(cgh, [=]() {
            ::cblas_sspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                 int64_t incx, const double* y, int64_t incy, double* ap,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dspr2_usm>(cgh, [=]() {
            ::cblas_dspr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, x, (const int)incx, y, (const int)incy, ap);
        });
    });
    return done;
}

sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* a,
                 int64_t lda, const float* x, int64_t incx, float beta, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ssymv_usm>(cgh, [=]() {
            ::cblas_ssymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, a, (const int)lda, x, (const int)incx,
                          (const float)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* a,
                 int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dsymv_usm>(cgh, [=]() {
            ::cblas_dsymv(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, a, (const int)lda, x, (const int)incx,
                          (const double)beta, y, (const int)incy);
        });
    });
    return done;
}

sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                int64_t incx, float* a, int64_t lda, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ssyr_usm>(cgh, [=]() {
            ::cblas_ssyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const float)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                int64_t incx, double* a, int64_t lda,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dsyr_usm>(cgh, [=]() {
            ::cblas_dsyr(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                         (const double)alpha, x, (const int)incx, a, (const int)lda);
        });
    });
    return done;
}

sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                 int64_t incx, const float* y, int64_t incy, float* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ssyr2_usm>(cgh, [=]() {
            ::cblas_ssyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const float)alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                 int64_t incx, const double* y, int64_t incy, double* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dsyr2_usm>(cgh, [=]() {
            ::cblas_dsyr2(MAJOR, convert_to_cblas_uplo(upper_lower), (const int)n,
                          (const double)alpha, x, (const int)incx, y, (const int)incy, a,
                          (const int)lda);
        });
    });
    return done;
}

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_stbmv_usm>(cgh, [=]() {
            ::cblas_stbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtbmv_usm>(cgh, [=]() {
            ::cblas_dtbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<float>* a, int64_t lda, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctbmv_usm>(cgh, [=]() {
            ::cblas_ctbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<double>* a, int64_t lda, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztbmv_usm>(cgh, [=]() {
            ::cblas_ztbmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_stbsv_usm>(cgh, [=]() {
            ::cblas_stbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtbsv_usm>(cgh, [=]() {
            ::cblas_dtbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<float>* a, int64_t lda, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctbsv_usm>(cgh, [=]() {
            ::cblas_ctbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<double>* a, int64_t lda, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztbsv_usm>(cgh, [=]() {
            ::cblas_ztbsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, (const int)k, a,
                          (const int)lda, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* ap, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_stpmv_usm>(cgh, [=]() {
            ::cblas_stpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* ap, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtpmv_usm>(cgh, [=]() {
            ::cblas_dtpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* ap, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctpmv_usm>(cgh, [=]() {
            ::cblas_ctpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* ap, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztpmv_usm>(cgh, [=]() {
            ::cblas_ztpmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* ap, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_stpsv_usm>(cgh, [=]() {
            ::cblas_stpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* ap, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtpsv_usm>(cgh, [=]() {
            ::cblas_dtpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* ap, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctpsv_usm>(cgh, [=]() {
            ::cblas_ctpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* ap, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztpsv_usm>(cgh, [=]() {
            ::cblas_ztpsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, ap, x, (const int)incx);
        });
    });
    return done;
}

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
                 const float* a, int64_t lda, float* b, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_strmv_usm>(cgh, [=]() {
            ::cblas_strmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
                 const double* a, int64_t lda, double* b, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtrmv_usm>(cgh, [=]() {
            ::cblas_dtrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* b, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctrmv_usm>(cgh, [=]() {
            ::cblas_ctrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose transa, diag unit_diag, int64_t n,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* b, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztrmv_usm>(cgh, [=]() {
            ::cblas_ztrmv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(transa),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, b,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_strsv_usm>(cgh, [=]() {
            ::cblas_strsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dtrsv_usm>(cgh, [=]() {
            ::cblas_dtrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ctrsv_usm>(cgh, [=]() {
            ::cblas_ctrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ztrsv_usm>(cgh, [=]() {
            ::cblas_ztrsv(MAJOR, convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans),
                          convert_to_cblas_diag(unit_diag), (const int)n, a, (const int)lda, x,
                          (const int)incx);
        });
    });
    return done;
}
