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

static inline void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& result) {
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
}

static inline void asum(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<double, 1>& result) {
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
}

static inline void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& result) {
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
}

static inline void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                        std::int64_t incx, sycl::buffer<double, 1>& result) {
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
}

static inline void axpy(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(sycl::queue& queue, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                        std::int64_t incy) {
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
}

static inline void axpy_batch(sycl::queue& queue, std::int64_t n, float alpha,
                              sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<float, 1>& y, std::int64_t incy, std::int64_t stridey,
                              std::int64_t batch_size) {
    detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void axpy_batch(sycl::queue& queue, std::int64_t n, double alpha,
                              sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<double, 1>& y, std::int64_t incy, std::int64_t stridey,
                              std::int64_t batch_size) {
    detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void axpby(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
                         std::int64_t incx, float beta, sycl::buffer<float, 1>& y,
                         std::int64_t incy) {
    detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy);
}

static inline void axpby(sycl::queue& queue, std::int64_t n, double alpha,
                         sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                         sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy);
}

static inline void axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                         std::int64_t incy) {
    detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy);
}

static inline void axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                         std::int64_t incy) {
    detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy);
}

static inline void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                        std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy) {
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void copy(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                              std::int64_t incx, std::int64_t stridex, sycl::buffer<float, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void copy_batch(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                              std::int64_t incx, std::int64_t stridex, sycl::buffer<double, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void copy_batch(sycl::queue& queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void copy_batch(sycl::queue& queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy, stridey,
                       batch_size);
}

static inline void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                       std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
                       sycl::buffer<float, 1>& result) {
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                       std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
                       sycl::buffer<double, 1>& result) {
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                       std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
                       sycl::buffer<double, 1>& result) {
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& result) {
    detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dotc(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                        sycl::buffer<std::complex<double>, 1>& result) {
    detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& result) {
    detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void dotu(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                        sycl::buffer<std::complex<double>, 1>& result) {
    detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result);
}

static inline void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, float alpha, sycl::buffer<float, 1>& a,
                        std::int64_t lda, sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                        sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, double alpha, sycl::buffer<double, 1>& a,
                        std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx,
                        double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy) {
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy) {
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                        std::int64_t lda, sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                        sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                        std::int64_t lda, sycl::buffer<double, 1>& b, std::int64_t ldb, double beta,
                        sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                        std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                        std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, sycl::half alpha,
                        sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                        sycl::buffer<sycl::half, 1>& b, std::int64_t ldb, sycl::half beta,
                        sycl::buffer<sycl::half, 1>& c, std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a,
                        std::int64_t lda, sycl::buffer<sycl::half, 1>& b, std::int64_t ldb,
                        float beta, sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, sycl::buffer<bfloat16, 1>& a,
                        std::int64_t lda, sycl::buffer<bfloat16, 1>& b, std::int64_t ldb,
                        float beta, sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<float, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                              float beta, sycl::buffer<float, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<double, 1>& b, std::int64_t ldb, std::int64_t stride_b,
                              double beta, sycl::buffer<double, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::complex<float> beta,
                              sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                              sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                              std::int64_t ldb, std::int64_t stride_b, sycl::half beta,
                              sycl::buffer<sycl::half, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<sycl::half, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<sycl::half, 1>& b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                              std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                              std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<std::int8_t, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<std::int8_t, 1>& b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              sycl::buffer<std::int32_t, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                             std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                             sycl::buffer<int8_t, 1>& a, std::int64_t lda, int8_t ao,
                             sycl::buffer<uint8_t, 1>& b, std::int64_t ldb, uint8_t bo, float beta,
                             sycl::buffer<int32_t, 1>& c, std::int64_t ldc,
                             sycl::buffer<int32_t, 1>& co) {
    detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                      ao, b, ldb, bo, beta, c, ldc, co);
}

static inline void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                             std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                             sycl::buffer<int8_t, 1>& a, std::int64_t lda, int8_t ao,
                             sycl::buffer<int8_t, 1>& b, std::int64_t ldb, int8_t bo, float beta,
                             sycl::buffer<int32_t, 1>& c, std::int64_t ldc,
                             sycl::buffer<int32_t, 1>& co) {
    detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                      ao, b, ldb, bo, beta, c, ldc, co);
}

static inline void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                             std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                             sycl::buffer<uint8_t, 1>& a, std::int64_t lda, uint8_t ao,
                             sycl::buffer<int8_t, 1>& b, std::int64_t ldb, int8_t bo, float beta,
                             sycl::buffer<int32_t, 1>& c, std::int64_t ldc,
                             sycl::buffer<int32_t, 1>& co) {
    detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                      ao, b, ldb, bo, beta, c, ldc, co);
}

static inline void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                             std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                             sycl::buffer<uint8_t, 1>& a, std::int64_t lda, uint8_t ao,
                             sycl::buffer<uint8_t, 1>& b, std::int64_t ldb, uint8_t bo, float beta,
                             sycl::buffer<int32_t, 1>& c, std::int64_t ldc,
                             sycl::buffer<int32_t, 1>& co) {
    detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                      ao, b, ldb, bo, beta, c, ldc, co);
}

static inline void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                         std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                         std::int64_t lda, sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
}

static inline void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                         std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                         std::int64_t lda, sycl::buffer<double, 1>& b, std::int64_t ldb,
                         double beta, sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
}

static inline void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                         std::int64_t n, std::int64_t k, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                         std::int64_t ldc) {
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
}

static inline void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                         std::int64_t n, std::int64_t k, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                         std::int64_t ldc) {
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
}

static inline void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                        sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                        sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                        sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, std::complex<float> beta,
                        sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, std::complex<double> beta,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                              float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<float, 1>& x, std::int64_t incx,
                              std::int64_t stridex, float beta, sycl::buffer<float, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                       stridex, beta, y, incy, stridey, batch_size);
}

static inline void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                              double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<double, 1>& x, std::int64_t incx,
                              std::int64_t stridex, double beta, sycl::buffer<double, 1>& y,
                              std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
    detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                       stridex, beta, y, incy, stridey, batch_size);
}

static inline void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                              std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, std::complex<float> beta,
                              sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
                              std::int64_t stridey, std::int64_t batch_size) {
    detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                       stridex, beta, y, incy, stridey, batch_size);
}

static inline void gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                              std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                              std::int64_t stridex, std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
                              std::int64_t stridey, std::int64_t batch_size) {
    detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x, incx,
                       stridex, beta, y, incy, stridey, batch_size);
}

static inline void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                              sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<float, 1>& x, std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stridec,
                              std::int64_t batch_size) {
    detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea, x, incx,
                       stridex, c, ldc, stridec, batch_size);
}

static inline void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                              sycl::buffer<double, 1>& a, std::int64_t lda, std::int64_t stridea,
                              sycl::buffer<double, 1>& x, std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stridec,
                              std::int64_t batch_size) {
    detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea, x, incx,
                       stridex, c, ldc, stridec, batch_size);
}

static inline void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<std::complex<float>, 1>& x,
                              std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                              std::int64_t stridec, std::int64_t batch_size) {
    detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea, x, incx,
                       stridex, c, ldc, stridec, batch_size);
}

static inline void dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                              std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& x,
                              std::int64_t incx, std::int64_t stridex,
                              sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                              std::int64_t stridec, std::int64_t batch_size) {
    detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea, x, incx,
                       stridex, c, ldc, stridec, batch_size);
}

static inline void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, float alpha,
                       sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                       std::int64_t incy, sycl::buffer<float, 1>& a, std::int64_t lda) {
    detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void ger(sycl::queue& queue, std::int64_t m, std::int64_t n, double alpha,
                       sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                       std::int64_t incy, sycl::buffer<double, 1>& a, std::int64_t lda) {
    detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda) {
    detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda) {
    detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda) {
    detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda) {
    detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, std::complex<float> beta,
                        sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, std::complex<double> beta,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                        std::int64_t ldc) {
    detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void hemm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                        std::int64_t ldc) {
    detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, std::complex<float> beta,
                        sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, std::complex<double> beta,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void her(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                       sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                       sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda) {
    detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                       sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                       sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda) {
    detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda) {
    detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda) {
    detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc) {
    detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb, double beta,
                         sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc) {
    detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, float beta, sycl::buffer<std::complex<float>, 1>& c,
                        std::int64_t ldc) {
    detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void herk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, double beta, sycl::buffer<std::complex<double>, 1>& c,
                        std::int64_t ldc) {
    detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy) {
    detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy) {
    detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                       sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                       sycl::buffer<std::complex<float>, 1>& a) {
    detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                       sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                       sycl::buffer<std::complex<double>, 1>& a) {
    detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<float>, 1>& a) {
    detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                        std::int64_t incy, sycl::buffer<std::complex<double>, 1>& a) {
    detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                         std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                         std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamax(sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::int64_t, 1>& result) {
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamax(sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::int64_t, 1>& result) {
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                         std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                         std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamin(sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::int64_t, 1>& result) {
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
}

static inline void iamin(sycl::queue& queue, std::int64_t n,
                         sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                         sycl::buffer<std::int64_t, 1>& result) {
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
}

static inline void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& result) {
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
}

static inline void nrm2(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<double, 1>& result) {
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
}

static inline void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& result) {
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
}

static inline void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                        std::int64_t incx, sycl::buffer<double, 1>& result) {
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
}

static inline void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                       std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                       std::int64_t incy, float c, float s) {
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
}

static inline void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                       std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y,
                       std::int64_t incy, double c, double s) {
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
}

static inline void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                       std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy, float c,
                       float s) {
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
}

static inline void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                       std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy, double c,
                       double s) {
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
}

static inline void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
                        sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s) {
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
}

static inline void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
                        sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s) {
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
}

static inline void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
                        sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
                        sycl::buffer<std::complex<float>, 1>& s) {
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
}

static inline void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
                        sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
                        sycl::buffer<std::complex<double>, 1>& s) {
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
}

static inline void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
                        sycl::buffer<float, 1>& param) {
    detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param);
}

static inline void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                        std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy,
                        sycl::buffer<double, 1>& param) {
    detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param);
}

static inline void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
                         sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param) {
    detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param);
}

static inline void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1,
                         sycl::buffer<double, 1>& d2, sycl::buffer<double, 1>& x1, double y1,
                         sycl::buffer<double, 1>& param) {
    detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param);
}

static inline void sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx, float beta,
                        sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                        sycl::buffer<double, 1>& x, std::int64_t incx, double beta,
                        sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
                        std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void scal(sycl::queue& queue, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& x, std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void scal(sycl::queue& queue, std::int64_t n, float alpha,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void scal(sycl::queue& queue, std::int64_t n, double alpha,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
}

static inline void sdsdot(sycl::queue& queue, std::int64_t n, float sb, sycl::buffer<float, 1>& x,
                          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
                          sycl::buffer<float, 1>& result) {
    detail::sdsdot(get_device_id(queue), queue, n, sb, x, incx, y, incy, result);
}

static inline void spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, std::int64_t incx,
                        float beta, sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, std::int64_t incx,
                        double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                       sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& a) {
    detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                       sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& a) {
    detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                        std::int64_t incy, sycl::buffer<float, 1>& a) {
    detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                        std::int64_t incy, sycl::buffer<double, 1>& a) {
    detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x,
                        std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy) {
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x,
                        std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy) {
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y,
                        std::int64_t incy) {
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void swap(sycl::queue& queue, std::int64_t n,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
                        sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
}

static inline void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                        sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                        sycl::buffer<double, 1>& b, std::int64_t ldb, double beta,
                        sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                        std::int64_t ldc) {
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void symm(sycl::queue& queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                        std::int64_t ldc) {
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
}

static inline void symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& x,
                        std::int64_t incx, float beta, sycl::buffer<float, 1>& y,
                        std::int64_t incy) {
    detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& x,
                        std::int64_t incx, double beta, sycl::buffer<double, 1>& y,
                        std::int64_t incy) {
    detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
}

static inline void syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                       sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& a,
                       std::int64_t lda) {
    detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                       sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& a,
                       std::int64_t lda) {
    detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& x, std::int64_t incx, sycl::buffer<float, 1>& y,
                        std::int64_t incy, sycl::buffer<float, 1>& a, std::int64_t lda) {
    detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& x, std::int64_t incx, sycl::buffer<double, 1>& y,
                        std::int64_t incy, sycl::buffer<double, 1>& a, std::int64_t lda) {
    detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                         sycl::buffer<float, 1>& b, std::int64_t ldb, float beta,
                         sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                         sycl::buffer<double, 1>& b, std::int64_t ldb, double beta,
                         sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                         std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                         std::int64_t ldc) {
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                         sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                         std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                         std::int64_t ldc) {
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
}

static inline void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                        float beta, sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                        double beta, sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                        std::int64_t ldc) {
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c,
                        std::int64_t ldc) {
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
}

static inline void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<float, 1>& a,
                              std::int64_t lda, std::int64_t stride_a, float beta,
                              sycl::buffer<float, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                              std::int64_t batch_size) {
    detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                       stride_a, beta, c, ldc, stride_c, batch_size);
}

static inline void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                              std::int64_t k, double alpha, sycl::buffer<double, 1>& a,
                              std::int64_t lda, std::int64_t stride_a, double beta,
                              sycl::buffer<double, 1>& c, std::int64_t ldc, std::int64_t stride_c,
                              std::int64_t batch_size) {
    detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                       stride_a, beta, c, ldc, stride_c, batch_size);
}

static inline void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                              std::int64_t k, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, std::complex<float> beta,
                              sycl::buffer<std::complex<float>, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                       stride_a, beta, c, ldc, stride_c, batch_size);
}

static inline void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, std::int64_t n,
                              std::int64_t k, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1>& c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                       stride_a, beta, c, ldc, stride_c, batch_size);
}

static inline void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx) {
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<double, 1>& a,
                        std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx) {
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx) {
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx) {
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx) {
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<double, 1>& a,
                        std::int64_t lda, sycl::buffer<double, 1>& x, std::int64_t incx) {
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<float>, 1>& x,
                        std::int64_t incx) {
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>, 1>& a,
                        std::int64_t lda, sycl::buffer<std::complex<double>, 1>& x,
                        std::int64_t incx) {
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x,
                        std::int64_t incx) {
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x,
                        std::int64_t incx) {
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<float>, 1>& a,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<double>, 1>& a,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x,
                        std::int64_t incx) {
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x,
                        std::int64_t incx) {
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<float>, 1>& a,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<double>, 1>& a,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
                        std::int64_t ldb) {
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
                        std::int64_t ldb) {
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb) {
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb) {
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx) {
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<double, 1>& a, std::int64_t lda,
                        sycl::buffer<double, 1>& x, std::int64_t incx) {
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        sycl::buffer<float, 1>& a, std::int64_t lda, sycl::buffer<float, 1>& b,
                        std::int64_t ldb) {
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        sycl::buffer<double, 1>& a, std::int64_t lda, sycl::buffer<double, 1>& b,
                        std::int64_t ldb) {
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb) {
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb) {
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
}

static inline void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<float, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size) {
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                              std::int64_t stride_a, sycl::buffer<double, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size) {
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size) {
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                              std::int64_t lda, std::int64_t stride_a,
                              sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                              std::int64_t stride_b, std::int64_t batch_size) {
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<float, 1>& a, std::int64_t lda,
                        sycl::buffer<float, 1>& x, std::int64_t incx) {
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<double, 1>& a, std::int64_t lda,
                        sycl::buffer<double, 1>& x, std::int64_t incx) {
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                        sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, float alpha, sycl::buffer<float, 1>& a,
                                  std::int64_t lda, std::int64_t stride_a,
                                  sycl::buffer<float, 1>& b, std::int64_t ldb,
                                  std::int64_t stride_b, std::int64_t batch_size) {
    detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stride_a, b,
                           ldb, stride_b, batch_size);
}

static inline void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
                                  std::int64_t lda, std::int64_t stride_a,
                                  sycl::buffer<double, 1>& b, std::int64_t ldb,
                                  std::int64_t stride_b, std::int64_t batch_size) {
    detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stride_a, b,
                           ldb, stride_b, batch_size);
}

static inline void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, std::complex<float> alpha,
                                  sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b,
                                  std::int64_t ldb, std::int64_t stride_b,
                                  std::int64_t batch_size) {
    detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stride_a, b,
                           ldb, stride_b, batch_size);
}

static inline void omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, std::complex<double> alpha,
                                  sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                                  std::int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                                  std::int64_t ldb, std::int64_t stride_b,
                                  std::int64_t batch_size) {
    detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stride_a, b,
                           ldb, stride_b, batch_size);
}

static inline void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, float alpha, sycl::buffer<float, 1>& ab,
                                  std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                  std::int64_t batch_size) {
    detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb, stride,
                           batch_size);
}

static inline void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, double alpha, sycl::buffer<double, 1>& ab,
                                  std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                  std::int64_t batch_size) {
    detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb, stride,
                           batch_size);
}

static inline void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, std::complex<float> alpha,
                                  sycl::buffer<std::complex<float>, 1>& ab, std::int64_t lda,
                                  std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb, stride,
                           batch_size);
}

static inline void imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                  std::int64_t n, std::complex<double> alpha,
                                  sycl::buffer<std::complex<double>, 1>& ab, std::int64_t lda,
                                  std::int64_t ldb, std::int64_t stride, std::int64_t batch_size) {
    detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb, stride,
                           batch_size);
}

static inline void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, float alpha,
                                 sycl::buffer<float, 1>& a, std::int64_t lda, std::int64_t stride_a,
                                 float beta, sycl::buffer<float, 1>& b, std::int64_t ldb,
                                 std::int64_t stride_b, sycl::buffer<float, 1>& c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                          stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

static inline void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, double alpha,
                                 sycl::buffer<double, 1>& a, std::int64_t lda,
                                 std::int64_t stride_a, double beta, sycl::buffer<double, 1>& b,
                                 std::int64_t ldb, std::int64_t stride_b,
                                 sycl::buffer<double, 1>& c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                          stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

static inline void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                 sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                                 std::int64_t stride_a, std::complex<float> beta,
                                 sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                                 std::int64_t stride_b, sycl::buffer<std::complex<float>, 1>& c,
                                 std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                          stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

static inline void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                 sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                                 std::int64_t stride_a, std::complex<double> beta,
                                 sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                                 std::int64_t stride_b, sycl::buffer<std::complex<double>, 1>& c,
                                 std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                          stride_a, beta, b, ldb, stride_b, c, ldc, stride_c, batch_size);
}

static inline void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                            sycl::buffer<float, 1>& b, std::int64_t ldb) {
    detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb);
}

static inline void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                            sycl::buffer<double, 1>& b, std::int64_t ldb) {
    detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb);
}

static inline void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                            std::int64_t lda, sycl::buffer<std::complex<float>, 1>& b,
                            std::int64_t ldb) {
    detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb);
}

static inline void omatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                            std::int64_t lda, sycl::buffer<std::complex<double>, 1>& b,
                            std::int64_t ldb) {
    detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb);
}

static inline void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                             float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                             std::int64_t stridea, sycl::buffer<float, 1>& b, std::int64_t ldb,
                             std::int64_t strideb) {
    detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, b, ldb,
                      strideb);
}

static inline void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                             double alpha, sycl::buffer<double, 1>& a, std::int64_t lda,
                             std::int64_t stridea, sycl::buffer<double, 1>& b, std::int64_t ldb,
                             std::int64_t strideb) {
    detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, b, ldb,
                      strideb);
}

static inline void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                             std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                             std::int64_t lda, std::int64_t stridea,
                             sycl::buffer<std::complex<float>, 1>& b, std::int64_t ldb,
                             std::int64_t strideb) {
    detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, b, ldb,
                      strideb);
}

static inline void omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                             std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                             std::int64_t lda, std::int64_t stridea,
                             sycl::buffer<std::complex<double>, 1>& b, std::int64_t ldb,
                             std::int64_t strideb) {
    detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, b, ldb,
                      strideb);
}

static inline void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            float alpha, sycl::buffer<float, 1>& ab, std::int64_t lda,
                            std::int64_t ldb) {
    detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb);
}

static inline void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            double alpha, sycl::buffer<double, 1>& ab, std::int64_t lda,
                            std::int64_t ldb) {
    detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb);
}

static inline void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                            std::int64_t lda, std::int64_t ldb) {
    detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb);
}

static inline void imatcopy(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                            std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                            std::int64_t lda, std::int64_t ldb) {
    detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb);
}

static inline void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                           std::int64_t n, float alpha, sycl::buffer<float, 1>& a, std::int64_t lda,
                           float beta, sycl::buffer<float, 1>& b, std::int64_t ldb,
                           sycl::buffer<float, 1>& c, std::int64_t ldc) {
    detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb,
                    c, ldc);
}

static inline void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                           std::int64_t n, double alpha, sycl::buffer<double, 1>& a,
                           std::int64_t lda, double beta, sycl::buffer<double, 1>& b,
                           std::int64_t ldb, sycl::buffer<double, 1>& c, std::int64_t ldc) {
    detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb,
                    c, ldc);
}

static inline void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                           std::int64_t n, std::complex<float> alpha,
                           sycl::buffer<std::complex<float>, 1>& a, std::int64_t lda,
                           std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& b,
                           std::int64_t ldb, sycl::buffer<std::complex<float>, 1>& c,
                           std::int64_t ldc) {
    detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb,
                    c, ldc);
}

static inline void omatadd(sycl::queue& queue, transpose transa, transpose transb, std::int64_t m,
                           std::int64_t n, std::complex<double> alpha,
                           sycl::buffer<std::complex<double>, 1>& a, std::int64_t lda,
                           std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& b,
                           std::int64_t ldb, sycl::buffer<std::complex<double>, 1>& c,
                           std::int64_t ldc) {
    detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb,
                    c, ldc);
}

// USM APIs

static inline sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                               std::int64_t incx, float* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                               std::int64_t incx, double* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event asum(sycl::queue& queue, std::int64_t n, const float* x,
                               std::int64_t incx, float* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event asum(sycl::queue& queue, std::int64_t n, const double* x,
                               std::int64_t incx, double* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event axpy(sycl::queue& queue, std::int64_t n, float alpha, const float* x,
                               std::int64_t incx, float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event axpy(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                               std::int64_t incx, double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float>* x, std::int64_t incx,
                               std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double>* x, std::int64_t incx,
                               std::complex<double>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, double* alpha,
                                     const double** x, std::int64_t* incx, double** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n, float* alpha,
                                     const float** x, std::int64_t* incx, float** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n,
                                     std::complex<double>* alpha, const std::complex<double>** x,
                                     std::int64_t* incx, std::complex<double>** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t* n,
                                     std::complex<float>* alpha, const std::complex<float>** x,
                                     std::int64_t* incx, std::complex<float>** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, float alpha,
                                     const float* x, std::int64_t incx, std::int64_t stridex,
                                     float* y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, double alpha,
                                     const double* x, std::int64_t incx, std::int64_t stridex,
                                     double* y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                                     const std::complex<float>* x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<float>* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event axpy_batch(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                                     const std::complex<double>* x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<double>* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event axpby(sycl::queue& queue, std::int64_t n, float alpha, const float* x,
                                std::int64_t incx, const float beta, float* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event axpby(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                                std::int64_t incx, const double beta, double* y, std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                                const std::complex<float>* x, std::int64_t incx,
                                const std::complex<float> beta, std::complex<float>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                                const std::complex<double>* x, std::int64_t incx,
                                const std::complex<double> beta, std::complex<double>* y,
                                std::int64_t incy,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::axpby(get_device_id(queue), queue, n, alpha, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event copy(sycl::queue& queue, std::int64_t n, const float* x,
                               std::int64_t incx, float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event copy(sycl::queue& queue, std::int64_t n, const double* x,
                               std::int64_t incx, double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                               std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                               std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const float** x,
                                     std::int64_t* incx, float** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, y, incy, group_count,
                                   group_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t* n, const double** x,
                                     std::int64_t* incx, double** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, y, incy, group_count,
                                   group_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t* n,
                                     const std::complex<float>** x, std::int64_t* incx,
                                     std::complex<float>** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, y, incy, group_count,
                                   group_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t* n,
                                     const std::complex<double>** x, std::int64_t* incx,
                                     std::complex<double>** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, y, incy, group_count,
                                   group_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const float* x,
                                     std::int64_t incx, std::int64_t stridex, float* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t n, const double* x,
                                     std::int64_t incx, std::int64_t stridex, double* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t n,
                                     const std::complex<float>* x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<float>* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event copy_batch(sycl::queue& queue, std::int64_t n,
                                     const std::complex<double>* x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<double>* y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::copy_batch(get_device_id(queue), queue, n, x, incx, stridex, y, incy,
                                   stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                              const float* y, std::int64_t incy, float* result,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dot(sycl::queue& queue, std::int64_t n, const double* x,
                              std::int64_t incx, const double* y, std::int64_t incy, double* result,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                              const float* y, std::int64_t incy, double* result,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::int64_t kl, std::int64_t ku, float alpha, const float* a,
                               std::int64_t lda, const float* x, std::int64_t incx, float beta,
                               float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::int64_t kl, std::int64_t ku, double alpha, const double* a,
                               std::int64_t lda, const double* x, std::int64_t incx, double beta,
                               double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                               const std::complex<float>* a, std::int64_t lda,
                               const std::complex<float>* x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event gbmv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                               const std::complex<double>* a, std::int64_t lda,
                               const std::complex<double>* x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double>* y,
                               std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const float* a, std::int64_t lda, const float* b, std::int64_t ldb,
                               float beta, float* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                               const double* a, std::int64_t lda, const double* b, std::int64_t ldb,
                               double beta, double* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                               const sycl::half* a, std::int64_t lda, const sycl::half* b,
                               std::int64_t ldb, sycl::half beta, sycl::half* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const sycl::half* a, std::int64_t lda, const sycl::half* b,
                               std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const bfloat16* a, std::int64_t lda, const bfloat16* b,
                               std::int64_t ldb, float beta, float* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     float* alpha, const float** a, std::int64_t* lda,
                                     const float** b, std::int64_t* ldb, float* beta, float** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     double* alpha, const double** a, std::int64_t* lda,
                                     const double** b, std::int64_t* ldb, double* beta, double** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     std::complex<float>* alpha, const std::complex<float>** a,
                                     std::int64_t* lda, const std::complex<float>** b,
                                     std::int64_t* ldb, std::complex<float>* beta,
                                     std::complex<float>** c, std::int64_t* ldc,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     std::complex<double>* alpha, const std::complex<double>** a,
                                     std::int64_t* lda, const std::complex<double>** b,
                                     std::int64_t* ldb, std::complex<double>* beta,
                                     std::complex<double>** c, std::int64_t* ldc,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     sycl::half* alpha, const sycl::half** a, std::int64_t* lda,
                                     const sycl::half** b, std::int64_t* ldb, sycl::half* beta,
                                     sycl::half** c, std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     float* alpha, const sycl::half** a, std::int64_t* lda,
                                     const sycl::half** b, std::int64_t* ldb, float* beta,
                                     float** c, std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     float* alpha, const std::int8_t** a, std::int64_t* lda,
                                     const std::int8_t** b, std::int64_t* ldb, float* beta,
                                     float** c, std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb,
                                     std::int64_t* m, std::int64_t* n, std::int64_t* k,
                                     float* alpha, const std::int8_t** a, std::int64_t* lda,
                                     const std::int8_t** b, std::int64_t* ldb, float* beta,
                                     std::int32_t** c, std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     const float* a, std::int64_t lda, std::int64_t stride_a,
                                     const float* b, std::int64_t ldb, std::int64_t stride_b,
                                     float beta, float* c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                     const double* a, std::int64_t lda, std::int64_t stride_a,
                                     const double* b, std::int64_t ldb, std::int64_t stride_b,
                                     double beta, double* c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(
    sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<float>* b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<float> beta, std::complex<float>* c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(
    sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double>* a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<double>* b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double>* c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     sycl::half alpha, const sycl::half* a, std::int64_t lda,
                                     std::int64_t stride_a, const sycl::half* b, std::int64_t ldb,
                                     std::int64_t stride_b, sycl::half beta, sycl::half* c,
                                     std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     const sycl::half* a, std::int64_t lda, std::int64_t stride_a,
                                     const sycl::half* b, std::int64_t ldb, std::int64_t stride_b,
                                     float beta, float* c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                                     const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                                     float beta, float* c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     const std::int8_t* a, std::int64_t lda, std::int64_t stride_a,
                                     const std::int8_t* b, std::int64_t ldb, std::int64_t stride_b,
                                     float beta, std::int32_t* c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, float alpha,
                                const float* a, std::int64_t lda, const float* b, std::int64_t ldb,
                                float beta, float* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, double alpha,
                                const double* a, std::int64_t lda, const double* b,
                                std::int64_t ldb, double beta, double* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha, const std::complex<float>* a,
                                std::int64_t lda, const std::complex<float>* b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha, const std::complex<double>* a,
                                std::int64_t lda, const std::complex<double>* b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::int8_t* a, std::int64_t lda,
                                    std::int8_t ao, const std::uint8_t* b, std::int64_t ldb,
                                    std::uint8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                                    const std::int32_t* co,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k,
                                  alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

static inline sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::int8_t* a, std::int64_t lda,
                                    std::int8_t ao, const std::int8_t* b, std::int64_t ldb,
                                    std::int8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                                    const std::int32_t* co,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k,
                                  alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

static inline sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::uint8_t* a, std::int64_t lda,
                                    std::uint8_t ao, const std::int8_t* b, std::int64_t ldb,
                                    std::int8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                                    const std::int32_t* co,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k,
                                  alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

static inline sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::uint8_t* a, std::int64_t lda,
                                    std::uint8_t ao, const std::uint8_t* b, std::int64_t ldb,
                                    std::uint8_t bo, float beta, std::int32_t* c, std::int64_t ldc,
                                    const std::int32_t* co,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k,
                                  alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
    return done;
}

static inline sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               float alpha, const float* a, std::int64_t lda, const float* x,
                               std::int64_t incx, float beta, float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               double alpha, const double* a, std::int64_t lda, const double* x,
                               std::int64_t incx, double beta, double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event gemv(sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double>* y,
                               std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                     std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                     std::int64_t stridea, const float* x, std::int64_t incx,
                                     std::int64_t stridex, float beta, float* y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x,
                           incx, stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                     std::int64_t n, double alpha, const double* a,
                                     std::int64_t lda, std::int64_t stridea, const double* x,
                                     std::int64_t incx, std::int64_t stridex, double beta,
                                     double* y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x,
                           incx, stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(
    sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float>* a, std::int64_t lda, std::int64_t stridea,
    const std::complex<float>* x, std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
    std::complex<float>* y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
    const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x,
                           incx, stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(
    sycl::queue& queue, transpose trans, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double>* a, std::int64_t lda, std::int64_t stridea,
    const std::complex<double>* x, std::int64_t incx, std::int64_t stridex,
    std::complex<double> beta, std::complex<double>* y, std::int64_t incy, std::int64_t stridey,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea, x,
                           incx, stridex, beta, y, incy, stridey, batch_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                     std::int64_t* n, float* alpha, const float** a,
                                     std::int64_t* lda, const float** x, std::int64_t* incx,
                                     float* beta, float** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx,
                                   beta, y, incy, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                     std::int64_t* n, double* alpha, const double** a,
                                     std::int64_t* lda, const double** x, std::int64_t* incx,
                                     double* beta, double** y, std::int64_t* incy,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx,
                                   beta, y, incy, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                     std::int64_t* n, std::complex<float>* alpha,
                                     const std::complex<float>** a, std::int64_t* lda,
                                     const std::complex<float>** x, std::int64_t* incx,
                                     std::complex<float>* beta, std::complex<float>** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx,
                                   beta, y, incy, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event gemv_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                     std::int64_t* n, std::complex<double>* alpha,
                                     const std::complex<double>** a, std::int64_t* lda,
                                     const std::complex<double>** x, std::int64_t* incx,
                                     std::complex<double>* beta, std::complex<double>** y,
                                     std::int64_t* incy, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gemv_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx,
                                   beta, y, incy, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m,
                                     std::int64_t n, const float* a, std::int64_t lda,
                                     std::int64_t stridea, const float* x, std::int64_t incx,
                                     std::int64_t stridex, float* c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea,
                                   x, incx, stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m,
                                     std::int64_t n, const double* a, std::int64_t lda,
                                     std::int64_t stridea, const double* x, std::int64_t incx,
                                     std::int64_t stridex, double* c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea,
                                   x, incx, stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m,
                                     std::int64_t n, const std::complex<float>* a, std::int64_t lda,
                                     std::int64_t stridea, const std::complex<float>* x,
                                     std::int64_t incx, std::int64_t stridex,
                                     std::complex<float>* c, std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea,
                                   x, incx, stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side left_right, std::int64_t m,
                                     std::int64_t n, const std::complex<double>* a,
                                     std::int64_t lda, std::int64_t stridea,
                                     const std::complex<double>* x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<double>* c,
                                     std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, stridea,
                                   x, incx, stridex, c, ldc, stridec, batch_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side* left_right, std::int64_t* m,
                                     std::int64_t* n, const float** a, std::int64_t* lda,
                                     const float** x, std::int64_t* incx, float** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, x, incx,
                                   c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side* left_right, std::int64_t* m,
                                     std::int64_t* n, const double** a, std::int64_t* lda,
                                     const double** x, std::int64_t* incx, double** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, x, incx,
                                   c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side* left_right, std::int64_t* m,
                                     std::int64_t* n, const std::complex<float>** a,
                                     std::int64_t* lda, const std::complex<float>** x,
                                     std::int64_t* incx, std::complex<float>** c, std::int64_t* ldc,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, x, incx,
                                   c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event dgmm_batch(sycl::queue& queue, side* left_right, std::int64_t* m,
                                     std::int64_t* n, const std::complex<double>** a,
                                     std::int64_t* lda, const std::complex<double>** x,
                                     std::int64_t* incx, std::complex<double>** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::dgmm_batch(get_device_id(queue), queue, left_right, m, n, a, lda, x, incx,
                                   c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, float alpha,
                              const float* x, std::int64_t incx, const float* y, std::int64_t incy,
                              float* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event ger(sycl::queue& queue, std::int64_t m, std::int64_t n, double alpha,
                              const double* x, std::int64_t incx, const double* y,
                              std::int64_t incy, double* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    return done;
}

static inline sycl::event gerc(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    return done;
}

static inline sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    return done;
}

static inline sycl::event geru(sycl::queue& queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    return done;
}

static inline sycl::event hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event hbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double>* y,
                               std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float>* a, std::int64_t lda,
                               const std::complex<float>* b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double>* a, std::int64_t lda,
                               const std::complex<double>* b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, const std::complex<float>* x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event hemv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, const std::complex<double>* x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double>* y,
                               std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event her(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                              const std::complex<float>* x, std::int64_t incx,
                              std::complex<float>* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event her(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                              const std::complex<double>* x, std::int64_t incx,
                              std::complex<double>* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    return done;
}

static inline sycl::event her2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    return done;
}

static inline sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                const std::complex<float>* b, std::int64_t ldb, float beta,
                                std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                const std::complex<double>* b, std::int64_t ldb, double beta,
                                std::complex<double>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               const std::complex<float>* a, std::int64_t lda, float beta,
                               std::complex<float>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               const std::complex<double>* a, std::int64_t lda, double beta,
                               std::complex<double>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* a,
                               const std::complex<float>* x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event hpmv(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* a,
                               const std::complex<double>* x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double>* y,
                               std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                              const std::complex<float>* x, std::int64_t incx,
                              std::complex<float>* a,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline sycl::event hpr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                              const std::complex<double>* x, std::int64_t incx,
                              std::complex<double>* a,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline sycl::event hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* x,
                               std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                               std::complex<float>* a,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    return done;
}

static inline sycl::event hpr2(sycl::queue& queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* x,
                               std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                               std::complex<double>* a,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    return done;
}

static inline sycl::event iamax(sycl::queue& queue, std::int64_t n, const float* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamax(sycl::queue& queue, std::int64_t n, const double* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamin(sycl::queue& queue, std::int64_t n, const float* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamin(sycl::queue& queue, std::int64_t n, const double* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                                std::int64_t incx, std::int64_t* result,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                               std::int64_t incx, float* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                               std::int64_t incx, double* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event nrm2(sycl::queue& queue, std::int64_t n, const float* x,
                               std::int64_t incx, float* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event nrm2(sycl::queue& queue, std::int64_t n, const double* x,
                               std::int64_t incx, double* result,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    return done;
}

static inline sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<float>* x,
                              std::int64_t incx, std::complex<float>* y, std::int64_t incy, float c,
                              float s, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<double>* x,
                              std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                              double c, double s,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline sycl::event rot(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx,
                              float* y, std::int64_t incy, float c, float s,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline sycl::event rot(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx,
                              double* y, std::int64_t incy, double c, double s,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    return done;
}

static inline sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    return done;
}

static inline sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b,
                               float* c, std::complex<float>* s,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    return done;
}

static inline sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b,
                               double* c, std::complex<double>* s,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    return done;
}

static inline sycl::event rotm(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx,
                               float* y, std::int64_t incy, float* param,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

static inline sycl::event rotm(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx,
                               double* y, std::int64_t incy, double* param,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

static inline sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, float y1,
                                float* param, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

static inline sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1, double y1,
                                double* param, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

static inline sycl::event sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                               float alpha, const float* a, std::int64_t lda, const float* x,
                               std::int64_t incx, float beta, float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event sbmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                               double alpha, const double* a, std::int64_t lda, const double* x,
                               std::int64_t incx, double beta, double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, float* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, double* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline sycl::event sdsdot(sycl::queue& queue, std::int64_t n, float sb, const float* x,
                                 std::int64_t incx, const float* y, std::int64_t incy,
                                 float* result, const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::sdsdot(get_device_id(queue), queue, n, sb, x, incx, y, incy, result, dependencies);
    return done;
}

static inline sycl::event spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                               const float* a, const float* x, std::int64_t incx, float beta,
                               float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event spmv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                               const double* a, const double* x, std::int64_t incx, double beta,
                               double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    return done;
}

static inline sycl::event spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float* x, std::int64_t incx, float* a,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline sycl::event spr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                              const double* x, std::int64_t incx, double* a,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline sycl::event spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                               const float* x, std::int64_t incx, const float* y, std::int64_t incy,
                               float* a, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    return done;
}

static inline sycl::event spr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                               const double* x, std::int64_t incx, const double* y,
                               std::int64_t incy, double* a,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    return done;
}

static inline sycl::event swap(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx,
                               float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event swap(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx,
                               double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<float>* x,
                               std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<double>* x,
                               std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, float alpha, const float* a,
                               std::int64_t lda, const float* b, std::int64_t ldb, float beta,
                               float* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, double alpha, const double* a,
                               std::int64_t lda, const double* b, std::int64_t ldb, double beta,
                               double* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float>* a, std::int64_t lda,
                               const std::complex<float>* b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double>* a, std::int64_t lda,
                               const std::complex<double>* b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                               const float* a, std::int64_t lda, const float* x, std::int64_t incx,
                               float beta, float* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event symv(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                               const double* a, std::int64_t lda, const double* x,
                               std::int64_t incx, double beta, double* y, std::int64_t incy,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    return done;
}

static inline sycl::event syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float* x, std::int64_t incx, float* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event syr(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                              const double* x, std::int64_t incx, double* a, std::int64_t lda,
                              const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    return done;
}

static inline sycl::event syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, float alpha,
                               const float* x, std::int64_t incx, const float* y, std::int64_t incy,
                               float* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    return done;
}

static inline sycl::event syr2(sycl::queue& queue, uplo upper_lower, std::int64_t n, double alpha,
                               const double* x, std::int64_t incx, const double* y,
                               std::int64_t incy, double* a, std::int64_t lda,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    return done;
}

static inline sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, float alpha, const float* a,
                                std::int64_t lda, const float* b, std::int64_t ldb, float beta,
                                float* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, double alpha, const double* a,
                                std::int64_t lda, const double* b, std::int64_t ldb, double beta,
                                double* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                const std::complex<float>* a, std::int64_t lda,
                                const std::complex<float>* b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                const std::complex<double>* a, std::int64_t lda,
                                const std::complex<double>* b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double>* c,
                                std::int64_t ldc,
                                const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha, const float* a,
                               std::int64_t lda, float beta, float* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha, const double* a,
                               std::int64_t lda, double beta, double* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               const std::complex<float>* a, std::int64_t lda,
                               std::complex<float> beta, std::complex<float>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               const std::complex<double>* a, std::int64_t lda,
                               std::complex<double> beta, std::complex<double>* c, std::int64_t ldc,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans,
                                     std::int64_t* n, std::int64_t* k, float* alpha,
                                     const float** a, std::int64_t* lda, float* beta, float** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans,
                                     std::int64_t* n, std::int64_t* k, double* alpha,
                                     const double** a, std::int64_t* lda, double* beta, double** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans,
                                     std::int64_t* n, std::int64_t* k, std::complex<float>* alpha,
                                     const std::complex<float>** a, std::int64_t* lda,
                                     std::complex<float>* beta, std::complex<float>** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans,
                                     std::int64_t* n, std::int64_t* k, std::complex<double>* alpha,
                                     const std::complex<double>** a, std::int64_t* lda,
                                     std::complex<double>* beta, std::complex<double>** c,
                                     std::int64_t* ldc, std::int64_t group_count,
                                     std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, beta, c, ldc, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, float alpha, const float* a,
                                     std::int64_t lda, std::int64_t stride_a, float beta, float* c,
                                     std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, double alpha, const double* a,
                                     std::int64_t lda, std::int64_t stride_a, double beta,
                                     double* c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                     const std::complex<float>* a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<float> beta,
                                     std::complex<float>* c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                     const std::complex<double>* a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<double> beta,
                                     std::complex<double>* c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::syrk_batch(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a,
                                   lda, stride_a, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const float* a,
                               std::int64_t lda, float* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const double* a,
                               std::int64_t lda, double* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<float>* a, std::int64_t lda,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<double>* a, std::int64_t lda,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const float* a,
                               std::int64_t lda, float* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const double* a,
                               std::int64_t lda, double* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<float>* a, std::int64_t lda,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<double>* a, std::int64_t lda,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    return done;
}

static inline sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float* a, float* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double* a, double* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float>* a,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double>* a,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float* a, float* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double* a, double* x,
                               std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float>* a,
                               std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double>* a,
                               std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    return done;
}

static inline sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, const float* a, std::int64_t lda, float* b,
                               std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, const double* a, std::int64_t lda, double* b,
                               std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float* a, std::int64_t lda,
                               float* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double* a, std::int64_t lda,
                               double* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float>* a,
                               std::int64_t lda, std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double>* a,
                               std::int64_t lda, std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, const float* a, std::int64_t lda, float* b,
                               std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, const double* a, std::int64_t lda, double* b,
                               std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float>* a,
                               std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double>* a,
                               std::int64_t lda, std::complex<double>* b, std::int64_t ldb,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                     std::int64_t stride_a, float* b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans,
                                   unit_diag, m, n, alpha, a, lda, stride_a, b, ldb, stride_b,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, double alpha, const double* a,
                                     std::int64_t lda, std::int64_t stride_a, double* b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans,
                                   unit_diag, m, n, alpha, a, lda, stride_a, b, ldb, stride_b,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<float> alpha,
                                     const std::complex<float>* a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<float>* b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans,
                                   unit_diag, m, n, alpha, a, lda, stride_a, b, ldb, stride_b,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<double> alpha,
                                     const std::complex<double>* a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<double>* b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans,
                                   unit_diag, m, n, alpha, a, lda, stride_a, b, ldb, stride_b,
                                   batch_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,
                                     transpose* trans, diag* unit_diag, std::int64_t* m,
                                     std::int64_t* n, float* alpha, const float** a,
                                     std::int64_t* lda, float** b, std::int64_t* ldb,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                           m, n, alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,
                                     transpose* trans, diag* unit_diag, std::int64_t* m,
                                     std::int64_t* n, double* alpha, const double** a,
                                     std::int64_t* lda, double** b, std::int64_t* ldb,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                           m, n, alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,
                                     transpose* trans, diag* unit_diag, std::int64_t* m,
                                     std::int64_t* n, std::complex<float>* alpha,
                                     const std::complex<float>** a, std::int64_t* lda,
                                     std::complex<float>** b, std::int64_t* ldb,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                           m, n, alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower,
                                     transpose* trans, diag* unit_diag, std::int64_t* m,
                                     std::int64_t* n, std::complex<double>* alpha,
                                     const std::complex<double>** a, std::int64_t* lda,
                                     std::complex<double>** b, std::int64_t* ldb,
                                     std::int64_t group_count, std::int64_t* group_size,
                                     const std::vector<sycl::event>& dependencies = {}) {
    auto done =
        detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                           m, n, alpha, a, lda, b, ldb, group_count, group_size, dependencies);
    return done;
}

static inline sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float* a, std::int64_t lda,
                               float* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double* a, std::int64_t lda,
                               double* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float>* a,
                               std::int64_t lda, std::complex<float>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double>* a,
                               std::int64_t lda, std::complex<double>* x, std::int64_t incx,
                               const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, const float* a,
                                         std::int64_t lda, std::int64_t stride_a, float* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda,
                                       stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, const double* a,
                                         std::int64_t lda, std::int64_t stride_a, double* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda,
                                       stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         const std::complex<float>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<float>* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda,
                                       stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         const std::complex<double>* a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<double>* b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda,
                                       stride_a, b, ldb, stride_b, batch_size, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, float* ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, stride, batch_size, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, double* ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, stride, batch_size, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         std::complex<float>* ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, stride, batch_size, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         std::complex<double>* ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, stride, batch_size, dependencies);
    return done;
}

static inline sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, float alpha, const float* a,
                                        std::int64_t lda, std::int64_t stride_a, float beta,
                                        const float* b, std::int64_t ldb, std::int64_t stride_b,
                                        float* c, std::int64_t ldc, std::int64_t stride_c,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a,
                                      lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c,
                                      batch_size, dependencies);
    return done;
}

static inline sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, double alpha,
                                        const double* a, std::int64_t lda, std::int64_t stride_a,
                                        double beta, const double* b, std::int64_t ldb,
                                        std::int64_t stride_b, double* c, std::int64_t ldc,
                                        std::int64_t stride_c, std::int64_t batch_size,
                                        const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a,
                                      lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c,
                                      batch_size, dependencies);
    return done;
}

static inline sycl::event omatadd_batch(
    sycl::queue& queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float>* a, std::int64_t lda,
    std::int64_t stride_a, std::complex<float> beta, const std::complex<float>* b, std::int64_t ldb,
    std::int64_t stride_b, std::complex<float>* c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a,
                                      lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c,
                                      batch_size, dependencies);
    return done;
}

static inline sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                        const std::complex<double>* a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<double> beta,
                                        const std::complex<double>* b, std::int64_t ldb,
                                        std::int64_t stride_b, std::complex<double>* c,
                                        std::int64_t ldc, std::int64_t stride_c,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd_batch(get_device_id(queue), queue, transa, transb, m, n, alpha, a,
                                      lda, stride_a, beta, b, ldb, stride_b, c, ldc, stride_c,
                                      batch_size, dependencies);
    return done;
}

static inline sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                   float* b, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                   double* b, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, std::complex<float> alpha,
                                   const std::complex<float>* a, std::int64_t lda,
                                   std::complex<float>* b, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event omatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, std::complex<double> alpha,
                                   const std::complex<double>* a, std::int64_t lda,
                                   std::complex<double>* b, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m,
                                    std::int64_t n, float alpha, const float* a, std::int64_t lda,
                                    std::int64_t stridea, float* b, std::int64_t ldb,
                                    std::int64_t strideb,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea,
                                  b, ldb, strideb, dependencies);
    return done;
}

static inline sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m,
                                    std::int64_t n, double alpha, const double* a, std::int64_t lda,
                                    std::int64_t stridea, double* b, std::int64_t ldb,
                                    std::int64_t strideb,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea,
                                  b, ldb, strideb, dependencies);
    return done;
}

static inline sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m,
                                    std::int64_t n, std::complex<float> alpha,
                                    const std::complex<float>* a, std::int64_t lda,
                                    std::int64_t stridea, std::complex<float>* b, std::int64_t ldb,
                                    std::int64_t strideb,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea,
                                  b, ldb, strideb, dependencies);
    return done;
}

static inline sycl::event omatcopy2(sycl::queue& queue, transpose trans, std::int64_t m,
                                    std::int64_t n, std::complex<double> alpha,
                                    const std::complex<double>* a, std::int64_t lda,
                                    std::int64_t stridea, std::complex<double>* b, std::int64_t ldb,
                                    std::int64_t strideb,
                                    const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy2(get_device_id(queue), queue, trans, m, n, alpha, a, lda, stridea,
                                  b, ldb, strideb, dependencies);
    return done;
}

static inline sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, float alpha, float* ab, std::int64_t lda,
                                   std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, double alpha, double* ab, std::int64_t lda,
                                   std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, std::complex<float> alpha,
                                   std::complex<float>* ab, std::int64_t lda, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event imatcopy(sycl::queue& queue, transpose trans, std::int64_t m,
                                   std::int64_t n, std::complex<double> alpha,
                                   std::complex<double>* ab, std::int64_t lda, std::int64_t ldb,
                                   const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy(get_device_id(queue), queue, trans, m, n, alpha, ab, lda, ldb,
                                 dependencies);
    return done;
}

static inline sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, float alpha, const float* a,
                                  std::int64_t lda, float beta, const float* b, std::int64_t ldb,
                                  float* c, std::int64_t ldc,
                                  const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                                beta, b, ldb, c, ldc, dependencies);
    return done;
}

static inline sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, double alpha, const double* a,
                                  std::int64_t lda, double beta, const double* b, std::int64_t ldb,
                                  double* c, std::int64_t ldc,
                                  const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                                beta, b, ldb, c, ldc, dependencies);
    return done;
}

static inline sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                  const std::complex<float>* a, std::int64_t lda,
                                  std::complex<float> beta, const std::complex<float>* b,
                                  std::int64_t ldb, std::complex<float>* c, std::int64_t ldc,
                                  const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                                beta, b, ldb, c, ldc, dependencies);
    return done;
}

static inline sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                  const std::complex<double>* a, std::int64_t lda,
                                  std::complex<double> beta, const std::complex<double>* b,
                                  std::int64_t ldb, std::complex<double>* c, std::int64_t ldc,
                                  const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatadd(get_device_id(queue), queue, transa, transb, m, n, alpha, a, lda,
                                beta, b, ldb, c, ldc, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, float* alpha, const float** a,
                                         std::int64_t* lda, float** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, double* alpha, const double** a,
                                         std::int64_t* lda, double** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, std::complex<float>* alpha,
                                         const std::complex<float>** a, std::int64_t* lda,
                                         std::complex<float>** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, std::complex<double>* alpha,
                                         const std::complex<double>** a, std::int64_t* lda,
                                         std::complex<double>** b, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::omatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, a, lda, b,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, float* alpha, float** ab,
                                         std::int64_t* lda, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, double* alpha, double** ab,
                                         std::int64_t* lda, std::int64_t* ldb,
                                         std::int64_t group_count, std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, std::complex<float>* alpha,
                                         std::complex<float>** ab, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}

static inline sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, std::int64_t* m,
                                         std::int64_t* n, std::complex<double>* alpha,
                                         std::complex<double>** ab, std::int64_t* lda,
                                         std::int64_t* ldb, std::int64_t group_count,
                                         std::int64_t* groupsize,
                                         const std::vector<sycl::event>& dependencies = {}) {
    auto done = detail::imatcopy_batch(get_device_id(queue), queue, trans, m, n, alpha, ab, lda,
                                       ldb, group_count, groupsize, dependencies);
    return done;
}
