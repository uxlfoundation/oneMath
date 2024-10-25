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

// Buffer APIs

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result) {
    blas_major::asum(queue, n, x, incx, result);
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result) {
    blas_major::asum(queue, n, x, incx, result);
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result) {
    blas_major::asum(queue, n, x, incx, result);
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result) {
    blas_major::asum(queue, n, x, incx, result);
}

void axpy(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy) {
    blas_major::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy) {
    blas_major::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    blas_major::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    blas_major::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpby(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
           std::int64_t incx, float beta, sycl::buffer<float, 1>& y, std::int64_t incy) {
    blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy);
}

void axpby(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
           std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy);
}

void axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy);
}

void axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy);
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy) {
    blas_major::copy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy) {
    blas_major::copy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    blas_major::copy(queue, n, x, incx, y, incy);
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    blas_major::copy(queue, n, x, incx, y, incy);
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& result) {
    blas_major::dot(queue, n, x, incx, y, incy, result);
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result) {
    blas_major::dot(queue, n, x, incx, y, incy, result);
}

void sdsdot(sycl::queue& queue, std::int64_t n, float sb, sycl::buffer<float, 1>& x,
            std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
            sycl::buffer<float, 1>& result) {
    blas_major::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result) {
    blas_major::dot(queue, n, x, incx, y, incy, result);
}

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    blas_major::dotc(queue, n, x, incx, y, incy, result);
}

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    blas_major::dotc(queue, n, x, incx, y, incy, result);
}

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    blas_major::dotu(queue, n, x, incx, y, incy, result);
}

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    blas_major::dotu(queue, n, x, incx, y, incy, result);
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result) {
    blas_major::nrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result) {
    blas_major::nrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result) {
    blas_major::nrm2(queue, n, x, incx, result);
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result) {
    blas_major::nrm2(queue, n, x, incx, result);
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, float c,
         float s) {
    blas_major::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, double c,
         double s) {
    blas_major::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, float c, float s) {
    blas_major::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, double c, double s) {
    blas_major::rot(queue, n, x, incx, y, incy, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
          sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s) {
    blas_major::rotg(queue, a, b, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
          sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s) {
    blas_major::rotg(queue, a, b, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
          sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
          sycl::buffer<std::complex<float>, 1>& s) {
    blas_major::rotg(queue, a, b, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
          sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
          sycl::buffer<std::complex<double>, 1>& s) {
    blas_major::rotg(queue, a, b, c, s);
}

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& param) {
    blas_major::rotm(queue, n, x, incx, y, incy, param);
}

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& param) {
    blas_major::rotm(queue, n, x, incx, y, incy, param);
}

void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
           sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param) {
    blas_major::rotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
           sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param) {
    blas_major::rotmg(queue, d1, d2, x1, y1, param);
}

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    blas_major::scal(queue, n, alpha, x, incx);
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy) {
    blas_major::swap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy) {
    blas_major::swap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    blas_major::swap(queue, n, x, incx, y, incy);
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    blas_major::swap(queue, n, x, incx, y, incy);
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamax(queue, n, x, incx, result);
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamax(queue, n, x, incx, result);
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamax(queue, n, x, incx, result);
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamax(queue, n, x, incx, result);
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamin(queue, n, x, incx, result);
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamin(queue, n, x, incx, result);
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamin(queue, n, x, incx, result);
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    blas_major::iamin(queue, n, x, incx, result);
}

// USM APIs

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::asum(queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::asum(queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::asum(queue, n, x, incx, result, dependencies);
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::asum(queue, n, x, incx, result, dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, float alpha, const float* x, std::int64_t incx,
                 float* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                 std::int64_t incx, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, float alpha, const float* x,
                  std::int64_t incx, float beta, float* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                  std::int64_t incx, double beta, double* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, std::int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, std::int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies);
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::copy(queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::copy(queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::copy(queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::copy(queue, n, x, incx, y, incy, dependencies);
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, float* result,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::dot(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                const double* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::dot(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event sdsdot(sycl::queue& queue, std::int64_t n, float sb, const float* x, std::int64_t incx,
                   const float* y, std::int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    return blas_major::sdsdot(queue, n, sb, x, incx, y, incy, result, dependencies);
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::dot(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::dotc(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::dotc(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::dotu(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::dotu(queue, n, x, incx, y, incy, result, dependencies);
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::nrm2(queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::nrm2(queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::nrm2(queue, n, x, incx, result, dependencies);
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::nrm2(queue, n, x, incx, result, dependencies);
}

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                std::complex<float>* y, std::int64_t incy, float c, float s,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                std::complex<double>* y, std::int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                std::int64_t incy, float c, float s, const std::vector<sycl::event>& dependencies) {
    return blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rot(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                std::int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies) {
    return blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::rotg(queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::rotg(queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b, float* c,
                 std::complex<float>* s, const std::vector<sycl::event>& dependencies) {
    return blas_major::rotg(queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b, double* c,
                 std::complex<double>* s, const std::vector<sycl::event>& dependencies) {
    return blas_major::rotg(queue, a, b, c, s, dependencies);
}

sycl::event rotm(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, float* param, const std::vector<sycl::event>& dependencies) {
    return blas_major::rotm(queue, n, x, incx, y, incy, param, dependencies);
}

sycl::event rotm(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, double* param, const std::vector<sycl::event>& dependencies) {
    return blas_major::rotm(queue, n, x, incx, y, incy, param, dependencies);
}

sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, float y1, float* param,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::rotmg(queue, d1, d2, x1, y1, param, dependencies);
}

sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1, double y1, double* param,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::rotmg(queue, d1, d2, x1, y1, param, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    return blas_major::scal(queue, n, alpha, x, incx, dependencies);
}

sycl::event swap(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::swap(queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    return blas_major::swap(queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::swap(queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    return blas_major::swap(queue, n, x, incx, y, incy, dependencies);
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::iamax(queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::iamax(queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::iamax(queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::iamax(queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::iamin(queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    return blas_major::iamin(queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::iamin(queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    return blas_major::iamin(queue, n, x, incx, result, dependencies);
}
