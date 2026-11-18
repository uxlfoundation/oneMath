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

// Workaround for oneMKL 2026.0 SSE2 bug: iamax/iamin return incorrect indices with incx > 1
// Only apply on MKLCPU backend where SSE2 fallback is used
#if defined(ONEMATH_MKLCPU_BACKEND) && defined(__INTEL_MKL__) && __INTEL_MKL__ == 2026
#define APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
#pragma message("oneMKL 2026.0 iamin/iamax workaround ENABLED for MKLCPU backend")

namespace {

// Helper: compute absolute value (magnitude) for iamin/iamax
template <typename T>
inline T abs_value(T val) {
    return sycl::fabs(val);
}

template <typename T>
inline T abs_value(std::complex<T> val) {
    return sycl::fabs(val.real()) + sycl::fabs(val.imag());
}

// SYCL reference implementation of iamin for buffer API
// Used as workaround for oneMKL 2026.0 SSE2 bug when |incx| > 1
template <typename T>
void sycl_iamin_impl(sycl::queue& queue, std::int64_t n, sycl::buffer<T, 1>& x, std::int64_t incx,
                     sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);

        cgh.single_task([=]() {
            if (n < 1 || incx < 1) {
                res_acc[0] = 0;
                return;
            }

            std::int64_t min_idx = 0;
            auto min_val = abs_value(x_acc[0]);

            if (sycl::isnan(min_val)) {
                res_acc[0] = (base == oneapi::math::index_base::zero ? 0 : 1);
                return;
            }

            std::int64_t abs_incx = (incx > 0) ? incx : -incx;
            for (std::int64_t logical_i = 1; logical_i < n; ++logical_i) {
                std::int64_t i = logical_i * abs_incx;
                auto curr_val = abs_value(x_acc[i]);

                if (sycl::isnan(curr_val)) {
                    res_acc[0] = logical_i + (base == oneapi::math::index_base::zero ? 0 : 1);
                    return;
                }

                if (curr_val < min_val) {
                    min_idx = logical_i;
                    min_val = curr_val;
                }
            }

            res_acc[0] = min_idx + (base == oneapi::math::index_base::zero ? 0 : 1);
        });
    });
}

// SYCL reference implementation of iamax for buffer API
template <typename T>
void sycl_iamax_impl(sycl::queue& queue, std::int64_t n, sycl::buffer<T, 1>& x, std::int64_t incx,
                     sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);

        cgh.single_task([=]() {
            if (n < 1 || incx < 1) {
                res_acc[0] = 0;
                return;
            }

            std::int64_t max_idx = 0;
            auto max_val = abs_value(x_acc[0]);

            if (sycl::isnan(max_val)) {
                res_acc[0] = (base == oneapi::math::index_base::zero ? 0 : 1);
                return;
            }

            std::int64_t abs_incx = (incx > 0) ? incx : -incx;
            for (std::int64_t logical_i = 1; logical_i < n; ++logical_i) {
                std::int64_t i = logical_i * abs_incx;
                auto curr_val = abs_value(x_acc[i]);

                if (sycl::isnan(curr_val)) {
                    res_acc[0] = logical_i + (base == oneapi::math::index_base::zero ? 0 : 1);
                    return;
                }

                if (curr_val > max_val) {
                    max_idx = logical_i;
                    max_val = curr_val;
                }
            }

            res_acc[0] = max_idx + (base == oneapi::math::index_base::zero ? 0 : 1);
        });
    });
}

// SYCL reference implementation of iamin for USM API
template <typename T>
sycl::event sycl_iamin_usm_impl(sycl::queue& queue, std::int64_t n, const T* x, std::int64_t incx,
                                std::int64_t* result, oneapi::math::index_base base,
                                const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.single_task([=]() {
            if (n < 1 || incx < 1) {
                *result = 0;
                return;
            }

            std::int64_t min_idx = 0;
            auto min_val = abs_value(x[0]);

            if (sycl::isnan(min_val)) {
                *result = (base == oneapi::math::index_base::zero ? 0 : 1);
                return;
            }

            std::int64_t abs_incx = (incx > 0) ? incx : -incx;
            for (std::int64_t logical_i = 1; logical_i < n; ++logical_i) {
                std::int64_t i = logical_i * abs_incx;
                auto curr_val = abs_value(x[i]);

                if (sycl::isnan(curr_val)) {
                    *result = logical_i + (base == oneapi::math::index_base::zero ? 0 : 1);
                    return;
                }

                if (curr_val < min_val) {
                    min_idx = logical_i;
                    min_val = curr_val;
                }
            }

            *result = min_idx + (base == oneapi::math::index_base::zero ? 0 : 1);
        });
    });
}

// SYCL reference implementation of iamax for USM API
template <typename T>
sycl::event sycl_iamax_usm_impl(sycl::queue& queue, std::int64_t n, const T* x, std::int64_t incx,
                                std::int64_t* result, oneapi::math::index_base base,
                                const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.single_task([=]() {
            if (n < 1 || incx < 1) {
                *result = 0;
                return;
            }

            std::int64_t max_idx = 0;
            auto max_val = abs_value(x[0]);

            if (sycl::isnan(max_val)) {
                *result = (base == oneapi::math::index_base::zero ? 0 : 1);
                return;
            }

            std::int64_t abs_incx = (incx > 0) ? incx : -incx;
            for (std::int64_t logical_i = 1; logical_i < n; ++logical_i) {
                std::int64_t i = logical_i * abs_incx;
                auto curr_val = abs_value(x[i]);

                if (sycl::isnan(curr_val)) {
                    *result = logical_i + (base == oneapi::math::index_base::zero ? 0 : 1);
                    return;
                }

                if (curr_val > max_val) {
                    max_idx = logical_i;
                    max_val = curr_val;
                }
            }

            *result = max_idx + (base == oneapi::math::index_base::zero ? 0 : 1);
        });
    });
}

} // anonymous namespace
#endif

// Buffer APIs

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::asum(queue, n, x, incx, result));
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::asum(queue, n, x, incx, result));
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::asum(queue, n, x, incx, result));
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::asum(queue, n, x, incx, result));
}

void axpy(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpy(queue, n, alpha, x, incx, y, incy));
}

void axpy(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpy(queue, n, alpha, x, incx, y, incy));
}

void axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpy(queue, n, alpha, x, incx, y, incy));
}

void axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpy(queue, n, alpha, x, incx, y, incy));
}

void axpby(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
           std::int64_t incx, float beta, sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy));
}

void axpby(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
           std::int64_t incx, double beta, sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy));
}

void axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy));
}

void axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy));
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::copy(queue, n, x, incx, y, incy));
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::copy(queue, n, x, incx, y, incy));
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::copy(queue, n, x, incx, y, incy));
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::copy(queue, n, x, incx, y, incy));
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dot(queue, n, x, incx, y, incy, result));
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dot(queue, n, x, incx, y, incy, result));
}

void sdsdot(sycl::queue& queue, std::int64_t n, float sb, sycl::buffer<float, 1>& x,
            std::int64_t incx, sycl::buffer<float, 1>& y, std::int64_t incy,
            sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::sdsdot(queue, n, sb, x, incx, y, incy, result));
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dot(queue, n, x, incx, y, incy, result));
}

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dotc(queue, n, x, incx, y, incy, result));
}

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dotc(queue, n, x, incx, y, incy, result));
}

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dotu(queue, n, x, incx, y, incy, result));
}

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::dotu(queue, n, x, incx, y, incy, result));
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::nrm2(queue, n, x, incx, result));
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::nrm2(queue, n, x, incx, result));
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::nrm2(queue, n, x, incx, result));
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& result) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::nrm2(queue, n, x, incx, result));
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy, float c,
         float s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rot(queue, n, x, incx, y, incy, c, s));
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy, double c,
         double s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rot(queue, n, x, incx, y, incy, c, s));
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, float c, float s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rot(queue, n, x, incx, y, incy, c, s));
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
         sycl::buffer<double, 1>& y, std::int64_t incy, double c, double s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rot(queue, n, x, incx, y, incy, c, s));
}

void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
          sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotg(queue, a, b, c, s));
}

void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
          sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotg(queue, a, b, c, s));
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
          sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
          sycl::buffer<std::complex<float>, 1>& s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotg(queue, a, b, c, s));
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
          sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
          sycl::buffer<std::complex<double>, 1>& s) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotg(queue, a, b, c, s));
}

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<float, 1>& param) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotm(queue, n, x, incx, y, incy, param));
}

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& param) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotm(queue, n, x, incx, y, incy, param));
}

void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
           sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotmg(queue, d1, d2, x1, y1, param));
}

void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
           sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::rotmg(queue, d1, d2, x1, y1, param));
}

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<float, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void scal(sycl::queue& queue, std::int64_t n, double alpha, sycl::buffer<double, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void scal(sycl::queue& queue, std::int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void scal(sycl::queue& queue, std::int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1>& x, std::int64_t incx) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::scal(queue, n, alpha, x, incx));
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
          sycl::buffer<float, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::swap(queue, n, x, incx, y, incy));
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
          sycl::buffer<double, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::swap(queue, n, x, incx, y, incy));
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<float>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::swap(queue, n, x, incx, y, incy));
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<double>, 1>& y, std::int64_t incy) {
    RETHROW_ONEMKL_EXCEPTIONS(blas_major::swap(queue, n, x, incx, y, incy));
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamax_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamax(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamax_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamax(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamax_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamax(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamax_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamax(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamin_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamin(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<double, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result, oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamin_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamin(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<float>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamin_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamin(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<double>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result,
           oneapi::math::index_base base) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        sycl_iamin_impl(queue, n, x, incx, result, base);
    }
    else {
#endif
        RETHROW_ONEMKL_EXCEPTIONS(
            blas_major::iamin(queue, n, x, incx, result, detail::get_onemkl_index_base(base)));
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    }
#endif
}

// USM APIs

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::asum(queue, n, x, incx, result, dependencies));
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::asum(queue, n, x, incx, result, dependencies));
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::asum(queue, n, x, incx, result, dependencies));
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::asum(queue, n, x, incx, result, dependencies));
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, float alpha, const float* x, std::int64_t incx,
                 float* y, std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies));
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                 std::int64_t incx, double* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies));
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, std::int64_t incx, std::complex<float>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies));
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, std::int64_t incx, std::complex<double>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpy(queue, n, alpha, x, incx, y, incy, dependencies));
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, float alpha, const float* x,
                  std::int64_t incx, float beta, float* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies));
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, double alpha, const double* x,
                  std::int64_t incx, double beta, double* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies));
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, std::int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies));
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, std::int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies));
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy(queue, n, x, incx, y, incy, dependencies));
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy(queue, n, x, incx, y, incy, dependencies));
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy(queue, n, x, incx, y, incy, dependencies));
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::copy(queue, n, x, incx, y, incy, dependencies));
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, float* result,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dot(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                const double* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dot(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event sdsdot(sycl::queue& queue, std::int64_t n, float sb, const float* x, std::int64_t incx,
                   const float* y, std::int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::sdsdot(queue, n, sb, x, incx, y, incy, result, dependencies));
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dot(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dotc(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dotc(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, const std::complex<float>* y, std::int64_t incy,
                 std::complex<float>* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dotu(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, const std::complex<double>* y, std::int64_t incy,
                 std::complex<double>* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::dotu(queue, n, x, incx, y, incy, result, dependencies));
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                 std::int64_t incx, float* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::nrm2(queue, n, x, incx, result, dependencies));
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                 std::int64_t incx, double* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::nrm2(queue, n, x, incx, result, dependencies));
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::nrm2(queue, n, x, incx, result, dependencies));
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::nrm2(queue, n, x, incx, result, dependencies));
}

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                std::complex<float>* y, std::int64_t incy, float c, float s,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies));
}

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                std::complex<double>* y, std::int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies));
}

sycl::event rot(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                std::int64_t incy, float c, float s, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies));
}

sycl::event rot(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                std::int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rot(queue, n, x, incx, y, incy, c, s, dependencies));
}

sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotg(queue, a, b, c, s, dependencies));
}

sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotg(queue, a, b, c, s, dependencies));
}

sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b, float* c,
                 std::complex<float>* s, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotg(queue, a, b, c, s, dependencies));
}

sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b, double* c,
                 std::complex<double>* s, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotg(queue, a, b, c, s, dependencies));
}

sycl::event rotm(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const float* param,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::rotm(queue, n, x, incx, y, incy, param, dependencies));
}

sycl::event rotm(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const double* param,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(
        blas_major::rotm(queue, n, x, incx, y, incy, param, dependencies));
}

sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, float y1, float* param,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotmg(queue, d1, d2, x1, y1, param, dependencies));
}

sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1, double y1, double* param,
                  const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::rotmg(queue, d1, d2, x1, y1, param, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, float* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, double* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<float> alpha,
                 std::complex<float>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<double> alpha,
                 std::complex<double>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, float alpha, std::complex<float>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event scal(sycl::queue& queue, std::int64_t n, double alpha, std::complex<double>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::scal(queue, n, alpha, x, incx, dependencies));
}

sycl::event swap(sycl::queue& queue, std::int64_t n, float* x, std::int64_t incx, float* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::swap(queue, n, x, incx, y, incy, dependencies));
}

sycl::event swap(sycl::queue& queue, std::int64_t n, double* x, std::int64_t incx, double* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::swap(queue, n, x, incx, y, incy, dependencies));
}

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<float>* x, std::int64_t incx,
                 std::complex<float>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::swap(queue, n, x, incx, y, incy, dependencies));
}

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<double>* x, std::int64_t incx,
                 std::complex<double>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::swap(queue, n, x, incx, y, incy, dependencies));
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamax_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamax(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamax_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamax(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamax_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamax(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamax_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamax(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                  std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamin_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamin(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const double* x, std::int64_t incx,
                  std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamin_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamin(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<float>* x,
                  std::int64_t incx, std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamin_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamin(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<double>* x,
                  std::int64_t incx, std::int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
#ifdef APPLY_ONEMKL_2026_IAMIN_IAMAX_WORKAROUND
    if (std::abs(incx) > 1) {
        return sycl_iamin_usm_impl(queue, n, x, incx, result, base, dependencies);
    }
#endif
    RETHROW_ONEMKL_EXCEPTIONS_RET(blas_major::iamin(
        queue, n, x, incx, result, detail::get_onemkl_index_base(base), dependencies));
}
