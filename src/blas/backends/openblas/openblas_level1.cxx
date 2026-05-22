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
#include <memory>
// Buffer APIs

void asum(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_sasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sasum((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void asum(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_dasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dasum((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void asum(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_scasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_scasum((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void asum(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_dzasum>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dzasum((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void axpy(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_saxpy>(cgh, [=]() {
            ::cblas_saxpy((const int)n, (const float)alpha, accessor_x.GET_MULTI_PTR,
                          (const int)incx, accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void axpy(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_daxpy>(cgh, [=]() {
            ::cblas_daxpy((const int)n, (const double)alpha, accessor_x.GET_MULTI_PTR,
                          (const int)incx, accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void axpy(sycl::queue& queue, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_caxpy>(cgh, [=]() {
            ::cblas_caxpy((const int)n, (const void*)&alpha, accessor_x.GET_MULTI_PTR,
                          (const int)incx, accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void axpy(sycl::queue& queue, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zaxpy>(cgh, [=]() {
            ::cblas_zaxpy((const int)n, (const void*)&alpha, accessor_x.GET_MULTI_PTR,
                          (const int)incx, accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_saxpby>(cgh, [=]() {
            ::cblas_saxpby((blasint)n, alpha, accessor_x.GET_MULTI_PTR, (blasint)incx, beta,
                           accessor_y.GET_MULTI_PTR, (blasint)incy);
        });
    });
}

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_daxpby>(cgh, [=]() {
            ::cblas_daxpby((blasint)n, alpha, accessor_x.GET_MULTI_PTR, (blasint)incx, beta,
                           accessor_y.GET_MULTI_PTR, (blasint)incy);
        });
    });
}

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_caxpby>(cgh, [=]() {
            ::cblas_caxpby((blasint)n, static_cast<const void*>(&alpha), accessor_x.GET_MULTI_PTR,
                           (blasint)incx, static_cast<const void*>(&beta), accessor_y.GET_MULTI_PTR,
                           (blasint)incy);
        });
    });
}

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_zaxpby>(cgh, [=]() {
            ::cblas_zaxpby((blasint)n, static_cast<const void*>(&alpha), accessor_x.GET_MULTI_PTR,
                           (blasint)incx, static_cast<const void*>(&beta), accessor_y.GET_MULTI_PTR,
                           (blasint)incy);
        });
    });
}

void copy(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_scopy>(cgh, [=]() {
            ::cblas_scopy((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void copy(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dcopy>(cgh, [=]() {
            ::cblas_dcopy((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void copy(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_ccopy>(cgh, [=]() {
            ::cblas_ccopy((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void copy(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zcopy>(cgh, [=]() {
            ::cblas_zcopy((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_sdot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sdot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                             accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void dot(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
         sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_ddot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_ddot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                             accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_dsdot>(cgh, [=]() {
            double sum = 0.0;
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += static_cast<double>(accessor_x.GET_MULTI_PTR[ix]) *
                       static_cast<double>(accessor_y.GET_MULTI_PTR[iy]);
                ix += incx;
                iy += incy;
            }

            accessor_result[0] = sum;
        });
    });
}

void dotc(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cdotc>(cgh, [=]() {
            ::cblas_cdotc_sub((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                              accessor_y.GET_MULTI_PTR, (const int)incy,
                              accessor_result.GET_MULTI_PTR);
        });
    });
}

void dotc(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zdotc>(cgh, [=]() {
            ::cblas_zdotc_sub((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                              accessor_y.GET_MULTI_PTR, (const int)incy,
                              accessor_result.GET_MULTI_PTR);
        });
    });
}

void dotu(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cdotu>(cgh, [=]() {
            ::cblas_cdotu_sub((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                              accessor_y.GET_MULTI_PTR, (const int)incy,
                              accessor_result.GET_MULTI_PTR);
        });
    });
}

void dotu(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zdotu>(cgh, [=]() {
            ::cblas_zdotu_sub((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                              accessor_y.GET_MULTI_PTR, (const int)incy,
                              accessor_result.GET_MULTI_PTR);
        });
    });
}

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_isamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_isamin((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_idamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_idamin((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_icamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_icamin((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_izamin>(cgh, [=]() {
            accessor_result[0] = ::cblas_izamin((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_isamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_isamax((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_idamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_idamax((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_icamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_icamax((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_izamax>(cgh, [=]() {
            accessor_result[0] = ::cblas_izamax((int)n, accessor_x.GET_MULTI_PTR, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_snrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_snrm2((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_dnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dnrm2((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_scnrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_scnrm2((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_dznrm2>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_dznrm2((const int)n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

void rot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, float c, float s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_srot>(cgh, [=]() {
            ::cblas_srot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_y.GET_MULTI_PTR, (const int)incy, (const float)c, (const float)s);
        });
    });
}

void rot(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
         sycl::buffer<double, 1>& y, int64_t incy, double c, double s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_drot>(cgh, [=]() {
            ::cblas_drot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                         accessor_y.GET_MULTI_PTR, (const int)incy, (const float)c, (const float)s);
        });
    });
}

void rot(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& y, int64_t incy, float c, float s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_csrot>(cgh, [=]() {
            ::cblas_csrot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, (const float)c,
                          (const float)s);
        });
    });
}

void rot(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& y, int64_t incy, double c, double s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zdrot>(cgh, [=]() {
            ::cblas_zdrot((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, (const double)c,
                          (const double)s);
        });
    });
}

void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
          sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_srotg>(cgh, [=]() {
            ::cblas_srotg(accessor_a.GET_MULTI_PTR, accessor_b.GET_MULTI_PTR,
                          accessor_c.GET_MULTI_PTR, accessor_s.GET_MULTI_PTR);
        });
    });
}

void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
          sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_drotg>(cgh, [=]() {
            ::cblas_drotg(accessor_a.GET_MULTI_PTR, accessor_b.GET_MULTI_PTR,
                          accessor_c.GET_MULTI_PTR, accessor_s.GET_MULTI_PTR);
        });
    });
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
          sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
          sycl::buffer<std::complex<float>, 1>& s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_crotg>(cgh, [=]() {
            ::cblas_crotg(
                const_cast<void*>(reinterpret_cast<const void*>(accessor_a.GET_MULTI_PTR)),
                const_cast<void*>(reinterpret_cast<const void*>(accessor_b.GET_MULTI_PTR)),
                accessor_c.GET_MULTI_PTR,
                const_cast<void*>(reinterpret_cast<const void*>(accessor_s.GET_MULTI_PTR)));
        });
    });
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
          sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
          sycl::buffer<std::complex<double>, 1>& s) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.get_access<sycl::access::mode::read>(cgh);
        auto accessor_c = c.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.get_access<sycl::access::mode::read_write>(cgh);

        host_task<class openblas_zrotg>(cgh, [=]() {
            void* a_ptr =
                const_cast<void*>(reinterpret_cast<const void*>(accessor_a.GET_MULTI_PTR));

            void* b_ptr =
                const_cast<void*>(reinterpret_cast<const void*>(accessor_b.GET_MULTI_PTR));

            double* c_ptr = accessor_c.GET_MULTI_PTR;

            void* s_ptr =
                const_cast<void*>(reinterpret_cast<const void*>(accessor_s.GET_MULTI_PTR));

            ::cblas_zrotg(a_ptr, b_ptr, c_ptr, s_ptr);
        });
    });
}

void rotm(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& param) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<sycl::access::mode::read>(cgh);
        host_task<class openblas_srotm>(cgh, [=]() {
            ::cblas_srotm((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_param.GET_MULTI_PTR);
        });
    });
}

void rotm(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& param) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<sycl::access::mode::read>(cgh);
        host_task<class openblas_drotm>(cgh, [=]() {
            ::cblas_drotm((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy, accessor_param.GET_MULTI_PTR);
        });
    });
}

void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
           sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_d1 = d1.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_d2 = d2.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_x1 = x1.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_srotmg>(cgh, [=]() {
            ::cblas_srotmg(accessor_d1.GET_MULTI_PTR, accessor_d2.GET_MULTI_PTR,
                           accessor_x1.GET_MULTI_PTR, (float)y1, accessor_param.GET_MULTI_PTR);
        });
    });
}

void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
           sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_d1 = d1.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_d2 = d2.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_x1 = x1.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_drotmg>(cgh, [=]() {
            ::cblas_drotmg(accessor_d1.GET_MULTI_PTR, accessor_d2.GET_MULTI_PTR,
                           accessor_x1.GET_MULTI_PTR, (double)y1, accessor_param.GET_MULTI_PTR);
        });
    });
}

void scal(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sscal>(cgh, [=]() {
            ::cblas_sscal((const int)n, (const float)alpha, accessor_x.GET_MULTI_PTR,
                          (const int)std::abs(incx));
        });
    });
}

void scal(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dscal>(cgh, [=]() {
            ::cblas_dscal((const int)n, (const double)alpha, accessor_x.GET_MULTI_PTR,
                          (const int)std::abs(incx));
        });
    });
}

void scal(sycl::queue& queue, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cscal>(cgh, [=]() {
            ::cblas_cscal((const int)n, (const void*)&alpha, accessor_x.GET_MULTI_PTR,
                          (const int)std::abs(incx));
        });
    });
}

void scal(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_csscal>(cgh, [=]() {
            ::cblas_csscal((const int)n, (const float)alpha, accessor_x.GET_MULTI_PTR,
                           (const int)std::abs(incx));
        });
    });
}

void scal(sycl::queue& queue, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zscal>(cgh, [=]() {
            ::cblas_zscal((const int)n, (const void*)&alpha, accessor_x.GET_MULTI_PTR,
                          (const int)std::abs(incx));
        });
    });
}

void scal(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zdscal>(cgh, [=]() {
            ::cblas_zdscal((const int)n, (const double)alpha, accessor_x.GET_MULTI_PTR,
                           (const int)std::abs(incx));
        });
    });
}

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
        host_task<class openblas_sdsdot>(cgh, [=]() {
            accessor_result[0] =
                ::cblas_sdsdot((const int)n, (const float)sb, accessor_x.GET_MULTI_PTR,
                               (const int)incx, accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void swap(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_sswap>(cgh, [=]() {
            ::cblas_sswap((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void swap(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_dswap>(cgh, [=]() {
            ::cblas_dswap((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void swap(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_cswap>(cgh, [=]() {
            ::cblas_cswap((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

void swap(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.get_access<sycl::access::mode::read_write>(cgh);
        host_task<class openblas_zswap>(cgh, [=]() {
            ::cblas_zswap((const int)n, accessor_x.GET_MULTI_PTR, (const int)incx,
                          accessor_y.GET_MULTI_PTR, (const int)incy);
        });
    });
}

// USM APIs

sycl::event asum(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sasum_usm>(
            cgh, [=]() { result[0] = ::cblas_sasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event asum(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dasum_usm>(
            cgh, [=]() { result[0] = ::cblas_dasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event asum(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_scasum_usm>(
            cgh, [=]() { result[0] = ::cblas_scasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event asum(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dzasum_usm>(
            cgh, [=]() { result[0] = ::cblas_dzasum((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event axpy(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_saxpy_usm>(cgh, [=]() {
            ::cblas_saxpy((const int)n, (const float)alpha, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

sycl::event axpy(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_daxpy_usm>(cgh, [=]() {
            ::cblas_daxpy((const int)n, (const double)alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event axpy(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_caxpy_usm>(cgh, [=]() {
            ::cblas_caxpy((const int)n, (const void*)&alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event axpy(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zaxpy_usm>(cgh, [=]() {
            ::cblas_zaxpy((const int)n, (const void*)&alpha, x, (const int)incx, y,
                          (const int)incy);
        });
    });
    return done;
}

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
#ifdef COLUMN_MAJOR
    throw unimplemented("blas", "axpby", "for column_major layout");
#endif
#ifdef ROW_MAJOR
    throw unimplemented("blas", "axpby", "for row_major layout");
#endif
}

sycl::event copy(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_scopy_usm>(
            cgh, [=]() { ::cblas_scopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event copy(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dcopy_usm>(
            cgh, [=]() { ::cblas_dcopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event copy(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ccopy_usm>(
            cgh, [=]() { ::cblas_ccopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event copy(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zcopy_usm>(
            cgh, [=]() { ::cblas_zcopy((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, float* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sdot_usm>(cgh, [=]() {
            result[0] = ::cblas_sdot((const int)n, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

sycl::event dot(sycl::queue& queue, int64_t n, const double* x, int64_t incx, const double* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_ddot_usm>(cgh, [=]() {
            result[0] = ::cblas_ddot((const int)n, x, (const int)incx, y, (const int)incy);
        });
    });
    return done;
}

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        for (const auto& dep : dependencies) {
            cgh.depends_on(dep);
        }

        host_task<class openblas_sdot_usm_compute>(cgh, [=]() {
            double sum = 0.0;
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += static_cast<double>(x[ix]) * static_cast<double>(y[iy]);
                ix += incx;
                iy += incy;
            }

            *result = sum;
        });
    });
}

sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 const std::complex<float>* y, int64_t incy, std::complex<float>* result,
                 const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cdotc_usm>(cgh, [=]() {
            std::complex<float> sum = { 0.0f, 0.0f };
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += std::conj(x[ix]) * y[iy];
                ix += incx;
                iy += incy;
            }

            *result = sum;
        });
    });
}

sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 const std::complex<double>* y, int64_t incy, std::complex<double>* result,
                 const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zdotc_usm>(cgh, [=]() {
            std::complex<double> sum = { 0.0, 0.0 };
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += std::conj(x[ix]) * y[iy];
                ix += incx;
                iy += incy;
            }

            *result = sum;
        });
    });
}

sycl::event dotu(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 const std::complex<float>* y, int64_t incy, std::complex<float>* result,
                 const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        for (const auto& dep : dependencies) {
            cgh.depends_on(dep);
        }

        host_task<class openblas_cdotu_usm>(cgh, [=]() {
            std::complex<float> sum = { 0.0f, 0.0f };
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += x[ix] * y[iy];
                ix += incx;
                iy += incy;
            }

            *result = sum;
        });
    });
}
sycl::event dotu(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 const std::complex<double>* y, int64_t incy, std::complex<double>* result,
                 const std::vector<sycl::event>& dependencies) {
    return queue.submit([&](sycl::handler& cgh) {
        for (const auto& dep : dependencies) {
            cgh.depends_on(dep);
        }

        host_task<class openblas_zdotu_usm>(cgh, [=]() {
            std::complex<double> sum = { 0.0, 0.0 };
            int64_t ix = (incx > 0) ? 0 : (1 - n) * incx;
            int64_t iy = (incy > 0) ? 0 : (1 - n) * incy;

            for (int64_t i = 0; i < n; ++i) {
                sum += x[ix] * y[iy];
                ix += incx;
                iy += incy;
            }

            *result = sum;
        });
    });
}

sycl::event iamin(sycl::queue& queue, int64_t n, const float* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_isamin_usm>(cgh, [=]() {
            result[0] = ::cblas_isamin((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamin(sycl::queue& queue, int64_t n, const double* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_idamin_usm>(cgh, [=]() {
            result[0] = ::cblas_idamin((const int)n, x, (const int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamin(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                  int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_icamin_usm>(cgh, [=]() {
            result[0] = ::cblas_icamin((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamin(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                  int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_izamin_usm>(cgh, [=]() {
            result[0] = ::cblas_izamin((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamax(sycl::queue& queue, int64_t n, const float* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_isamax_usm>(cgh, [=]() {
            result[0] = ::cblas_isamax((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamax(sycl::queue& queue, int64_t n, const double* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_idamax_usm>(cgh, [=]() {
            result[0] = ::cblas_idamax((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamax(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                  int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_icamax_usm>(cgh, [=]() {
            result[0] = ::cblas_icamax((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event iamax(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                  int64_t* result, oneapi::math::index_base base,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_izamax_usm>(cgh, [=]() {
            result[0] = ::cblas_izamax((int)n, x, (int)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

sycl::event nrm2(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_snrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_snrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event nrm2(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dnrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_dnrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event nrm2(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_scnrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_scnrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event nrm2(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dznrm2_usm>(
            cgh, [=]() { result[0] = ::cblas_dznrm2((const int)n, x, (const int)std::abs(incx)); });
    });
    return done;
}

sycl::event rot(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                float c, float s, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_srot_usm>(cgh, [=]() {
            ::cblas_srot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                         (const float)s);
        });
    });
    return done;
}

sycl::event rot(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                double c, double s, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_drot_usm>(cgh, [=]() {
            ::cblas_drot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                         (const float)s);
        });
    });
    return done;
}

sycl::event rot(sycl::queue& queue, int64_t n, std::complex<float>* x, int64_t incx,
                std::complex<float>* y, int64_t incy, float c, float s,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_csrot_usm>(cgh, [=]() {
            ::cblas_csrot((const int)n, x, (const int)incx, y, (const int)incy, (const float)c,
                          (const float)s);
        });
    });
    return done;
}

sycl::event rot(sycl::queue& queue, int64_t n, std::complex<double>* x, int64_t incx,
                std::complex<double>* y, int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zdrot_usm>(cgh, [=]() {
            ::cblas_zdrot((const int)n, x, (const int)incx, y, (const int)incy, (const double)c,
                          (const double)s);
        });
    });
    return done;
}

sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_srotg_usm>(cgh, [=]() { ::cblas_srotg(a, b, c, s); });
    });
    return done;
}

sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_drotg_usm>(cgh, [=]() { ::cblas_drotg(a, b, c, s); });
    });
    return done;
}

sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b, float* c,
                 std::complex<float>* s, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_crotg_usm>(cgh, [=]() { ::cblas_crotg(a, b, c, s); });
    });
    return done;
}

sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b, double* c,
                 std::complex<double>* s, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zrotg_usm>(cgh, [=]() { ::cblas_zrotg(a, b, c, s); });
    });
    return done;
}

sycl::event rotm(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                 float* param, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_srotm_usm>(cgh, [=]() {
            ::cblas_srotm((const int)n, x, (const int)incx, y, (const int)incy, param);
        });
    });
    return done;
}

sycl::event rotm(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                 double* param, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_drotm_usm>(cgh, [=]() {
            ::cblas_drotm((const int)n, x, (const int)incx, y, (const int)incy, param);
        });
    });
    return done;
}

sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, float y1, float* param,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_srotmg_usm>(
            cgh, [=]() { ::cblas_srotmg(d1, d2, x1, (float)y1, param); });
    });
    return done;
}

sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1, double y1, double* param,
                  const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_drotmg_usm>(
            cgh, [=]() { ::cblas_drotmg(d1, d2, x1, (double)y1, param); });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, float alpha, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sscal_usm>(cgh, [=]() {
            ::cblas_sscal((const int)n, (const float)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, double alpha, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dscal_usm>(cgh, [=]() {
            ::cblas_dscal((const int)n, (const double)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, std::complex<float> alpha, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cscal_usm>(cgh, [=]() {
            ::cblas_cscal((const int)n, (const void*)&alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, float alpha, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_csscal_usm>(cgh, [=]() {
            ::cblas_csscal((const int)n, (const float)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, std::complex<double> alpha, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zscal_usm>(cgh, [=]() {
            ::cblas_zscal((const int)n, (const void*)&alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event scal(sycl::queue& queue, int64_t n, double alpha, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zdscal_usm>(cgh, [=]() {
            ::cblas_zdscal((const int)n, (const double)alpha, x, (const int)std::abs(incx));
        });
    });
    return done;
}

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sdsdot_usm>(cgh, [=]() {
            result[0] = ::cblas_sdsdot((const int)n, (const float)sb, x, (const int)incx, y,
                                       (const int)incy);
        });
    });
    return done;
}

sycl::event swap(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_sswap_usm>(
            cgh, [=]() { ::cblas_sswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event swap(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_dswap_usm>(
            cgh, [=]() { ::cblas_dswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event swap(sycl::queue& queue, int64_t n, std::complex<float>* x, int64_t incx,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_cswap_usm>(
            cgh, [=]() { ::cblas_cswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}

sycl::event swap(sycl::queue& queue, int64_t n, std::complex<double>* x, int64_t incx,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class openblas_zswap_usm>(
            cgh, [=]() { ::cblas_zswap((const int)n, x, (const int)incx, y, (const int)incy); });
    });
    return done;
}
