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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "netlib_common.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/blas/detail/netlib/onemkl_blas_netlib.hpp"

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

int cblas_isamin(int n, const float* x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < n; ++logical_i) {
        int i = logical_i * std::abs(incx);
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

int cblas_idamin(int n, const double* x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < n; ++logical_i) {
        int i = logical_i * std::abs(incx);
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

int cblas_icamin(int n, const std::complex<float>* x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < n; ++logical_i) {
        int i = logical_i * std::abs(incx);
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

int cblas_izamin(int n, const std::complex<double>* x, int incx) {
    if (n < 1 || incx < 1) {
        return 0;
    }
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < n; ++logical_i) {
        int i = logical_i * std::abs(incx);
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

void cblas_csrot(const int n, std::complex<float>* cx, const int incx, std::complex<float>* cy,
                 const int incy, const float c, const float s) {
    if (n < 1)
        return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            std::complex<float> ctemp = c * cx[i] + s * cy[i];
            cy[i] = c * cy[i] - s * cx[i];
            cx[i] = ctemp;
        }
    }
    else {
        int ix = 0, iy = 0;
        if (incx < 0)
            ix = (-n + 1) * incx;
        if (incy < 0)
            iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            std::complex<float> ctemp = c * cx[ix] + s * cy[iy];
            cy[iy] = c * cy[iy] - s * cx[ix];
            cx[ix] = ctemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

void cblas_zdrot(const int n, std::complex<double>* zx, const int incx, std::complex<double>* zy,
                 const int incy, const double c, const double s) {
    if (n < 1)
        return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            std::complex<double> ctemp = c * zx[i] + s * zy[i];
            zy[i] = c * zy[i] - s * zx[i];
            zx[i] = ctemp;
        }
    }
    else {
        int ix = 0, iy = 0;
        if (incx < 0)
            ix = (-n + 1) * incx;
        if (incy < 0)
            iy = (-n + 1) * incy;
        for (int i = 0; i < n; i++) {
            std::complex<double> ctemp = c * zx[ix] + s * zy[iy];
            zy[iy] = c * zy[iy] - s * zx[ix];
            zx[ix] = ctemp;
            ix = ix + incx;
            iy = iy + incy;
        }
    }
}

void cblas_crotg(std::complex<float>* ca, const std::complex<float>* cb, float* c,
                 std::complex<float>* s) {
    if (std::abs(ca[0]) == 0) {
        c[0] = 0.0;
        s[0] = std::complex<float>(1.0, 0.0);
        ca[0] = cb[0];
    }
    else {
        float scale = std::abs(ca[0]) + std::abs(cb[0]);
        float norm = scale * std::sqrt(std::pow(std::abs(ca[0] / scale), 2) +
                                       std::pow(std::abs(cb[0] / scale), 2));
        std::complex<float> alpha = ca[0] / std::abs(ca[0]);
        c[0] = std::abs(ca[0]) / norm;
        s[0] = alpha * std::conj(cb[0]) / norm;
        ca[0] = alpha * norm;
    }
}

void cblas_zrotg(std::complex<double>* ca, const std::complex<double>* cb, double* c,
                 std::complex<double>* s) {
    if (std::abs(ca[0]) == 0) {
        c[0] = 0.0;
        s[0] = std::complex<double>(1.0, 0.0);
        ca[0] = cb[0];
    }
    else {
        double scale = std::abs(ca[0]) + std::abs(cb[0]);
        double norm = scale * std::sqrt(std::pow(std::abs(ca[0] / scale), 2) +
                                        std::pow(std::abs(cb[0] / scale), 2));
        std::complex<double> alpha = ca[0] / std::abs(ca[0]);
        c[0] = std::abs(ca[0]) / norm;
        s[0] = alpha * std::conj(cb[0]) / norm;
        ca[0] = alpha * norm;
    }
}

namespace oneapi {
namespace mkl {
namespace blas {
namespace netlib {
namespace column_major {

#define COLUMN_MAJOR
#include "netlib_level1.cxx"
#undef COLUMN_MAJOR

} // namespace column_major
namespace row_major {

#define ROW_MAJOR
#include "netlib_level1.cxx"
#undef ROW_MAJOR

} // namespace row_major
} // namespace netlib
} // namespace blas
} // namespace mkl
} // namespace oneapi
