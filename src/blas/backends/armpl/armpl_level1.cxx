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

template <typename T, typename U, typename CBLAS_FUNC>
void asum(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
          sycl::buffer<U, 1>& result, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_asum>(cgh, [=]() {
            accessor_result[0] = cblas_func(n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

#define ASUM_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                    \
    void asum(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<RESULT_TYPE, 1>& result) {                                \
        asum(queue, n, x, incx, result, ROUTINE);                                    \
    }

ASUM_LAUNCHER(float, float, ::cblas_sasum)
ASUM_LAUNCHER(double, double, ::cblas_dasum)
ASUM_LAUNCHER(std::complex<float>, float, ::cblas_scasum)
ASUM_LAUNCHER(std::complex<double>, double, ::cblas_dzasum)

template <typename T, typename CBLAS_FUNC>
void axpy(sycl::queue& queue, int64_t n, T alpha, sycl::buffer<T, 1>& x, int64_t incx,
          sycl::buffer<T, 1>& y, int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_axpy>(cgh, [=]() {
            cblas_func(n, cast_to_void_if_complex(alpha), accessor_x.GET_MULTI_PTR, incx,
                       accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define AXPY_LAUNCHER(TYPE, ROUTINE)                                                             \
    void axpy(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        axpy(queue, n, alpha, x, incx, y, incy, ROUTINE);                                        \
    }

AXPY_LAUNCHER(float, ::cblas_saxpy)
AXPY_LAUNCHER(double, ::cblas_daxpy)
AXPY_LAUNCHER(std::complex<float>, ::cblas_caxpy)
AXPY_LAUNCHER(std::complex<double>, ::cblas_zaxpy)

template <typename T, typename CBLAS_FUNC>
void axpby(sycl::queue& queue, int64_t n, T alpha, sycl::buffer<T, 1>& x, int64_t incx, T beta,
           sycl::buffer<T, 1>& y, int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_axpby>(cgh, [=]() {
            cblas_func(n, cast_to_void_if_complex(alpha), accessor_x.GET_MULTI_PTR, incx,
                       cast_to_void_if_complex(beta), accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define AXPBY_LAUNCHER(TYPE, ROUTINE)                                                             \
    void axpby(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
               TYPE beta, sycl::buffer<TYPE, 1>& y, int64_t incy) {                               \
        axpby(queue, n, alpha, x, incx, beta, y, incy, ROUTINE);                                  \
    }

AXPBY_LAUNCHER(float, ::cblas_saxpby)
AXPBY_LAUNCHER(double, ::cblas_daxpby)
AXPBY_LAUNCHER(std::complex<float>, ::cblas_caxpby)
AXPBY_LAUNCHER(std::complex<double>, ::cblas_zaxpby)

template <typename T, typename CBLAS_FUNC>
void copy(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
          int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_copy>(cgh, [=]() {
            cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define COPY_LAUNCHER(TYPE, ROUTINE)                                                 \
    void copy(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        copy(queue, n, x, incx, y, incy, ROUTINE);                                   \
    }

COPY_LAUNCHER(float, ::cblas_scopy)
COPY_LAUNCHER(double, ::cblas_dcopy)
COPY_LAUNCHER(std::complex<float>, ::cblas_ccopy)
COPY_LAUNCHER(std::complex<double>, ::cblas_zcopy)

template <typename T, typename U, typename CBLAS_FUNC>
void dot(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
         int64_t incy, sycl::buffer<U, 1>& result, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_dot>(cgh, [=]() {
            accessor_result[0] =
                cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define DOT_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                             \
    void dot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,          \
             sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<RESULT_TYPE, 1>& result) { \
        dot(queue, n, x, incx, y, incy, result, ROUTINE);                                    \
    }

DOT_LAUNCHER(float, float, ::cblas_sdot)
DOT_LAUNCHER(double, double, ::cblas_ddot)
DOT_LAUNCHER(float, double, ::cblas_dsdot)

template <typename T, typename U, typename CBLAS_FUNC>
void dotc(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
          int64_t incy, sycl::buffer<U, 1>& result, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_dotc>(cgh, [=]() {
            cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy,
                       accessor_result.GET_MULTI_PTR);
        });
    });
}

#define DOTC_LAUNCHER(TYPE, ROUTINE)                                                   \
    void dotc(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,   \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dotc(queue, n, x, incx, y, incy, result, ROUTINE);                             \
    }

DOTC_LAUNCHER(std::complex<float>, ::cblas_cdotc_sub)
DOTC_LAUNCHER(std::complex<double>, ::cblas_zdotc_sub)

#define DOTU_LAUNCHER(TYPE, ROUTINE)                                                   \
    void dotu(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,   \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dotc(queue, n, x, incx, y, incy, result, ROUTINE);                             \
    }

DOTU_LAUNCHER(std::complex<float>, ::cblas_cdotu_sub)
DOTU_LAUNCHER(std::complex<double>, ::cblas_zdotu_sub)

template <typename T, typename CBLAS_FUNC>
void iamin(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_iamin>(cgh, [=]() {
            accessor_result[0] =
                cblas_func((armpl_int_t)n, accessor_x.GET_MULTI_PTR, (armpl_int_t)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

#define IAMIN_LAUNCHER(TYPE, ROUTINE)                                                 \
    void iamin(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
               sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {     \
        iamin(queue, n, x, incx, result, base, ROUTINE);                              \
    }

IAMIN_LAUNCHER(float, ::cblas_isamin)
IAMIN_LAUNCHER(double, ::cblas_idamin)
IAMIN_LAUNCHER(std::complex<float>, ::cblas_icamin)
IAMIN_LAUNCHER(std::complex<double>, ::cblas_izamin)

template <typename T, typename CBLAS_FUNC>
void iamax(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_iamax>(cgh, [=]() {
            accessor_result[0] =
                cblas_func((armpl_int_t)n, accessor_x.GET_MULTI_PTR, (armpl_int_t)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                accessor_result[0]++;
        });
    });
}

#define IAMAX_LAUNCHER(TYPE, ROUTINE)                                                 \
    void iamax(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
               sycl::buffer<int64_t, 1>& result, oneapi::math::index_base base) {     \
        iamax(queue, n, x, incx, result, base, ROUTINE);                              \
    }

IAMAX_LAUNCHER(float, ::cblas_isamax)
IAMAX_LAUNCHER(double, ::cblas_idamax)
IAMAX_LAUNCHER(std::complex<float>, ::cblas_icamax)
IAMAX_LAUNCHER(std::complex<double>, ::cblas_izamax)

template <typename T, typename U, typename CBLAS_FUNC>
void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
          sycl::buffer<U, 1>& result, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_nrm2>(cgh, [=]() {
            accessor_result[0] = cblas_func(n, accessor_x.GET_MULTI_PTR, (const int)std::abs(incx));
        });
    });
}

#define NRM2_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                    \
    void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<RESULT_TYPE, 1>& result) {                                \
        nrm2(queue, n, x, incx, result, ROUTINE);                                    \
    }

NRM2_LAUNCHER(float, float, ::cblas_snrm2)
NRM2_LAUNCHER(double, double, ::cblas_dnrm2)
NRM2_LAUNCHER(std::complex<float>, float, ::cblas_scnrm2)
NRM2_LAUNCHER(std::complex<double>, double, ::cblas_dznrm2)

template <typename T, typename U, typename CBLAS_FUNC>
void rot(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
         int64_t incy, U c, U s, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_rot>(cgh, [=]() {
            cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy, c, s);
        });
    });
}

#define ROT_LAUNCHER(TYPE, TYPEC, ROUTINE)                                          \
    void rot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
             sycl::buffer<TYPE, 1>& y, int64_t incy, TYPEC c, TYPEC s) {            \
        rot(queue, n, x, incx, y, incy, c, s, ROUTINE);                             \
    }

ROT_LAUNCHER(float, float, ::cblas_srot)
ROT_LAUNCHER(double, double, ::cblas_drot)
ROT_LAUNCHER(std::complex<float>, float, ::cblas_csrot)
ROT_LAUNCHER(std::complex<double>, double, ::cblas_zdrot)

template <typename T, typename U, typename CBLAS_FUNC>
void rotg(sycl::queue& queue, sycl::buffer<T, 1>& a, sycl::buffer<T, 1>& b, sycl::buffer<U, 1>& c,
          sycl::buffer<T, 1>& s, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_a = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_b = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_c = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_s = s.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_rotg>(cgh, [=]() {
            cblas_func(accessor_a.GET_MULTI_PTR, accessor_b.GET_MULTI_PTR, accessor_c.GET_MULTI_PTR,
                       accessor_s.GET_MULTI_PTR);
        });
    });
}

#define ROTG_LAUNCHER(TYPE, TYPEC, ROUTINE)                                           \
    void rotg(sycl::queue& queue, sycl::buffer<TYPE, 1>& a, sycl::buffer<TYPE, 1>& b, \
              sycl::buffer<TYPEC, 1>& c, sycl::buffer<TYPE, 1>& s) {                  \
        rotg(queue, a, b, c, s, ROUTINE);                                             \
    }

ROTG_LAUNCHER(float, float, ::cblas_srotg)
ROTG_LAUNCHER(double, double, ::cblas_drotg)
ROTG_LAUNCHER(std::complex<float>, float, ::cblas_crotg)
ROTG_LAUNCHER(std::complex<double>, double, ::cblas_zrotg)

template <typename T, typename CBLAS_FUNC>
void rotm(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
          int64_t incy, sycl::buffer<T, 1>& param, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.template get_access<sycl::access::mode::read>(cgh);
        host_task<class armpl_kernel_rotm>(cgh, [=]() {
            cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy,
                       accessor_param.GET_MULTI_PTR);
        });
    });
}

#define ROTM_LAUNCHER(TYPE, ROUTINE)                                                  \
    void rotm(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,  \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& param) { \
        rotm(queue, n, x, incx, y, incy, param, ROUTINE);                             \
    }

ROTM_LAUNCHER(float, ::cblas_srotm)
ROTM_LAUNCHER(double, ::cblas_drotm)

template <typename T, typename CBLAS_FUNC>
void rotmg(sycl::queue& queue, sycl::buffer<T, 1>& d1, sycl::buffer<T, 1>& d2,
           sycl::buffer<T, 1>& x1, T y1, sycl::buffer<T, 1>& param, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_d1 = d1.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_d2 = d2.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_x1 = x1.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_param = param.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_rotmg>(cgh, [=]() {
            cblas_func(accessor_d1.GET_MULTI_PTR, accessor_d2.GET_MULTI_PTR,
                       accessor_x1.GET_MULTI_PTR, y1, accessor_param.GET_MULTI_PTR);
        });
    });
}

#define ROTMG_LAUNCHER(TYPE, ROUTINE)                                                    \
    void rotmg(sycl::queue& queue, sycl::buffer<TYPE, 1>& d1, sycl::buffer<TYPE, 1>& d2, \
               sycl::buffer<TYPE, 1>& x1, TYPE y1, sycl::buffer<TYPE, 1>& param) {       \
        rotmg(queue, d1, d2, x1, y1, param, ROUTINE);                                    \
    }

ROTMG_LAUNCHER(float, ::cblas_srotmg)
ROTMG_LAUNCHER(double, ::cblas_drotmg)

template <typename T, typename U, typename CBLAS_FUNC>
void scal(sycl::queue& queue, int64_t n, T alpha, sycl::buffer<U, 1>& x, int64_t incx,
          CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_scal>(cgh, [=]() {
            cblas_func(n, cast_to_void_if_complex(alpha), accessor_x.GET_MULTI_PTR, incx);
        });
    });
}

#define SCAL_LAUNCHER(TYPE, ALPHA_TYPE, ROUTINE)                                         \
    void scal(sycl::queue& queue, int64_t n, ALPHA_TYPE alpha, sycl::buffer<TYPE, 1>& x, \
              int64_t incx) {                                                            \
        scal(queue, n, alpha, x, incx, ROUTINE);                                         \
    }

SCAL_LAUNCHER(float, float, ::cblas_sscal)
SCAL_LAUNCHER(double, double, ::cblas_dscal)
SCAL_LAUNCHER(std::complex<float>, std::complex<float>, ::cblas_cscal)
SCAL_LAUNCHER(std::complex<double>, std::complex<double>, ::cblas_zscal)
SCAL_LAUNCHER(std::complex<float>, float, ::cblas_csscal)
SCAL_LAUNCHER(std::complex<double>, double, ::cblas_zdscal)

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read>(cgh);
        auto accessor_result = result.template get_access<sycl::access::mode::write>(cgh);
        host_task<class armpl_kernel_sdsdot>(cgh, [=]() {
            accessor_result[0] = ::cblas_sdsdot(n, sb, accessor_x.GET_MULTI_PTR, incx,
                                                accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

template <typename T, typename CBLAS_FUNC>
void swap(sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx, sycl::buffer<T, 1>& y,
          int64_t incy, CBLAS_FUNC cblas_func) {
    queue.submit([&](sycl::handler& cgh) {
        auto accessor_x = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto accessor_y = y.template get_access<sycl::access::mode::read_write>(cgh);
        host_task<class armpl_kernel_swap>(cgh, [=]() {
            cblas_func(n, accessor_x.GET_MULTI_PTR, incx, accessor_y.GET_MULTI_PTR, incy);
        });
    });
}

#define SWAP_LAUNCHER(TYPE, ROUTINE)                                                 \
    void swap(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        swap(queue, n, x, incx, y, incy, ROUTINE);                                   \
    }

SWAP_LAUNCHER(float, ::cblas_sswap)
SWAP_LAUNCHER(double, ::cblas_dswap)
SWAP_LAUNCHER(std::complex<float>, ::cblas_cswap)
SWAP_LAUNCHER(std::complex<double>, ::cblas_zswap)

// USM APIs

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event asum(sycl::queue& queue, int64_t n, const T* x, int64_t incx, U* result,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_asum>(
            cgh, [=]() { result[0] = cblas_func(n, x, (const int)std::abs(incx)); });
    });
    return done;
}

#define ASUM_USM_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                     \
    sycl::event asum(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx,          \
                     RESULT_TYPE* result, const std::vector<sycl::event>& dependencies) { \
        return asum(queue, n, x, incx, result, dependencies, ROUTINE);                    \
    }

ASUM_USM_LAUNCHER(float, float, ::cblas_sasum)
ASUM_USM_LAUNCHER(double, double, ::cblas_dasum)
ASUM_USM_LAUNCHER(std::complex<float>, float, ::cblas_scasum)
ASUM_USM_LAUNCHER(std::complex<double>, double, ::cblas_dzasum)

template <typename T, typename CBLAS_FUNC>
sycl::event axpy(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies,
                 CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_axpy>(
            cgh, [=]() { cblas_func(n, cast_to_void_if_complex(alpha), x, incx, y, incy); });
    });
    return done;
}

#define AXPY_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event axpy(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                     TYPE* y, int64_t incy, const std::vector<sycl::event>& dependencies) {  \
        return axpy(queue, n, alpha, x, incx, y, incy, dependencies, ROUTINE);               \
    }

AXPY_USM_LAUNCHER(float, ::cblas_saxpy)
AXPY_USM_LAUNCHER(double, ::cblas_daxpy)
AXPY_USM_LAUNCHER(std::complex<float>, ::cblas_caxpy)
AXPY_USM_LAUNCHER(std::complex<double>, ::cblas_zaxpy)

template <typename T, typename CBLAS_FUNC>
sycl::event axpby(sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx, T beta, T* y,
                  int64_t incy, const std::vector<sycl::event>& dependencies,
                  CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_axpby>(cgh, [=]() {
            cblas_func(n, cast_to_void_if_complex(alpha), x, incx, cast_to_void_if_complex(beta), y,
                       incy);
        });
    });
    return done;
}

#define AXPBY_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event axpby(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                      TYPE beta, TYPE* y, int64_t incy,                                       \
                      const std::vector<sycl::event>& dependencies) {                         \
        return axpby(queue, n, alpha, x, incx, beta, y, incy, dependencies, ROUTINE);         \
    }

AXPBY_USM_LAUNCHER(float, ::cblas_saxpby)
AXPBY_USM_LAUNCHER(double, ::cblas_daxpby)
AXPBY_USM_LAUNCHER(std::complex<float>, ::cblas_caxpby)
AXPBY_USM_LAUNCHER(std::complex<double>, ::cblas_zaxpby)

template <typename T, typename CBLAS_FUNC>
sycl::event copy(sycl::queue& queue, int64_t n, const T* x, int64_t incx, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_copy>(cgh, [=]() { cblas_func(n, x, incx, y, incy); });
    });
    return done;
}

#define COPY_USM_LAUNCHER(TYPE, ROUTINE)                                                  \
    sycl::event copy(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {        \
        return copy(queue, n, x, incx, y, incy, dependencies, ROUTINE);                   \
    }

COPY_USM_LAUNCHER(float, ::cblas_scopy)
COPY_USM_LAUNCHER(double, ::cblas_dcopy)
COPY_USM_LAUNCHER(std::complex<float>, ::cblas_ccopy)
COPY_USM_LAUNCHER(std::complex<double>, ::cblas_zcopy)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event dot(sycl::queue& queue, int64_t n, const T* x, int64_t incx, const T* y, int64_t incy,
                U* result, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_dot>(cgh,
                                          [=]() { result[0] = cblas_func(n, x, incx, y, incy); });
    });
    return done;
}

#define DOT_USM_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                           \
    sycl::event dot(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, const TYPE* y, \
                    int64_t incy, RESULT_TYPE* result,                                         \
                    const std::vector<sycl::event>& dependencies) {                            \
        return dot(queue, n, x, incx, y, incy, result, dependencies, ROUTINE);                 \
    }

DOT_USM_LAUNCHER(float, float, ::cblas_sdot)
DOT_USM_LAUNCHER(double, double, ::cblas_ddot)
DOT_USM_LAUNCHER(float, double, ::cblas_dsdot)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event dotc(sycl::queue& queue, int64_t n, const T* x, int64_t incx, const T* y, int64_t incy,
                 U* result, const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_dotc>(cgh, [=]() { cblas_func(n, x, incx, y, incy, result); });
    });
    return done;
}

#define DOTC_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event dotc(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, const TYPE* y,  \
                     int64_t incy, TYPE* result, const std::vector<sycl::event>& dependencies) { \
        return dotc(queue, n, x, incx, y, incy, result, dependencies, ROUTINE);                  \
    }

DOTC_USM_LAUNCHER(std::complex<float>, ::cblas_cdotc_sub)
DOTC_USM_LAUNCHER(std::complex<double>, ::cblas_zdotc_sub)

#define DOTU_USM_LAUNCHER(TYPE, ROUTINE)                                                         \
    sycl::event dotu(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, const TYPE* y,  \
                     int64_t incy, TYPE* result, const std::vector<sycl::event>& dependencies) { \
        return dotc(queue, n, x, incx, y, incy, result, dependencies, ROUTINE);                  \
    }

DOTU_USM_LAUNCHER(std::complex<float>, ::cblas_cdotu_sub)
DOTU_USM_LAUNCHER(std::complex<double>, ::cblas_zdotu_sub)

template <typename T, typename CBLAS_FUNC>
sycl::event iamin(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies,
                  CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_iamin>(cgh, [=]() {
            result[0] = cblas_func((armpl_int_t)n, x, (armpl_int_t)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

#define IAMIN_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event iamin(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, int64_t* result, \
                      oneapi::math::index_base base,                                               \
                      const std::vector<sycl::event>& dependencies) {                              \
        return iamin(queue, n, x, incx, result, base, dependencies, ROUTINE);                      \
    }

IAMIN_USM_LAUNCHER(float, ::cblas_isamin)
IAMIN_USM_LAUNCHER(double, ::cblas_idamin)
IAMIN_USM_LAUNCHER(std::complex<float>, ::cblas_icamin)
IAMIN_USM_LAUNCHER(std::complex<double>, ::cblas_izamin)

template <typename T, typename CBLAS_FUNC>
sycl::event iamax(sycl::queue& queue, int64_t n, const T* x, int64_t incx, int64_t* result,
                  oneapi::math::index_base base, const std::vector<sycl::event>& dependencies,
                  CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_iamax>(cgh, [=]() {
            result[0] = cblas_func((armpl_int_t)n, x, (armpl_int_t)incx);
            if (base == oneapi::math::index_base::one && n >= 1 && incx >= 1)
                result[0]++;
        });
    });
    return done;
}

#define IAMAX_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event iamax(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, int64_t* result, \
                      oneapi::math::index_base base,                                               \
                      const std::vector<sycl::event>& dependencies) {                              \
        return iamax(queue, n, x, incx, result, base, dependencies, ROUTINE);                      \
    }

IAMAX_USM_LAUNCHER(float, ::cblas_isamax)
IAMAX_USM_LAUNCHER(double, ::cblas_idamax)
IAMAX_USM_LAUNCHER(std::complex<float>, ::cblas_icamax)
IAMAX_USM_LAUNCHER(std::complex<double>, ::cblas_izamax)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event nrm2(sycl::queue& queue, int64_t n, const T* x, int64_t incx, U* result,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_nrm2>(
            cgh, [=]() { result[0] = cblas_func(n, x, (const int)std::abs(incx)); });
    });
    return done;
}

#define NRM2_USM_LAUNCHER(TYPE, RESULT_TYPE, ROUTINE)                                     \
    sycl::event nrm2(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx,          \
                     RESULT_TYPE* result, const std::vector<sycl::event>& dependencies) { \
        return nrm2(queue, n, x, incx, result, dependencies, ROUTINE);                    \
    }

NRM2_USM_LAUNCHER(float, float, ::cblas_snrm2)
NRM2_USM_LAUNCHER(double, double, ::cblas_dnrm2)
NRM2_USM_LAUNCHER(std::complex<float>, float, ::cblas_scnrm2)
NRM2_USM_LAUNCHER(std::complex<double>, double, ::cblas_dznrm2)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event rot(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy, U c, U s,
                const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_rot>(cgh, [=]() { cblas_func(n, x, incx, y, incy, c, s); });
    });
    return done;
}

#define ROT_USM_LAUNCHER(TYPE, TYPEC, ROUTINE)                                                   \
    sycl::event rot(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                    TYPEC c, TYPEC s, const std::vector<sycl::event>& dependencies) {            \
        return rot(queue, n, x, incx, y, incy, c, s, dependencies, ROUTINE);                     \
    }

ROT_USM_LAUNCHER(float, float, ::cblas_srot)
ROT_USM_LAUNCHER(double, double, ::cblas_drot)
ROT_USM_LAUNCHER(std::complex<float>, float, ::cblas_csrot)
ROT_USM_LAUNCHER(std::complex<double>, double, ::cblas_zdrot)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event rotg(sycl::queue& queue, T* a, T* b, U* c, T* s,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_rotg>(cgh, [=]() { cblas_func(a, b, c, s); });
    });
    return done;
}

#define ROTG_USM_LAUNCHER(TYPE, TYPEC, ROUTINE)                               \
    sycl::event rotg(sycl::queue& queue, TYPE* a, TYPE* b, TYPEC* c, TYPE* s, \
                     const std::vector<sycl::event>& dependencies) {          \
        return rotg(queue, a, b, c, s, dependencies, ROUTINE);                \
    }

ROTG_USM_LAUNCHER(float, float, ::cblas_srotg)
ROTG_USM_LAUNCHER(double, double, ::cblas_drotg)
ROTG_USM_LAUNCHER(std::complex<float>, float, ::cblas_crotg)
ROTG_USM_LAUNCHER(std::complex<double>, double, ::cblas_zrotg)

template <typename T, typename CBLAS_FUNC>
sycl::event rotm(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy, T* param,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_rotm>(cgh, [=]() { cblas_func(n, x, incx, y, incy, param); });
    });
    return done;
}

#define ROTM_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event rotm(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     TYPE* param, const std::vector<sycl::event>& dependencies) {                 \
        return rotm(queue, n, x, incx, y, incy, param, dependencies, ROUTINE);                    \
    }

ROTM_USM_LAUNCHER(float, ::cblas_srotm)
ROTM_USM_LAUNCHER(double, ::cblas_drotm)

template <typename T, typename CBLAS_FUNC>
sycl::event rotmg(sycl::queue& queue, T* d1, T* d2, T* x1, T y1, T* param,
                  const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_rotmg>(cgh, [=]() { cblas_func(d1, d2, x1, y1, param); });
    });
    return done;
}

#define ROTMG_USM_LAUNCHER(TYPE, ROUTINE)                                                     \
    sycl::event rotmg(sycl::queue& queue, TYPE* d1, TYPE* d2, TYPE* x1, TYPE y1, TYPE* param, \
                      const std::vector<sycl::event>& dependencies) {                         \
        return rotmg(queue, d1, d2, x1, y1, param, dependencies, ROUTINE);                    \
    }

ROTMG_USM_LAUNCHER(float, ::cblas_srotmg)
ROTMG_USM_LAUNCHER(double, ::cblas_drotmg)

template <typename T, typename U, typename CBLAS_FUNC>
sycl::event scal(sycl::queue& queue, int64_t n, T alpha, U* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_scal>(cgh, [=]() {
            cblas_func(n, cast_to_void_if_complex(alpha), x, (const int)std::abs(incx));
        });
    });
    return done;
}

#define SCAL_USM_LAUNCHER(TYPE, ALPHA_TYPE, ROUTINE)                                         \
    sycl::event scal(sycl::queue& queue, int64_t n, ALPHA_TYPE alpha, TYPE* x, int64_t incx, \
                     const std::vector<sycl::event>& dependencies) {                         \
        return scal(queue, n, alpha, x, incx, dependencies, ROUTINE);                        \
    }

SCAL_USM_LAUNCHER(float, float, ::cblas_sscal)
SCAL_USM_LAUNCHER(double, double, ::cblas_dscal)
SCAL_USM_LAUNCHER(std::complex<float>, std::complex<float>, ::cblas_cscal)
SCAL_USM_LAUNCHER(std::complex<double>, std::complex<double>, ::cblas_zscal)
SCAL_USM_LAUNCHER(std::complex<float>, float, ::cblas_csscal)
SCAL_USM_LAUNCHER(std::complex<double>, double, ::cblas_zdscal)

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_sdsdot_usm>(
            cgh, [=]() { result[0] = ::cblas_sdsdot(n, sb, x, incx, y, incy); });
    });
    return done;
}

template <typename T, typename CBLAS_FUNC>
sycl::event swap(sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies, CBLAS_FUNC cblas_func) {
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; ++i) {
            cgh.depends_on(dependencies[i]);
        }
        host_task<class armpl_kernel_swap>(cgh, [=]() {
            cblas_func(n, x, (const int)std::abs(incx), y, (const int)std::abs(incy));
        });
    });
    return done;
}

#define SWAP_USM_LAUNCHER(TYPE, ROUTINE)                                                          \
    sycl::event swap(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                              \
        return swap(queue, n, x, incx, y, incy, dependencies, ROUTINE);                           \
    }

SWAP_USM_LAUNCHER(float, ::cblas_sswap)
SWAP_USM_LAUNCHER(double, ::cblas_dswap)
SWAP_USM_LAUNCHER(std::complex<float>, ::cblas_cswap)
SWAP_USM_LAUNCHER(std::complex<double>, ::cblas_zswap)
