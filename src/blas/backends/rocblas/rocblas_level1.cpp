/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include "rocblas_helper.hpp"
#include "rocblas_task.hpp"

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/blas/detail/rocblas/onemath_blas_rocblas.hpp"

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {
namespace column_major {

// Buffer APIs

template <typename Func, typename T1, typename T2>
inline void asum(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                 const int64_t incx, sycl::buffer<T2, 1>& result) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType1*>(x_acc);
            auto res_ = sc.get_mem<rocDataType2*>(res_acc);
            rocblas_status err;
            // ASUM does not support negative index
            rocblas_native_func(func, err, handle, n, x_, std::abs(incx), res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define ASUM_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                        \
    void asum(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        asum(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                   \
    }

ASUM_LAUNCHER(float, float, rocblas_sasum)
ASUM_LAUNCHER(double, double, rocblas_dasum)
ASUM_LAUNCHER(std::complex<float>, float, rocblas_scasum)
ASUM_LAUNCHER(std::complex<double>, double, rocblas_dzasum)

#undef ASUM_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void scal(Func func, sycl::queue& queue, int64_t n, T1 a, sycl::buffer<T2, 1>& x,
                 int64_t incx) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto x_ = sc.get_mem<rocDataType2*>(x_acc);
            rocblas_status err;
            // SCAL does not support negative incx
            rocblas_native_func(func, err, handle, n, (rocDataType1*)&a, x_, std::abs(incx));
        });
    });
}

#define SCAL_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                             \
    void scal(sycl::queue& queue, int64_t n, TYPE1 a, sycl::buffer<TYPE2, 1>& x, int64_t incx) { \
        scal(ROCBLAS_ROUTINE, queue, n, a, x, incx);                                             \
    }

SCAL_LAUNCHER(float, float, rocblas_sscal)
SCAL_LAUNCHER(double, double, rocblas_dscal)
SCAL_LAUNCHER(std::complex<float>, std::complex<float>, rocblas_cscal)
SCAL_LAUNCHER(std::complex<double>, std::complex<double>, rocblas_zscal)
SCAL_LAUNCHER(float, std::complex<float>, rocblas_csscal)
SCAL_LAUNCHER(double, std::complex<double>, rocblas_zdscal)

#undef SCAL_LAUNCHER

template <typename Func, typename T>
inline void axpy(Func func, sycl::queue& queue, int64_t n, T alpha, sycl::buffer<T, 1>& x,
                 int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto y_ = sc.get_mem<rocDataType*>(y_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, (rocDataType*)&alpha, x_, incx, y_, incy);
        });
    });
}

#define AXPY_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void axpy(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        axpy(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy);                                \
    }

AXPY_LAUNCHER(float, rocblas_saxpy)
AXPY_LAUNCHER(double, rocblas_daxpy)
AXPY_LAUNCHER(std::complex<float>, rocblas_caxpy)
AXPY_LAUNCHER(std::complex<double>, rocblas_zaxpy)

#undef AXPY_LAUNCHER

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

template <typename Func, typename T1, typename T2>
inline void rotg(Func func, sycl::queue& queue, sycl::buffer<T1, 1>& a, sycl::buffer<T1, 1>& b,
                 sycl::buffer<T2, 1>& c, sycl::buffer<T1, 1>& s) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto a_ = sc.get_mem<rocDataType1*>(a_acc);
            auto b_ = sc.get_mem<rocDataType1*>(b_acc);
            auto c_ = sc.get_mem<rocDataType2*>(c_acc);
            auto s_ = sc.get_mem<rocDataType1*>(s_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, a_, b_, c_, s_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define ROTG_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                    \
    void rotg(sycl::queue& queue, sycl::buffer<TYPE1, 1>& a, sycl::buffer<TYPE1, 1>& b, \
              sycl::buffer<TYPE2, 1>& c, sycl::buffer<TYPE1, 1>& s) {                   \
        rotg(ROCBLAS_ROUTINE, queue, a, b, c, s);                                       \
    }

ROTG_LAUNCHER(float, float, rocblas_srotg)
ROTG_LAUNCHER(double, double, rocblas_drotg)
ROTG_LAUNCHER(std::complex<float>, float, rocblas_crotg)
ROTG_LAUNCHER(std::complex<double>, double, rocblas_zrotg)

#undef ROTG_LAUNCHER

template <typename Func, typename T>
inline void rotm(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& param) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        auto param_acc = param.template get_access<sycl::access::mode::read>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto y_ = sc.get_mem<rocDataType*>(y_acc);
            auto param_ = sc.get_mem<rocDataType*>(param_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, param_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define ROTM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                          \
    void rotm(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,  \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& param) { \
        rotm(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, param);                     \
    }

ROTM_LAUNCHER(float, rocblas_srotm)
ROTM_LAUNCHER(double, rocblas_drotm)

#undef ROTM_LAUNCHER

template <typename Func, typename T>
inline void copy(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto y_ = sc.get_mem<rocDataType*>(y_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define COPY_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void copy(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        copy(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy);                           \
    }

COPY_LAUNCHER(float, rocblas_scopy)
COPY_LAUNCHER(double, rocblas_dcopy)
COPY_LAUNCHER(std::complex<float>, rocblas_ccopy)
COPY_LAUNCHER(std::complex<double>, rocblas_zcopy)

#undef COPY_LAUNCHER

template <typename Func, typename T>
inline void dot(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, const int64_t incx,
                sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& result) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto y_ = sc.get_mem<rocDataType*>(y_acc);
            auto res_ = sc.get_mem<rocDataType*>(res_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define DOT_LAUNCHER(EXT, TYPE, ROCBLAS_ROUTINE)                                                 \
    void dot##EXT(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx,   \
                  sycl::buffer<TYPE, 1>& y, const int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, result);                                \
    }

DOT_LAUNCHER(, float, rocblas_sdot)
DOT_LAUNCHER(, double, rocblas_ddot)
DOT_LAUNCHER(u, std::complex<float>, rocblas_cdotu)
DOT_LAUNCHER(c, std::complex<float>, rocblas_cdotc)
DOT_LAUNCHER(u, std::complex<double>, rocblas_zdotu)
DOT_LAUNCHER(c, std::complex<double>, rocblas_zdotc)

#undef DOT_LAUNCHER

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename Func, typename T1, typename T2, typename T3>
inline void rot(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                const int64_t incx, sycl::buffer<T1, 1>& y, int64_t incy, T2 c, T3 s) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    using rocDataType3 = typename RocEquivalentType<T3>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            // rocblas_set_pointer_mode(handle, rocblas_set_pointer_mode);
            auto x_ = sc.get_mem<rocDataType1*>(x_acc);
            auto y_ = sc.get_mem<rocDataType1*>(y_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, (rocDataType2*)&c,
                                (rocDataType3*)&s);
        });
    });
}

#define ROT_LAUNCHER(TYPE1, TYPE2, TYPE3, ROCBLAS_ROUTINE)                                 \
    void rot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
             sycl::buffer<TYPE1, 1>& y, int64_t incy, TYPE2 c, TYPE3 s) {                  \
        rot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s);                            \
    }

ROT_LAUNCHER(float, float, float, rocblas_srot)
ROT_LAUNCHER(double, double, double, rocblas_drot)
ROT_LAUNCHER(std::complex<float>, float, float, rocblas_csrot)
ROT_LAUNCHER(std::complex<double>, double, double, rocblas_zdrot)

#undef ROT_LAUNCHER

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    overflow_check(n, incx, incy);

    // rocBLAS does not support sdot so we need to mimic sdot.
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.get_access<sycl::access::mode::read>(cgh);
        auto y_acc = y.get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<float*>(x_acc);
            auto y_ = sc.get_mem<float*>(y_acc);
            auto res_ = sc.get_mem<float*>(res_acc);
            rocblas_status err;
            rocblas_native_func(rocblas_sdot, err, handle, n, x_, incx, y_, incy, res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    // Since SB is a host pointer we need to bring the result back to the host and
    // add sb to it.
    result.get_access<sycl::access::mode::read_write>()[0] += sb;
}

template <typename Func, typename T>
inline void rotmg(Func func, sycl::queue& queue, sycl::buffer<T, 1>& d1, sycl::buffer<T, 1>& d2,
                  sycl::buffer<T, 1>& x1, T y1, sycl::buffer<T, 1>& param) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    sycl::buffer<T, 1> y1_buff(&y1, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
        auto d1_acc = d1.template get_access<sycl::access::mode::read_write>(cgh);
        auto d2_acc = d2.template get_access<sycl::access::mode::read_write>(cgh);
        auto x1_acc = x1.template get_access<sycl::access::mode::read_write>(cgh);
        auto y1_acc = y1_buff.template get_access<sycl::access::mode::read>(cgh);
        auto param_acc = param.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto d1_ = sc.get_mem<rocDataType*>(d1_acc);
            auto d2_ = sc.get_mem<rocDataType*>(d2_acc);
            auto x1_ = sc.get_mem<rocDataType*>(x1_acc);
            auto y1_ = sc.get_mem<rocDataType*>(y1_acc);
            auto param_ = sc.get_mem<rocDataType*>(param_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, d1_, d2_, x1_, y1_, param_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define ROTMG_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                            \
    void rotmg(sycl::queue& queue, sycl::buffer<TYPE, 1>& d1, sycl::buffer<TYPE, 1>& d2, \
               sycl::buffer<TYPE, 1>& x1, TYPE y1, sycl::buffer<TYPE, 1>& param) {       \
        rotmg(ROCBLAS_ROUTINE, queue, d1, d2, x1, y1, param);                            \
    }

ROTMG_LAUNCHER(float, rocblas_srotmg)
ROTMG_LAUNCHER(double, rocblas_drotmg)

#undef ROTMG_LAUNCHER

template <typename Func, typename T>
inline void iamax(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x,
                  const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);

    // rocBLAS does not support int64_t as return type for the data by default. So we need to
    // mimic iamax. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size. Alternatively either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    sycl::buffer<int, 1> int_res_buff{ sycl::range<1>(1) };
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto int_res_ = sc.get_mem<int*>(int_res_acc);
            rocblas_status err;
            // For negative incx, iamax returns 0. This behaviour is similar to that of
            // reference netlib BLAS.
            rocblas_native_func(func, err, handle, n, x_, incx, int_res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task(
            [=]() { result_acc[0] = std::max((int64_t)int_res_acc[0] - 1, (int64_t)0); });
    });
}

#define IAMAX_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void iamax(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamax(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                  \
    }

IAMAX_LAUNCHER(float, rocblas_isamax)
IAMAX_LAUNCHER(double, rocblas_idamax)
IAMAX_LAUNCHER(std::complex<float>, rocblas_icamax)
IAMAX_LAUNCHER(std::complex<double>, rocblas_izamax)

#undef IAMAX_LAUNCHER

template <typename Func, typename T>
inline void swap(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read_write>(cgh);
        auto y_acc = y.template get_access<sycl::access::mode::read_write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto y_ = sc.get_mem<rocDataType*>(y_acc);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });
}

#define SWAP_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void swap(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        swap(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy);                           \
    }

SWAP_LAUNCHER(float, rocblas_sswap)
SWAP_LAUNCHER(double, rocblas_dswap)
SWAP_LAUNCHER(std::complex<float>, rocblas_cswap)
SWAP_LAUNCHER(std::complex<double>, rocblas_zswap)

#undef SWAP_LAUNCHER

template <typename Func, typename T>
inline void iamin(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x,
                  const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);

    // rocBLAS does not support int64_t as return type for the data by default. So we need to
    // mimic iamin we are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size. Alternatively, either we need to write a sycl kernel
    // to elementwise copy the data between two buffer, or allow reinterpret cast
    // to convert to different type with different typesize size.
    sycl::buffer<int, 1> int_res_buff{ sycl::range<1>(1) };
    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType*>(x_acc);
            auto int_res_ = sc.get_mem<int*>(int_res_acc);
            rocblas_status err;
            // For negative incx, iamin returns 0. This behaviour is similar to that of
            // implemented as a reference IAMIN.
            rocblas_native_func(func, err, handle, n, x_, incx, int_res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto int_res_acc = int_res_buff.template get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task(
            [=]() { result_acc[0] = std::max((int64_t)int_res_acc[0] - 1, (int64_t)0); });
    });
}

#define IAMIN_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void iamin(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamin(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                  \
    }

IAMIN_LAUNCHER(float, rocblas_isamin)
IAMIN_LAUNCHER(double, rocblas_idamin)
IAMIN_LAUNCHER(std::complex<float>, rocblas_icamin)
IAMIN_LAUNCHER(std::complex<double>, rocblas_izamin)

#undef IAMIN_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void nrm2(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                 const int64_t incx, sycl::buffer<T2, 1>& result) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    queue.submit([&](sycl::handler& cgh) {
        auto x_acc = x.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = result.template get_access<sycl::access::mode::write>(cgh);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            // By default the pointer mode is the rocblas_pointer_mode_host
            // when the data is on buffer, it must be set to
            // rocblas_set_pointer_mode mode otherwise it causes the segmentation
            // fault. When it is set to device it is users responsibility to
            // synchronise as the function is completely asynchronous.
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = sc.get_mem<rocDataType1*>(x_acc);
            auto res_ = sc.get_mem<rocDataType2*>(res_acc);
            rocblas_status err;
            // NRM2 does not support negative index
            rocblas_native_func(func, err, handle, n, x_, std::abs(incx), res_);
            // Higher level BLAS functions expect rocblas_pointer_mode_host
            // to be set, therfore we need to reset this to the default value
            // in order to avoid invalid memory accesses
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });
}

#define NRM2_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                        \
    void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        nrm2(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                   \
    }

NRM2_LAUNCHER(float, float, rocblas_snrm2)
NRM2_LAUNCHER(double, double, rocblas_dnrm2)
NRM2_LAUNCHER(std::complex<float>, float, rocblas_scnrm2)
NRM2_LAUNCHER(std::complex<double>, double, rocblas_dznrm2)

#undef NRM2_LAUNCHER

// USM APIs

template <typename Func, typename T1, typename T2>
inline sycl::event asum(Func func, sycl::queue& queue, int64_t n, const T1* x, const int64_t incx,
                        T2* result, const std::vector<sycl::event>& dependencies) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

            auto x_ = reinterpret_cast<const rocDataType1*>(x);
            auto res_ = reinterpret_cast<rocDataType2*>(result);
            rocblas_status err;
            // ASUM does not support negative index
            rocblas_native_func(func, err, handle, n, x_, std::abs(incx), res_);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    return done;
}

#define ASUM_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                \
    sycl::event asum(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx, \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {     \
        return asum(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);          \
    }

ASUM_LAUNCHER_USM(float, float, rocblas_sasum)
ASUM_LAUNCHER_USM(double, double, rocblas_dasum)
ASUM_LAUNCHER_USM(std::complex<float>, float, rocblas_scasum)
ASUM_LAUNCHER_USM(std::complex<double>, double, rocblas_dzasum)

#undef ASUM_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event scal(Func func, sycl::queue& queue, int64_t n, T1 a, T2* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<rocDataType2*>(x);
            rocblas_status err;
            // SCAL does not support negative incx
            rocblas_native_func(func, err, handle, n, (rocDataType1*)&a, x_, std::abs(incx));
        });
    });

    return done;
}

#define SCAL_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                             \
    sycl::event scal(sycl::queue& queue, int64_t n, TYPE1 a, TYPE2* x, int64_t incx, \
                     const std::vector<sycl::event>& dependencies) {                 \
        return scal(ROCBLAS_ROUTINE, queue, n, a, x, incx, dependencies);            \
    }

SCAL_LAUNCHER_USM(float, float, rocblas_sscal)
SCAL_LAUNCHER_USM(double, double, rocblas_dscal)
SCAL_LAUNCHER_USM(std::complex<float>, std::complex<float>, rocblas_cscal)
SCAL_LAUNCHER_USM(std::complex<double>, std::complex<double>, rocblas_zscal)
SCAL_LAUNCHER_USM(float, std::complex<float>, rocblas_csscal)
SCAL_LAUNCHER_USM(double, std::complex<double>, rocblas_zdscal)

#undef SCAL_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy(Func func, sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx,
                        T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const rocDataType*>(x);
            auto y_ = reinterpret_cast<rocDataType*>(y);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, (rocDataType*)&alpha, x_, incx, y_, incy);
        });
    });

    return done;
}

#define AXPY_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event axpy(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                     TYPE* y, int64_t incy, const std::vector<sycl::event>& dependencies) {  \
        return axpy(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy, dependencies);       \
    }

AXPY_LAUNCHER_USM(float, rocblas_saxpy)
AXPY_LAUNCHER_USM(double, rocblas_daxpy)
AXPY_LAUNCHER_USM(std::complex<float>, rocblas_caxpy)
AXPY_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy)

#undef AXPY_LAUNCHER_USM

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for column_major layout");
}

template <typename Func, typename T1, typename T2>
inline sycl::event rotg(Func func, sycl::queue& queue, T1* a, T1* b, T2* c, T1* s,
                        const std::vector<sycl::event>& dependencies) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto a_ = reinterpret_cast<rocDataType1*>(a);
            auto b_ = reinterpret_cast<rocDataType1*>(b);
            auto c_ = reinterpret_cast<rocDataType2*>(c);
            auto s_ = reinterpret_cast<rocDataType1*>(s);
            rocblas_status err;
            rocblas_native_func(func, err, handle, a_, b_, c_, s_);
        });
    });

    return done;
}

#define ROTG_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                         \
    sycl::event rotg(sycl::queue& queue, TYPE1* a, TYPE1* b, TYPE2* c, TYPE1* s, \
                     const std::vector<sycl::event>& dependencies) {             \
        return rotg(ROCBLAS_ROUTINE, queue, a, b, c, s, dependencies);           \
    }

ROTG_LAUNCHER_USM(float, float, rocblas_srotg)
ROTG_LAUNCHER_USM(double, double, rocblas_drotg)
ROTG_LAUNCHER_USM(std::complex<float>, float, rocblas_crotg)
ROTG_LAUNCHER_USM(std::complex<double>, double, rocblas_zrotg)

#undef ROTG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event rotm(Func func, sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y,
                        int64_t incy, T* param, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<rocDataType*>(x);
            auto y_ = reinterpret_cast<rocDataType*>(y);
            auto param_ = reinterpret_cast<rocDataType*>(param);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, param_);
        });
    });

    return done;
}

#define ROTM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event rotm(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     TYPE* param, const std::vector<sycl::event>& dependencies) {                 \
        return rotm(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, param, dependencies);            \
    }

ROTM_LAUNCHER_USM(float, rocblas_srotm)
ROTM_LAUNCHER_USM(double, rocblas_drotm)

#undef ROTM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy(Func func, sycl::queue& queue, int64_t n, const T* x, int64_t incx, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const rocDataType*>(x);
            auto y_ = reinterpret_cast<rocDataType*>(y);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });

    return done;
}

#define COPY_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event copy(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {        \
        return copy(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);           \
    }

COPY_LAUNCHER_USM(float, rocblas_scopy)
COPY_LAUNCHER_USM(double, rocblas_dcopy)
COPY_LAUNCHER_USM(std::complex<float>, rocblas_ccopy)
COPY_LAUNCHER_USM(std::complex<double>, rocblas_zcopy)

#undef COPY_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dot(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                       const T* y, int64_t incy, T* result,
                       const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const rocDataType*>(x);
            auto y_ = reinterpret_cast<const rocDataType*>(y);
            auto res_ = reinterpret_cast<rocDataType*>(result);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, res_);
        });
    });

    return done;
}

#define DOT_LAUNCHER_USM(EXT, TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event dot##EXT(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                         const TYPE* y, const int64_t incy, TYPE* result,                  \
                         const std::vector<sycl::event>& dependencies) {                   \
        return dot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, result, dependencies);     \
    }

DOT_LAUNCHER_USM(, float, rocblas_sdot)
DOT_LAUNCHER_USM(, double, rocblas_ddot)
DOT_LAUNCHER_USM(u, std::complex<float>, rocblas_cdotu)
DOT_LAUNCHER_USM(c, std::complex<float>, rocblas_cdotc)
DOT_LAUNCHER_USM(u, std::complex<double>, rocblas_zdotu)
DOT_LAUNCHER_USM(c, std::complex<double>, rocblas_zdotc)

#undef DOT_LAUNCHER_USM

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", "for column_major layout");
}

template <typename Func, typename T1, typename T2, typename T3>
inline sycl::event rot(Func func, sycl::queue& queue, int64_t n, T1* x, const int64_t incx, T1* y,
                       int64_t incy, T2 c, T3 s, const std::vector<sycl::event>& dependencies) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    using rocDataType3 = typename RocEquivalentType<T3>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<rocDataType1*>(x);
            auto y_ = reinterpret_cast<rocDataType1*>(y);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy, (rocDataType2*)&c,
                                (rocDataType3*)&s);
        });
    });

    return done;
}

#define ROT_LAUNCHER_USM(TYPE1, TYPE2, TYPE3, ROCBLAS_ROUTINE)                             \
    sycl::event rot(sycl::queue& queue, int64_t n, TYPE1* x, const int64_t incx, TYPE1* y, \
                    int64_t incy, TYPE2 c, TYPE3 s,                                        \
                    const std::vector<sycl::event>& dependencies) {                        \
        return rot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s, dependencies);       \
    }

ROT_LAUNCHER_USM(float, float, float, rocblas_srot)
ROT_LAUNCHER_USM(double, double, double, rocblas_drot)
ROT_LAUNCHER_USM(std::complex<float>, float, float, rocblas_csrot)
ROT_LAUNCHER_USM(std::complex<double>, double, double, rocblas_zdrot)

#undef ROT_LAUNCHER_USM

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    overflow_check(n, incx, incy);

    // rocBLAS does not support sdot so we need to mimic sdot.
    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<const float*>(x);
            auto y_ = reinterpret_cast<const float*>(y);
            auto res_ = reinterpret_cast<float*>(result);
            rocblas_status err;
            rocblas_native_func(rocblas_sdot, err, handle, n, x_, incx, y_, incy, res_);
        });
    });

    done.wait_and_throw();
    result[0] = result[0] + sb;
    return done;
}

template <typename Func, typename T>
inline sycl::event rotmg(Func func, sycl::queue& queue, T* d1, T* d2, T* x1, T y1, T* param,
                         const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto d1_ = reinterpret_cast<rocDataType*>(d1);
            auto d2_ = reinterpret_cast<rocDataType*>(d2);
            auto x1_ = reinterpret_cast<rocDataType*>(x1);
            auto y1_ = reinterpret_cast<const rocDataType*>(&y1);
            auto param_ = reinterpret_cast<rocDataType*>(param);
            rocblas_status err;
            rocblas_native_func(func, err, handle, d1_, d2_, x1_, y1_, param_);
        });
    });

    return done;
}

#define ROTMG_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event rotmg(sycl::queue& queue, TYPE* d1, TYPE* d2, TYPE* x1, TYPE y1, TYPE* param, \
                      const std::vector<sycl::event>& dependencies) {                         \
        return rotmg(ROCBLAS_ROUTINE, queue, d1, d2, x1, y1, param, dependencies);            \
    }

ROTMG_LAUNCHER_USM(float, rocblas_srotmg)
ROTMG_LAUNCHER_USM(double, rocblas_drotmg)

#undef ROTMG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamax(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                         int64_t* result, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    // rocBLAS does not support int64_t as return type for the data by default. So we need to
    // mimic iamax. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size.
    auto int_res_p = (int*)sycl::aligned_alloc_shared(64, sizeof(rocblas_int), queue.get_device(),
                                                      queue.get_context());
    *int_res_p = 0;

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
            auto x_ = reinterpret_cast<const rocDataType*>(x);
            auto int_res_p_ = reinterpret_cast<int*>(int_res_p);
            rocblas_status err;
            // For negative incx, iamax returns 0. This behaviour is similar to that of
            // reference iamax.
            rocblas_native_func(func, err, handle, n, x_, incx, int_res_p_);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    done.wait_and_throw();
    result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
    return done;
}

#define IAMAX_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event iamax(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {  \
        return iamax(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);         \
    }

IAMAX_LAUNCHER_USM(float, rocblas_isamax)
IAMAX_LAUNCHER_USM(double, rocblas_idamax)
IAMAX_LAUNCHER_USM(std::complex<float>, rocblas_icamax)
IAMAX_LAUNCHER_USM(std::complex<double>, rocblas_izamax)

#undef IAMAX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event swap(Func func, sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx, incy);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);

            auto x_ = reinterpret_cast<rocDataType*>(x);
            auto y_ = reinterpret_cast<rocDataType*>(y);
            rocblas_status err;
            rocblas_native_func(func, err, handle, n, x_, incx, y_, incy);
        });
    });

    return done;
}

#define SWAP_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event swap(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                              \
        return swap(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);                   \
    }

SWAP_LAUNCHER_USM(float, rocblas_sswap)
SWAP_LAUNCHER_USM(double, rocblas_dswap)
SWAP_LAUNCHER_USM(std::complex<float>, rocblas_cswap)
SWAP_LAUNCHER_USM(std::complex<double>, rocblas_zswap)

#undef SWAP_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamin(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                         int64_t* result, const std::vector<sycl::event>& dependencies) {
    using rocDataType = typename RocEquivalentType<T>::Type;
    overflow_check(n, incx);
    // rocBLAS does not support int64_t as return type for the data by default. So we need to
    // mimic iamin. We are converting the result to be the int and then we convert
    // it back to the actual data on the host.
    // This change may cause failure as the result of integer overflow
    // based on the size.
    auto int_res_p = (int*)sycl::aligned_alloc_shared(64, sizeof(rocblas_int), queue.get_device(),
                                                      queue.get_context());
    *int_res_p = 0;

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

            auto x_ = reinterpret_cast<const rocDataType*>(x);
            auto int_res_p_ = reinterpret_cast<int*>(int_res_p);
            rocblas_status err;
            // For negative incx, iamin returns 0. This behaviour is similar to that of
            // implemented iamin.
            rocblas_native_func(func, err, handle, n, x_, incx, int_res_p_);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    done.wait_and_throw();
    result[0] = std::max((int64_t)(*int_res_p - 1), int64_t{ 0 });
    return done;
}

#define IAMIN_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event iamin(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {  \
        return iamin(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);         \
    }

IAMIN_LAUNCHER_USM(float, rocblas_isamin)
IAMIN_LAUNCHER_USM(double, rocblas_idamin)
IAMIN_LAUNCHER_USM(std::complex<float>, rocblas_icamin)
IAMIN_LAUNCHER_USM(std::complex<double>, rocblas_izamin)

#undef IAMIN_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event nrm2(Func func, sycl::queue& queue, int64_t n, const T1* x, const int64_t incx,
                        T2* result, const std::vector<sycl::event>& dependencies) {
    using rocDataType1 = typename RocEquivalentType<T1>::Type;
    using rocDataType2 = typename RocEquivalentType<T2>::Type;
    overflow_check(n, incx);

    auto done = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        onemath_rocblas_host_task(cgh, queue, [=](RocblasScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

            auto x_ = reinterpret_cast<const rocDataType1*>(x);
            auto res_ = reinterpret_cast<rocDataType2*>(result);
            rocblas_status err;
            // NRM2 does not support negative index
            rocblas_native_func(func, err, handle, n, x_, std::abs(incx), res_);
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        });
    });

    return done;
}

#define NRM2_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                \
    sycl::event nrm2(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx, \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {     \
        return nrm2(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);          \
    }

NRM2_LAUNCHER_USM(float, float, rocblas_snrm2)
NRM2_LAUNCHER_USM(double, double, rocblas_dnrm2)
NRM2_LAUNCHER_USM(std::complex<float>, float, rocblas_scnrm2)
NRM2_LAUNCHER_USM(std::complex<double>, double, rocblas_dznrm2)

#undef NRM2_LAUNCHER_USM

} // namespace column_major
namespace row_major {

// Buffer APIs

template <typename Func, typename T1, typename T2>
inline void asum(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                 const int64_t incx, sycl::buffer<T2, 1>& result) {
    column_major::asum(func, queue, n, x, incx, result);
}

#define ASUM_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                        \
    void asum(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        asum(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                   \
    }

ASUM_LAUNCHER(float, float, rocblas_sasum)
ASUM_LAUNCHER(double, double, rocblas_dasum)
ASUM_LAUNCHER(std::complex<float>, float, rocblas_scasum)
ASUM_LAUNCHER(std::complex<double>, double, rocblas_dzasum)

#undef ASUM_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void scal(Func func, sycl::queue& queue, int64_t n, T1 a, sycl::buffer<T2, 1>& x,
                 int64_t incx) {
    column_major::scal(func, queue, n, a, x, incx);
}

#define SCAL_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                             \
    void scal(sycl::queue& queue, int64_t n, TYPE1 a, sycl::buffer<TYPE2, 1>& x, int64_t incx) { \
        scal(ROCBLAS_ROUTINE, queue, n, a, x, incx);                                             \
    }

SCAL_LAUNCHER(float, float, rocblas_sscal)
SCAL_LAUNCHER(double, double, rocblas_dscal)
SCAL_LAUNCHER(std::complex<float>, std::complex<float>, rocblas_cscal)
SCAL_LAUNCHER(std::complex<double>, std::complex<double>, rocblas_zscal)
SCAL_LAUNCHER(float, std::complex<float>, rocblas_csscal)
SCAL_LAUNCHER(double, std::complex<double>, rocblas_zdscal)

#undef SCAL_LAUNCHER

template <typename Func, typename T>
inline void axpy(Func func, sycl::queue& queue, int64_t n, T alpha, sycl::buffer<T, 1>& x,
                 int64_t incx, sycl::buffer<T, 1>& y, int64_t incy) {
    column_major::axpy(func, queue, n, alpha, x, incx, y, incy);
}

#define AXPY_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                                     \
    void axpy(sycl::queue& queue, int64_t n, TYPE alpha, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                                          \
        axpy(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy);                                \
    }

AXPY_LAUNCHER(float, rocblas_saxpy)
AXPY_LAUNCHER(double, rocblas_daxpy)
AXPY_LAUNCHER(std::complex<float>, rocblas_caxpy)
AXPY_LAUNCHER(std::complex<double>, rocblas_zaxpy)

#undef AXPY_LAUNCHER

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

template <typename Func, typename T1, typename T2>
inline void rotg(Func func, sycl::queue& queue, sycl::buffer<T1, 1>& a, sycl::buffer<T1, 1>& b,
                 sycl::buffer<T2, 1>& c, sycl::buffer<T1, 1>& s) {
    column_major::rotg(func, queue, a, b, c, s);
}

#define ROTG_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                    \
    void rotg(sycl::queue& queue, sycl::buffer<TYPE1, 1>& a, sycl::buffer<TYPE1, 1>& b, \
              sycl::buffer<TYPE2, 1>& c, sycl::buffer<TYPE1, 1>& s) {                   \
        rotg(ROCBLAS_ROUTINE, queue, a, b, c, s);                                       \
    }

ROTG_LAUNCHER(float, float, rocblas_srotg)
ROTG_LAUNCHER(double, double, rocblas_drotg)
ROTG_LAUNCHER(std::complex<float>, float, rocblas_crotg)
ROTG_LAUNCHER(std::complex<double>, double, rocblas_zrotg)

#undef ROTG_LAUNCHER

template <typename Func, typename T>
inline void rotm(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& param) {
    column_major::rotm(func, queue, n, x, incx, y, incy, param);
}

#define ROTM_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                          \
    void rotm(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx,  \
              sycl::buffer<TYPE, 1>& y, int64_t incy, sycl::buffer<TYPE, 1>& param) { \
        rotm(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, param);                     \
    }

ROTM_LAUNCHER(float, rocblas_srotm)
ROTM_LAUNCHER(double, rocblas_drotm)

#undef ROTM_LAUNCHER

template <typename Func, typename T>
inline void copy(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    column_major::copy(func, queue, n, x, incx, y, incy);
}

#define COPY_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void copy(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        copy(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy);                           \
    }

COPY_LAUNCHER(float, rocblas_scopy)
COPY_LAUNCHER(double, rocblas_dcopy)
COPY_LAUNCHER(std::complex<float>, rocblas_ccopy)
COPY_LAUNCHER(std::complex<double>, rocblas_zcopy)

#undef COPY_LAUNCHER

template <typename Func, typename T>
inline void dot(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, const int64_t incx,
                sycl::buffer<T, 1>& y, int64_t incy, sycl::buffer<T, 1>& result) {
    column_major::dot(func, queue, n, x, incx, y, incy, result);
}

#define DOT_LAUNCHER(EXT, TYPE, ROCBLAS_ROUTINE)                                                 \
    void dot##EXT(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx,   \
                  sycl::buffer<TYPE, 1>& y, const int64_t incy, sycl::buffer<TYPE, 1>& result) { \
        dot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, result);                                \
    }

DOT_LAUNCHER(, float, rocblas_sdot)
DOT_LAUNCHER(, double, rocblas_ddot)
DOT_LAUNCHER(u, std::complex<float>, rocblas_cdotu)
DOT_LAUNCHER(c, std::complex<float>, rocblas_cdotc)
DOT_LAUNCHER(u, std::complex<double>, rocblas_zdotu)
DOT_LAUNCHER(c, std::complex<double>, rocblas_zdotc)

#undef DOT_LAUNCHER

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename Func, typename T1, typename T2, typename T3>
inline void rot(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                const int64_t incx, sycl::buffer<T1, 1>& y, int64_t incy, T2 c, T3 s) {
    column_major::rot(func, queue, n, x, incx, y, incy, c, s);
}

#define ROT_LAUNCHER(TYPE1, TYPE2, TYPE3, ROCBLAS_ROUTINE)                                 \
    void rot(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
             sycl::buffer<TYPE1, 1>& y, int64_t incy, TYPE2 c, TYPE3 s) {                  \
        rot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s);                            \
    }

ROT_LAUNCHER(float, float, float, rocblas_srot)
ROT_LAUNCHER(double, double, double, rocblas_drot)
ROT_LAUNCHER(std::complex<float>, float, float, rocblas_csrot)
ROT_LAUNCHER(std::complex<double>, double, double, rocblas_zdrot)

#undef ROT_LAUNCHER

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result) {
    column_major::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

template <typename Func, typename T>
inline void rotmg(Func func, sycl::queue& queue, sycl::buffer<T, 1>& d1, sycl::buffer<T, 1>& d2,
                  sycl::buffer<T, 1>& x1, T y1, sycl::buffer<T, 1>& param) {
    column_major::rotmg(func, queue, d1, d2, x1, y1, param);
}

#define ROTMG_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                            \
    void rotmg(sycl::queue& queue, sycl::buffer<TYPE, 1>& d1, sycl::buffer<TYPE, 1>& d2, \
               sycl::buffer<TYPE, 1>& x1, TYPE y1, sycl::buffer<TYPE, 1>& param) {       \
        rotmg(ROCBLAS_ROUTINE, queue, d1, d2, x1, y1, param);                            \
    }

ROTMG_LAUNCHER(float, rocblas_srotmg)
ROTMG_LAUNCHER(double, rocblas_drotmg)

#undef ROTMG_LAUNCHER

template <typename Func, typename T>
inline void iamax(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x,
                  const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    column_major::iamax(func, queue, n, x, incx, result);
}

#define IAMAX_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void iamax(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamax(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                  \
    }

IAMAX_LAUNCHER(float, rocblas_isamax)
IAMAX_LAUNCHER(double, rocblas_idamax)
IAMAX_LAUNCHER(std::complex<float>, rocblas_icamax)
IAMAX_LAUNCHER(std::complex<double>, rocblas_izamax)

#undef IAMAX_LAUNCHER

template <typename Func, typename T>
inline void swap(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x, int64_t incx,
                 sycl::buffer<T, 1>& y, int64_t incy) {
    column_major::swap(func, queue, n, x, incx, y, incy);
}

#define SWAP_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                         \
    void swap(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, int64_t incx, \
              sycl::buffer<TYPE, 1>& y, int64_t incy) {                              \
        swap(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy);                           \
    }

SWAP_LAUNCHER(float, rocblas_sswap)
SWAP_LAUNCHER(double, rocblas_dswap)
SWAP_LAUNCHER(std::complex<float>, rocblas_cswap)
SWAP_LAUNCHER(std::complex<double>, rocblas_zswap)

#undef SWAP_LAUNCHER

template <typename Func, typename T>
inline void iamin(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T, 1>& x,
                  const int64_t incx, sycl::buffer<int64_t, 1>& result) {
    column_major::iamin(func, queue, n, x, incx, result);
}

#define IAMIN_LAUNCHER(TYPE, ROCBLAS_ROUTINE)                                               \
    void iamin(sycl::queue& queue, int64_t n, sycl::buffer<TYPE, 1>& x, const int64_t incx, \
               sycl::buffer<int64_t, 1>& result) {                                          \
        iamin(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                  \
    }

IAMIN_LAUNCHER(float, rocblas_isamin)
IAMIN_LAUNCHER(double, rocblas_idamin)
IAMIN_LAUNCHER(std::complex<float>, rocblas_icamin)
IAMIN_LAUNCHER(std::complex<double>, rocblas_izamin)

#undef IAMIN_LAUNCHER

template <typename Func, typename T1, typename T2>
inline void nrm2(Func func, sycl::queue& queue, int64_t n, sycl::buffer<T1, 1>& x,
                 const int64_t incx, sycl::buffer<T2, 1>& result) {
    column_major::nrm2(func, queue, n, x, incx, result);
}

#define NRM2_LAUNCHER(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                        \
    void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<TYPE1, 1>& x, const int64_t incx, \
              sycl::buffer<TYPE2, 1>& result) {                                             \
        nrm2(ROCBLAS_ROUTINE, queue, n, x, incx, result);                                   \
    }

NRM2_LAUNCHER(float, float, rocblas_snrm2)
NRM2_LAUNCHER(double, double, rocblas_dnrm2)
NRM2_LAUNCHER(std::complex<float>, float, rocblas_scnrm2)
NRM2_LAUNCHER(std::complex<double>, double, rocblas_dznrm2)

#undef NRM2_LAUNCHER

// USM APIs

template <typename Func, typename T1, typename T2>
inline sycl::event asum(Func func, sycl::queue& queue, int64_t n, const T1* x, const int64_t incx,
                        T2* result, const std::vector<sycl::event>& dependencies) {
    return column_major::asum(func, queue, n, x, incx, result, dependencies);
}

#define ASUM_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                \
    sycl::event asum(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx, \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {     \
        return asum(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);          \
    }

ASUM_LAUNCHER_USM(float, float, rocblas_sasum)
ASUM_LAUNCHER_USM(double, double, rocblas_dasum)
ASUM_LAUNCHER_USM(std::complex<float>, float, rocblas_scasum)
ASUM_LAUNCHER_USM(std::complex<double>, double, rocblas_dzasum)

#undef ASUM_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event scal(Func func, sycl::queue& queue, int64_t n, T1 a, T2* x, int64_t incx,
                        const std::vector<sycl::event>& dependencies) {
    return column_major::scal(func, queue, n, a, x, incx, dependencies);
}

#define SCAL_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                             \
    sycl::event scal(sycl::queue& queue, int64_t n, TYPE1 a, TYPE2* x, int64_t incx, \
                     const std::vector<sycl::event>& dependencies) {                 \
        return scal(ROCBLAS_ROUTINE, queue, n, a, x, incx, dependencies);            \
    }

SCAL_LAUNCHER_USM(float, float, rocblas_sscal)
SCAL_LAUNCHER_USM(double, double, rocblas_dscal)
SCAL_LAUNCHER_USM(std::complex<float>, std::complex<float>, rocblas_cscal)
SCAL_LAUNCHER_USM(std::complex<double>, std::complex<double>, rocblas_zscal)
SCAL_LAUNCHER_USM(float, std::complex<float>, rocblas_csscal)
SCAL_LAUNCHER_USM(double, std::complex<double>, rocblas_zdscal)

#undef SCAL_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event axpy(Func func, sycl::queue& queue, int64_t n, T alpha, const T* x, int64_t incx,
                        T* y, int64_t incy, const std::vector<sycl::event>& dependencies) {
    return column_major::axpy(func, queue, n, alpha, x, incx, y, incy, dependencies);
}

#define AXPY_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event axpy(sycl::queue& queue, int64_t n, TYPE alpha, const TYPE* x, int64_t incx, \
                     TYPE* y, int64_t incy, const std::vector<sycl::event>& dependencies) {  \
        return axpy(ROCBLAS_ROUTINE, queue, n, alpha, x, incx, y, incy, dependencies);       \
    }

AXPY_LAUNCHER_USM(float, rocblas_saxpy)
AXPY_LAUNCHER_USM(double, rocblas_daxpy)
AXPY_LAUNCHER_USM(std::complex<float>, rocblas_caxpy)
AXPY_LAUNCHER_USM(std::complex<double>, rocblas_zaxpy)

#undef AXPY_LAUNCHER_USM

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}
sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", "for row_major layout");
}

template <typename Func, typename T1, typename T2>
inline sycl::event rotg(Func func, sycl::queue& queue, T1* a, T1* b, T2* c, T1* s,
                        const std::vector<sycl::event>& dependencies) {
    return column_major::rotg(func, queue, a, b, c, s, dependencies);
}

#define ROTG_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                         \
    sycl::event rotg(sycl::queue& queue, TYPE1* a, TYPE1* b, TYPE2* c, TYPE1* s, \
                     const std::vector<sycl::event>& dependencies) {             \
        return rotg(ROCBLAS_ROUTINE, queue, a, b, c, s, dependencies);           \
    }

ROTG_LAUNCHER_USM(float, float, rocblas_srotg)
ROTG_LAUNCHER_USM(double, double, rocblas_drotg)
ROTG_LAUNCHER_USM(std::complex<float>, float, rocblas_crotg)
ROTG_LAUNCHER_USM(std::complex<double>, double, rocblas_zrotg)

#undef ROTG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event rotm(Func func, sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y,
                        int64_t incy, T* param, const std::vector<sycl::event>& dependencies) {
    return column_major::rotm(func, queue, n, x, incx, y, incy, param, dependencies);
}

#define ROTM_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event rotm(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     TYPE* param, const std::vector<sycl::event>& dependencies) {                 \
        return rotm(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, param, dependencies);            \
    }

ROTM_LAUNCHER_USM(float, rocblas_srotm)
ROTM_LAUNCHER_USM(double, rocblas_drotm)

#undef ROTM_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event copy(Func func, sycl::queue& queue, int64_t n, const T* x, int64_t incx, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    return column_major::copy(func, queue, n, x, incx, y, incy, dependencies);
}

#define COPY_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                          \
    sycl::event copy(sycl::queue& queue, int64_t n, const TYPE* x, int64_t incx, TYPE* y, \
                     int64_t incy, const std::vector<sycl::event>& dependencies) {        \
        return copy(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);           \
    }

COPY_LAUNCHER_USM(float, rocblas_scopy)
COPY_LAUNCHER_USM(double, rocblas_dcopy)
COPY_LAUNCHER_USM(std::complex<float>, rocblas_ccopy)
COPY_LAUNCHER_USM(std::complex<double>, rocblas_zcopy)

#undef COPY_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event dot(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                       const T* y, int64_t incy, T* result,
                       const std::vector<sycl::event>& dependencies) {
    return column_major::dot(func, queue, n, x, incx, y, incy, result, dependencies);
}

#define DOT_LAUNCHER_USM(EXT, TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event dot##EXT(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                         const TYPE* y, const int64_t incy, TYPE* result,                  \
                         const std::vector<sycl::event>& dependencies) {                   \
        return dot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, result, dependencies);     \
    }

DOT_LAUNCHER_USM(, float, rocblas_sdot)
DOT_LAUNCHER_USM(, double, rocblas_ddot)
DOT_LAUNCHER_USM(u, std::complex<float>, rocblas_cdotu)
DOT_LAUNCHER_USM(c, std::complex<float>, rocblas_cdotc)
DOT_LAUNCHER_USM(u, std::complex<double>, rocblas_zdotu)
DOT_LAUNCHER_USM(c, std::complex<double>, rocblas_zdotc)

#undef DOT_LAUNCHER_USM

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", "for row_major layout");
}

template <typename Func, typename T1, typename T2, typename T3>
inline sycl::event rot(Func func, sycl::queue& queue, int64_t n, T1* x, const int64_t incx, T1* y,
                       int64_t incy, T2 c, T3 s, const std::vector<sycl::event>& dependencies) {
    return column_major::rot(func, queue, n, x, incx, y, incy, c, s, dependencies);
}

#define ROT_LAUNCHER_USM(TYPE1, TYPE2, TYPE3, ROCBLAS_ROUTINE)                             \
    sycl::event rot(sycl::queue& queue, int64_t n, TYPE1* x, const int64_t incx, TYPE1* y, \
                    int64_t incy, TYPE2 c, TYPE3 s,                                        \
                    const std::vector<sycl::event>& dependencies) {                        \
        return rot(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, c, s, dependencies);       \
    }

ROT_LAUNCHER_USM(float, float, float, rocblas_srot)
ROT_LAUNCHER_USM(double, double, double, rocblas_drot)
ROT_LAUNCHER_USM(std::complex<float>, float, float, rocblas_csrot)
ROT_LAUNCHER_USM(std::complex<double>, double, double, rocblas_zdrot)

#undef ROT_LAUNCHER_USM

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies) {
    return column_major::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

template <typename Func, typename T>
inline sycl::event rotmg(Func func, sycl::queue& queue, T* d1, T* d2, T* x1, T y1, T* param,
                         const std::vector<sycl::event>& dependencies) {
    return column_major::rotmg(func, queue, d1, d2, x1, y1, param, dependencies);
}

#define ROTMG_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                             \
    sycl::event rotmg(sycl::queue& queue, TYPE* d1, TYPE* d2, TYPE* x1, TYPE y1, TYPE* param, \
                      const std::vector<sycl::event>& dependencies) {                         \
        return rotmg(ROCBLAS_ROUTINE, queue, d1, d2, x1, y1, param, dependencies);            \
    }

ROTMG_LAUNCHER_USM(float, rocblas_srotmg)
ROTMG_LAUNCHER_USM(double, rocblas_drotmg)

#undef ROTMG_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamax(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                         int64_t* result, const std::vector<sycl::event>& dependencies) {
    return column_major::iamax(func, queue, n, x, incx, result, dependencies);
}

#define IAMAX_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event iamax(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {  \
        return iamax(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);         \
    }

IAMAX_LAUNCHER_USM(float, rocblas_isamax)
IAMAX_LAUNCHER_USM(double, rocblas_idamax)
IAMAX_LAUNCHER_USM(std::complex<float>, rocblas_icamax)
IAMAX_LAUNCHER_USM(std::complex<double>, rocblas_izamax)

#undef IAMAX_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event swap(Func func, sycl::queue& queue, int64_t n, T* x, int64_t incx, T* y,
                        int64_t incy, const std::vector<sycl::event>& dependencies) {
    return column_major::swap(func, queue, n, x, incx, y, incy, dependencies);
}

#define SWAP_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                                  \
    sycl::event swap(sycl::queue& queue, int64_t n, TYPE* x, int64_t incx, TYPE* y, int64_t incy, \
                     const std::vector<sycl::event>& dependencies) {                              \
        return swap(ROCBLAS_ROUTINE, queue, n, x, incx, y, incy, dependencies);                   \
    }

SWAP_LAUNCHER_USM(float, rocblas_sswap)
SWAP_LAUNCHER_USM(double, rocblas_dswap)
SWAP_LAUNCHER_USM(std::complex<float>, rocblas_cswap)
SWAP_LAUNCHER_USM(std::complex<double>, rocblas_zswap)

#undef SWAP_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event iamin(Func func, sycl::queue& queue, int64_t n, const T* x, const int64_t incx,
                         int64_t* result, const std::vector<sycl::event>& dependencies) {
    return column_major::iamin(func, queue, n, x, incx, result, dependencies);
}

#define IAMIN_LAUNCHER_USM(TYPE, ROCBLAS_ROUTINE)                                       \
    sycl::event iamin(sycl::queue& queue, int64_t n, const TYPE* x, const int64_t incx, \
                      int64_t* result, const std::vector<sycl::event>& dependencies) {  \
        return iamin(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);         \
    }

IAMIN_LAUNCHER_USM(float, rocblas_isamin)
IAMIN_LAUNCHER_USM(double, rocblas_idamin)
IAMIN_LAUNCHER_USM(std::complex<float>, rocblas_icamin)
IAMIN_LAUNCHER_USM(std::complex<double>, rocblas_izamin)

#undef IAMIN_LAUNCHER_USM

template <typename Func, typename T1, typename T2>
inline sycl::event nrm2(Func func, sycl::queue& queue, int64_t n, const T1* x, const int64_t incx,
                        T2* result, const std::vector<sycl::event>& dependencies) {
    return column_major::nrm2(func, queue, n, x, incx, result, dependencies);
}

#define NRM2_LAUNCHER_USM(TYPE1, TYPE2, ROCBLAS_ROUTINE)                                \
    sycl::event nrm2(sycl::queue& queue, int64_t n, const TYPE1* x, const int64_t incx, \
                     TYPE2* result, const std::vector<sycl::event>& dependencies) {     \
        return nrm2(ROCBLAS_ROUTINE, queue, n, x, incx, result, dependencies);          \
    }

NRM2_LAUNCHER_USM(float, float, rocblas_snrm2)
NRM2_LAUNCHER_USM(double, double, rocblas_dnrm2)
NRM2_LAUNCHER_USM(std::complex<float>, float, rocblas_scnrm2)
NRM2_LAUNCHER_USM(std::complex<double>, double, rocblas_dznrm2)

#undef NRM2_LAUNCHER_USM

} // namespace row_major
} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
