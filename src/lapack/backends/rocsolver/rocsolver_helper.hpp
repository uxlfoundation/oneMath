/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright 2022 Intel Corporation
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

/**
 * @file solver_*.cpp : contain the implementation of all the routines
 * for HIP backend
 */
#ifndef _ROCSOLVER_HELPER_HPP_
#define _ROCSOLVER_HELPER_HPP_

#include <CL/sycl.hpp>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>
#include <complex>

#include "oneapi/mkl/types.hpp"
#include "runtime_support_helper.hpp"
#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/lapack/exceptions.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace rocsolver {

// The static assert to make sure that all index types used in
// oneMKL/include/oneapi/mkl/lapack.hpp interface are int64_t
template <typename... Next>
struct is_int64 : std::false_type {};

template <typename First>
struct is_int64<First> : std::is_same<std::int64_t, First> {};

template <typename First, typename... Next>
struct is_int64<First, Next...>
        : std::integral_constant<bool, std::is_same<std::int64_t, First>::value &&
                                           is_int64<Next...>::value> {};

template <typename... T>
struct Overflow {
    static void inline check(T...) {}
};

template <typename Index, typename... T>
struct Overflow<Index, T...> {
    static void inline check(Index index, T... next) {
        if (std::abs(index) >= (1LL << 31)) {
            throw std::runtime_error(
                "rocsolver index overflow. rocsolver legacy API does not support 64 bit "
                "integer as data size. Thus, the data size should not be greater than "
                "maximum supported size by 32 bit integer.");
        }
        Overflow<T...>::check(next...);
    }
};

template <typename Index, typename... Next>
void overflow_check(Index index, Next... indices) {
    static_assert(is_int64<Index, Next...>::value, "oneMKL index type must be 64 bit integer.");
    Overflow<Index, Next...>::check(index, indices...);
}

class rocsolver_error : virtual public std::runtime_error {
protected:
    inline const char* rocsolver_error_map(rocblas_status error) {
        return rocblas_status_to_string(error);
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, rocblas_status ).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit rocsolver_error(std::string message, rocblas_status result)
            : std::runtime_error((message + std::string(rocsolver_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~rocsolver_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class hip_error : virtual public std::runtime_error {
protected:
    inline const char* hip_error_map(hipError_t result) {
        return hipGetErrorName(result);
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, hipError_t).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit hip_error(std::string message, hipError_t result)
            : std::runtime_error((message + std::string(hip_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~hip_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define HIP_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                           \
    if (err != HIP_SUCCESS) {                                          \
        throw hip_error(std::string(#name) + std::string(" : "), err); \
    }

#define ROCSOLVER_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                                 \
    if (err != rocblas_status_success) {                                     \
        throw rocsolver_error(std::string(#name) + std::string(" : "), err); \
    }

#define ROCSOLVER_ERROR_FUNC_T(name, func, err, ...)                        \
    err = func(__VA_ARGS__);                                                \
    if (err != rocblas_status_success) {                                    \
        throw rocsolver_error(std::string(name) + std::string(" : "), err); \
    }

#define ROCSOLVER_ERROR_FUNC_T_SYNC(name, func, err, handle, ...)            \
    err = func(handle, __VA_ARGS__);                                         \
    if (err != rocblas_status_success) {                                     \
        throw rocsolver_error(std::string(name) + std::string(" : "), err);  \
    }                                                                        \
    hipStream_t currentStreamId;                                             \
    ROCSOLVER_ERROR_FUNC(rocblas_get_stream, err, handle, &currentStreamId); \
    hipError_t hip_err;                                                      \
    HIP_ERROR_FUNC(hipStreamSynchronize, hip_err, currentStreamId);

template <class Func, class... Types>
inline void rocsolver_native_named_func(const char* func_name, Func func, rocsolver_status err,
                                        rocsolver_handle handle, Types... args) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    ROCSOLVER_ERROR_FUNC_T(func_name, func, err, handle, args...)
#else
    ROCSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, args...)
#endif
};

inline rocblas_eform get_rocsolver_itype(std::int64_t itype) {
    switch (itype) {
        case 1: return rocblas_eform_ax;
        case 2: return rocblas_eform_abx;
        case 3: return rocblas_eform_bax;
        default: throw "Wrong itype.";
    }
}

inline rocblas_evect get_rocsolver_job(oneapi::mkl::job jobz) {
    switch (jobz) {
        case oneapi::mkl::job::V: return rocblas_evect_original;
        case oneapi::mkl::job::N: return rocblas_evect_none;
        default: throw "Wrong jobz.";
    }
}

inline rocblas_svect get_rocsolver_jobsvd(oneapi::mkl::jobsvd job) {
    switch (job) {
        case oneapi::mkl::jobsvd::N: return rocblas_svect_none;
        case oneapi::mkl::jobsvd::A: return rocblas_svect_all;
        case oneapi::mkl::jobsvd::O: return rocblas_svect_overwrite;
        case oneapi::mkl::jobsvd::S: return rocblas_svect_singular;
        default: throw "Wrong jobsvd.";
    }
}

inline rocblas_operation get_rocblas_operation(oneapi::mkl::transpose trn) {
    switch (trn) {
        case oneapi::mkl::transpose::nontrans: return rocblas_operation_none;
        case oneapi::mkl::transpose::trans: return rocblas_operation_transpose;
        case oneapi::mkl::transpose::conjtrans: return rocblas_operation_conjugate_transpose;
        default: throw "Wrong transpose Operation.";
    }
}

inline rocblas_fill get_rocblas_fill_mode(oneapi::mkl::uplo ul) {
    switch (ul) {
        case oneapi::mkl::uplo::upper: return rocblas_fill_upper;
        case oneapi::mkl::uplo::lower: return rocblas_fill_lower;
        default: throw "Wrong fill mode.";
    }
}

inline rocblas_side get_rocblas_side_mode(oneapi::mkl::side lr) {
    switch (lr) {
        case oneapi::mkl::side::left: return rocblas_side_left;
        case oneapi::mkl::side::right: return rocblas_side_right;
        default: throw "Wrong side mode.";
    }
}

inline rocblas_storev get_rocblas_generate(oneapi::mkl::generate qp) {
    switch (qp) {
        case oneapi::mkl::generate::Q: return rocblas_column_wise;
        case oneapi::mkl::generate::P: return rocblas_row_wise;
        default: throw "Wrong generate.";
    }
}

/*converting std::complex<T> to cu<T>Complex*/
/*converting sycl::half to __half*/
template <typename T>
struct RocmEquivalentType {
    using Type = T;
};
template <>
struct RocmEquivalentType<sycl::half> {
    using Type = rocblas_half;
};
template <>
struct RocmEquivalentType<std::complex<float>> {
    using Type = rocblas_float_complex;
};
template <>
struct RocmEquivalentType<std::complex<double>> {
    using Type = rocblas_double_complex;
};

/* devinfo */

inline int get_rocsolver_devinfo(sycl::queue& queue, sycl::buffer<int>& devInfo) {
    sycl::host_accessor<int, 1, sycl::access::mode::read> dev_info_{ devInfo };
    return dev_info_[0];
}

inline int get_rocsolver_devinfo(sycl::queue& queue, const int* devInfo) {
    int dev_info_;
    queue.memcpy(&dev_info_, devInfo, sizeof(int));
    queue.wait();
    return dev_info_;
}

template <typename DEVINFO_T>
inline void lapack_info_check(sycl::queue& queue, DEVINFO_T devinfo, const char* func_name,
                              const char* cufunc_name) {
    queue.wait();
    const int devinfo_ = get_rocsolver_devinfo(queue, devinfo);
    if (devinfo_ > 0)
        throw oneapi::mkl::lapack::computation_error(
            func_name, std::string(cufunc_name) + " failed with info = " + std::to_string(devinfo_),
            devinfo_);
}

} // namespace rocsolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif // _ROCSOLVER_HELPER_HPP_
