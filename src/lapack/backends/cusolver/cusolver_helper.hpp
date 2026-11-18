/***************************************************************************
*  Copyright (C) Codeplay Software Limited
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
 * @file cusolver_*.cpp : contain the implementation of all the routines
 * for CUDA backend
 */
#ifndef _CUSOLVER_HELPER_HPP_
#define _CUSOLVER_HELPER_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda.h>
#include <complex>
#include <cstdint>
#include <vector>

#include "oneapi/math/types.hpp"
#include "runtime_support_helper.hpp"
#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/lapack/exceptions.hpp"

namespace oneapi {
namespace math {
namespace lapack {
namespace cusolver {

// The static assert to make sure that all index types used in
// oneMath/include/oneapi/math/lapack.hpp interface are int64_t
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
                "Cusolver index overflow. Cusolver legacy API does not support 64 bit "
                "integer as data size. Thus, the data size should not be greater that "
                "maximum supported size by 32 bit integer.");
        }
        Overflow<T...>::check(next...);
    }
};

template <typename Index, typename... Next>
void overflow_check(Index index, Next... indices) {
    static_assert(is_int64<Index, Next...>::value, "oneMath index type must be 64 bit integer.");
    Overflow<Index, Next...>::check(index, indices...);
}

class cusolver_error : virtual public std::runtime_error {
protected:
    inline const char* cusolver_error_map(cusolverStatus_t error) {
        switch (error) {
            case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";

            case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";

            case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";

            case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";

            case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";

            case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";

            case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";

            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, cusolverStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit cusolver_error(std::string message, cusolverStatus_t result)
            : std::runtime_error((message + std::string(cusolver_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cusolver_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class cuda_error : virtual public std::runtime_error {
protected:
    inline const char* cuda_error_map(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS: return "CUDA_SUCCESS";
            case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
            case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";
            case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
            case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
            case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, CUresult).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit cuda_error(std::string message, CUresult result)
            : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cuda_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define CUDA_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                            \
    if (err != CUDA_SUCCESS) {                                          \
        throw cuda_error(std::string(#name) + std::string(" : "), err); \
    }

#define CUSOLVER_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                                \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                   \
        throw cusolver_error(std::string(#name) + std::string(" : "), err); \
    }

#define CUSOLVER_ERROR_FUNC_T(name, func, err, ...)                        \
    err = func(__VA_ARGS__);                                               \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
        throw cusolver_error(std::string(name) + std::string(" : "), err); \
    }

#define CUSOLVER_SYNC(err, handle)                                           \
    cudaStream_t currentStreamId;                                            \
    CUSOLVER_ERROR_FUNC(cusolverDnGetStream, err, handle, &currentStreamId); \
    {                                                                        \
        CUresult __cuda_err;                                                 \
        CUDA_ERROR_FUNC(cuStreamSynchronize, __cuda_err, currentStreamId);   \
    }

#define CUSOLVER_ERROR_FUNC_T_SYNC(name, func, err, handle, ...)           \
    err = func(handle, __VA_ARGS__);                                       \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
        throw cusolver_error(std::string(name) + std::string(" : "), err); \
    }                                                                      \
    CUSOLVER_SYNC(err, handle)

template <class Func, class... Types>
inline void cusolver_native_named_func(const char* func_name, Func func, cusolverStatus_t err,
                                       cusolverDnHandle_t handle, Types... args) {
#ifdef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    CUSOLVER_ERROR_FUNC_T(func_name, func, err, handle, args...)
#else
    CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, args...)
#endif
};

inline cusolverEigType_t get_cusolver_itype(std::int64_t itype) {
    switch (itype) {
        case 1: return CUSOLVER_EIG_TYPE_1;
        case 2: return CUSOLVER_EIG_TYPE_2;
        case 3: return CUSOLVER_EIG_TYPE_3;
        default: throw "Wrong itype.";
    }
}

inline cusolverEigMode_t get_cusolver_job(oneapi::math::job jobz) {
    switch (jobz) {
        case oneapi::math::job::N: return CUSOLVER_EIG_MODE_NOVECTOR;
        case oneapi::math::job::V: return CUSOLVER_EIG_MODE_VECTOR;
        default: throw "Wrong jobz.";
    }
}

inline signed char get_cusolver_jobsvd(oneapi::math::jobsvd job) {
    switch (job) {
        case oneapi::math::jobsvd::N: return 'N';
        case oneapi::math::jobsvd::A: return 'A';
        case oneapi::math::jobsvd::O: return 'O';
        case oneapi::math::jobsvd::S: return 'S';
    }
}

inline cublasOperation_t get_cublas_operation(oneapi::math::transpose trn) {
    switch (trn) {
        case oneapi::math::transpose::nontrans: return CUBLAS_OP_N;
        case oneapi::math::transpose::trans: return CUBLAS_OP_T;
        case oneapi::math::transpose::conjtrans: return CUBLAS_OP_C;
        default: throw "Wrong transpose Operation.";
    }
}

inline cublasFillMode_t get_cublas_fill_mode(oneapi::math::uplo ul) {
    switch (ul) {
        case oneapi::math::uplo::upper: return CUBLAS_FILL_MODE_UPPER;
        case oneapi::math::uplo::lower: return CUBLAS_FILL_MODE_LOWER;
        default: throw "Wrong fill mode.";
    }
}

inline cublasSideMode_t get_cublas_side_mode(oneapi::math::side lr) {
    switch (lr) {
        case oneapi::math::side::left: return CUBLAS_SIDE_LEFT;
        case oneapi::math::side::right: return CUBLAS_SIDE_RIGHT;
        default: throw "Wrong side mode.";
    }
}

inline cublasSideMode_t get_cublas_generate(oneapi::math::generate qp) {
    switch (qp) {
        case oneapi::math::generate::Q: return CUBLAS_SIDE_LEFT;
        case oneapi::math::generate::P: return CUBLAS_SIDE_RIGHT;
        default: throw "Wrong generate.";
    }
}

/*converting std::complex<T> to cu<T>Complex*/
/*converting sycl::half to __half*/
template <typename T>
struct CudaEquivalentType {
    using Type = T;
};
template <>
struct CudaEquivalentType<sycl::half> {
    using Type = __half;
};
template <>
struct CudaEquivalentType<std::complex<float>> {
    using Type = cuComplex;
};
template <>
struct CudaEquivalentType<std::complex<double>> {
    using Type = cuDoubleComplex;
};

/*converting T to the cudaDataType tag expected by the cuSOLVER 64-bit API*/
template <typename T>
struct CudaDataType;
template <>
struct CudaDataType<float> {
    static constexpr cudaDataType Type = CUDA_R_32F;
};
template <>
struct CudaDataType<double> {
    static constexpr cudaDataType Type = CUDA_R_64F;
};
template <>
struct CudaDataType<std::complex<float>> {
    static constexpr cudaDataType Type = CUDA_C_32F;
};
template <>
struct CudaDataType<std::complex<double>> {
    static constexpr cudaDataType Type = CUDA_C_64F;
};

/* 64-bit pivot API */

#if !defined(CUSOLVER_VERSION)
#define ONEMATH_CUSOLVER_VERSION 0
#else
#define ONEMATH_CUSOLVER_VERSION CUSOLVER_VERSION
#endif

// cuSOLVER exposes getrf/getrs entry points taking int64_t pivots, which lets oneMath
// hand the user's ipiv array straight through instead of converting it. They first
// appeared in CUDA 11.0 (CUSOLVER_VERSION 10600) as cusolverDnGetrf/cusolverDnGetrs and
// were renamed with an X prefix in CUDA 11.1 (11000); the old names are deprecated from
// 11.1 and removed in CUDA 12. CUDA 10.x has no 64-bit pivot API at all.
#define ONEMATH_CUSOLVER_HAS_64BIT_PIVOTS (ONEMATH_CUSOLVER_VERSION >= 10600)
#define ONEMATH_CUSOLVER_HAS_X_NAMES      (ONEMATH_CUSOLVER_VERSION >= 11000)

#if ONEMATH_CUSOLVER_HAS_64BIT_PIVOTS

#if !ONEMATH_CUSOLVER_HAS_X_NAMES
// CUDA 11.0 spelling. The pre-X getrf has no host workspace, so the wrapper reports a
// host requirement of zero and drops the host buffer arguments.
inline cusolverStatus_t cusolverDnXgetrf_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverDnParams_t params, int64_t m, int64_t n,
                                                    cudaDataType dataTypeA, const void* A,
                                                    int64_t lda, cudaDataType computeType,
                                                    size_t* workspaceInBytesOnDevice,
                                                    size_t* workspaceInBytesOnHost) {
    *workspaceInBytesOnHost = 0;
    return cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType,
                                      workspaceInBytesOnDevice);
}

inline cusolverStatus_t cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                         int64_t m, int64_t n, cudaDataType dataTypeA, void* A,
                                         int64_t lda, int64_t* ipiv, cudaDataType computeType,
                                         void* bufferOnDevice, size_t workspaceInBytesOnDevice,
                                         void* bufferOnHost, size_t workspaceInBytesOnHost,
                                         int* info) {
    return cusolverDnGetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType,
                           bufferOnDevice, workspaceInBytesOnDevice, info);
}

inline cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                         cublasOperation_t trans, int64_t n, int64_t nrhs,
                                         cudaDataType dataTypeA, const void* A, int64_t lda,
                                         const int64_t* ipiv, cudaDataType dataTypeB, void* B,
                                         int64_t ldb, int* info) {
    return cusolverDnGetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B,
                           ldb, info);
}
#endif // !ONEMATH_CUSOLVER_HAS_X_NAMES

// Owns a cusolverDnParams_t for the duration of one call. Caching it across calls is a
// possible follow-up, tracked together with the handle churn described in issue #298.
class CusolverDnParams {
public:
    CusolverDnParams() {
        cusolverStatus_t err;
        CUSOLVER_ERROR_FUNC(cusolverDnCreateParams, err, &params_);
    }
    ~CusolverDnParams() {
        cusolverDnDestroyParams(params_);
    }
    CusolverDnParams(const CusolverDnParams&) = delete;
    CusolverDnParams& operator=(const CusolverDnParams&) = delete;

    cusolverDnParams_t get() const {
        return params_;
    }

private:
    cusolverDnParams_t params_;
};

// Queries the workspace cusolverDnXgetrf needs for an m-by-n factorisation. The device part
// is served by the oneMath scratchpad; the host part has no oneMath counterpart and is
// reported separately so that call sites can provide it.
template <typename T>
inline void cusolver_xgetrf_buffer_size(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                        std::int64_t m, std::int64_t n, std::int64_t lda,
                                        size_t* device_bytes, size_t* host_bytes) {
    constexpr cudaDataType data_type = CudaDataType<T>::Type;
    cusolverStatus_t err;
    CUSOLVER_ERROR_FUNC(cusolverDnXgetrf_bufferSize, err, handle, params, m, n, data_type, nullptr,
                        lda, data_type, device_bytes, host_bytes);
}

// Runs cusolverDnXgetrf with `device_workspace` (of `device_bytes` bytes) as the device
// workspace. A non-zero host requirement is allocated here and the stream is synchronised
// before the allocation goes out of scope, since oneMath has no host scratchpad to hold it.
template <typename T>
inline void cusolver_xgetrf(const char* func_name, cusolverDnHandle_t handle,
                            cusolverDnParams_t params, std::int64_t m, std::int64_t n, void* a,
                            std::int64_t lda, std::int64_t* ipiv, void* device_workspace,
                            size_t device_bytes, size_t host_bytes, int* devinfo) {
    constexpr cudaDataType data_type = CudaDataType<T>::Type;
    cusolverStatus_t err;

    if (host_bytes > 0) {
        std::vector<char> host_workspace(host_bytes);
        CUSOLVER_ERROR_FUNC_T_SYNC(func_name, cusolverDnXgetrf, err, handle, params, m, n,
                                   data_type, a, lda, ipiv, data_type, device_workspace,
                                   device_bytes, host_workspace.data(), host_bytes, devinfo)
    }
    else {
        cusolver_native_named_func(func_name, cusolverDnXgetrf, err, handle, params, m, n,
                                   data_type, a, lda, ipiv, data_type, device_workspace,
                                   device_bytes, nullptr, size_t{ 0 }, devinfo);
    }
}

// Runs cusolverDnXgetrs. It takes no workspace at all, so nothing else is needed here.
template <typename T>
inline void cusolver_xgetrs(const char* func_name, cusolverDnHandle_t handle,
                            cusolverDnParams_t params, cublasOperation_t trans, std::int64_t n,
                            std::int64_t nrhs, const void* a, std::int64_t lda,
                            const std::int64_t* ipiv, void* b, std::int64_t ldb) {
    constexpr cudaDataType data_type = CudaDataType<T>::Type;
    cusolverStatus_t err;
    cusolver_native_named_func(func_name, cusolverDnXgetrs, err, handle, params, trans, n, nrhs,
                               data_type, a, lda, ipiv, data_type, b, ldb, nullptr);
}

#endif // ONEMATH_CUSOLVER_HAS_64BIT_PIVOTS

// Number of T elements needed to store `count` 32-bit pivots. cuSOLVER's sytrf and cuBLAS'
// getriBatched have no 64-bit pivot counterpart, so their pivot array is carved out of the
// tail of the oneMath scratchpad rather than allocated separately. That removes the separate
// allocation and the host synchronisation its release would otherwise need; routines that
// also report info still synchronise in lapack_info_check. sizeof(T) is a multiple of 4 for
// every supported type, so an offset expressed in T elements is always suitably aligned for int.
template <typename T>
inline std::int64_t pivot32_scratchpad_size(std::int64_t count) {
    return (count * static_cast<std::int64_t>(sizeof(int)) + sizeof(T) - 1) / sizeof(T);
}

/* devinfo */

inline void get_cusolver_devinfo(sycl::queue& queue, sycl::buffer<int>& devInfo,
                                 std::vector<int>& dev_info_) {
    sycl::host_accessor<int, 1, sycl::access::mode::read> dev_info_acc{ devInfo };
    for (unsigned int i = 0; i < dev_info_.size(); ++i)
        dev_info_[i] = dev_info_acc[i];
}

inline void get_cusolver_devinfo(sycl::queue& queue, const int* devInfo,
                                 std::vector<int>& dev_info_) {
    queue.wait();
    // The copy must complete before dev_info_ is read: callers inspect it as soon as this
    // returns and then release devInfo, both of which race with an outstanding copy.
    queue.memcpy(dev_info_.data(), devInfo, sizeof(int) * dev_info_.size()).wait();
}

template <typename DEVINFO_T>
inline void lapack_info_check(sycl::queue& queue, DEVINFO_T devinfo, const char* func_name,
                              const char* cufunc_name, int dev_info_size = 1) {
    std::vector<int> dev_info_(dev_info_size);
    get_cusolver_devinfo(queue, devinfo, dev_info_);
    for (const auto& val : dev_info_) {
        if (val > 0)
            throw oneapi::math::lapack::computation_error(
                func_name, std::string(cufunc_name) + " failed with info = " + std::to_string(val),
                val);
    }
}

/* batched helpers */

// Creates list of matrix/vector pointers from initial ptr and stride
// Note: user is responsible for deallocating memory
template <typename T>
T** create_ptr_list_from_stride(T* ptr, int64_t ptr_stride, int64_t batch_size) {
    T** ptr_list = (T**)malloc(sizeof(T*) * batch_size);
    for (int64_t i = 0; i < batch_size; i++)
        ptr_list[i] = ptr + i * ptr_stride;

    return ptr_list;
}

} // namespace cusolver
} // namespace lapack
} // namespace math
} // namespace oneapi
#endif // _CUSOLVER_HELPER_HPP_
