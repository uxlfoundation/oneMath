/*******************************************************************************
 * Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) 
 * and Computing Centre (URZ)
 * cuRAND back-end Copyright (c) 2021, The Regents of the University of
 * California, through Lawrence Berkeley National Laboratory (subject to receipt
 * of any required approvals from the U.S. Dept. of Energy). All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * (1) Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * (2) Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * (3) Neither the name of the University of California, Lawrence Berkeley
 * National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches,
 * or upgrades to the features, functionality or performance of the source
 * code ("Enhancements") to anyone; however, if you choose to make your
 * Enhancements available either publicly, or directly to Lawrence Berkeley
 * National Laboratory, without imposing a separate written license agreement
 * for such Enhancements, then you hereby grant the following license: a
 * non-exclusive, royalty-free perpetual license to install, use, modify,
 * prepare derivative works, incorporate into other computer software,
 * distribute, and sublicense such enhancements or derivative works thereof,
 * in binary and source code form.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Intellectual Property Office at
 * IPO@lbl.gov.
 *
 * NOTICE.  This Software was developed under funding from the U.S. Department
 * of Energy and the U.S. Government consequently retains certain rights.  As
 * such, the U.S. Government has been granted for itself and others acting on
 * its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, distribute copies to the public, prepare derivative
 * works, and perform publicly and display publicly, and to permit others to do
 * so.
 ******************************************************************************/

/**
 * @file rocrand_helper.cpp : contains the implementation of all the routines
 * for HIP backend
 */
#ifndef ONEMATH_RNG_ROCRAND_HELPER_HPP_
#define ONEMATH_RNG_ROCRAND_HELPER_HPP_

#include <rocrand.h>
#include <complex>
#include "oneapi/math/types.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace rocrand {

// Static template functions oneapi::math::rng::rocrand::range_transform_fp for
// Buffer and USM APIs
//
// rocRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `rocrand_generate_uniform' generates uniform random numbers on
// [0, 1). This function is used to convert to range [a, b).
//
// Supported types:
//      float
//      double
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      a     - range lower bound (inclusive)
//      b     - range upper bound (exclusive)
//      r     - buffer to store transformed random numbers
template <typename T>
static inline void range_transform_fp(sycl::queue& queue, T a, T b, std::int64_t n,
                                      sycl::buffer<T, 1>& r) {
    queue.submit([&](sycl::handler& cgh) {
        auto acc = r.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::range<1>(n),
                         [=](sycl::id<1> id) { acc[id[0]] = acc[id[0]] * (b - a) + a; });
    });
}
template <typename T>
static inline sycl::event range_transform_fp(sycl::queue& queue, T a, T b, std::int64_t n, T* r) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n),
                         [=](sycl::id<1> id) { r[id[0]] = r[id[0]] * (b - a) + a; });
    });
}
template <typename T>
static inline void range_transform_fp_accurate(sycl::queue& queue, T a, T b, std::int64_t n,
                                               sycl::buffer<T, 1>& r) {
    queue.submit([&](sycl::handler& cgh) {
        auto acc = r.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            acc[id[0]] = acc[id[0]] * (b - a) + a;
            if (acc[id[0]] < a) {
                acc[id[0]] = a;
            }
            else if (acc[id[0]] > b) {
                acc[id[0]] = b;
            }
        });
    });
}
template <typename T>
static inline sycl::event range_transform_fp_accurate(sycl::queue& queue, T a, T b, std::int64_t n,
                                                      T* r) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            r[id[0]] = r[id[0]] * (b - a) + a;
            if (r[id[0]] < a) {
                r[id[0]] = a;
            }
            else if (r[id[0]] > b) {
                r[id[0]] = b;
            }
        });
    });
}

// Static template functions oneapi::math::rng::rocrand::range_transform_int for
// Buffer and USM APIs
//
// rocRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `rocrand_generate_uniform' generates uniform random numbers on
// [0, 1). This function is used to convert to range [a, b).
//
// Supported types:
//      std::int32_t
//      std::uint32_t
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      a     - range lower bound (inclusive)
//      b     - range upper bound (exclusive)
//      r     - buffer to store transformed random numbers
template <typename T>
inline void range_transform_int(sycl::queue& queue, T a, T b, std::int64_t n,
                                sycl::buffer<std::uint32_t, 1>& in, sycl::buffer<T, 1>& out) {
    queue.submit([&](sycl::handler& cgh) {
        auto acc_in = in.template get_access<sycl::access::mode::read>(cgh);
        auto acc_out = out.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(n),
                         [=](sycl::id<1> id) { acc_out[id[0]] = a + acc_in[id[0]] % (b - a); });
    });
}
template <typename T>
inline sycl::event range_transform_int(sycl::queue& queue, T a, T b, std::int64_t n,
                                       std::uint32_t* in, T* out) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n),
                         [=](sycl::id<1> id) { out[id[0]] = a + in[id[0]] % (b - a); });
    });
}

// Static template functions oneapi::math::rng::rocrand::sample_bernoulli for
// Buffer and USM APIs
//
// rocRAND has no built-in functionality to sample from a Bernoulli distribution.
// The implementation here uses uniformly-generated random numbers and returns
// the corresponding Bernoulli distribution based on a probability.
//
// Supported types:
//      std::int32_t
//      std::uint32_t
//
// Input arguments:
//      queue - the queue to submit the kernel to
//      p     - success probablity of a trial
//      in    - buffer containing uniformly-generated random numbers
//      out   - buffer to store Bernoulli
template <typename T>
static inline void sample_bernoulli_from_uniform(sycl::queue& queue, float p, std::int64_t n,
                                                 sycl::buffer<float, 1> in,
                                                 sycl::buffer<T, 1>& out) {
    queue.submit([&](sycl::handler& cgh) {
        auto acc_in = in.template get_access<sycl::access::mode::read>(cgh);
        auto acc_out = out.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>(n),
                         [=](sycl::id<1> id) { acc_out[id[0]] = acc_in[id[0]] < p; });
    });
}
template <typename T>
static inline sycl::event sample_bernoulli_from_uniform(sycl::queue& queue, float p, std::int64_t n,
                                                        float* in, T* out) {
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { out[id[0]] = in[id[0]] < p; });
    });
}

class rocrand_error : virtual public std::runtime_error {
protected:
    inline const char* rocrand_error_map(rocrand_status error) {
        switch (error) {
            case ROCRAND_STATUS_SUCCESS: return "ROCRAND_STATUS_SUCCESS";

            case ROCRAND_STATUS_NOT_CREATED: return "ROCRAND_STATUS_NOT_CREATED";

            case ROCRAND_STATUS_ALLOCATION_FAILED: return "ROCRAND_STATUS_ALLOCATION_FAILED";

            case ROCRAND_STATUS_TYPE_ERROR: return "ROCRAND_STATUS_TYPE_ERROR";

            case ROCRAND_STATUS_OUT_OF_RANGE: return "ROCRAND_STATUS_OUT_OF_RANGE";

            case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: return "ROCRAND_STATUS_LENGTH_NOT_MULTIPLE";

            case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED";

            case ROCRAND_STATUS_LAUNCH_FAILURE: return "ROCRAND_STATUS_LAUNCH_FAILURE";

            case ROCRAND_STATUS_VERSION_MISMATCH: return "ROCRAND_STATUS_VERSION_MISMATCH";

            case ROCRAND_STATUS_INTERNAL_ERROR: return "ROCRAND_STATUS_INTERNAL_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, rocrand_status).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit rocrand_error(std::string message, rocrand_status result)
            : std::runtime_error((message + std::string(rocrand_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~rocrand_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class rocm_error : virtual public std::runtime_error {
protected:
    inline const char* rocm_error_map(hipError_t result) {
        switch (result) {
            case hipSuccess: return "hipSuccess";
            case hipErrorInvalidContext: return "hipErrorInvalidContext";
            case hipErrorInvalidKernelFile: return "hipErrorInvalidKernelFile";
            case hipErrorMemoryAllocation: return "hipErrorMemoryAllocation";
            case hipErrorInitializationError: return "hipErrorInitializationError";
            case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
            case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
            case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
            case hipErrorInvalidValue: return "hipErrorInvalidValue";
            case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
            case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
            case hipErrorUnknown: return "hipErrorUnknown";
            case hipErrorInvalidResourceHandle: return "hipErrorInvalidResourceHandle";
            case hipErrorNotReady: return "hipErrorNotReady";
            case hipErrorNoDevice: return "hipErrorNoDevice";
            case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";
            case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";
            case hipErrorRuntimeMemory: return "hipErrorRuntimeMemory";
            case hipErrorRuntimeOther: return "hipErrorRuntimeOther";
            case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";
            case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";
            case hipErrorMapBufferObjectFailed: return "hipErrorMapBufferObjectFailed";

            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, hipError_t).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit rocm_error(std::string message, hipError_t result)
            : std::runtime_error((message + std::string(rocm_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~rocm_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define HIP_ERROR_FUNC(name, err, ...)                                  \
    err = name(__VA_ARGS__);                                            \
    if (err != HIP_SUCCESS) {                                           \
        throw rocm_error(std::string(#name) + std::string(" : "), err); \
    }

#define ROCRAND_CALL(func, status, ...)                                       \
    status = func(__VA_ARGS__);                                               \
    if (status != ROCRAND_STATUS_SUCCESS) {                                   \
        throw rocrand_error(std::string(#func) + std::string(" : "), status); \
    }

} // namespace rocrand
} // namespace rng
} // namespace math
} // namespace oneapi

#endif // ONEMATH_RNG_ROCRAND_HELPER_HPP_
