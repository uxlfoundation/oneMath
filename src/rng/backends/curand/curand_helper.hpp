/*******************************************************************************
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
 * @file curand_helper.cpp : contains the implementation of all the routines
 * for CUDA backend
 */
#ifndef ONEMATH_RNG_CURAND_HELPER_HPP_
#define ONEMATH_RNG_CURAND_HELPER_HPP_
#include <cuda.h>
#include <curand.h>

#include <complex>

#include "oneapi/math/types.hpp"

namespace oneapi {
namespace math {
namespace rng {
namespace curand {

class curand_error : virtual public std::runtime_error {
protected:
    inline const char* curand_error_map(curandStatus_t error) {
        switch (error) {
            case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";

            case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";

            case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";

            case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";

            case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";

            case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";

            case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

            case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";

            case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";

            case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";

            case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";

            case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, curandStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit curand_error(std::string message, curandStatus_t result)
            : std::runtime_error((message + std::string(curand_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~curand_error() throw() {}

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

#define CURAND_CALL(func, status, ...)                                       \
    status = func(__VA_ARGS__);                                              \
    if (status != CURAND_STATUS_SUCCESS) {                                   \
        throw curand_error(std::string(#func) + std::string(" : "), status); \
    }

// Static template functions oneapi::math::rng::curand::range_transform_fp for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `curandGenerateUniform' generates uniform random numbers on
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
        cgh.parallel_for(n, [=](sycl::id<1> id) { acc[id] = acc[id] * (b - a) + a; });
    });
}

template <typename T>
static inline sycl::event range_transform_fp(sycl::queue& queue, T a, T b, std::int64_t n, T* r,
                                             sycl::event dependency) {
    return queue.parallel_for(n, dependency, [=](sycl::id<1> id) { r[id] = r[id] * (b - a) + a; });
}

template <typename T>
static inline void range_transform_fp_accurate(sycl::queue& queue, T a, T b, std::int64_t n,
                                               sycl::buffer<T, 1>& r) {
    queue.submit([&](sycl::handler& cgh) {
        auto acc = r.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(n, [=](sycl::id<1> id) {
            acc[id] = acc[id] * (b - a) + a;
            if (acc[id] < a) {
                acc[id] = a;
            }
            else if (acc[id] > b) {
                acc[id] = b;
            }
        });
    });
}

template <typename T>
static inline sycl::event range_transform_fp_accurate(sycl::queue& queue, T a, T b, std::int64_t n,
                                                      T* r, sycl::event dependency) {
    return queue.parallel_for(n, dependency, [=](sycl::id<1> id) {
        r[id] = r[id] * (b - a) + a;
        if (r[id] < a) {
            r[id] = a;
        }
        else if (r[id] > b) {
            r[id] = b;
        }
    });
}

// Static template functions oneapi::math::rng::curand::range_transform_int for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to specify a custom range for sampling
// random numbers; `curandGenerateUniform' generates uniform random numbers on
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
        cgh.parallel_for(n, [=](sycl::id<1> id) { acc_out[id] = a + acc_in[id] % (b - a); });
    });
}

template <typename T>
inline sycl::event range_transform_int(sycl::queue& queue, T a, T b, std::int64_t n,
                                       std::uint32_t* in, T* out, sycl::event dependency) {
    return queue.parallel_for(n, dependency,
                              [=](sycl::id<1> id) { out[id] = a + in[id] % (b - a); });
}

// Static template functions oneapi::math::rng::curand::sample_bernoulli for
// Buffer and USM APIs
//
// cuRAND has no built-in functionality to sample from a Bernoulli distribution.
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
        cgh.parallel_for(n, [=](sycl::id<1> id) { acc_out[id] = acc_in[id] < p; });
    });
}

template <typename T>
static inline sycl::event sample_bernoulli_from_uniform(sycl::queue& queue, float p, std::int64_t n,
                                                        float* in, T* out) {
    return queue.parallel_for(n, [=](sycl::id<1> id) { out[id] = in[id] < p; });
}

} // namespace curand
} // namespace rng
} // namespace math
} // namespace oneapi

#endif // ONEMATH_RNG_CURAND_HELPER_HPP_
