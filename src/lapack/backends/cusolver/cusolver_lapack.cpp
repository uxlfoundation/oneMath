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
#include "cusolver_helper.hpp"
#include "cusolver_task.hpp"

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/lapack/detail/cusolver/onemkl_lapack_cusolver.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace cusolver {

// BUFFER APIs

template <typename Func, typename T_A, typename T_B>
inline void gebrd(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tauq, sycl::buffer<T_A>& taup,
                  sycl::buffer<T_A>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    if (m < n)
        throw unimplemented("lapack", "gebrd", "cusolver gebrd does not support m < n");

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tauq_acc = tauq.template get_access<sycl::access::mode::write>(cgh);
        auto taup_acc = taup.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A*>(a_acc);
            auto d_ = sc.get_mem<cuDataType_B*>(d_acc);
            auto e_ = sc.get_mem<cuDataType_B*>(e_acc);
            auto tauq_ = sc.get_mem<cuDataType_A*>(tauq_acc);
            auto taup_ = sc.get_mem<cuDataType_A*>(taup_acc);
            auto scratch_ = sc.get_mem<cuDataType_A*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, d_, e_, tauq_,
                                       taup_, scratch_, scratchpad_size, nullptr);
        });
    });
}

#define GEBRD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                    \
    void gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, \
               std::int64_t lda, sycl::buffer<TYPE_B>& d, sycl::buffer<TYPE_B>& e,          \
               sycl::buffer<TYPE_A>& tauq, sycl::buffer<TYPE_A>& taup,                      \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {            \
        gebrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup,   \
              scratchpad, scratchpad_size);                                                 \
    }

GEBRD_LAUNCHER(float, float, cusolverDnSgebrd)
GEBRD_LAUNCHER(double, double, cusolverDnDgebrd)
GEBRD_LAUNCHER(std::complex<float>, float, cusolverDnCgebrd)
GEBRD_LAUNCHER(std::complex<double>, double, cusolverDnZgebrd)

#undef GEBRD_LAUNCHER

void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& tau, sycl::buffer<float>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& tau, sycl::buffer<double>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<std::complex<float>>& a,
           std::int64_t lda, sycl::buffer<std::complex<float>>& tau,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}
void gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline void geqrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, tau_, scratch_,
                                       scratchpad_size, nullptr);
        });
    });
}

#define GEQRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                            \
    void geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                            \
        geqrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad,  \
              scratchpad_size);                                                           \
    }

GEQRF_LAUNCHER(float, cusolverDnSgeqrf)
GEQRF_LAUNCHER(double, cusolverDnDgeqrf)
GEQRF_LAUNCHER(std::complex<float>, cusolverDnCgeqrf)
GEQRF_LAUNCHER(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_LAUNCHER

template <typename Func, typename T>
void getrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m, std::int64_t n,
           sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer with 32-bit ints then copy over results
    std::uint64_t ipiv_size = std::min(n, m);
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });
    sycl::buffer<int> devInfo{ 1 };

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, scratch_,
                                       ipiv32_, devInfo_);
        });
    });

    // Copy from 32-bit buffer to 64-bit
    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv_acc[index] = static_cast<std::int64_t>(ipiv32_acc[index]);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define GETRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, sycl::buffer<TYPE>& a,          \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        getrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad,          \
              scratchpad_size);                                                                    \
    }

GETRF_LAUNCHER(float, cusolverDnSgetrf)
GETRF_LAUNCHER(double, cusolverDnDgetrf)
GETRF_LAUNCHER(std::complex<float>, cusolverDnCgetrf)
GETRF_LAUNCHER(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_LAUNCHER

#define GETRI_LAUNCHER(TYPE)                                                                    \
    void getri(sycl::queue& queue, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda,     \
               sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad,                \
               std::int64_t scratchpad_size) {                                                  \
        return getri_batch(queue, n, a, lda, lda * n, ipiv, n, 1, scratchpad, scratchpad_size); \
    }

GETRI_LAUNCHER(float)
GETRI_LAUNCHER(double)
GETRI_LAUNCHER(std::complex<float>)
GETRI_LAUNCHER(std::complex<double>)

#undef GETRI_LAUNCHER

// cusolverDnXgetrs does not use scratchpad memory
template <typename Func, typename T>
inline void getrs(const char* func_name, Func func, sycl::queue& queue,
                  oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
                  sycl::buffer<T>& b, std::int64_t ldb, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = ipiv.size();
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32_acc[index] = static_cast<std::int32_t>(ipiv_acc[index]);
        });
    });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto ipiv_ = sc.get_mem<std::int32_t*>(ipiv_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), n,
                                       nrhs, a_, lda, ipiv_, b_, ldb, nullptr);
        });
    });
}

#define GETRS_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                  \
    void getrs(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,                \
               std::int64_t nrhs, sycl::buffer<TYPE>& a, std::int64_t lda,                      \
               sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& b, std::int64_t ldb,       \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                  \
        getrs(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, ldb, \
              scratchpad, scratchpad_size);                                                     \
    }

GETRS_LAUNCHER(float, cusolverDnSgetrs)
GETRS_LAUNCHER(double, cusolverDnDgetrs)
GETRS_LAUNCHER(std::complex<float>, cusolverDnCgetrs)
GETRS_LAUNCHER(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void gesvd(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::jobsvd jobu,
                  oneapi::mkl::jobsvd jobvt, std::int64_t m, std::int64_t n, sycl::buffer<T_A>& a,
                  std::int64_t lda, sycl::buffer<T_B>& s, sycl::buffer<T_A>& u, std::int64_t ldu,
                  sycl::buffer<T_A>& vt, std::int64_t ldvt, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, m, lda, ldu, ldvt, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto s_acc = s.template get_access<sycl::access::mode::write>(cgh);
        auto u_acc = u.template get_access<sycl::access::mode::write>(cgh);
        auto vt_acc = vt.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A*>(a_acc);
            auto s_ = sc.get_mem<cuDataType_B*>(s_acc);
            auto u_ = sc.get_mem<cuDataType_A*>(u_acc);
            auto vt_ = sc.get_mem<cuDataType_A*>(vt_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType_A*>(scratch_acc);
            cusolverStatus_t err;
            // rwork is set to nullptr. If set it is filled with information from the superdiagonal.
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_jobsvd(jobu),
                                       get_cusolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu, vt_,
                                       ldvt, scratch_, scratchpad_size, nullptr, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define GESVD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                        \
    void gesvd(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,         \
               std::int64_t m, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda,       \
               sycl::buffer<TYPE_B>& s, sycl::buffer<TYPE_A>& u, std::int64_t ldu,              \
               sycl::buffer<TYPE_A>& vt, std::int64_t ldvt, sycl::buffer<TYPE_A>& scratchpad,   \
               std::int64_t scratchpad_size) {                                                  \
        gesvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, ldu, \
              vt, ldvt, scratchpad, scratchpad_size);                                           \
    }

GESVD_LAUNCHER(float, float, cusolverDnSgesvd)
GESVD_LAUNCHER(double, double, cusolverDnDgesvd)
GESVD_LAUNCHER(std::complex<float>, float, cusolverDnCgesvd)
GESVD_LAUNCHER(std::complex<double>, double, cusolverDnZgesvd)

#undef GESVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void heevd(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda,
                  sycl::buffer<T_B>& w, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A*>(a_acc);
            auto w_ = sc.get_mem<cuDataType_B*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType_A*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_job(jobz),
                                       get_cublas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                       scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define HEEVD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                          \
    void heevd(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, \
               sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& w,                \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                  \
        heevd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w, scratchpad,   \
              scratchpad_size);                                                                   \
    }

HEEVD_LAUNCHER(std::complex<float>, float, cusolverDnCheevd)
HEEVD_LAUNCHER(std::complex<double>, double, cusolverDnZheevd)

#undef HEEVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hegvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                  sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_A>& b, std::int64_t ldb,
                  sycl::buffer<T_B>& w, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A*>(a_acc);
            auto b_ = sc.get_mem<cuDataType_A*>(b_acc);
            auto w_ = sc.get_mem<cuDataType_B*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType_A*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_itype(itype),
                                       get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n, a_,
                                       lda, b_, ldb, w_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define HEGVD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                           \
    void hegvd(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,                      \
               oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE_A>& a, std::int64_t lda,  \
               sycl::buffer<TYPE_A>& b, std::int64_t ldb, sycl::buffer<TYPE_B>& w,                 \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                   \
        hegvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w, \
              scratchpad, scratchpad_size);                                                        \
    }

HEGVD_LAUNCHER(std::complex<float>, float, cusolverDnChegvd)
HEGVD_LAUNCHER(std::complex<double>, double, cusolverDnZhegvd)

#undef HEGVD_LAUNCHER

template <typename Func, typename T_A, typename T_B>
inline void hetrd(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T_A>& a, std::int64_t lda, sycl::buffer<T_B>& d,
                  sycl::buffer<T_B>& e, sycl::buffer<T_A>& tau, sycl::buffer<T_A>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType_A*>(a_acc);
            auto d_ = sc.get_mem<cuDataType_B*>(d_acc);
            auto e_ = sc.get_mem<cuDataType_B*>(e_acc);
            auto tau_ = sc.get_mem<cuDataType_A*>(tau_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType_A*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, d_, e_, tau_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define HETRD_LAUNCHER(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                          \
    void hetrd(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,                        \
               sycl::buffer<TYPE_A>& a, std::int64_t lda, sycl::buffer<TYPE_B>& d,                \
               sycl::buffer<TYPE_B>& e, sycl::buffer<TYPE_A>& tau,                                \
               sycl::buffer<TYPE_A>& scratchpad, std::int64_t scratchpad_size) {                  \
        hetrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad, \
              scratchpad_size);                                                                   \
    }

HETRD_LAUNCHER(std::complex<float>, float, cusolverDnChetrd)
HETRD_LAUNCHER(std::complex<double>, double, cusolverDnZhetrd)

#undef HETRD_LAUNCHER

void hetrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda, sycl::buffer<std::int64_t>& ipiv,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}
void hetrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::int64_t>& ipiv, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "hetrf");
}

template <typename Func, typename T>
inline void orgbr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_generate(vec), m, n,
                                       k, a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
}

#define ORGBR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                   \
    void orgbr(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,    \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        orgbr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                                  \
    }

ORGBR_LAUNCHER(float, cusolverDnSorgbr)
ORGBR_LAUNCHER(double, cusolverDnDorgbr)

#undef ORGBR_LAUNCHER

template <typename Func, typename T>
inline void orgqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
}

#define ORGQR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                              \
    void orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,            \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        orgqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                             \
    }

ORGQR_LAUNCHER(float, cusolverDnSorgqr)
ORGQR_LAUNCHER(double, cusolverDnDorgqr)

#undef ORGQR_LAUNCHER

template <typename Func, typename T>
inline void orgtr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
}

#define ORGTR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void orgtr(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,         \
               std::int64_t scratchpad_size) {                                                    \
        orgtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad,       \
              scratchpad_size);                                                                   \
    }

ORGTR_LAUNCHER(float, cusolverDnSorgtr)
ORGTR_LAUNCHER(double, cusolverDnDorgtr)

#undef ORGTR_LAUNCHER

template <typename Func, typename T>
inline void ormtr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read_write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_fill_mode(uplo), get_cublas_operation(trans), m,
                                       n, a_, lda, tau_, c_, ldc, scratch_, scratchpad_size,
                                       nullptr);
        });
    });
}

#define ORMTR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void ormtr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,                 \
               oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,                       \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                   \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,            \
               std::int64_t scratchpad_size) {                                                     \
        ormtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

ORMTR_LAUNCHER(float, cusolverDnSormtr)
ORMTR_LAUNCHER(double, cusolverDnDormtr)

#undef ORMTR_LAUNCHER

void ormrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<float>& a, std::int64_t lda,
           sycl::buffer<float>& tau, sycl::buffer<float>& c, std::int64_t ldc,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}
void ormrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<double>& a, std::int64_t lda,
           sycl::buffer<double>& tau, sycl::buffer<double>& c, std::int64_t ldc,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "ormrq");
}

template <typename Func, typename T>
inline void ormqr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::read>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_operation(trans), m, n, k, a_, lda, tau_, c_, ldc,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
}

#define ORMQR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void ormqr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,           \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        ormqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c,    \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

ORMQR_LAUNCHER(float, cusolverDnSormqr)
ORMQR_LAUNCHER(double, cusolverDnDormqr)

#undef ORMQR_LAUNCHER

template <typename Func, typename T>
inline void potrf(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define POTRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void potrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {  \
        potrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad,            \
              scratchpad_size);                                                                   \
    }

POTRF_LAUNCHER(float, cusolverDnSpotrf)
POTRF_LAUNCHER(double, cusolverDnDpotrf)
POTRF_LAUNCHER(std::complex<float>, cusolverDnCpotrf)
POTRF_LAUNCHER(std::complex<double>, cusolverDnZpotrf)

#undef POTRF_LAUNCHER

template <typename Func, typename T>
inline void potri(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define POTRI_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void potri(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {  \
        potri(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad,            \
              scratchpad_size);                                                                   \
    }

POTRI_LAUNCHER(float, cusolverDnSpotri)
POTRI_LAUNCHER(double, cusolverDnDpotri)
POTRI_LAUNCHER(std::complex<float>, cusolverDnCpotri)
POTRI_LAUNCHER(std::complex<double>, cusolverDnZpotri)

#undef POTRI_LAUNCHER

// cusolverDnXpotrs does not use scratchpad memory
template <typename Func, typename T>
inline void potrs(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, std::int64_t nrhs, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& b, std::int64_t ldb, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       nrhs, a_, lda, b_, ldb, nullptr);
        });
    });
}

#define POTRS_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                   \
    void potrs(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs,    \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& b, std::int64_t ldb, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        potrs(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb,         \
              scratchpad, scratchpad_size);                                                      \
    }

POTRS_LAUNCHER(float, cusolverDnSpotrs)
POTRS_LAUNCHER(double, cusolverDnDpotrs)
POTRS_LAUNCHER(std::complex<float>, cusolverDnCpotrs)
POTRS_LAUNCHER(std::complex<double>, cusolverDnZpotrs)

#undef POTRS_LAUNCHER

template <typename Func, typename T>
inline void syevd(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::job jobz,
                  oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& w, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto w_ = sc.get_mem<cuDataType*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_job(jobz),
                                       get_cublas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                       scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYEVD_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void syevd(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& w,                    \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                    \
        syevd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w, scratchpad,   \
              scratchpad_size);                                                                   \
    }

SYEVD_LAUNCHER(float, cusolverDnSsyevd)
SYEVD_LAUNCHER(double, cusolverDnDsyevd)

#undef SYEVD_LAUNCHER

template <typename Func, typename T>
inline void sygvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& b, std::int64_t ldb, sycl::buffer<T>& w,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto b_acc = b.template get_access<sycl::access::mode::read_write>(cgh);
        auto w_acc = w.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto b_ = sc.get_mem<cuDataType*>(b_acc);
            auto w_ = sc.get_mem<cuDataType*>(w_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_itype(itype),
                                       get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n, a_,
                                       lda, b_, ldb, w_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYGVD_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void sygvd(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,                      \
               oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, std::int64_t lda,    \
               sycl::buffer<TYPE>& b, std::int64_t ldb, sycl::buffer<TYPE>& w,                     \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        sygvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, ldb, w, \
              scratchpad, scratchpad_size);                                                        \
    }

SYGVD_LAUNCHER(float, cusolverDnSsygvd)
SYGVD_LAUNCHER(double, cusolverDnDsygvd)

#undef SYGVD_LAUNCH

template <typename Func, typename T>
inline void sytrd(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& d,
                  sycl::buffer<T>& e, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto d_acc = d.template get_access<sycl::access::mode::write>(cgh);
        auto e_acc = e.template get_access<sycl::access::mode::write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto d_ = sc.get_mem<cuDataType*>(d_acc);
            auto e_ = sc.get_mem<cuDataType*>(e_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, d_, e_, tau_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYTRD_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void sytrd(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& d, sycl::buffer<TYPE>& e,                    \
               sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,                           \
               std::int64_t scratchpad_size) {                                                    \
        sytrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, scratchpad, \
              scratchpad_size);                                                                   \
    }

SYTRD_LAUNCHER(float, cusolverDnSsytrd)
SYTRD_LAUNCHER(double, cusolverDnDsytrd)

#undef SYTRD_LAUNCHER

template <typename Func, typename T>
inline void sytrf(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<std::int64_t>& ipiv, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    sycl::buffer<int> devInfo{ 1 };

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer with 32-bit ints then copy over results
    std::uint64_t ipiv_size = n;
    sycl::buffer<int, 1> ipiv32(sycl::range<1>{ ipiv_size });

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::write>(cgh);
        auto devInfo_acc = devInfo.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto ipiv32_ = sc.get_mem<int*>(ipiv32_acc);
            auto devInfo_ = sc.get_mem<int*>(devInfo_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, ipiv32_, scratch_, scratchpad_size, devInfo_);
        });
    });

    // Copy from 32-bit buffer to 64-bit
    queue.submit([&](sycl::handler& cgh) {
        auto ipiv32_acc = ipiv32.template get_access<sycl::access::mode::read>(cgh);
        auto ipiv_acc = ipiv.template get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv_acc[index] = static_cast<std::int64_t>(ipiv32_acc[index]);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
}

#define SYTRF_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void sytrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a,  \
               std::int64_t lda, sycl::buffer<std::int64_t>& ipiv, sycl::buffer<TYPE>& scratchpad, \
               std::int64_t scratchpad_size) {                                                     \
        sytrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, ipiv, scratchpad,       \
              scratchpad_size);                                                                    \
    }

SYTRF_LAUNCHER(float, cusolverDnSsytrf)
SYTRF_LAUNCHER(double, cusolverDnDsytrf)
SYTRF_LAUNCHER(std::complex<float>, cusolverDnCsytrf)
SYTRF_LAUNCHER(std::complex<double>, cusolverDnZsytrf)

#undef SYTRF_LAUNCHER

void trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<float>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<double>& a,
           std::int64_t lda, sycl::buffer<double>& b, std::int64_t ldb,
           sycl::buffer<double>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, sycl::buffer<float>& a,
           std::int64_t lda, sycl::buffer<float>& b, std::int64_t ldb,
           sycl::buffer<float>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}
void trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
           oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
           sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& b, std::int64_t ldb,
           sycl::buffer<std::complex<double>>& scratchpad, std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "trtrs");
}

template <typename Func, typename T>
inline void ungbr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::generate vec,
                  std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<T>& a,
                  std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_generate(vec), m, n,
                                       k, a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
}

#define UNGBR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                   \
    void ungbr(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,    \
               std::int64_t k, sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                   \
        ungbr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                                  \
    }

UNGBR_LAUNCHER(std::complex<float>, cusolverDnCungbr)
UNGBR_LAUNCHER(std::complex<double>, cusolverDnZungbr)

#undef UNGBR_LAUNCHER

template <typename Func, typename T>
inline void ungqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                  std::int64_t n, std::int64_t k, sycl::buffer<T>& a, std::int64_t lda,
                  sycl::buffer<T>& tau, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
}

#define UNGQR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                              \
    void ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k,          \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,            \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {              \
        ungqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
              scratchpad_size);                                                             \
    }

UNGQR_LAUNCHER(std::complex<float>, cusolverDnCungqr)
UNGQR_LAUNCHER(std::complex<double>, cusolverDnZungqr)

#undef UNGQR_LAUNCHER

template <typename Func, typename T>
inline void ungtr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::uplo uplo,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
}

#define UNGTR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                    \
    void ungtr(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, sycl::buffer<TYPE>& a, \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& scratchpad,         \
               std::int64_t scratchpad_size) {                                                    \
        ungtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad,       \
              scratchpad_size);                                                                   \
    }

UNGTR_LAUNCHER(std::complex<float>, cusolverDnCungtr)
UNGTR_LAUNCHER(std::complex<double>, cusolverDnZungtr)

#undef UNGTR_LAUNCHER

void unmrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<float>>& a, std::int64_t lda,
           sycl::buffer<std::complex<float>>& tau, sycl::buffer<std::complex<float>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<float>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}
void unmrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
           std::int64_t n, std::int64_t k, sycl::buffer<std::complex<double>>& a, std::int64_t lda,
           sycl::buffer<std::complex<double>>& tau, sycl::buffer<std::complex<double>>& c,
           std::int64_t ldc, sycl::buffer<std::complex<double>>& scratchpad,
           std::int64_t scratchpad_size) {
    throw unimplemented("lapack", "unmrq");
}

template <typename Func, typename T>
inline void unmqr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, std::int64_t k,
                  sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau, sycl::buffer<T>& c,
                  std::int64_t ldc, sycl::buffer<T>& scratchpad, std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_operation(trans), m, n, k, a_, lda, tau_, c_, ldc,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
}

#define UNMQR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void unmqr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,           \
               std::int64_t m, std::int64_t n, std::int64_t k, sycl::buffer<TYPE>& a,              \
               std::int64_t lda, sycl::buffer<TYPE>& tau, sycl::buffer<TYPE>& c, std::int64_t ldc, \
               sycl::buffer<TYPE>& scratchpad, std::int64_t scratchpad_size) {                     \
        unmqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda, tau, c,    \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

UNMQR_LAUNCHER(std::complex<float>, cusolverDnCunmqr)
UNMQR_LAUNCHER(std::complex<double>, cusolverDnZunmqr)

#undef UNMQR_LAUNCHER

template <typename Func, typename T>
inline void unmtr(const char* func_name, Func func, sycl::queue& queue, oneapi::mkl::side side,
                  oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, std::int64_t m,
                  std::int64_t n, sycl::buffer<T>& a, std::int64_t lda, sycl::buffer<T>& tau,
                  sycl::buffer<T>& c, std::int64_t ldc, sycl::buffer<T>& scratchpad,
                  std::int64_t scratchpad_size) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a.template get_access<sycl::access::mode::read_write>(cgh);
        auto tau_acc = tau.template get_access<sycl::access::mode::write>(cgh);
        auto c_acc = c.template get_access<sycl::access::mode::read_write>(cgh);
        auto scratch_acc = scratchpad.template get_access<sycl::access::mode::read_write>(cgh);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = sc.get_mem<cuDataType*>(a_acc);
            auto tau_ = sc.get_mem<cuDataType*>(tau_acc);
            auto c_ = sc.get_mem<cuDataType*>(c_acc);
            auto scratch_ = sc.get_mem<cuDataType*>(scratch_acc);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_fill_mode(uplo), get_cublas_operation(trans), m,
                                       n, a_, lda, tau_, c_, ldc, scratch_, scratchpad_size,
                                       nullptr);
        });
    });
}

#define UNMTR_LAUNCHER(TYPE, CUSOLVER_ROUTINE)                                                     \
    void unmtr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,                 \
               oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,                       \
               sycl::buffer<TYPE>& a, std::int64_t lda, sycl::buffer<TYPE>& tau,                   \
               sycl::buffer<TYPE>& c, std::int64_t ldc, sycl::buffer<TYPE>& scratchpad,            \
               std::int64_t scratchpad_size) {                                                     \
        unmtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, tau, c, \
              ldc, scratchpad, scratchpad_size);                                                   \
    }

UNMTR_LAUNCHER(std::complex<float>, cusolverDnCunmtr)
UNMTR_LAUNCHER(std::complex<double>, cusolverDnZunmtr)

#undef UNMTR_LAUNCHER

// USM APIs

template <typename Func, typename T_A, typename T_B>
inline sycl::event gebrd(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* d, T_B* e, T_A* tauq,
                         T_A* taup, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    if (m < n)
        throw unimplemented("lapack", "gebrd", "cusolver gebrd does not support m < n");

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A*>(a);
            auto d_ = reinterpret_cast<cuDataType_B*>(d);
            auto e_ = reinterpret_cast<cuDataType_B*>(e);
            auto tauq_ = reinterpret_cast<cuDataType_A*>(tauq);
            auto taup_ = reinterpret_cast<cuDataType_A*>(taup);
            auto scratch_ = reinterpret_cast<cuDataType_A*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, d_, e_, tauq_,
                                       taup_, scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define GEBRD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                     \
    sycl::event gebrd(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE_A* a,             \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tauq, TYPE_A* taup,        \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                          \
                      const std::vector<sycl::event>& dependencies) {                            \
        return gebrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, d, e, tauq, taup, \
                     scratchpad, scratchpad_size, dependencies);                                 \
    }

GEBRD_LAUNCHER_USM(float, float, cusolverDnSgebrd)
GEBRD_LAUNCHER_USM(double, double, cusolverDnDgebrd)
GEBRD_LAUNCHER_USM(std::complex<float>, float, cusolverDnCgebrd)
GEBRD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZgebrd)

#undef GEBRD_LAUNCHER_USM

sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, float* a, std::int64_t lda,
                  float* tau, float* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, double* a, std::int64_t lda,
                  double* tau, double* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}
sycl::event gerqf(sycl::queue& queue, std::int64_t m, std::int64_t n, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "gerqf");
}

template <typename Func, typename T>
inline sycl::event geqrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, tau_, scratch_,
                                       scratchpad_size, nullptr);
        });
    });
    return done;
}

#define GEQRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event geqrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,                 \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return geqrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, tau, scratchpad,    \
                     scratchpad_size, dependencies);                                               \
    }

GEQRF_LAUNCHER_USM(float, cusolverDnSgeqrf)
GEQRF_LAUNCHER_USM(double, cusolverDnDgeqrf)
GEQRF_LAUNCHER_USM(std::complex<float>, cusolverDnCgeqrf)
GEQRF_LAUNCHER_USM(std::complex<double>, cusolverDnZgeqrf)

#undef GEQRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event getrf(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, T* a, std::int64_t lda, std::int64_t* ipiv, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    std::uint64_t ipiv_size = std::min(n, m);
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, a_, lda, scratch_, ipiv_,
                                       devInfo_);
        });
    });

    // Copy from 32-bit USM to 64-bit
    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv[index] = static_cast<std::int64_t>(ipiv32[index]);
        });
    });

    queue.wait();

    free(ipiv32, queue);

    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done_casting;
}

#define GETRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                               \
    sycl::event getrf(sycl::queue& queue, std::int64_t m, std::int64_t n, TYPE* a,               \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,                    \
                      std::int64_t scratchpad_size,                                              \
                      const std::vector<sycl::event>& dependencies) {                            \
        return getrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, a, lda, ipiv, scratchpad, \
                     scratchpad_size, dependencies);                                             \
    }

GETRF_LAUNCHER_USM(float, cusolverDnSgetrf)
GETRF_LAUNCHER_USM(double, cusolverDnDgetrf)
GETRF_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrf)
GETRF_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrf)

#undef GETRF_LAUNCHER_USM

#define GETRI_LAUNCHER_USM(TYPE)                                                               \
    sycl::event getri(sycl::queue& queue, std::int64_t n, TYPE* a, std::int64_t lda,           \
                      std::int64_t* ipiv, TYPE* scratchpad, std::int64_t scratchpad_size,      \
                      const std::vector<sycl::event>& dependencies) {                          \
        return getri_batch(queue, n, a, lda, lda * n, ipiv, n, 1, scratchpad, scratchpad_size, \
                           dependencies);                                                      \
    }

GETRI_LAUNCHER_USM(float)
GETRI_LAUNCHER_USM(double)
GETRI_LAUNCHER_USM(std::complex<float>)
GETRI_LAUNCHER_USM(std::complex<double>)

#undef GETRI_LAUNCHER_USM

// cusolverDnXgetrs does not use scratchpad memory
template <typename Func, typename T>
inline sycl::event getrs(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::transpose trans, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, std::int64_t* ipiv, T* b, std::int64_t ldb,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Create new buffer and convert 64-bit values.
    std::uint64_t ipiv_size = n;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv32[index] = static_cast<std::int32_t>(ipiv[index]);
        });
    });

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        cgh.depends_on(done_casting);
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_operation(trans), n,
                                       nrhs, a_, lda, ipiv_, b_, ldb, nullptr);
        });
    });

    queue.wait();

    free(ipiv32, queue);

    return done;
}

#define GETRS_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                \
    sycl::event getrs(sycl::queue& queue, oneapi::mkl::transpose trans, std::int64_t n,           \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, std::int64_t* ipiv, TYPE* b,  \
                      std::int64_t ldb, TYPE* scratchpad, std::int64_t scratchpad_size,           \
                      const std::vector<sycl::event>& dependencies) {                             \
        return getrs(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, trans, n, nrhs, a, lda, ipiv, b, \
                     ldb, scratchpad, scratchpad_size, dependencies);                             \
    }

GETRS_LAUNCHER_USM(float, cusolverDnSgetrs)
GETRS_LAUNCHER_USM(double, cusolverDnDgetrs)
GETRS_LAUNCHER_USM(std::complex<float>, cusolverDnCgetrs)
GETRS_LAUNCHER_USM(std::complex<double>, cusolverDnZgetrs)

#undef GETRS_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event gesvd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m,
                         std::int64_t n, T_A* a, std::int64_t lda, T_B* s, T_A* u, std::int64_t ldu,
                         T_A* vt, std::int64_t ldvt, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(m, n, lda, ldu, ldvt, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A*>(a);
            auto s_ = reinterpret_cast<cuDataType_B*>(s);
            auto u_ = reinterpret_cast<cuDataType_A*>(u);
            auto vt_ = reinterpret_cast<cuDataType_A*>(vt);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType_A*>(scratchpad);
            cusolverStatus_t err;
            // rwork is set to nullptr. If set it is filled with information from the superdiagonal.
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_jobsvd(jobu),
                                       get_cusolver_jobsvd(jobvt), m, n, a_, lda, s_, u_, ldu, vt_,
                                       ldvt, scratch_, scratchpad_size, nullptr, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define GESVD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                      \
    sycl::event gesvd(sycl::queue& queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,    \
                      std::int64_t m, std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* s,     \
                      TYPE_A* u, std::int64_t ldu, TYPE_A* vt, std::int64_t ldvt,                 \
                      TYPE_A* scratchpad, std::int64_t scratchpad_size,                           \
                      const std::vector<sycl::event>& dependencies) {                             \
        return gesvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, a, lda, s, u, \
                     ldu, vt, ldvt, scratchpad, scratchpad_size, dependencies);                   \
    }

GESVD_LAUNCHER_USM(float, float, cusolverDnSgesvd)
GESVD_LAUNCHER_USM(double, double, cusolverDnDgesvd)
GESVD_LAUNCHER_USM(std::complex<float>, float, cusolverDnCgesvd)
GESVD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZgesvd)

#undef GESVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event heevd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_B*& w, T_A*& scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A*>(a);
            auto w_ = reinterpret_cast<cuDataType_B*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType_A*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_job(jobz),
                                       get_cublas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                       scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define HEEVD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                      \
    sycl::event heevd(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,          \
                      std::int64_t n, TYPE_A* a, std::int64_t lda, TYPE_B* w, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return heevd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w,        \
                     scratchpad, scratchpad_size, dependencies);                                  \
    }

HEEVD_LAUNCHER_USM(std::complex<float>, float, cusolverDnCheevd)
HEEVD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZheevd)

#undef HEEVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hegvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, T_A*& a,
                         std::int64_t lda, T_A*& b, std::int64_t ldb, T_B*& w, T_A*& scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A*>(a);
            auto b_ = reinterpret_cast<cuDataType_A*>(b);
            auto w_ = reinterpret_cast<cuDataType_B*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType_A*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_itype(itype),
                                       get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n, a_,
                                       lda, b_, ldb, w_, scratch_, scratchpad_size, devInfo);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define HEGVD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                      \
    sycl::event hegvd(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,              \
                      oneapi::mkl::uplo uplo, std::int64_t n, TYPE_A* a, std::int64_t lda,        \
                      TYPE_A* b, std::int64_t ldb, TYPE_B* w, TYPE_A* scratchpad,                 \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return hegvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, \
                     ldb, w, scratchpad, scratchpad_size, dependencies);                          \
    }

HEGVD_LAUNCHER_USM(std::complex<float>, float, cusolverDnChegvd)
HEGVD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZhegvd)

#undef HEGVD_LAUNCHER_USM

template <typename Func, typename T_A, typename T_B>
inline sycl::event hetrd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T_A* a, std::int64_t lda, T_B* d,
                         T_B* e, T_A* tau, T_A* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType_A = typename CudaEquivalentType<T_A>::Type;
    using cuDataType_B = typename CudaEquivalentType<T_B>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType_A*>(a);
            auto d_ = reinterpret_cast<cuDataType_B*>(d);
            auto e_ = reinterpret_cast<cuDataType_B*>(e);
            auto tau_ = reinterpret_cast<cuDataType_A*>(tau);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType_A*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, d_, e_, tau_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define HETRD_LAUNCHER_USM(TYPE_A, TYPE_B, CUSOLVER_ROUTINE)                                   \
    sycl::event hetrd(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE_A* a,   \
                      std::int64_t lda, TYPE_B* d, TYPE_B* e, TYPE_A* tau, TYPE_A* scratchpad, \
                      std::int64_t scratchpad_size,                                            \
                      const std::vector<sycl::event>& dependencies) {                          \
        return hetrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau,   \
                     scratchpad, scratchpad_size, dependencies);                               \
    }

HETRD_LAUNCHER_USM(std::complex<float>, float, cusolverDnChetrd)
HETRD_LAUNCHER_USM(std::complex<double>, double, cusolverDnZhetrd)

#undef HETRD_LAUNCHER_USM

sycl::event hetrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<float>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "hetrf");
}
sycl::event hetrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,
                  std::complex<double>* a, std::int64_t lda, std::int64_t* ipiv,
                  std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "hetrf");
}

template <typename Func, typename T>
inline sycl::event orgbr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_generate(vec), m, n,
                                       k, a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define ORGBR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event orgbr(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,        \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau, \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                      const std::vector<sycl::event>& dependencies) {                       \
        return orgbr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, \
                     scratchpad, scratchpad_size, dependencies);                            \
    }

ORGBR_LAUNCHER_USM(float, cusolverDnSorgbr)
ORGBR_LAUNCHER_USM(double, cusolverDnDorgbr)

#undef ORGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define ORGQR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event orgqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
                     scratchpad_size, dependencies);                                               \
    }

ORGQR_LAUNCHER_USM(float, cusolverDnSorgqr)
ORGQR_LAUNCHER_USM(double, cusolverDnDorgqr)

#undef ORGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event orgtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define ORGTR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event orgtr(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a,         \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return orgtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, \
                     scratchpad_size, dependencies);                                               \
    }

ORGTR_LAUNCHER_USM(float, cusolverDnSorgtr)
ORGTR_LAUNCHER_USM(double, cusolverDnDorgtr)

#undef ORGTR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ormtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_fill_mode(uplo), get_cublas_operation(trans), m,
                                       n, a_, lda, tau_, c_, ldc, scratch_, scratchpad_size,
                                       nullptr);
        });
    });
    return done;
}

#define ORMTR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                \
    sycl::event ormtr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,         \
                      oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,      \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,   \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return ormtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                     \
    }

ORMTR_LAUNCHER_USM(float, cusolverDnSormtr)
ORMTR_LAUNCHER_USM(double, cusolverDnDormtr)

#undef ORMTR_LAUNCHER_USM

sycl::event ormrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, float* a, std::int64_t lda,
                  float* tau, float* c, std::int64_t ldc, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ormrq");
}
sycl::event ormrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, double* a, std::int64_t lda,
                  double* tau, double* c, std::int64_t ldc, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "ormrq");
}

template <typename Func, typename T>
inline sycl::event ormqr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_operation(trans), m, n, k, a_, lda, tau_, c_, ldc,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define ORMQR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                               \
    sycl::event ormqr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,  \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                    \
                      std::int64_t scratchpad_size,                                              \
                      const std::vector<sycl::event>& dependencies) {                            \
        return ormqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda,   \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                    \
    }

ORMQR_LAUNCHER_USM(float, cusolverDnSormqr)
ORMQR_LAUNCHER_USM(double, cusolverDnDormqr)

#undef ORMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potrf(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define POTRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                            \
    sycl::event potrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a,    \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,       \
                      const std::vector<sycl::event>& dependencies) {                         \
        return potrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, \
                     scratchpad_size, dependencies);                                          \
    }

POTRF_LAUNCHER_USM(float, cusolverDnSpotrf)
POTRF_LAUNCHER_USM(double, cusolverDnDpotrf)
POTRF_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrf)
POTRF_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrf)

#undef POTRF_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event potri(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define POTRI_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                            \
    sycl::event potri(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a,    \
                      std::int64_t lda, TYPE* scratchpad, std::int64_t scratchpad_size,       \
                      const std::vector<sycl::event>& dependencies) {                         \
        return potri(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, scratchpad, \
                     scratchpad_size, dependencies);                                          \
    }

POTRI_LAUNCHER_USM(float, cusolverDnSpotri)
POTRI_LAUNCHER_USM(double, cusolverDnDpotri)
POTRI_LAUNCHER_USM(std::complex<float>, cusolverDnCpotri)
POTRI_LAUNCHER_USM(std::complex<double>, cusolverDnZpotri)

#undef POTRI_LAUNCHER_USM

// cusolverDnXpotrs does not use scratchpad memory
template <typename Func, typename T>
inline sycl::event potrs(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t nrhs, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, nrhs, lda, ldb, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       nrhs, a_, lda, b_, ldb, nullptr);
        });
    });
    return done;
}

#define POTRS_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                              \
    sycl::event potrs(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n,               \
                      std::int64_t nrhs, TYPE* a, std::int64_t lda, TYPE* b, std::int64_t ldb,  \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                           \
                      const std::vector<sycl::event>& dependencies) {                           \
        return potrs(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, nrhs, a, lda, b, ldb, \
                     scratchpad, scratchpad_size, dependencies);                                \
    }

POTRS_LAUNCHER_USM(float, cusolverDnSpotrs)
POTRS_LAUNCHER_USM(double, cusolverDnDpotrs)
POTRS_LAUNCHER_USM(std::complex<float>, cusolverDnCpotrs)
POTRS_LAUNCHER_USM(std::complex<double>, cusolverDnZpotrs)

#undef POTRS_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event syevd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* w, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto w_ = reinterpret_cast<cuDataType*>(w);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_job(jobz),
                                       get_cublas_fill_mode(uplo), n, a_, lda, w_, scratch_,
                                       scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define SYEVD_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event syevd(sycl::queue& queue, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,    \
                      std::int64_t n, TYPE* a, std::int64_t lda, TYPE* w, TYPE* scratchpad, \
                      std::int64_t scratchpad_size,                                         \
                      const std::vector<sycl::event>& dependencies) {                       \
        return syevd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, a, lda, w,  \
                     scratchpad, scratchpad_size, dependencies);                            \
    }

SYEVD_LAUNCHER_USM(float, cusolverDnSsyevd)
SYEVD_LAUNCHER_USM(double, cusolverDnDsyevd)

#undef SYEVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sygvd(const char* func_name, Func func, sycl::queue& queue, std::int64_t itype,
                         oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n, T* a,
                         std::int64_t lda, T* b, std::int64_t ldb, T* w, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, ldb, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto b_ = reinterpret_cast<cuDataType*>(b);
            auto w_ = reinterpret_cast<cuDataType*>(w);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cusolver_itype(itype),
                                       get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n, a_,
                                       lda, b_, ldb, w_, scratch_, scratchpad_size, devInfo);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define SYGVD_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                \
    sycl::event sygvd(sycl::queue& queue, std::int64_t itype, oneapi::mkl::job jobz,              \
                      oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a, std::int64_t lda, TYPE* b, \
                      std::int64_t ldb, TYPE* w, TYPE* scratchpad, std::int64_t scratchpad_size,  \
                      const std::vector<sycl::event>& dependencies) {                             \
        return sygvd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n, a, lda, b, \
                     ldb, w, scratchpad, scratchpad_size, dependencies);                          \
    }

SYGVD_LAUNCHER_USM(float, cusolverDnSsygvd)
SYGVD_LAUNCHER_USM(double, cusolverDnDsygvd)

#undef SYGVD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrd(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* d, T* e,
                         T* tau, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto d_ = reinterpret_cast<cuDataType*>(d);
            auto e_ = reinterpret_cast<cuDataType*>(e);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, d_, e_, tau_, scratch_, scratchpad_size, devInfo_);
        });
    });
    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done;
}

#define SYTRD_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                           \
    sycl::event sytrd(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a,   \
                      std::int64_t lda, TYPE* d, TYPE* e, TYPE* tau, TYPE* scratchpad,       \
                      std::int64_t scratchpad_size,                                          \
                      const std::vector<sycl::event>& dependencies) {                        \
        return sytrd(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, d, e, tau, \
                     scratchpad, scratchpad_size, dependencies);                             \
    }

SYTRD_LAUNCHER_USM(float, cusolverDnSsytrd)
SYTRD_LAUNCHER_USM(double, cusolverDnDsytrd)

#undef SYTRD_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event sytrf(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda,
                         std::int64_t* ipiv, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    int* devInfo = (int*)malloc_device(sizeof(int), queue);

    // cuSolver legacy api does not accept 64-bit ints.
    // To get around the limitation.
    // Allocate memory with 32-bit ints then copy over results
    std::uint64_t ipiv_size = n;
    int* ipiv32 = (int*)malloc_device(sizeof(int) * ipiv_size, queue);

    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            auto ipiv_ = reinterpret_cast<int*>(ipiv32);
            auto devInfo_ = reinterpret_cast<int*>(devInfo);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, ipiv_, scratch_, scratchpad_size, devInfo_);
        });
    });

    // Copy from 32-bit USM to 64-bit
    auto done_casting = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(done);
        cgh.parallel_for(sycl::range<1>{ ipiv_size }, [=](sycl::id<1> index) {
            ipiv[index] = static_cast<std::int64_t>(ipiv32[index]);
        });
    });

    queue.wait();

    free(ipiv32, queue);

    lapack_info_check(queue, devInfo, __func__, func_name);
    free(devInfo, queue);
    return done_casting;
}

#define SYTRF_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                         \
    sycl::event sytrf(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a, \
                      std::int64_t lda, std::int64_t* ipiv, TYPE* scratchpad,              \
                      std::int64_t scratchpad_size,                                        \
                      const std::vector<sycl::event>& dependencies) {                      \
        return sytrf(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, ipiv,    \
                     scratchpad, scratchpad_size, dependencies);                           \
    }

SYTRF_LAUNCHER_USM(float, cusolverDnSsytrf)
SYTRF_LAUNCHER_USM(double, cusolverDnDsytrf)
SYTRF_LAUNCHER_USM(std::complex<float>, cusolverDnCsytrf)
SYTRF_LAUNCHER_USM(std::complex<double>, cusolverDnZsytrf)

#undef SYTRF_LAUNCHER_USM

sycl::event trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* b, std::int64_t ldb,
                  std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, double* a,
                  std::int64_t lda, double* b, std::int64_t ldb, double* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs, float* a,
                  std::int64_t lda, float* b, std::int64_t ldb, float* scratchpad,
                  std::int64_t scratchpad_size, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}
sycl::event trtrs(sycl::queue& queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                  oneapi::mkl::diag diag, std::int64_t n, std::int64_t nrhs,
                  std::complex<double>* a, std::int64_t lda, std::complex<double>* b,
                  std::int64_t ldb, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "trtrs");
}

template <typename Func, typename T>
inline sycl::event ungbr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::generate vec, std::int64_t m, std::int64_t n, std::int64_t k,
                         T* a, std::int64_t lda, T* tau, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_generate(vec), m, n,
                                       k, a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define UNGBR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                          \
    sycl::event ungbr(sycl::queue& queue, oneapi::mkl::generate vec, std::int64_t m,        \
                      std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, TYPE* tau, \
                      TYPE* scratchpad, std::int64_t scratchpad_size,                       \
                      const std::vector<sycl::event>& dependencies) {                       \
        return ungbr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, a, lda, tau, \
                     scratchpad, scratchpad_size, dependencies);                            \
    }

UNGBR_LAUNCHER_USM(std::complex<float>, cusolverDnCungbr)
UNGBR_LAUNCHER_USM(std::complex<double>, cusolverDnZungbr)

#undef UNGBR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungqr(const char* func_name, Func func, sycl::queue& queue, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, k, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, m, n, k, a_, lda, tau_,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define UNGQR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event ungqr(sycl::queue& queue, std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, a, lda, tau, scratchpad, \
                     scratchpad_size, dependencies);                                               \
    }

UNGQR_LAUNCHER_USM(std::complex<float>, cusolverDnCungqr)
UNGQR_LAUNCHER_USM(std::complex<double>, cusolverDnZungqr)

#undef UNGQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event ungtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::uplo uplo, std::int64_t n, T* a, std::int64_t lda, T* tau,
                         T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_fill_mode(uplo), n,
                                       a_, lda, tau_, scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define UNGTR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                 \
    sycl::event ungtr(sycl::queue& queue, oneapi::mkl::uplo uplo, std::int64_t n, TYPE* a,         \
                      std::int64_t lda, TYPE* tau, TYPE* scratchpad, std::int64_t scratchpad_size, \
                      const std::vector<sycl::event>& dependencies) {                              \
        return ungtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, a, lda, tau, scratchpad, \
                     scratchpad_size, dependencies);                                               \
    }

UNGTR_LAUNCHER_USM(std::complex<float>, cusolverDnCungtr)
UNGTR_LAUNCHER_USM(std::complex<double>, cusolverDnZungtr)

#undef UNGTR_LAUNCHER_USM

sycl::event unmrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float>* a,
                  std::int64_t lda, std::complex<float>* tau, std::complex<float>* c,
                  std::int64_t ldc, std::complex<float>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "unmrq");
}
sycl::event unmrq(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,
                  std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double>* a,
                  std::int64_t lda, std::complex<double>* tau, std::complex<double>* c,
                  std::int64_t ldc, std::complex<double>* scratchpad, std::int64_t scratchpad_size,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("lapack", "unmrq");
}

template <typename Func, typename T>
inline sycl::event unmqr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m,
                         std::int64_t n, std::int64_t k, T* a, std::int64_t lda, T* tau, T* c,
                         std::int64_t ldc, T* scratchpad, std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(n, lda, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_operation(trans), m, n, k, a_, lda, tau_, c_, ldc,
                                       scratch_, scratchpad_size, nullptr);
        });
    });
    return done;
}

#define UNMQR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                               \
    sycl::event unmqr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::transpose trans,  \
                      std::int64_t m, std::int64_t n, std::int64_t k, TYPE* a, std::int64_t lda, \
                      TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,                    \
                      std::int64_t scratchpad_size,                                              \
                      const std::vector<sycl::event>& dependencies) {                            \
        return unmqr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k, a, lda,   \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                    \
    }

UNMQR_LAUNCHER_USM(std::complex<float>, cusolverDnCunmqr)
UNMQR_LAUNCHER_USM(std::complex<double>, cusolverDnZunmqr)

#undef UNMQR_LAUNCHER_USM

template <typename Func, typename T>
inline sycl::event unmtr(const char* func_name, Func func, sycl::queue& queue,
                         oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                         oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, T* a,
                         std::int64_t lda, T* tau, T* c, std::int64_t ldc, T* scratchpad,
                         std::int64_t scratchpad_size,
                         const std::vector<sycl::event>& dependencies) {
    using cuDataType = typename CudaEquivalentType<T>::Type;
    overflow_check(m, n, lda, ldc, scratchpad_size);
    auto done = queue.submit([&](sycl::handler& cgh) {
        int64_t num_events = dependencies.size();
        for (int64_t i = 0; i < num_events; i++) {
            cgh.depends_on(dependencies[i]);
        }
        onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
            auto handle = sc.get_handle(queue);
            auto a_ = reinterpret_cast<cuDataType*>(a);
            auto tau_ = reinterpret_cast<cuDataType*>(tau);
            auto c_ = reinterpret_cast<cuDataType*>(c);
            auto scratch_ = reinterpret_cast<cuDataType*>(scratchpad);
            cusolverStatus_t err;
            cusolver_native_named_func(func_name, func, err, handle, get_cublas_side_mode(side),
                                       get_cublas_fill_mode(uplo), get_cublas_operation(trans), m,
                                       n, a_, lda, tau_, c_, ldc, scratch_, scratchpad_size,
                                       nullptr);
        });
    });
    return done;
}

#define UNMTR_LAUNCHER_USM(TYPE, CUSOLVER_ROUTINE)                                                \
    sycl::event unmtr(sycl::queue& queue, oneapi::mkl::side side, oneapi::mkl::uplo uplo,         \
                      oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n, TYPE* a,      \
                      std::int64_t lda, TYPE* tau, TYPE* c, std::int64_t ldc, TYPE* scratchpad,   \
                      std::int64_t scratchpad_size,                                               \
                      const std::vector<sycl::event>& dependencies) {                             \
        return unmtr(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, a, lda, \
                     tau, c, ldc, scratchpad, scratchpad_size, dependencies);                     \
    }

UNMTR_LAUNCHER_USM(std::complex<float>, cusolverDnCunmtr)
UNMTR_LAUNCHER_USM(std::complex<double>, cusolverDnZunmtr)

#undef UNMTR_LAUNCHER_USM

// SCRATCHPAD APIs

template <typename Func>
inline void gebrd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, scratch_size);
            });
        })
        .wait();
}

#define GEBRD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t gebrd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        gebrd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda,              \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

GEBRD_LAUNCHER_SCRATCH(float, cusolverDnSgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(double, cusolverDnDgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgebrd_bufferSize)
GEBRD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgebrd_bufferSize)

#undef GEBRD_LAUNCHER_SCRATCH

template <>
std::int64_t gerqf_scratchpad_size<float>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                          std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<double>(sycl::queue& queue, std::int64_t m, std::int64_t n,
                                           std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<float>>(sycl::queue& queue, std::int64_t m,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}
template <>
std::int64_t gerqf_scratchpad_size<std::complex<double>>(sycl::queue& queue, std::int64_t m,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "gerqf_scratchpad_size");
}

template <typename Func>
inline void geqrf_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, nullptr, lda,
                                           scratch_size);
            });
        })
        .wait();
}

#define GEQRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t geqrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        geqrf_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda,              \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

GEQRF_LAUNCHER_SCRATCH(float, cusolverDnSgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(double, cusolverDnDgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgeqrf_bufferSize)
GEQRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgeqrf_bufferSize)

#undef GEQRF_LAUNCHER_SCRATCH

template <typename Func>
inline void gesvd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  std::int64_t ldu, std::int64_t ldvt, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, scratch_size);
            });
        })
        .wait();
}

#define GESVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t gesvd_scratchpad_size<TYPE>(                                                     \
        sycl::queue & queue, oneapi::mkl::jobsvd jobu, oneapi::mkl::jobsvd jobvt, std::int64_t m, \
        std::int64_t n, std::int64_t lda, std::int64_t ldu, std::int64_t ldvt) {                  \
        int scratch_size;                                                                         \
        gesvd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobu, jobvt, m, n, lda, \
                              ldu, ldvt, &scratch_size);                                          \
        return scratch_size;                                                                      \
    }

GESVD_LAUNCHER_SCRATCH(float, cusolverDnSgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(double, cusolverDnDgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgesvd_bufferSize)
GESVD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgesvd_bufferSize)

#undef GESVD_LAUNCHER_SCRATCH

template <typename Func>
inline void getrf_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, nullptr, lda,
                                           scratch_size);
            });
        })
        .wait();
}

#define GETRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t getrf_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t lda) {                                  \
        int scratch_size;                                                                         \
        getrf_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, lda,              \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

GETRF_LAUNCHER_SCRATCH(float, cusolverDnSgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(double, cusolverDnDgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCgetrf_bufferSize)
GETRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZgetrf_bufferSize)

#undef GETRF_LAUNCHER_SCRATCH

#define GETRI_LAUNCHER_SCRATCH(TYPE)                                              \
    template <>                                                                   \
    std::int64_t getri_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t n, \
                                             std::int64_t lda) {                  \
        return lda * n;                                                           \
    }

GETRI_LAUNCHER_SCRATCH(float)
GETRI_LAUNCHER_SCRATCH(double)
GETRI_LAUNCHER_SCRATCH(std::complex<float>)
GETRI_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRI_LAUNCHER_SCRATCH

// cusolverDnXgetrs does not use scratchpad memory
#define GETRS_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t getrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::transpose trans,   \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

GETRS_LAUNCHER_SCRATCH(float)
GETRS_LAUNCHER_SCRATCH(double)
GETRS_LAUNCHER_SCRATCH(std::complex<float>)
GETRS_LAUNCHER_SCRATCH(std::complex<double>)

#undef GETRS_LAUNCHER_SCRATCH

template <typename Func>
inline void heevd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                  std::int64_t lda, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cusolver_job(jobz),
                                           get_cublas_fill_mode(uplo), n, nullptr, lda, nullptr,
                                           scratch_size);
            });
        })
        .wait();
}

#define HEEVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                        \
    template <>                                                                               \
    std::int64_t heevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::job jobz,      \
                                             oneapi::mkl::uplo uplo, std::int64_t n,          \
                                             std::int64_t lda) {                              \
        int scratch_size;                                                                     \
        heevd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, lda, \
                              &scratch_size);                                                 \
        return scratch_size;                                                                  \
    }

HEEVD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCheevd_bufferSize)
HEEVD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZheevd_bufferSize)

#undef HEEVD_LAUNCHER_SCRATCH

template <typename Func>
inline void hegvd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                  std::int64_t n, std::int64_t lda, std::int64_t ldb,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cusolver_itype(itype),
                                           get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n,
                                           nullptr, lda, nullptr, ldb, nullptr, scratch_size);
            });
        })
        .wait();
}

#define HEGVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                             \
    template <>                                                                                    \
    std::int64_t hegvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        int scratch_size;                                                                          \
        hegvd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n,    \
                              lda, ldb, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

HEGVD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnChegvd_bufferSize)
HEGVD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZhegvd_bufferSize)

#undef HEGVD_LAUNCHER_SCRATCH

template <typename Func>
inline void hetrd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, nullptr, nullptr, nullptr,
                                           scratch_size);
            });
        })
        .wait();
}

#define HETRD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t hetrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        hetrd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

HETRD_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnChetrd_bufferSize)
HETRD_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZhetrd_bufferSize)

#undef HETRD_LAUNCHER_SCRATCH

template <>
std::int64_t hetrf_scratchpad_size<std::complex<float>>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}
template <>
std::int64_t hetrf_scratchpad_size<std::complex<double>>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         std::int64_t n, std::int64_t lda) {
    throw unimplemented("lapack", "hetrf_scratchpad_size");
}

template <typename Func>
inline void orgbr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                  std::int64_t k, std::int64_t lda, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_generate(vec),
                                           m, n, k, nullptr, lda, nullptr, scratch_size);
            });
        })
        .wait();
}

#define ORGBR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                       \
    template <>                                                                              \
    std::int64_t orgbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k, \
                                             std::int64_t lda) {                             \
        int scratch_size;                                                                    \
        orgbr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, lda, \
                              &scratch_size);                                                \
        return scratch_size;                                                                 \
    }

ORGBR_LAUNCHER_SCRATCH(float, cusolverDnSorgbr_bufferSize)
ORGBR_LAUNCHER_SCRATCH(double, cusolverDnDorgbr_bufferSize)

#undef ORGBR_LAUNCHER_SCRATCH

template <typename Func>
inline void orgtr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, nullptr, scratch_size);
            });
        })
        .wait();
}

#define ORGTR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t orgtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        orgtr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

ORGTR_LAUNCHER_SCRATCH(float, cusolverDnSorgtr_bufferSize)
ORGTR_LAUNCHER_SCRATCH(double, cusolverDnDorgtr_bufferSize)

#undef ORGTR_LAUNCHER_SCRATCH

template <typename Func>
inline void orgqr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, k, nullptr, lda,
                                           nullptr, scratch_size);
            });
        })
        .wait();
}

#define ORGQR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t orgqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        int scratch_size;                                                                         \
        orgqr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,           \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

ORGQR_LAUNCHER_SCRATCH(float, cusolverDnSorgqr_bufferSize)
ORGQR_LAUNCHER_SCRATCH(double, cusolverDnDorgqr_bufferSize)

#undef ORGQR_LAUNCHER_SCRATCH

template <>
std::int64_t ormrq_scratchpad_size<float>(sycl::queue& queue, oneapi::mkl::side side,
                                          oneapi::mkl::transpose trans, std::int64_t m,
                                          std::int64_t n, std::int64_t k, std::int64_t lda,
                                          std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}
template <>
std::int64_t ormrq_scratchpad_size<double>(sycl::queue& queue, oneapi::mkl::side side,
                                           oneapi::mkl::transpose trans, std::int64_t m,
                                           std::int64_t n, std::int64_t k, std::int64_t lda,
                                           std::int64_t ldc) {
    throw unimplemented("lapack", "ormrq_scratchpad_size");
}

template <typename Func>
inline void ormqr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_side_mode(side),
                                           get_cublas_operation(trans), m, n, k, nullptr, lda,
                                           nullptr, nullptr, ldc, scratch_size);
            });
        })
        .wait();
}

#define ORMQRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                    \
    std::int64_t ormqr_scratchpad_size<TYPE>(                                                      \
        sycl::queue & queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, \
        std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) {                      \
        int scratch_size;                                                                          \
        ormqr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k,    \
                              lda, ldc, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

ORMQRF_LAUNCHER_SCRATCH(float, cusolverDnSormqr_bufferSize)
ORMQRF_LAUNCHER_SCRATCH(double, cusolverDnDormqr_bufferSize)

#undef ORMQRF_LAUNCHER_SCRATCH

template <typename Func>
inline void ormtr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, std::int64_t ldc, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_side_mode(side),
                                           get_cublas_fill_mode(uplo), get_cublas_operation(trans),
                                           m, n, nullptr, lda, nullptr, nullptr, ldc, scratch_size);
            });
        })
        .wait();
}

#define ORMTR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                             \
    template <>                                                                                    \
    std::int64_t ormtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::side side,          \
                                             oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, \
                                             std::int64_t m, std::int64_t n, std::int64_t lda,     \
                                             std::int64_t ldc) {                                   \
        int scratch_size;                                                                          \
        ormtr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, \
                              lda, ldc, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

ORMTR_LAUNCHER_SCRATCH(float, cusolverDnSormtr_bufferSize)
ORMTR_LAUNCHER_SCRATCH(double, cusolverDnDormtr_bufferSize)

#undef ORMTR_LAUNCHER_SCRATCH

template <typename Func>
inline void potrf_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, scratch_size);
            });
        })
        .wait();
}

#define POTRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t potrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        potrf_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

POTRF_LAUNCHER_SCRATCH(float, cusolverDnSpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(double, cusolverDnDpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCpotrf_bufferSize)
POTRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZpotrf_bufferSize)

#undef POTRF_LAUNCHER_SCRATCH

// cusolverDnXpotrs does not use scratchpad memory
#define POTRS_LAUNCHER_SCRATCH(TYPE)                                                              \
    template <>                                                                                   \
    std::int64_t potrs_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo,         \
                                             std::int64_t n, std::int64_t nrhs, std::int64_t lda, \
                                             std::int64_t ldb) {                                  \
        return 0;                                                                                 \
    }

POTRS_LAUNCHER_SCRATCH(float)
POTRS_LAUNCHER_SCRATCH(double)
POTRS_LAUNCHER_SCRATCH(std::complex<float>)
POTRS_LAUNCHER_SCRATCH(std::complex<double>)

#undef POTRS_LAUNCHER_SCRATCH

template <typename Func>
inline void potri_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, scratch_size);
            });
        })
        .wait();
}

#define POTRI_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t potri_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        potri_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

POTRI_LAUNCHER_SCRATCH(float, cusolverDnSpotri_bufferSize)
POTRI_LAUNCHER_SCRATCH(double, cusolverDnDpotri_bufferSize)
POTRI_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCpotri_bufferSize)
POTRI_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZpotri_bufferSize)

#undef POTRI_LAUNCHER_SCRATCH

template <typename Func>
inline void sytrf_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, n, nullptr, lda,
                                           scratch_size);
            });
        })
        .wait();
}

#define SYTRF_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t sytrf_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        sytrf_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

SYTRF_LAUNCHER_SCRATCH(float, cusolverDnSsytrf_bufferSize)
SYTRF_LAUNCHER_SCRATCH(double, cusolverDnDsytrf_bufferSize)
SYTRF_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCsytrf_bufferSize)
SYTRF_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZsytrf_bufferSize)

#undef SYTRF_LAUNCHER_SCRATCH

template <typename Func>
inline void syevd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::job jobz, oneapi::mkl::uplo uplo, std::int64_t n,
                                  std::int64_t lda, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cusolver_job(jobz),
                                           get_cublas_fill_mode(uplo), n, nullptr, lda, nullptr,
                                           scratch_size);
            });
        })
        .wait();
}

#define SYEVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                        \
    template <>                                                                               \
    std::int64_t syevd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::job jobz,      \
                                             oneapi::mkl::uplo uplo, std::int64_t n,          \
                                             std::int64_t lda) {                              \
        int scratch_size;                                                                     \
        syevd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, jobz, uplo, n, lda, \
                              &scratch_size);                                                 \
        return scratch_size;                                                                  \
    }

SYEVD_LAUNCHER_SCRATCH(float, cusolverDnSsyevd_bufferSize)
SYEVD_LAUNCHER_SCRATCH(double, cusolverDnDsyevd_bufferSize)

#undef SYEVD_LAUNCHER_SCRATCH

template <typename Func>
inline void sygvd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t itype, oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,
                                  std::int64_t n, std::int64_t lda, std::int64_t ldb,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cusolver_itype(itype),
                                           get_cusolver_job(jobz), get_cublas_fill_mode(uplo), n,
                                           nullptr, lda, nullptr, ldb, nullptr, scratch_size);
            });
        })
        .wait();
}

#define SYGVD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                             \
    template <>                                                                                    \
    std::int64_t sygvd_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t itype,              \
                                             oneapi::mkl::job jobz, oneapi::mkl::uplo uplo,        \
                                             std::int64_t n, std::int64_t lda, std::int64_t ldb) { \
        int scratch_size;                                                                          \
        sygvd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, itype, jobz, uplo, n,    \
                              lda, ldb, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

SYGVD_LAUNCHER_SCRATCH(float, cusolverDnSsygvd_bufferSize)
SYGVD_LAUNCHER_SCRATCH(double, cusolverDnDsygvd_bufferSize)

#undef SYGVD_LAUNCHER_SCRATCH

template <typename Func>
inline void sytrd_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, nullptr, nullptr, nullptr,
                                           scratch_size);
            });
        })
        .wait();
}

#define SYTRD_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t sytrd_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        sytrd_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

SYTRD_LAUNCHER_SCRATCH(float, cusolverDnSsytrd_bufferSize)
SYTRD_LAUNCHER_SCRATCH(double, cusolverDnDsytrd_bufferSize)

#undef SYTRD_LAUNCHER_SCRATCH

template <>
std::int64_t trtrs_scratchpad_size<float>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                          oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                          std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                          std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<double>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                           oneapi::mkl::transpose trans, oneapi::mkl::diag diag,
                                           std::int64_t n, std::int64_t nrhs, std::int64_t lda,
                                           std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<float>>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                        oneapi::mkl::transpose trans,
                                                        oneapi::mkl::diag diag, std::int64_t n,
                                                        std::int64_t nrhs, std::int64_t lda,
                                                        std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}
template <>
std::int64_t trtrs_scratchpad_size<std::complex<double>>(sycl::queue& queue, oneapi::mkl::uplo uplo,
                                                         oneapi::mkl::transpose trans,
                                                         oneapi::mkl::diag diag, std::int64_t n,
                                                         std::int64_t nrhs, std::int64_t lda,
                                                         std::int64_t ldb) {
    throw unimplemented("lapack", "trtrs_scratchpad_size");
}

template <typename Func>
inline void ungbr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::generate vec, std::int64_t m, std::int64_t n,
                                  std::int64_t k, std::int64_t lda, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_generate(vec),
                                           m, n, k, nullptr, lda, nullptr, scratch_size);
            });
        })
        .wait();
}

#define UNGBR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                       \
    template <>                                                                              \
    std::int64_t ungbr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::generate vec, \
                                             std::int64_t m, std::int64_t n, std::int64_t k, \
                                             std::int64_t lda) {                             \
        int scratch_size;                                                                    \
        ungbr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, vec, m, n, k, lda, \
                              &scratch_size);                                                \
        return scratch_size;                                                                 \
    }

UNGBR_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCungbr_bufferSize)
UNGBR_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZungbr_bufferSize)

#undef UNGBR_LAUNCHER_SCRATCH

template <typename Func>
inline void ungqr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, m, n, k, nullptr, lda,
                                           nullptr, scratch_size);
            });
        })
        .wait();
}

#define UNGQR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                            \
    template <>                                                                                   \
    std::int64_t ungqr_scratchpad_size<TYPE>(sycl::queue & queue, std::int64_t m, std::int64_t n, \
                                             std::int64_t k, std::int64_t lda) {                  \
        int scratch_size;                                                                         \
        ungqr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, m, n, k, lda,           \
                              &scratch_size);                                                     \
        return scratch_size;                                                                      \
    }

UNGQR_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCungqr_bufferSize)
UNGQR_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZungqr_bufferSize)

#undef UNGQR_LAUNCHER_SCRATCH

template <typename Func>
inline void ungtr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t lda,
                                  int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_fill_mode(uplo),
                                           n, nullptr, lda, nullptr, scratch_size);
            });
        })
        .wait();
}

#define UNGTR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                    \
    template <>                                                                           \
    std::int64_t ungtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::uplo uplo, \
                                             std::int64_t n, std::int64_t lda) {          \
        int scratch_size;                                                                 \
        ungtr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, uplo, n, lda,   \
                              &scratch_size);                                             \
        return scratch_size;                                                              \
    }

UNGTR_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCungtr_bufferSize)
UNGTR_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZungtr_bufferSize)

#undef UNGTR_LAUNCHER_SCRATCH

template <>
std::int64_t unmrq_scratchpad_size<std::complex<float>>(sycl::queue& queue, oneapi::mkl::side side,
                                                        oneapi::mkl::transpose trans,
                                                        std::int64_t m, std::int64_t n,
                                                        std::int64_t k, std::int64_t lda,
                                                        std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}
template <>
std::int64_t unmrq_scratchpad_size<std::complex<double>>(sycl::queue& queue, oneapi::mkl::side side,
                                                         oneapi::mkl::transpose trans,
                                                         std::int64_t m, std::int64_t n,
                                                         std::int64_t k, std::int64_t lda,
                                                         std::int64_t ldc) {
    throw unimplemented("lapack", "unmrq_scratchpad_size");
}

template <typename Func>
inline void unmqr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::side side, oneapi::mkl::transpose trans,
                                  std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t lda,
                                  std::int64_t ldc, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_side_mode(side),
                                           get_cublas_operation(trans), m, n, k, nullptr, lda,
                                           nullptr, nullptr, ldc, scratch_size);
            });
        })
        .wait();
}

#define UNMQR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                             \
    template <>                                                                                    \
    std::int64_t unmqr_scratchpad_size<TYPE>(                                                      \
        sycl::queue & queue, oneapi::mkl::side side, oneapi::mkl::transpose trans, std::int64_t m, \
        std::int64_t n, std::int64_t k, std::int64_t lda, std::int64_t ldc) {                      \
        int scratch_size;                                                                          \
        unmqr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, trans, m, n, k,    \
                              lda, ldc, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

UNMQR_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCunmqr_bufferSize)
UNMQR_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZunmqr_bufferSize)

#undef UNMQR_LAUNCHER_SCRATCH

template <typename Func>
inline void unmtr_scratchpad_size(const char* func_name, Func func, sycl::queue& queue,
                                  oneapi::mkl::side side, oneapi::mkl::uplo uplo,
                                  oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
                                  std::int64_t lda, std::int64_t ldc, int* scratch_size) {
    queue
        .submit([&](sycl::handler& cgh) {
            onemkl_cusolver_host_task(cgh, queue, [=](CusolverScopedContextHandler& sc) {
                auto handle = sc.get_handle(queue);
                cusolverStatus_t err;
                CUSOLVER_ERROR_FUNC_T_SYNC(func_name, func, err, handle, get_cublas_side_mode(side),
                                           get_cublas_fill_mode(uplo), get_cublas_operation(trans),
                                           m, n, nullptr, lda, nullptr, nullptr, ldc, scratch_size);
            });
        })
        .wait();
}

#define UNMTR_LAUNCHER_SCRATCH(TYPE, CUSOLVER_ROUTINE)                                             \
    template <>                                                                                    \
    std::int64_t unmtr_scratchpad_size<TYPE>(sycl::queue & queue, oneapi::mkl::side side,          \
                                             oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans, \
                                             std::int64_t m, std::int64_t n, std::int64_t lda,     \
                                             std::int64_t ldc) {                                   \
        int scratch_size;                                                                          \
        unmtr_scratchpad_size(#CUSOLVER_ROUTINE, CUSOLVER_ROUTINE, queue, side, uplo, trans, m, n, \
                              lda, ldc, &scratch_size);                                            \
        return scratch_size;                                                                       \
    }

UNMTR_LAUNCHER_SCRATCH(std::complex<float>, cusolverDnCunmtr_bufferSize)
UNMTR_LAUNCHER_SCRATCH(std::complex<double>, cusolverDnZunmtr_bufferSize)

#undef UNMTR_LAUNCHER_SCRATCH

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
