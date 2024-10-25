/*******************************************************************************
* Copyright Codeplay Software
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

void dotc(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& result) {
    throw unimplemented("blas", "dotc", "");
}

void dotu(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy,
          sycl::buffer<std::complex<real_t>, 1>& result) {
    throw unimplemented("blas", "dotu", "");
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    CALL_PORTBLAS_FN(::blas::_iamax, queue, n, x, incx, result);
}

void iamax(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    throw unimplemented("blas", "iamax", "");
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
           sycl::buffer<std::int64_t, 1>& result) {
    CALL_PORTBLAS_FN(::blas::_iamin, queue, n, x, incx, result);
}

void iamin(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
           std::int64_t incx, sycl::buffer<std::int64_t, 1>& result) {
    throw unimplemented("blas", "iamin", "");
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<real_t, 1>& result) {
    throw unimplemented("blas", "asum", "");
}

void asum(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
          sycl::buffer<real_t, 1>& result) {
    // portBLAS asum implementation requires that result is initialized to zero
    // before performing the computation.
    queue.submit([&](sycl::handler& cgh) {
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task([=]() { result_acc[0] = real_t(0); });
    });
    CALL_PORTBLAS_FN(::blas::_asum, queue, n, x, incx, result);
}

void axpy(sycl::queue& queue, std::int64_t n, real_t alpha, sycl::buffer<real_t, 1>& x,
          std::int64_t incx, sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_axpy, queue, n, alpha, x, incx, y, incy);
}

void axpy(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx,
          sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "axpy", "for complex");
}

void axpby(sycl::queue& queue, std::int64_t n, real_t alpha, sycl::buffer<real_t, 1>& x,
           std::int64_t incx, real_t beta, sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "axpby", "");
}

void axpby(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
           sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx, std::complex<real_t> beta,
           sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "axpby", "");
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
          sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_copy, queue, n, x, incx, y, incy);
}

void copy(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "copy", " for complex.");
}

void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
         sycl::buffer<real_t, 1>& y, std::int64_t incy, sycl::buffer<real_t, 1>& result) {
    // portBLAS dot implementation requires that result is initialized to zero
    // before performing the computation.
    queue.submit([&](sycl::handler& cgh) {
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task([=]() { result_acc[0] = real_t(0); });
    });
    CALL_PORTBLAS_FN(::blas::_dot, queue, n, x, incx, y, incy, result);
}

#ifdef ENABLE_MIXED_PRECISION_WITH_DOUBLE
void dot(sycl::queue& queue, std::int64_t n, sycl::buffer<float, 1>& x, std::int64_t incx,
         sycl::buffer<float, 1>& y, std::int64_t incy, sycl::buffer<double, 1>& result) {
    throw unimplemented("blas", "dot", " for unmatched return type");
}
#endif

void sdsdot(sycl::queue& queue, std::int64_t n, real_t sb, sycl::buffer<real_t, 1>& x,
            std::int64_t incx, sycl::buffer<real_t, 1>& y, std::int64_t incy,
            sycl::buffer<real_t, 1>& result) {
    // portBLAS sdsdot implementation requires that result is initialized to zero
    // before performing the computation.
    queue.submit([&](sycl::handler& cgh) {
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task([=]() { result_acc[0] = real_t(0); });
    });
    CALL_PORTBLAS_FN(::blas::_sdsdot, queue, n, sb, x, incx, y, incy, result);
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<real_t, 1>& result) {
    throw unimplemented("blas", "nrm2", " for complex");
}

void nrm2(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
          sycl::buffer<real_t, 1>& result) {
    // portBLAS nrm2 implementation requires that result is initialized to zero
    // before performing the computation.
    queue.submit([&](sycl::handler& cgh) {
        auto result_acc = result.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task([=]() { result_acc[0] = real_t(0); });
    });
    CALL_PORTBLAS_FN(::blas::_nrm2, queue, n, x, incx, result);
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
         std::int64_t incx, sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy, real_t c,
         real_t s) {
    throw unimplemented("blas", "rot", " for complex");
}

void rot(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
         sycl::buffer<real_t, 1>& y, std::int64_t incy, real_t c, real_t s) {
    CALL_PORTBLAS_FN(::blas::_rot, queue, n, x, incx, y, incy, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<real_t, 1>& a, sycl::buffer<real_t, 1>& b,
          sycl::buffer<real_t, 1>& c, sycl::buffer<real_t, 1>& s) {
    CALL_PORTBLAS_FN(::blas::_rotg, queue, a, b, c, s);
}

void rotg(sycl::queue& queue, sycl::buffer<std::complex<real_t>, 1>& a,
          sycl::buffer<std::complex<real_t>, 1>& b, sycl::buffer<real_t, 1>& c,
          sycl::buffer<std::complex<real_t>, 1>& s) {
    throw unimplemented("blas", "rotg", " for complex");
}

void rotm(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
          sycl::buffer<real_t, 1>& y, std::int64_t incy, sycl::buffer<real_t, 1>& param) {
    CALL_PORTBLAS_FN(::blas::_rotm, queue, n, x, incx, y, incy, param);
}

void rotmg(sycl::queue& queue, sycl::buffer<real_t, 1>& d1, sycl::buffer<real_t, 1>& d2,
           sycl::buffer<real_t, 1>& x1, real_t y1, sycl::buffer<real_t, 1>& param) {
    CALL_PORTBLAS_FN(::blas::_rotmg, queue, d1, d2, x1, y1, param);
}

void scal(sycl::queue& queue, std::int64_t n, real_t alpha, sycl::buffer<real_t, 1>& x,
          std::int64_t incx) {
    CALL_PORTBLAS_FN(::blas::_scal, queue, n, alpha, x, incx);
}

void scal(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "scal", " for complex");
}

void scal(sycl::queue& queue, std::int64_t n, real_t alpha,
          sycl::buffer<std::complex<real_t>, 1>& x, std::int64_t incx) {
    throw unimplemented("blas", "scal", " for complex");
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<real_t, 1>& x, std::int64_t incx,
          sycl::buffer<real_t, 1>& y, std::int64_t incy) {
    CALL_PORTBLAS_FN(::blas::_swap, queue, n, x, incx, y, incy);
}

void swap(sycl::queue& queue, std::int64_t n, sycl::buffer<std::complex<real_t>, 1>& x,
          std::int64_t incx, sycl::buffer<std::complex<real_t>, 1>& y, std::int64_t incy) {
    throw unimplemented("blas", "swap", " for complex");
}

// USM APIs

sycl::event dotc(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                 std::int64_t incx, const std::complex<real_t>* y, std::int64_t incy,
                 std::complex<real_t>* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dotc", " for USM");
}

sycl::event dotu(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                 std::int64_t incx, const std::complex<real_t>* y, std::int64_t incy,
                 std::complex<real_t>* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dotu", " for USM");
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_iamax, queue, n, x, incx, result, dependencies);
}

sycl::event iamax(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "iamax", " for USM");
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx,
                  std::int64_t* result, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_iamin, queue, n, x, incx, result, dependencies);
}

sycl::event iamin(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                  std::int64_t incx, std::int64_t* result,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "iamin", " for USM");
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                 std::int64_t incx, real_t* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "asum", " for USM");
}

sycl::event asum(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx,
                 real_t* result, const std::vector<sycl::event>& dependencies) {
    // portBLAS asum implementation requires result to be initializes to zero
    // before starting the computation.
    auto init_res_val = queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task([=]() { result[0] = real_t(0); }); });
    std::vector<sycl::event> new_dependencies = dependencies;
    new_dependencies.push_back(init_res_val);
    CALL_PORTBLAS_USM_FN(::blas::_asum, queue, n, x, incx, result, new_dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, real_t alpha, const real_t* x,
                 std::int64_t incx, real_t* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_axpy, queue, n, alpha, x, incx, y, incy, dependencies);
}

sycl::event axpy(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
                 const std::complex<real_t>* x, std::int64_t incx, std::complex<real_t>* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpy", " for USM");
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, real_t alpha, const real_t* x,
                  std::int64_t incx, const real_t beta, real_t* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", " for USM");
}

sycl::event axpby(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
                  const std::complex<real_t>* x, std::int64_t incx, const std::complex<real_t> beta,
                  std::complex<real_t>* y, std::int64_t incy,
                  const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "axpby", " for USM");
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx, real_t* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_copy, queue, n, x, incx, y, incy, dependencies);
}

sycl::event copy(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                 std::int64_t incx, std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "copy", " for USM");
}

sycl::event dot(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx,
                const real_t* y, std::int64_t incy, real_t* result,
                const std::vector<sycl::event>& dependencies) {
    // portBLAS dot implementation requires result to be initializes to zero
    // before starting the computation.
    auto init_res_val = queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task([=]() { result[0] = real_t(0); }); });
    std::vector<sycl::event> new_dependencies = dependencies;
    new_dependencies.emplace_back(init_res_val);
    CALL_PORTBLAS_USM_FN(::blas::_dot, queue, n, x, incx, y, incy, result, new_dependencies);
}

#ifdef ENABLE_MIXED_PRECISION_WITH_DOUBLE
sycl::event dot(sycl::queue& queue, std::int64_t n, const float* x, std::int64_t incx,
                const float* y, std::int64_t incy, double* result,
                const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "dot", " for USM");
}
#endif

sycl::event sdsdot(sycl::queue& queue, std::int64_t n, real_t sb, const real_t* x,
                   std::int64_t incx, const real_t* y, std::int64_t incy, real_t* result,
                   const std::vector<sycl::event>& dependencies) {
    // portBLAS sdsdot implementation requires result to be initializes to zero
    // before starting the computation.
    auto init_res_val = queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task([=]() { result[0] = real_t(0); }); });
    std::vector<sycl::event> new_dependencies = dependencies;
    new_dependencies.emplace_back(init_res_val);
    CALL_PORTBLAS_USM_FN(::blas::_sdsdot, queue, n, sb, x, incx, y, incy, result, new_dependencies);
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const std::complex<real_t>* x,
                 std::int64_t incx, real_t* result, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "nrm2", " for USM");
}

sycl::event nrm2(sycl::queue& queue, std::int64_t n, const real_t* x, std::int64_t incx,
                 real_t* result, const std::vector<sycl::event>& dependencies) {
    // portBLAS nrm2 implementation requires result to be initializes to zero
    // before starting the computation.
    auto init_res_val = queue.submit(
        [&](sycl::handler& cgh) { cgh.single_task([=]() { result[0] = real_t(0); }); });
    std::vector<sycl::event> new_dependencies = dependencies;
    new_dependencies.push_back(init_res_val);
    CALL_PORTBLAS_USM_FN(::blas::_nrm2, queue, n, x, incx, result, new_dependencies);
}

sycl::event rot(sycl::queue& queue, std::int64_t n, std::complex<real_t>* x, std::int64_t incx,
                std::complex<real_t>* y, std::int64_t incy, real_t c, real_t s,
                const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rot", " for USM");
}

sycl::event rot(sycl::queue& queue, std::int64_t n, real_t* x, std::int64_t incx, real_t* y,
                std::int64_t incy, real_t c, real_t s,
                const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_rot, queue, n, x, incx, y, incy, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, real_t* a, real_t* b, real_t* c, real_t* s,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_rotg, queue, a, b, c, s, dependencies);
}

sycl::event rotg(sycl::queue& queue, std::complex<real_t>* a, std::complex<real_t>* b, real_t* c,
                 std::complex<real_t>* s, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "rotg", " for USM");
}

sycl::event rotm(sycl::queue& queue, std::int64_t n, real_t* x, std::int64_t incx, real_t* y,
                 std::int64_t incy, real_t* param, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_rotm, queue, n, x, incx, y, incy, param, dependencies);
}

sycl::event rotmg(sycl::queue& queue, real_t* d1, real_t* d2, real_t* x1, real_t y1, real_t* param,
                  const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_rotmg, queue, d1, d2, x1, y1, param, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, real_t alpha, real_t* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_scal, queue, n, alpha, x, incx, dependencies);
}

sycl::event scal(sycl::queue& queue, std::int64_t n, std::complex<real_t> alpha,
                 std::complex<real_t>* x, std::int64_t incx,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "scal", " for USM");
}

sycl::event scal(sycl::queue& queue, std::int64_t n, real_t alpha, std::complex<real_t>* x,
                 std::int64_t incx, const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "scal", " for USM");
}

sycl::event swap(sycl::queue& queue, std::int64_t n, real_t* x, std::int64_t incx, real_t* y,
                 std::int64_t incy, const std::vector<sycl::event>& dependencies) {
    CALL_PORTBLAS_USM_FN(::blas::_swap, queue, n, x, incx, y, incy, dependencies);
}

sycl::event swap(sycl::queue& queue, std::int64_t n, std::complex<real_t>* x, std::int64_t incx,
                 std::complex<real_t>* y, std::int64_t incy,
                 const std::vector<sycl::event>& dependencies) {
    throw unimplemented("blas", "swap", " for USM");
}
