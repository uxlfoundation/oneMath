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

// Buffer APIs

void asum(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result);

void asum(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result);

void asum(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result);

void asum(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result);

void axpy(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy);

void axpy(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy);

void axpy(sycl::queue& queue, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void axpy(sycl::queue& queue, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void axpy_batch(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void axpy_batch(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size);

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void axpby(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx,
           float beta, sycl::buffer<float, 1>& y, int64_t incy);

void axpby(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx,
           double beta, sycl::buffer<double, 1>& y, int64_t incy);

void axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void copy(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy);

void copy(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy);

void copy(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void copy(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<float, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
                int64_t stridex, sycl::buffer<double, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size);

void copy_batch(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<std::complex<double>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size);

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result);

void dot(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
         sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& result);

void dot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<double, 1>& result);

void dotc(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result);

void dotc(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result);

void dotu(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& result);

void dotu(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& result);

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamin(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void iamax(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
           sycl::buffer<int64_t, 1>& result);

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result);

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result);

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& result);

void nrm2(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& result);

void rot(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& y, int64_t incy, float c, float s);

void rot(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& y, int64_t incy, double c, double s);

void rot(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
         sycl::buffer<float, 1>& y, int64_t incy, float c, float s);

void rot(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
         sycl::buffer<double, 1>& y, int64_t incy, double c, double s);

void rotg(sycl::queue& queue, sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b,
          sycl::buffer<float, 1>& c, sycl::buffer<float, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& b,
          sycl::buffer<double, 1>& c, sycl::buffer<double, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<std::complex<float>, 1>& a,
          sycl::buffer<std::complex<float>, 1>& b, sycl::buffer<float, 1>& c,
          sycl::buffer<std::complex<float>, 1>& s);

void rotg(sycl::queue& queue, sycl::buffer<std::complex<double>, 1>& a,
          sycl::buffer<std::complex<double>, 1>& b, sycl::buffer<double, 1>& c,
          sycl::buffer<std::complex<double>, 1>& s);

void rotm(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& param);

void rotm(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& param);

void rotmg(sycl::queue& queue, sycl::buffer<float, 1>& d1, sycl::buffer<float, 1>& d2,
           sycl::buffer<float, 1>& x1, float y1, sycl::buffer<float, 1>& param);

void rotmg(sycl::queue& queue, sycl::buffer<double, 1>& d1, sycl::buffer<double, 1>& d2,
           sycl::buffer<double, 1>& x1, double y1, sycl::buffer<double, 1>& param);

void scal(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<float, 1>& x, int64_t incx);

void scal(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<double, 1>& x, int64_t incx);

void scal(sycl::queue& queue, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx);

void scal(sycl::queue& queue, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx);

void scal(sycl::queue& queue, int64_t n, float alpha, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx);

void scal(sycl::queue& queue, int64_t n, double alpha, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx);

void sdsdot(sycl::queue& queue, int64_t n, float sb, sycl::buffer<float, 1>& x, int64_t incx,
            sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& result);

void swap(sycl::queue& queue, int64_t n, sycl::buffer<float, 1>& x, int64_t incx,
          sycl::buffer<float, 1>& y, int64_t incy);

void swap(sycl::queue& queue, int64_t n, sycl::buffer<double, 1>& x, int64_t incx,
          sycl::buffer<double, 1>& y, int64_t incy);

void swap(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void swap(sycl::queue& queue, int64_t n, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          float alpha, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx, float beta, sycl::buffer<float, 1>& y, int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          double alpha, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx, double beta, sycl::buffer<double, 1>& y, int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx,
          float beta, sycl::buffer<float, 1>& y, int64_t incy);

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx,
          double beta, sycl::buffer<double, 1>& y, int64_t incy);

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stridea, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stridex, float beta, sycl::buffer<float, 1>& y, int64_t incy,
                int64_t stridey, int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stridea,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stridex, double beta,
                sycl::buffer<double, 1>& y, int64_t incy, int64_t stridey, int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stridea, sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
                int64_t stridex, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
                int64_t incy, int64_t stridey, int64_t batch_size);

void gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stridea, sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
                int64_t stridex, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& y, int64_t incy, int64_t stridey,
                int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<float, 1>& a, int64_t lda, int64_t stridea, sycl::buffer<float, 1>& x,
                int64_t incx, int64_t stridex, sycl::buffer<float, 1>& c, int64_t ldc,
                int64_t stridec, int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<double, 1>& a, int64_t lda, int64_t stridea,
                sycl::buffer<double, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stridec, int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stridea,
                sycl::buffer<std::complex<float>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stridec,
                int64_t batch_size);

void dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stridea,
                sycl::buffer<std::complex<double>, 1>& x, int64_t incx, int64_t stridex,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stridec,
                int64_t batch_size);

void ger(sycl::queue& queue, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& a,
         int64_t lda);

void ger(sycl::queue& queue, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& a,
         int64_t lda);

void gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda);

void gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda);

void geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda);

void geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda);

void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy);

void hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy);

void her(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a, int64_t lda);

void her(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a, int64_t lda);

void her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda);

void her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda);

void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& y,
          int64_t incy);

void hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx, std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& y,
          int64_t incy);

void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
         sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<float>, 1>& a);

void hpr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
         sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
         sycl::buffer<std::complex<double>, 1>& a);

void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<float>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<float>, 1>& a);

void hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx,
          sycl::buffer<std::complex<double>, 1>& y, int64_t incy,
          sycl::buffer<std::complex<double>, 1>& a);

void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx,
          float beta, sycl::buffer<float, 1>& y, int64_t incy);

void sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx,
          double beta, sycl::buffer<double, 1>& y, int64_t incy);

void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& a,
          sycl::buffer<float, 1>& x, int64_t incx, float beta, sycl::buffer<float, 1>& y,
          int64_t incy);

void spmv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& a,
          sycl::buffer<double, 1>& x, int64_t incx, double beta, sycl::buffer<double, 1>& y,
          int64_t incy);

void spr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& a);

void spr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& a);

void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
          int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& a);

void spr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
          int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& a);

void symv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& a,
          int64_t lda, sycl::buffer<float, 1>& x, int64_t incx, float beta,
          sycl::buffer<float, 1>& y, int64_t incy);

void symv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& a,
          int64_t lda, sycl::buffer<double, 1>& x, int64_t incx, double beta,
          sycl::buffer<double, 1>& y, int64_t incy);

void syr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
         int64_t incx, sycl::buffer<float, 1>& a, int64_t lda);

void syr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
         int64_t incx, sycl::buffer<double, 1>& a, int64_t lda);

void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, sycl::buffer<float, 1>& x,
          int64_t incx, sycl::buffer<float, 1>& y, int64_t incy, sycl::buffer<float, 1>& a,
          int64_t lda);

void syr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, sycl::buffer<double, 1>& x,
          int64_t incx, sycl::buffer<double, 1>& y, int64_t incy, sycl::buffer<double, 1>& a,
          int64_t lda);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx);

void tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x,
          int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x,
          int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx);

void tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          int64_t k, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx);

void tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& x, int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, sycl::buffer<double, 1>& x, int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, sycl::buffer<std::complex<float>, 1>& x,
          int64_t incx);

void tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, sycl::buffer<std::complex<double>, 1>& x,
          int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx);

void trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& x, int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& x, int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& x, int64_t incx);

void trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& x, int64_t incx);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          float alpha, sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& b,
          int64_t ldb, float beta, sycl::buffer<float, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          double alpha, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& b,
          int64_t ldb, double beta, sycl::buffer<double, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
          sycl::buffer<sycl::half, 1>& b, int64_t ldb, sycl::half beta,
          sycl::buffer<sycl::half, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda, sycl::buffer<sycl::half, 1>& b,
          int64_t ldb, float beta, sycl::buffer<float, 1>& c, int64_t ldc);

void gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n, int64_t k,
          float alpha, sycl::buffer<bfloat16, 1>& a, int64_t lda, sycl::buffer<bfloat16, 1>& b,
          int64_t ldb, float beta, sycl::buffer<float, 1>& c, int64_t ldc);

void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, float alpha,
          sycl::buffer<std::complex<float>, 1>& a, int64_t lda, float beta,
          sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, double alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda, double beta,
          sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
           sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, float beta,
           sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
           sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, double beta,
           sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& b, int64_t ldb,
          float beta, sycl::buffer<float, 1>& c, int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& b, int64_t ldb,
          double beta, sycl::buffer<double, 1>& c, int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, float alpha,
          sycl::buffer<float, 1>& a, int64_t lda, float beta, sycl::buffer<float, 1>& c,
          int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, double alpha,
          sycl::buffer<double, 1>& a, int64_t lda, double beta, sycl::buffer<double, 1>& c,
          int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                double beta, sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size);

void syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
                int64_t stride_a, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k, float alpha,
           sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& b, int64_t ldb,
           float beta, sycl::buffer<float, 1>& c, int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           double alpha, sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& b,
           int64_t ldb, double beta, sycl::buffer<double, 1>& c, int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
           sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
           sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a, int64_t lda,
          sycl::buffer<float, 1>& b, int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a, int64_t lda,
          sycl::buffer<double, 1>& b, int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
          int64_t lda, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb);

void trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a, int64_t lda,
          sycl::buffer<float, 1>& b, int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a, int64_t lda,
          sycl::buffer<double, 1>& b, int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
          int64_t lda, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb);

void trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          int64_t m, int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
          sycl::buffer<std::complex<double>, 1>& b, int64_t ldb);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, float beta,
                sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, double beta,
                sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                int64_t stride_b, std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c,
                int64_t ldc, int64_t stride_c, int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                int64_t ldb, int64_t stride_b, std::complex<double> beta,
                sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                sycl::half beta, sycl::buffer<sycl::half, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<sycl::half, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<sycl::half, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                int64_t k, float alpha, sycl::buffer<std::int8_t, 1>& a, int64_t lda,
                int64_t stride_a, sycl::buffer<std::int8_t, 1>& b, int64_t ldb, int64_t stride_b,
                float beta, sycl::buffer<std::int32_t, 1>& c, int64_t ldc, int64_t stride_c,
                int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, float alpha, sycl::buffer<float, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<float, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, double alpha, sycl::buffer<double, 1>& a,
                int64_t lda, int64_t stride_a, sycl::buffer<double, 1>& b, int64_t ldb,
                int64_t stride_b, int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                sycl::buffer<std::complex<float>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size);

void trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                sycl::buffer<std::complex<double>, 1>& a, int64_t lda, int64_t stride_a,
                sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                int64_t batch_size);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1>& a, int64_t lda,
           sycl::buffer<float, 1>& b, int64_t ldb, float beta, sycl::buffer<float, 1>& c,
           int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1>& a, int64_t lda,
           sycl::buffer<double, 1>& b, int64_t ldb, double beta, sycl::buffer<double, 1>& c,
           int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
           int64_t lda, sycl::buffer<std::complex<double>, 1>& b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1>& a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1>& b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co);

void gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1>& a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1>& b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1>& c, int64_t ldc, sycl::buffer<int32_t, 1>& co);

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                    sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b, int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                    int64_t stride_a, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
                    int64_t stride_b, int64_t batch_size);

void omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                    int64_t lda, int64_t stride_a, sycl::buffer<std::complex<double>, 1>& b,
                    int64_t ldb, int64_t stride_b, int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                    sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                    sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb, int64_t stride,
                    int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size);

void imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& ab,
                    int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   float alpha, sycl::buffer<float, 1>& a, int64_t lda, int64_t stride_a,
                   float beta, sycl::buffer<float, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<float, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   double alpha, sycl::buffer<double, 1>& a, int64_t lda, int64_t stride_a,
                   double beta, sycl::buffer<double, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<double, 1>& c, int64_t ldc, int64_t stride_c, int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
                   int64_t stride_a, std::complex<float> beta,
                   sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<float>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size);

void omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                   std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a,
                   int64_t lda, int64_t stride_a, std::complex<double> beta,
                   sycl::buffer<std::complex<double>, 1>& b, int64_t ldb, int64_t stride_b,
                   sycl::buffer<std::complex<double>, 1>& c, int64_t ldc, int64_t stride_c,
                   int64_t batch_size);

void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1>& a, int64_t lda, sycl::buffer<float, 1>& b, int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1>& a, int64_t lda, sycl::buffer<double, 1>& b, int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
              sycl::buffer<std::complex<float>, 1>& b, int64_t ldb);

void omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
              sycl::buffer<std::complex<double>, 1>& b, int64_t ldb);

void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
               sycl::buffer<float, 1>& a, int64_t lda, std::int64_t stridea,
               sycl::buffer<float, 1>& b, int64_t ldb, std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
               sycl::buffer<double, 1>& a, int64_t lda, std::int64_t stridea,
               sycl::buffer<double, 1>& b, int64_t ldb, std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
               sycl::buffer<std::complex<float>, 1>& a, int64_t lda, std::int64_t stridea,
               sycl::buffer<std::complex<float>, 1>& b, int64_t ldb, std::int64_t strideb);

void omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
               std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
               std::int64_t stridea, sycl::buffer<std::complex<double>, 1>& b, int64_t ldb,
               std::int64_t strideb);

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
              sycl::buffer<float, 1>& ab, int64_t lda, int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
              sycl::buffer<double, 1>& ab, int64_t lda, int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<float> alpha,
              sycl::buffer<std::complex<float>, 1>& ab, int64_t lda, int64_t ldb);

void imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, std::complex<double> alpha,
              sycl::buffer<std::complex<double>, 1>& ab, int64_t lda, int64_t ldb);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
             float alpha, sycl::buffer<float, 1>& a, int64_t lda, float beta,
             sycl::buffer<float, 1>& b, int64_t ldb, sycl::buffer<float, 1>& c, int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
             double alpha, sycl::buffer<double, 1>& a, int64_t lda, double beta,
             sycl::buffer<double, 1>& b, int64_t ldb, sycl::buffer<double, 1>& c, int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<float> alpha, sycl::buffer<std::complex<float>, 1>& a, int64_t lda,
             std::complex<float> beta, sycl::buffer<std::complex<float>, 1>& b, int64_t ldb,
             sycl::buffer<std::complex<float>, 1>& c, int64_t ldc);

void omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
             std::complex<double> alpha, sycl::buffer<std::complex<double>, 1>& a, int64_t lda,
             std::complex<double> beta, sycl::buffer<std::complex<double>, 1>& b, int64_t ldb,
             sycl::buffer<std::complex<double>, 1>& c, int64_t ldc);

// USM APIs

sycl::event asum(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event asum(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event axpy(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, float* alpha, const float** x, int64_t* incx,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, double* alpha, const double** x,
                       int64_t* incx, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                       int64_t stridex, float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpy_batch(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, int64_t n, float alpha, const float* x, int64_t incx,
                  const float beta, float* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, int64_t n, double alpha, const double* x, int64_t incx,
                  const double beta, double* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<float> alpha,
                  const std::complex<float>* x, int64_t incx, const std::complex<float> beta,
                  std::complex<float>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event axpby(sycl::queue& queue, int64_t n, std::complex<double> alpha,
                  const std::complex<double>* x, int64_t incx, const std::complex<double> beta,
                  std::complex<double>* y, int64_t incy,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event copy(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const float** x, int64_t* incx, float** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const double** x, int64_t* incx, double** y,
                       int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<float>** x, int64_t* incx,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t* n, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t n, const float* x, int64_t incx, int64_t stridex,
                       float* y, int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t n, const double* x, int64_t incx,
                       int64_t stridex, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                       int64_t stridex, std::complex<float>* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event copy_batch(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                       int64_t stridex, std::complex<double>* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, float* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, int64_t n, const double* x, int64_t incx, const double* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dot(sycl::queue& queue, int64_t n, const float* x, int64_t incx, const float* y,
                int64_t incy, double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 const std::complex<float>* y, int64_t incy, std::complex<float>* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event dotc(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 const std::complex<double>* y, int64_t incy, std::complex<double>* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event dotu(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 const std::complex<float>* y, int64_t incy, std::complex<float>* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event dotu(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 const std::complex<double>* y, int64_t incy, std::complex<double>* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, int64_t n, const float* x, int64_t incx, int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, int64_t n, const double* x, int64_t incx, int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                  int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamin(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                  int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, int64_t n, const float* x, int64_t incx, int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, int64_t n, const double* x, int64_t incx, int64_t* result,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                  int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event iamax(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                  int64_t* result, const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, int64_t n, const std::complex<float>* x, int64_t incx,
                 float* result, const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, int64_t n, const std::complex<double>* x, int64_t incx,
                 double* result, const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, int64_t n, const float* x, int64_t incx, float* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event nrm2(sycl::queue& queue, int64_t n, const double* x, int64_t incx, double* result,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, int64_t n, std::complex<float>* x, int64_t incx,
                std::complex<float>* y, int64_t incy, float c, float s,
                const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, int64_t n, std::complex<double>* x, int64_t incx,
                std::complex<double>* y, int64_t incy, double c, double s,
                const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                float c, float s, const std::vector<sycl::event>& dependencies = {});

sycl::event rot(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                double c, double s, const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, float* a, float* b, float* c, float* s,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, double* a, double* b, double* c, double* s,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, std::complex<float>* a, std::complex<float>* b, float* c,
                 std::complex<float>* s, const std::vector<sycl::event>& dependencies = {});

sycl::event rotg(sycl::queue& queue, std::complex<double>* a, std::complex<double>* b, double* c,
                 std::complex<double>* s, const std::vector<sycl::event>& dependencies = {});

sycl::event rotm(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                 float* param, const std::vector<sycl::event>& dependencies = {});

sycl::event rotm(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                 double* param, const std::vector<sycl::event>& dependencies = {});

sycl::event rotmg(sycl::queue& queue, float* d1, float* d2, float* x1, float y1, float* param,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event rotmg(sycl::queue& queue, double* d1, double* d2, double* x1, double y1, double* param,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, float alpha, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, double alpha, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, std::complex<float> alpha, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, std::complex<double> alpha, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, float alpha, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event scal(sycl::queue& queue, int64_t n, double alpha, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event sdsdot(sycl::queue& queue, int64_t n, float sb, const float* x, int64_t incx,
                   const float* y, int64_t incy, float* result,
                   const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, int64_t n, float* x, int64_t incx, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, int64_t n, double* x, int64_t incx, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, int64_t n, std::complex<float>* x, int64_t incx,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event swap(sycl::queue& queue, int64_t n, std::complex<double>* x, int64_t incx,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 float alpha, const float* a, int64_t lda, const float* x, int64_t incx, float beta,
                 float* y, int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 double alpha, const double* a, int64_t lda, const double* x, int64_t incx,
                 double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gbmv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, int64_t kl, int64_t ku,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                 const float* a, int64_t lda, const float* x, int64_t incx, float beta, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                 const double* a, int64_t lda, const double* x, int64_t incx, double beta,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemv(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                       const float* a, int64_t lda, int64_t stridea, const float* x, int64_t incx,
                       int64_t stridex, float beta, float* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                       const double* a, int64_t lda, int64_t stridea, const double* x, int64_t incx,
                       int64_t stridex, double beta, double* y, int64_t incy, int64_t stridey,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stridea, const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float> beta, std::complex<float>* y, int64_t incy,
                       int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stridea, const std::complex<double>* x, int64_t incx,
                       int64_t stridex, std::complex<double> beta, std::complex<double>* y,
                       int64_t incy, int64_t stridey, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n, float* alpha,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float* beta,
                       float** y, int64_t* incy, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n, double* alpha,
                       const double** a, int64_t* lda, const double** x, int64_t* incx,
                       double* beta, double** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                       std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                       const std::complex<float>** x, int64_t* incx, std::complex<float>* beta,
                       std::complex<float>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gemv_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                       std::complex<double>* alpha, const std::complex<double>** a, int64_t* lda,
                       const std::complex<double>** x, int64_t* incx, std::complex<double>* beta,
                       std::complex<double>** y, int64_t* incy, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const float* a,
                       int64_t lda, int64_t stridea, const float* x, int64_t incx, int64_t stridex,
                       float* c, int64_t ldc, int64_t stridec, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n, const double* a,
                       int64_t lda, int64_t stridea, const double* x, int64_t incx, int64_t stridex,
                       double* c, int64_t ldc, int64_t stridec, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<float>* a, int64_t lda, int64_t stridea,
                       const std::complex<float>* x, int64_t incx, int64_t stridex,
                       std::complex<float>* c, int64_t ldc, int64_t stridec, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side left_right, int64_t m, int64_t n,
                       const std::complex<double>* a, int64_t lda, int64_t stridea,
                       const std::complex<double>* x, int64_t incx, int64_t stridex,
                       std::complex<double>* c, int64_t ldc, int64_t stridec, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const float** a, int64_t* lda, const float** x, int64_t* incx, float** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const double** a, int64_t* lda, const double** x, int64_t* incx, double** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** x,
                       int64_t* incx, std::complex<float>** c, int64_t* ldc, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event dgmm_batch(sycl::queue& queue, side* left_right, int64_t* m, int64_t* n,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** x,
                       int64_t* incx, std::complex<double>** c, int64_t* ldc, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, float alpha, const float* x, int64_t incx,
                const float* y, int64_t incy, float* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event ger(sycl::queue& queue, int64_t m, int64_t n, double alpha, const double* x,
                int64_t incx, const double* y, int64_t incy, double* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gerc(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event geru(sycl::queue& queue, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* x, int64_t incx, std::complex<float> beta,
                 std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* x, int64_t incx, std::complex<double> beta,
                 std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, int64_t lda, const std::complex<float>* x,
                 int64_t incx, std::complex<float> beta, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, int64_t lda, const std::complex<double>* x,
                 int64_t incx, std::complex<double> beta, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
                const std::complex<float>* x, int64_t incx, std::complex<float>* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event her(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
                const std::complex<double>* x, int64_t incx, std::complex<double>* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event her2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, const std::complex<float>* x, int64_t incx,
                 std::complex<float> beta, std::complex<float>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hpmv(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, const std::complex<double>* x, int64_t incx,
                 std::complex<double> beta, std::complex<double>* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha,
                const std::complex<float>* x, int64_t incx, std::complex<float>* a,
                const std::vector<sycl::event>& dependencies = {});

sycl::event hpr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha,
                const std::complex<double>* x, int64_t incx, std::complex<double>* a,
                const std::vector<sycl::event>& dependencies = {});

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* x, int64_t incx, const std::complex<float>* y,
                 int64_t incy, std::complex<float>* a,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hpr2(sycl::queue& queue, uplo upper_lower, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* x, int64_t incx, const std::complex<double>* y,
                 int64_t incy, std::complex<double>* a,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t lda, const float* x, int64_t incx, float beta, float* y,
                 int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event sbmv(sycl::queue& queue, uplo upper_lower, int64_t n, int64_t k, double alpha,
                 const double* a, int64_t lda, const double* x, int64_t incx, double beta,
                 double* y, int64_t incy, const std::vector<sycl::event>& dependencies = {});

sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* a,
                 const float* x, int64_t incx, float beta, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spmv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* a,
                 const double* x, int64_t incx, double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                int64_t incx, float* a, const std::vector<sycl::event>& dependencies = {});

sycl::event spr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                int64_t incx, double* a, const std::vector<sycl::event>& dependencies = {});

sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                 int64_t incx, const float* y, int64_t incy, float* a,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event spr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                 int64_t incx, const double* y, int64_t incy, double* a,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* a,
                 int64_t lda, const float* x, int64_t incx, float beta, float* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symv(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* a,
                 int64_t lda, const double* x, int64_t incx, double beta, double* y, int64_t incy,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                int64_t incx, float* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event syr(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                int64_t incx, double* a, int64_t lda,
                const std::vector<sycl::event>& dependencies = {});

sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, float alpha, const float* x,
                 int64_t incx, const float* y, int64_t incy, float* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syr2(sycl::queue& queue, uplo upper_lower, int64_t n, double alpha, const double* x,
                 int64_t incx, const double* y, int64_t incy, double* a, int64_t lda,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<float>* a, int64_t lda, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<double>* a, int64_t lda, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<float>* a, int64_t lda, std::complex<float>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tbsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 int64_t k, const std::complex<double>* a, int64_t lda, std::complex<double>* x,
                 int64_t incx, const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* a, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* a, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* a, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* a, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* a, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* a, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* a, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event tpsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* a, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const float* a, int64_t lda, float* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const double* a, int64_t lda, double* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsv(sycl::queue& queue, uplo upper_lower, transpose trans, diag unit_diag, int64_t n,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* x, int64_t incx,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
                 float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, double alpha, const double* a, int64_t lda, const double* b,
                 int64_t ldb, double beta, double* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                 std::complex<float>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* b, int64_t ldb, std::complex<double> beta,
                 std::complex<double>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda, const sycl::half* b,
                 int64_t ldb, sycl::half beta, sycl::half* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const sycl::half* a, int64_t lda, const sycl::half* b,
                 int64_t ldb, float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event gemm(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                 int64_t k, float alpha, const bfloat16* a, int64_t lda, const bfloat16* b,
                 int64_t ldb, float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                 std::complex<float>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event hemm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* b, int64_t ldb, std::complex<double> beta,
                 std::complex<double>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 float alpha, const std::complex<float>* a, int64_t lda, float beta,
                 std::complex<float>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event herk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 double alpha, const std::complex<double>* a, int64_t lda, double beta,
                 std::complex<double>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                  const std::complex<float>* b, int64_t ldb, float beta, std::complex<float>* c,
                  int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event her2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                  const std::complex<double>* b, int64_t ldb, double beta, std::complex<double>* c,
                  int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 float alpha, const float* a, int64_t lda, const float* b, int64_t ldb, float beta,
                 float* c, int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,
                 double beta, double* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                 std::complex<float>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event symm(sycl::queue& queue, side left_right, uplo upper_lower, int64_t m, int64_t n,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 const std::complex<double>* b, int64_t ldb, std::complex<double> beta,
                 std::complex<double>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 float alpha, const float* a, int64_t lda, float beta, float* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 double alpha, const double* a, int64_t lda, double beta, double* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                 std::complex<float> beta, std::complex<float>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                 std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                 std::complex<double> beta, std::complex<double>* c, int64_t ldc,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, float* alpha, const float** a, int64_t* lda, float* beta,
                       float** c, int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, double* alpha, const double** a, int64_t* lda, double* beta,
                       double** c, int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<float>* alpha, const std::complex<float>** a,
                       int64_t* lda, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo* upper_lower, transpose* trans, int64_t* n,
                       int64_t* k, std::complex<double>* alpha, const std::complex<double>** a,
                       int64_t* lda, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       float alpha, const float* a, int64_t lda, int64_t stride_a, float beta,
                       float* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       double alpha, const double* a, int64_t lda, int64_t stride_a, double beta,
                       double* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                       int64_t stride_a, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syrk_batch(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                       int64_t stride_a, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  float alpha, const float* a, int64_t lda, const float* b, int64_t ldb, float beta,
                  float* c, int64_t ldc, const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  double alpha, const double* a, int64_t lda, const double* b, int64_t ldb,
                  double beta, double* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                  const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                  std::complex<float>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event syr2k(sycl::queue& queue, uplo upper_lower, transpose trans, int64_t n, int64_t k,
                  std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                  const std::complex<double>* b, int64_t ldb, std::complex<double> beta,
                  std::complex<double>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, float alpha, const float* a, int64_t lda,
                 float* b, int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, double alpha, const double* a, int64_t lda,
                 double* b, int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* b, int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trmm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* b, int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, float alpha, const float* a, int64_t lda,
                 float* b, int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, double alpha, const double* a, int64_t lda,
                 double* b, int64_t ldb, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                 const std::complex<float>* a, int64_t lda, std::complex<float>* b, int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                 const std::complex<double>* a, int64_t lda, std::complex<double>* b, int64_t ldb,
                 const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, float alpha, const float* a,
                       int64_t lda, int64_t stride_a, float* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, double alpha, const double* a,
                       int64_t lda, int64_t stride_a, double* b, int64_t ldb, int64_t stride_b,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float>* a, int64_t lda, int64_t stride_a,
                       std::complex<float>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side left_right, uplo upper_lower, transpose trans,
                       diag unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double>* a, int64_t lda, int64_t stride_a,
                       std::complex<double>* b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, float* alpha, const float** a,
                       int64_t* lda, float** b, int64_t* ldb, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, double* alpha, const double** a,
                       int64_t* lda, double** b, int64_t* ldb, int64_t group_count,
                       int64_t* group_size, const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, std::complex<float>** b,
                       int64_t* ldb, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event trsm_batch(sycl::queue& queue, side* left_right, uplo* upper_lower, transpose* trans,
                       diag* unit_diag, int64_t* m, int64_t* n, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, std::complex<double>** b,
                       int64_t* ldb, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const float** a, int64_t* lda,
                       const float** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, double* alpha, const double** a, int64_t* lda,
                       const double** b, int64_t* ldb, double* beta, double** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<float>* alpha,
                       const std::complex<float>** a, int64_t* lda, const std::complex<float>** b,
                       int64_t* ldb, std::complex<float>* beta, std::complex<float>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, std::complex<double>* alpha,
                       const std::complex<double>** a, int64_t* lda, const std::complex<double>** b,
                       int64_t* ldb, std::complex<double>* beta, std::complex<double>** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, sycl::half* alpha, const sycl::half** a,
                       int64_t* lda, const sycl::half** b, int64_t* ldb, sycl::half* beta,
                       sycl::half** c, int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const sycl::half** a, int64_t* lda,
                       const sycl::half** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, float** c, int64_t* ldc,
                       int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose* transa, transpose* transb, int64_t* m,
                       int64_t* n, int64_t* k, float* alpha, const std::int8_t** a, int64_t* lda,
                       const std::int8_t** b, int64_t* ldb, float* beta, std::int32_t** c,
                       int64_t* ldc, int64_t group_count, int64_t* group_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const float* a, int64_t lda, int64_t stride_a,
                       const float* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, double alpha, const double* a, int64_t lda, int64_t stride_a,
                       const double* b, int64_t ldb, int64_t stride_b, double beta, double* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                       int64_t lda, int64_t stride_a, const std::complex<float>* b, int64_t ldb,
                       int64_t stride_b, std::complex<float> beta, std::complex<float>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                       int64_t lda, int64_t stride_a, const std::complex<double>* b, int64_t ldb,
                       int64_t stride_b, std::complex<double> beta, std::complex<double>* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, sycl::half alpha, const sycl::half* a, int64_t lda,
                       int64_t stride_a, const sycl::half* b, int64_t ldb, int64_t stride_b,
                       sycl::half beta, sycl::half* c, int64_t ldc, int64_t stride_c,
                       int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const sycl::half* a, int64_t lda, int64_t stride_a,
                       const sycl::half* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta, float* c,
                       int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                       int64_t k, float alpha, const std::int8_t* a, int64_t lda, int64_t stride_a,
                       const std::int8_t* b, int64_t ldb, int64_t stride_b, float beta,
                       std::int32_t* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float* a, int64_t lda, const float* b,
                  int64_t ldb, float beta, float* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double* a, int64_t lda, const double* b,
                  int64_t ldb, double beta, double* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float>* a,
                  int64_t lda, const std::complex<float>* b, int64_t ldb, std::complex<float> beta,
                  std::complex<float>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemmt(sycl::queue& queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double>* a,
                  int64_t lda, const std::complex<double>* b, int64_t ldb,
                  std::complex<double> beta, std::complex<double>* c, int64_t ldc,
                  const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const std::int8_t* a,
                      int64_t lda, std::int8_t ao, const std::uint8_t* b, int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t* c, int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const std::int8_t* a,
                      int64_t lda, std::int8_t ao, const std::int8_t* b, int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t* c, int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const std::uint8_t* a,
                      int64_t lda, std::uint8_t ao, const std::int8_t* b, int64_t ldb,
                      std::int8_t bo, float beta, std::int32_t* c, int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

sycl::event gemm_bias(sycl::queue& queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const std::uint8_t* a,
                      int64_t lda, std::uint8_t ao, const std::uint8_t* b, int64_t ldb,
                      std::uint8_t bo, float beta, std::int32_t* c, int64_t ldc,
                      const std::int32_t* co, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           const float* a, int64_t lda, int64_t stride_a, float* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           const double* a, int64_t lda, int64_t stride_a, double* b, int64_t ldb,
                           int64_t stride_b, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                           int64_t stride_a, std::complex<float>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                           int64_t stride_a, std::complex<double>* b, int64_t ldb, int64_t stride_b,
                           int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                           float* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                           double* ab, int64_t lda, int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<float> alpha, std::complex<float>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                           std::complex<double> alpha, std::complex<double>* ab, int64_t lda,
                           int64_t ldb, int64_t stride, int64_t batch_size,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, float alpha, const float* a, int64_t lda, int64_t stride_a,
                          float beta, const float* b, int64_t ldb, int64_t stride_b, float* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, double alpha, const double* a, int64_t lda, int64_t stride_a,
                          double beta, const double* b, int64_t ldb, int64_t stride_b, double* c,
                          int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<float> alpha, const std::complex<float>* a,
                          int64_t lda, int64_t stride_a, std::complex<float> beta,
                          const std::complex<float>* b, int64_t ldb, int64_t stride_b,
                          std::complex<float>* c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                          const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd_batch(sycl::queue& queue, transpose transa, transpose transb, int64_t m,
                          int64_t n, std::complex<double> alpha, const std::complex<double>* a,
                          int64_t lda, int64_t stride_a, std::complex<double> beta,
                          const std::complex<double>* b, int64_t ldb, int64_t stride_b,
                          std::complex<double>* c, int64_t ldc, int64_t stride_c,
                          int64_t batch_size, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                     const float* a, int64_t lda, float* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                     const double* a, int64_t lda, double* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                     std::complex<float>* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                     std::complex<double>* b, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                      const float* a, int64_t lda, std::int64_t stridea, float* b, int64_t ldb,
                      std::int64_t strideb, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                      const double* a, int64_t lda, std::int64_t stridea, double* b, int64_t ldb,
                      std::int64_t strideb, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                      std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                      std::int64_t stridea, std::complex<float>* b, int64_t ldb,
                      std::int64_t strideb, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy2(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                      std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                      std::int64_t stridea, std::complex<double>* b, int64_t ldb,
                      std::int64_t strideb, const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, float alpha,
                     float* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n, double alpha,
                     double* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<float> alpha, std::complex<float>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy(sycl::queue& queue, transpose trans, int64_t m, int64_t n,
                     std::complex<double> alpha, std::complex<double>* ab, int64_t lda, int64_t ldb,
                     const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    float alpha, const float* a, int64_t lda, float beta, const float* b,
                    int64_t ldb, float* c, int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    double alpha, const double* a, int64_t lda, double beta, const double* b,
                    int64_t ldb, double* c, int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<float> alpha, const std::complex<float>* a, int64_t lda,
                    std::complex<float> beta, const std::complex<float>* b, int64_t ldb,
                    std::complex<float>* c, int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatadd(sycl::queue& queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    std::complex<double> alpha, const std::complex<double>* a, int64_t lda,
                    std::complex<double> beta, const std::complex<double>* b, int64_t ldb,
                    std::complex<double>* c, int64_t ldc,
                    const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, const float** a, int64_t* lda, float** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, const double** a, int64_t* lda, double** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, const std::complex<float>** a, int64_t* lda,
                           std::complex<float>** b, int64_t* ldb, int64_t group_count,
                           int64_t* groupsize, const std::vector<sycl::event>& dependencies = {});

sycl::event omatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, const std::complex<double>** a,
                           int64_t* lda, std::complex<double>** b, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           float* alpha, float** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           double* alpha, double** ab, int64_t* lda, int64_t* ldb,
                           int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<float>* alpha, std::complex<float>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});

sycl::event imatcopy_batch(sycl::queue& queue, transpose* trans, int64_t* m, int64_t* n,
                           std::complex<double>* alpha, std::complex<double>** ab, int64_t* lda,
                           int64_t* ldb, int64_t group_count, int64_t* groupsize,
                           const std::vector<sycl::event>& dependencies = {});
