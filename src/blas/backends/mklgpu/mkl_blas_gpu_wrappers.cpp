/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "blas/function_table.hpp"
#include "oneapi/mkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT oneapi::mkl::blas::detail::function_table_t mkl_blas_table = {
    WRAPPER_VERSION,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dotc,
    oneapi::mkl::mklgpu::column_major::dotc,
    oneapi::mkl::mklgpu::column_major::dotu,
    oneapi::mkl::mklgpu::column_major::dotu,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotm,
    oneapi::mkl::mklgpu::column_major::rotm,
    oneapi::mkl::mklgpu::column_major::rotmg,
    oneapi::mkl::mklgpu::column_major::rotmg,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::sdsdot,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::ger,
    oneapi::mkl::mklgpu::column_major::ger,
    oneapi::mkl::mklgpu::column_major::gerc,
    oneapi::mkl::mklgpu::column_major::gerc,
    oneapi::mkl::mklgpu::column_major::geru,
    oneapi::mkl::mklgpu::column_major::geru,
    oneapi::mkl::mklgpu::column_major::hbmv,
    oneapi::mkl::mklgpu::column_major::hbmv,
    oneapi::mkl::mklgpu::column_major::hemv,
    oneapi::mkl::mklgpu::column_major::hemv,
    oneapi::mkl::mklgpu::column_major::her,
    oneapi::mkl::mklgpu::column_major::her,
    oneapi::mkl::mklgpu::column_major::her2,
    oneapi::mkl::mklgpu::column_major::her2,
    oneapi::mkl::mklgpu::column_major::hpmv,
    oneapi::mkl::mklgpu::column_major::hpmv,
    oneapi::mkl::mklgpu::column_major::hpr,
    oneapi::mkl::mklgpu::column_major::hpr,
    oneapi::mkl::mklgpu::column_major::hpr2,
    oneapi::mkl::mklgpu::column_major::hpr2,
    oneapi::mkl::mklgpu::column_major::sbmv,
    oneapi::mkl::mklgpu::column_major::sbmv,
    oneapi::mkl::mklgpu::column_major::spmv,
    oneapi::mkl::mklgpu::column_major::spmv,
    oneapi::mkl::mklgpu::column_major::spr,
    oneapi::mkl::mklgpu::column_major::spr,
    oneapi::mkl::mklgpu::column_major::spr2,
    oneapi::mkl::mklgpu::column_major::spr2,
    oneapi::mkl::mklgpu::column_major::symv,
    oneapi::mkl::mklgpu::column_major::symv,
    oneapi::mkl::mklgpu::column_major::syr,
    oneapi::mkl::mklgpu::column_major::syr,
    oneapi::mkl::mklgpu::column_major::syr2,
    oneapi::mkl::mklgpu::column_major::syr2,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::hemm,
    oneapi::mkl::mklgpu::column_major::hemm,
    oneapi::mkl::mklgpu::column_major::herk,
    oneapi::mkl::mklgpu::column_major::herk,
    oneapi::mkl::mklgpu::column_major::her2k,
    oneapi::mkl::mklgpu::column_major::her2k,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::trsm_batch,
    oneapi::mkl::mklgpu::column_major::trsm_batch,
    oneapi::mkl::mklgpu::column_major::trsm_batch,
    oneapi::mkl::mklgpu::column_major::trsm_batch,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemm_bias,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::asum,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy,
    oneapi::mkl::mklgpu::column_major::axpy_batch,
    oneapi::mkl::mklgpu::column_major::axpy_batch,
    oneapi::mkl::mklgpu::column_major::axpy_batch,
    oneapi::mkl::mklgpu::column_major::axpy_batch,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::copy,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dot,
    oneapi::mkl::mklgpu::column_major::dotc,
    oneapi::mkl::mklgpu::column_major::dotc,
    oneapi::mkl::mklgpu::column_major::dotu,
    oneapi::mkl::mklgpu::column_major::dotu,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamin,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::iamax,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::nrm2,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rot,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotg,
    oneapi::mkl::mklgpu::column_major::rotm,
    oneapi::mkl::mklgpu::column_major::rotm,
    oneapi::mkl::mklgpu::column_major::rotmg,
    oneapi::mkl::mklgpu::column_major::rotmg,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::scal,
    oneapi::mkl::mklgpu::column_major::sdsdot,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::swap,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gbmv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::gemv,
    oneapi::mkl::mklgpu::column_major::ger,
    oneapi::mkl::mklgpu::column_major::ger,
    oneapi::mkl::mklgpu::column_major::gerc,
    oneapi::mkl::mklgpu::column_major::gerc,
    oneapi::mkl::mklgpu::column_major::geru,
    oneapi::mkl::mklgpu::column_major::geru,
    oneapi::mkl::mklgpu::column_major::hbmv,
    oneapi::mkl::mklgpu::column_major::hbmv,
    oneapi::mkl::mklgpu::column_major::hemv,
    oneapi::mkl::mklgpu::column_major::hemv,
    oneapi::mkl::mklgpu::column_major::her,
    oneapi::mkl::mklgpu::column_major::her,
    oneapi::mkl::mklgpu::column_major::her2,
    oneapi::mkl::mklgpu::column_major::her2,
    oneapi::mkl::mklgpu::column_major::hpmv,
    oneapi::mkl::mklgpu::column_major::hpmv,
    oneapi::mkl::mklgpu::column_major::hpr,
    oneapi::mkl::mklgpu::column_major::hpr,
    oneapi::mkl::mklgpu::column_major::hpr2,
    oneapi::mkl::mklgpu::column_major::hpr2,
    oneapi::mkl::mklgpu::column_major::sbmv,
    oneapi::mkl::mklgpu::column_major::sbmv,
    oneapi::mkl::mklgpu::column_major::spmv,
    oneapi::mkl::mklgpu::column_major::spmv,
    oneapi::mkl::mklgpu::column_major::spr,
    oneapi::mkl::mklgpu::column_major::spr,
    oneapi::mkl::mklgpu::column_major::spr2,
    oneapi::mkl::mklgpu::column_major::spr2,
    oneapi::mkl::mklgpu::column_major::symv,
    oneapi::mkl::mklgpu::column_major::symv,
    oneapi::mkl::mklgpu::column_major::syr,
    oneapi::mkl::mklgpu::column_major::syr,
    oneapi::mkl::mklgpu::column_major::syr2,
    oneapi::mkl::mklgpu::column_major::syr2,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbmv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tbsv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpmv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::tpsv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trmv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::trsv,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::gemm,
    oneapi::mkl::mklgpu::column_major::hemm,
    oneapi::mkl::mklgpu::column_major::hemm,
    oneapi::mkl::mklgpu::column_major::herk,
    oneapi::mkl::mklgpu::column_major::herk,
    oneapi::mkl::mklgpu::column_major::her2k,
    oneapi::mkl::mklgpu::column_major::her2k,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::symm,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syrk,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::syr2k,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trmm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::trsm,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemm_batch,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::column_major::gemmt,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dotc,
    oneapi::mkl::mklgpu::row_major::dotc,
    oneapi::mkl::mklgpu::row_major::dotu,
    oneapi::mkl::mklgpu::row_major::dotu,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotm,
    oneapi::mkl::mklgpu::row_major::rotm,
    oneapi::mkl::mklgpu::row_major::rotmg,
    oneapi::mkl::mklgpu::row_major::rotmg,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::sdsdot,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::ger,
    oneapi::mkl::mklgpu::row_major::ger,
    oneapi::mkl::mklgpu::row_major::gerc,
    oneapi::mkl::mklgpu::row_major::gerc,
    oneapi::mkl::mklgpu::row_major::geru,
    oneapi::mkl::mklgpu::row_major::geru,
    oneapi::mkl::mklgpu::row_major::hbmv,
    oneapi::mkl::mklgpu::row_major::hbmv,
    oneapi::mkl::mklgpu::row_major::hemv,
    oneapi::mkl::mklgpu::row_major::hemv,
    oneapi::mkl::mklgpu::row_major::her,
    oneapi::mkl::mklgpu::row_major::her,
    oneapi::mkl::mklgpu::row_major::her2,
    oneapi::mkl::mklgpu::row_major::her2,
    oneapi::mkl::mklgpu::row_major::hpmv,
    oneapi::mkl::mklgpu::row_major::hpmv,
    oneapi::mkl::mklgpu::row_major::hpr,
    oneapi::mkl::mklgpu::row_major::hpr,
    oneapi::mkl::mklgpu::row_major::hpr2,
    oneapi::mkl::mklgpu::row_major::hpr2,
    oneapi::mkl::mklgpu::row_major::sbmv,
    oneapi::mkl::mklgpu::row_major::sbmv,
    oneapi::mkl::mklgpu::row_major::spmv,
    oneapi::mkl::mklgpu::row_major::spmv,
    oneapi::mkl::mklgpu::row_major::spr,
    oneapi::mkl::mklgpu::row_major::spr,
    oneapi::mkl::mklgpu::row_major::spr2,
    oneapi::mkl::mklgpu::row_major::spr2,
    oneapi::mkl::mklgpu::row_major::symv,
    oneapi::mkl::mklgpu::row_major::symv,
    oneapi::mkl::mklgpu::row_major::syr,
    oneapi::mkl::mklgpu::row_major::syr,
    oneapi::mkl::mklgpu::row_major::syr2,
    oneapi::mkl::mklgpu::row_major::syr2,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::hemm,
    oneapi::mkl::mklgpu::row_major::hemm,
    oneapi::mkl::mklgpu::row_major::herk,
    oneapi::mkl::mklgpu::row_major::herk,
    oneapi::mkl::mklgpu::row_major::her2k,
    oneapi::mkl::mklgpu::row_major::her2k,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::trsm_batch,
    oneapi::mkl::mklgpu::row_major::trsm_batch,
    oneapi::mkl::mklgpu::row_major::trsm_batch,
    oneapi::mkl::mklgpu::row_major::trsm_batch,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemm_bias,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::asum,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy,
    oneapi::mkl::mklgpu::row_major::axpy_batch,
    oneapi::mkl::mklgpu::row_major::axpy_batch,
    oneapi::mkl::mklgpu::row_major::axpy_batch,
    oneapi::mkl::mklgpu::row_major::axpy_batch,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::copy,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dot,
    oneapi::mkl::mklgpu::row_major::dotc,
    oneapi::mkl::mklgpu::row_major::dotc,
    oneapi::mkl::mklgpu::row_major::dotu,
    oneapi::mkl::mklgpu::row_major::dotu,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamin,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::iamax,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::nrm2,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rot,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotg,
    oneapi::mkl::mklgpu::row_major::rotm,
    oneapi::mkl::mklgpu::row_major::rotm,
    oneapi::mkl::mklgpu::row_major::rotmg,
    oneapi::mkl::mklgpu::row_major::rotmg,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::scal,
    oneapi::mkl::mklgpu::row_major::sdsdot,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::swap,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gbmv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::gemv,
    oneapi::mkl::mklgpu::row_major::ger,
    oneapi::mkl::mklgpu::row_major::ger,
    oneapi::mkl::mklgpu::row_major::gerc,
    oneapi::mkl::mklgpu::row_major::gerc,
    oneapi::mkl::mklgpu::row_major::geru,
    oneapi::mkl::mklgpu::row_major::geru,
    oneapi::mkl::mklgpu::row_major::hbmv,
    oneapi::mkl::mklgpu::row_major::hbmv,
    oneapi::mkl::mklgpu::row_major::hemv,
    oneapi::mkl::mklgpu::row_major::hemv,
    oneapi::mkl::mklgpu::row_major::her,
    oneapi::mkl::mklgpu::row_major::her,
    oneapi::mkl::mklgpu::row_major::her2,
    oneapi::mkl::mklgpu::row_major::her2,
    oneapi::mkl::mklgpu::row_major::hpmv,
    oneapi::mkl::mklgpu::row_major::hpmv,
    oneapi::mkl::mklgpu::row_major::hpr,
    oneapi::mkl::mklgpu::row_major::hpr,
    oneapi::mkl::mklgpu::row_major::hpr2,
    oneapi::mkl::mklgpu::row_major::hpr2,
    oneapi::mkl::mklgpu::row_major::sbmv,
    oneapi::mkl::mklgpu::row_major::sbmv,
    oneapi::mkl::mklgpu::row_major::spmv,
    oneapi::mkl::mklgpu::row_major::spmv,
    oneapi::mkl::mklgpu::row_major::spr,
    oneapi::mkl::mklgpu::row_major::spr,
    oneapi::mkl::mklgpu::row_major::spr2,
    oneapi::mkl::mklgpu::row_major::spr2,
    oneapi::mkl::mklgpu::row_major::symv,
    oneapi::mkl::mklgpu::row_major::symv,
    oneapi::mkl::mklgpu::row_major::syr,
    oneapi::mkl::mklgpu::row_major::syr,
    oneapi::mkl::mklgpu::row_major::syr2,
    oneapi::mkl::mklgpu::row_major::syr2,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbmv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tbsv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpmv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::tpsv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trmv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::trsv,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::gemm,
    oneapi::mkl::mklgpu::row_major::hemm,
    oneapi::mkl::mklgpu::row_major::hemm,
    oneapi::mkl::mklgpu::row_major::herk,
    oneapi::mkl::mklgpu::row_major::herk,
    oneapi::mkl::mklgpu::row_major::her2k,
    oneapi::mkl::mklgpu::row_major::her2k,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::symm,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syrk,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::syr2k,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trmm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::trsm,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemm_batch,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
    oneapi::mkl::mklgpu::row_major::gemmt,
};
