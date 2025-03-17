/*******************************************************************************
* Copyright 2025 SiPearl
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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <array>
#include <algorithm>
#include <optional>

#include "oneapi/math/exceptions.hpp"

#include "oneapi/math/dft/detail/commit_impl.hpp"
#include "oneapi/math/dft/detail/descriptor_impl.hpp"
#include "oneapi/math/dft/detail/armpl/onemath_dft_armpl.hpp"
#include "oneapi/math/dft/types.hpp"

#include "../stride_helper.hpp"
#include "fftw3.h"

namespace oneapi::math::dft::armpl {
namespace detail {

/// Commit impl class specialization for armpl/fftw.
template <dft::precision prec, dft::domain dom>
class armpl_commit final : public dft::detail::commit_impl<prec, dom> {
private:
    using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;
    using planType = typename std::conditional_t<prec == oneapi::math::dft::precision::SINGLE,
                                                 fftwf_plan, fftw_plan>;
    // plans[0] is forward, plans[1] is backward
    std::array<std::optional<planType>, 2> plans = { std::nullopt, std::nullopt };
    std::int64_t offset_fwd_in, offset_fwd_out, offset_bwd_in, offset_bwd_out;

public:
    armpl_commit(sycl::queue& queue, const dft::detail::dft_values<prec, dom>& config_values)
            : oneapi::math::dft::detail::commit_impl<prec, dom>(queue, backend::armpl,
                                                                config_values) {
        if constexpr (prec == dft::detail::precision::DOUBLE) {
            if (!queue.get_device().has(sycl::aspect::fp64)) {
                throw math::exception("DFT", "commit", "Device does not support double precision.");
            }
        }
    }

    void clean_plans() {
        if (plans[0] && plans[0] != plans[1]) {
            if constexpr (prec == dft::detail::precision::SINGLE) {
                fftwf_destroy_plan(plans[0].value());
            }
            else {
                fftw_destroy_plan(plans[0].value());
            }

            plans[0] = std::nullopt;
        }
        if (plans[1]) {
            if constexpr (prec == dft::detail::precision::SINGLE) {
                fftwf_destroy_plan(plans[1].value());
            }
            else {
                fftw_destroy_plan(plans[1].value());
            }
            plans[1] = std::nullopt;
        }
    }
    enum class Direction { Forward = FFTW_FORWARD, Backward = FFTW_BACKWARD };
    template <Direction dir, typename forward_data_type, typename backward_data_type,
              typename planType, typename dimsType>
    planType create_plan(unsigned int rank, const dimsType* dims, int howmany_rank,
                         const dimsType* howmany_dims, forward_data_type* in,
                         backward_data_type* out, unsigned flags) {
        constexpr bool is_real_in = std::is_floating_point_v<forward_data_type>;
        constexpr bool is_real_out = std::is_floating_point_v<backward_data_type>;
        using single_type = std::conditional_t<is_real_in, float, std::complex<float>>;
        constexpr bool is_single = std::is_same_v<forward_data_type, single_type>;
        planType plan;
        if constexpr (is_real_in) {
            if constexpr (dir == Direction::Forward) {
                if constexpr (is_single) {
                    plan =
                        fftwf_plan_guru64_dft_r2c((int)rank, dims, howmany_rank, howmany_dims, in,
                                                  reinterpret_cast<fftwf_complex*>(out), flags);
                }
                else {
                    plan = fftw_plan_guru64_dft_r2c((int)rank, dims, howmany_rank, howmany_dims, in,
                                                    reinterpret_cast<fftw_complex*>(out), flags);
                }
            }
        }
        else if constexpr (is_real_out) {
            if constexpr (is_single) {
                plan = fftwf_plan_guru64_dft_c2r((int)rank, dims, howmany_rank, howmany_dims,
                                                 reinterpret_cast<fftwf_complex*>(in), out, flags);
            }
            else {
                plan = fftw_plan_guru64_dft_c2r((int)rank, dims, howmany_rank, howmany_dims,
                                                reinterpret_cast<fftw_complex*>(in), out, flags);
            }
        }
        else {
            if constexpr (is_single) {
                plan =
                    fftwf_plan_guru64_dft((int)rank, dims, howmany_rank, howmany_dims,
                                          reinterpret_cast<fftwf_complex*>(in),
                                          reinterpret_cast<fftwf_complex*>(out), (int)dir, flags);
            }
            else {
                plan = fftw_plan_guru64_dft((int)rank, dims, howmany_rank, howmany_dims,
                                            reinterpret_cast<fftw_complex*>(in),
                                            reinterpret_cast<fftw_complex*>(out), (int)dir, flags);
            }
        }
        return plan;
    }

    template <Direction dir, typename data_type, typename planType, typename dimsType>
    planType create_plan_split(unsigned int rank, const dimsType* dims, int howmany_rank,
                               const dimsType* howmany_dims, data_type* r_in, data_type* i_in,
                               data_type* r_out, data_type* i_out, unsigned flags) {
        constexpr bool is_real_in = std::is_floating_point_v<data_type>;
        using single_type = std::conditional_t<is_real_in, float, std::complex<float>>;
        constexpr bool is_single = std::is_same_v<data_type, single_type>;
        planType plan;
        if constexpr (is_single) {
            plan = fftwf_plan_guru64_split_dft((int)rank, dims, howmany_rank, howmany_dims, r_in,
                                               i_in, r_out, i_out, flags);
        }
        else {
            plan = fftw_plan_guru64_split_dft((int)rank, dims, howmany_rank, howmany_dims, r_in,
                                              i_in, r_out, i_out, flags);
        }
        return plan;
    }

    void commit(const dft::detail::dft_values<prec, dom>& config_values) override {
        // this could be a recommit
        this->external_workspace_helper_ =
            oneapi::math::dft::detail::external_workspace_helper<prec, dom>(
                config_values.workspace_placement ==
                oneapi::math::dft::detail::config_value::WORKSPACE_EXTERNAL);
        clean_plans();

        constexpr std::size_t max_supported_dims = 7;
        std::array<std::uint64_t, max_supported_dims> n_copy;
        std::copy(config_values.dimensions.begin(), config_values.dimensions.end(), n_copy.data());
        const unsigned int rank = static_cast<unsigned int>(config_values.dimensions.size());

        auto stride_api_choice = dft::detail::get_stride_api(config_values);
        dft::detail::throw_on_invalid_stride_api("FFTW commit", stride_api_choice);
        dft::detail::stride_vectors<int> stride_vecs(config_values, stride_api_choice);
        offset_fwd_in = stride_vecs.offset_fwd_in;
        offset_fwd_out = stride_vecs.offset_fwd_out;
        offset_bwd_in = stride_vecs.offset_bwd_in;
        offset_bwd_out = stride_vecs.offset_bwd_out;
        if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE &&
            offset_fwd_in != offset_fwd_out)
            throw oneapi::math::unimplemented(
                "FFTW - Forward offsets should match for inplace transform");
        if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE &&
            offset_bwd_in != offset_bwd_out)
            throw oneapi::math::unimplemented(
                "FFTW - Backward offsets should match for inplace transform");
        if (config_values.fwd_scale != 1.0 || config_values.bwd_scale != 1.0) {
            throw math::unimplemented(
                "dft/backends/armpl", __FUNCTION__,
                "FFTW does not support values other than 1 for FORWARD/BACKWARD_SCALE, scaling has to be done outside of the backend calls");
        }

        const unsigned int batch = static_cast<unsigned int>(config_values.number_of_transforms);
        const int fwd_dist = static_cast<int>(config_values.fwd_dist);
        const int bwd_dist = static_cast<int>(config_values.bwd_dist);

        //FFTW_PATIENT is slower than MEASURE, but this should be done only once in a while and could save a lot later
        unsigned flags = FFTW_PATIENT;

        using FwdInputType = typename std::conditional_t<dom == oneapi::math::dft::domain::REAL,
                                                         scalar_type, std::complex<scalar_type>>;
        using FwdOutputType = std::complex<scalar_type>;
        std::uint64_t num_elements = 1;
        for (unsigned int i = 0; i < rank; i++)
            num_elements *= n_copy[i];

        //need to temporarily allocate arrays for plan optimization
        FwdInputType* in = nullptr;
        FwdOutputType* out = nullptr;
        scalar_type *r_in = nullptr, *i_in = nullptr;
        scalar_type *r_out = nullptr, *i_out = nullptr;

        if (config_values.complex_storage != oneapi::math::dft::detail::config_value::REAL_REAL) {
            in = (FwdInputType*)malloc(batch * num_elements * sizeof(FwdInputType));
            if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE)
                out = (FwdOutputType*)in;
            else
                out = (FwdOutputType*)malloc(batch * num_elements * sizeof(FwdOutputType));
        }
        else {
            if (config_values.complex_storage ==
                oneapi::math::dft::detail::config_value::REAL_REAL) {
                i_in = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
                r_in = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
                if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE) {
                    i_out = i_in;
                    r_out = r_in;
                }
                else {
                    i_out = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
                    r_out = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
                }
            }
        }

        using iodim64Type =
            typename std::conditional_t<prec == oneapi::math::dft::precision::SINGLE, fftwf_iodim64,
                                        fftw_iodim64>;
        iodim64Type* dims = new iodim64Type[rank];
        for (unsigned int i = 0; i < rank; i++) {
            dims[i].n = (ptrdiff_t)(n_copy[i]);
            dims[i].is = stride_vecs.fwd_in[i + 1];
            dims[i].os = stride_vecs.fwd_out[i + 1];
        }

        iodim64Type* howmany_dims = new iodim64Type[1];
        for (unsigned int i = 0; i < 1; i++) {
            howmany_dims[i].n = batch;
            howmany_dims[i].is = fwd_dist;
            howmany_dims[i].os = bwd_dist;
        }
        planType fwd_plan;
        if (config_values.complex_storage == oneapi::math::dft::detail::config_value::REAL_REAL)
            fwd_plan = create_plan_split<Direction::Forward, scalar_type, planType, iodim64Type>(
                rank, dims, 1, howmany_dims, r_in, i_in, r_out, i_out, flags);
        else
            fwd_plan =
                create_plan<Direction::Forward, FwdInputType, FwdOutputType, planType, iodim64Type>(
                    rank, dims, 1, howmany_dims, in, out, flags);
        if (fwd_plan == NULL)
            throw math::exception("dft/backends/armpl", __FUNCTION__, "Invalid plan.");
        plans[0] = fwd_plan;
        if ((void*)in != (void*)out)
            free(in);
        free(out);
        if ((void*)i_in != (void*)i_out) {
            free(i_in);
            free(r_in);
        }
        free(i_out);
        free(r_out);

        if (config_values.complex_storage != oneapi::math::dft::detail::config_value::REAL_REAL) {
            out = (FwdOutputType*)malloc(batch * num_elements * sizeof(FwdOutputType));

            if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE)
                in = (FwdInputType*)out;
            else
                in = (FwdInputType*)malloc(batch * num_elements * sizeof(FwdInputType));
        }
        else {
            i_in = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
            r_in = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
            if (config_values.placement == oneapi::math::dft::detail::config_value::INPLACE) {
                i_out = i_in;
                r_out = r_in;
            }
            else {
                i_out = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
                r_out = (scalar_type*)malloc(batch * num_elements * sizeof(scalar_type));
            }
        }

        for (unsigned int i = 0; i < rank; i++) {
            dims[i].n = (ptrdiff_t)(n_copy[i]);
            dims[i].is = stride_vecs.bwd_in[i + 1];
            dims[i].os = stride_vecs.bwd_out[i + 1];
        }

        for (unsigned int i = 0; i < 1; i++) {
            howmany_dims[i].n = batch;
            howmany_dims[i].is = bwd_dist;
            howmany_dims[i].os = fwd_dist;
        }
        planType bwd_plan = NULL;
        if (config_values.complex_storage == oneapi::math::dft::detail::config_value::REAL_REAL)
            bwd_plan = fwd_plan;
        else
            bwd_plan = create_plan<Direction::Backward, FwdOutputType, FwdInputType, planType,
                                   iodim64Type>(rank, dims, 1, howmany_dims, out, in, flags);

        if (bwd_plan == NULL)
            throw math::exception("dft/backends/armpl", __FUNCTION__, "Invalid plan.");
        plans[1] = bwd_plan;
        if ((void*)in != (void*)out)
            free(in);
        free(out);

        if ((void*)i_in != (void*)i_out) {
            free(i_in);
            free(r_in);
        }

        free(i_out);
        free(r_out);
    }
    ~armpl_commit() override {
        clean_plans();
    }
    void* get_handle() noexcept override {
        return plans.data();
    }

    std::array<std::int64_t, 2> get_offsets_fwd() noexcept {
        return { offset_fwd_in, offset_fwd_out };
    }

    std::array<std::int64_t, 2> get_offsets_bwd() noexcept {
        return { offset_bwd_in, offset_bwd_out };
    }

#define BACKEND armpl
#include "../backend_compute_signature.cxx"
#undef BACKEND
};
} // namespace detail

template <dft::precision prec, dft::domain dom>
dft::detail::commit_impl<prec, dom>* create_commit(const dft::detail::descriptor<prec, dom>& desc,
                                                   sycl::queue& sycl_queue) {
    return new detail::armpl_commit<prec, dom>(sycl_queue, desc.get_values());
}

template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>&,
    sycl::queue&);
template dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*
create_commit(
    const dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>&,
    sycl::queue&);

namespace detail {
template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_fwd(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<armpl_commit<prec, dom>*>(commit)->get_offsets_fwd();
}

template <dft::precision prec, dft::domain dom>
std::array<std::int64_t, 2> get_offsets_bwd(dft::detail::commit_impl<prec, dom>* commit) {
    return static_cast<armpl_commit<prec, dom>*>(commit)->get_offsets_bwd();
}

template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::SINGLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_fwd<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*);

template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::SINGLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>*);
template std::array<std::int64_t, 2>
get_offsets_bwd<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>(
    dft::detail::commit_impl<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>*);

} //namespace detail

} // namespace oneapi::math::dft::armpl
