/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

namespace oneapi {
namespace mkl {
namespace lapack {

class exception {
public:
    exception(oneapi::mkl::exception* _ex, std::int64_t info, std::int64_t detail = 0)
            : _info(info),
              _detail(detail),
              _ex(_ex) {}
    std::int64_t info() const {
        return _info;
    }
    std::int64_t detail() const {
        return _detail;
    }
    const char* what() const {
        return _ex->what();
    }

private:
    std::int64_t _info;
    std::int64_t _detail;
    mkl::exception* _ex;
};

class computation_error : public oneapi::mkl::computation_error,
                          public oneapi::mkl::lapack::exception {
public:
    computation_error(const std::string& function, const std::string& info, std::int64_t code)
            : oneapi::mkl::computation_error("LAPACK", function, info),
              oneapi::mkl::lapack::exception(this, code) {}
    using oneapi::mkl::computation_error::what;
};

class batch_error : public oneapi::mkl::batch_error, public oneapi::mkl::lapack::exception {
public:
    batch_error(const std::string& function, const std::string& info, std::int64_t num_errors,
                std::vector<std::int64_t> ids = {}, std::vector<std::exception_ptr> exceptions = {})
            : oneapi::mkl::batch_error("LAPACK", function, info),
              oneapi::mkl::lapack::exception(this, num_errors),
              _ids(ids),
              _exceptions(exceptions) {}
    using oneapi::mkl::batch_error::what;
    const std::vector<std::int64_t>& ids() const {
        return _ids;
    }
    const std::vector<std::exception_ptr>& exceptions() const {
        return _exceptions;
    }

private:
    std::vector<std::int64_t> _ids;
    std::vector<std::exception_ptr> _exceptions;
};

class invalid_argument : public oneapi::mkl::invalid_argument,
                         public oneapi::mkl::lapack::exception {
public:
    invalid_argument(const std::string& function, const std::string& info,
                     std::int64_t arg_position = 0, std::int64_t detail = 0)
            : oneapi::mkl::invalid_argument("LAPACK", function, info),
              oneapi::mkl::lapack::exception(this, arg_position, detail) {}
    using oneapi::mkl::invalid_argument::what;
};

} // namespace lapack
} // namespace mkl
} // namespace oneapi
