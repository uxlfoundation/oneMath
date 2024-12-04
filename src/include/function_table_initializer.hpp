/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _LOADER_HPP_
#define _LOADER_HPP_

#include <cstdint>
#include <map>

#include "oneapi/math/detail/backends_table.hpp"
#include "oneapi/math/detail/exceptions.hpp"

#define SPEC_VERSION 1

#ifdef __linux__
#include <dlfcn.h>
#define LIB_TYPE                 void*
#define GET_LIB_HANDLE(libname)  dlopen((libname), RTLD_LAZY | RTLD_GLOBAL)
#define GET_FUNC(lib, fn)        dlsym(lib, (fn))
#define FREE_LIB_HANDLE(libname) dlclose(libname)
#define ERROR_MSG                dlerror()
#elif defined(_WIN64)
#include <windows.h>
#define LIB_TYPE                 HINSTANCE
#define GET_LIB_HANDLE(libname)  LoadLibrary(libname)
#define GET_FUNC(lib, fn)        GetProcAddress((lib), (fn))
#define FREE_LIB_HANDLE(libname) FreeLibrary(libname)
#define ERROR_MSG                GetLastErrorStdStr()
#endif

namespace oneapi {
namespace math {
namespace detail {

template <oneapi::math::domain domain_id, typename function_table_t>
class table_initializer {
    struct handle_deleter {
        using pointer = LIB_TYPE;
        void operator()(pointer p) const {
            ::FREE_LIB_HANDLE(p);
        }
    };
    using dlhandle = std::unique_ptr<LIB_TYPE, handle_deleter>;

public:
    function_table_t& operator[](std::pair<oneapi::math::device, sycl::queue> device_queue_pair) {
        auto lib = tables.find(device_queue_pair.first);
        if (lib != tables.end())
            return lib->second;
        return add_table(device_queue_pair.first, device_queue_pair.second);
    }

private:
#if defined(ENABLE_PORTBLAS_BACKEND) || defined(ENABLE_PORTFFT_BACKEND)
    static constexpr bool is_generic_device_supported = true;
#else
    static constexpr bool is_generic_device_supported = false;
#endif

#ifdef _WIN64
    // Create a string with last error message
    std::string GetLastErrorStdStr() {
        DWORD error = GetLastError();
        if (error) {
            LPVOID lpMsgBuf;
            DWORD bufLen = FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
            if (bufLen) {
                LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
                std::string result(lpMsgStr, lpMsgStr + bufLen);

                LocalFree(lpMsgBuf);

                return result;
            }
        }
        return std::string();
    }
#endif

    function_table_t& add_table(oneapi::math::device key, sycl::queue& q) {
        dlhandle handle;
        // check all available libraries for the key(device)
        for (const char* libname : libraries[domain_id][key]) {
            handle = dlhandle{ ::GET_LIB_HANDLE(libname) };
            if (handle)
                break;
        }
        if (!handle) {
            if (!is_generic_device_supported && key == oneapi::math::device::generic_device) {
                throw math::unsupported_device("", "", q.get_device());
            }
            else {
                std::cerr << ERROR_MSG << '\n';
                throw math::backend_not_found();
            }
        }
        auto t =
            reinterpret_cast<function_table_t*>(::GET_FUNC(handle.get(), table_names[domain_id]));

        if (!t) {
            std::cerr << ERROR_MSG << '\n';
            throw math::function_not_found();
        }
        if (t->version != SPEC_VERSION)
            throw math::specification_mismatch();

        handles[key] = std::move(handle);
        tables[key] = *t;
        return *t;
    }

    std::map<oneapi::math::device, function_table_t> tables;
    std::map<oneapi::math::device, dlhandle> handles;
};

} //namespace detail
} // namespace math
} // namespace oneapi

#endif //_LOADER_HPP_
