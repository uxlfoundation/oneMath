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

#include <complex>
#include <sstream>
#include <string>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "lapack_common.hpp"
#include "oneapi/math/exceptions.hpp"

template <class T>
std::istream& operator>>(std::istream& is, T& t) {
    int64_t i;
    is >> i;
    t = static_cast<T>(i);
    return is;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::job& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::jobsvd& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::transpose& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::uplo& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::side& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::diag& t) {
    os << static_cast<int64_t>(t);
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const oneapi::math::generate& t) {
    os << static_cast<int64_t>(t);
    return os;
}

class result_T {
public:
    enum class result { fail, pass, exception };

    result_T() : result_{ result::pass } {}
    result_T(bool b) : result_{ b ? result::pass : result::fail } {}
    result_T(const std::exception& e, result t = result::exception)
            : result_{ t },
              what_{ e.what() } {}

    operator bool() const& {
        return result_ == result::pass;
    }

    friend bool operator==(const result_T& lhs, const result_T& rhs);
    friend std::ostream& operator<<(std::ostream& os, result_T result);

private:
    result result_;
    std::string what_;
};

inline bool operator==(const result_T& lhs, const result_T& rhs) {
    return (lhs.result_ == rhs.result_ && lhs.what_ == rhs.what_);
}
inline bool operator!=(const result_T& lhs, const result_T& rhs) {
    return !(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, result_T result) {
    switch (result.result_) {
        case result_T::result::pass: os << "PASS"; break;
        case result_T::result::fail: os << "FAIL"; break;
        case result_T::result::exception: os << "EXCEPTION " << result.what_; break;
    }
    return os;
}

template <typename T>
struct function_info;

template <typename... Args>
struct function_info<bool(const sycl::device&, Args...)> {
    using arg_type = std::tuple<Args...>;
    static constexpr size_t arg_count = sizeof...(Args);

    template <size_t n>
    struct arg {
        using type = typename std::tuple_element<n, std::tuple<Args...>>::type;
    };
};

template <typename T>
struct InputTestController {
    using TestPointer = T;
    using ArgTuple_T = typename function_info<T>::arg_type;
    static constexpr size_t arg_count = function_info<T>::arg_count;
    std::vector<ArgTuple_T> vargs;

    InputTestController(const char* input) {
        if constexpr (arg_count == 0) /* test does not take input */
            return;

        if (input) {
            std::stringstream input_stream(input);
            if (input_stream.fail())
                std::cout << "Failed to process input: \'" << input << "\'" << std::endl;
            else
                store_input(input_stream, std::make_index_sequence<arg_count>());
        }
        else { /* search for input file */
            std::cout << "Test parameters not found" << std::endl;
        }
    }

    template <size_t... I>
    void store_input(std::istream& input_stream, std::index_sequence<I...>) {
        if constexpr (arg_count == 0) /* test does not take input */
            return;
        else {
            ArgTuple_T args;
            while ((..., (input_stream >> std::get<I>(args)))) {
                vargs.push_back(args);
                input_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    template <size_t... I>
    void print_result(size_t input_file_line, result_T result, const ArgTuple_T& args = {},
                      std::index_sequence<I...> = std::make_index_sequence<0>{}) {
        std::cout.clear();
        std::cout << test_log::padding << "[" << input_file_line << "]: ";
        (..., (std::cout << std::get<I>(args) << " "));
        std::cout << "# " << result << std::endl;
        test_log::print();
    }

    result_T call_test(TestPointer tp, const sycl::device& dev, ArgTuple_T args) {
        auto tp_args = tuple_cat(std::make_tuple(dev), args);
        result_T result;
        try {
            result = std::apply(tp, tp_args);
        }
        catch (const oneapi::math::unsupported_device& e) {
            result = result_T{ e, result_T::result::pass };
        }
        catch (const oneapi::math::unimplemented& e) {
            result = result_T{ e, result_T::result::pass };
        }
        catch (const std::exception& e) {
            result = result_T{ e };
        }
        return result;
    }

    result_T run(TestPointer tp, const sycl::device& dev) {
        print_device_info(dev);
        if constexpr (arg_count == 0) { /* test does not take input */
            result_T result = call_test(tp, dev, {});
            print_result(0, result);
            return result;
        }
        else {
            if (!vargs.size()) {
                test_log::lout << arg_count << " inputs expected, found none" << std::endl;
                print_result(1, false);
            }
            result_T aggregate_result;
            size_t input_file_line = 1;
            for (auto& args : vargs) {
                result_T result = call_test(tp, dev, args);
                if (!result) {
                    aggregate_result = result;
                }
                print_result(input_file_line++, result, args,
                             std::make_index_sequence<arg_count>());
            }
            return aggregate_result;
        }
    }

    result_T run_print_on_fail(TestPointer tp, const sycl::device& dev) {
        print_device_info(dev);
        if constexpr (arg_count == 0) { /* test does not take input */
            result_T result = call_test(tp, dev, {});
            if (!result) {
                print_result(0, result);
            }
            else {
                test_log::lout.str("");
                test_log::lout.clear();
            }
            return result;
        }
        else {
            if (!vargs.size()) {
                test_log::lout << arg_count << " inputs expected, found none" << std::endl;
                print_result(1, false);
            }
            result_T aggregate_result;
            size_t input_file_line = 0;
            for (auto& args : vargs) {
                input_file_line++;
                result_T result = call_test(tp, dev, args);
                if (!result) {
                    print_result(input_file_line, result, args,
                                 std::make_index_sequence<arg_count>());
                }
                else {
                    test_log::lout.str("");
                    test_log::lout.clear();
                }
                if (!result)
                    aggregate_result = result;
            }
            return aggregate_result;
        }
    }
};
