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

#include <CL/sycl.hpp>
#include <chrono>
#include <thread>

#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"

namespace global {
std::vector<int64_t> host_data(1024);
int64_t* device_data = nullptr;
} // namespace global

sycl::event create_dependent_event(sycl::queue queue) {
    global::device_data = device_alloc<int64_t>(queue, global::host_data.size());
    return host_to_device_copy(queue, global::host_data.data(), global::device_data,
                               global::host_data.size());
}

Dependency_Result get_result(sycl::info::event_command_status in_status,
                             sycl::info::event_command_status func_status) {
    /*   in\func | submitted | running  | complete */
    /* submitted |   inc.    |   fail   |  fail    */
    /* running   |   pass    |   fail   |  fail    */
    /* complete  |   inc.    |   inc.   |  inc.    */
    if (in_status == sycl::info::event_command_status::submitted) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::fail;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::fail;
    }
    else if (in_status == sycl::info::event_command_status::running) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::pass;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::fail;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::fail;
    }
    else if (in_status == sycl::info::event_command_status::complete) {
        if (func_status == sycl::info::event_command_status::submitted)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::running)
            return Dependency_Result::inconclusive;
        else if (func_status == sycl::info::event_command_status::complete)
            return Dependency_Result::inconclusive;
    }

    return Dependency_Result::unknown;
}

void print_status(const char* name, sycl::info::event_command_status status) {
    global::log << name << " command execution status: ";
    if (sycl::info::event_command_status::submitted == status)
        global::log << "submitted";
    else if (sycl::info::event_command_status::running == status)
        global::log << "running";
    else if (sycl::info::event_command_status::complete == status)
        global::log << "complete";
    else
        global::log << "status unknown";
    global::log << " (" << static_cast<int64_t>(status) << ")" << std::endl;
}

bool check_dependency(sycl::queue queue, sycl::event in_event, sycl::event func_event) {
    auto result = Dependency_Result::inconclusive;
    sycl::info::event_command_status in_status;
    sycl::info::event_command_status func_status;

    do {
        in_status = in_event.get_info<sycl::info::event::command_execution_status>();
        func_status = func_event.get_info<sycl::info::event::command_execution_status>();

        auto temp_result = get_result(in_status, func_status);
        if (temp_result == Dependency_Result::pass || temp_result == Dependency_Result::fail)
            result = temp_result;

    } while (in_status != sycl::info::event_command_status::complete &&
             result != Dependency_Result::fail);

    /* Print results */
    if (result == Dependency_Result::pass)
        global::log << "Dependency Test: Successful" << std::endl;
    if (result == Dependency_Result::inconclusive)
        global::log << "Dependency Test: Inconclusive" << std::endl;
    if (result == Dependency_Result::fail)
        global::log << "Dependency Test: Failed" << std::endl;
    print_status("in_event", in_status);
    print_status("func_event", func_status);

    device_free(queue, global::device_data);
    return (result == Dependency_Result::pass || result == Dependency_Result::inconclusive) ? true
                                                                                            : false;
}
