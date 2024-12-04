#!/usr/bin/env python
#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

from sys import argv, exit, stdin
from subprocess import call
from pprint import pprint
from collections import defaultdict
import errno
import re
import os

from func_parser import create_func_db, get_namespaces

def usage(err = None):
    if err:
        print('error: %s' % err)
    print('''\
Script to generate CT API instantiations for backend based on general_ct_templates.hpp
Note: requires clang-format tool to be installed
Usage:

    {script} <path/to/general_ct_templates.hpp> <path/to/out_ct_header.hpp> <path/to/backend_include.hpp> <backend> <namespace>

Example:
The command below will generate:
"blas_ct.hpp" header with compile-time BLAS API based on "blas_ct_templates.hpp" for "mklgpu" backend.
API from the backend library will be called from "oneapi::math::mklgpu::blas" namespace.

{script}  include/oneapi/math/blas/detail/blas_ct_templates.hpp include/oneapi/math/blas/detail/mklgpu/blas_ct.hpp include/oneapi/math/blas/detail/mklgpu/onemath_blas_mklgpu.hpp mklgpu oneapi::math::mklgpu::blas
'''.format(script = argv[0]))

if len(argv) < 6:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_filename = argv[1]
out_filename = argv[2]
include = argv[3]
backend = argv[4]
namespace = argv[5]

namespace_list=namespace.split("::")

header_db = create_func_db(in_filename)
external_namespace_list=get_namespaces(in_filename)

print("Generate " + out_filename)

def print_funcs(func_list):
    code=""
    for data in func_list:
        code +="""
template<>
{ret_type} {name}<backend::{backend}>{par_str} {{
    {name}_precondition{call_str};
    {namespace}::{name}{call_str};
    {name}_postcondition{call_str};
}}
""".format(namespace=namespace, backend=backend, **data)
    return code

try:
    os.makedirs(os.path.dirname(out_filename))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(out_filename, "w+")
out_file.write("""//
// Generated based on {in_header}
//

#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/types.hpp"
#include "oneapi/math/detail/backends.hpp"
#include "{internal_api}"
#include "{ct_teplates}"

""".format(in_header=in_filename, ct_teplates=in_filename.strip("include/"), internal_api=include.strip("include/")))


for nmsp in external_namespace_list:
    out_file.write("""namespace {name} {{
""".format(name=nmsp))

for func_name, func_list in header_db.items():
    out_file.write("""
{funcs}""".format(funcs=print_funcs(func_list)))


for nmsp in reversed(external_namespace_list):
    out_file.write("""}} // namespace {name} {{
""".format(name=nmsp))

out_file.close()

print("Formatting with clang-format " + out_filename)
try:
    lc = ["clang-format", "-style=file", "-i", out_filename]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")

