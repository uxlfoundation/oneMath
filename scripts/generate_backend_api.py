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
Script to generate backend library header based on base_header.h
Note: requires clang-format tool to be installed
Usage:

    {script} <path/to/base_header.hpp> <path/to/backend_include.hpp> <namespace>

Example:
The command below will generate:
"onemkl_blas_mklgpu.hpp" header with declaration of all backend library APIs.
API from backend library will be called from "oneapi::mkl::mklgpu::blas" namespace.

{script}  include/oneapi/mkl/blas.hpp include/oneapi/mkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp oneapi::mkl::mklgpu::blas
'''.format(script = argv[0]))

if len(argv) < 3:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_filename = argv[1]
out_headername = argv[2]
namespace = argv[3]

namespace_list=namespace.split("::")

header_db = create_func_db(in_filename)

print("Generate " + out_headername)

def print_declaration(func_list):
    code=""
    for data in func_list:
        code +="""
{ret_type} {name}{par_str};

""".format(**data)
    return code

try:
    os.makedirs(os.path.dirname(out_headername))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(out_headername, "w+")
out_file.write("""//
// Generated based on {in_filename}
//

#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
""".format(in_filename=in_filename))

for nmsp in namespace_list:
    out_file.write("""namespace {name} {{
""".format(name=nmsp))

for func_name, func_list in header_db.items():
    out_file.write("""
{funcs}""".format(funcs=print_declaration(func_list)))

for nmsp in reversed(namespace_list):
    out_file.write("""}} // namespace {name} {{
""".format(name=nmsp))

out_file.close()

print("Formatting with clang-format " + out_headername)
try:
    lc = ["clang-format", "-style=file", "-i", out_headername]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")

