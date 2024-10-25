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
Script to generate blank wrappers and pointers table based on header.hpp
Note: requires clang-format tool
Usage:

    {script} <path/to/header.hpp> <path/to/table.hpp> <path/to/out_wrappers.cpp> <libname>

Example:

    {script}  include/oneapi/mkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp src/blas/function_table.hpp src/blas/backend/mklgpu/wrappers.cpp mklgpu
'''.format(script = argv[0]))

if len(argv) <= 4:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_filename = argv[1]
in_table = argv[2]
out_filename = argv[3]
libname = argv[4]

table_list = argv[0].rsplit('/', 1)[0] + "/blas_list.txt"
table_file = out_filename.rsplit('/', 1)[0] + "/" + libname + "_wrappers_table_dyn.cpp"

cmake_file = out_filename.rsplit('/', 1)[0] + "/CMakeLists.txt"

header_db = create_func_db(in_filename)
namespace_list = get_namespaces(in_filename)

# Generate wrappers
print("Generate " + out_filename)

def print_funcs(func_list):
    code=""
    for data in func_list:
        code +="""
{ret_type} {name}{par_str} {{
    throw std::runtime_error("Not implemented for {libname}");
}}
""".format(libname=libname, **data)
    return code

try:
    os.makedirs(os.path.dirname(out_filename))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(out_filename, "w+")
out_file.write("""//
// generated file
//

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/types.hpp"

#include "{header}"

""".format(header=in_filename.strip("include/")))

for nmsp in namespace_list:
    out_file.write("""namespace {name} {{
""".format(name=nmsp))

for func_name, func_list in header_db.items():
    out_file.write("""
{funcs}""".format(funcs=print_funcs(func_list)))

out_file.write("\n")
for nmsp in reversed(namespace_list):
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
    else:
        raise

# Generate table
print("Generate " + table_file)

try:
    os.makedirs(os.path.dirname(table_file))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

out_file = open(table_file, "w+")
out_file.write("""//
// generated file
//

#include "{header}"
#include "{table}"

#define WRAPPER_VERSION 1

extern "C" function_table_t mkl_blas_table = {{
    WRAPPER_VERSION,
""".format(table=in_table.strip('src/'), header=in_filename.strip('include/')))

namespace = ""
for nmsp in namespace_list:
    namespace = namespace + nmsp.strip() + "::"
with open(table_list, "r") as f:
    table = f.readlines()

for t in table:
    out_file.write("    " + namespace + t.strip() + ",\n")


out_file.write("\n};\n")
out_file.close()

print("Formatting with clang-format " + table_file)
try:
    lc = ["clang-format", "-style=file", "-i", table_file]
    call(lc)
except OSError as exc:
    if exc.errno == errno.ENOENT:
        print("Error: clang-format is not found")
    else:
        raise
