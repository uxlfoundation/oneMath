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
import re
import os

from func_parser import create_func_db, get_namespaces

def usage(err = None):
    if err:
        print('error: %s' % err)
    print('''\
Script to generate CMakeLists.txt for all files in the specified directory
Usage:

    {script} <path/to/directory> <libname>

Example:

    {script}  include/oneapi/math/blas/detail/mklgpu mklgpu
'''.format(script = argv[0]))

if len(argv) <= 2:
    usage()
    exit(0)

if re.search(r'[-]*\b[h]([e][l][p])?\b' ,argv[1]):
    usage()
    exit(0)

in_dir = argv[1]
libname = argv[2]

if not os.path.exists(in_dir):
    print("Error: directory " + in_dir + " doesn't exist\n")
    exit(1)

cmake_file = in_dir + "/CMakeLists.txt"

if os.path.exists(cmake_file):
    print("Error: file " + cmake_file + " already exists\n")
    exit(1)
else:
    print("Generate " + cmake_file)

file_list = os.listdir(in_dir)

out_file = open(cmake_file, "w+")

out_file.write("""#
# generated file
#

set(LIB_NAME onemath_blas_{libname})
set(LIB_OBJ ${{LIB_NAME}}_obj)

# Add third-party library
# find_package(XXX REQUIRED)

add_library(${{LIB_NAME}})
add_library(${{LIB_OBJ}} OBJECT
""".format(libname=libname))

for f in file_list:
    if re.search('_dyn.c', f):
        out_file.write("""  $<$<BOOL:${{ONEMATH_BUILD_SHARED_LIBS}}>: {filename}>
""".format(filename=f))
    else:
        out_file.write("""  {filename}
""".format(filename=f))

out_file.write("""
)

target_include_directories(${{LIB_OBJ}}
  PRIVATE ${{PROJECT_SOURCE_DIR}}/include
          ${{PROJECT_SOURCE_DIR}}/src
)

target_link_libraries(${{LIB_OBJ}}
    PUBLIC ONEMATH::SYCL::SYCL
    # Add third party library to link with here
)

target_compile_features(${{LIB_OBJ}} PUBLIC cxx_std_14)
set_target_properties(${{LIB_OBJ}} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${{LIB_NAME}} PUBLIC ${{LIB_OBJ}})

# Add major version to the library
set_target_properties(${{LIB_NAME}} PROPERTIES
  SOVERSION ${{PROJECT_VERSION_MAJOR}}
)

# Add dependencies rpath to the library
list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${{LIB_NAME}}>)

# Add the library to install package
install(TARGETS ${{LIB_OBJ}} EXPORT oneMathTargets)
install(TARGETS ${{LIB_NAME}} EXPORT oneMathTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
""".format())

out_file.close()
