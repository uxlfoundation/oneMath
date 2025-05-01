#===============================================================================
# Copyright 2025 SiPearl
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

include_guard()
set(ARMPL_SEQ armpl_int64)
set(ARMPL_OMP armpl_int64_mp)

include(FindPackageHandleStandardArgs)
if(ENABLE_ARMPL_OMP)
  message(STATUS "Use OpenMP version of ArmPL")
  set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
  find_library(ARMPL_LIBRARY NAMES ${ARMPL_OMP} HINTS ${ARMPL_ROOT} $ENV{ARMPLROOT} PATH_SUFFIXES lib lib64)
else()
  message(STATUS "Use Sequential version of ArmPL")
  find_library(ARMPL_LIBRARY NAMES ${ARMPL_SEQ} HINTS ${ARMPL_ROOT} $ENV{ARMPLROOT} PATH_SUFFIXES lib lib64)
endif()
find_package_handle_standard_args(ARMPL REQUIRED_VARS ARMPL_LIBRARY)

get_filename_component(ARMPL_LIB_DIR ${ARMPL_LIBRARY} DIRECTORY)
find_path(ARMPL_INCLUDE armpl.h HINTS ${ARMPL_ROOT} $ENV{ARMPLROOT} PATH_SUFFIXES include)
#cmake replaces fullpath to libarmpl by -larmpl (because SONAME is absent) and -Wl,-rpath is not enough for some compilers as hint
#so we need to add -L to compiler, otherwise we need to set LIBRARY_PATH manually when building
if(UNIX)
  list(APPEND ARMPL_LINK "-Wl,-rpath,${ARMPL_LIB_DIR} -L${ARMPL_LIB_DIR}")
endif()
list(APPEND ARMPL_LINK ${ARMPL_LIBRARY})
list(APPEND ARMPL_LINK ${ARMPL_LIBRARY})
message(${ARMPL_LINK})
find_package_handle_standard_args(ARMPL REQUIRED_VARS ARMPL_INCLUDE ARMPL_LINK)

# Check ARMPL version (only versions higher or equal to 22.0.1 are supported)
set(ARMPL_MAJOR 22)
set(ARMPL_MINOR 0)
set(ARMPL_BUILD 1)
file(WRITE ${CMAKE_BINARY_DIR}/armplversion.cpp
"#include <stdio.h>\n"
"\n"
"#include \"armpl.h\"\n"
"\n"
"int main(void) {\n"
"  int major, minor, build;\n"
"  char *tag;\n"
"  armplversion(&major, &minor, &build, (const char **)&tag);\n"
"  if (major > MAJOR) {\n"
"    return 0;\n"
"  }\n"
"  else if (major == MAJOR && minor > MINOR) {\n"
"    return 0;\n"
"  }\n"
"  else if (major == MAJOR && minor == MINOR && build >= BUILD) {\n"
"    return 0;\n"
"  }\n"
"  printf(\"You are using version %d.%d.%d\\n\", major, minor, build);\n"
"  return 1;\n"
"}\n")
execute_process(COMMAND ${CMAKE_CXX_COMPILER} armplversion.cpp -O0 -I${ARMPL_INCLUDE} -Wl,-rpath,${ARMPL_LIB_DIR} -larmpl -DMAJOR=${ARMPL_MAJOR} -DMINOR=${ARMPL_MINOR} -DBUILD=${ARMPL_BUILD} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
execute_process(COMMAND ./a.out WORKING_DIRECTORY ${CMAKE_BINARY_DIR} RESULT_VARIABLE ARMPL_CHECK_VERSION)
execute_process(COMMAND rm ./a.out WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
execute_process(COMMAND rm armplversion.cpp WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
if(ARMPL_CHECK_VERSION)
  message(FATAL_ERROR "ARMPL backend does not support ARMPL version prior to version ${ARMPL_MAJOR}.${ARMPL_MINOR}.${ARMPL_BUILD}")
endif()

add_library(ONEMKL::ARMPL::ARMPL UNKNOWN IMPORTED)
set_target_properties(ONEMKL::ARMPL::ARMPL PROPERTIES IMPORTED_LOCATION ${ARMPL_LIBRARY})
