#===============================================================================
# Copyright 2023 Intel Corporation
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

#===================================================================
# CMake Config file for Intel(R) oneAPI Math Kernel Library (oneMKL)
#===================================================================

#===============================================================================
# Input parameters
#=================
#-------------
# Main options
#-------------
# MKL_ROOT: oneMKL root directory (May be required for non-standard install locations. Optional otherwise.)
#    Default: use location from MKLROOT environment variable or <Full path to this file>/../../../ if MKLROOT is not defined
# MKL_ARCH
#    Values:  ia32 [ia32 support is deprecated]
#             intel64
#    Default: intel64
# MKL_LINK
#    Values:  static, dynamic, sdl
#    Default: dynamic
#       Exceptions:- SYCL doesn't support sdl
# MKL_THREADING
#    Values:  sequential,
#             intel_thread (Intel OpenMP),
#             gnu_thread (GNU OpenMP),
#             pgi_thread (PGI OpenMP) [PGI support is deprecated],
#             tbb_thread
#    Default: intel_thread
#       Exceptions:- SYCL defaults to oneTBB, PGI compiler on Windows defaults to pgi_thread
# MKL_INTERFACE (for MKL_ARCH=intel64 only)
#    Values:  lp64, ilp64
#       GNU or INTEL interface will be selected based on Compiler.
#    Default: ilp64
# MKL_MPI
#    Values:  intelmpi, mpich, openmpi, msmpi, mshpc
#    Default: intelmpi
#-----------------------------------
# Special options (OFF by default)
#-----------------------------------
# ENABLE_BLAS95:           Enables BLAS Fortran95 API in MKL::MKL
# ENABLE_LAPACK95:         Enables LAPACK Fortran95 API in MKL::MKL
# ENABLE_BLACS:            Enables cluster BLAS library in MKL::MKL
# ENABLE_CDFT:             Enables cluster DFT library in MKL::MKL
# ENABLE_SCALAPACK:        Enables cluster LAPACK library in MKL::MKL
# ENABLE_OMP_OFFLOAD:      Enables OpenMP Offload functionality in MKL::MKL
# ENABLE_TRY_SYCL_COMPILE: Enables compiling a test program that calls a oneMKL DPC++ API
#
#==================
# Output parameters
#==================
# MKL_ROOT
#     oneMKL root directory.
# MKL_INCLUDE
#     Use of target_include_directories() is recommended.
#     INTERFACE_INCLUDE_DIRECTORIES property is set on mkl_core and mkl_rt libraries.
#     Alternatively, this variable can be used directly (not recommended as per Modern CMake)
# MKL_ENV
#     Provides all environment variables based on input parameters.
#     Currently useful for mkl_rt linking and BLACS on Windows.
#     Must be set as an ENVIRONMENT property.
#     Example:
#     add_test(NAME mytest COMMAND myexe)
#     if(MKL_ENV)
#       set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
#     endif()
#
# MKL::<library name>
#     IMPORTED targets to link oneMKL libraries individually or when using a custom link-line.
#     mkl_core and mkl_rt have INTERFACE_* properties set to them.
#     Please refer to Intel(R) oneMKL Link Line Advisor for help with linking.
#
# Below INTERFACE targets provide full link-lines for direct use.
# Example:
#     target_link_options(<my_linkable_target> PUBLIC MKL::MKL)
#
# MKL::MKL
#     Link line for C and Fortran API
# MKL::MKL_SYCL
#     Link line for SYCL API
# MKL::MKL_SYCL::<domain>
#     Link line for specific domain SYCL API
#     Where <domain> could be: BLAS, LAPACK, DFT, SPARSE, RNG, STATS, VM, DATA_FITTING (experimental)
# MKL::MKL_CDFT
#     Link line for CDFT and Cluster FFTW API (includes MKL::MKL and MKL::MKL_BLACS)
#     !IMPORTANT!: Because of specific link order it must not be used together
#     with any other oneMKL targets in case of MKL_LINK=static on Linux
# MKL::MKL_SCALAPACK
#     Link line for ScaLAPACK and PBLAS API (includes MKL::MKL and MKL::MKL_BLACS)
# MKL::MKL_BLACS
#     Link line for BLACS and CPARDISO API (includes MKL::MKL)
#
# Note: For Device API, library linking is not required.
#       Compile options can be added from the INTERFACE_COMPILE_OPTIONS property on MKL::MKL_SYCL
#       Include directories can be added from the INTERFACE_INCLUDE_DIRECTORIES property on MKL::MKL_SYCL
#
# Note: Output parameters' and targets' availability can change
# based on Input parameters and application project languages.
#===============================================================================

include_guard()

if(NOT TARGET MKL::MKL)

function(mkl_message MSG_MODE MSG_TEXT)
  if(MSG_MODE STREQUAL "FATAL_ERROR")
    message(${MSG_MODE} ${MSG_TEXT})
  else()
    if(NOT MKL_FIND_QUIETLY)
      message(${MSG_MODE} ${MSG_TEXT})
    endif()
  endif()
endfunction()

macro(mkl_not_found_and_return NOT_FOUND_MSG)
  set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${NOT_FOUND_MSG}")
  set(${CMAKE_FIND_PACKAGE_NAME}_FOUND FALSE)
  return()
endmacro()

if(CMAKE_VERSION VERSION_LESS "3.13")
  mkl_not_found_and_return("The minimum supported CMake version is 3.13. You are running version ${CMAKE_VERSION}.")
endif()

# Set CMake policies for well-defined behavior across CMake versions
cmake_policy(SET CMP0011 NEW)
cmake_policy(SET CMP0057 NEW)

# Project Languages
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
list(APPEND MKL_LANGS C CXX Fortran)
foreach(lang ${languages})
  if(${lang} IN_LIST MKL_LANGS)
    list(APPEND CURR_LANGS ${lang})
  endif()
endforeach()
list(REMOVE_DUPLICATES CURR_LANGS)

option(ENABLE_BLAS95           "Enables BLAS Fortran95 API"                    OFF)
option(ENABLE_LAPACK95         "Enables LAPACK Fortran95 API"                  OFF)
option(ENABLE_BLACS            "Enables cluster BLAS library"                  OFF)
option(ENABLE_CDFT             "Enables cluster DFT library"                   OFF)
option(ENABLE_SCALAPACK        "Enables cluster LAPACK library"                OFF)
option(ENABLE_OMP_OFFLOAD      "Enables OpenMP Offload functionality"          OFF)
option(ENABLE_TRY_SYCL_COMPILE "Enables compiling a oneMKL DPC++ test program" OFF)

# Use MPI if any of these are enabled
if(ENABLE_BLACS OR ENABLE_CDFT OR ENABLE_SCALAPACK)
  set(USE_MPI ON)
endif()

# Check Parameters
function(define_param TARGET_PARAM DEFAULT_PARAM SUPPORTED_LIST)
  if(NOT DEFINED ${TARGET_PARAM} AND NOT DEFINED ${DEFAULT_PARAM})
    mkl_message(STATUS "${TARGET_PARAM}: Undefined")
  elseif(NOT DEFINED ${TARGET_PARAM} AND DEFINED ${DEFAULT_PARAM})
    set(${TARGET_PARAM} "${${DEFAULT_PARAM}}" CACHE STRING "Choose ${TARGET_PARAM} options are: ${${SUPPORTED_LIST}}")
    foreach(opt ${${DEFAULT_PARAM}})
      set(STR_LIST "${STR_LIST} ${opt}")
    endforeach()
    mkl_message(STATUS "${TARGET_PARAM}: None, set to `${STR_LIST}` by default")
  elseif(${SUPPORTED_LIST})
    set(ITEM_FOUND 1)
    foreach(opt ${${TARGET_PARAM}})
      if(NOT ${opt} IN_LIST ${SUPPORTED_LIST})
        set(ITEM_FOUND 0)
      endif()
    endforeach()
    if(ITEM_FOUND EQUAL 0)
      foreach(opt ${${SUPPORTED_LIST}})
        set(STR_LIST "${STR_LIST} ${opt}")
      endforeach()
      if(${ARGC} EQUAL 3)
        mkl_message(WARNING "Invalid ${TARGET_PARAM} `${${TARGET_PARAM}}`, options are: ${STR_LIST}")
        set(${TARGET_PARAM} "${${TARGET_PARAM}}_MKL_INVALID_PARAM" PARENT_SCOPE)
      elseif(${ARGC} EQUAL 4)
        mkl_message(${ARGV3} "Invalid ${TARGET_PARAM} `${${TARGET_PARAM}}`, options are: ${STR_LIST}")
        set(${TARGET_PARAM} "" PARENT_SCOPE)
      endif()
    else()
      mkl_message(STATUS "${TARGET_PARAM}: ${${TARGET_PARAM}}")
    endif()
  else()
    mkl_message(STATUS "${TARGET_PARAM}: ${${TARGET_PARAM}}")
  endif()
endfunction()

macro(check_required_vars)
  foreach(var IN ITEMS ${ARGV})
    if(NOT ${var})
      mkl_not_found_and_return("The required variable ${var} has an invalid value \"${${var}}\".")
    elseif(${${var}} MATCHES "_MKL_INVALID_PARAM$")
      string(REPLACE "_MKL_INVALID_PARAM" "" INVALID_PARAM ${${var}})
      mkl_not_found_and_return("The required variable ${var} has an invalid value \"${INVALID_PARAM}\".")
    endif()
  endforeach()
endmacro()

#================
# Compiler checks
#================

if(CMAKE_C_COMPILER)
  get_filename_component(C_COMPILER_NAME ${CMAKE_C_COMPILER} NAME)
endif()
if(CMAKE_CXX_COMPILER)
  get_filename_component(CXX_COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
endif()
if(CMAKE_Fortran_COMPILER)
  get_filename_component(Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)
endif()

# Determine Compiler Family
if(CXX_COMPILER_NAME STREQUAL "dpcpp" OR CXX_COMPILER_NAME STREQUAL "dpcpp.exe"
    OR CXX_COMPILER_NAME STREQUAL "icpx" OR CXX_COMPILER_NAME STREQUAL "icx.exe")
  set(SYCL_COMPILER ON)
endif()
if(C_COMPILER_NAME MATCHES "^clang" OR CXX_COMPILER_NAME MATCHES "^clang")
  set(CLANG_COMPILER ON)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "PGI" OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI" OR CMAKE_Fortran_COMPILER_ID STREQUAL "PGI"
    OR CMAKE_C_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"
    OR CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC") # PGI 22.9
  mkl_message(WARNING "PGI support is deprecated and will be removed in the oneMKL 2025.0 release.")
  set(PGI_COMPILER ON)
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel"
        OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(INTEL_COMPILER ON)
else()
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GNU_C_COMPILER ON)
  endif()
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(GNU_Fortran_COMPILER ON)
  endif()
endif()
# CMake identifies IntelLLVM compilers only after 3.20
if(NOT INTEL_COMPILER)
  if(C_COMPILER_NAME STREQUAL "icx" OR C_COMPILER_NAME STREQUAL "icx.exe"
      OR CXX_COMPILER_NAME STREQUAL "icpx" OR CXX_COMPILER_NAME STREQUAL "icx.exe"
      OR Fortran_COMPILER_NAME STREQUAL "ifx" OR Fortran_COMPILER_NAME STREQUAL "ifx.exe")
    set(INTEL_COMPILER ON)
  endif()
endif()
# CMake supports IntelLLVM compilers only after 3.25.2
if(CMAKE_VERSION VERSION_LESS "3.25.2")
  if(C_COMPILER_NAME STREQUAL "icx" OR C_COMPILER_NAME STREQUAL "icx.exe" OR CXX_COMPILER_NAME STREQUAL "icx.exe")
    list(APPEND INTEL_LLVM_COMPILERS_IN_USE "icx")
  endif()
  if(CXX_COMPILER_NAME STREQUAL "icpx")
    list(APPEND INTEL_LLVM_COMPILERS_IN_USE "icpx")
  endif()
  if(Fortran_COMPILER_NAME STREQUAL "ifx" OR Fortran_COMPILER_NAME STREQUAL "ifx.exe")
    list(APPEND INTEL_LLVM_COMPILERS_IN_USE "ifx")
  endif()
  if(INTEL_LLVM_COMPILERS_IN_USE)
    list(JOIN INTEL_LLVM_COMPILERS_IN_USE ", " INTEL_LLVM_COMPILERS_IN_USE_COMMA)
    mkl_message(STATUS "Upgrade to CMake version 3.25.2 or later for native support of Intel compiler(s) ${INTEL_LLVM_COMPILERS_IN_USE_COMMA}. You are running version ${CMAKE_VERSION}.")
  endif()
endif()

if(USE_MPI AND (C_COMPILER_NAME MATCHES "^mpi" OR Fortran_COMPILER_NAME MATCHES "^mpi"))
  set(USE_MPI_SCRIPT ON)
endif()

#================

#================
# System-specific
#================

# Extensions
set(SO_VER "2")
set(SYCL_SO_VER "4")
if(UNIX)
  set(LIB_PREFIX "lib")
  set(LIB_EXT ".a")
  set(DLL_EXT ".so")
  if(APPLE)
    set(DLL_EXT ".dylib")
  endif()
  set(LINK_PREFIX "-l")
  set(LINK_SUFFIX "")
else()
  set(LIB_PREFIX "")
  set(LIB_EXT ".lib")
  set(DLL_EXT "_dll.lib")
  set(LINK_PREFIX "")
  set(LINK_SUFFIX ".lib")
endif()

#================

#=============
# Setup oneMKL
#=============

# Set MKL_ROOT directory
if(NOT DEFINED MKL_ROOT)
  if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT})
    # Verify that the version in MKL_ROOT is the same as MKL_VERSION
    find_file(MKL_VERSION_H mkl_version.h
      HINTS ${MKL_ROOT}
      PATH_SUFFIXES include
      NO_DEFAULT_PATH)
    check_required_vars(MKL_VERSION_H)
    file(READ ${MKL_VERSION_H} MKL_VERSION_H_CONTENT)
    string(REGEX MATCH "__INTEL_MKL__ +([0-9]+)" MKL_VERSION_INFO ${MKL_VERSION_H_CONTENT})
    set(MKL_ROOT_MAJOR_VERSION ${CMAKE_MATCH_1})
    string(REGEX MATCH "__INTEL_MKL_UPDATE__ +([0-9]+)" MKL_VERSION_INFO ${MKL_VERSION_H_CONTENT})
    set(MKL_ROOT_UPDATE_VERSION ${CMAKE_MATCH_1})
    set(MKL_ROOT_VERSION ${MKL_ROOT_MAJOR_VERSION}.${MKL_ROOT_UPDATE_VERSION})
    if(NOT MKL_ROOT_VERSION VERSION_EQUAL ${CMAKE_FIND_PACKAGE_NAME}_VERSION)
      mkl_not_found_and_return("oneMKL ${MKL_ROOT_VERSION} specified by the environment variable MKLROOT \
                                mismatches the found version ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} \
                                indicated by ${CMAKE_CURRENT_LIST_DIR}/MKLConfigVersion.cmake")
    endif()
  else()
    get_filename_component(MKL_CMAKE_PATH "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
    get_filename_component(MKL_ROOT "${MKL_CMAKE_PATH}/../../../" ABSOLUTE)
  endif()
endif()
string(REPLACE "\\" "/" MKL_ROOT ${MKL_ROOT})
check_required_vars(MKL_ROOT)
mkl_message(STATUS "${CMAKE_FIND_PACKAGE_NAME}_VERSION: ${${CMAKE_FIND_PACKAGE_NAME}_VERSION}")
mkl_message(STATUS "MKL_ROOT: ${MKL_ROOT}")

# Set target system architecture
if(SYCL_COMPILER)
  set(DEFAULT_MKL_SYCL_ARCH intel64)
  set(MKL_SYCL_ARCH_LIST intel64)
  if(NOT DEFINED MKL_SYCL_ARCH)
    set(MKL_SYCL_ARCH ${MKL_ARCH})
  endif()
  define_param(MKL_SYCL_ARCH DEFAULT_MKL_SYCL_ARCH MKL_SYCL_ARCH_LIST STATUS)
  if(NOT MKL_SYCL_ARCH)
    set(SYCL_COMPILER OFF)
    mkl_message(STATUS "MKL::MKL_SYCL target will not be available.")
  endif()
endif()
set(DEFAULT_MKL_ARCH intel64)
if(PGI_COMPILER OR ENABLE_OMP_OFFLOAD OR USE_MPI)
  set(MKL_ARCH_LIST intel64)
else()
  set(MKL_ARCH_LIST ia32 intel64)
endif()
define_param(MKL_ARCH DEFAULT_MKL_ARCH MKL_ARCH_LIST)
check_required_vars(MKL_ARCH)
if(MKL_ARCH STREQUAL "ia32")
  set(MKL_ARCH_DIR "32")
  mkl_message(WARNING "ia32 support is deprecated and will be removed in the oneMKL 2025.0 release.")
else()
  set(MKL_ARCH_DIR "")
endif()

# Define MKL_LINK
if(SYCL_COMPILER)
  set(DEFAULT_MKL_SYCL_LINK dynamic)
  set(MKL_SYCL_LINK_LIST static dynamic)
  if(NOT DEFINED MKL_SYCL_LINK)
    set(MKL_SYCL_LINK ${MKL_LINK})
  endif()
  define_param(MKL_SYCL_LINK DEFAULT_MKL_SYCL_LINK MKL_SYCL_LINK_LIST STATUS)
  if(NOT MKL_SYCL_LINK)
    set(SYCL_COMPILER OFF)
    mkl_message(STATUS "MKL::MKL_SYCL target will not be available.")
  endif()
endif()
set(DEFAULT_MKL_LINK dynamic)
if(USE_MPI)
  set(MKL_LINK_LIST static dynamic)
else()
  set(MKL_LINK_LIST static dynamic sdl)
endif()
define_param(MKL_LINK DEFAULT_MKL_LINK MKL_LINK_LIST)
check_required_vars(MKL_LINK)

# Define MKL_INTERFACE
if(SYCL_COMPILER)
  if(MKL_INTERFACE AND NOT DEFINED MKL_SYCL_INTERFACE_FULL)
    set(MKL_SYCL_INTERFACE_FULL intel_${MKL_INTERFACE})
  endif()
  set(DEFAULT_MKL_SYCL_INTERFACE intel_ilp64)
  set(MKL_SYCL_INTERFACE_LIST intel_lp64 intel_ilp64)
  define_param(MKL_SYCL_INTERFACE_FULL DEFAULT_MKL_SYCL_INTERFACE MKL_SYCL_INTERFACE_LIST STATUS)
  if(NOT MKL_SYCL_INTERFACE_FULL)
    set(SYCL_COMPILER OFF)
    mkl_message(STATUS "MKL::MKL_SYCL target will not be available.")
  endif()
endif()
if(MKL_ARCH STREQUAL "intel64")
  set(IFACE_TYPE intel)
  if(GNU_Fortran_COMPILER)
    set(IFACE_TYPE gf)
  endif()
  if(MKL_INTERFACE)
    set(MKL_INTERFACE_FULL ${IFACE_TYPE}_${MKL_INTERFACE})
  endif()
  set(DEFAULT_MKL_INTERFACE ${IFACE_TYPE}_ilp64)
  set(MKL_INTERFACE_LIST ${IFACE_TYPE}_ilp64 ${IFACE_TYPE}_lp64)
  define_param(MKL_INTERFACE_FULL DEFAULT_MKL_INTERFACE MKL_INTERFACE_LIST)
else()
  if(WIN32)
    set(MKL_INTERFACE_FULL intel_c)
  elseif(NOT APPLE)
    if(GNU_Fortran_COMPILER)
      set(MKL_INTERFACE_FULL gf)
    else()
      set(MKL_INTERFACE_FULL intel)
    endif()
  else()
    mkl_not_found_and_return("OSX does not support MKL_ARCH ia32.")
  endif()
endif()
check_required_vars(MKL_INTERFACE_FULL)
if(MKL_INTERFACE_FULL MATCHES "ilp64")
  set(MKL_INTERFACE "ilp64")
else()
  set(MKL_INTERFACE "lp64")
endif()

# Define oneMKL headers
find_path(MKL_INCLUDE mkl.h
  HINTS ${MKL_ROOT}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
check_required_vars(MKL_INCLUDE)

# Add pre-built F95 Interface Modules
if(INTEL_COMPILER AND (ENABLE_BLAS95 OR ENABLE_LAPACK95))
  if(MKL_ARCH STREQUAL "intel64")
    list(APPEND MKL_INCLUDE "${MKL_ROOT}/include/mkl/${MKL_ARCH}/${MKL_INTERFACE}")
  else()
    list(APPEND MKL_INCLUDE "${MKL_ROOT}/include/mkl/${MKL_ARCH}")
  endif()
endif()

# Define MKL_THREADING
# All APIs support sequential threading
# SYCL API supports oneTBB and OpenMP threadings, but OpenMP threading might have composability problem on CPU device with other SYCL kernels
if(SYCL_COMPILER)
  set(MKL_SYCL_THREADING_LIST "sequential" "intel_thread" "tbb_thread")
  set(DEFAULT_MKL_SYCL_THREADING tbb_thread)
  if(NOT DEFINED MKL_SYCL_THREADING)
    set(MKL_SYCL_THREADING ${MKL_THREADING})
  endif()
  define_param(MKL_SYCL_THREADING DEFAULT_MKL_SYCL_THREADING MKL_SYCL_THREADING_LIST STATUS)
  if(NOT MKL_SYCL_THREADING)
    set(SYCL_COMPILER OFF)
    mkl_message(STATUS "MKL::MKL_SYCL target will not be available.")
  endif()
  if(MKL_SYCL_THREADING STREQUAL "intel_thread")
    mkl_message(STATUS "Using MKL::MKL_SYCL* targets with intel_thread may have potential composability problems on CPU device with other SYCL kernels.")
    add_custom_target(MKL_SYCL_MESSAGE
                      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red
                      "Warning: Using MKL::MKL_SYCL* targets with intel_thread may have potential composability problems on CPU device with other SYCL kernels.")
  endif()
endif()
# C, Fortran API
set(MKL_THREADING_LIST "sequential" "intel_thread" "tbb_thread")
set(DEFAULT_MKL_THREADING intel_thread)
if(PGI_COMPILER)
  # PGI compiler supports PGI OpenMP threading, additionally
  list(APPEND MKL_THREADING_LIST pgi_thread)
  # PGI compiler does not support oneTBB threading
  list(REMOVE_ITEM MKL_THREADING_LIST tbb_thread)
  if(WIN32)
    # PGI 19.10 and 20.1 on Windows, do not support Intel OpenMP threading
    list(REMOVE_ITEM MKL_THREADING_LIST intel_thread)
    set(DEFAULT_MKL_THREADING pgi_thread)
  endif()
elseif(GNU_C_COMPILER OR GNU_Fortran_COMPILER OR CLANG_COMPILER)
  list(APPEND MKL_THREADING_LIST gnu_thread)
else()
  # Intel and Microsoft compilers
  # Nothing to do, only for completeness
endif()
define_param(MKL_THREADING DEFAULT_MKL_THREADING MKL_THREADING_LIST)
check_required_vars(MKL_THREADING)

# Define MKL_MPI
if(NOT MKL_LINK STREQUAL "sdl" AND NOT MKL_ARCH STREQUAL "ia32")
  set(DEFAULT_MKL_MPI intelmpi)
  if(UNIX)
    if(APPLE)
      # Override defaults for OSX
      set(DEFAULT_MKL_MPI mpich)
      set(MKL_MPI_LIST mpich)
    else()
      set(MKL_MPI_LIST intelmpi openmpi mpich mpich2)
    endif()
  else()
    # Windows
    set(MKL_MPI_LIST intelmpi mshpc msmpi)
  endif()
  define_param(MKL_MPI DEFAULT_MKL_MPI MKL_MPI_LIST)
  # MSMPI is now called MSHPC. MSMPI option exists for backward compatibility.
  if(MKL_MPI STREQUAL "mshpc")
    set(MKL_MPI msmpi)
  endif()
  check_required_vars(MKL_MPI)
else()
  mkl_message(STATUS "MKL_MPI: Selected configuration is not supported by oneMKL cluster components")
endif()

# Provides a list of IMPORTED targets for the project
if(NOT DEFINED MKL_IMPORTED_TARGETS)
  set(MKL_IMPORTED_TARGETS "")
endif()

# Clear temporary variables
set(MKL_C_COPT "")
set(MKL_F_COPT "")
set(MKL_SDL_COPT "")
set(MKL_CXX_COPT "")
set(MKL_SYCL_COPT "")
set(MKL_SYCL_LOPT "")
set(MKL_OFFLOAD_COPT "")
set(MKL_OFFLOAD_LOPT "")

set(MKL_SUPP_LINK "")        # Other link options. Usually at the end of the link-line.
set(MKL_SYCL_SUPP_LINK "")
set(MKL_LINK_LINE "")
set(MKL_SYCL_LINK_LINE "")
set(MKL_ENV_PATH "")         # Temporary variable to work with PATH
set(MKL_ENV "")              # Exported environment variables

# Modify PATH variable to make it CMake-friendly
set(OLD_PATH $ENV{PATH})
string(REPLACE ";" "\;" OLD_PATH "${OLD_PATH}")
# Modify LIBRARY_PATH variable to make it CMake-friendly
set(ENV_LIBRARY_PATH $ENV{LIBRARY_PATH})
string(REPLACE ":" ";" ENV_LIBRARY_PATH "${ENV_LIBRARY_PATH}")

# Compiler options
if(GNU_C_COMPILER OR GNU_Fortran_COMPILER)
  if(MKL_ARCH STREQUAL "ia32")
    list(APPEND MKL_C_COPT   -m32)
    list(APPEND MKL_CXX_COPT -m32)
    list(APPEND MKL_F_COPT   -m32)
  else()
    list(APPEND MKL_C_COPT   -m64)
    list(APPEND MKL_CXX_COPT -m64)
    list(APPEND MKL_F_COPT   -m64)
  endif()
endif()

# Additonal compiler & linker options
if(SYCL_COMPILER)
  list(APPEND MKL_SYCL_COPT "-fsycl")
  list(APPEND MKL_SYCL_LOPT "-fsycl")
  if(MKL_SYCL_LINK STREQUAL "static")
    list(APPEND MKL_SYCL_LOPT "-fsycl-device-code-split=per_kernel")
  endif()
endif()
if(ENABLE_OMP_OFFLOAD)
  if(MKL_LINK STREQUAL "static")
    list(APPEND MKL_OFFLOAD_LOPT "-fsycl-device-code-split=per_kernel")
  endif()
endif()

# For OpenMP Offload
if(ENABLE_OMP_OFFLOAD)
  if(WIN32)
    if("Fortran" IN_LIST CURR_LANGS)
      list(APPEND MKL_OFFLOAD_COPT -Qiopenmp -Qopenmp-targets:spir64)
    else()
      list(APPEND MKL_OFFLOAD_COPT -Qiopenmp -Qopenmp-targets:spir64 -Qopenmp-version:51)
    endif()
    # -MD and -MDd are manually added here because offload functionality uses SYCL runtime.
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      list(APPEND MKL_OFFLOAD_COPT -MDd)
    else()
      list(APPEND MKL_OFFLOAD_COPT -MD)
    endif()
    list(APPEND MKL_OFFLOAD_LOPT -Qiopenmp -Qopenmp-targets:spir64 -fsycl)
    set(SKIP_LIBPATH ON)
  else()
    if("Fortran" IN_LIST CURR_LANGS)
      list(APPEND MKL_OFFLOAD_COPT -fiopenmp -fopenmp-targets=spir64)
    else()
      list(APPEND MKL_OFFLOAD_COPT -fiopenmp -fopenmp-targets=spir64 -fopenmp-version=51)
    endif()
    list(APPEND MKL_OFFLOAD_LOPT -fiopenmp -fopenmp-targets=spir64 -fsycl)
    if(APPLE)
      list(APPEND MKL_SUPP_LINK -lc++)
    else()
      list(APPEND MKL_SUPP_LINK -lstdc++)
    endif()
  endif()
endif()

# For selected Interface
if(SYCL_COMPILER)
  if(MKL_INTERFACE STREQUAL "ilp64")
    list(INSERT MKL_SYCL_COPT 0 "-DMKL_ILP64")
  else()
    mkl_message(STATUS "Experimental oneMKL Data Fitting SYCL API does not support LP64 on CPU")
  endif()
endif()

if(MKL_INTERFACE_FULL)
  if(MKL_ARCH STREQUAL "ia32")
    if(GNU_Fortran_COMPILER)
      set(MKL_SDL_IFACE_ENV "GNU")
    endif()
  else()
    if(GNU_Fortran_COMPILER)
      set(MKL_SDL_IFACE_ENV "GNU,${MKL_INTERFACE}")
    else()
      set(MKL_SDL_IFACE_ENV "${MKL_INTERFACE}")
    endif()
    if(MKL_INTERFACE STREQUAL "ilp64")
      if("Fortran" IN_LIST CURR_LANGS)
        if(INTEL_COMPILER)
          if(WIN32)
            list(APPEND MKL_F_COPT "-4I8")
          else()
            list(APPEND MKL_F_COPT "-i8")
          endif()
        elseif(GNU_Fortran_COMPILER)
          list(APPEND MKL_F_COPT "-fdefault-integer-8")
        elseif(PGI_COMPILER)
          list(APPEND MKL_F_COPT "-i8")
        endif()
      endif()
      list(INSERT MKL_C_COPT 0 "-DMKL_ILP64")
      list(INSERT MKL_SDL_COPT 0 "-DMKL_ILP64")
      list(INSERT MKL_CXX_COPT 0 "-DMKL_ILP64")
      list(INSERT MKL_OFFLOAD_COPT 0 "-DMKL_ILP64")
    else()
      # lp64
    endif()
  endif()
  if(MKL_SDL_IFACE_ENV)
    string(TOUPPER ${MKL_SDL_IFACE_ENV} MKL_SDL_IFACE_ENV)
  endif()
endif() # MKL_INTERFACE_FULL

# All oneMKL Libraries
if(SYCL_COMPILER)
  set(MKL_SYCL_IFACE_LIB mkl_${MKL_SYCL_INTERFACE_FULL})
  if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug" AND MKL_SYCL_THREADING STREQUAL "tbb_thread")
    set(MKL_SYCL_THREAD mkl_tbb_threadd)
  else()
    set(MKL_SYCL_THREAD mkl_${MKL_SYCL_THREADING})
  endif()
endif()
set(MKL_SYCL)
set(MKL_SYCL_LIBS)
list(APPEND MKL_SYCL_LIBS mkl_sycl_blas)
list(APPEND MKL_SYCL_LIBS mkl_sycl_lapack)
list(APPEND MKL_SYCL_LIBS mkl_sycl_dft)
list(APPEND MKL_SYCL_LIBS mkl_sycl_sparse)
list(APPEND MKL_SYCL_LIBS mkl_sycl_data_fitting)
list(APPEND MKL_SYCL_LIBS mkl_sycl_rng)
list(APPEND MKL_SYCL_LIBS mkl_sycl_stats)
list(APPEND MKL_SYCL_LIBS mkl_sycl_vm)
if(NOT MKL_LINK STREQUAL "static")
  if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
    list(TRANSFORM MKL_SYCL_LIBS APPEND "d")
  endif()
  list(APPEND MKL_SYCL ${MKL_SYCL_LIBS})
  # List for tracking incomplete onemKL package
  set(MISSED_MKL_SYCL_LIBS)
else()
  if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
    set(MKL_SYCL         mkl_sycld)
  else()
    set(MKL_SYCL         mkl_sycl)
  endif()
endif()

set(MKL_IFACE_LIB     mkl_${MKL_INTERFACE_FULL})
set(MKL_CORE          mkl_core)
if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug" AND MKL_THREADING STREQUAL "tbb_thread")
  set(MKL_THREAD        mkl_tbb_threadd)
else()
  set(MKL_THREAD        mkl_${MKL_THREADING})
endif()
set(MKL_SDL           mkl_rt)
if(MKL_ARCH STREQUAL "ia32")
  set(MKL_BLAS95      mkl_blas95)
  set(MKL_LAPACK95    mkl_lapack95)
else()
  set(MKL_BLAS95      mkl_blas95_${MKL_INTERFACE})
  set(MKL_LAPACK95    mkl_lapack95_${MKL_INTERFACE})
endif()
# BLACS
set(MKL_BLACS mkl_blacs_${MKL_MPI}_${MKL_INTERFACE})
if(UNIX AND NOT APPLE AND MKL_MPI MATCHES "mpich")
  # MPICH is compatible with INTELMPI Wrappers on Linux
  set(MKL_BLACS mkl_blacs_intelmpi_${MKL_INTERFACE})
endif()
if(WIN32)
  if(MKL_MPI STREQUAL "msmpi")
    if("Fortran" IN_LIST CURR_LANGS)
      list(APPEND MKL_SUPP_LINK "msmpifec.lib")
    endif()
    # MSMPI and MSHPC are supported with the same BLACS library
    set(MKL_BLACS mkl_blacs_msmpi_${MKL_INTERFACE})
    if(NOT MKL_LINK STREQUAL "static")
      set(MKL_BLACS mkl_blacs_${MKL_INTERFACE})
      set(MKL_BLACS_ENV MSMPI)
    endif()
  elseif(MKL_MPI STREQUAL "intelmpi" AND NOT MKL_LINK STREQUAL "static")
    set(MKL_BLACS mkl_blacs_${MKL_INTERFACE})
    set(MKL_BLACS_ENV INTELMPI)
  endif()
endif()
# CDFT & SCALAPACK
set(MKL_CDFT      mkl_cdft_core)
set(MKL_SCALAPACK mkl_scalapack_${MKL_INTERFACE})


if(UNIX AND NOT APPLE)
  if(MKL_LINK STREQUAL "static" OR MKL_SYCL_LINK STREQUAL "static")
    set(START_GROUP "-Wl,--start-group")
    set(END_GROUP "-Wl,--end-group")
    if(SYCL_COMPILER)
      set(SYCL_EXPORT_DYNAMIC "-Wl,-export-dynamic")
    endif()
    if(ENABLE_OMP_OFFLOAD)
      set(EXPORT_DYNAMIC "-Wl,-export-dynamic")
    endif()
  endif()
  if(MKL_LINK STREQUAL "dynamic")
    set(MKL_RPATH "-Wl,-rpath=$<TARGET_FILE_DIR:MKL::${MKL_CORE}>")
    if((GNU_Fortran_COMPILER OR PGI_COMPILER) AND "Fortran" IN_LIST CURR_LANGS)
      set(NO_AS_NEEDED -Wl,--no-as-needed)
    endif()
  endif()
  if(MKL_SYCL_LINK STREQUAL "dynamic")
    set(MKL_SYCL_RPATH "-Wl,-rpath=$<TARGET_FILE_DIR:MKL::${MKL_CORE}>")
  endif()
  if(MKL_LINK STREQUAL "sdl")
    set(MKL_RPATH "-Wl,-rpath=$<TARGET_FILE_DIR:MKL::${MKL_SDL}>")
  endif()
endif()

# Create a list of requested libraries, based on input options (MKL_LIBRARIES)
# Create full link-line in MKL_LINK_LINE
if(SYCL_COMPILER)
  list(APPEND MKL_SYCL_LIBRARIES ${MKL_SYCL} ${MKL_SYCL_IFACE_LIB} ${MKL_SYCL_THREAD} ${MKL_CORE})
  list(TRANSFORM MKL_SYCL PREPEND MKL:: OUTPUT_VARIABLE MKL_SYCL_T)
  list(APPEND MKL_SYCL_LINK_LINE ${MKL_SYCL_LOPT} ${SYCL_EXPORT_DYNAMIC} ${NO_AS_NEEDED} ${MKL_SYCL_RPATH}
       ${MKL_SYCL_T} ${START_GROUP} MKL::${MKL_SYCL_IFACE_LIB} MKL::${MKL_SYCL_THREAD} MKL::${MKL_CORE} ${END_GROUP})
endif()
list(APPEND MKL_LINK_LINE $<IF:$<BOOL:${ENABLE_OMP_OFFLOAD}>,${MKL_OFFLOAD_LOPT},>
     ${EXPORT_DYNAMIC} ${NO_AS_NEEDED} ${MKL_RPATH})
if(ENABLE_BLAS95)
  list(APPEND MKL_LIBRARIES ${MKL_BLAS95})
  list(APPEND MKL_LINK_LINE MKL::${MKL_BLAS95})
endif()
if(ENABLE_LAPACK95)
  list(APPEND MKL_LIBRARIES ${MKL_LAPACK95})
  list(APPEND MKL_LINK_LINE MKL::${MKL_LAPACK95})
endif()
if(NOT MKL_LINK STREQUAL "sdl" AND NOT MKL_ARCH STREQUAL "ia32")
  list(APPEND MKL_LIBRARIES ${MKL_SCALAPACK})
  if(ENABLE_SCALAPACK)
    list(APPEND MKL_LINK_LINE MKL::${MKL_SCALAPACK})
  endif()
endif()
if(ENABLE_OMP_OFFLOAD AND NOT MKL_LINK STREQUAL "sdl")
  list(APPEND MKL_LIBRARIES ${MKL_SYCL})
  list(TRANSFORM MKL_SYCL PREPEND MKL:: OUTPUT_VARIABLE MKL_SYCL_T)
  list(APPEND MKL_LINK_LINE ${MKL_SYCL_T})
endif()
list(APPEND MKL_LINK_LINE ${START_GROUP})
if(NOT MKL_LINK STREQUAL "sdl" AND NOT MKL_ARCH STREQUAL "ia32")
  list(APPEND MKL_LIBRARIES ${MKL_CDFT})
  if(ENABLE_CDFT)
    list(APPEND MKL_LINK_LINE MKL::${MKL_CDFT})
  endif()
endif()
if(MKL_LINK STREQUAL "sdl")
  list(APPEND MKL_LIBRARIES ${MKL_SDL})
  list(APPEND MKL_LINK_LINE MKL::${MKL_SDL})
else()
  list(APPEND MKL_LIBRARIES ${MKL_IFACE_LIB} ${MKL_THREAD} ${MKL_CORE})
  list(APPEND MKL_LINK_LINE MKL::${MKL_IFACE_LIB} MKL::${MKL_THREAD} MKL::${MKL_CORE})
endif()
if(NOT MKL_LINK STREQUAL "sdl" AND NOT MKL_ARCH STREQUAL "ia32")
  list(APPEND MKL_LIBRARIES ${MKL_BLACS})
  if(USE_MPI)
    list(APPEND MKL_LINK_LINE MKL::${MKL_BLACS})
  endif()
endif()
list(APPEND MKL_LINK_LINE ${END_GROUP})

# Find all requested libraries
list(APPEND MKL_REQUESTED_LIBRARIES ${MKL_LIBRARIES})
if(SYCL_COMPILER)
  # If SYCL_COMPILER is still ON, MKL_SYCL_ARCH, MKL_SYCL_LINK, and MKL_SYCL_IFACE_LIB are the same as MKL_ARCH, MKL_LINK, and MKL_IFACE_LIB.
  # Hence we can combine the libraries and find them in the following for loop.
  # Note that MKL_SYCL_THREADING and MKL_THREADING could be different because of the default value.
  list(APPEND MKL_REQUESTED_LIBRARIES ${MKL_SYCL_LIBRARIES})
  list(REMOVE_DUPLICATES MKL_REQUESTED_LIBRARIES)
endif()
foreach(lib ${MKL_REQUESTED_LIBRARIES})
  unset(${lib}_file CACHE)
  if(MKL_LINK STREQUAL "static" AND NOT ${lib} STREQUAL ${MKL_SDL})
    find_library(${lib}_file ${LIB_PREFIX}${lib}${LIB_EXT}
                  PATHS ${MKL_ROOT}
                  PATH_SUFFIXES "lib${MKL_ARCH_DIR}"
                  NO_DEFAULT_PATH)
    add_library(MKL::${lib} STATIC IMPORTED)
  else()
    find_library(${lib}_file NAMES ${LIB_PREFIX}${lib}${DLL_EXT}
                  ${LIB_PREFIX}${lib}${DLL_EXT}.${SO_VER}
                  ${LIB_PREFIX}${lib}${DLL_EXT}.${SYCL_SO_VER}
                  ${lib}
                  PATHS ${MKL_ROOT}
                  PATH_SUFFIXES "lib${MKL_ARCH_DIR}"
                  NO_DEFAULT_PATH)
    add_library(MKL::${lib} SHARED IMPORTED)
  endif()
  if(NOT MKL_LINK STREQUAL "static" AND ${lib} MATCHES "mkl_sycl" AND ${${lib}_file} STREQUAL "${lib}_file-NOTFOUND")
    list(APPEND MISSED_MKL_SYCL_LIBS ${lib})
    set(MKL_SYCL_DOMAIN "")
    string(REGEX REPLACE "mkl_sycl_" "" MKL_SYCL_DOMAIN ${lib})
    if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
      string(REGEX REPLACE "d$" "" MKL_SYCL_DOMAIN ${MKL_SYCL_DOMAIN})
    endif()
    string(TOUPPER ${MKL_SYCL_DOMAIN} MKL_SYCL_DOMAIN)
    mkl_message(WARNING "Could NOT find MKL ${lib} for target MKL::MKL_SYCL::${MKL_SYCL_DOMAIN}")
  else()
    if(NOT USE_MPI AND (${lib} MATCHES "mkl_scalapack" OR ${lib} MATCHES "mkl_blacs" OR ${lib} MATCHES "mkl_cdft")
            AND ${${lib}_file} STREQUAL "${lib}_file-NOTFOUND")
      if(${lib} MATCHES "mkl_scalapack")
        mkl_message(STATUS "Could NOT find MKL ${lib} for target MKL::MKL_SCALAPACK")
      endif()
      if(${lib} MATCHES "mkl_cdft")
        mkl_message(STATUS "Could NOT find MKL ${lib} for target MKL::MKL_CDFT")
      endif()
      if(${lib} MATCHES "mkl_blacs")
        mkl_message(STATUS "Could NOT find MKL ${lib} for targets MKL::MKL_SCALAPACK, MKL::MKL_CDFT, and MKL::MKL_BLACS")
      endif()
    else()
      check_required_vars(${lib}_file)
      mkl_message(STATUS "Found ${${lib}_file}")
    endif()
  endif()
  # CMP0111, implemented in CMake 3.20+ requires a shared library target on Windows
  # to be defined with IMPLIB and LOCATION property.
  # It also requires a static library target to be defined with LOCATION property.
  # Setting the policy to OLD usage, using cmake_policy() does not work as of 3.20.0, hence the if-else below.
  if(WIN32 AND NOT MKL_LINK STREQUAL "static")
    set_target_properties(MKL::${lib} PROPERTIES IMPORTED_IMPLIB "${${lib}_file}")
    # Find corresponding DLL
    set(MKL_DLL_GLOB ${lib}.*.dll)
    file(GLOB MKL_DLL_FILE "${MKL_ROOT}/bin${MKL_ARCH_DIR}/${MKL_DLL_GLOB}"
        # Legacy oneAPI layout support below
        "${MKL_ROOT}/redist/${MKL_ARCH}/${MKL_DLL_GLOB}"
        "${MKL_ROOT}/../redist/${MKL_ARCH}/${MKL_DLL_GLOB}"
        "${MKL_ROOT}/../redist/${MKL_ARCH}/mkl/${MKL_DLL_GLOB}"
        # Support for Conda directory layout
        "${MKL_ROOT}/bin/${MKL_DLL_GLOB}"
    )
    if(NOT ${lib} STREQUAL ${MKL_IFACE_LIB} AND NOT ${lib} STREQUAL ${MKL_BLAS95} AND NOT ${lib} STREQUAL ${MKL_LAPACK95})  # Windows IFACE libs are static only
      list(LENGTH MKL_DLL_FILE MKL_DLL_FILE_LEN)
      if(MKL_DLL_FILE_LEN)
        # in case multiple versions of the same dll are found, select the highest version
        list(SORT MKL_DLL_FILE)
        list(REVERSE MKL_DLL_FILE)
        list(GET MKL_DLL_FILE 0 MKL_DLL_FILE)

        mkl_message(STATUS "Found DLL: ${MKL_DLL_FILE}")
        set_target_properties(MKL::${lib} PROPERTIES IMPORTED_LOCATION "${MKL_DLL_FILE}")
      else()
        if(${lib} MATCHES "mkl_sycl" AND ${${lib}_file} STREQUAL "${lib}_file-NOTFOUND")
          mkl_message(WARNING "Could NOT find ${MKL_DLL_GLOB} for target MKL::MKL_SYCL::${MKL_SYCL_DOMAIN}")
        else()
          mkl_not_found_and_return("${MKL_DLL_GLOB} not found")
        endif()
      endif()
    endif()
    if(NOT DEFINED MKL_DLL_DIR AND MKL_DLL_FILE)
      get_filename_component(MKL_DLL_DIR ${MKL_DLL_FILE} DIRECTORY)
    endif()
  else()
    set_target_properties(MKL::${lib} PROPERTIES IMPORTED_LOCATION "${${lib}_file}")
  endif()
  list(APPEND MKL_IMPORTED_TARGETS MKL::${lib})
endforeach()

# Threading selection
if(MKL_THREADING STREQUAL "tbb_thread" OR MKL_SYCL_THREADING STREQUAL "tbb_thread")
  find_package(TBB CONFIG COMPONENTS tbb)
  if(TARGET TBB::tbb)
    if(MKL_THREADING STREQUAL "tbb_thread")
      set(MKL_THREAD_LIB $<TARGET_LINKER_FILE:TBB::tbb>)
      set(MKL_SDL_THREAD_ENV "TBB")
    endif()
    if(MKL_SYCL_THREADING STREQUAL "tbb_thread")
      set(MKL_SYCL_THREAD_LIB $<TARGET_LINKER_FILE:TBB::tbb>)
    endif()
    get_property(TBB_LIB TARGET TBB::tbb PROPERTY IMPORTED_LOCATION_RELEASE)
    get_filename_component(TBB_LIB_DIR ${TBB_LIB} DIRECTORY)
  else()
    if(UNIX)
      set(TBB_LIBNAME libtbb.so)
    else()
      set(TBB_LIBNAME tbb.lib)
    endif()
    find_path(TBB_LIB_DIR ${TBB_LIBNAME}
        HINTS $ENV{TBBROOT} $ENV{MKLROOT} ${MKL_ROOT} ${TBB_ROOT}
        PATH_SUFFIXES "lib" "lib/intel64/gcc4.4" "lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/gcc4.4" "../tbb/lib/intel64/gcc4.8"
                 "../../tbb/latest/lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/vc14" "lib/intel64/vc14"
    )
    find_library(TBB_LIBRARIES NAMES tbb
        HINTS $ENV{TBBROOT} $ENV{MKLROOT} ${MKL_ROOT} ${TBB_ROOT}
        PATH_SUFFIXES "lib" "lib/intel64/gcc4.4" "lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/gcc4.4" "../tbb/lib/intel64/gcc4.8"
                 "../../tbb/latest/lib/intel64/gcc4.8"
                 "../tbb/lib/intel64/vc14" "lib/intel64/vc14"
    )
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(MKL REQUIRED_VARS TBB_LIBRARIES)
  endif()
  if(UNIX)
    if(CMAKE_SKIP_BUILD_RPATH)
      set(TBB_LINK "-L${TBB_LIB_DIR} -ltbb")
    else()
      set(TBB_LINK "-Wl,-rpath,${TBB_LIB_DIR} -L${TBB_LIB_DIR} -ltbb")
    endif()
    if(MKL_THREADING STREQUAL "tbb_thread")
      list(APPEND MKL_SUPP_LINK ${TBB_LINK})
      if(APPLE)
        list(APPEND MKL_SUPP_LINK -lc++)
      else()
        list(APPEND MKL_SUPP_LINK -lstdc++)
        # force clang to link libstdc++
        if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
          list(APPEND MKL_SUPP_LINK -stdlib=libstdc++ )
        endif()
      endif()
    endif()
    if(MKL_SYCL_THREADING STREQUAL "tbb_thread")
      list(APPEND MKL_SYCL_SUPP_LINK ${TBB_LINK})
    endif()
  endif()
  if(WIN32 OR APPLE)
    set(MKL_ENV_PATH ${TBB_LIB_DIR})
  endif()
endif()
if(NOT MKL_THREADING STREQUAL "tbb_thread" AND MKL_THREADING MATCHES "_thread")
  if(MKL_THREADING STREQUAL "pgi_thread")
    list(APPEND MKL_SUPP_LINK -mp -pgf90libs)
    set(MKL_SDL_THREAD_ENV "PGI")
  elseif(MKL_THREADING STREQUAL "gnu_thread")
    list(APPEND MKL_SUPP_LINK -lgomp)
    set(MKL_SDL_THREAD_ENV "GNU")
  else()
    # intel_thread
    if(UNIX)
      set(MKL_OMP_LIB iomp5)
      set(LIB_EXT ".so")
      if(APPLE)
        set(LIB_EXT ".dylib")
      endif()
    else()
      set(MKL_OMP_LIB libiomp5md)
    endif()
    set(MKL_SDL_THREAD_ENV "INTEL")
    set(OMP_LIBNAME ${LIB_PREFIX}${MKL_OMP_LIB}${LIB_EXT})

    find_library(OMP_LIBRARY ${OMP_LIBNAME}
      HINTS $ENV{LIB} ${ENV_LIBRARY_PATH} $ENV{MKLROOT} ${MKL_ROOT} $ENV{CMPLR_ROOT}
      PATH_SUFFIXES "lib" "lib/${MKL_ARCH}"
             "lib/${MKL_ARCH}_lin" "lib/${MKL_ARCH}_win"
             "linux/compiler/lib/${MKL_ARCH}"
             "linux/compiler/lib/${MKL_ARCH}_lin"
             "windows/compiler/lib/${MKL_ARCH}"
             "windows/compiler/lib/${MKL_ARCH}_win"
             "../compiler/lib/${MKL_ARCH}_lin" "../compiler/lib/${MKL_ARCH}_win"
             "../compiler/lib/${MKL_ARCH}" "../compiler/lib" "compiler/lib"
             "../../compiler/latest/lib"
             "../../compiler/latest/linux/compiler/lib/${MKL_ARCH}"
             "../../compiler/latest/linux/compiler/lib/${MKL_ARCH}_lin"
             "../../compiler/latest/windows/compiler/lib/${MKL_ARCH}"
             "../../compiler/latest/windows/compiler/lib/${MKL_ARCH}_win"
             "../../compiler/latest/mac/compiler/lib"
      NO_DEFAULT_PATH)
    if(WIN32)
      set(OMP_DLLNAME ${LIB_PREFIX}${MKL_OMP_LIB}.dll)
      find_path(OMP_DLL_DIR ${OMP_DLLNAME}
        HINTS $ENV{LIB} ${ENV_LIBRARY_PATH} $ENV{MKLROOT} ${MKL_ROOT} $ENV{CMPLR_ROOT}
        PATH_SUFFIXES "bin"
              # Legacy layout support for oneMKL
              "redist/${MKL_ARCH}"
              "redist/${MKL_ARCH}_win" "redist/${MKL_ARCH}_win/compiler"
              "../redist/${MKL_ARCH}/compiler" "../compiler/lib"
              "../../compiler/latest/bin"
              "../../compiler/latest/windows/redist/${MKL_ARCH}_win"
              "../../compiler/latest/windows/redist/${MKL_ARCH}_win/compiler"
              "../../compiler/latest/windows/compiler/redist/${MKL_ARCH}_win"
              "../../compiler/latest/windows/compiler/redist/${MKL_ARCH}_win/compiler"
        NO_DEFAULT_PATH)
      check_required_vars(OMP_DLL_DIR)
      set(MKL_ENV_PATH "${OMP_DLL_DIR}")
    endif()

    if(WIN32 AND SKIP_LIBPATH)
      # Only for Intel OpenMP Offload
      set(OMP_LINK "libiomp5md.lib")
    else()
      set(OMP_LINK "${OMP_LIBRARY}")
      if(CMAKE_C_COMPILER_ID STREQUAL "PGI" OR CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
        # Disable PGI OpenMP runtime for correct work of Intel OpenMP runtime
        list(APPEND MKL_SUPP_LINK -nomp)
      endif()
    endif()
    check_required_vars(OMP_LIBRARY OMP_LINK)
    mkl_message(STATUS "Found ${OMP_LIBRARY}")
    if(MKL_SYCL_THREADING STREQUAL "intel_thread")
      set(MKL_SYCL_THREAD_LIB ${OMP_LINK})
    endif()
    set(MKL_THREAD_LIB ${OMP_LINK})
  endif()
elseif(MKL_THREADING STREQUAL "sequential")
  # Sequential threading
  set(MKL_SDL_THREAD_ENV "SEQUENTIAL")
endif() # MKL_THREADING

if(UNIX)
  if(SYCL_COMPILER)
    list(APPEND MKL_SYCL_SUPP_LINK -lm -ldl -lpthread)
  endif()
  list(APPEND MKL_SUPP_LINK -lm -ldl -lpthread)
endif()

if(SYCL_COMPILER OR ENABLE_OMP_OFFLOAD)
  if(SYCL_COMPILER)
    if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
      list(APPEND MKL_SYCL_SUPP_LINK ${LINK_PREFIX}sycld${LINK_SUFFIX})
    else()
      list(APPEND MKL_SYCL_SUPP_LINK ${LINK_PREFIX}sycl${LINK_SUFFIX})
    endif()
    list(APPEND MKL_SYCL_SUPP_LINK ${LINK_PREFIX}OpenCL${LINK_SUFFIX})
  endif()
  if(ENABLE_OMP_OFFLOAD)
    if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
      list(APPEND MKL_SUPP_LINK ${LINK_PREFIX}sycld${LINK_SUFFIX})
    else()
      list(APPEND MKL_SUPP_LINK ${LINK_PREFIX}sycl${LINK_SUFFIX})
    endif()
    list(APPEND MKL_SUPP_LINK ${LINK_PREFIX}OpenCL${LINK_SUFFIX})
  endif()
endif()

# Setup link types based on input options
set(LINK_TYPES "")

if(SYCL_COMPILER OR ENABLE_OMP_OFFLOAD)
# Remove missed mkl_sycl libraries in case of incomplete oneMKL package
  if(MISSED_MKL_SYCL_LIBS)
    list(REMOVE_ITEM MKL_SYCL_LIBS ${MISSED_MKL_SYCL_LIBS})
    list(TRANSFORM MISSED_MKL_SYCL_LIBS PREPEND MKL:: OUTPUT_VARIABLE MISSED_MKL_SYCL_TARGETS)
    list(REMOVE_ITEM MKL_SYCL_LINK_LINE ${MISSED_MKL_SYCL_TARGETS})
    list(REMOVE_ITEM MKL_LINK_LINE ${MISSED_MKL_SYCL_TARGETS})
  endif()
endif()

if(SYCL_COMPILER)
  if(NOT TARGET MKL::MKL_SYCL)
    add_library(MKL::MKL_SYCL INTERFACE IMPORTED GLOBAL)
    add_library(MKL::MKL_DPCPP ALIAS MKL::MKL_SYCL)
    add_dependencies(MKL::MKL_SYCL MKL_SYCL_MESSAGE)
  endif()
  target_compile_options(MKL::MKL_SYCL INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${MKL_SYCL_COPT}>)
  target_link_libraries(MKL::MKL_SYCL INTERFACE ${MKL_SYCL_LINK_LINE} ${MKL_SYCL_THREAD_LIB} ${MKL_SYCL_SUPP_LINK})
  list(APPEND LINK_TYPES MKL::MKL_SYCL)
  foreach(lib ${MKL_SYCL_LIBS})
    set(MKL_SYCL_DOMAIN "")
    string(REGEX REPLACE "mkl_sycl_" "" MKL_SYCL_DOMAIN ${lib})
    if(WIN32 AND CMAKE_BUILD_TYPE MATCHES "Debug")
      string(REGEX REPLACE "d$" "" MKL_SYCL_DOMAIN ${MKL_SYCL_DOMAIN})
    endif()
    string(TOUPPER ${MKL_SYCL_DOMAIN} MKL_SYCL_DOMAIN)
    add_library(MKL::MKL_SYCL::${MKL_SYCL_DOMAIN} INTERFACE IMPORTED GLOBAL)
    add_dependencies(MKL::MKL_SYCL::${MKL_SYCL_DOMAIN} MKL_SYCL_MESSAGE)
    target_compile_options(MKL::MKL_SYCL::${MKL_SYCL_DOMAIN} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${MKL_SYCL_COPT}>)
    # Only dynamic link has domain specific libraries
    # Domain specific targets still use mkl_sycl for static
    if(MKL_LINK STREQUAL "static")
      target_link_libraries(MKL::MKL_SYCL::${MKL_SYCL_DOMAIN} INTERFACE ${MKL_SYCL_LINK_LINE} ${MKL_SYCL_THREAD_LIB} ${MKL_SYCL_SUPP_LINK})
    else()
      list(TRANSFORM MKL_SYCL_LINK_LINE REPLACE ".*mkl_sycl.*" "TBD")
      list(REMOVE_DUPLICATES MKL_SYCL_LINK_LINE)
      list(TRANSFORM MKL_SYCL_LINK_LINE REPLACE "TBD" "MKL::${lib}")
      target_link_libraries(MKL::MKL_SYCL::${MKL_SYCL_DOMAIN} INTERFACE ${MKL_SYCL_LINK_LINE} ${MKL_SYCL_THREAD_LIB} ${MKL_SYCL_SUPP_LINK})
    endif()
    list(APPEND LINK_TYPES MKL::MKL_SYCL::${MKL_SYCL_DOMAIN})
  endforeach(lib) # MKL_SYCL_LIBS
endif()
# Single target for all C, Fortran link-lines
if(NOT TARGET MKL::MKL)
  add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
endif()
target_compile_options(MKL::MKL INTERFACE
    $<$<COMPILE_LANGUAGE:C>:${MKL_C_COPT}>
    $<$<COMPILE_LANGUAGE:Fortran>:${MKL_F_COPT}>
    $<$<COMPILE_LANGUAGE:CXX>:${MKL_CXX_COPT}>
    $<IF:$<BOOL:${ENABLE_OMP_OFFLOAD}>,${MKL_OFFLOAD_COPT},>)
target_link_libraries(MKL::MKL INTERFACE ${MKL_LINK_LINE} ${MKL_THREAD_LIB} ${MKL_SUPP_LINK})
list(APPEND LINK_TYPES MKL::MKL)

# Define cluster components
if(NOT ${${MKL_CDFT}_file} STREQUAL "${MKL_CDFT}_file-NOTFOUND")
  if(NOT TARGET MKL::MKL_BLACS)
    add_library(MKL::MKL_BLACS INTERFACE IMPORTED GLOBAL)
  endif()
  if(MKL_LINK STREQUAL "static")
    # Static link requires duplications for cross library dependency resolutions
    target_link_libraries(MKL::MKL_BLACS INTERFACE ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_BLACS INTERFACE ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_BLACS INTERFACE MKL::MKL)
  else()
    target_link_libraries(MKL::MKL_BLACS INTERFACE MKL::MKL MKL::${MKL_BLACS})
  endif()
endif()
if(NOT ${${MKL_CDFT}_file} STREQUAL "${MKL_CDFT}_file-NOTFOUND"
        AND NOT ${${MKL_BLACS}_file} STREQUAL "${MKL_BLACS}_file-NOTFOUND")
  if(NOT TARGET MKL::MKL_CDFT)
    add_library(MKL::MKL_CDFT INTERFACE IMPORTED GLOBAL)
  endif()
  if(UNIX AND NOT APPLE AND MKL_LINK STREQUAL "static")
    # Static link requires duplications for cross library dependency resolutions
    target_link_libraries(MKL::MKL_CDFT INTERFACE ${${MKL_CDFT}_file} ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_CDFT INTERFACE ${${MKL_CDFT}_file} ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_CDFT INTERFACE MKL::MKL)
  else()
    target_link_libraries(MKL::MKL_CDFT INTERFACE MKL::${MKL_CDFT} MKL::MKL_BLACS)
  endif()
endif()
if(NOT ${${MKL_SCALAPACK}_file} STREQUAL "${MKL_SCALAPACK}_file-NOTFOUND"
        AND NOT ${${MKL_BLACS}_file} STREQUAL "${MKL_BLACS}_file-NOTFOUND")
  if(NOT TARGET MKL::MKL_SCALAPACK)
    add_library(MKL::MKL_SCALAPACK INTERFACE IMPORTED GLOBAL)
  endif()
  if(UNIX AND NOT APPLE AND MKL_LINK STREQUAL "static")
    # Static link requires duplications for cross library dependency resolutions
    target_link_libraries(MKL::MKL_SCALAPACK INTERFACE ${${MKL_SCALAPACK}_file} ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_SCALAPACK INTERFACE ${${MKL_SCALAPACK}_file} ${${MKL_IFACE_LIB}_file} ${${MKL_THREAD}_file} ${${MKL_CORE}_file} ${${MKL_BLACS}_file})
    target_link_libraries(MKL::MKL_SCALAPACK INTERFACE MKL::MKL)
  else()
    target_link_libraries(MKL::MKL_SCALAPACK INTERFACE MKL::${MKL_SCALAPACK} MKL::MKL_BLACS)
  endif()
endif()

foreach(link ${LINK_TYPES})
  # Set properties on all INTERFACE targets
  target_include_directories(${link} BEFORE INTERFACE "${MKL_INCLUDE}")
  list(APPEND MKL_IMPORTED_TARGETS ${link})
endforeach(link) # LINK_TYPES
# oneMKL could be added to implicit directories when it's defined in CPATH
# In order to avoid dependency on CPATH, remove oneMKL from implicit directories
if(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
  list(REMOVE_ITEM CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "${MKL_INCLUDE}")
endif()
if(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
  list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${MKL_INCLUDE}")
endif()

if(MKL_LINK STREQUAL "sdl")
  list(APPEND MKL_ENV "MKL_INTERFACE_LAYER=${MKL_SDL_IFACE_ENV}" "MKL_THREADING_LAYER=${MKL_SDL_THREAD_ENV}")
endif()
if(WIN32 AND NOT MKL_LINK STREQUAL "static")
  list(APPEND MKL_ENV "MKL_BLACS_MPI=${MKL_BLACS_ENV}")
endif()

# Add oneMKL dynamic libraries if RPATH is not defined on Unix
if(UNIX AND CMAKE_SKIP_BUILD_RPATH)
  if(MKL_LINK STREQUAL "sdl")
    set(MKL_LIB_DIR $<TARGET_FILE_DIR:MKL::${MKL_SDL}>)
  else()
    set(MKL_LIB_DIR $<TARGET_FILE_DIR:MKL::${MKL_CORE}>)
  endif()
  if(APPLE)
    list(APPEND MKL_ENV "DYLD_LIBRARY_PATH=${MKL_LIB_DIR}\;$ENV{DYLD_LIBRARY_PATH}")
  else()
    list(APPEND MKL_ENV "LD_LIBRARY_PATH=${MKL_LIB_DIR}\;$ENV{LD_LIBRARY_PATH}")
  endif()
endif()

# Add oneMKL dynamic libraries to PATH on Windows
if(WIN32 AND NOT MKL_LINK STREQUAL "static")
  set(MKL_ENV_PATH "${MKL_DLL_DIR}\;${MKL_ENV_PATH}")
endif()

if(MKL_ENV_PATH)
  list(APPEND MKL_ENV "PATH=${MKL_ENV_PATH}\;${OLD_PATH}")
  if(APPLE)
    list(APPEND MKL_ENV "DYLD_LIBRARY_PATH=${MKL_ENV_PATH}\:${OLD_PATH}")
  endif()
endif()

# Additional checks
if(ENABLE_TRY_SYCL_COMPILE AND "CXX" IN_LIST CURR_LANGS AND SYCL_COMPILER AND MKL_SYCL_LIBS)
  # The check is run only once with the result cached
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES MKL::MKL_SYCL)
  foreach(lib IN LISTS MKL_SYCL_LIBS)
    if(lib STREQUAL "mkl_sycl_blas")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/blas.hpp\"
        
        int main()
        {
            sycl::queue q;
            float x[1], res[1];
            oneapi::mkl::blas::asum(q, 1, x, 1, res);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_lapack")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/lapack.hpp\"
        
        int main()
        {
            sycl::queue q;
            float a[1], scratchpad[1];
            std::int64_t ipiv[1];
            oneapi::mkl::lapack::getrf(q, 1, 1, a, 1, ipiv, scratchpad, 1);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_dft")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/dfti.hpp\"
        
        int main()
        {
            namespace dft = oneapi::mkl::dft;
            dft::descriptor<dft::precision::SINGLE, dft::domain::REAL> desc(1);
            sycl::queue q;
            desc.commit(q);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_sparse")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/spblas.hpp\"
        
        int main()
        {
            sycl::queue q;
            oneapi::mkl::sparse::matrix_handle_t handle;
            float x[1], y[1];
            oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1, handle, x, 1, y);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_data_fitting")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/experimental/data_fitting.hpp\"
        
        int main()
        {
            namespace df = oneapi::mkl::experimental::data_fitting;
            sycl::queue q;
            df::spline<double, df::linear_spline::default_type> spl(q);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_rng")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/rng.hpp\"
        
        int main()
        {
            sycl::queue q;
            oneapi::mkl::rng::default_engine engine(q, 0);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_stats")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/stats.hpp\"
        
        int main()
        {
            namespace stats = oneapi::mkl::stats;
            sycl::queue q;
            float x[1], min[1];
            stats::min(q, stats::make_dataset<stats::layout::row_major>(1, 1, x), min);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    elseif(lib STREQUAL "mkl_sycl_vm")
      check_cxx_source_compiles("
        #include <sycl/sycl.hpp>
        #include \"oneapi/mkl/vm.hpp\"
        
        int main()
        {
            sycl::queue q;
            float a[1], b[1], y[1];
            oneapi::mkl::vm::add(q, 1, a, b, y);
            return 0;
        }
        " MKL_TRY_SYCL_COMPILE)
      break()
    endif()
  endforeach()
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT MKL_TRY_SYCL_COMPILE)
    mkl_not_found_and_return("The SYCL compiler \"${CMAKE_CXX_COMPILER}\" is not able to \
                              compile a simple test program that calls a oneMKL DPC++ API. \
                              See \"CMakeError.log\" for details. Besides environment issues, \
                              this could be caused by a compiler version that is incompatible \
                              with oneMKL ${${CMAKE_FIND_PACKAGE_NAME}_VERSION}.")
  endif()
endif()

unset(MKL_DLL_FILE)
unset(MKL_LIBRARIES)

endif() # MKL::MKL
