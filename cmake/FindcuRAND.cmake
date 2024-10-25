#--===============================================================================
# cuRAND back-end Copyright (c) 2021, The Regents of the University of
# California, through Lawrence Berkeley National Laboratory (subject to receipt
# of any required approvals from the U.S. Dept. of Energy). All rights
# reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# (1) Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# (2) Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# (3) Neither the name of the University of California, Lawrence Berkeley
# National Laboratory, U.S. Dept. of Energy nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches,
# or upgrades to the features, functionality or performance of the source
# code ("Enhancements") to anyone; however, if you choose to make your
# Enhancements available either publicly, or directly to Lawrence Berkeley
# National Laboratory, without imposing a separate written license agreement
# for such Enhancements, then you hereby grant the following license: a
# non-exclusive, royalty-free perpetual license to install, use, modify,
# prepare derivative works, incorporate into other computer software,
# distribute, and sublicense such enhancements or derivative works thereof,
# in binary and source code form.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at
# IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.  As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do
# so.
#=================================================================================

find_package(CUDA 10.0 REQUIRED)
get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

if (NOT (ONEMKL_SYCL_IMPLEMENTATION STREQUAL "hipsycl"))
# the OpenCL include file from cuda is opencl 1.1 and it is not compatible with DPC++
# the OpenCL include headers 1.2 onward is required. This is used to bypass NVIDIA OpenCL headers
find_path(OPENCL_INCLUDE_DIR CL/cl.h OpenCL/cl.h 
HINTS 
${OPENCL_INCLUDE_DIR}
${SYCL_BINARY_DIR}/../include/sycl/
${SYCL_BINARY_DIR}/../include/
)
endif()

# this is work around to avoid duplication half creation in both cuda and SYCL
add_compile_definitions(CUDA_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)

if (ONEMKL_SYCL_IMPLEMENTATION STREQUAL "hipsycl")
find_package_handle_standard_args(cuRAND
    REQUIRED_VARS
	CUDA_TOOLKIT_INCLUDE
	CUDA_curand_LIBRARY
        CUDA_LIBRARIES
        CUDA_CUDA_LIBRARY
)

  if(NOT TARGET ONEMKL::cuRAND::cuRAND)
  add_library(ONEMKL::cuRAND::cuRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::cuRAND::cuRAND PROPERTIES
    IMPORTED_LOCATION ${CUDA_curand_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}"
  )
  endif()
else()
find_package_handle_standard_args(cuRAND
    REQUIRED_VARS
	CUDA_TOOLKIT_INCLUDE
	CUDA_curand_LIBRARY
        CUDA_LIBRARIES
        CUDA_CUDA_LIBRARY
        OPENCL_INCLUDE_DIR
)

  if(NOT TARGET ONEMKL::cuRAND::cuRAND)
  add_library(ONEMKL::cuRAND::cuRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::cuRAND::cuRAND PROPERTIES
    IMPORTED_LOCATION ${CUDA_curand_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR};${CUDA_TOOLKIT_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CUDA_CUDA_LIBRARY};${CUDA_LIBRARIES}"
  )
  endif()
endif()
