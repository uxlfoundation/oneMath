#--===============================================================================
# Copyright 2020-2022 Intel Corporation
#=================================================================================

if (NOT DEFINED HIP_PATH)
if (NOT DEFINED ENV{HIP_PATH})
set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed") 
else() 
set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed") 
endif() 
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH}) 
list(APPEND CMAKE_PREFIX_PATH "${HIP_PATH}/lib/cmake" "${HIP_PATH}/../lib/cmake" 
     "${HIP_PATH}/../lib/cmake/rocrand")

#find_package(HIP QUIET)
find_package(hip QUIET)
find_package(rocrand REQUIRED)

get_filename_component(SYCL_BINARY_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

# this is work around to avoid duplication half creation in both hip and SYCL
add_compile_definitions(HIP_NO_HALF)

find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rocRAND
    REQUIRED_VARS
      HIP_INCLUDE_DIRS
      HIP_LIBRARIES
      rocrand_INCLUDE_DIR
      rocrand_LIBRARIES
     
)

if(NOT TARGET ONEMKL::rocRAND::rocRAND)
  add_library(ONEMKL::rocRAND::rocRAND SHARED IMPORTED)
  set_target_properties(ONEMKL::rocRAND::rocRAND PROPERTIES
      IMPORTED_LOCATION "${HIP_PATH}/../lib/librocrand.so"
      INTERFACE_INCLUDE_DIRECTORIES "${rocrand_INCLUDE_DIR};${HIP_INCLUDE_DIRS};"
      INTERFACE_LINK_LIBRARIES "Threads::Threads;${rocrand_LIBRARIES};"
  )

endif()
