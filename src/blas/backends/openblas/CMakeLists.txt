#===============================================================================
# OpenBLAS Backend for oneMath BLAS
#===============================================================================

set(LIB_NAME onemath_blas_openblas)
set(LIB_OBJ ${LIB_NAME}_obj)

# ------------------------------------------------------------------------------
# Find OpenBLAS (uses custom FindOpenBLAS.cmake)
# ------------------------------------------------------------------------------
find_package(OpenBLAS REQUIRED)

set(SOURCES
  openblas_common.hpp
  openblas_level1.cpp
  openblas_level2.cpp
  openblas_level3.cpp
  openblas_batch.cpp
  openblas_extensions.cpp
  $<$<BOOL:${BUILD_SHARED_LIBS}>:openblas_wrappers.cpp>
)

# ------------------------------------------------------------------------------
# Create libraries
# ------------------------------------------------------------------------------

add_library(${LIB_NAME})
add_deprecated_library(${LIB_NAME})

add_library(${LIB_OBJ} OBJECT ${SOURCES})
add_dependencies(onemath_backend_libs_blas ${LIB_NAME})

# ------------------------------------------------------------------------------
# SYCL integration
# ------------------------------------------------------------------------------

if (USE_ADD_SYCL_TO_TARGET_INTEGRATION)
  add_sycl_to_target(TARGET ${LIB_OBJ} SOURCES ${SOURCES})
endif()

# ------------------------------------------------------------------------------
# Include directories
# ------------------------------------------------------------------------------

target_include_directories(${LIB_OBJ}
  PUBLIC
      ${ONEMATH_INCLUDE_DIRS}
      ${OPENBLAS_INCLUDE}           
  PRIVATE
      ${PROJECT_SOURCE_DIR}/src/include
      ${PROJECT_SOURCE_DIR}/src
      ${CMAKE_BINARY_DIR}/bin
      ${ONEMATH_GENERATED_INCLUDE_PATH}
)

# ------------------------------------------------------------------------------
# Compile options
# ------------------------------------------------------------------------------

target_compile_options(${LIB_OBJ}
  PRIVATE ${ONEMATH_BUILD_COPT}
)

# ------------------------------------------------------------------------------
# Link OpenBLAS properly (via imported target)
# ------------------------------------------------------------------------------

target_link_libraries(${LIB_OBJ}
  PUBLIC
      ONEMATH::SYCL::SYCL
      ONEMATH::OPENBLAS::OPENBLAS  
)

set_target_properties(${LIB_OBJ} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------------------------
# Link object library into main backend lib
# ------------------------------------------------------------------------------

target_link_libraries(${LIB_NAME}
  PRIVATE ${LIB_OBJ}
)

target_include_directories(${LIB_NAME}
  PUBLIC ${ONEMATH_INCLUDE_DIRS}
)

if(BUILD_SHARED_LIBS)
  set_target_properties(${LIB_NAME} PROPERTIES
    INTERFACE_LINK_LIBRARIES ONEMATH::SYCL::SYCL
  )
endif()

set_target_properties(${LIB_NAME} PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ------------------------------------------------------------------------------
# RPATH handling
# ------------------------------------------------------------------------------

list(APPEND CMAKE_BUILD_RPATH $<TARGET_FILE_DIR:${LIB_NAME}>)

# ------------------------------------------------------------------------------
# Install
# ------------------------------------------------------------------------------

install(TARGETS ${LIB_OBJ} EXPORT oneMathTargets)

install(TARGETS ${LIB_NAME}
  EXPORT oneMathTargets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

