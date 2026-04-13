#===============================================================================
#  FindOpenBLAS.cmake
#
#  Finds OpenBLAS library and headers.
#
#  Result variables:
#
#    OpenBLAS_FOUND
#    OPENBLAS_LIBRARY
#    OPENBLAS_INCLUDE
#
#  Imported target:
#
#    ONEMATH::OPENBLAS::OPENBLAS
#
#===============================================================================

include_guard()
include(FindPackageHandleStandardArgs)

# ------------------------------------------------------------------------------
# User hints
# ------------------------------------------------------------------------------

# Highest priority: OPENBLAS_DIR
if(DEFINED OPENBLAS_DIR)
    set(_OPENBLAS_HINTS ${OPENBLAS_DIR})
elseif(DEFINED ENV{OPENBLAS_DIR})
    set(_OPENBLAS_HINTS $ENV{OPENBLAS_DIR})
elseif(CMAKE_PREFIX_PATH)
    set(_OPENBLAS_HINTS ${CMAKE_PREFIX_PATH})
endif()

# ------------------------------------------------------------------------------
# Find library
# ------------------------------------------------------------------------------

find_library(OPENBLAS_LIBRARY
    NAMES openblas libopenblas
    HINTS ${_OPENBLAS_HINTS}
    PATH_SUFFIXES lib lib64
)

# ------------------------------------------------------------------------------
# Find include directory
# ------------------------------------------------------------------------------

find_path(OPENBLAS_INCLUDE
    NAMES cblas.h
    HINTS ${_OPENBLAS_HINTS}
    PATH_SUFFIXES include include/openblas
)

# ------------------------------------------------------------------------------
# Handle result
# ------------------------------------------------------------------------------

find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OPENBLAS_LIBRARY OPENBLAS_INCLUDE
)

if(OpenBLAS_FOUND)

    get_filename_component(OPENBLAS_LIB_DIR
        ${OPENBLAS_LIBRARY}
        DIRECTORY
    )

    # ----------------------------------------------------------
    # Create imported target
    # ----------------------------------------------------------
    add_library(ONEMATH::OPENBLAS::OPENBLAS UNKNOWN IMPORTED)

    set_target_properties(ONEMATH::OPENBLAS::OPENBLAS PROPERTIES
        IMPORTED_LOCATION ${OPENBLAS_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDE}
    )

    # ----------------------------------------------------------
    # RPATH handling (Linux)
    # ----------------------------------------------------------
    if(UNIX AND OPENBLAS_LIB_DIR)
        set_target_properties(ONEMATH::OPENBLAS::OPENBLAS PROPERTIES
            INTERFACE_LINK_OPTIONS "-Wl,-rpath,${OPENBLAS_LIB_DIR}"
        )
    endif()

endif()

mark_as_advanced(
    OPENBLAS_LIBRARY
    OPENBLAS_INCLUDE
)

