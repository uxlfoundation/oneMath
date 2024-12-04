..
  Copyright 2020 Intel Corporation

.. _create_backend_wrappers:

Integrating a Third-Party Library to oneAPI Math Library (oneMath)
==================================================================

This step-by-step tutorial provides examples for enabling new third-party libraries in oneMath.

oneMath has a header-based implementation of the interface layer (``include`` directory) and a source-based implementation of the backend layer for each third-party library (``src`` directory). To enable a third-party library, you must update both parts of oneMath and integrate the new third-party library to the oneMath build and test systems.

For the new backend library and header naming please use the following template:

.. code-block::

    onemath_<domain>_<3rd-party library short name>[<wrapper for specific target>]

Where ``<wrapper for specific target>`` is required only if multiple wrappers are provided from the same 3rd-party library, e.g., wrappers with Intel oneMKL C API for CPU target ``onemath_blas_mklcpu.so`` and wrappers with Intel oneMKL DPC++ API for GPU target ``onemath_blas_mklgpu.so``.

If there is no need for multiple wrappers only ``<domain>`` and ``<3rd-party library short name>`` are required, e.g. ``onemath_rng_curand.so``

`1. Create Header Files`_

`2. Integrate Header Files`_

`3. Create Wrappers`_

`4. Integrate Wrappers To the Build System`_

`5. Update the Test System`_

.. _generate_header_files:

1. Create Header Files
----------------------

For each new backend library, you should create the following two header files:

* Header file with a declaration of entry points to the new third-party library wrappers
* Compiler-time dispatching interface (see `oneMath Usage Models <../README.md#supported-usage-models>`_) for new third-party libraries

**Header File Example**: command to generate the header file with a declaration of BLAS entry points in the oneapi::math::newlib namespace 

.. code-block:: bash

    python scripts/generate_backend_api.py include/oneapi/math/blas.hpp \                                  # Base header file
                                           include/oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp \ # Output header file
                                           oneapi::math::newlib                                            # Wrappers namespace

Code snippet of the generated header file ``include/oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp``

.. code-block:: cpp

    namespace oneapi {
    namespace math {
    namespace newlib {
    
    void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
              sycl::buffer<float, 1> &result);



**Compile-time Dispatching Interface Example**: command to generate the compile-time dispatching interface template instantiations for ``newlib`` and supported device ``newdevice``

.. code-block:: bash

    python scripts/generate_ct_instant.py   include/oneapi/math/blas/detail/blas_ct_templates.hpp \         # Base header file
                                            include/oneapi/math/blas/detail/newlib/blas_ct.hpp \            # Output header file
                                            include/oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp \ # Header file with declaration of entry points to wrappers
                                            newlib \                                                       # Library name
                                            newdevice \                                                    # Backend name
                                            oneapi::math::newlib                                            # Wrappers namespace

Code snippet of the generated header file ``include/oneapi/math/blas/detail/newlib/blas_ct.hpp``

.. code-block:: cpp

    namespace oneapi {
    namespace math {
    namespace blas {
    
    template <>
    void asum<library::newlib, backend::newdevice>(sycl::queue &queue, std::int64_t n,
                                                   sycl::buffer<float, 1> &x, std::int64_t incx,
                                                   sycl::buffer<float, 1> &result) {
        asum_precondition(queue, n, x, incx, result);
        oneapi::math::newlib::asum(queue, n, x, incx, result);
        asum_postcondition(queue, n, x, incx, result);
    }


.. _integrate_header_files:

2. Integrate Header Files
-------------------------

Below you can see structure of oneMath top-level include directory:

::

    include/
        oneapi/
            math/
                math.hpp -> oneMath spec APIs
                types.hpp  -> oneMath spec types
                blas.hpp   -> oneMath BLAS APIs w/ pre-check/dispatching/post-check
                detail/    -> implementation specific header files
                    exceptions.hpp        -> oneMath exception classes
                    backends.hpp          -> list of oneMath backends
                    backends_table.hpp    -> table of backend libraries for each domain and device
                    get_device_id.hpp     -> function to query device information from queue for Run-time dispatching
                blas/
                    predicates.hpp -> oneMath BLAS pre-check post-check
                    detail/        -> BLAS domain specific implementation details
                        blas_loader.hpp       -> oneMath Run-time BLAS API
                        blas_ct_templates.hpp -> oneMath Compile-time BLAS API general templates
                        cublas/
                            blas_ct.hpp            -> oneMath Compile-time BLAS API template instantiations for <cublas>
                            onemath_blas_cublas.hpp -> backend wrappers library API
                        mklcpu/
                            blas_ct.hpp            -> oneMath Compile-time BLAS API template instantiations for <mklcpu>
                            onemath_blas_mklcpu.hpp -> backend wrappers library API
                        <other backends>/
                <other domains>/


To integrate the new third-party library to a oneMath header-based part, following files from this structure should be updated:

* ``include/oneapi/math/detail/backends.hpp``: add the new backend

  **Example**: add the ``newbackend`` backend

  .. code-block:: diff

        enum class backend { mklcpu,
     +                       newbackend,


  .. code-block:: diff

        static backendmap backend_map = { { backend::mklcpu, "mklcpu" },
     +                                    { backend::newbackend, "newbackend" },

* ``include/oneapi/math/detail/backends_table.hpp``: add new backend library for supported domain(s) and device(s)

  **Example**: enable ``newlib`` for ``blas`` domain and ``newdevice`` device

  .. code-block:: diff
    
        enum class device : uint16_t { x86cpu,
                                       ...
     +                                 newdevice
                                     };
        
        static std::map<domain, std::map<device, std::vector<const char*>>> libraries = {
            { domain::blas,
              { { device::x86cpu,
                  {
        #ifdef ONEMATH_ENABLE_MKLCPU_BACKEND
                      LIB_NAME("blas_mklcpu")
        #endif
                   } },
     +          { device::newdevice,
     +            {
     +  #ifdef ONEMATH_ENABLE_NEWLIB_BACKEND
     +                 LIB_NAME("blas_newlib")
     +  #endif
     +             } },

* ``include/oneapi/math/detail/get_device_id.hpp``: add new device detection mechanism for Run-time dispatching

  **Example**: enable ``newdevice`` if the queue is targeted for the Host

  .. code-block:: diff
    
        inline oneapi::math::device get_device_id(sycl::queue &queue) {
            oneapi::math::device device_id;
     +      if (queue.is_host())
     +          device_id=device::newdevice;

* ``include/oneapi/math/blas.hpp``: include the generated header file for the compile-time dispatching interface (see `oneMath Usage Models <../README.md#supported-usage-models>`_)

  **Example**: add ``include/oneapi/math/blas/detail/newlib/blas_ct.hpp`` generated at the `1. Create Header Files`_ step
    
  .. code-block:: diff
    
        #include "oneapi/math/blas/detail/mklcpu/blas_ct.hpp"
        #include "oneapi/math/blas/detail/mklgpu/blas_ct.hpp"
     +  #include "oneapi/math/blas/detail/newlib/blas_ct.hpp"


The new files generated at the `1. Create Header Files`_ step result in the following updated structure of the BLAS domain header files.

.. code-block:: diff

    include/
        oneapi/
            math/
                blas.hpp -> oneMath BLAS APIs w/ pre-check/dispatching/post-check
                blas/
                    predicates.hpp -> oneMath BLAS pre-check post-check
                    detail/        -> BLAS domain specific implementation details
                        blas_loader.hpp       -> oneMath Run-time BLAS API
                        blas_ct_templates.hpp -> oneMath Compile-time BLAS API general templates
                        cublas/
                            blas_ct.hpp            -> oneMath Compile-time BLAS API template instantiations for <cublas>
                            onemath_blas_cublas.hpp -> backend wrappers library API
                        mklcpu/
                            blas_ct.hpp            -> oneMath Compile-time BLAS API template instantiations for <mklcpu>
                            onemath_blas_mklcpu.hpp -> backend wrappers library API
        +              newlib/
        +                  blas_ct.hpp            -> oneMath Compile-time BLAS API template instantiations for <newbackend>
        +                  onemath_blas_newlib.hpp -> backend wrappers library API
                        <other backends>/
                <other domains>/

.. _generate_wrappers_and_cmake:

3. Create Wrappers
------------------
Wrappers convert Data Parallel C++ (DPC++) input data types to third-party library data types and call corresponding implementation from the third-party library. Wrappers for each third-party library are built to separate oneMath backend libraries. The ``libonemath.so`` dispatcher library loads the wrappers at run-time if you are using the interface for run-time dispatching, or you will link with them directly in case you are using the interface for compile-time dispatching (for more information see `oneMath Usage Models <../README.md#supported-usage-models>`_).

All wrappers and dispatcher library implementations are in the ``src`` directory:

::

    src/
        include/
            function_table_initializer.hpp -> general loader implementation w/ global libraries table
        blas/
            function_table.hpp -> loaded BLAS functions declaration
            blas_loader.cpp -> BLAS wrappers for loader
            backends/
                cublas/ -> cuBLAS wrappers
                mklcpu/ -> Intel oneMKL CPU wrappers
                mklgpu/ -> Intel oneMKL GPU wrappers
                <other backend libraries>/
        <other domains>/

Each backend library should contain a table of all functions from the chosen domain.

``scripts/generate_wrappers.py`` can help to generate wrappers with the "Not implemented" exception for all functions based on the provided header file.

You can modify wrappers generated with this script to enable third-party library functionality.

**Example**: generate wrappers for ``newlib`` based on the header files generated and integrated previously, and enable only one ``asum`` function

The command below generates two new files:

* ``src/blas/backends/newlib/newlib_wrappers.cpp`` - DPC++ wrappers for all functions from ``include/oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp``
* ``src/blas/backends/newlib/newlib_wrappers_table_dyn.cpp`` - structure of symbols for run-time dispatcher (in the same location as wrappers), suffix ``_dyn`` indicates that this file is required for dynamic library only.

.. code-block:: bash

    python scripts/generate_wrappers.py include/oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp \ # Base header file
                                        src/blas/function_table.hpp \                                  # Declaration for structure of symbols
                                        src/blas/backends/newlib/newlib_wrappers.cpp \                 # Output wrappers
                                        newlib                                                         # Library name

You can then modify ``src/blas/backends/newlib/newlib_wrappers.cpp`` to enable the C function ``newlib_sasum`` from the third-party library ``libnewlib.so``.

To enable this function:

* Include the header file ``newlib.h`` with the ``newlib_sasum`` function declaration
* Convert all DPC++ parameters to proper C types: use the ``get_access`` method for input and output DPC++ buffers to get row pointers
* Submit the DPC++ kernel with a C function call to ``newlib`` as ``single_task``

The following code snippet is updated for ``src/blas/backends/newlib/newlib_wrappers.cpp``:

.. code-block:: diff

        #if __has_include(<sycl/sycl.hpp>)
        #include <sycl/sycl.hpp>
        #else
        #include <CL/sycl.hpp>
        #endif
        
        #include "oneapi/math/types.hpp"
        
        #include "oneapi/math/blas/detail/newlib/onemath_blas_newlib.hpp"
    +    
    +    #include "newlib.h"
        
        namespace oneapi {
        namespace math {
        namespace newlib {
        
        void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x, std::int64_t incx,
                   sycl::buffer<float, 1> &result) {
    -       throw std::runtime_error("Not implemented for newlib");
    +       queue.submit([&](sycl::handler &cgh) {
    +           auto accessor_x      = x.get_access<sycl::access::mode::read>(cgh);
    +           auto accessor_result = result.get_access<sycl::access::mode::write>(cgh);
    +           cgh.single_task<class newlib_sasum>([=]() {
    +               accessor_result[0] = ::newlib_sasum((const int)n, accessor_x.get_pointer(), (const int)incx);
    +           });
    +       });
        }
        
        void asum(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x, std::int64_t incx,
                  sycl::buffer<double, 1> &result) {
            throw std::runtime_error("Not implemented for newlib");
        }

Updated structure of the ``src`` folder with the ``newlib`` wrappers:

.. code-block:: diff

    src/
        blas/
            loader.hpp -> general loader implementation w/ global libraries table
            function_table.hpp -> loaded BLAS functions declaration
            blas_loader.cpp -> BLAS wrappers for loader
            backends/
                cublas/ -> cuBLAS wrappers
                mklcpu/ -> Intel oneMKL CPU wrappers
                mklgpu/ -> Intel oneMKL GPU wrappers
     +          newlib/
     +              newlib.h
     +              newlib_wrappers.cpp
     +              newlib_wrappers_table_dyn.cpp
                <other backend libraries>/
        <other domains>/

.. _integrate_backend_to_build_system:

4. Integrate Wrappers to the Build System
-----------------------------------------
Here is the list of files that should be created/updated to integrate the new wrappers for the third-party library to the oneMath build system:

* Add the new option ``ENABLE_XXX_BACKEND`` for the new third-party library to the top of the ``CMakeList.txt`` file.

  **Example**: changes for ``newlib`` in the top of the ``CMakeList.txt`` file

  .. code-block:: diff

            option(ENABLE_MKLCPU_BACKEND "" ON)
            option(ENABLE_MKLGPU_BACKEND "" ON)
        +   option(ENABLE_NEWLIB_BACKEND "" ON)

* Add the new directory (``src/<domain>/backends/<new_directory>``) with the wrappers for the new third-party library under the ``ENABLE_XXX_BACKEND`` condition to the ``src/<domain>/backends/CMakeList.txt`` file.

  **Example**: changes for ``newlib`` in ``src/blas/backends/CMakeLists.txt``

  .. code-block:: diff
    
            if(ENABLE_MKLCPU_BACKEND)
                add_subdirectory(mklcpu)
            endif()
        +    
        +   if(ENABLE_NEWLIB_BACKEND)
        +       add_subdirectory(newlib)
        +   endif()

* Create the ``cmake/FindXXX.cmake`` cmake config file to find the new third-party library and its dependencies.

  **Example**: new config file ``cmake/FindNEWLIB.cmake`` for ``newlib``
    
  .. code-block:: cmake
    
        include_guard()
        # Find library by name in NEWLIB_ROOT cmake variable or environment variable NEWLIBROOT
        find_library(NEWLIB_LIBRARY NAMES newlib
            HINTS ${NEWLIB_ROOT} $ENV{NEWLIBROOT}
            PATH_SUFFIXES "lib")
        # Make sure that the library was found
        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(NEWLIB REQUIRED_VARS NEWLIB_LIBRARY)
        # Set cmake target for the library
        add_library(ONEMATH::NEWLIB::NEWLIB UNKNOWN IMPORTED)
        set_target_properties(ONEMATH::NEWLIB::NEWLIB PROPERTIES
            IMPORTED_LOCATION ${NEWLIB_LIBRARY})

* Create the ``src/<domain>/backends/<new_directory>/CMakeList.txt`` cmake config file to specify how to build the backend layer for the new third-party library.

  ``scripts/generate_cmake.py`` can help to generate the initial ``src/<domain>/backends/<new_directory>/CMakeList.txt`` config file automatically for all files in the directory.
  Note: all source files with the ``_dyn`` suffix are added to build if the target is a dynamic library only.
  
  **Example**: command to generate the cmake config file for the ``src/blas/backends/newlib`` directory

  .. code-block:: bash

    python scripts/generate_cmake.py src/blas/backends/newlib \ # Full path to the directory
                                     newlib                     # Library name

  You should manually update the generated config file with information about the new ``cmake/FindXXX.cmake`` file and instructions about how to link with the third-party library.
  
  **Example**: update the generated ``src/blas/backends/newlib/CMakeLists.txt`` file

  .. code-block:: diff

            # Add third-party library
        -   # find_package(XXX REQUIRED)
        +   find_package(NEWLIB REQUIRED)
    
  .. code-block:: diff

            target_link_libraries(${LIB_OBJ}
                PUBLIC ONEMATH::SYCL::SYCL
        -       # Add third-party library to link with here
        +       PUBLIC ONEMATH::NEWLIB::NEWLIB
            )

Now you can build the backend library for ``newlib`` to make sure the third-party library integration was completed successfully (for more information, see `Build with cmake <../README.md#building-with-cmake>`_)

.. code-block:: bash

    cd build/
    cmake .. -DNEWLIB_ROOT=<path/to/newlib> \
        -DENABLE_MKLCPU_BACKEND=OFF \
        -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_NEWLIB_BACKEND=ON \           # Enable new third-party library backend
        -DBUILD_FUNCTIONAL_TESTS=OFF           # At this step we want build only
    cmake --build . -j4

.. _integrate_backend_to_test_system:

5. Update the Test System
-------------------------

Update the following files to enable the new third-party library for unit tests:

* ``src/config.hpp.in``: add a cmake option for the new third-party library so this macro can be propagated to unit tests
    
  **Example**: add ``ENABLE_NEWLIB_BACKEND``

  .. code-block:: diff
    
        #cmakedefine ONEMATH_ENABLE_MKLCPU_BACKEND
     +  #cmakedefine ONEMATH_ENABLE_NEWLIB_BACKEND

* ``tests/unit_tests/CMakeLists.txt``: add instructions about how to link tests with the new backend library

  **Example**: add the ``newlib`` backend library

  .. code-block:: diff
    
        if(ENABLE_MKLCPU_BACKEND)
            add_dependencies(test_main_ct onemath_blas_mklcpu)
            if(BUILD_SHARED_LIBS)
                list(APPEND ONEMATH_LIBRARIES onemath_blas_mklcpu)
            else()
                list(APPEND ONEMATH_LIBRARIES -foffload-static-lib=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonemath_blas_mklcpu.a)
                find_package(MKL REQUIRED)
                list(APPEND ONEMATH_LIBRARIES ${MKL_LINK_C})
            endif()
        endif()
     +
     +    if(ENABLE_NEWLIB_BACKEND)
     +       add_dependencies(test_main_ct onemath_blas_newlib)
     +       if(BUILD_SHARED_LIBS)
     +           list(APPEND ONEMATH_LIBRARIES onemath_blas_newlib)
     +       else()
     +           list(APPEND ONEMATH_LIBRARIES -foffload-static-lib=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libonemath_blas_newlib.a)
     +           find_package(NEWLIB REQUIRED)
     +           list(APPEND ONEMATH_LIBRARIES ONEMATH::NEWLIB::NEWLIB)
     +       endif()
     +   endif()

* ``tests/unit_tests/include/test_helper.hpp``: add the helper function for the compile-time dispatching interface with the new backend, and specify the device for which it should be called

  **Example**: add the helper function for the ``newlib`` compile-time dispatching interface with ``newdevice`` if it is the Host

  .. code-block:: diff
    
        #ifdef ONEMATH_ENABLE_MKLGPU_BACKEND
            #define TEST_RUN_INTELGPU(q, func, args) \
                func<oneapi::math::backend::mklgpu> args
        #else
            #define TEST_RUN_INTELGPU(q, func, args)
        #endif
     +    
     +  #ifdef ONEMATH_ENABLE_NEWLIB_BACKEND
     +     #define TEST_RUN_NEWDEVICE(q, func, args) \
     +         func<oneapi::math::backend::newbackend> args
     +  #else
     +      #define TEST_RUN_NEWDEVICE(q, func, args)
     +  #endif
 
  .. code-block:: diff
 
        #define TEST_RUN_CT(q, func, args)               \
            do {                                         \
     +          if (q.is_host())                         \
     +              TEST_RUN_NEWDEVICE(q, func, args);   \ 


* ``tests/unit_tests/main_test.cpp``: add the targeted device to the vector of devices to test

  **Example**: add the targeted device CPU for ``newlib``

  .. code-block:: diff
    
                }
            }
     +           
     +  #ifdef ONEMATH_ENABLE_NEWLIB_BACKEND
     +      devices.push_back(sycl::device(sycl::host_selector()));
     +  #endif

Now you can build and run functional testing for enabled third-party libraries (for more information see `Build with cmake <../README.md#building-with-cmake>`_).

.. code-block:: bash

    cd build/
    cmake .. -DNEWLIB_ROOT=<path/to/newlib> \
        -DENABLE_MKLCPU_BACKEND=OFF \
        -DENABLE_MKLGPU_BACKEND=OFF \
        -DENABLE_NEWLIB_BACKEND=ON  \
        -DBUILD_FUNCTIONAL_TESTS=ON
    cmake --build . -j4
    ctest
