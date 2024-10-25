.. _using_onemkl_interface_library_with_cmake:

Using the oneMKL Interfaces in your project with CMake
=============================================================

The CMake build tool can help you use oneMKL Interfaces in your own project.
Instead of manually linking and including directories, you can use the CMake targets
exported by the oneMKL Interfaces project. You can use oneMKL in one of two
forms, with the target names depending on the approach taken: 

* you can use a previously installed copy, either from a binary distribution or
  built from source. This can be imported using CMake's ``find_package``
  command. See the section `using_from_installed_binary`_.
* or you can have CMake automatically download and build oneMKL as part of the
  build process using CMake's FetchContent_ functionality.
  See the section `using_with_fetchcontent`_.


.. _using_from_installed_binary:

Using an installed oneMKL Interfaces
####################################

If the oneMKL Interfaces have been previously installed, either by building from
source or as a distributed binary, they can be consumed using CMake using
``find_package(oneMKL REQUIRED)``. The compiler used for the target library or
application should match that used to build oneMKL Interfaces.

For example:

.. code-block:: cmake

    find_package(oneMKL REQUIRED)
    target_link_libraries(myTarget PRIVATE MKL::onemkl)

Different targets can be used depending on the requirements of oneMKL. 
To link against the entire library, the ``MKL::onemkl`` target should be used.
For specific domains, ``MKL::onemkl_<domain>`` should be used.
And for specific backends, ``MKL::onemkl_<domain>_<backend>`` should be used.

When using a binary, it may be useful to know the backends that were enabled
during the build. To check for the existence of backends, CMake's ``if(TARGET
<target>)`` construct can be used. For example, with the ``cufft`` backend:

.. code-block:: cmake

    if(TARGET MKL::onemkl_dft_cufft)
        target_link_libraries(myTarget PRIVATE MKL::onemkl_dft_cufft)
    else()
        message(FATAL_ERROR "oneMKL Interfaces was not built with CuFFT backend")
    endif()

.. _using_with_fetchcontent:

Using CMake's FetchContent
##########################


The FetchContent_ functionality of CMake can be used to download, build and
install oneMKL Interfaces as part of the build.

For example:

.. code-block:: cmake

    include(FetchContent)
    set(BUILD_FUNCTIONAL_TESTS False)
    set(BUILD_EXAMPLES False)
    set(ENABLE_<BACKEND_NAME>_BACKEND True)
    FetchContent_Declare(
            onemkl_interface_library
            GIT_REPOSITORY https://github.com/oneapi-src/oneMKL.git
            GIT_TAG develop
    )
    FetchContent_MakeAvailable(onemkl_interface_library)

    target_link_libraries(myTarget PRIVATE onemkl)

The build parameters should be appropriately set before
``FetchContent_Declare``. See :ref:`building_the_project_with_dpcpp` or
:ref:`building_the_project_with_adaptivecpp`.

To link against the main library with run-time dispatching, use the target
``onemkl``. To link against particular domains, use the target
``onemkl_<domain>``. For example, ``onemkl_blas`` or ``onemkl_dft``. To link
against particular backends (as required for static dispatch of oneAPI calls to
a particular backend), use the target ``onemkl_<domain>_<backend>``. For
example, ``onemkl_dft_cufft``.

.. _FetchContent: https://cmake.org/cmake/help/latest/module/FetchContent.html
