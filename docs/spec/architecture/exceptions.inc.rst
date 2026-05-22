.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_exceptions:

Exceptions and Error Handling
------------------------------

oneMath error handling relies on the mechanism of C++ exceptions. Should error occur, it will be propagated at the point of a function call where it is caught using standard C++ error handling mechanism.

.. _onemath_exception_classification:

Exception classification
++++++++++++++++++++++++

Exception classification in oneMath is aligned with C++ Standard Library classification. oneMath introduces class that defines the base class in the hierarchy of oneMath exception classes. All oneMath routines throw exceptions inherited from this base class.
In the hierarchy of oneMath exceptions, ``oneapi::math::exception`` is the base class inherited from ``std::exception`` class. All other oneMath exception classes are derived from this base class.

This specification does not require implementations to perform error-checking. However, if an implementation does provide error-checking, it shall use the following exception classes. Additional implementation-specific exception classes can be used for exceptional conditions not fitting any of these classes.

.. _onemath_common_exceptions:

Common exceptions
*****************

.. csv-table::
    :header: "Exception class", "Description"
    :widths: 40, 60

    ".. _onemath_exception:

    ``oneapi::math::exception``", "Reports general unspecified problem"
    ".. _onemath_exception_unsupported_device:

    ``oneapi::math::unsupported_device``", "Reports a problem when the routine is not supported on a specific device"
    ".. _onemath_exception_host_bad_alloc:

    ``oneapi::math::host_bad_alloc``", "Reports a problem that occurred during memory allocation on the host"
    ".. _onemath_exception_device_bad_alloc:

    ``oneapi::math::device_bad_alloc``", "Reports a problem that occurred during memory allocation on a specific device"
    ".. _onemath_exception_unimplemented:

    ``oneapi::math::unimplemented``", "Reports a problem when a specific routine has not been implemented for the specified parameters"
    ".. _onemath_exception_invalid_argument:

    ``oneapi::math::invalid_argument``", "Reports problem when arguments to the routine were rejected"
    ".. _onemath_exception_uninitialized:

    ``oneapi::math::uninitialized``", "Reports problem when a handle (descriptor) has not been initialized"
    ".. _onemath_exception_computation_error:

    ``oneapi::math::computation_error``", "Reports any computation errors that have occurred inside a oneMath routine"
    ".. _onemath_exception_batch_error:

    ``oneapi::math::batch_error``", "Reports errors that have occurred inside a batch oneMath routine"


.. _onemath_lapack_specific_exceptions:

LAPACK specific exceptions
**************************

.. csv-table::
    :header: "Exception class", "Description"
    :widths: 40, 60

    ".. _onemath_lapack_exception:

    ``oneapi::math::lapack::exception``", "Base class for all LAPACK exceptions providing access to info code familiar to users of conventional LAPACK API. All LAPACK related exceptions can be handled with catch block for this class."
    ".. _onemath_lapack_exception_invalid_argument:
    
    ``oneapi::math::lapack::invalid_argument``", "Reports errors when arguments provided to the LAPACK subroutine are inconsistent or do not match expected values. Class extends base ``oneapi::math::invalid_argument`` with ability to access conventional status info code."
    ".. _onemath_lapack_exception_computation_error:
    
    ``oneapi::math::lapack::computation_error``", "Reports computation errors that have occurred during call to LAPACK subroutine. Class extends base ``oneapi::math::computation_error`` with ability to access conventional status info code familiar to LAPACK users."
    ".. _onemath_lapack_exception_batch_error:
    
    ``oneapi::math::lapack::batch_error``", "Reports errors that have occurred during batch LAPACK computations. Class extends base ``oneapi::math::batch_error`` with ability to access individual exception objects for each of the issues observed in a batch and an info code. The info code contains the number of errors that occurred in a batch. Positions of problems in a supplied batch that experienced issues during computations can be retrieved with `ids()` method, and list of particular exceptions can be obtained with `exceptions()` method of the exception object. Possible exceptions for a batch are documented for corresponding non-batch API."

