.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_descriptor:

The ``descriptor`` class template
=================================

Instances of any ``oneapi::math::dft::descriptor`` class define DFT(s) to be
computed. The usage of prepended namespace specifiers ``oneapi::math::dft`` is
omitted below for conciseness.

Description
+++++++++++

Any desired DFT is fully defined by an instance of a specialization of the
``descriptor`` class template, declared in the ``oneapi::math::dft`` namespace.
The :ref:`scoped enumeration types<onemath_dft_enums>` ``precision``, ``domain``,
``config_param`` and ``config_value`` defined in the same namespace (and the
corresponding ranges of values) are relevant to the definition and
configurations of such objects. Users can set several (resp. query all)
configuration parameters for (resp. from) any ``descriptor`` object by using
its ``set_value`` (resp. ``get_value``) member functions.

Invoking the ``commit`` member function of a ``descriptor`` object effectively
commits it to the desired DFT calculations (as it defines it) on the specific
device encapsulated by the ``sycl::queue`` object required by that function.

The desired forward (resp. backward) DFT calculations may then be computed by
passing such a committed ``descriptor`` object to a ``compute_forward`` (resp.
``compute_backward``) function (defined in the ``oneapi::math::dft`` namespace as
well), along with the relevant data containers (``sycl::buffer`` object(s) or
pointer(s) to a device-accessible USM allocations) for the  desired DFT(s). This
function makes the ``descriptor`` object enqueue the operations relevant for the
desired calculations to the ``sycl::queue`` object it was given when committing it.

.. note::
   The ``compute_forward`` and ``compute_backward`` functions may need to be
   able to access the internals of the ``descriptor`` object to compute the
   desired transform(s), this could be done for instance, by labeling them as
   friend functions of the ``descriptor`` class template.

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      class descriptor {
       private:
          using real_scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;

       public:
          
          // Constructor for 1-dimensional DFT
          descriptor(std::int64_t length); // d = 1;
          
          // Constructor for d-dimensional DFT
          descriptor(std::vector<std::int64_t> lengths); // d = lengths.size();

          descriptor(const descriptor&);

          descriptor(descriptor&&);

          descriptor& operator=(const descriptor&);

          descriptor& operator=(descriptor&&);

          ~descriptor();

          void set_value(config_param param, config_value value);
          void set_value(config_param param, std::int64_t value);
          void set_value(config_param param, real_scalar_t value);
          [[deprecated("Use set_value(config_param, const std::vector<std::int64_t>&) instead.")]]
          void set_value(config_param param, const std::int64_t* value);
          void set_value(config_param param, const std::vector<std::int64_t>& value);
          template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
          void set_value(config_param param, T value) {
              set_value(param, static_cast<std::int64_t>(value));
          }
          template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
          void set_value(config_param param, T value) {
              set_value(param, static_cast<real_scalar_t>(value));
          }
          [[deprecated("This set_value member function is deprecated.")]]
          void set_value(config_param param, ...);

          void get_value(config_param param, config_value* value_ptr) const;
          void get_value(config_param param, domain* value_ptr) const;
          void get_value(config_param param, precision* value_ptr) const;
          void get_value(config_param param, std::int64_t* value_ptr) const;
          void get_value(config_param param, real_scalar_t* value_ptr) const;
          void get_value(config_param param, std::vector<std::int64_t>* value_ptr) const;
          [[deprecated("This get_value member function is deprecated.")]]
          void get_value(config_param param, ...) const;
         
          void set_workspace(sycl::buffer<real_scalar_t, 1> &workspaceBuf);
          void set_workspace(real_scalar_t* workspaceUSM);
      
          void commit(sycl::queue &queue);
      
       };
   }

.. _onemath_dft_descriptor_template_parameters:

.. rubric:: Template parameters

``precision prec``
  Specifies the floating-point format of the user-provided data, the results,
  and the precision of the floating-point operations to be enqueued. The
  possible specialization values are ``precision::SINGLE`` and
  ``precision::DOUBLE``, corresponding to single-precision (FP32) and
  double-precision (FP64) floating-point formats, respectively . Objects of a
  ``descriptor`` class specialized with ``precision::SINGLE`` (resp.
  ``precision::DOUBLE``) as ``prec`` are referred to as "single-precision
  descriptors" (resp. "double-precision descriptors").

``domain dom``
  Specifies the type of forward domain for the transform. The possible
  specialization values are ``domain::COMPLEX`` and ``domain::REAL``,
  corresponding to complex and real forward domains, respectively. Objects of
  the ``descriptor`` class specialized with ``domain::COMPLEX`` (resp.
  ``domain::REAL``) as ``dom`` are referred to as "complex descriptors" (resp.
  "real descriptors").

.. _onemath_dft_descriptor_member_table:

.. rubric:: Member functions of the ``descriptor`` class template

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * -     Routines
     -     Description
   * -     :ref:`constructors<onemath_dft_descriptor_constructors>`
     -     Create a ``descriptor`` object. The parameterized constructors
           enable the (one-time) definition of the length(s)
           :math:`\lbrace n_1, \ldots, n_d\rbrace` (the dimension :math:`d` is
           deduced accordingly). The parameterized constructors default-initialize
           the object; copy and move constructors do not.
   * -     :ref:`assignment operators<onemath_dft_descriptor_assignment_operators>`
     -     Perform a deep copy of or moves the argument.
   * -     ``set_value`` :ref:`member functions<onemath_dft_descriptor_set_value>`
     -     Set a configuration value for a specific configuration parameter.
   * -     ``get_value`` :ref:`member functions<onemath_dft_descriptor_get_value>`
     -     Query the configuration value associated with a particular
           configuration parameter.
   * -     ``set_workspace`` :ref:`member function<onemath_dft_descriptor_set_workspace>`
     -     Equips the ``descriptor`` object with an external workspace.
   * -     ``commit`` :ref:`member function<onemath_dft_descriptor_commit>`
     -     Commits the ``descriptor`` object to enqueue the operations relevant
           to the DFT(s) it determines to a given, user-provided ``sycl::queue``
           object; completes all initialization work relevant to and required by
           the chosen, device-compliant implementation for the particular DFT,
           as defined by the ``descriptor`` object.

.. _onemath_dft_descriptor_constructors:

Constructors
++++++++++++

The parameterized constructors for a ``descriptor`` object instantiate it with
all the relevant default configuration settings (which may depend on the
specialization values for ``prec`` and ``dom``). The constructors do not perform
any significant initialization work as changes in the object's configuration(s)
may be operated thereafter (via its ``set_value``
:ref:`member functions<onemath_dft_descriptor_set_value>`) and modify
significantly the nature of that work.

The copy constructor performs a deep copy of ``descriptor`` objects.

The move constructor transfers the resources owned by a ``descriptor`` object,
without copying them.

.. rubric:: Parameterized constructor (one-dimensional transform)

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>::descriptor(std::int64_t length);

   }


.. rubric:: Parameterized constructor (transform of any dimension)

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>::descriptor(std::vector<std::int64_t> lengths);

   }

.. rubric:: Copy constructor

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>::descriptor(const descriptor<prec,dom>& other);

   }

.. rubric:: Move constructor

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>::descriptor(descriptor<prec,dom>&& other);

   }


.. container:: section

   .. rubric:: Input Parameters

   ``length``
      Length :math:`n_1 > 0` of the data sequence(s) for one-dimensional
      transform(s).

   ``lengths``
      Vector of :math:`d > 0` lengths :math:`\lbrace n_1, \ldots, n_d\rbrace`
      of the data sequence(s) for :math:`d`-dimensional transform(s). The values
      are to be provided in that order and such that
      :math:`n_j > 0,\ \forall j \in \lbrace 1, \ldots, d \rbrace`.

   ``other``
      Another ``descriptor`` object of the same type to copy or move.

.. container:: section

   .. rubric:: Throws

   The constructors shall throw the following
   :ref:`exceptions<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:

   ``oneapi::math::host_bad_alloc()``
      If any memory allocations on host have failed, for instance due to
      insufficient memory.

   ``oneapi::math::unimplemented()``
      If the dimension :math:`d`, *i.e.*, the size of ``lengths``, is larger
      than what is supported by the library implementation.
   

**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`


.. _onemath_dft_descriptor_assignment_operators:

Assignment operators
++++++++++++++++++++

The copy assignment operator results in a deep copy.

.. rubric:: Copy assignment

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>& descriptor<prec,dom>::operator=(const descriptor<prec,dom>& other);

   }

.. rubric:: Move assignment

.. code-block:: cpp
   
   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      descriptor<prec,dom>& descriptor<prec,dom>::operator=(descriptor<prec,dom>&& other);

   }


.. container:: section

   .. rubric:: Input Parameters

   ``other``
      Another ``descriptor`` object to copy or move from.

.. container:: section

   .. rubric:: Throws

   The assignment operators shall throw the following
   :ref:`exception<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:

   ``oneapi::math::host_bad_alloc()``
      If any memory allocations on host have failed, for instance due to
      insufficient memory.

**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`

.. _onemath_dft_descriptor_set_value:

``set_value`` member functions
++++++++++++++++++++++++++++++

The ``set_value`` member functions of any ``descriptor`` object set a
configuration value corresponding to a (read-write) configuration parameter for
the DFT(s) that it defines. These functions are to be used as many times as
required for all the necessary configuration parameters to be set prior to
committing the object (by calling its ``commit``
:ref:`member function<onemath_dft_descriptor_commit>`).

All these functions require and expect exactly **two** arguments: they set the
given configuration value (second argument) for a desired configuration
parameter (first argument), represented by ``param`` of type ``config_param``.
The expected type of the associated configuration value (second argument)
depends on ``param`` and is specified in the
:ref:`section<onemath_dft_enum_config_param>` dedicated to the ``config_param``
type and its enumerators (unless a deprecated version is used). The expected
type of configuration value determines which of the ``set_value`` overloads is
to be used for a specific value of ``param``.

.. rubric:: Syntax for integer-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, std::int64_t value);
   }

This version of ``set_value`` supports the following values of ``param``:

- ``config_param::NUMBER_OF_TRANSFORMS``;
- ``config_param::FWD_DISTANCE``;
- ``config_param::BWD_DISTANCE``.

.. rubric:: Syntax for real-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, real_scalar_t value);
   }

This version of ``set_value`` supports the following values of ``param``:

- ``config_param::FORWARD_SCALE``;
- ``config_param::BACKWARD_SCALE``.


.. rubric:: Syntax for vector-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, const std::vector<std::int64_t>& value);
   }

This version of ``set_value`` supports the following values of ``param``:

- ``config_param::FWD_STRIDES``;
- ``config_param::BWD_STRIDES``;
- ``config_param::INPUT_STRIDES`` (deprecated);
- ``config_param::OUTPUT_STRIDES`` (deprecated).

``value`` must be a vector of :math:`\left(d+1\right)` ``std::int64_t`` elements.
More information about setting strides may be found in the page dedicated to
the :ref:`configuration of data layouts<onemath_dft_config_data_layouts>`.

.. rubric:: Syntax for parameters associated with non-numeric values

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, config_value value);
   }

This version of ``set_value`` supports the following values of ``param``:

- ``config_param::COMPLEX_STORAGE``;
- ``config_param::PLACEMENT``;
- ``config_param::WORKSPACE_PLACEMENT``.

.. rubric:: Deprecated syntax for vector-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, const std::int64_t* value);
   }

This version of ``set_value`` supports the following values of ``param``:

- ``config_param::FWD_STRIDES``;
- ``config_param::BWD_STRIDES``;
- ``config_param::INPUT_STRIDES`` (deprecated);
- ``config_param::OUTPUT_STRIDES`` (deprecated);

and behaves as if redirecting to
``set_value(param, std::vector<std::int64_t>(value, d + 1))``. As a consequence,
``value`` must be a valid pointer to :math:`\left(d+1\right)` contiguous
``std::int64_t`` elements.

This version is deprecated and it is recommended to use the alternative version
recommended by the compile-time deprecation warning.

.. rubric:: Deprecated variadic syntax 

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_value(config_param param, ...);

   }

This version supports all values of ``param`` corresponding to a writable
configuration parameter. The variadic argument list must contain a unique
element. When reading the latter (after default argument promotions of variadic
arguments, if applicable), oneMath *assumes* that it is

- an ``std::int64_t`` value if ``param`` is any of ``config_param::NUMBER_OF_TRANSFORMS``, ``config_param::FWD_DISTANCE``, or ``config_param::BWD_DISTANCE``;
- a ``double`` value if ``param`` is any of ``FORWARD_SCALE``, ``BACKWARD_SCALE``;
- a ``config_value`` value if ``param`` is any of ``config_param::COMPLEX_STORAGE``, ``config_param::PLACEMENT``, or ``config_param::WORKSPACE_PLACEMENT``;
- an ``std::int64_t*`` value (address of the first of :math:`\left(d + 1\right)` contiguous ``std::int64_t`` values) if ``param`` is any of ``config_param::FWD_STRIDES``, ``config_param::BWD_STRIDES``, ``config_param::INPUT_STRIDES``, or ``config_param::OUTPUT_STRIDES``.

This variadic function is deprecated; it may emit runtime deprecation warnings
to inform about the recommended alternative.

.. container:: section

   .. rubric:: Input Parameters

   ``param``
      One of the possible values of type ``config_param`` representing the
      (writable) configuration parameter to be set.

   ``value``
      The value to be set for the targeted configuration parameter. The type of
      this input argument depends on ``param`` as specified above.

   ``...``
      The value to be set for the targeted configuration parameter, passed as
      a variadic argument list of **one** element. This usage is deprecated.
      Note the type assumed by oneMath when reading that value (specified above).

.. container:: section

   .. rubric:: Throws

   The ``set_value`` member functions shall throw the following
   :ref:`exceptions<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:

   ``oneapi::math::invalid_argument()``
      - If the provided ``param`` corresponds to a read-only configuration parameter;
      - If the overloaded version being used does not support ``param``;
      - If the provided ``param`` and/or configuration value are/is not valid.

   ``oneapi::math::unimplemented()``
      If the provided ``param`` and configuration value are valid, but not
      supported by the library implementation.

**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`


.. _onemath_dft_descriptor_get_value:

``get_value`` member functions
++++++++++++++++++++++++++++++

The ``get_value`` member functions of any ``descriptor`` object query the
configuration value corresponding to a configuration parameter for the DFT that
it defines. The ``get_value`` member functions do not modify the calling object.

These functions require and expect exactly **two** arguments: they return the
configuration value (into the element pointed by the second argument)
corresponding to the queried configuration parameter (first argument) ``param``
of type ``config_param``. The second argument is a valid *pointer* to a
configuration value whose type corresponds to ``param``, as specified in the
:ref:`section<onemath_dft_enum_config_param>` dedicated to the ``config_param``
type and its enumerators (unless a deprecated version is used). The expected
type of configuration value determines which of the ``get_value`` overloads is
to be used for a specific value of ``param``.

.. note::
   When querying the value associated with a writable configuration parameter,
   the returned value corresponds to the latest value that was set, even if it
   was set after committing the descriptor. If the value was never set
   explicitly, the corresponding default value is returned.

.. rubric:: Syntax for querying the kind of forward domain

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, domain* value_ptr) const;
   }

This version of ``get_value`` supports only ``config_param::FORWARD_DOMAIN`` for
``param``.

.. rubric:: Syntax for querying the considered floating-point format

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, precision* value_ptr) const;
   }

This version of ``get_value`` supports only ``config_param::PRECISION`` for
``param``.

.. rubric:: Syntax for integer-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, std::int64_t* value_ptr) const;
   }

This version of ``get_value`` supports the following values of ``param``:

- ``config_param::NUMBER_OF_TRANSFORMS``;
- ``config_param::FWD_DISTANCE``;
- ``config_param::BWD_DISTANCE``;
- ``config_param::DIMENSION``;
- ``config_param::WORKSPACE_EXTERNAL_BYTES`` (requires the calling object to be committed);
- ``config_param::LENGTHS`` (deprecated usage if :math:`d > 1`, :math:`d` contiguous ``std::int64_t`` written by oneMath)
- ``config_param::INPUT_STRIDES`` (deprecated usage, :math:`\left(d+1\right)` contiguous ``std::int64_t`` written by oneMath);
- ``config_param::OUTPUT_STRIDES`` (deprecated usage, :math:`\left(d+1\right)` contiguous ``std::int64_t`` written by oneMath);
- ``config_param::FWD_STRIDES`` (deprecated usage, :math:`\left(d+1\right)` contiguous ``std::int64_t`` written by oneMath);
- ``config_param::BWD_STRIDES`` (deprecated usage, :math:`\left(d+1\right)` contiguous ``std::int64_t`` written by oneMath);

Using this version for querying configuration values encapsulating more than one
``std::int64_t`` values is deprecated. A runtime deprecation warning may be
emitted to inform about the recommended alternative in such cases.

.. rubric:: Syntax for real-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, real_scalar_t* value_ptr) const;
   }

This version of ``get_value`` supports the following values of ``param``:

- ``config_param::FORWARD_SCALE``;
- ``config_param::BACKWARD_SCALE``.

Note that ``real_scalar_t`` is defined as ``float`` (resp. ``double``) for
single-precision (resp. double-precision) descriptors.

.. rubric:: Syntax for vector-valued parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, std::vector<std::int64_t>* value_ptr) const;
   }

This version of ``get_value`` supports the following values of ``param``:

- ``config_param::NUMBER_OF_TRANSFORMS`` (requires ``value_ptr->size() == 1``);
- ``config_param::FWD_DISTANCE`` (requires ``value_ptr->size() == 1``);
- ``config_param::BWD_DISTANCE`` (requires ``value_ptr->size() == 1``);
- ``config_param::DIMENSION`` (requires ``value_ptr->size() == 1``);
- ``config_param::WORKSPACE_EXTERNAL_BYTES`` (requires ``value_ptr->size() == 1``);
- ``config_param::LENGTHS`` (requires ``value_ptr->size() == d``);
- ``config_param::INPUT_STRIDES`` (requires ``value_ptr->size() == d + 1``);
- ``config_param::OUTPUT_STRIDES`` (requires ``value_ptr->size() == d + 1``);
- ``config_param::FWD_STRIDES`` (requires ``value_ptr->size() == d + 1``);
- ``config_param::BWD_STRIDES`` (requires ``value_ptr->size() == d + 1``).

.. rubric:: Syntax for other non-numeric parameters

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, config_value* value_ptr) const;
   }

This version of ``get_value`` supports the following values of ``param``:

- ``config_param::COMMIT_STATUS``;
- ``config_param::COMPLEX_STORAGE``;
- ``config_param::PLACEMENT``;
- ``config_param::WORKSPACE_PLACEMENT``.

.. rubric:: Deprecated variadic syntax

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::get_value(config_param param, ...) const;

   }

This version supports all values of ``param``. The variadic argument list must
contain a unique element. When reading the latter (after default argument
promotions of variadic arguments, if applicable), oneMath assumes that it is of
type

- ``domain*`` if ``param`` is ``config_param::FORWARD_DOMAIN``;
- ``precision*`` if ``param`` is ``config_param::PRECISION``;
- ``std::int64_t*`` if ``param`` is any of ``config_param::NUMBER_OF_TRANSFORMS``, ``config_param::FWD_DISTANCE``, ``config_param::BWD_DISTANCE``, ``config_param::DIMENSION``, ``config_param::WORKSPACE_EXTERNAL_BYTES``, ``config_param::LENGTHS``, ``config_param::INPUT_STRIDES``, ``config_param::OUTPUT_STRIDES``, ``config_param::FWD_STRIDES``, or ``config_param::BWD_STRIDES``;
- ``float*`` (resp. ``double*``) if ``param`` is any of ``config_param::FORWARD_SCALE`` or ``config_param::BACKWARD_SCALE``, for single-precision (resp. double-precision) descriptors;
- ``config_value*`` if ``param`` is any of ``config::param::COMMIT_STATUS``, ``config::param::COMPLEX_STORAGE``, ``config::param::PLACEMENT``, or ``config::param::WORKSPACE_PLACEMENT``.

This variadic function is deprecated and behaves as if redirecting to the
overloaded non-variadic overloaded alternative (possibly deprecated itself) that
is consistent with that assumed type. It may emit runtime deprecation warnings
to inform about the recommended alternative.

.. container:: section

   .. rubric:: Input Parameters

   ``param``
      One of the possible values of type ``config_param`` representing the
      configuration parameter being queried.

.. container:: section

   .. rubric:: Output Parameters

   ``value_ptr``
      A valid *pointer* to a configuration value (or configuration values) in
      which oneMath is allowed to write (return) the queried value(s). The type
      of this input argument depends on ``param`` as specified above.

   ``...``
      A valid *pointer* to a configuration value (or configuration values),
      passed as a variadic argument list of **one** element. This usage is
      deprecated. Note the type assumed by oneMath when accessing that pointer
      (specified above)

.. container:: section

   .. rubric:: Throws

   The ``get_value`` member functions shall throw the following
   :ref:`exceptions<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:
   
   ``oneapi::math::invalid_argument()``
      - If the overloaded version being used does not support ``param``;
      - If ``value_ptr`` is ``nullptr``;
      - If ``value_ptr->size()`` is not as expected when querying a vector-valued parameter.

   ``oneapi::math::uninitialized``
      If ``param`` is ``config_param::WORKSPACE_EXTERNAL_BYTES`` and the
      calling object is not committed.

   ``oneapi::math::unimplemented()``
      If the queried ``param`` is valid, but not supported by the library
      implementation.

**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`

.. _onemath_dft_descriptor_set_workspace:

``set_workspace`` member function
+++++++++++++++++++++++++++++++++

The ``set_workspace`` member function of any ``descriptor`` object sets the
workspace (possible additional memory required by the object for computation
purposes) to use when computing DFTs.

This function may only be called after the ``descriptor`` object has been
committed. The size of the provided workspace must be equal to or larger than
the required workspace size, *i.e.*, the configuration value associated with
``config_param::WORKSPACE_EXTERNAL_BYTES`` (queryable via the ``get_value``
member function for integer-valued parameters).

A ``descriptor`` object where ``config_value::WORKSPACE_EXTERNAL`` is specified
for  ``config_param::WORKSPACE_PLACEMENT`` is not a valid object for compute
calls until its workspace has been successfully set using this member function.

The type of workspace must match the compute calls for which it is used.
That is, if the workspace is provided as a ``sycl::buffer``, the compute
calls must also use ``sycl::buffer`` for their arguments. Likewise, a USM
allocated workspace must only be used with USM compute calls.
Failing to do this will result in an invalid descriptor for compute calls.

If the workspace is a USM allocation, the user must not use it for other purposes
in parallel whilst the DFT ``compute_forward`` or ``compute_backward`` are in progress.

This function can be called on committed descriptors where the workspace placement
is not ``config_value::WORKSPACE_EXTERNAL``. The provided workspace may or may not
be used in compute calls. However, the aforementioned restrictions will still apply.

.. rubric:: Syntax (buffer workspace)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_workspace(sycl::buffer<real_scalar_t, 1> &workspaceBuf);
   }

.. rubric:: Syntax (USM workspace)

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::set_workspace(real_scalar_t* workspaceUSM);

   }

.. container:: section

   .. rubric:: Input Parameters

   ``workspaceBuf``
      A workspace buffer where ``real_scalar_t`` is the floating-point type
      according to ``prec``. This buffer must be sufficiently large or an
      exception will be thrown. A sub-buffer cannot be used.

   ``workspaceUSM``
      A workspace USM allocation where ``real_scalar_t`` is the floating-point
      type according to ``prec``. This allocation must be accessible on the
      device on which the descriptor is committed. It is assumed that this USM
      allocation is sufficiently large. The pointer is expected to be aligned
      to ``real_scalar_t``.

.. container:: section

   .. rubric:: Throws

   The ``set_workspace`` member function shall throw the following
   :ref:`exceptions<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here:
   
   ``oneapi::math::invalid_argument()``
      If the provided buffer ``workspaceBuf`` is not sufficiently large or is a
      sub-buffer, or if the provided USM allocation ``workspaceUSM`` is
      ``nullptr`` when an external workspace of size greater than zero is
      required, or if the provided USM allocation ``workspaceUSM`` is not
      accessible by the device.

   ``oneapi::math::uninitialized()``
      If ``set_workspace`` is called before the descriptor is committed.


**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`

.. _onemath_dft_descriptor_commit:

``commit`` member function
++++++++++++++++++++++++++

The ``commit`` member function commits a ``descriptor`` object to the DFT
calculations it defines consistently with its configuration settings, by
completing all the initialization work (*e.g.*, algorithm selection, algorithm
tuning, choice of factorization, memory allocations, calculation of twiddle
factors, etc.) required by the chosen implementation for the desired DFT(s) on
the targeted device. Objects of any ``descriptor`` class **must** be committed
prior to using them in any call to ``compute_forward`` or ``compute_backward``
(which trigger actual DFT calculations).

As specified :ref:`above<onemath_dft_descriptor_set_value>`, all required
configuration parameters must be set before this function is called. Any change
in configuration operated on a ``descriptor`` object via a call to its
``set_value`` member function *after* it was committed results in an undefined
state not suitable for computation until this ``commit`` member function is
called again.

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::math::dft {

      template <precision prec, domain dom>
      void descriptor<prec,dom>::commit(sycl::queue& queue);
   }

.. container:: section

   .. rubric:: Input Parameters

   ``queue``
      Valid ``sycl::queue`` object to which the operations relevant to the
      desired DFT(s) are to be enqueued.

.. container:: section

   .. rubric:: Throws

   The ``commit`` member function shall throw the following
   :ref:`exceptions<onemath_common_exceptions>` if the associated condition is
   detected. An implementation may throw additional implementation-specific
   exception(s) in case of error conditions not covered here (if the
   ``descriptor`` object's configuration was found to be inconsistent, for
   instance):
   
   ``oneapi::math::invalid_argument()``
      If ``queue`` is found to be invalid in any way.

   ``oneapi::math::host_bad_alloc()``
      If any host side only memory allocations fail, for instance due to lack of
      memory.

   ``oneapi::math::device_bad_alloc()``
      If any device or shared memory allocation fail.
 
**Descriptor class member table:** :ref:`onemath_dft_descriptor_member_table`

**Parent topic:** :ref:`onemath_dft`
