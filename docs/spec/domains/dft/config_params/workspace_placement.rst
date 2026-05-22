.. SPDX-FileCopyrightText: Codeplay Software
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemath_dft_config_workspace_placement:

Workspace placement
-------------------

The usage of prepended namespace specifiers ``oneapi::math::dft`` is
omitted below for conciseness.

DFT implementations often require temporary storage for intermediate data whilst computing DFTs.
This temporary storage is referred to as a *workspace*.
Whilst ``descriptor`` objects manage their own workspace automatically by default
(``config_value::WORKSPACE_AUTOMATIC`` set for ``config_param::WORKSPACE_PLACEMENT``),
it may be preferable to provide them with an external workspace
(``config_value::WORKSPACE_EXTERNAL`` set for ``config_param::WORKSPACE_PLACEMENT``)
for the following reasons:

* to reduce the number of mallocs / frees;
* to reduce memory consumption.

For some backends and configurations, externally-managed workspaces may reduce
performance.

A typical workflow for using externally-managed workspaces is given
:ref:`below<onemath_dft_typical_usage_of_workspace_external>`.

Automatically- and externally-managed workspaces
++++++++++++++++++++++++++++++++++++++++++++++++

For any ``descriptor`` object, its configuration value associated with
configuration parameter ``config_param::WORKSPACE_PLACEMENT`` determines whether
it manages its workspace automatically (configuration value
``config_value::WORKSPACE_AUTOMATIC``) or if it is to be provided with an
external workspace (configuration value ``config_value::WORKSPACE_EXTERNAL``).

.. _onemath_dft_config_value_workspace_automatic:

.. rubric:: Automatically-managed workspace

The default value for the ``config_param::WORKSPACE_PLACEMENT`` is ``config_value::WORKSPACE_AUTOMATIC``. 

When set to ``config_value::WORKSPACE_AUTOMATIC`` the user does not need to provide an external workspace. The workspace will be automatically managed by the backend library.

.. _onemath_dft_config_value_workspace_external:

.. rubric:: Externally-managed workspace

The configuration parameter ``config_param::WORKSPACE_PLACEMENT`` can be set to
``config_value::WORKSPACE_EXTERNAL`` to allow the workspace to be set manually. 

When a descriptor is committed with ``config_value::WORKSPACE_EXTERNAL`` set 
for ``config_param::WORKSPACE_PLACEMENT``, the user must provide an external 
workspace before calling any compute function.
More details can be found in the section dedicated to the ``set_workspace``
:ref:`member function<onemath_dft_descriptor_set_workspace>` and in the typical
usage illustrated :ref:`below<onemath_dft_typical_usage_of_workspace_external>`.

.. _onemath_dft_typical_usage_of_workspace_external:

Typical usage of externally-managed workspaces
++++++++++++++++++++++++++++++++++++++++++++++

The usage of ``config_value::WORKSPACE_EXTERNAL`` typically involves the
following order of operations:

#. ``config_value::WORKSPACE_EXTERNAL`` is set for the uncommitted descriptor's ``config_param::WORKSPACE_PLACEMENT``.
#. The descriptor is committed.
#. The required workspace size is queried.
#. A workspace of sufficient size is provided to the descriptor.
#. Compute functions following the type of external workspace provided are called.
#. The user is responsible for freeing the external workspace.

This is shown in the following example code:

.. code-block:: cpp

   namespace dft = oneapi::math::dft;
   // Create a descriptor
   dft::descriptor<dft::precision::SINGLE, dom> desc(n);
   // 1. Set the workspace placement to WORKSPACE_EXTERNAL
   desc.set_value(dft::config_param::WORKSPACE_PLACEMENT,
                  dft::config_value::WORKSPACE_EXTERNAL);
   // Set further configuration parameters
   // ...
   // 2. Commit the descriptor
   desc.commit(myQueue);
   // 3. Query the required workspace size
   std::int64_t workspaceBytes{0};
   desc.get_value(dft::config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes);
   // Obtain a sufficiently large USM allocation or buffer. For this example, a USM allocation is used.
   float* workspaceUsm = sycl::malloc_device<float>(workspaceBytes / sizeof(float), myQueue);
   // 4. Set the workspace
   desc.set_workspace(workspaceUsm);
   // 5. Now USM compute functions can be called.


**Parent topic:** :ref:`onemath_dft_enums`

