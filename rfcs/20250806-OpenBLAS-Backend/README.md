# Design Document

### Revision


|Date       |Revision | Comments                                                                 |
|-----------|---------|--------------------------------------------------------------------------|
|  20250806 |  1.0    | Initial version                                                          |


## Motivation

We aim to integrate OpenBLAS as a backend for the BLAS domain in the oneAPI Math Library (oneMath) to provide a high performance, open source alternative to vendor specific BLAS implementations. The integration will target CPU devices on x86 and ARM architectures, ensuring optimized performance across supported platforms.

### What problem do we solve?

Currently, oneMath employs Intel oneMKL for x86 platforms and Arm Performance Libraries (ArmPL) for ARM architectures, both of which deliver high performance but are tied to specific hardware ecosystems. Netlib BLAS is also supported as a fallback backend; however, it primarily serves as a reference and lacks the performance tuning required for modern CPUs. To improve performance consistency and broaden platform support, we propose integrating OpenBLAS as an additional backend. OpenBLAS is a widely adopted, open-source BLAS library with architecture-aware optimizations for both x86 and ARM CPUs. Its integration will strengthen oneMath’s portability and performance across diverse CPU environments.

### What is the user impact if we don't do this?

Users will face reduced performance, limited portability, and continued reliance on proprietary or unoptimized libraries, restricting oneMath’s broader adoption across diverse platforms.

### What is the timeline?

Implementation, build system integration, and testing are targeted for completion within 6 months, with initial focus on BLAS functionality and support across ParaS, AdaptiveCpp, and DPC++ Compilers.


## Outline

1. [Introduction](#introduction)
2. [Proposal](#proposal)
3. [Changes in the Project](#changes-in-the-project)
4. [Open questions](#open-questions)

## Introduction

The oneAPI Math Library (oneMath) provides a unified interface for high-performance mathematical operations across heterogeneous platforms. It currently supports multiple BLAS backends, including oneMKL for x86 architectures, ArmPL for ARM platforms, and Netlib BLAS as a reference implementation. While these backends enable core functionality, their limitations restrict oneMath’s broader applicability and performance portability.
Vendor-specific libraries like oneMKL and ArmPL offer strong performance but are tightly coupled to specific hardware and may not be freely available or suitable for open-source or cross-platform deployments. Netlib BLAS, though portable, is not optimized for modern processors and fails to deliver competitive performance in real-world HPC applications. As a result, users working on diverse CPU architectures face a trade-off between performance and portability.
To overcome these challenges, this proposal introduces OpenBLAS as an additional backend for the BLAS domain. OpenBLAS is an open-source, highly optimized library that supports a wide range of architectures, including x86 and ARM, offering a practical and performant alternative to existing backends. Its integration into oneMath will enhance cross-platform performance, reduce reliance on vendor-specific solutions, and improve the overall flexibility of the math library.

## Proposal

This proposal outlines the integration of OpenBLAS as a backend for the oneAPI Math Library (oneMath). The primary objective is to enhance oneMath’s modular and extensible design by enabling support for OpenBLAS through a SYCL-compatible interface. This integration aims to improve performance portability across diverse CPU architectures by leveraging OpenBLAS’s optimized implementations while preserving a unified, vendor-agnostic programming model.
The proposed integration introduces OpenBLAS support within the oneMath library through a structured enhancement of its interface and implementation layers. The key steps in this process include:

+ **Wrapper Header Definition:** 
New headers will be added under the include/ directory to declare OpenBLAS-specific interfaces. These headers will conform to the existing oneMath API structure and act as the entry point for invoking OpenBLAS-based functionality.

+ **Compile-Time Interface Binding:**
The compile-time dispatch layer will be updated to route mathematical API calls directly to OpenBLAS implementations. This ensures type-safe, efficient linkage without runtime overhead, adhering to oneMath’s modular dispatch model.

+ **SYCL-Compatible Wrapper Implementation:**
The src/ directory will contain SYCL-compliant wrappers that internally call OpenBLAS routines. These wrappers will translate SYCL buffer abstractions into raw pointers compatible with OpenBLAS, enabling transparent use within SYCL applications.

+ **CMake Build System Integration:**
The build configuration will be extended to detect and optionally enable the OpenBLAS backend. This includes defining CMake options, linking rules, and dependency resolution to support seamless integration into the oneMath build process.

+ **Unit Test Integration:**
The testing infrastructure will be expanded to validate the correctness of OpenBLAS-based implementations. Tests will be incorporated into the existing functional and unit test pipelines to ensure compatibility and maintain overall reliability.

## Changes in the Project

### Header-Level Integration:

New headers will be introduced to declare SYCL-facing function prototypes for the OpenBLAS backend. These will follow oneMath’s modular design principles and enable seamless dispatch integration through backend-specific routing support.

### Backend Implementation:

The backend layer will implement wrapper functions that interface between SYCL-accessible data and native OpenBLAS routines. These implementations will ensure compatibility with SYCL compilers by structuring operations to support device execution and maintain portability across platforms.

### Build System Enhancements:

The CMake build infrastructure will be extended to support OpenBLAS as an optional backend. This includes introducing a dedicated build flag, defining linkage behavior, and integrating a mechanism to locate OpenBLAS during configuration time to support flexible deployment.

### Testing and Validation:

The test infrastructure will be updated to validate OpenBLAS backend functionality. This includes enabling conditional compilation, adjusting test linkage, and ensuring dispatch logic is exercised correctly through existing unit test workflows.


## Timeline

The proposed work—including implementation, integration, build configuration, and testing—is expected to be completed within six months.

## Examples

The proposed integration of OpenBLAS does not introduce any changes to the user-facing API. Developers continue to use oneMath's standardized SYCL interface for linear algebra operations without requiring modifications to their application code.
For instance, general matrix-matrix multiplication (GEMM) can be performed using the following call:

```bash
gemm_done = oneapi::math::blas::column_major::gemm(
    main_queue, transA, transB, m, n, k, alpha,
    dev_A, ldA, dev_B, ldB, beta, dev_C, ldC);
```

This interface remains consistent regardless of the underlying backend. When OpenBLAS is enabled and applicable, it is transparently leveraged to execute the computation, allowing users to benefit from optimized performance while preserving portability.

## User Impact 

### Portability 

 SYCL applications can leverage OpenBLAS as a backend without requiring modifications to existing user code, ensuring consistent behavior across supported platforms.

### Performance

 The integration benefits from OpenBLAS’s highly optimized CPU kernels, including support for advanced vector instruction sets such as SVE, AVX2, and AVX-512.

### Modularity

 OpenBLAS support can be enabled through a configurable option within the oneMath build system, promoting clean separation of concerns and backend extensibility.

