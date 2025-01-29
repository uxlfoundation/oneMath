<img src="https://github.com/uxlfoundation/artwork/blob/main/foundation/uxl-foundation-logo-horizontal-color.png" alt="UXL Foundation Logo" width="250"/>

# oneAPI Math Library (oneMath)

oneMath is an open-source implementation of the [oneMath specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/). It can work with multiple devices using multiple libraries (backends) underneath. The oneMath project was previously referred to as oneMKL Interfaces.

oneMath is part of the [UXL Foundation](http://www.uxlfoundation.org).
<br/><br/>

<table>
    <thead>
        <tr align="center" >
            <th>User Application</th>
            <th>oneMath Layer</th>
            <th>Third-Party Library</th>
            <th>Hardware Backend</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=15 align="center">oneMath</td>
            <td rowspan=15 align="center">oneMath selector</td>
            <td align="center"><a href="https://software.intel.com/en-us/oneapi/onemkl">Intel(R) oneAPI Math Kernel Library (oneMKL)</a></td>
            <td align="center">x86 CPU, Intel GPU</td>
        </tr>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cublas"> NVIDIA cuBLAS</a></td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cusolver"> NVIDIA cuSOLVER</a></td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/curand"> NVIDIA cuRAND</a></td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cufft"> NVIDIA cuFFT</a></td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://developer.nvidia.com/cusparse"> NVIDIA cuSPARSE</a></td>
            <td align="center">NVIDIA GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://www.netlib.org"> NETLIB LAPACK</a> </td>
            <td align="center">x86 and aarch64 CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://www.arm.com/products/development-tools/server-and-hpc/allinea-studio/performance-libraries">Arm Performance Libraries</a></td>
            <td align="center">aarch64 CPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://rocblas.readthedocs.io/en/rocm-4.5.2/"> AMD rocBLAS</a></td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocSOLVER"> AMD rocSOLVER</a></td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocRAND"> AMD rocRAND</a></td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocFFT"> AMD rocFFT</a></td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/ROCmSoftwarePlatform/rocSPARSE"> AMD rocSPARSE</a></td>
            <td align="center">AMD GPU</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/uxlfoundation/generic-sycl-components/tree/main/onemath/sycl/blas"> generic SYCL BLAS </a></td>
            <td align="center">x86 CPU, Intel GPU, NVIDIA GPU, AMD GPU, Other SYCL devices (unsupported)</td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/codeplaysoftware/portFFT"> portFFT </a></td>
            <td align="center">x86 CPU, Intel GPU, NVIDIA GPU, AMD GPU, Other SYCL devices (unsupported)</td>
        </tr>
    </tbody>
</table>


## Table of Contents

- [Support and Requirements](#support-and-requirements)
- [Documentation](#documentation)
- [FAQs](#faqs)
- [Legal Information](#legal-information)

---

## Support and Requirements

### Supported Usage Models:

#### Host API

There are two oneMath selector layer implementations:

- **Run-time dispatching**: The application is linked with the oneMath library and the required backend is loaded at run-time based on device vendor (all libraries should be dynamic).

  Example of app.cpp with run-time dispatching:
  
  ```cpp
  #include "oneapi/math.hpp"
  
  ...
  cpu_dev = sycl::device(sycl::cpu_selector());
  gpu_dev = sycl::device(sycl::gpu_selector());
  
  sycl::queue cpu_queue(cpu_dev);
  sycl::queue gpu_queue(gpu_dev);
  
  oneapi::math::blas::column_major::gemm(cpu_queue, transA, transB, m, ...);
  oneapi::math::blas::column_major::gemm(gpu_queue, transA, transB, m, ...);
  ```
  How to build an application with run-time dispatching:
  
  if OS is Linux, use icpx compiler. If OS is Windows, use icx compiler.
  Linux example:
  ```cmd
  $> icpx -fsycl –I$ONEMATH/include app.cpp
  $> icpx -fsycl app.o –L$ONEMATH/lib –lonemath
  ```

- **Compile-time dispatching**: The application uses a templated backend selector API where the template parameters specify the required backends and third-party libraries and the application is linked with the required oneMath backend wrapper libraries (libraries can be static or dynamic).

  Example of app.cpp with compile-time dispatching:
  
  ```cpp
  #include "oneapi/math.hpp"
  
  ...
  cpu_dev = sycl::device(sycl::cpu_selector());
  gpu_dev = sycl::device(sycl::gpu_selector());
  
  sycl::queue cpu_queue(cpu_dev);
  sycl::queue gpu_queue(gpu_dev);
  
  oneapi::math::backend_selector<oneapi::math::backend::mklcpu> cpu_selector(cpu_queue);
  
  oneapi::math::blas::column_major::gemm(cpu_selector, transA, transB, m, ...);
  oneapi::math::blas::column_major::gemm(oneapi::math::backend_selector<oneapi::math::backend::cublas> {gpu_queue}, transA, transB, m, ...);
  ```
  How to build an application with compile-time dispatching:
  
  ```cmd
  $> clang++ -fsycl –I$ONEMATH/include app.cpp
  $> clang++ -fsycl app.o –L$ONEMATH/lib –lonemath_blas_mklcpu –lonemath_blas_cublas
  ```
  
*Refer to [Selecting a Compiler](https://uxlfoundation.github.io/oneMath/selecting_a_compiler.html) for the choice between `icpx/icx` and `clang++` compilers.*

#### Device API

Header-based and backend-independent Device API can be called within ```sycl kernel``` or work from Host code ([device-rng-usage-model-example](https://spec.oneapi.io/versions/latest/elements/oneMath/source/domains/rng/device_api/device-rng-usage-model.html#id2)). Currently, the following domains support the Device API:

- **RNG**. To use RNG Device API functionality it's required to include ```oneapi/math/rng/device.hpp``` header file.

### Supported Configurations:

Supported domains include: BLAS, LAPACK, RNG, DFT, SPARSE_BLAS

Supported compilers include:
- [Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler): Intel proprietary compiler that supports CPUs and Intel GPUs. Intel(R) oneAPI DPC++ Compiler will be referred to as "Intel DPC++" in the "Supported Compiler" column of the tables below.
- [oneAPI DPC++ Compiler](https://github.com/intel/llvm): Open source compiler that supports CPUs and Intel, NVIDIA, and AMD GPUs. oneAPI DPC++ Compiler will be referred to as "Open DPC++" in the "Supported Compiler" column of the tables below.
- [AdaptiveCpp Compiler](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly known as hipSYCL): Open source compiler that supports CPUs and Intel, NVIDIA, and AMD GPUs.</br>**Note**: The source code and some documents in this project still use the previous name hipSYCL during this transition period.

#### Linux*

<table>
    <thead>
        <tr align="center" >
            <th>Domain</th>
            <th>Backend</th>
            <th>Library</th>
            <th>Supported Compiler</th>		
            <th>Supported Link Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=12 align="center">BLAS</td>
            <td rowspan=3 align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NETLIB LAPACK</td>
            <td align="center">Intel DPC++</br>Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">generic SYCL BLAS</td>
            <td align="center">Intel DPC++</br>Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">aarch64 CPU</td>
            <td align="center">Arm Performance Libraries</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NETLIB LAPACK</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">generic SYCL BLAS</td>
            <td align="center">Intel DPC++</br>Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuBLAS</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">generic SYCL BLAS</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">AMD GPU</td>
            <td align="center">AMD rocBLAS</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">generic SYCL BLAS</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=1 align="center">Other SYCL devices (unsupported)</td>
            <td align="center">generic SYCL BLAS</td>
            <td align="center">Intel DPC++</br>Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=5 align="center">LAPACK</td>
            <td align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">aarch64 CPU</td>
            <td align="center">Arm Performance Libraries</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuSOLVER</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocSOLVER</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">RNG</td>
            <td align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuRAND</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocRAND</td>
            <td align="center">Open DPC++</br>AdaptiveCpp</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=9 align="center">DFT</td>
            <td rowspan=2 align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">portFFT (<a href="https://github.com/codeplaysoftware/portFFT#supported-configurations">limited API support</a>)</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">portFFT (<a href="https://github.com/codeplaysoftware/portFFT#supported-configurations">limited API support</a>)</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuFFT</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">portFFT (<a href="https://github.com/codeplaysoftware/portFFT#supported-configurations">limited API support</a>)</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">AMD GPU</td>
            <td align="center">AMD rocFFT</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">portFFT (<a href="https://github.com/codeplaysoftware/portFFT#supported-configurations">limited API support</a>)</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=1 align="center">Other SYCL devices (unsupported)</td>
            <td align="center">portFFT</td>
            <td align="center">Open DPC++</br>Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=4 align="center">SPARSE_BLAS</td>
            <td align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NVIDIA GPU</td>
            <td align="center">NVIDIA cuSPARSE</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">AMD GPU</td>
            <td align="center">AMD rocSPARSE</td>
            <td align="center">Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
    </tbody>
</table>

#### Windows*

<table>
    <thead>
        <tr align="center" >
            <th>Domain</th>
            <th>Backend</th>
            <th>Library</th>
            <th>Supported Compiler</th>	
            <th>Supported Link Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3 align="center">BLAS</td>
            <td rowspan=2 align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">NETLIB LAPACK</td>
            <td align="center">Intel DPC++</br>Open DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">LAPACK</td>
            <td align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td rowspan=2 align="center">RNG</td>
            <td align="center">x86 CPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
        <tr>
            <td align="center">Intel GPU</td>
            <td align="center">Intel(R) oneMKL</td>
            <td align="center">Intel DPC++</td>
            <td align="center">Dynamic, Static</td>
        </tr>
    </tbody>
</table>

---

### Hardware Platform Support

- CPU
    - Intel Atom(R) Processors
    - Intel(R) Core(TM) Processor Family
    - Intel(R) Xeon(R) Processor Family
    - Arm Neoverse Processor Family (tested on N1, V1, V2)
- Accelerators
    - Intel(R) Arc(TM) A-Series Graphics
    - Intel(R) Data Center GPU Max Series
    - NVIDIA(R) A100 (Linux* only)
    - AMD(R) GPUs see [here](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support) tested on AMD Vega 20 (gfx906)
    - Other SYCL devices can be used, but are not supported
    
---
### Supported Operating Systems

#### Linux*

Backend | Supported Operating System
:--- | :---
x86 CPU    | Red Hat Enterprise Linux* 9 (RHEL* 9)
aarch64 CPU| Red Hat Enterprise Linux* 9 (RHEL* 9)
Intel GPU  | Ubuntu 24.04 LTS
NVIDIA GPU | Ubuntu 22.04 LTS

#### Windows*

Backend | Supported Operating System
:--- | :---
x86 CPU   | Microsoft Windows* Server 2022
Intel GPU | Microsoft Windows* 11
---

### Software Requirements

**What should I download?**

#### General:
<table>
    <thead>
        <tr align="center">
            <th> Functional Testing </th>
            <th> Build Only </th>
            <th>Documentation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=3 align=center> CMake (version 3.13 or newer) </td>
        </tr>
        <tr>
            <td colspan=3 align=center> Linux* : GNU* GCC 5.1 or higher <br> Windows* : MSVS* 2017 or MSVS* 2019 (version 16.5 or newer) </td>
        </tr>
        <tr>
            <tr>
                <td colspan=3 align=center> Ninja (optional) </td>
            </tr>
            <tr>
                <td> GNU* FORTRAN Compiler </td>
                <td> - </td>
                <td> Sphinx </td>
            </tr>
            <tr>
                <td> NETLIB LAPACK </td>
                <td> - </td>
                <td> - </td>
            </tr>
        </tr>
    </tbody>
</table>

#### Hardware and OS Specific:
<table>
    <thead>
        <tr align="center">
            <th>Operating System</th>
            <th>Device</th>
            <th>Package</th>
        </tr>
    </thead>
    <tbody>
        <td rowspan=5> Linux*/Windows* </td>
        <td rowspan=2> x86 CPU </td>
        <td> Intel(R) oneAPI DPC++ Compiler <br> or <br> oneAPI DPC++ Compiler </td>
        <tr>
            <td> Intel(R) oneAPI Math Kernel Library </td>
        </tr>
        <td rowspan=3> Intel GPU </td>
        <td> Intel(R) oneAPI DPC++ Compiler </td>
        <tr>
            <td> Intel GPU driver </td>
        </tr>
        <tr>
            <td> Intel(R) oneAPI Math Kernel Library </td>
        </tr>
        <td rowspan=2> Linux* only </td>
        <td> NVIDIA GPU </td>
        <td> oneAPI DPC++ Compiler <br> or <br> AdaptiveCpp with CUDA backend and dependencies </td>
        <tr>
            <td> AMD GPU </td>
            <td> oneAPI DPC++ Compiler <br> or <br> AdaptiveCpp with ROCm backend and dependencies </td>
        </tr>
    </tbody>
</table>

#### Product and Version Information:

Product | Supported Version | License
:--- | :--- | :---
[CMake](https://cmake.org/download/) | 3.13 or higher | [The OSI-approved BSD 3-clause License](https://gitlab.kitware.com/cmake/cmake/raw/master/Copyright.txt)
[Ninja](https://ninja-build.org/) | 1.10.0 | [Apache License v2.0](https://github.com/ninja-build/ninja/blob/master/COPYING)
[GNU* FORTRAN Compiler](https://gcc.gnu.org/wiki/GFortran) | 7.4.0 or higher | [GNU General Public License, version 3](https://gcc.gnu.org/onlinedocs/gcc-7.5.0/gfortran/Copying.html)
[Intel(R) oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) | Latest | [End User License Agreement for the Intel(R) Software Development Products](https://software.intel.com/en-us/license/eula-for-intel-software-development-products)
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) | Later than [2cfa530](https://github.com/AdaptiveCpp/AdaptiveCpp/commit/2cfa5303fd88b8f84e539b5bb6ed41e49c6d6118) | [BSD-2-Clause License ](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/LICENSE)
[oneAPI DPC++ Compiler binary for x86 CPU](https://github.com/intel/llvm/releases) | Daily builds | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[oneAPI DPC++ Compiler source for NVIDIA and AMD GPUs](https://github.com/intel/llvm) | Daily source releases | [Apache License v2](https://github.com/intel/llvm/blob/sycl/sycl/LICENSE.TXT)
[Intel(R) oneAPI Math Kernel Library](https://software.intel.com/en-us/oneapi/onemkl) | Latest | [Intel Simplified Software License](https://software.intel.com/en-us/license/intel-simplified-software-license)
[NVIDIA CUDA SDK](https://developer.nvidia.com/hpc-sdk) | 12.0 | [End User License Agreement](https://docs.nvidia.com/cuda/eula/index.html)
[AMD rocBLAS](https://github.com/ROCm/rocblas) | 4.5 | [AMD License](https://github.com/ROCm/rocBLAS/blob/develop/LICENSE.md)
[AMD rocRAND](https://github.com/ROCm/rocRAND) | 5.1.0 | [AMD License](https://github.com/ROCm/rocRAND/blob/develop/LICENSE.txt)
[AMD rocSOLVER](https://github.com/ROCm/rocSOLVER) | 5.0.0 | [AMD License](https://github.com/ROCm/rocSOLVER/blob/develop/LICENSE.md)
[AMD rocFFT](https://github.com/ROCm/rocFFT) | rocm-5.4.3 | [AMD License](https://github.com/ROCm/rocFFT/blob/rocm-5.4.3/LICENSE.md)
[AMD rocSPARSE](https://github.com/ROCm/rocSPARSE) | 3.1.2 | [AMD License](https://github.com/ROCm/rocSPARSE/blob/develop/LICENSE.md)
[NETLIB LAPACK](https://www.netlib.org/) | [5d4180c](https://github.com/Reference-LAPACK/lapack/commit/5d4180cf8288ae6ad9a771d18793d15bd0c5643c) | [BSD like license](http://www.netlib.org/lapack/LICENSE.txt)
[Generic SYCL BLAS](https://github.com/uxlfoundation/generic-sycl-components/tree/main/onemath/sycl/blas) | 0.1 | [Apache License v2.0](https://github.com/uxlfoundation/generic-sycl-components/blob/main/LICENSE)
[portFFT](https://github.com/codeplaysoftware/portFFT) | 0.1 | [Apache License v2.0](https://github.com/codeplaysoftware/portFFT/blob/main/LICENSE)
[Arm Performance Libraries](https://developer.arm.com/downloads/-/arm-performance-libraries) | 22.0.1 or higher | [EULA](https://developer.arm.com/downloads/-/arm-performance-libraries/eula)

---

## Documentation
- [Contents](https://uxlfoundation.github.io/oneMath/)
- [About](https://uxlfoundation.github.io/oneMath/introduction.html)
- Get Started
  - [Selecting a Compiler](https://uxlfoundation.github.io/oneMath/selecting_a_compiler.html)
  - [Building the Project with DPC++](https://uxlfoundation.github.io/oneMath/building_the_project_with_dpcpp.html)
  - [Building the Project with AdaptiveCpp](https://uxlfoundation.github.io/oneMath/building_the_project_with_adaptivecpp.html)
- Developer Reference
  - [oneMath Defined Datatypes](https://uxlfoundation.github.io/oneMath/onemath-datatypes.html)
  - [Dense Linear Algebra](https://uxlfoundation.github.io/oneMath/domains/dense_linear_algebra.html)
- [Integrating a Third-Party Library](https://uxlfoundation.github.io/oneMath/create_new_backend.html)

---

## Governance

The oneMath project is governed by the UXL Foundation and you can get involved in this project in multiple ways. It is possible to join the [Math Special Interest Group (SIG)](https://github.com/uxlfoundation/foundation/tree/main/math) meetings where the group discusses and demonstrates work using this project. Members can also join the Open Source and Specification Working Group meetings.

You can also join the mailing lists for the [UXL Foundation](https://lists.uxlfoundation.org/g/main/subgroups) to be informed of when meetings are happening and receive the latest information and discussions.

---

## Contributing

You can contribute to this project and also contribute to [the specification for this project](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/). Please read the [CONTRIBUTING](CONTRIBUTING.md) page for more information. You can also contact oneMath developers and maintainers via [UXL Foundation Slack](https://slack-invite.uxlfoundation.org/) using [#onemath](https://uxlfoundation.slack.com/archives/onemath) channel.

---

## License

Distributed under the Apache license 2.0. See [LICENSE](LICENSE) for more information.

---

## FAQs

### oneMath

**Q: What is the difference between the following items?**
   - The [oneAPI Specification for oneMath](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/)
   - The [oneAPI Math Library (oneMath)](https://github.com/uxlfoundation/oneMath) project
   - The [Intel(R) oneAPI Math Kernel Library (oneMKL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) Product

**A:**
- The [oneAPI Specification for oneMath](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemath/source/) defines the SYCL interfaces for performance math library functions. The oneMath specification can evolve faster and more frequently than implementations of the specification.

- The [oneAPI Math Library (oneMath)](https://github.com/uxlfoundation/oneMath) project is an open source implementation of the specification. The project goal is to demonstrate how the SYCL interfaces documented in the oneMath specification can be implemented for any math library and work for any target hardware. While the implementation provided here may not yet be the full implementation of the specification, the goal is to build it out over time. We encourage the community to contribute to this project and help to extend support to multiple hardware targets and other math libraries.

- The [Intel(R) oneAPI Math Kernel Library (oneMKL)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) project is an Intel product provided as part of the Intel(R) oneAPI Base Toolkit. It is used for the Intel backends of oneMath. Its C++ API is very similar to the oneMath specification. It is highly optimized for Intel CPU and Intel GPU hardware.

**Q: I'm trying to use oneMath in my project using `FetchContent`**, but I keep running into `ONEMATH::SYCL::SYCL target was not found` problem when I try to build the project. What should I do?

**A:**
Make sure you set the compiler when you configure your project.
E.g. `cmake -Bbuild . -DCMAKE_CXX_COMPILER=icpx`.

**Q: I'm trying to use oneMath in my project using `find_package(oneMath)`.** I set oneMath/oneTBB and Compiler environment first, then I built and installed oneMath, and finally I tried to build my project using installed oneMath (e.g. like this `cmake -Bbuild -GNinja -DCMAKE_CXX_COMPILER=icpx -DoneMath_ROOT=<path_to_installed_oneMath> .`) and I noticed that cmake includes installed oneMath headers as a system include which ends up as a lower priority than the installed Intel(R) oneAPI Math Kernel Library package includes which I set before for building oneMath. As a result, I get conflicts between Intel(R) oneAPI Math Kernel Library and installed oneMath headers. What should I do?

**A:**
Having installed oneMath headers as `-I` instead on system includes (as `-isystem`) helps to resolve this problem. We use `INTERFACE_INCLUDE_DIRECTORIES` to add paths to installed oneMath headers (check `oneMathTargets.cmake` in `lib/cmake` to find it). It's a known limitation that `INTERFACE_INCLUDE_DIRECTORIES` puts headers paths as system headers. To avoid that:
- Option 1: Use CMake >=3.25. In this case oneMath will be built with `EXPORT_NO_SYSTEM` property set to `true` and you won't see the issue.
- Option 2: If you use CMake < 3.25, set `PROPERTIES NO_SYSTEM_FROM_IMPORTED true` for your target. E.g: `set_target_properties(test PROPERTIES NO_SYSTEM_FROM_IMPORTED true)`.

---


#### [Legal information](legal_information.md)
