# MSCCL-EXECUTOR-NCCL

Microsoft Collective Communication Library Exector on NCCL (MSCCL-EXECUTOR-NCCL) is an inter-accelerator communication framework that is built on top of [NCCL](https://github.com/nvidia/nccl) and uses its building blocks to execute custom-written collective communication algorithms.

## Introduction

MSCCL-EXECUTOR-NCCL is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, all-to-all, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. MSCCL-EXECUTOR-NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications. To achieve this, MSCCL has multiple capabilities:

- Programmibility: Inter-connection among accelerators have different latencies and bandwidths. Therefore, a generic collective communication algorithm does not necessarily well for all topologies and buffer sizes. MSCCL-EXECUTOR-NCCL allows a user to write a hyper-optimized collective communication algorithm for a given topology and a buffer size. This is possbile through two main components: [MSCCL toolkit](https://github.com/microsoft/msccl-tools) and [MSCCL-EXECUTOR-NCCL](https://github.com/Azure/msccl-executor-nccl) (this repo). MSCCL toolkit contains a high-level DSL (MSCCLang) and a compiler which generate an IR for the MSCCL runtime (this repo) to run on the backend. MSCCL will automatically fall back to a NCCL's generic algorithm in case there is no custom algorithm. [Example](#Example) provides some instances on how MSCCL toolkit with the runtime works. Please refer to [MSCCL toolkit](https://github.com/microsoft/msccl-tools) for more information.
- Profiling: MSCCL-EXECUTOR-NCCL has a profiling tool [NPKit](https://github.com/microsoft/npkit) which provides detailed timeline for each primitive send and receive operation to understand the bottlenecks in a given collective communication algorithms.

**Please note:** MSCCL customized algorithms and NPKit only support single GPU per process mode.

## Build

To build the library :

```sh
$ git clone https://github.com/Azure/msccl.git --recurse-submodules
$ cd msccl/executor/msccl-executor-nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```sh
$ make src.build CUDA_HOME=<path to cuda install>
```

MSCCL-EXECUTOR-NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, MSCCL-EXECUTOR-NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```sh
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

## Install

To install MSCCL-EXECUTOR-NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```sh
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build MSCCL-EXECUTOR-NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```sh
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build MSCCL-EXECUTOR-NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```sh
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for MSCCL-EXECUTOR-NCCL are maintained separately at https://github.com/Azure/msccl-tests-nccl.

```sh
$ git clone https://github.com/Azure/msccl-tests-nccl.git
$ cd msccl-tests-nccl
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [CLA](https://cla.opensource.microsoft.com).

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2022-2024, Microsoft Corporation. All rights reserved.
