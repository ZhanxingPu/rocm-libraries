# GemmDq — Fused GEMM + Dequantization 算子 (RDNA3 WMMA)

## 概述

本项目在 MIOpen 框架中新增了 `GemmDq` 算子，实现了 **FP16 GEMM + UINT4 权重反量化融合** 的高性能 GPU Kernel。

**计算公式**：`C[M×N] = A[M×K] × dequant(B_packed[N×K/2])^T`

| 张量 | 数据类型 | 布局 | 说明 |
|------|----------|------|------|
| A | FP16 | col-major (M×K) | 输入激活 |
| B_packed | UINT4 packed | N×K/2 bytes | 每 byte 存 2 个 4-bit 权重 |
| scales | FP16 | N × num_groups_k | 每列每组一个 scale |
| zeros | FP16 | N × num_groups_k | 每列每组一个 zero point |
| C | FP16 | col-major (M×N) | 输出 |

**硬件要求**：AMD RDNA3+ GPU（gfx1100 / gfx1101 / gfx1102 / gfx1150 / gfx1151 / gfx12xx）

---

## 目录结构

```
rocm-libraries/
├── projects/miopen/
│   ├── include/miopen/miopen.h              # C API 声明 (miopenGemmDqForward)
│   ├── src/
│   │   ├── include/miopen/
│   │   │   ├── gemmdq.hpp                   # C++ API
│   │   │   └── gemmdq/
│   │   │       ├── problem_description.hpp  # 参数校验 + NetworkConfig
│   │   │       ├── invoke_params.hpp        # Kernel 调用参数封装
│   │   │       └── solvers.hpp             # Solver 声明
│   │   ├── gemmdq/
│   │   │   └── problem_description.cpp     # MakeNetworkConfig 实现
│   │   ├── gemmdq.cpp                       # 主实现
│   │   ├── gemmdq_api.cpp                   # C API 封装
│   │   ├── solver/gemmdq/
│   │   │   └── forward_gemmdq.cpp          # Solver (Grid/Block 配置 + Invoker)
│   │   ├── kernels/MIOpenGemmDq.cpp         # ★ GPU Kernel (JIT 编译)
│   │   ├── solver.cpp                       # (已修改) 注册 GemmDq solver
│   │   └── CMakeLists.txt                   # (已修改) 添加 4 处源文件引用
│   └── ...
└── op_examples/gemmdq/
    ├── test_gemm_dq.cpp                     # 正确性验证 + 性能测试
    ├── gen_gemm_dq_data.py                  # 测试数据生成 + NumPy 参考实现
    └── Makefile                             # 跨平台编译/运行脚本（WSL + Windows）
```

---

## 一、前置环境准备

### 1.1 安装 HIP SDK

从 [AMD ROCm/HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub.html) 下载安装。

安装完成后记录安装路径，后续用 `<HIP_SDK>` 表示，例如 `C:\AMD\Rocm\7.1`。

确认 GPU 架构（后续编译测试程序时需要）：

```powershell
# 在 PowerShell 中查看
& "<HIP_SDK>\bin\hipInfo.exe" | Select-String "gcnArchName"
# 或在 WSL 中查看
/mnt/c/AMD/ROCm/7.1/bin/hipInfo.exe 2>/dev/null | grep gcnArch
```

常见架构：`gfx1100`（RX 7900）、`gfx1150`（Radeon 890M）、`gfx1151` 等。

### 1.2 安装构建工具

| 工具 | 安装方式 | 说明 |
|------|----------|------|
| CMake ≥ 3.15 | https://cmake.org/download/ 安装 MSI，勾选加入 PATH | 编译 MIOpen 库 |
| Ninja | `pip install ninja` | CMake 构建后端 |
| GNU Make | `choco install make` | 编译和运行测试程序 |
| Python + NumPy | `pip install numpy` | 生成测试数据 |

> **WSL 是可选的**。整个流程（编译 MIOpen、编译/运行测试）均可在 Windows CMD/PowerShell 中完成。如果你更习惯 WSL，Makefile 也同样支持。

### 1.3 准备 vcpkg（MIOpen 第三方依赖管理）

在你的工作目录下（例如 `<WORKSPACE>`）：

```powershell
cd <WORKSPACE>
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

安装 MIOpen 所需的库依赖：

```powershell
.\vcpkg.exe install sqlite3:x64-windows bzip2:x64-windows nlohmann-json:x64-windows half:x64-windows boost-filesystem:x64-windows boost-system:x64-windows
```

> **⚠️ half 库路径修复**
>
> vcpkg 安装的 `half.hpp` 在 `include/half.hpp`，但 MIOpen 代码需要 `#include <half/half.hpp>`。
> 必须手动创建子目录并拷贝：
>
> ```powershell
> mkdir <WORKSPACE>\vcpkg\installed\x64-windows\include\half
> copy <WORKSPACE>\vcpkg\installed\x64-windows\include\half.hpp <WORKSPACE>\vcpkg\installed\x64-windows\include\half\half.hpp
> ```

### 1.4 准备 rocm-cmake

```powershell
cd <WORKSPACE>
git clone --depth 1 https://github.com/ROCm/rocm-cmake.git
```

### 1.5 创建 amd_comgr CMake 配置文件

HIP SDK 包含 `amd_comgr` 库但缺少 CMake 配置文件，需手动创建。

**创建文件**：`<HIP_SDK>\lib\cmake\amd_comgr\amd_comgrConfig.cmake`

```cmake
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_ROCM_ROOT "${_DIR}/../../.." ABSOLUTE)

if(NOT TARGET amd_comgr)
    add_library(amd_comgr SHARED IMPORTED)
    set_target_properties(amd_comgr PROPERTIES
        IMPORTED_LOCATION "${_ROCM_ROOT}/bin/amd_comgr_3.dll"
        IMPORTED_IMPLIB   "${_ROCM_ROOT}/lib/amd_comgr_3.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${_ROCM_ROOT}/include/amd_comgr"
    )
endif()

set(amd_comgr_FOUND TRUE)
set(amd_comgr_VERSION "3.0.0")
set(amd_comgr_VERSION_MAJOR 3)
set(amd_comgr_VERSION_MINOR 0)
set(amd_comgr_VERSION_PATCH 0)
```

> 如果 `amd_comgr` 的 dll/lib 文件名不是 `amd_comgr_3`，请根据实际文件名修改。
> 版本号 `3.0.0` 必须设置，否则 MIOpen 编译会报 `AMD COMgr older than 1.7.0 is not supported`。

---

## 二、编译 MIOpen 库（含 GemmDq 算子）

### 2.1 Clone 本仓库

```powershell
cd <WORKSPACE>
git clone <本仓库地址> rocm-libraries
```

### 2.2 映射短路径（必须）

MIOpen 的部分内核汇编文件名极长，加上源码路径后容易超过 Windows 260 字符限制。
**必须**用 `subst` 映射一个短盘符：

```powershell
subst M: "<WORKSPACE>\rocm-libraries\projects\miopen"
```

> 此映射重启后消失，每次重新开机后需重新执行。

### 2.3 CMake 配置

```powershell
mkdir M:\build -ErrorAction SilentlyContinue
cd M:\build

cmake -G "Ninja" `
  -DMIOPEN_BACKEND=HIP `
  -DCMAKE_PREFIX_PATH="<HIP_SDK>" `
  -DCMAKE_CXX_COMPILER="<HIP_SDK>/bin/clang++.exe" `
  -DCMAKE_C_COMPILER="<HIP_SDK>/bin/clang.exe" `
  -DCMAKE_TOOLCHAIN_FILE="<WORKSPACE>/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DROCmCMakeBuildTools_DIR="<WORKSPACE>/rocm-cmake/share/rocmcmakebuildtools/cmake" `
  -DMIOPEN_USE_COMPOSABLEKERNEL=OFF `
  -DMIOPEN_USE_MLIR=OFF `
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF `
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF `
  -DMIOPEN_BUILD_DRIVER=OFF `
  -DBUILD_TESTING=OFF `
  -DPKG_CONFIG_EXECUTABLE="PKG_CONFIG_EXECUTABLE-NOTFOUND" `
  -DHALF_INCLUDE_DIR="<WORKSPACE>/vcpkg/installed/x64-windows/include" `
  ..
```

**请将所有 `<HIP_SDK>` 和 `<WORKSPACE>` 替换为你的实际路径**，例如：
- `<HIP_SDK>` → `C:/AMD/Rocm/7.1`
- `<WORKSPACE>` → `C:/Users/yourname/Desktop/Workspace/Rocm`

> **各参数说明**：
>
> | 参数 | 说明 |
> |------|------|
> | `-DMIOPEN_BACKEND=HIP` | 使用 HIP 后端 |
> | `-DCMAKE_PREFIX_PATH` | HIP SDK 安装路径 |
> | `-DCMAKE_CXX_COMPILER` | HIP SDK 自带的 clang++ |
> | `-DCMAKE_TOOLCHAIN_FILE` | vcpkg 工具链（自动查找 SQLite3、BZip2 等） |
> | `-DROCmCMakeBuildTools_DIR` | rocm-cmake 模块路径 |
> | `-DMIOPEN_USE_COMPOSABLEKERNEL=OFF` | 禁用 CK（需额外编译，非必需） |
> | `-DMIOPEN_USE_MLIR=OFF` | 禁用 MLIR（Windows 下不可用） |
> | `-DMIOPEN_BUILD_DRIVER=OFF` | 禁用 MIOpenDriver（与 vcpkg half 版本不兼容） |
> | `-DPKG_CONFIG_EXECUTABLE=...NOTFOUND` | 禁用 pkg-config（Windows 路径解析有 bug） |
> | `-DHALF_INCLUDE_DIR` | half 浮点库头文件路径 |

### 2.4 编译

```powershell
cd M:\build
ninja -j8
```

首次编译约 **10-20 分钟**（~635 个编译目标），后续增量编译很快（几十秒）。

编译产物：
- `M:\build\bin\MIOpen.dll` — 动态链接库（~490 MB）
- `M:\build\lib\MIOpen.lib` — 导入库

---

## 三、编译和运行测试程序

测试文件位于本仓库的 `op_examples/gemmdq/` 目录下：

| 文件 | 说明 |
|------|------|
| `test_gemm_dq.cpp` | 正确性验证 + 性能基准测试（调用 `miopenGemmDqForward` C API） |
| `gen_gemm_dq_data.py` | 生成测试数据（随机 FP16/UINT4）+ NumPy 参考结果 |
| `Makefile` | 跨平台编译/运行（WSL 和 Windows CMD/PowerShell） |

Makefile 通过 `$(OS)` 自动检测运行环境，WSL 和 Windows CMD/PowerShell 均可直接使用。

### 3.0 前提：安装 GNU Make（仅 Windows 原生方式）

在 Windows CMD/PowerShell 中使用 Makefile 前，需先安装 GNU Make：

```powershell
# 方式一：Chocolatey（推荐）
choco install make

# 方式二：手动下载
# 从 https://gnuwin32.sourceforge.net/packages/make.htm 下载并添加到 PATH
```

> WSL 中已自带 `make`，无需额外安装。

### 3.1 修改 Makefile

打开 `op_examples/gemmdq/Makefile`，修改顶部的变量以匹配你的环境：

```makefile
# GPU 架构
OFFLOAD := --offload-arch=gfx1150

# vcpkg include 路径（默认在 rocm-libraries 同级目录，按实际情况修改）
VCPKG_INC := ../../../vcpkg/installed/x64-windows/include
```

HIP SDK 路径会根据环境自动设置默认值：

| 环境 | 默认 HIP_SDK | 覆盖方式 |
|------|-------------|----------|
| WSL | `/mnt/c/AMD/ROCm/7.1` | `make HIP_SDK=/mnt/c/AMD/ROCm/7.1` |
| Windows | `C:/AMD/ROCm/7.1` | `make HIP_SDK=C:/AMD/ROCm/7.1` |

> MIOpen 源码和 build 路径已通过相对路径（`../../projects/miopen`）自动指向仓库内的正确位置，无需手动配置。

### 3.2 编译

**WSL**：

```bash
cd /mnt/c/<WORKSPACE>/rocm-libraries/op_examples/gemmdq
make clean && make
```

**Windows CMD/PowerShell**：

```powershell
cd <WORKSPACE>\rocm-libraries\op_examples\gemmdq
make clean && make
```

> hipcc 编译时会产生大量 `__declspec(dllimport) is not supported` 警告，这是正常的，可忽略。

### 3.3 生成测试数据

```bash
# 生成默认尺寸 (128x128x128, group_size=128)
make gendata

# 自定义尺寸
make gendata SIZE=256x512x256 GS=128
```

需要 Python + NumPy（WSL 中使用 `python3`，Windows 中使用 `python`，Makefile 会自动选择）。
生成的 `.bin` 文件保存在 `data/` 子目录中。

### 3.4 运行

```bash
# 运行单个测试（需先 gendata）
make run SIZE=128x128x128 GS=128

# 自定义参数
make run_custom ARGS="2048x4096x4096 128"

# 生成数据 + 运行（一步到位）
make test SIZE=256x256x256 GS=128

# 运行全部预设测试用例（4 组小尺寸正确性验证）
make test_all
```

**约束条件**：M 必须是 128 的倍数，N 必须是 128 的倍数，K 必须是 32 的倍数。

> **运行方式差异**：WSL 中 Makefile 通过 `cmd.exe /c "set PATH=... && exe"` 运行（自动用 `wslpath` 转换路径），
> Windows 中则直接 `set PATH=... && .\exe`。用户不需要关心这些细节，只需执行 `make run` 即可。

### 3.5 预期输出

```
MIOpen GemmDq (Fused GEMM + Dequantization) Verification
========================================================
GPU: AMD Radeon(TM) 890M Graphics (arch: gfx1150)

=== Test GemmDq M=128 N=128 K=128 group_size=128 ===
  Loaded input data from data/
  Warmup... OK (1823.45 ms), iters=5
  Benchmarking (5 rounds x 5 iters)... done

  === Performance ===
  Median: 2.134 ms, 2.018 GFLOPS, 0.123 GB/s
  Range:  2.098 ~ 2.201 ms  (jitter 4.8%)

  === GPU vs Python Reference ===
  Verified 16384 elements, 0 errors
  ...
  Result: PASSED

========================================================
Overall: ALL PASSED
```

> 第一次调用时 MIOpen 需要 JIT 编译 kernel（约 500-600ms），后续调用已缓存，通常 2-3ms。

> **注意**：测试程序使用 `hipcc` 编译。MIOpen 库本身在 PowerShell 中编译（第二节），测试程序可以在 WSL 或 Windows CMD/PowerShell 中编译和运行（本节）。Windows 原生方式需要先安装 GNU Make（见 3.0 节）。

---

## 四、C API 接口

```c
#define MIOPEN_BETA_API 1    // 必须在 include 之前定义！
#include <miopen/miopen.h>

miopenStatus_t miopenGemmDqForward(
    miopenHandle_t handle,
    int M, int N, int K,         // 矩阵维度
    const void* A, int lda,      // FP16 输入矩阵 (col-major, lda >= M)
    const void* B_packed,        // UINT4 packed 权重 (N × K/2 bytes)
    const void* scales,          // FP16 scale (N × num_groups_k)
    const void* zeros,           // FP16 zero point (N × num_groups_k)
    int group_size,              // 量化分组大小 (沿 K 维)
    int num_groups_k,            // K 维分组数 (= K / group_size)
    void* C, int ldc);           // FP16 输出矩阵 (col-major, ldc >= M)
```

> **重要**：调用方必须在 `#include <miopen/miopen.h>` **之前** 定义 `#define MIOPEN_BETA_API 1`，
> 否则编译器看不到 `miopenGemmDqForward` 的声明。

---

## 五、不同 GPU 架构适配

### MIOpen 库编译

MIOpen 库编译时 **无需指定 GPU 架构**。GemmDq 的 GPU kernel 是 JIT 编译的（运行时由
MIOpen 自动检测当前 GPU 并编译），所以同一份 `MIOpen.dll` 可以在不同 RDNA3+ GPU 上运行。

### 测试程序编译

测试程序（`test_gemm_dq.cpp`）使用 `hipcc` 编译，**需要指定目标 GPU 架构**。
修改 Makefile 中的 `OFFLOAD` 变量：

```makefile
# 按你的 GPU 选择：
OFFLOAD := --offload-arch=gfx1100    # RX 7900 XT/XTX
OFFLOAD := --offload-arch=gfx1101    # RX 7800 XT / 7700 XT
OFFLOAD := --offload-arch=gfx1102    # RX 7600
OFFLOAD := --offload-arch=gfx1150    # Radeon 890M (Strix Point)
OFFLOAD := --offload-arch=gfx1151    # Radeon 880M
OFFLOAD := --offload-arch=gfx1200    # RDNA4 (Radeon RX 9070 等)
```

> 如果不确定你的 GPU 架构，运行 `hipInfo.exe` 查看 `gcnArchName` 字段。

---

## 六、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Could NOT find amd_comgr` | HIP SDK 缺少 CMake 配置 | 按 1.5 节创建 `amd_comgrConfig.cmake` |
| `AMD COMgr older than 1.7.0` | amd_comgr 版本变量未设置 | 确保 config 文件中设置了 `VERSION_MAJOR/MINOR/PATCH` |
| `fatal error: 'hip/hip_runtime.h' file not found` + `Code object build failed` | 内核 JIT 编译时找不到 HIP 头文件 | 内核文件 (`MIOpenGemmDq.cpp`) 不能包含任何 `#include` |
| 路径超 260 字符导致编译失败 | Windows MAX_PATH 限制 | 用 `subst M: "..."` 映射短盘符 |
| `half_float::detail::expr` 大量错误 | MIOpenDriver 与 vcpkg half 不兼容 | 加 `-DMIOPEN_BUILD_DRIVER=OFF` |
| `Invalid character escape '\U'` | pkg-config 解析 Windows 路径 | 加 `-DPKG_CONFIG_EXECUTABLE="PKG_CONFIG_EXECUTABLE-NOTFOUND"` |
| `use of undeclared identifier 'miopenGemmDqForward'` | 测试代码没定义宏 | 在 `#include` 前加 `#define MIOPEN_BETA_API 1` |
| 270+ `dllimport is not supported` 警告 | hipcc/clang 不支持 MSVC declspec | 无害，忽略 |
| Windows .exe 在 WSL 中找不到 DLL | DLL 搜索走 Windows 规则 | 用 `cmd.exe /c "set PATH=... && xxx.exe"` 运行 |
| `make: hipcc: No such file or directory` | WSL 中没找到 hipcc | Makefile 中用完整路径 `/mnt/c/AMD/ROCm/7.1/bin/hipcc.exe` |

---

## 七、开发者备忘

### 修改 kernel 后的增量编译

修改 `src/kernels/MIOpenGemmDq.cpp` 后，`ninja` 会自动重新 inline kernel 并重链接。
通常只需编译 2-4 个目标，十几秒即可完成。

```powershell
cd M:\build
ninja -j8
```

### GemmDq 在 MIOpen 中的调用链

```
用户调用
  └── miopenGemmDqForward()              [gemmdq_api.cpp — C API]
        └── miopen::GemmDqForward()      [gemmdq.cpp — C++ API]
              ├── ProblemDescription       [参数校验: M%128, N%128, K%32]
              ├── GemmDqInvokeParams       [封装所有指针和标量]
              └── SolverContainer::ExecutePrimitive()
                    ├── IsApplicable()     [检查维度对齐]
                    └── GetSolution()      [配置 2D grid + 512 threads/block]
                          └── GemmDqFusedWmmaForward   [GPU Kernel]
```

### Kernel 约束（MIOpen JIT 内核规则）

- **禁止 `#include`**：JIT 编译环境没有 HIP 头文件
- **使用 `_Float16`** 而非 `half`（编译器内置类型）
- **使用 `unsigned int`** 而非 `uint32_t`（不需要 `<stdint.h>`）
- **使用 `unsigned char`** 而非 `uint8_t`
- **不要自定义 `uint2`**：HIPRTC runtime 已内置该类型
- 函数必须声明为 **`extern "C" __global__`**
