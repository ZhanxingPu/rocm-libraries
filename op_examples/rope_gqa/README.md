# RoPE + GQA Flash Attention — MIOpen 集成

## 一、概述

将 **Rotary Position Embedding (RoPE)** 与 **Grouped Query Attention (GQA)** 作为一个融合操作 `rope_gqa` 集成到 MIOpen 中。

### 流水线

```
Input: Q, K, V, cos_table, sin_table
  ↓
RoPE(Q, cos, sin) → Q'    (compiled HIP kernel, in-place)
RoPE(K, cos, sin) → K'    (compiled HIP kernel, in-place)
  ↓
FMHA_GQA(Q', K', V) → O   (CK Tile Flash Attention)
  ↓
Output: O
```

### 内核构成

RoPE 和 FMHA GQA 的全部 GPU 计算均在同一个文件中实现，编译期链接进 `MIOpen.dll`：

| 组件 | 实现方式 | 位置 |
|------|---------|------|
| RoPE (旋转位置编码) | HIP `__global__` kernel (`__half`) | `MIOpenRopeGqa.cpp` Part 1 |
| FMHA GQA (Flash Attention) | CK Tile `FmhaFwdKernel` 模板实例化 | `MIOpenRopeGqa.cpp` Part 2 |
| 入口函数 `RunRopeGqaKernel()` | RoPE(Q) → RoPE(K) → FMHA | `MIOpenRopeGqa.cpp` Part 3 |

Solver 通过 `extern` 声明直接调用 `RunRopeGqaKernel()`。

> **注意：** 不使用 JIT 编译，修改 kernel 后只需 `ninja` 增量编译，**无需清除 JIT 缓存**。

### 约束

- 数据类型：FP16
- 布局：`[batch, nhead, seqlen, hdim]` (row-major)
- `nhead_q % nhead_k == 0` (支持 MHA/GQA/MQA)
- `rotary_dim` 为偶数，且 `<= hdim`
- 当前 CK Tile FMHA 仅支持 `hdim = 128`
- 需要 RDNA3+ GPU (gfx1100/gfx1150)

---

## 二、前置环境准备

> 如果你已经按照 `README_GEMM_FP16_U4.md` 完成了环境搭建，可直接跳到第三节。
> 下面的步骤与 gemm_fp16_u4 完全一致。

### 2.1 安装 HIP SDK

从 [AMD ROCm/HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub.html) 下载安装。

安装完成后记录安装路径，后续用 `<HIP_SDK>` 表示，例如 `C:\AMD\Rocm\7.1`。

### 2.2 安装构建工具

| 工具 | 安装方式 | 说明 |
|------|----------|------|
| CMake ≥ 3.15 | https://cmake.org/download/ 安装 MSI，勾选加入 PATH | 编译 MIOpen 库 |
| Ninja | `pip install ninja` | CMake 构建后端 |
| GNU Make | `choco install make` | 编译和运行测试程序 |
| Python + NumPy | `pip install numpy` | 生成测试数据 |

### 2.3 准备 vcpkg（MIOpen 第三方依赖管理）

```powershell
cd <WORKSPACE>
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
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

### 2.4 准备 rocm-cmake

```powershell
cd <WORKSPACE>
git clone --depth 1 https://github.com/ROCm/rocm-cmake.git
```

### 2.5 准备 Composable Kernel（CK）头文件

rope_gqa 的 FMHA 部分依赖 CK Tile 头文件（仅需 headers，不需要编译 CK 库）：

```powershell
cd <WORKSPACE>
git clone --depth 1 https://github.com/ROCm/composable_kernel.git
```

CK 头文件路径示例：`C:\AMD\Rocm\composable_kernel\include`

### 2.6 创建 amd_comgr CMake 配置文件

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

---

## 三、编译 MIOpen（启用 CK Tile FMHA）

### 3.1 映射短路径（必须）

MIOpen 的部分内核汇编文件名极长，加上源码路径后容易超过 Windows 260 字符限制。
**必须**用 `subst` 映射一个短盘符：

```powershell
subst M: "<WORKSPACE>\rocm-libraries\projects\miopen"
```

> 此映射重启后消失，每次重新开机后需重新执行。

### 3.2 CMake 配置

```powershell
# ===== 按实际环境修改这两个变量 =====
$HIP_SDK   = "C:/AMD/Rocm/7.1"
$WORKSPACE = "C:/Users/zhanxipu/Desktop/Workspace/Rocm/rocm_lib_zhanxipu"

# ===== 以下无需修改 =====
mkdir M:\build -ErrorAction SilentlyContinue
cd M:\build

cmake -G "Ninja" `
  -DMIOPEN_BACKEND=HIP `
  -DCMAKE_PREFIX_PATH="$HIP_SDK" `
  -DCMAKE_CXX_COMPILER="$HIP_SDK/bin/clang++.exe" `
  -DCMAKE_C_COMPILER="$HIP_SDK/bin/clang.exe" `
  -DCMAKE_TOOLCHAIN_FILE="$WORKSPACE/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DROCmCMakeBuildTools_DIR="$WORKSPACE/rocm-cmake/share/rocmcmakebuildtools/cmake" `
  -DMIOPEN_USE_COMPOSABLEKERNEL=OFF `
  -DMIOPEN_USE_MLIR=OFF `
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF `
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF `
  -DMIOPEN_BUILD_DRIVER=OFF `
  -DBUILD_TESTING=OFF `
  -DPKG_CONFIG_EXECUTABLE="PKG_CONFIG_EXECUTABLE-NOTFOUND" `
  -DHALF_INCLUDE_DIR="$WORKSPACE/vcpkg/installed/x64-windows/include" `
  -DCK_TILE_INCLUDE_DIR="$WORKSPACE/composable_kernel/include" `
  -DGPU_TARGETS=gfx1150 `
  ..
```

只需修改最上面的 `$HIP_SDK` 和 `$WORKSPACE` 两个变量，下面的 cmake 命令完全不用动。
PowerShell 的 `$var` 在双引号里会自动展开，且保持正斜杠不会产生转义问题。

> **各参数说明**：
>
> | 参数 | 说明 |
> |------|------|
> | `-DMIOPEN_BACKEND=HIP` | 使用 HIP 后端 |
> | `-DCMAKE_PREFIX_PATH` | HIP SDK 安装路径 |
> | `-DCMAKE_CXX_COMPILER` | HIP SDK 自带的 clang++ |
> | `-DCMAKE_TOOLCHAIN_FILE` | vcpkg 工具链（自动查找 SQLite3、BZip2 等） |
> | `-DROCmCMakeBuildTools_DIR` | rocm-cmake 模块路径 |
> | `-DMIOPEN_USE_COMPOSABLEKERNEL=OFF` | 禁用 CK 原始 CMake 集成（非必需） |
> | `-DMIOPEN_USE_MLIR=OFF` | 禁用 MLIR（Windows 下不可用） |
> | `-DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF` | 禁用 AI 调优（需要 frugally-deep） |
> | `-DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF` | 禁用 AI fallback（需要 frugally-deep） |
> | `-DMIOPEN_BUILD_DRIVER=OFF` | 禁用 MIOpenDriver（与 vcpkg half 版本不兼容） |
> | `-DBUILD_TESTING=OFF` | 禁用单元测试构建（需要 GTest） |
> | `-DPKG_CONFIG_EXECUTABLE=...NOTFOUND` | 禁用 pkg-config（Windows 路径解析有 bug） |
> | `-DHALF_INCLUDE_DIR` | half 浮点库头文件路径 |
> | `-DCK_TILE_INCLUDE_DIR` | **CK Tile 头文件路径（rope_gqa 新增）** |
> | `-DGPU_TARGETS` | 目标 GPU 架构 |

配置成功后应看到：`CK Tile FMHA enabled: <CK_INCLUDE_PATH>`

### 3.3 编译

```powershell
cd M:\build
ninja -j8
```

首次编译约 **10-20 分钟**（~680 个编译目标），后续增量编译很快（几十秒）。

编译产物：
- `M:\build\bin\MIOpen.dll` — 动态链接库
- `M:\build\lib\MIOpen.lib` — 导入库

### 3.4 增量编译

| 修改的文件 | 需要做 | 是否需要清除缓存 |
|-----------|--------|--------------|
| `src/kernels/MIOpenRopeGqa.cpp` | `ninja` | 否 |
| `src/solver/rope_gqa/forward_rope_gqa.cpp` | `ninja` | 否 |
| `src/rope_gqa.cpp` / `src/rope_gqa_api.cpp` | `ninja` | 否 |
| 头文件 (`.hpp`) | `ninja` | 否 |

> rope_gqa 的所有 GPU 代码都是编译期链接的（非 JIT），因此**不涉及 JIT 缓存问题**。
> 所有情况都**不需要 `ninja clean`**。`ninja` 的依赖追踪会自动识别需要重编的文件。

---

## 四、API 接口

### C API

```c
#define MIOPEN_BETA_API 1    // 必须在 include 之前定义！
#include <miopen/miopen.h>

miopenStatus_t miopenRopeGqaForward(
    miopenHandle_t handle,
    int batch,          // batch size
    int seqlen_q,       // query sequence length
    int seqlen_k,       // key/value sequence length
    int nhead_q,        // number of query heads
    int nhead_k,        // number of KV heads (nhead_q % nhead_k == 0)
    int hdim,           // head dimension
    int rotary_dim,     // dimensions to rotate (even, <= hdim)
    void* Q,            // [batch, nhead_q, seqlen_q, hdim] FP16, modified in-place
    void* K,            // [batch, nhead_k, seqlen_k, hdim] FP16, modified in-place
    const void* V,      // [batch, nhead_k, seqlen_k, hdim] FP16
    void* O,            // [batch, nhead_q, seqlen_q, hdim] FP16 output
    const void* cos_t,  // [max(seqlen_q,seqlen_k), rotary_dim/2] FP16
    const void* sin_t   // [max(seqlen_q,seqlen_k), rotary_dim/2] FP16
);
```

**注意：** Q 和 K 被 RoPE **原地修改**。调用后它们的内容已改变。

> **重要**：调用方必须在 `#include <miopen/miopen.h>` **之前** 定义 `#define MIOPEN_BETA_API 1`，
> 否则编译器看不到 `miopenRopeGqaForward` 的声明。

---

## 五、编译和运行测试程序

测试文件位于 `op_examples/rope_gqa/` 目录下：

| 文件 | 说明 |
|------|------|
| `test_rope_gqa.cpp` | 正确性验证 + 性能测试（调用 `miopenRopeGqaForward` C API） |
| `gen_rope_gqa_data.py` | 生成测试数据 + NumPy 参考结果 |
| `Makefile` | 编译/运行脚本 |

### 5.1 生成测试数据

```powershell
cd op_examples\rope_gqa
python gen_rope_gqa_data.py --batch 2 --seqlen_q 256 --seqlen_k 256 --nhead_q 32 --nhead_k 8 --hdim 128
```

### 5.2 编译测试程序

```powershell
make clean && make
```

### 5.3 运行正确性测试

```powershell
make test BATCH=2 SEQLEN_Q=256 SEQLEN_K=256 NHEAD_Q=32 NHEAD_K=8 HDIM=128
```

### 5.4 运行基准测试（无验证）

```powershell
make run BATCH=2 SEQLEN_Q=1024 SEQLEN_K=1024 NHEAD_Q=32 NHEAD_K=8 HDIM=128
```

### 5.5 批量测试

```powershell
make test_all
```

---

## 六、文件结构

```
rocm-libraries/
├── README_ROPE_GQA.md                                # 本文档
├── projects/miopen/
│   ├── include/miopen/miopen.h                       # C API 声明
│   ├── src/
│   │   ├── include/miopen/
│   │   │   ├── rope_gqa.hpp                          # C++ API header
│   │   │   └── rope_gqa/
│   │   │       ├── problem_description.hpp
│   │   │       ├── invoke_params.hpp
│   │   │       └── solvers.hpp
│   │   ├── rope_gqa.cpp                              # C++ 实现
│   │   ├── rope_gqa_api.cpp                          # C API 包装
│   │   ├── kernels/MIOpenRopeGqa.cpp                  # ★ Kernel: RoPE + CK FMHA 全部 GPU 计算
│   │   ├── rope_gqa/
│   │   │   └── problem_description.cpp
│   │   └── solver/rope_gqa/
│   │       └── forward_rope_gqa.cpp                  # Solver: extern 调用 RunRopeGqaKernel()
│   └── CMakeLists.txt                                # 新增 CK_TILE_INCLUDE_DIR
└── op_examples/rope_gqa/
    ├── test_rope_gqa.cpp                             # 测试程序
    ├── gen_rope_gqa_data.py                          # Python 数据生成
    └── Makefile                                      # 构建脚本
```

---

## 七、RoPE 算法说明

采用 LLaMA 风格的 **half-rotated** RoPE：

```
θ_i = 1 / 10000^(2i / rotary_dim),  i = 0, 1, ..., rotary_dim/2 - 1

x'[d]              = x[d] × cos(pos × θ_d) − x[d + half] × sin(pos × θ_d)
x'[d + half]       = x[d] × sin(pos × θ_d) + x[d + half] × cos(pos × θ_d)
```

其中 `half = rotary_dim / 2`，`pos` 为 token 在序列中的位置。

超过 `rotary_dim` 的维度不做旋转。

---

## 八、CK Tile FMHA 说明

FMHA 使用 CK Tile 的 `FmhaFwdKernel` 模板，编译期实例化。

当前配置：
- Block tile: `<128, 64, 32, 128, 32, 128>`
- FP16 输入/输出，FP32 累加
- 无 bias / mask / dropout / LSE
- 支持 GQA (通过 `nhead_q / nhead_k` ratio)
- 目标架构：RDNA3+ (gfx1100/gfx1150)

如需支持其他 `hdim` 值，需要新增模板实例化。

---

## 九、RopeGqa 在 MIOpen 中的调用链

```
用户调用
  └── miopenRopeGqaForward()                  [rope_gqa_api.cpp — C API]
        └── miopen::RopeGqaForward()          [rope_gqa.cpp — C++ API]
              ├── ProblemDescription            [参数校验: hdim=128, rotary_dim 偶数]
              ├── RopeGqaInvokeParams           [封装所有指针和标量]
              └── SolverContainer::ExecutePrimitive()
                    ├── IsApplicable()          [检查 hdim == 128]
                    └── GetSolution()           [invoker 调用 RunRopeGqaKernel()]
                          └── RunRopeGqaKernel  [MIOpenRopeGqa.cpp]
                                ├── RoPE(Q)     [HIP __global__ kernel]
                                ├── RoPE(K)     [HIP __global__ kernel]
                                └── FMHA GQA    [CK Tile FmhaFwdKernel]
```

---

## 十、常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Could NOT find amd_comgr` | HIP SDK 缺少 CMake 配置 | 按 2.6 节创建 `amd_comgrConfig.cmake` |
| 路径超 260 字符导致编译失败 / `File not found` | Windows MAX_PATH 限制 | **必须**用 `subst M: "..."` 映射短盘符（3.1 节） |
| `Could not find rocMLIR` | Windows 下不可用 | 加 `-DMIOPEN_USE_MLIR=OFF` |
| `Could not find frugally-deep` | AI 调优依赖 | 加 `-DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF` |
| `Could NOT find GTest` | 单元测试依赖 | 加 `-DBUILD_TESTING=OFF` |
| `Invalid character escape '\U'` | pkg-config 解析 Windows 路径 | 加 `-DPKG_CONFIG_EXECUTABLE="PKG_CONFIG_EXECUTABLE-NOTFOUND"` |
| `half_float::detail::expr` 大量错误 | MIOpenDriver 与 vcpkg half 不兼容 | 加 `-DMIOPEN_BUILD_DRIVER=OFF` |
| 编译报错找不到 `ck_tile/core.hpp` | 缺少 CK 头文件 | 加 `-DCK_TILE_INCLUDE_DIR=<path>` |
| 运行时报 "No applicable solver" | CK 未启用 / hdim 不支持 | 确认 cmake 日志有 `CK Tile FMHA enabled`，且 `hdim=128` |
| `use of undeclared identifier 'miopenRopeGqaForward'` | 没定义宏 | 在 `#include` 前加 `#define MIOPEN_BETA_API 1` |
| 修改 kernel 后行为没变 | — | rope_gqa 是编译期链接的，`ninja` 重编即可，**无需清缓存** |
| 能不能不用 CK？ | FMHA 依赖 CK Tile | 不设 `CK_TILE_INCLUDE_DIR` 则 rope_gqa 不可用 |
