# ============================================================
# MIOpen GemmFp16U4 (Fused GEMM + Dequantization) Test
#
# Works in both WSL and Windows CMD/PowerShell.
#
# Usage:
#   make                                          # Build
#   make gendata                                  # Generate test data
#   make run SIZE=128x128x128 GS=128              # Run
#   make test SIZE=128x128x128 GS=128             # gendata + run
#   make test_all                                 # Run all preset cases
# ============================================================

# ---- OS detection ----

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    # Force cmd.exe as shell — WSL installs /bin/sh which GNU Make picks up
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- User-configurable (modify for your environment) ----

# GPU architecture
OFFLOAD := --offload-arch=gfx1150

# vcpkg include path (defaults to rocm-libraries sibling)
VCPKG_INC := ../../../vcpkg/installed/x64-windows/include

# OS-specific defaults
ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= C:/AMD/ROCm/7.1
    PYTHON  ?= python
else
    HIP_SDK ?= /mnt/c/AMD/ROCm/7.1
    PYTHON  ?= python3
endif

# ---- Paths derived from repo layout (no need to change) ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
MIOPEN_SRC   := ../../projects/miopen
MIOPEN_BUILD := $(MIOPEN_SRC)/build

# ---- Build settings (shared) ----

CXXFLAGS := -O3 -x hip

INCLUDES := -I$(MIOPEN_SRC)/include \
            -I$(MIOPEN_BUILD)/include \
            -I$(VCPKG_INC)

LDFLAGS := -L$(MIOPEN_BUILD)/lib -lMIOpen

TARGET := test_gemm_fp16_u4.exe

# Default test parameters
SIZE     ?= 128x128x128
GS       ?= 128
DATA_DIR ?= data

# ---- OS-specific: run command + clean ----

ifeq ($(DETECTED_OS),Windows)
    # Windows: set PATH with relative paths and run directly
    MIOPEN_BUILD_WIN := $(subst /,\,$(MIOPEN_BUILD))
    HIP_SDK_WIN      := $(subst /,\,$(HIP_SDK))
    RUN_CMD = set "PATH=$(MIOPEN_BUILD_WIN)\bin;$(MIOPEN_BUILD_WIN)\lib;$(HIP_SDK_WIN)\bin;%PATH%" && .\$(TARGET)
    define DO_CLEAN
	if exist $(TARGET) del /q $(TARGET)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
    endef
else
    # WSL: convert paths to Windows format for DLL loading via cmd.exe
    MIOPEN_BUILD_WIN := $(shell wslpath -w "$(abspath $(MIOPEN_BUILD))")
    HIP_SDK_WIN      := $(shell wslpath -w "$(HIP_SDK)")
    RUN_CMD = cmd.exe /c "set PATH=$(MIOPEN_BUILD_WIN)\bin;$(MIOPEN_BUILD_WIN)\lib;$(HIP_SDK_WIN)\bin;%PATH% && %CD%\$(TARGET)"
    define DO_CLEAN
	rm -f $(TARGET)
	rm -rf $(DATA_DIR)
    endef
endif

# ---- Targets ----

all: $(TARGET)

$(TARGET): test_gemm_fp16_u4.cpp
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD) $< -o $@ $(INCLUDES) $(LDFLAGS)

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_gemm_fp16_u4_data.py $(SIZE) --group-size $(GS) --dir $(DATA_DIR)

run: $(TARGET)
	$(RUN_CMD) $(SIZE) $(GS) $(DATA_DIR)

run_custom: $(TARGET)
	$(RUN_CMD) $(ARGS)

test: gendata $(TARGET)
	$(RUN_CMD) $(SIZE) $(GS) $(DATA_DIR)

test_all: $(TARGET)
	$(PYTHON) gen_gemm_fp16_u4_data.py 128x128x128 --group-size 128 --dir $(DATA_DIR)
	$(RUN_CMD) 128x128x128 128 $(DATA_DIR)
	$(PYTHON) gen_gemm_fp16_u4_data.py 128x256x128 --group-size 128 --dir $(DATA_DIR)
	$(RUN_CMD) 128x256x128 128 $(DATA_DIR)
	$(PYTHON) gen_gemm_fp16_u4_data.py 256x256x256 --group-size 128 --dir $(DATA_DIR)
	$(RUN_CMD) 256x256x256 128 $(DATA_DIR)
	$(PYTHON) gen_gemm_fp16_u4_data.py 256x512x256 --group-size 128 --dir $(DATA_DIR)
	$(RUN_CMD) 256x512x256 128 $(DATA_DIR)

.PHONY: all clean run run_custom gendata test test_all
