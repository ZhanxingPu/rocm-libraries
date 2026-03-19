# ============================================================
# MIOpen GemmDq (Fused GEMM + Dequantization) Test
#
# Usage (run in WSL):
#   make                                          # Build
#   make gendata                                  # Generate test data with Python
#   make run SIZE=128x128x128 GS=128              # Run with MxKxN group_size
#   make test SIZE=128x128x128 GS=128             # gendata + run
#   make test_all                                 # Run all preset test cases
# ============================================================

# ---- User-configurable paths (modify these for your environment) ----

# 1. HIP SDK install path
HIP_SDK    := /mnt/c/AMD/ROCm/7.1

# 2. GPU architecture
OFFLOAD    := --offload-arch=gfx1150

# 3. vcpkg include path (defaults to rocm-libraries sibling, adjust if needed)
VCPKG_INC  := ../../../vcpkg/installed/x64-windows/include

# ---- Paths derived from repo layout (no need to change) ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
MIOPEN_SRC   := ../../projects/miopen
MIOPEN_BUILD := $(MIOPEN_SRC)/build

# ---- Build settings ----

CXXFLAGS := -O3 -x hip

INCLUDES := -I$(MIOPEN_SRC)/include \
            -I$(MIOPEN_BUILD)/include \
            -I$(VCPKG_INC)

LDFLAGS := -L$(MIOPEN_BUILD)/lib -lMIOpen

TARGET := test_gemm_dq.exe

# Default test parameters
SIZE     ?= 128x128x128
GS       ?= 128
DATA_DIR ?= data

# Auto-convert WSL paths to Windows paths for DLL loading at runtime
MIOPEN_BUILD_WIN := $(shell wslpath -w "$(abspath $(MIOPEN_BUILD))")
HIP_SDK_WIN      := $(shell wslpath -w "$(HIP_SDK)")
RUN_ENV := set PATH=$(MIOPEN_BUILD_WIN)\bin;$(MIOPEN_BUILD_WIN)\lib;$(HIP_SDK_WIN)\bin;%PATH%

# ---- Targets ----

all: $(TARGET)

$(TARGET): test_gemm_dq.cpp
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD) $< -o $@ $(INCLUDES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf $(DATA_DIR)

gendata:
	python3 gen_gemm_dq_data.py $(SIZE) --group-size $(GS) --dir $(DATA_DIR)

run: $(TARGET)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) $(SIZE) $(GS) $(DATA_DIR)"

run_custom: $(TARGET)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) $(ARGS)"

test: gendata $(TARGET)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) $(SIZE) $(GS) $(DATA_DIR)"

test_all: $(TARGET)
	python3 gen_gemm_dq_data.py 128x128x128 --group-size 128 --dir $(DATA_DIR)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) 128x128x128 128 $(DATA_DIR)"
	python3 gen_gemm_dq_data.py 128x256x128 --group-size 128 --dir $(DATA_DIR)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) 128x256x128 128 $(DATA_DIR)"
	python3 gen_gemm_dq_data.py 256x256x256 --group-size 128 --dir $(DATA_DIR)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) 256x256x256 128 $(DATA_DIR)"
	python3 gen_gemm_dq_data.py 256x512x256 --group-size 128 --dir $(DATA_DIR)
	cmd.exe /c "$(RUN_ENV) && %CD%\$(TARGET) 256x512x256 128 $(DATA_DIR)"

.PHONY: all clean run run_custom gendata test test_all
