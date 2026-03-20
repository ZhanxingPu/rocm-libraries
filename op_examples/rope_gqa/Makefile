# ============================================================
# MIOpen RoPE + GQA Flash Attention Test
#
# Works in both WSL and Windows CMD/PowerShell.
#
# Usage:
#   make                                    # Build
#   make gendata                            # Generate test data (Python)
#   make run                                # Run benchmark (random data)
#   make test                               # gendata + run with validation
#   make test_all                           # Sweep GQA ratios
# ============================================================

# ---- OS detection ----

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := WSL
endif

# ---- User-configurable ----

OFFLOAD := --offload-arch=gfx1150
VCPKG_INC := ../../../vcpkg/installed/x64-windows/include

ifeq ($(DETECTED_OS),Windows)
    HIP_SDK ?= C:/AMD/ROCm/7.1
    PYTHON  ?= python
else
    HIP_SDK ?= /mnt/c/AMD/ROCm/7.1
    PYTHON  ?= python3
endif

# ---- Paths ----

HIPCC        := $(HIP_SDK)/bin/hipcc.exe
MIOPEN_SRC   := ../../projects/miopen
MIOPEN_BUILD := $(MIOPEN_SRC)/build

# ---- Build settings ----

CXXFLAGS := -O3 -x hip

INCLUDES := -I$(MIOPEN_SRC)/include \
            -I$(MIOPEN_BUILD)/include \
            -I$(VCPKG_INC)

LDFLAGS := -L$(MIOPEN_BUILD)/lib -lMIOpen

TARGET := test_rope_gqa.exe

# Default test parameters
BATCH    ?= 1
SEQLEN_Q ?= 4096
SEQLEN_K ?= 4096
NHEAD_Q  ?= 32
NHEAD_K  ?= 32
HDIM     ?= 128
DATA_DIR ?= data

# ---- OS-specific ----

ifeq ($(DETECTED_OS),Windows)
    MIOPEN_BUILD_WIN := $(subst /,\,$(MIOPEN_BUILD))
    HIP_SDK_WIN      := $(subst /,\,$(HIP_SDK))
    RUN_CMD = set "PATH=$(MIOPEN_BUILD_WIN)\bin;$(MIOPEN_BUILD_WIN)\lib;$(HIP_SDK_WIN)\bin;%PATH%" && .\$(TARGET)
    define DO_CLEAN
	if exist $(TARGET) del /q $(TARGET)
	if exist $(DATA_DIR) rmdir /s /q $(DATA_DIR)
    endef
else
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

$(TARGET): test_rope_gqa.cpp
	$(HIPCC) $(CXXFLAGS) $(OFFLOAD) $< -o $@ $(INCLUDES) $(LDFLAGS)

clean:
	$(DO_CLEAN)

gendata:
	$(PYTHON) gen_rope_gqa_data.py --batch $(BATCH) --seqlen_q $(SEQLEN_Q) --seqlen_k $(SEQLEN_K) --nhead_q $(NHEAD_Q) --nhead_k $(NHEAD_K) --hdim $(HDIM) --dir $(DATA_DIR)

run: $(TARGET)
	$(RUN_CMD) $(BATCH) $(SEQLEN_Q) $(SEQLEN_K) $(NHEAD_Q) $(NHEAD_K) $(HDIM)

run_data: $(TARGET)
	$(RUN_CMD) $(BATCH) $(SEQLEN_Q) $(SEQLEN_K) $(NHEAD_Q) $(NHEAD_K) $(HDIM) $(DATA_DIR)

run_custom: $(TARGET)
	$(RUN_CMD) $(ARGS)

test: gendata $(TARGET)
	$(RUN_CMD) $(BATCH) $(SEQLEN_Q) $(SEQLEN_K) $(NHEAD_Q) $(NHEAD_K) $(HDIM) $(DATA_DIR)

test_all: $(TARGET)
	$(PYTHON) gen_rope_gqa_data.py --batch 2 --seqlen_q 128 --seqlen_k 128 --nhead_q 32 --nhead_k 8 --hdim 128 --dir $(DATA_DIR)
	$(RUN_CMD) 2 128 128 32 8 128 $(DATA_DIR)
	$(PYTHON) gen_rope_gqa_data.py --batch 2 --seqlen_q 256 --seqlen_k 256 --nhead_q 32 --nhead_k 8 --hdim 128 --dir $(DATA_DIR)
	$(RUN_CMD) 2 256 256 32 8 128 $(DATA_DIR)
	$(PYTHON) gen_rope_gqa_data.py --batch 1 --seqlen_q 512 --seqlen_k 512 --nhead_q 32 --nhead_k 4 --hdim 128 --dir $(DATA_DIR)
	$(RUN_CMD) 1 512 512 32 4 128 $(DATA_DIR)

.PHONY: all clean run run_data run_custom gendata test test_all
