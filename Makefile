# Run from the repository root. The default target only prints help.
.DEFAULT_GOAL := help

CC ?= cc
AR ?= ar
NVCC ?= nvcc
CPPFLAGS += -Iinclude -Isrc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O2 -std=c++14
BUILD := build

CPU_SOURCES := src/matrix/cpu.c src/convolution/cpu.c
CPU_OBJECTS := $(patsubst src/%.c,$(BUILD)/cpu/%.o,$(CPU_SOURCES))
CUDA_SOURCES := src/matrix/naive.cu src/matrix/tiled.cu \
                src/matrix/optimized.cu src/matrix/cublas.cu \
                src/convolution/cuda.cu src/bindings/exports.cu
CUDA_OBJECTS := $(patsubst src/%.cu,$(BUILD)/cuda/%.o,$(CUDA_SOURCES))
CUDA_HEADERS := include/lab6.h src/common/cuda_utils.cuh

.PHONY: help all cpu cuda clean
help:
	@echo "make cpu   - build/liblab6_cpu.a and build/bench_cpu"
	@echo "make cuda  - build/liblab6_cuda.so and build/bench_cuda (Linux + CUDA Toolkit)"
	@echo "make all   - both of the above"
	@echo "make clean - remove build/"
	@echo "See README.md for ownership and pending benchmark entry points."

all: cpu cuda

cpu: $(BUILD)/liblab6_cpu.a $(BUILD)/bench_cpu

cuda: $(BUILD)/liblab6_cuda.so $(BUILD)/bench_cuda

clean:
	rm -rf $(BUILD)

$(BUILD)/liblab6_cpu.a: $(CPU_OBJECTS)
	$(AR) rcs $@ $^

$(BUILD)/cpu/%.o: src/%.c include/lab6.h
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/bench_cpu: apps/bench_cpu.c $(BUILD)/liblab6_cpu.a include/lab6.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/liblab6_cpu.a $(LDFLAGS) -o $@

# Each .cu compiles on its own so one broken file is reported by name and
# the others rebuild incrementally.
$(BUILD)/cuda/%.o: src/%.cu $(CUDA_HEADERS)
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Xcompiler=-fPIC -c $< -o $@

$(BUILD)/liblab6_cuda.so: $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -shared $^ $(LDFLAGS) -lcublas -o $@

$(BUILD)/bench_cuda: apps/bench_cuda.cu $(BUILD)/liblab6_cuda.so $(CUDA_HEADERS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $< -L$(BUILD) -llab6_cuda \
		-Xlinker -rpath -Xlinker '$$ORIGIN' $(LDFLAGS) -o $@

-include $(CPU_OBJECTS:.o=.d)
