# Run from the repository root. The default target only prints help.
.DEFAULT_GOAL := help

CC ?= cc
AR ?= ar
NVCC ?= nvcc
CPPFLAGS += -Iinclude -Isrc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O2 -std=c++14
C_DEFS := -D_POSIX_C_SOURCE=200809L
BUILD := build

CPU_SOURCES := src/matrix/cpu.c src/convolution/cpu.c
CPU_OBJECTS := $(patsubst src/%.c,$(BUILD)/cpu/%.o,$(CPU_SOURCES))
CUDA_SOURCES := src/matrix/naive.cu src/matrix/tiled.cu \
                src/matrix/optimized.cu src/matrix/cublas.cu \
                src/convolution/cuda.cu src/bindings/exports.cu
CUDA_OBJECTS := $(patsubst src/%.cu,$(BUILD)/cuda/%.o,$(CUDA_SOURCES))
CUDA_HEADERS := include/lab6.h src/common/timer.h src/common/validate.h \
                src/common/cuda_utils.cuh
HARNESS := $(BUILD)/apps/harness.o

.PHONY: help all cpu cuda test print-flags clean
help:
	@echo "make cpu         - build/liblab6_cpu.a and build/bench_cpu"
	@echo "make cuda        - build/liblab6_cuda.so and build/bench_cuda (Linux + CUDA Toolkit)"
	@echo "make all         - both of the above"
	@echo "make test        - build/bench_cpu selftest"
	@echo "make print-flags - compilers and flags, for environment.txt"
	@echo "make clean       - remove build/"
	@echo "See README.md for ownership and benchmark commands."

all: cpu cuda

cpu: $(BUILD)/liblab6_cpu.a $(BUILD)/bench_cpu

cuda: $(BUILD)/liblab6_cuda.so $(BUILD)/bench_cuda

test: $(BUILD)/bench_cpu
	$(BUILD)/bench_cpu selftest

print-flags:
	@echo "CC=$(CC) CPPFLAGS=$(CPPFLAGS) $(C_DEFS) CFLAGS=$(CFLAGS)"
	@echo "NVCC=$(NVCC) NVCCFLAGS=$(NVCCFLAGS) LDFLAGS=$(LDFLAGS)"

clean:
	rm -rf $(BUILD)

$(BUILD)/liblab6_cpu.a: $(CPU_OBJECTS)
	$(AR) rcs $@ $^

$(BUILD)/cpu/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(C_DEFS) $(CFLAGS) -MMD -MP -c $< -o $@

$(HARNESS): apps/harness.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(C_DEFS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/bench_cpu: apps/bench_cpu.c apps/harness.h $(HARNESS) $(BUILD)/liblab6_cpu.a
	$(CC) $(CPPFLAGS) $(C_DEFS) $(CFLAGS) $< $(HARNESS) $(BUILD)/liblab6_cpu.a \
		$(LDFLAGS) -lm -o $@

# Each .cu compiles on its own so one broken file is reported by name and
# the others rebuild incrementally.
$(BUILD)/cuda/%.o: src/%.cu $(CUDA_HEADERS)
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -Xcompiler=-fPIC -c $< -o $@

$(BUILD)/liblab6_cuda.so: $(CUDA_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -shared $^ $(LDFLAGS) -lcublas -o $@

$(BUILD)/bench_cuda: apps/bench_cuda.cu apps/harness.h $(HARNESS) $(BUILD)/liblab6_cuda.so
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $< $(HARNESS) -L$(BUILD) -llab6_cuda \
		-Xlinker -rpath -Xlinker '$$ORIGIN' $(LDFLAGS) -lm -o $@

-include $(CPU_OBJECTS:.o=.d) $(HARNESS:.o=.d)
