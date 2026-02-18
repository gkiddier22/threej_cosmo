# Makefile for GPU Wigner 3j coupling matrix implementations
# Requires NVIDIA HPC SDK (nvc compiler with OpenMP target offloading)

# Compiler and flags
CC := nvc
CFLAGS = -O3 -mp=gpu -gpu=cc80 -Minfo=mp

# For different GPU architectures, adjust -gpu flag:
# A100: -gpu=cc80
# V100: -gpu=cc70
# H100: -gpu=cc90

LIBS = -lm

# Targets
TARGETS = thrj_000_gpu thrj_220_gpu

.PHONY: all clean info

all: $(TARGETS)

thrj_000_gpu: thrj_000_gpu.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

thrj_220_gpu: thrj_220_gpu.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS) *.o

info:
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo "Targets: $(TARGETS)"
