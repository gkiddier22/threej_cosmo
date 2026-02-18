# Makefile for CPU Wigner 3j coupling matrix implementations
# Supports Mac (Apple Silicon/Intel) and Linux

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform detection
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon - requires Homebrew GCC for OpenMP
        CC := $(shell which /opt/homebrew/bin/gcc-15 /opt/homebrew/bin/gcc-14 /opt/homebrew/bin/gcc-13 2>/dev/null | head -1)
    else
        # Intel Mac
        CC := gcc
    endif
else
    # Linux
    CC := gcc
endif

# Core flags: -O3 -march=native -fopenmp -ffast-math
# These are the only flags that provide measurable speedup
CFLAGS = -O3 -march=native -fopenmp -ffast-math
LIBS = -lm

TARGETS = thrj_000 thrj_220

.PHONY: all clean info

all: $(TARGETS)

thrj_%: thrj_%.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
