# Makefile for CPU Wigner 3j coupling matrix implementations
# Supports Mac (Apple Silicon/Intel) and Linux

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform detection
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon - requires Homebrew GCC for OpenMP
        CXX := $(shell which /opt/homebrew/bin/g++-15 /opt/homebrew/bin/g++-14 /opt/homebrew/bin/g++-13 2>/dev/null | head -1)
    else
        # Intel Mac
        CXX := g++
    endif
else
    # Linux
    CXX := g++
endif

# Core flags: -O3 -march=native -fopenmp -ffast-math
# These are the only flags that provide measurable speedup
CXXFLAGS = -O3 -march=native -fopenmp -ffast-math

TARGETS = thrj_all

.PHONY: all clean info

all: $(TARGETS)

thrj_%: thrj_%.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CFLAGS)"
