"""
Utility functions for threej_cosmo package.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


def get_package_dir() -> Path:
    """Get the package installation directory."""
    return Path(__file__).parent


def get_src_dir() -> Path:
    """Get the C source directory."""
    return get_package_dir().parent / "src"


def get_src_gpu_dir() -> Path:
    """Get the GPU C source directory."""
    return get_package_dir().parent / "src_gpu"


def get_binary_path(name: str, backend: str = "cpu") -> Path:
    """
    Get path to a compiled binary.

    Parameters
    ----------
    name : str
        Binary name: 'thrj_000' or 'thrj_220'
    backend : str
        'cpu' or 'gpu'

    Returns
    -------
    Path
        Path to the binary
    """
    if backend == "gpu":
        return get_src_gpu_dir() / f"{name}_gpu"
    return get_src_dir() / name


def find_cpu_compiler() -> Optional[str]:
    """Find an available C compiler with OpenMP support for CPU."""
    compilers = ["g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++"]
    homebrew_paths = ["/opt/homebrew/bin", "/usr/local/bin"]

    for compiler in compilers:
        for prefix in homebrew_paths + [""]:
            if prefix:
                full_path = os.path.join(prefix, compiler)
            else:
                full_path = compiler

            if shutil.which(full_path):
                try:
                    result = subprocess.run(
                        [full_path, "-fopenmp", "-x", "c", "-E", "-"],
                        input="int main() { return 0; }",
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return full_path
                except Exception:
                    continue

    return None


def find_gpu_compiler() -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Find an available GPU compiler with OpenMP target offloading support.

    Returns
    -------
    Tuple[Optional[str], Optional[str], Optional[List[str]]]
        (compiler_path, vendor, compile_flags) or (None, None, None) if not found
    """
    # NVIDIA HPC SDK (nvc)
    for nvc in ["nvc", "/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin/nvc"]:
        nvc_path = shutil.which(nvc)
        if nvc_path:
            try:
                result = subprocess.run(
                    [nvc_path, "--version"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and "nvc" in result.stdout.lower():
                    # Default to a common GPU architecture, user can override
                    flags = ["-O3", "-mp=gpu", "-gpu=cc70,cc80,cc90", "-lm"]
                    return nvc_path, "nvidia", flags
            except Exception:
                pass

    # AMD ROCm (amdclang)
    for amdclang in ["amdclang", "/opt/rocm/bin/amdclang"]:
        amdclang_path = shutil.which(amdclang)
        if amdclang_path:
            try:
                result = subprocess.run(
                    [amdclang_path, "--version"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    flags = [
                        "-O3", "-fopenmp",
                        "-fopenmp-targets=amdgcn-amd-amdhsa",
                        "-Xopenmp-target=amdgcn-amd-amdhsa",
                        "-march=gfx90a",  # Default MI200 series
                        "-lm"
                    ]
                    return amdclang_path, "amd", flags
            except Exception:
                pass

    # Clang with CUDA (less common but possible)
    for clang in ["clang-17", "clang-16", "clang-15", "clang"]:
        clang_path = shutil.which(clang)
        if clang_path:
            try:
                # Check if it has CUDA offloading support
                result = subprocess.run(
                    [clang_path, "--print-targets"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and "nvptx" in result.stdout:
                    flags = [
                        "-O3", "-fopenmp",
                        "-fopenmp-targets=nvptx64-nvidia-cuda",
                        "-lm"
                    ]
                    return clang_path, "nvidia-clang", flags
            except Exception:
                pass

    return None, None, None


def get_available_backends() -> List[str]:
    """
    Get list of available computation backends.

    Returns
    -------
    List[str]
        List of available backends ('cpu', 'gpu')
    """
    backends = []

    # CPU is available if we can find a compiler
    if find_cpu_compiler() is not None:
        backends.append("cpu")

    # GPU is available if we can find a GPU compiler
    gpu_compiler, _, _ = find_gpu_compiler()
    if gpu_compiler is not None:
        backends.append("gpu")

    return backends


def ensure_cpu_binaries_compiled(force: bool = False) -> None:
    """
    Ensure CPU C++ binaries are compiled.

    Parameters
    ----------
    force : bool
        If True, recompile even if binaries exist
    """
    src_dir = get_src_dir()
    binaries = ["thrj_all"]

    needs_compile = force or any(
        not (src_dir / name).exists() for name in binaries
    )

    if not needs_compile:
        return

    compiler = find_cpu_compiler()
    if compiler is None:
        raise RuntimeError(
            "No C++ compiler with OpenMP support found. "
            "Install gcc with: brew install gcc (macOS) or apt install gcc (Linux)"
        )

    flags = ["-O3", "-march=native", "-fopenmp", "-ffast-math", "-lm"]

    for name in binaries:
        source = src_dir / f"{name}.cc"
        output = src_dir / name

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if output.exists() and not force:
            continue

        cmd = [compiler] + flags + ["-o", str(output), str(source)] + ["-lm"]
        print(f"Compiling {name} (CPU)...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed for {name}:\n{result.stderr}")

    print("CPU compilation complete.")


def ensure_gpu_binaries_compiled(force: bool = False) -> None:
    """
    Ensure GPU C binaries are compiled.

    Parameters
    ----------
    force : bool
        If True, recompile even if binaries exist

    Raises
    ------
    RuntimeError
        If no GPU compiler is found
    """
    src_gpu_dir = get_src_gpu_dir()
    binaries = ["thrj_000_gpu", "thrj_220_gpu"]

    needs_compile = force or any(
        not (src_gpu_dir / name).exists() for name in binaries
    )

    if not needs_compile:
        return

    compiler, vendor, flags = find_gpu_compiler()
    if compiler is None:
        raise RuntimeError(
            "No GPU compiler found. Install one of:\n"
            "  - NVIDIA HPC SDK (nvc): https://developer.nvidia.com/hpc-sdk\n"
            "  - AMD ROCm (amdclang): https://rocm.docs.amd.com/\n"
        )

    print(f"Found GPU compiler: {compiler} ({vendor})")

    for name in binaries:
        source = src_gpu_dir / f"{name}.c"
        output = src_gpu_dir / name

        if not source.exists():
            raise FileNotFoundError(f"GPU source file not found: {source}")

        if output.exists() and not force:
            continue

        cmd = [compiler] + flags + ["-o", str(output), str(source)]
        print(f"Compiling {name} (GPU)...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"GPU compilation failed for {name}:\n{result.stderr}\n"
                f"Command was: {' '.join(cmd)}"
            )

    print("GPU compilation complete.")


def ensure_binaries_compiled(backend: str = "cpu", force: bool = False) -> None:
    """
    Ensure binaries are compiled for the specified backend.

    Parameters
    ----------
    backend : str
        'cpu' or 'gpu'
    force : bool
        If True, recompile even if binaries exist
    """
    if backend == "cpu":
        ensure_cpu_binaries_compiled(force=force)
    elif backend == "gpu":
        ensure_gpu_binaries_compiled(force=force)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cpu' or 'gpu'.")


def check_gpu_runtime() -> bool:
    """
    Check if GPU runtime is actually working by running a simple test.

    Returns
    -------
    bool
        True if GPU offloading works, False otherwise
    """
    src_gpu_dir = get_src_gpu_dir()
    test_binary = src_gpu_dir / "thrj_000_gpu"

    if not test_binary.exists():
        return False

    # Try to run with minimal input to check if GPU works
    # This is a basic sanity check - actual GPU availability
    # is verified when the binary runs
    try:
        result = subprocess.run(
            [str(test_binary)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # If it prints usage, the binary at least runs
        return "Usage:" in result.stderr or result.returncode == 1
    except Exception:
        return False
