#!/usr/bin/env python
"""
Build script for nvjpeg-torch - Python bindings for NVIDIA nvJPEG library with PyTorch support.

This module provides GPU-accelerated JPEG encoding/decoding using NVIDIA's nvJPEG,
with native support for PyTorch tensors.
"""

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Base version - CUDA and torch suffix will be appended
BASE_VERSION = "0.1.0"


def get_cuda_version():
    """Detect CUDA version and return as string like 'cu128' for CUDA 12.8."""
    # Try nvcc first
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse output like "Cuda compilation tools, release 12.4, V12.4.131"
        match = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if match:
            major, minor = match.groups()
            return f"cu{major}{minor}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try reading from CUDA path
    cuda_path = get_cuda_path()
    if cuda_path:
        # Check version.json (CUDA 11+)
        version_json = os.path.join(cuda_path, "version.json")
        if os.path.exists(version_json):
            try:
                import json
                with open(version_json) as f:
                    data = json.load(f)
                    version = data.get("cuda", {}).get("version", "")
                    match = re.match(r"(\d+)\.(\d+)", version)
                    if match:
                        major, minor = match.groups()
                        return f"cu{major}{minor}"
            except Exception:
                pass

        # Try parsing from path name (e.g., /usr/local/cuda-12.4)
        match = re.search(r"cuda-?(\d+)\.?(\d+)?", cuda_path)
        if match:
            major = match.group(1)
            minor = match.group(2) or "0"
            return f"cu{major}{minor}"

    # Default fallback
    return "cu12"


def get_torch_version():
    """Get PyTorch version string like 'torch2.5'."""
    try:
        import torch
        version = torch.__version__
        # Parse version like "2.5.0+cu124" or "2.5.0"
        match = re.match(r"(\d+)\.(\d+)", version)
        if match:
            major, minor = match.groups()
            return f"torch{major}.{minor}"
    except ImportError:
        pass
    return "torch2.0"  # Default fallback


def get_version():
    """Get full version string with CUDA and torch suffix."""
    cuda_suffix = get_cuda_version()
    torch_suffix = get_torch_version()
    return f"{BASE_VERSION}+{cuda_suffix}{torch_suffix}"


class NvJpegBuildExt(build_ext):
    """Custom build extension that handles CUDA and PyTorch library configuration."""

    def build_extensions(self):
        # Customize compiler flags
        for ext in self.extensions:
            if self.compiler.compiler_type == "unix":
                # Get torch library path for rpath
                torch_lib_path = get_torch_library_paths()[0] if get_torch_library_paths() else ""
                
                # Add rpath so the extension can find bundled libraries and torch libs
                ext.extra_link_args.extend([
                    "-Wl,-rpath,$ORIGIN/nvjpeg_libs",
                    "-Wl,-rpath,$ORIGIN",
                ])
                if torch_lib_path:
                    ext.extra_link_args.append(f"-Wl,-rpath,{torch_lib_path}")
        super().build_extensions()


def get_cuda_path():
    """Find CUDA installation path."""
    # Check environment variable first
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path

    # Check common locations
    common_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def get_torch_include_paths():
    """Get PyTorch include directories."""
    from torch.utils.cpp_extension import include_paths
    return include_paths()


def get_torch_library_paths():
    """Get PyTorch library directories."""
    from torch.utils.cpp_extension import library_paths
    return library_paths()


def get_extension():
    """Configure the nvjpeg extension module."""
    import torch
    
    cuda_path = get_cuda_path()

    if platform.system() == "Linux":
        include_dirs = [
            "include",
            numpy.get_include(),
        ] + get_torch_include_paths()
        
        library_dirs = get_torch_library_paths()
        libraries = ["cudart", "nvjpeg", "c10", "torch", "torch_cpu", "torch_python"]
        extra_compile_args = ["-std=c++17", "-O3", "-D_GLIBCXX_USE_CXX11_ABI=0"]
        extra_link_args = []

        # Check if torch was built with CXX11 ABI
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            extra_compile_args = ["-std=c++17", "-O3", "-D_GLIBCXX_USE_CXX11_ABI=1"]

        if cuda_path:
            include_dirs.append(os.path.join(cuda_path, "include"))
            library_dirs.append(os.path.join(cuda_path, "lib64"))

        return Extension(
            "_nvjpeg",  # Underscore prefix - internal module, wrapped by nvjpeg/__init__.py
            sources=["nvjpeg-python.cpp", "src/x86/JpegCoder.cpp"],
            include_dirs=include_dirs,
            define_macros=[("JPEGCODER_ARCH", "x86")],
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )

    elif platform.system() == "Windows":
        if not cuda_path:
            raise RuntimeError(
                "CUDA_PATH environment variable not set. "
                "Please install CUDA Toolkit and set CUDA_PATH."
            )

        cuda_include = os.path.join(cuda_path, "include")
        if platform.machine().endswith("64"):
            cuda_lib = os.path.join(cuda_path, "lib", "x64")
        else:
            cuda_lib = os.path.join(cuda_path, "lib", "Win32")

        return Extension(
            "_nvjpeg",  # Underscore prefix - internal module, wrapped by nvjpeg/__init__.py
            sources=["nvjpeg-python.cpp", "src/x86/JpegCoder.cpp"],
            include_dirs=["include", numpy.get_include(), cuda_include] + get_torch_include_paths(),
            define_macros=[("JPEGCODER_ARCH", "x86")],
            library_dirs=[cuda_lib] + get_torch_library_paths(),
            libraries=["cudart", "nvjpeg", "c10", "torch", "torch_cpu", "torch_python"],
            extra_compile_args=["/std:c++17"],
        )

    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")


setup(
    version=get_version(),
    packages=["nvjpeg"],  # Python wrapper package that imports torch first
    ext_modules=[get_extension()],
    cmdclass={"build_ext": NvJpegBuildExt},
)
