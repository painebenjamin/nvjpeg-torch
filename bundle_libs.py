#!/usr/bin/env python
"""
Bundle CUDA libraries into the wheel.

This script copies the required CUDA shared libraries into the wheel
and fixes the RPATH so the extension can find them at runtime.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def get_cuda_lib_path():
    """Find CUDA library path."""
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        lib_path = os.path.join(cuda_path, "lib64")
        if os.path.exists(lib_path):
            return lib_path

    common_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-11/lib64",
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def find_library(lib_name: str, search_paths: list) -> Path | None:
    """Find a library file in the search paths."""
    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue
        # Look for the library (handle symlinks)
        for lib_file in path.glob(f"{lib_name}*"):
            if lib_file.is_file() and not lib_file.is_symlink():
                return lib_file
    return None


def get_library_dependencies(lib_path: Path) -> list[str]:
    """Get the shared library dependencies of a file using ldd."""
    try:
        result = subprocess.run(
            ["ldd", str(lib_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        deps = []
        for line in result.stdout.strip().split("\n"):
            if "=>" in line:
                parts = line.split("=>")
                if len(parts) >= 2:
                    lib_path_str = parts[1].strip().split()[0]
                    if lib_path_str and lib_path_str != "not" and os.path.exists(lib_path_str):
                        deps.append(lib_path_str)
        return deps
    except subprocess.CalledProcessError:
        return []


def patch_rpath(lib_path: Path, new_rpath: str):
    """Patch the RPATH of a library using patchelf."""
    try:
        subprocess.run(
            ["patchelf", "--set-rpath", new_rpath, str(lib_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to patch RPATH for {lib_path}: {e}")
    except FileNotFoundError:
        print("Warning: patchelf not found. RPATH not patched.")


def bundle_wheel(wheel_path: str, output_dir: str = None) -> str:
    """
    Bundle CUDA libraries into a wheel.

    Args:
        wheel_path: Path to the input wheel file
        output_dir: Output directory for the bundled wheel

    Returns:
        Path to the bundled wheel
    """
    wheel_path = Path(wheel_path)
    if output_dir is None:
        output_dir = wheel_path.parent
    output_dir = Path(output_dir)

    cuda_lib_path = get_cuda_lib_path()
    if not cuda_lib_path:
        raise RuntimeError("Could not find CUDA library path")

    print(f"Using CUDA libraries from: {cuda_lib_path}")

    # Libraries to bundle
    required_libs = [
        "libcudart.so",
        "libnvjpeg.so",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extract_dir = tmpdir / "wheel"

        # Extract wheel
        print(f"Extracting wheel: {wheel_path}")
        with ZipFile(wheel_path, "r") as zf:
            zf.extractall(extract_dir)

        # Create libs directory
        libs_dir = extract_dir / "nvjpeg_libs"
        libs_dir.mkdir(exist_ok=True)

        # Copy required libraries
        bundled_libs = []
        for lib_name in required_libs:
            lib_file = find_library(lib_name, [cuda_lib_path])
            if lib_file:
                dest = libs_dir / lib_file.name
                print(f"Bundling: {lib_file} -> {dest.name}")
                shutil.copy2(lib_file, dest)
                bundled_libs.append(dest)

                # Also create the soname symlink if needed
                soname = lib_name.replace(".so", ".so.12")
                symlink_dest = libs_dir / soname
                if not symlink_dest.exists():
                    symlink_dest.symlink_to(dest.name)
            else:
                print(f"Warning: Could not find {lib_name}")

        # Find and patch the extension module
        for ext_file in extract_dir.glob("*.so"):
            print(f"Patching RPATH for: {ext_file.name}")
            patch_rpath(ext_file, "$ORIGIN/nvjpeg_libs:$ORIGIN")

        # Patch bundled libraries to use relative paths
        for lib_file in bundled_libs:
            patch_rpath(lib_file, "$ORIGIN")

        # Update RECORD file
        record_files = list(extract_dir.glob("*.dist-info/RECORD"))
        if record_files:
            record_file = record_files[0]
            with open(record_file, "a") as f:
                for lib_file in libs_dir.iterdir():
                    rel_path = lib_file.relative_to(extract_dir)
                    f.write(f"{rel_path},,\n")

        # Repack wheel with new name indicating bundled libs
        wheel_name = wheel_path.stem
        # Keep the original name but ensure it's in the output directory
        output_wheel = output_dir / wheel_path.name

        print(f"Creating bundled wheel: {output_wheel}")
        with ZipFile(output_wheel, "w") as zf:
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zf.write(file_path, arcname)

        return str(output_wheel)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle CUDA libraries into a wheel"
    )
    parser.add_argument(
        "wheel",
        help="Path to the wheel file to bundle",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for the bundled wheel",
    )
    args = parser.parse_args()

    try:
        output = bundle_wheel(args.wheel, args.output_dir)
        print(f"\nBundled wheel created: {output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

