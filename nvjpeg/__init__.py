"""
nvjpeg-torch: GPU-accelerated JPEG encoding/decoding with PyTorch support.

This module provides hardware-accelerated JPEG operations using NVIDIA's nvJPEG library,
with native support for PyTorch tensors. The interface is designed to be compatible with
TurboJPEG for easy swapping between CPU and GPU implementations.
"""

# Import torch first to ensure its libraries are loaded before our C extension
# This is necessary because our extension links against libc10.so and other torch libraries
import torch  # noqa: F401

# Now import the C extension
from _nvjpeg import NvJPEG

# Import constants for TurboJPEG compatibility
from _nvjpeg import (
    # Pixel formats
    TJPF_RGB,
    TJPF_BGR,
    TJPF_RGBX,
    TJPF_BGRX,
    TJPF_XBGR,
    TJPF_XRGB,
    TJPF_GRAY,
    TJPF_RGBA,
    TJPF_BGRA,
    TJPF_ABGR,
    TJPF_ARGB,
    TJPF_CMYK,
    # Chroma subsampling
    TJSAMP_444,
    TJSAMP_422,
    TJSAMP_420,
    TJSAMP_GRAY,
    TJSAMP_440,
    TJSAMP_411,
    TJSAMP_441,
)

__all__ = [
    "NvJPEG",
    # Pixel formats
    "TJPF_RGB",
    "TJPF_BGR",
    "TJPF_RGBX",
    "TJPF_BGRX",
    "TJPF_XBGR",
    "TJPF_XRGB",
    "TJPF_GRAY",
    "TJPF_RGBA",
    "TJPF_BGRA",
    "TJPF_ABGR",
    "TJPF_ARGB",
    "TJPF_CMYK",
    # Chroma subsampling
    "TJSAMP_444",
    "TJSAMP_422",
    "TJSAMP_420",
    "TJSAMP_GRAY",
    "TJSAMP_440",
    "TJSAMP_411",
    "TJSAMP_441",
]
