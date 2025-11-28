#!/usr/bin/env python3
"""
Test suite for nvjpeg-torch - tests numpy and PyTorch tensor encoding/decoding.
"""

import sys
import os
import glob

import numpy as np
import torch

# Add project root and build directory to path for local testing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
for lib in glob.glob(os.path.join(project_root, "build/lib.*")):
    sys.path.insert(0, lib)

from nvjpeg import NvJPEG

# Try to import cv2 for visual comparison (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available, skipping visual tests")


def test_numpy_encode_decode():
    """Test basic numpy array encoding and decoding."""
    print("Testing numpy encode/decode...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Read and decode
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    
    decoded = nj.decode(img_bytes)
    assert isinstance(decoded, np.ndarray), "Decoded should be numpy array"
    assert decoded.ndim == 3, "Should be 3D array (HWC)"
    assert decoded.shape[2] == 3, "Should have 3 channels"
    assert decoded.dtype == np.uint8, "Should be uint8"
    print(f"  Decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
    
    # Encode
    encoded = nj.encode(decoded)
    assert isinstance(encoded, bytes), "Encoded should be bytes"
    assert len(encoded) > 0, "Encoded should not be empty"
    print(f"  Encoded size: {len(encoded)} bytes")
    
    print("  ✓ numpy encode/decode passed")


def test_torch_tensor_uint8():
    """Test PyTorch tensor encoding with uint8 CHW RGB format."""
    print("Testing torch tensor (uint8 CHW RGB)...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Read image as numpy (BGR format)
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)  # BGR
    
    # Convert BGR to RGB, then to CHW torch tensor
    # Torch tensors are expected in RGB format
    decoded_rgb = decoded_np[:, :, ::-1].copy()  # BGR -> RGB
    tensor_chw = torch.from_numpy(decoded_rgb).permute(2, 0, 1)  # HWC -> CHW
    assert tensor_chw.shape[0] == 3, "Should be CHW format"
    print(f"  Tensor shape: {tensor_chw.shape}, dtype: {tensor_chw.dtype}")
    
    # Encode torch tensor (RGB) - will be converted to BGR internally
    encoded = nj.encode(tensor_chw)
    assert isinstance(encoded, bytes), "Encoded should be bytes"
    assert len(encoded) > 0, "Encoded should not be empty"
    print(f"  Encoded size: {len(encoded)} bytes")
    
    # Decode and verify (decoded is BGR, original is BGR)
    re_decoded = nj.decode(encoded)
    # Images won't be exactly equal due to JPEG compression, but should be close
    diff = np.abs(decoded_np.astype(np.float32) - re_decoded.astype(np.float32)).mean()
    print(f"  Mean absolute difference after re-encode: {diff:.2f}")
    assert diff < 10, "Re-encoded image should be similar to original"
    
    print("  ✓ torch uint8 CHW RGB passed")


def test_torch_tensor_float_0_1():
    """Test PyTorch tensor encoding with float [0, 1] CHW RGB format."""
    print("Testing torch tensor (float [0,1] CHW RGB)...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Read image as numpy (BGR format)
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)  # BGR
    
    # Convert BGR to RGB, then to float [0, 1] CHW tensor
    decoded_rgb = decoded_np[:, :, ::-1].copy()  # BGR -> RGB
    tensor_chw = torch.from_numpy(decoded_rgb).permute(2, 0, 1).float() / 255.0
    assert tensor_chw.shape[0] == 3, "Should be CHW format"
    assert tensor_chw.min() >= 0 and tensor_chw.max() <= 1, "Should be in [0, 1] range"
    print(f"  Tensor shape: {tensor_chw.shape}, dtype: {tensor_chw.dtype}")
    print(f"  Value range: [{tensor_chw.min():.3f}, {tensor_chw.max():.3f}]")
    
    # Encode torch tensor
    encoded = nj.encode(tensor_chw)
    assert isinstance(encoded, bytes), "Encoded should be bytes"
    assert len(encoded) > 0, "Encoded should not be empty"
    print(f"  Encoded size: {len(encoded)} bytes")
    
    # Decode and verify
    re_decoded = nj.decode(encoded)
    diff = np.abs(decoded_np.astype(np.float32) - re_decoded.astype(np.float32)).mean()
    print(f"  Mean absolute difference after re-encode: {diff:.2f}")
    assert diff < 10, "Re-encoded image should be similar to original"
    
    print("  ✓ torch float [0,1] CHW passed")


def test_torch_tensor_float_neg1_1():
    """Test PyTorch tensor encoding with float [-1, 1] CHW RGB format."""
    print("Testing torch tensor (float [-1,1] CHW RGB)...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Read image as numpy (BGR format)
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)  # BGR
    
    # Convert BGR to RGB, then to float [-1, 1] CHW tensor
    decoded_rgb = decoded_np[:, :, ::-1].copy()  # BGR -> RGB
    tensor_chw = torch.from_numpy(decoded_rgb).permute(2, 0, 1).float() / 255.0
    tensor_chw = tensor_chw * 2 - 1  # [0, 1] -> [-1, 1]
    assert tensor_chw.shape[0] == 3, "Should be CHW format"
    assert tensor_chw.min() >= -1 and tensor_chw.max() <= 1, "Should be in [-1, 1] range"
    print(f"  Tensor shape: {tensor_chw.shape}, dtype: {tensor_chw.dtype}")
    print(f"  Value range: [{tensor_chw.min():.3f}, {tensor_chw.max():.3f}]")
    
    # Encode torch tensor (RGB) - will be converted to BGR internally
    encoded = nj.encode(tensor_chw)
    assert isinstance(encoded, bytes), "Encoded should be bytes"
    assert len(encoded) > 0, "Encoded should not be empty"
    print(f"  Encoded size: {len(encoded)} bytes")
    
    # Decode and verify (decoded is BGR, original is BGR)
    re_decoded = nj.decode(encoded)
    diff = np.abs(decoded_np.astype(np.float32) - re_decoded.astype(np.float32)).mean()
    print(f"  Mean absolute difference after re-encode: {diff:.2f}")
    assert diff < 10, "Re-encoded image should be similar to original"
    
    print("  ✓ torch float [-1,1] CHW RGB passed")


def test_torch_tensor_4d_batch():
    """Test PyTorch tensor encoding with 4D NCHW RGB format (batch size 1)."""
    print("Testing torch tensor (4D NCHW RGB with N=1)...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Read image as numpy (BGR format)
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)  # BGR
    
    # Convert BGR to RGB, then to 4D NCHW tensor with batch size 1
    decoded_rgb = decoded_np[:, :, ::-1].copy()  # BGR -> RGB
    tensor_nchw = torch.from_numpy(decoded_rgb).permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    assert tensor_nchw.ndim == 4, "Should be 4D"
    assert tensor_nchw.shape[0] == 1, "Batch size should be 1"
    assert tensor_nchw.shape[1] == 3, "Should have 3 channels"
    print(f"  Tensor shape: {tensor_nchw.shape}, dtype: {tensor_nchw.dtype}")
    
    # Encode torch tensor (RGB) - will be converted to BGR internally
    encoded = nj.encode(tensor_nchw)
    assert isinstance(encoded, bytes), "Encoded should be bytes"
    assert len(encoded) > 0, "Encoded should not be empty"
    print(f"  Encoded size: {len(encoded)} bytes")
    
    print("  ✓ torch 4D NCHW RGB passed")


def test_torch_write():
    """Test writing torch tensor directly to file."""
    print("Testing torch tensor write to file...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    output_file = os.path.join(os.path.dirname(__file__), "out", "torch_output.jpg")
    
    # Read image as numpy (BGR format)
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)  # BGR
    
    # Convert BGR to RGB, then to CHW torch tensor
    decoded_rgb = decoded_np[:, :, ::-1].copy()  # BGR -> RGB
    tensor_chw = torch.from_numpy(decoded_rgb).permute(2, 0, 1)
    
    # Write directly
    written_bytes = nj.write(output_file, tensor_chw, 90)
    assert written_bytes > 0, "Should write bytes"
    assert os.path.exists(output_file), "Output file should exist"
    print(f"  Written {written_bytes} bytes to {output_file}")
    
    # Verify by reading back
    with open(output_file, "rb") as fp:
        re_read = fp.read()
    assert len(re_read) == written_bytes, "File size should match"
    
    print("  ✓ torch write passed")


def test_read_write():
    """Test read and write convenience functions."""
    print("Testing read/write functions...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    output_file = os.path.join(os.path.dirname(__file__), "out", "test_output.jpg")
    
    # Read using nj.read()
    img = nj.read(image_file)
    assert isinstance(img, np.ndarray), "Should return numpy array"
    print(f"  Read image shape: {img.shape}")
    
    # Write using nj.write()
    written = nj.write(output_file, img, 85)
    assert written > 0, "Should write bytes"
    print(f"  Written {written} bytes")
    
    print("  ✓ read/write passed")


def test_quality_parameter():
    """Test different quality settings."""
    print("Testing quality parameter...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    decoded_np = nj.decode(img_bytes)
    
    # Test different quality levels
    sizes = {}
    for quality in [10, 50, 90, 100]:
        encoded = nj.encode(decoded_np, quality)
        sizes[quality] = len(encoded)
    
    print(f"  Quality 10:  {sizes[10]:,} bytes")
    print(f"  Quality 50:  {sizes[50]:,} bytes")
    print(f"  Quality 90:  {sizes[90]:,} bytes")
    print(f"  Quality 100: {sizes[100]:,} bytes")
    
    # Higher quality should generally produce larger files
    assert sizes[10] < sizes[100], "Lower quality should produce smaller files"
    
    print("  ✓ quality parameter passed")


def visual_comparison():
    """Visual comparison test (requires OpenCV and display)."""
    if not HAS_CV2:
        print("Skipping visual comparison (OpenCV not available)")
        return
    
    print("Running visual comparison...")
    nj = NvJPEG()
    
    image_file = os.path.join(os.path.dirname(__file__), "test-image", "test.jpg")
    
    # Decode with nvjpeg
    with open(image_file, "rb") as fp:
        img_bytes = fp.read()
    nj_np = nj.decode(img_bytes)
    
    # Decode with OpenCV
    cv_np = cv2.imread(image_file)
    
    # Display
    cv2.imshow("NvJPEG Decode Image", nj_np)
    cv2.imshow("OpenCV Decode Image", cv_np)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("  ✓ visual comparison done")


def run_speed_tests():
    """Run speed benchmarks for different data types and image sizes."""
    import time
    
    print("=" * 60)
    print("Speed Tests")
    print("=" * 60)
    print()
    
    nj = NvJPEG()
    
    # Test shapes: (height, width)
    shapes = [
        (256, 256),
        (512, 512),
        (640, 480),
        (1024, 1024),
        (1920, 1080),
        (3840, 2160),  # 4K
    ]
    
    num_iterations = 50
    warmup_iterations = 5
    
    def benchmark(func, iterations=num_iterations, warmup=warmup_iterations):
        """Run function multiple times and return average time in ms."""
        # Warmup
        for _ in range(warmup):
            func()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        elapsed = time.perf_counter() - start
        return (elapsed / iterations) * 1000  # ms
    
    # Generate test images for each shape
    def create_test_images(h, w):
        """Create test images in all supported formats."""
        # Random uint8 numpy array (HWC BGR format)
        np_hwc_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # Torch uint8 CHW RGB
        torch_uint8_chw = torch.from_numpy(
            np_hwc_bgr[:, :, ::-1].copy()
        ).permute(2, 0, 1).contiguous()
        
        # Torch float32 [0, 1] CHW RGB
        torch_float32_0_1 = torch_uint8_chw.float() / 255.0
        
        # Torch float32 [-1, 1] CHW RGB
        torch_float32_neg1_1 = torch_float32_0_1 * 2 - 1
        
        # Torch float16 (half) [0, 1] CHW RGB
        torch_float16_0_1 = torch_float32_0_1.half()
        
        # Torch bfloat16 [0, 1] CHW RGB
        torch_bfloat16_0_1 = torch_float32_0_1.bfloat16()
        
        # Torch float64 (double) [0, 1] CHW RGB
        torch_float64_0_1 = torch_float32_0_1.double()
        
        # Torch 4D NCHW RGB (batch=1)
        torch_nchw = torch_uint8_chw.unsqueeze(0)
        
        return {
            "numpy_uint8_hwc": np_hwc_bgr,
            "torch_uint8_chw": torch_uint8_chw,
            "torch_float16_0_1": torch_float16_0_1,
            "torch_bfloat16_0_1": torch_bfloat16_0_1,
            "torch_float32_0_1": torch_float32_0_1,
            "torch_float32_neg1_1": torch_float32_neg1_1,
            "torch_float64_0_1": torch_float64_0_1,
            "torch_uint8_nchw": torch_nchw,
        }
    
    # Results storage
    encode_results = {shape: {} for shape in shapes}
    decode_results = {shape: {} for shape in shapes}
    
    print(f"Running {num_iterations} iterations per test (+ {warmup_iterations} warmup)")
    print()
    
    # Test encoding for each shape and data type
    print("-" * 60)
    print("ENCODE BENCHMARKS")
    print("-" * 60)
    
    for h, w in shapes:
        print(f"\nShape: {w}x{h} ({w*h/1e6:.2f} MP)")
        images = create_test_images(h, w)
        
        for dtype_name, img in images.items():
            time_ms = benchmark(lambda img=img: nj.encode(img))
            encode_results[(h, w)][dtype_name] = time_ms
            fps = 1000 / time_ms if time_ms > 0 else float('inf')
            print(f"  {dtype_name:25s}: {time_ms:8.2f} ms ({fps:7.1f} fps)")
    
    # Test decoding for each shape and output type
    print()
    print("-" * 60)
    print("DECODE BENCHMARKS (decode + conversion to output type)")
    print("-" * 60)
    
    def decode_to_numpy(jpeg_bytes):
        """Decode to numpy uint8 HWC BGR (native output)."""
        return nj.decode(jpeg_bytes)
    
    def decode_to_torch_uint8(jpeg_bytes):
        """Decode and convert to torch uint8 CHW RGB."""
        np_img = nj.decode(jpeg_bytes)
        return torch.from_numpy(np_img[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    
    def decode_to_torch_float16(jpeg_bytes):
        """Decode and convert to torch float16 CHW RGB [0, 1]."""
        np_img = nj.decode(jpeg_bytes)
        return torch.from_numpy(np_img[:, :, ::-1].copy()).permute(2, 0, 1).half() / 255.0
    
    def decode_to_torch_bfloat16(jpeg_bytes):
        """Decode and convert to torch bfloat16 CHW RGB [0, 1]."""
        np_img = nj.decode(jpeg_bytes)
        return torch.from_numpy(np_img[:, :, ::-1].copy()).permute(2, 0, 1).bfloat16() / 255.0
    
    def decode_to_torch_float32(jpeg_bytes):
        """Decode and convert to torch float32 CHW RGB [0, 1]."""
        np_img = nj.decode(jpeg_bytes)
        return torch.from_numpy(np_img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    
    def decode_to_torch_float64(jpeg_bytes):
        """Decode and convert to torch float64 CHW RGB [0, 1]."""
        np_img = nj.decode(jpeg_bytes)
        return torch.from_numpy(np_img[:, :, ::-1].copy()).permute(2, 0, 1).double() / 255.0
    
    decode_funcs = {
        "numpy_uint8_hwc": decode_to_numpy,
        "torch_uint8_chw": decode_to_torch_uint8,
        "torch_float16_0_1": decode_to_torch_float16,
        "torch_bfloat16_0_1": decode_to_torch_bfloat16,
        "torch_float32_0_1": decode_to_torch_float32,
        "torch_float64_0_1": decode_to_torch_float64,
    }
    
    for h, w in shapes:
        print(f"\nShape: {w}x{h} ({w*h/1e6:.2f} MP)")
        
        # Create a JPEG to decode
        np_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        jpeg_bytes = nj.encode(np_img, 90)
        
        for dtype_name, decode_func in decode_funcs.items():
            time_ms = benchmark(lambda jpeg_bytes=jpeg_bytes, f=decode_func: f(jpeg_bytes))
            decode_results[(h, w)][dtype_name] = time_ms
            fps = 1000 / time_ms if time_ms > 0 else float('inf')
            print(f"  {dtype_name:25s}: {time_ms:8.2f} ms ({fps:7.1f} fps)")
    
    # Summary table
    print()
    print("-" * 60)
    print("SUMMARY TABLE - Encode Times (ms)")
    print("-" * 60)
    
    # Header
    encode_dtypes = list(next(iter(encode_results.values())).keys())
    header = f"{'Resolution':>12s}"
    for dtype in encode_dtypes:
        short_name = dtype.replace("torch_", "t_").replace("numpy_", "np_")
        header += f" | {short_name:>12s}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for h, w in shapes:
        row = f"{w}x{h:>5d}"
        for dtype in encode_dtypes:
            row += f" | {encode_results[(h, w)][dtype]:>12.2f}"
        print(row)
    
    print()
    print("-" * 60)
    print("SUMMARY TABLE - Decode Times (ms)")
    print("-" * 60)
    
    # Header
    decode_dtypes = list(next(iter(decode_results.values())).keys())
    header = f"{'Resolution':>12s}"
    for dtype in decode_dtypes:
        short_name = dtype.replace("torch_", "t_").replace("numpy_", "np_")
        header += f" | {short_name:>12s}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for h, w in shapes:
        row = f"{w}x{h:>5d}"
        for dtype in decode_dtypes:
            row += f" | {decode_results[(h, w)][dtype]:>12.2f}"
        print(row)
    
    print()
    print("  ✓ speed tests completed")


if __name__ == "__main__":
    print("=" * 60)
    print("nvjpeg-torch Test Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    test_numpy_encode_decode()
    print()
    
    test_torch_tensor_uint8()
    print()
    
    test_torch_tensor_float_0_1()
    print()
    
    test_torch_tensor_float_neg1_1()
    print()
    
    test_torch_tensor_4d_batch()
    print()
    
    test_torch_write()
    print()
    
    test_read_write()
    print()
    
    test_quality_parameter()
    print()
    
    # Visual comparison is optional and interactive
    if "--visual" in sys.argv:
        visual_comparison()
        print()
    
    # Speed tests are optional
    if "--speed" in sys.argv:
        run_speed_tests()
        print()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
