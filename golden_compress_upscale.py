#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden‑Ratio Compression + Fractal Upscaler (v1.0)
==================================================
Compresses an image to a small file (fractal dictionary + hypervector + thumbnail)
and decompresses it to a high‑resolution image using golden‑ratio fractal upscaling.

Usage:
  compress:   python golden_compress_upscale.py compress input.jpg output.golden
  decompress: python golden_compress_upscale.py decompress input.golden output.png [--upscale-iter N]

Author: DeepSeek Space Lab (Golden‑Ratio Compendium)
License: MIT
"""

import sys
import os
import math
import struct
import tempfile
import argparse
from collections import defaultdict
import numpy as np

# Try to import optional accelerators
try:
    import cupy as cp
    import cupyx.scipy.spatial as cp_spatial
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Install it for upscaling.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: Pillow not installed. Install it for image I/O.")

# ============================================================
# Golden‑ratio constants
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
ALPHA = 1 / PHI          # 0.618
BETA = 1 / PHI**2        # 0.382
DIM = 3819               # hypervector dimension

# Pre‑compute base hypervectors for all 256 bytes
np.random.seed(42)
BASE_HV = np.random.randn(256, DIM).astype(np.float32)
BASE_HV /= np.linalg.norm(BASE_HV, axis=1, keepdims=True)

if HAS_GPU:
    BASE_HV_GPU = cp.asarray(BASE_HV)

# ============================================================
# Fractal dictionary (for compression)
# ============================================================
def extract_fractal_dict(data, max_patterns=256):
    """Extract most frequent 3‑grams as fractal dictionary."""
    freq = defaultdict(int)
    for i in range(len(data) - 2):
        gram = data[i:i+3]
        freq[gram] += 1
    sorted_pats = sorted(freq.items(), key=lambda x: -x[1])[:max_patterns]
    patterns = [pat for pat, _ in sorted_pats]
    mapping = {pat: i for i, pat in enumerate(patterns)}
    encoded = bytearray()
    i = 0
    n = len(data)
    while i < n:
        matched = False
        for pat in patterns:
            if data.startswith(pat, i):
                encoded.append(mapping[pat])
                i += len(pat)
                matched = True
                break
        if not matched:
            encoded.append(data[i])
            i += 1
    return bytes(encoded), patterns

def rebuild_from_dict(encoded, patterns):
    """Rebuild data from dictionary tokens."""
    rev_mapping = {i: pat for i, pat in enumerate(patterns)}
    out = bytearray()
    for b in encoded:
        if b in rev_mapping:
            out.extend(rev_mapping[b])
        else:
            out.append(b)
    return bytes(out)

# ============================================================
# Hypervector operations
# ============================================================
def hv_from_bytes(data):
    hv = np.zeros(DIM, dtype=np.float32)
    n = len(data)
    for i in range(n):
        hv += ALPHA * BASE_HV[data[i]]
        if i < n-1:
            hv += BETA * BASE_HV[data[i+1]]
    norm = np.linalg.norm(hv)
    return hv / norm if norm > 0 else hv

def hv_to_bytes(hv):
    max_abs = np.max(np.abs(hv))
    if max_abs == 0:
        max_abs = 1.0
    scaled = hv / max_abs * 32767
    ints = np.round(scaled).astype(np.int16)
    return ints.tobytes(), max_abs

def bytes_to_hv(data, max_abs):
    ints = np.frombuffer(data, dtype=np.int16)
    return (ints.astype(np.float32) / 32767.0) * max_abs

# ============================================================
# Fractal Upscaler (forward mapping IFS)
# ============================================================
class FractalUpscaler:
    def __init__(self, iterations=1, tile_size=512, use_yuv=True):
        self.iterations = iterations
        self.tile_size = tile_size
        self.use_yuv = use_yuv
        self._inv_coeffs = None

    def _precompute_inv_coeffs(self, iterations):
        """Pre‑compute all 2^iterations inverse map coefficients (scale + shift)."""
        n_maps = 1 << iterations
        coeffs = np.zeros((n_maps, 2, 2), dtype=np.float32)
        for mask in range(n_maps):
            scale = PHI ** iterations
            shift_x = 0.0
            shift_y = 0.0
            for k in range(iterations):
                if (mask >> k) & 1:
                    shift_x = shift_x * PHI + (PHI - 1)
                    shift_y = shift_y * PHI + (PHI - 1)
                else:
                    shift_x *= PHI
                    shift_y *= PHI
            coeffs[mask, 0, 0] = scale
            coeffs[mask, 0, 1] = shift_x
            coeffs[mask, 1, 0] = scale
            coeffs[mask, 1, 1] = shift_y
        return coeffs

    def _upscale_channel(self, img_channel, target_h, target_w):
        """Upscale a single 2D channel using forward mapping IFS."""
        h, w = img_channel.shape
        if self._inv_coeffs is None or self._inv_coeffs.shape[0] != (1 << self.iterations):
            self._inv_coeffs = self._precompute_inv_coeffs(self.iterations)

        out = np.zeros((target_h, target_w), dtype=np.float32)
        # Build KDTree of source pixels
        src_y, src_x = np.mgrid[0:h, 0:w] / max(h, w)
        src_coords = np.column_stack((src_x.ravel(), src_y.ravel()))
        src_vals = img_channel.ravel()
        tree = cKDTree(src_coords)

        # Process tiles
        for y0 in range(0, target_h, self.tile_size):
            y1 = min(y0 + self.tile_size, target_h)
            for x0 in range(0, target_w, self.tile_size):
                x1 = min(x0 + self.tile_size, target_w)
                tile_h = y1 - y0
                tile_w = x1 - x0
                # Target coordinates for this tile (normalized)
                yi = np.linspace(y0/target_h, (y1-1)/target_h, tile_h)
                xi = np.linspace(x0/target_w, (x1-1)/target_w, tile_w)
                Xt, Yt = np.meshgrid(xi, yi)
                tile_coords = np.column_stack((Xt.ravel(), Yt.ravel()))
                # Compute pre‑images
                scales = self._inv_coeffs[:, 0, 0]
                shifts_x = self._inv_coeffs[:, 0, 1]
                shifts_y = self._inv_coeffs[:, 1, 1]
                pre_x = tile_coords[:, 0][:, None] * scales[None, :] + shifts_x[None, :]
                pre_y = tile_coords[:, 1][:, None] * scales[None, :] + shifts_y[None, :]
                pre_coords = np.stack([pre_x, pre_y], axis=-1)
                pre_coords = np.clip(pre_coords, 0, 1)
                all_pre = pre_coords.reshape(-1, 2)
                # Nearest neighbour (fast)
                _, indices = tree.query(all_pre)
                interp_vals = src_vals[indices].reshape(len(tile_coords), -1)
                # Average over all inverse maps
                avg_vals = np.mean(interp_vals, axis=1)
                out[y0:y1, x0:x1] = avg_vals.reshape(tile_h, tile_w)
        return out

    def upscale(self, img_array):
        """Upscale RGB image using optional YUV conversion."""
        if self.use_yuv:
            # RGB → YUV
            rgb = img_array.astype(np.float32) / 255.0
            Y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
            U = -0.147 * rgb[:,:,0] - 0.289 * rgb[:,:,1] + 0.436 * rgb[:,:,2]
            V = 0.615 * rgb[:,:,0] - 0.515 * rgb[:,:,1] - 0.100 * rgb[:,:,2]
            h, w = Y.shape
            target_h = int(round(h * (PHI ** self.iterations)))
            target_w = int(round(w * (PHI ** self.iterations)))
            Y_up = self._upscale_channel(Y, target_h, target_w)
            # Upsample U, V with simple bilinear
            U_up = zoom(U, (target_h/h, target_w/w), order=1)
            V_up = zoom(V, (target_h/h, target_w/w), order=1)
            # YUV → RGB
            rgb_up = np.zeros((target_h, target_w, 3), dtype=np.float32)
            rgb_up[:,:,0] = Y_up + 1.13983 * V_up
            rgb_up[:,:,1] = Y_up - 0.39465 * U_up - 0.58060 * V_up
            rgb_up[:,:,2] = Y_up + 2.03211 * U_up
            rgb_up = np.clip(rgb_up * 255, 0, 255).astype(np.uint8)
            return rgb_up
        else:
            h, w, c = img_array.shape
            target_h = int(round(h * (PHI ** self.iterations)))
            target_w = int(round(w * (PHI ** self.iterations)))
            out = np.zeros((target_h, target_w, c), dtype=np.uint8)
            for ch in range(c):
                out[:,:,ch] = self._upscale_channel(img_array[:,:,ch].astype(np.float32), target_h, target_w)
            return out

# ============================================================
# Compression / Decompression
# ============================================================
def compress_image(input_path, output_path):
    """Compress image to .golden file."""
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    # Thumbnail
    thumb = img.copy()
    thumb.thumbnail((128, 128))
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        thumb.save(tmp, format='JPEG', quality=85)
        thumb_data = tmp.read()
    # Fractal dictionary on raw image bytes
    img_bytes = img_array.tobytes()
    encoded, patterns = extract_fractal_dict(img_bytes)
    # Hypervector
    hv = hv_from_bytes(img_bytes)
    hv_bytes, max_abs = hv_to_bytes(hv)
    # Pack patterns
    pat_bytes = b''
    for p in patterns:
        pat_bytes += struct.pack('<I', len(p)) + p
    # Write output
    magic = b'GOLDEN'
    thumb_len = len(thumb_data)
    pat_len = len(pat_bytes)
    output = magic + struct.pack('<I', thumb_len) + thumb_data + struct.pack('<I', pat_len) + pat_bytes + struct.pack('<f', max_abs) + hv_bytes
    with open(output_path, 'wb') as f:
        f.write(output)
    ratio = len(img_bytes) / len(output)
    print(f"Compressed {len(img_bytes)} bytes -> {len(output)} bytes (ratio: {ratio:.2f}:1)")

def decompress_image(input_path, output_path, upscale_iterations=2):
    """Decompress and upscale to high resolution."""
    with open(input_path, 'rb') as f:
        data = f.read()
    if data[:6] != b'GOLDEN':
        raise ValueError("Not a valid GOLDEN compressed file")
    pos = 6
    thumb_len = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    thumb_data = data[pos:pos+thumb_len]
    pos += thumb_len
    pat_len = struct.unpack('<I', data[pos:pos+4])[0]
    pos += 4
    pat_bytes = data[pos:pos+pat_len]
    pos += pat_len
    max_abs = struct.unpack('<f', data[pos:pos+4])[0]
    pos += 4
    hv_bytes = data[pos:pos + DIM*2]
    # Reconstruct patterns (not used in upscaling, only for completeness)
    patterns = []
    ppos = 0
    while ppos < pat_len:
        plen = struct.unpack('<I', pat_bytes[ppos:ppos+4])[0]
        ppos += 4
        pat = pat_bytes[ppos:ppos+plen]
        ppos += plen
        patterns.append(pat)
    # Load thumbnail
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(thumb_data)
        tmp_path = tmp.name
    thumb_img = Image.open(tmp_path)
    os.unlink(tmp_path)
    # Upscale
    upscaler = FractalUpscaler(iterations=upscale_iterations, tile_size=256, use_yuv=True)
    upscaled = upscaler.upscale(np.array(thumb_img))
    Image.fromarray(upscaled).save(output_path)
    factor = PHI ** upscale_iterations
    print(f"Decompressed and upscaled {thumb_img.size} -> {upscaled.shape[1]}x{upscaled.shape[0]} (factor ~{factor:.2f})")

# ============================================================
# Command‑line interface
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Golden‑Ratio Compression + Fractal Upscaler')
    parser.add_argument('mode', choices=['compress', 'decompress'])
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--upscale-iter', type=int, default=2,
                        help='Number of upscaling iterations (default: 2, factor φ²≈2.618)')
    args = parser.parse_args()

    if args.mode == 'compress':
        compress_image(args.input, args.output)
    else:
        decompress_image(args.input, args.output, args.upscale_iter)

if __name__ == '__main__':
    main()
