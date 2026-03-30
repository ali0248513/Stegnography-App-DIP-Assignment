"""
Image Steganography Application
Author: Ali Mustafa
Reg No: 235141
Description: Hide one image inside another using LSB (Least Significant Bit) technique.
"""

import numpy as np
from PIL import Image
import struct
import os


# ─────────────────────────────────────────────
# CORE LSB STEGANOGRAPHY FUNCTIONS
# ─────────────────────────────────────────────

def encode_image_in_image(cover_path: str, secret_path: str, output_path: str) -> bool:
  
    cover  = Image.open(cover_path).convert("RGB")
    secret = Image.open(secret_path).convert("RGB")

    cw, ch = cover.size
    sw, sh = secret.size

    # Header: 4 bytes width + 4 bytes height = 8 bytes = 64 bits
    header = struct.pack(">II", sw, sh)
    secret_bytes = np.array(secret, dtype=np.uint8).tobytes()
    payload = header + secret_bytes  # raw bytes to embed

    bits_needed = len(payload) * 8
    bits_available = cw * ch * 3      # 1 LSB per channel

    if bits_needed > bits_available:
        raise ValueError(
            f"Cover image too small. Need {bits_needed} bits, have {bits_available}."
        )

    # Convert payload to a flat bit array
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    cover_array = np.array(cover, dtype=np.uint8)
    flat = cover_array.flatten().copy()

    # Embed bits into LSBs
    flat[:len(payload_bits)] = (flat[:len(payload_bits)] & 0xFE) | payload_bits

    stego_array = flat.reshape(cover_array.shape)
    stego_img   = Image.fromarray(stego_array, "RGB")
    stego_img.save(output_path)
    print(f"[✓] Secret image embedded → saved to '{output_path}'")
    return True


def decode_image_from_image(stego_path: str, output_path: str) -> Image.Image:
    """
    Extract a hidden image from a stego image.

    Args:
        stego_path:  Path to the stego image
        output_path: Path to save the extracted secret image

    Returns:
        The recovered PIL Image
    """
    stego = Image.open(stego_path).convert("RGB")
    flat  = np.array(stego, dtype=np.uint8).flatten()

    # Extract first 64 bits → 8 bytes header
    header_bits  = flat[:64] & 1
    header_bytes = np.packbits(header_bits).tobytes()
    sw, sh = struct.unpack(">II", header_bytes)

    if sw <= 0 or sh <= 0 or sw > 10000 or sh > 10000:
        raise ValueError("Invalid dimensions decoded — image may not contain hidden data.")

    # Extract secret pixel bytes
    secret_byte_count = sw * sh * 3
    secret_bit_count  = secret_byte_count * 8
    secret_bits  = flat[64 : 64 + secret_bit_count] & 1
    secret_bytes = np.packbits(secret_bits).tobytes()

    secret_array = np.frombuffer(secret_bytes, dtype=np.uint8).reshape((sh, sw, 3))
    secret_img   = Image.fromarray(secret_array, "RGB")
    secret_img.save(output_path)
    print(f"[✓] Hidden image extracted → saved to '{output_path}'")
    return secret_img


# ─────────────────────────────────────────────
# METADATA READER  (Q.2)
# ─────────────────────────────────────────────

def read_image_metadata(image_path: str) -> dict:
    """Return basic metadata from an image file."""
    img  = Image.open(image_path)
    meta = {
        "filename"  : os.path.basename(image_path),
        "format"    : img.format,
        "mode"      : img.mode,
        "size_px"   : img.size,
        "file_size" : f"{os.path.getsize(image_path) / 1024:.1f} KB",
    }
    exif = img._getexif() if hasattr(img, "_getexif") else None
    if exif:
        meta["exif_tags"] = len(exif)
    info = img.info
    if info:
        meta["info_keys"] = list(info.keys())
    return meta


# ─────────────────────────────────────────────
# VIEW NON-UINT8 IMAGES  (Q.1)
# ─────────────────────────────────────────────

def view_non_uint8_image(array: np.ndarray, title: str = "Image") -> Image.Image:
    """
    Normalise any numeric numpy array to uint8 range [0, 255] so it can
    be displayed / saved as a standard image.
    """
    arr = array.astype(np.float64)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr if arr.ndim == 2 else arr, "L" if arr.ndim == 2 else "RGB")


# ─────────────────────────────────────────────
# SIMPLE CLI  (optional)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  Image Steganography App — Ali Mustafa (235141)")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python steganography_app.py encode <cover> <secret> <output>")
    print("  python steganography_app.py decode <stego>  <output>")
    print()

    if len(sys.argv) < 2:
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "encode" and len(sys.argv) == 5:
        encode_image_in_image(sys.argv[2], sys.argv[3], sys.argv[4])

    elif cmd == "decode" and len(sys.argv) == 4:
        decode_image_from_image(sys.argv[2], sys.argv[3])

    else:
        print("Invalid arguments. See usage above.")
