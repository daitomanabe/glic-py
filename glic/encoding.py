"""
Encoding methods for compressed data.
Supports 6 different encoding strategies.
"""

import numpy as np
from typing import List, Tuple
from .config import EncodingMethod
from .bitio import BitWriter, BitReader


def quantize(value: int, quant_value: int) -> int:
    """Quantize a value."""
    if quant_value <= 0:
        return value
    return value // quant_value


def dequantize(value: int, quant_value: int) -> int:
    """Dequantize a value."""
    if quant_value <= 0:
        return value
    return value * quant_value


def get_bits_needed(max_val: int) -> int:
    """Calculate bits needed to represent a value."""
    if max_val <= 0:
        return 1
    return max(1, int(np.ceil(np.log2(max_val + 1))))


# ============ RAW Encoding ============

def encode_raw(data: np.ndarray, writer: BitWriter, bits_per_value: int = 16) -> None:
    """Encode data as raw values."""
    flat = data.flatten()

    # Write dimensions
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_byte(bits_per_value)

    # Write values
    for val in flat:
        # Handle signed values
        if val < 0:
            val = (1 << bits_per_value) + val
        writer.write_bits(int(val) & ((1 << bits_per_value) - 1), bits_per_value)


def decode_raw(reader: BitReader) -> np.ndarray:
    """Decode raw values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    bits_per_value = reader.read_byte()

    data = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            val = reader.read_bits(bits_per_value)
            # Handle signed values
            if val >= (1 << (bits_per_value - 1)):
                val -= (1 << bits_per_value)
            data[y, x] = val

    return data


# ============ PACKED Encoding ============

def encode_packed(data: np.ndarray, writer: BitWriter) -> None:
    """Encode data using packed bit representation."""
    flat = data.flatten()

    # Find range
    min_val = int(np.min(flat))
    max_val = int(np.max(flat))

    # Shift to positive
    shifted = flat - min_val
    range_val = max_val - min_val

    bits_needed = get_bits_needed(range_val)

    # Write header
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_int16(min_val)
    writer.write_byte(bits_needed)

    # Write packed values
    for val in shifted:
        writer.write_bits(int(val), bits_needed)


def decode_packed(reader: BitReader) -> np.ndarray:
    """Decode packed values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    min_val = reader.read_int16()
    bits_needed = reader.read_byte()

    data = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            val = reader.read_bits(bits_needed)
            data[y, x] = val + min_val

    return data


# ============ RLE Encoding ============

def encode_rle(data: np.ndarray, writer: BitWriter) -> None:
    """Encode data using Run-Length Encoding."""
    flat = data.flatten().astype(np.int32)

    # Find range for value encoding
    min_val = int(np.min(flat))
    max_val = int(np.max(flat))
    shifted = flat - min_val
    range_val = max_val - min_val
    value_bits = get_bits_needed(range_val)

    # Write header
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_int16(min_val)
    writer.write_byte(value_bits)

    # Build runs
    runs = []
    i = 0
    while i < len(shifted):
        val = shifted[i]
        count = 1
        while i + count < len(shifted) and shifted[i + count] == val and count < 255:
            count += 1
        runs.append((int(val), count))
        i += count

    # Write number of runs
    writer.write_uint32(len(runs))

    # Write runs
    for val, count in runs:
        writer.write_bits(val, value_bits)
        writer.write_byte(count)


def decode_rle(reader: BitReader) -> np.ndarray:
    """Decode RLE values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    min_val = reader.read_int16()
    value_bits = reader.read_byte()
    num_runs = reader.read_uint32()

    flat = []
    for _ in range(num_runs):
        val = reader.read_bits(value_bits)
        count = reader.read_byte()
        flat.extend([val + min_val] * count)

    # Truncate or pad to exact size
    total_size = height * width
    if len(flat) > total_size:
        flat = flat[:total_size]
    elif len(flat) < total_size:
        flat.extend([0] * (total_size - len(flat)))

    return np.array(flat, dtype=np.int32).reshape(height, width)


# ============ DELTA Encoding ============

def encode_delta(data: np.ndarray, writer: BitWriter) -> None:
    """Encode data using delta/differential encoding."""
    flat = data.flatten().astype(np.int32)

    # Compute deltas
    deltas = np.zeros_like(flat)
    deltas[0] = flat[0]
    deltas[1:] = flat[1:] - flat[:-1]

    # Find range
    min_val = int(np.min(deltas))
    max_val = int(np.max(deltas))
    shifted = deltas - min_val
    range_val = max_val - min_val
    bits_needed = get_bits_needed(range_val)

    # Write header
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_int16(min_val)
    writer.write_byte(bits_needed)

    # Write deltas
    for val in shifted:
        writer.write_bits(int(val), bits_needed)


def decode_delta(reader: BitReader) -> np.ndarray:
    """Decode delta values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    min_val = reader.read_int16()
    bits_needed = reader.read_byte()

    total_size = height * width
    deltas = []
    for _ in range(total_size):
        val = reader.read_bits(bits_needed)
        deltas.append(val + min_val)

    # Reconstruct original values
    flat = np.cumsum(deltas).astype(np.int32)

    return flat.reshape(height, width)


# ============ XOR Encoding ============

def encode_xor(data: np.ndarray, writer: BitWriter) -> None:
    """Encode data using XOR with previous value."""
    flat = data.flatten().astype(np.int32)

    # Shift to positive for XOR
    min_val = int(np.min(flat))
    shifted = flat - min_val

    # Compute XOR chain
    xored = np.zeros_like(shifted)
    xored[0] = shifted[0]
    for i in range(1, len(shifted)):
        xored[i] = shifted[i] ^ shifted[i-1]

    # Find max for bit width
    max_val = int(np.max(xored))
    bits_needed = get_bits_needed(max_val)

    # Write header
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_int16(min_val)
    writer.write_byte(bits_needed)

    # Write XORed values
    for val in xored:
        writer.write_bits(int(val), bits_needed)


def decode_xor(reader: BitReader) -> np.ndarray:
    """Decode XOR values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    min_val = reader.read_int16()
    bits_needed = reader.read_byte()

    total_size = height * width
    xored = []
    for _ in range(total_size):
        xored.append(reader.read_bits(bits_needed))

    # Reconstruct values
    shifted = [xored[0]]
    for i in range(1, len(xored)):
        shifted.append(shifted[i-1] ^ xored[i])

    flat = np.array(shifted, dtype=np.int32) + min_val

    return flat.reshape(height, width)


# ============ ZIGZAG Encoding ============

def zigzag_encode_val(val: int) -> int:
    """Encode signed integer using zigzag encoding."""
    return (val << 1) ^ (val >> 31)


def zigzag_decode_val(val: int) -> int:
    """Decode zigzag encoded integer."""
    return (val >> 1) ^ -(val & 1)


def encode_zigzag(data: np.ndarray, writer: BitWriter) -> None:
    """Encode data using zigzag encoding (maps signed to unsigned)."""
    flat = data.flatten().astype(np.int32)

    # Apply zigzag encoding
    zigzag = np.array([zigzag_encode_val(int(v)) for v in flat], dtype=np.uint32)

    # Find max for bit width
    max_val = int(np.max(zigzag))
    bits_needed = get_bits_needed(max_val)

    # Write header
    writer.write_uint16(data.shape[0])
    writer.write_uint16(data.shape[1])
    writer.write_byte(bits_needed)

    # Write values
    for val in zigzag:
        writer.write_bits(int(val), bits_needed)


def decode_zigzag(reader: BitReader) -> np.ndarray:
    """Decode zigzag values."""
    height = reader.read_uint16()
    width = reader.read_uint16()
    bits_needed = reader.read_byte()

    total_size = height * width
    zigzag = []
    for _ in range(total_size):
        zigzag.append(reader.read_bits(bits_needed))

    flat = np.array([zigzag_decode_val(v) for v in zigzag], dtype=np.int32)

    return flat.reshape(height, width)


# ============ Encoding Dispatcher ============

_ENCODERS = {
    EncodingMethod.RAW: encode_raw,
    EncodingMethod.PACKED: encode_packed,
    EncodingMethod.RLE: encode_rle,
    EncodingMethod.DELTA: encode_delta,
    EncodingMethod.XOR: encode_xor,
    EncodingMethod.ZIGZAG: encode_zigzag,
}

_DECODERS = {
    EncodingMethod.RAW: decode_raw,
    EncodingMethod.PACKED: decode_packed,
    EncodingMethod.RLE: decode_rle,
    EncodingMethod.DELTA: decode_delta,
    EncodingMethod.XOR: decode_xor,
    EncodingMethod.ZIGZAG: decode_zigzag,
}


def encode_data(data: np.ndarray, method: EncodingMethod) -> bytes:
    """Encode 2D data using the specified method."""
    writer = BitWriter()
    encoder = _ENCODERS.get(method, encode_packed)
    encoder(data, writer)
    return writer.get_bytes()


def decode_data(data: bytes, method: EncodingMethod) -> np.ndarray:
    """Decode data using the specified method."""
    reader = BitReader(data)
    decoder = _DECODERS.get(method, decode_packed)
    return decoder(reader)


def encode_segments_data(segments_data: List[np.ndarray], method: EncodingMethod) -> bytes:
    """Encode multiple segment data arrays."""
    writer = BitWriter()

    # Write number of segments
    writer.write_uint32(len(segments_data))

    for seg_data in segments_data:
        # Encode each segment
        seg_bytes = encode_data(seg_data, method)
        writer.write_uint32(len(seg_bytes))
        for b in seg_bytes:
            writer.write_byte(b)

    return writer.get_bytes()


def decode_segments_data(data: bytes, method: EncodingMethod) -> List[np.ndarray]:
    """Decode multiple segment data arrays."""
    reader = BitReader(data)

    num_segments = reader.read_uint32()
    segments_data = []

    for _ in range(num_segments):
        seg_len = reader.read_uint32()
        seg_bytes = bytes([reader.read_byte() for _ in range(seg_len)])
        seg_data = decode_data(seg_bytes, method)
        segments_data.append(seg_data)

    return segments_data
