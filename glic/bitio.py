"""
Bit-level I/O utilities for reading and writing individual bits.
"""

from typing import List, Optional
import struct


class BitWriter:
    """Writer for bit-level data output."""

    def __init__(self):
        self._buffer: bytearray = bytearray()
        self._current_byte: int = 0
        self._bit_position: int = 0

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        if bit:
            self._current_byte |= (1 << (7 - self._bit_position))
        self._bit_position += 1

        if self._bit_position == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_position = 0

    def write_bits(self, value: int, num_bits: int) -> None:
        """Write multiple bits from value (MSB first)."""
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def write_byte(self, value: int) -> None:
        """Write a full byte."""
        self.write_bits(value & 0xFF, 8)

    def write_uint16(self, value: int) -> None:
        """Write a 16-bit unsigned integer (big-endian)."""
        self.write_bits((value >> 8) & 0xFF, 8)
        self.write_bits(value & 0xFF, 8)

    def write_uint32(self, value: int) -> None:
        """Write a 32-bit unsigned integer (big-endian)."""
        self.write_bits((value >> 24) & 0xFF, 8)
        self.write_bits((value >> 16) & 0xFF, 8)
        self.write_bits((value >> 8) & 0xFF, 8)
        self.write_bits(value & 0xFF, 8)

    def write_int16(self, value: int) -> None:
        """Write a 16-bit signed integer (big-endian)."""
        if value < 0:
            value = (1 << 16) + value
        self.write_uint16(value)

    def write_int32(self, value: int) -> None:
        """Write a 32-bit signed integer (big-endian)."""
        if value < 0:
            value = (1 << 32) + value
        self.write_uint32(value)

    def write_float(self, value: float) -> None:
        """Write a 32-bit float."""
        packed = struct.pack('>f', value)
        for byte in packed:
            self.write_byte(byte)

    def align(self) -> None:
        """Align to byte boundary by padding with zeros."""
        if self._bit_position > 0:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_position = 0

    def get_bytes(self) -> bytes:
        """Get the written data as bytes, aligned to byte boundary."""
        self.align()
        return bytes(self._buffer)

    def get_bit_count(self) -> int:
        """Get total number of bits written."""
        return len(self._buffer) * 8 + self._bit_position


class BitReader:
    """Reader for bit-level data input."""

    def __init__(self, data: bytes):
        self._data = data
        self._byte_position: int = 0
        self._bit_position: int = 0

    def read_bit(self) -> int:
        """Read a single bit."""
        if self._byte_position >= len(self._data):
            raise EOFError("End of data reached")

        bit = (self._data[self._byte_position] >> (7 - self._bit_position)) & 1
        self._bit_position += 1

        if self._bit_position == 8:
            self._byte_position += 1
            self._bit_position = 0

        return bit

    def read_bits(self, num_bits: int) -> int:
        """Read multiple bits and return as integer (MSB first)."""
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value

    def read_byte(self) -> int:
        """Read a full byte."""
        return self.read_bits(8)

    def read_uint16(self) -> int:
        """Read a 16-bit unsigned integer (big-endian)."""
        return self.read_bits(16)

    def read_uint32(self) -> int:
        """Read a 32-bit unsigned integer (big-endian)."""
        return self.read_bits(32)

    def read_int16(self) -> int:
        """Read a 16-bit signed integer (big-endian)."""
        value = self.read_uint16()
        if value >= (1 << 15):
            value -= (1 << 16)
        return value

    def read_int32(self) -> int:
        """Read a 32-bit signed integer (big-endian)."""
        value = self.read_uint32()
        if value >= (1 << 31):
            value -= (1 << 32)
        return value

    def read_float(self) -> float:
        """Read a 32-bit float."""
        data = bytes([self.read_byte() for _ in range(4)])
        return struct.unpack('>f', data)[0]

    def align(self) -> None:
        """Align to byte boundary."""
        if self._bit_position > 0:
            self._byte_position += 1
            self._bit_position = 0

    def get_position(self) -> int:
        """Get current bit position."""
        return self._byte_position * 8 + self._bit_position

    def has_more(self) -> bool:
        """Check if there is more data to read."""
        return self._byte_position < len(self._data)

    def remaining_bits(self) -> int:
        """Get number of remaining bits."""
        total_bits = len(self._data) * 8
        current_pos = self._byte_position * 8 + self._bit_position
        return total_bits - current_pos
