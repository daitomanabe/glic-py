"""
Color space conversion utilities.
Supports 16 different color spaces with bidirectional RGB conversion.
"""

import math
import numpy as np
from typing import Tuple
from .config import ColorSpace


def clamp(value: float, min_val: float = 0.0, max_val: float = 255.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def clamp_int(value: float) -> int:
    """Clamp and convert to int."""
    return int(clamp(value, 0, 255))


# ============ RGB (identity) ============

def rgb_to_rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (r, g, b)


def rgb_from_rgb(c0: int, c1: int, c2: int) -> Tuple[int, int, int]:
    return (c0, c1, c2)


# ============ HSB/HSV ============

def rgb_to_hsb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to HSB (Hue, Saturation, Brightness)."""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    # Brightness
    brightness = max_c

    # Saturation
    if max_c == 0:
        saturation = 0.0
    else:
        saturation = delta / max_c

    # Hue
    if delta == 0:
        hue = 0.0
    elif max_c == r_norm:
        hue = 60 * (((g_norm - b_norm) / delta) % 6)
    elif max_c == g_norm:
        hue = 60 * (((b_norm - r_norm) / delta) + 2)
    else:
        hue = 60 * (((r_norm - g_norm) / delta) + 4)

    if hue < 0:
        hue += 360

    return (
        clamp_int(hue * 255 / 360),
        clamp_int(saturation * 255),
        clamp_int(brightness * 255)
    )


def hsb_to_rgb(h: int, s: int, b: int) -> Tuple[int, int, int]:
    """Convert HSB to RGB."""
    hue = h * 360 / 255.0
    saturation = s / 255.0
    brightness = b / 255.0

    c = brightness * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = brightness - c

    if hue < 60:
        r, g, b_val = c, x, 0
    elif hue < 120:
        r, g, b_val = x, c, 0
    elif hue < 180:
        r, g, b_val = 0, c, x
    elif hue < 240:
        r, g, b_val = 0, x, c
    elif hue < 300:
        r, g, b_val = x, 0, c
    else:
        r, g, b_val = c, 0, x

    return (
        clamp_int((r + m) * 255),
        clamp_int((g + m) * 255),
        clamp_int((b_val + m) * 255)
    )


# ============ HWB ============

def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to HWB (Hue, Whiteness, Blackness)."""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    # Hue (same as HSB)
    if delta == 0:
        hue = 0.0
    elif max_c == r_norm:
        hue = 60 * (((g_norm - b_norm) / delta) % 6)
    elif max_c == g_norm:
        hue = 60 * (((b_norm - r_norm) / delta) + 2)
    else:
        hue = 60 * (((r_norm - g_norm) / delta) + 4)

    if hue < 0:
        hue += 360

    whiteness = min_c
    blackness = 1 - max_c

    return (
        clamp_int(hue * 255 / 360),
        clamp_int(whiteness * 255),
        clamp_int(blackness * 255)
    )


def hwb_to_rgb(h: int, w: int, b: int) -> Tuple[int, int, int]:
    """Convert HWB to RGB."""
    hue = h * 360 / 255.0
    whiteness = w / 255.0
    blackness = b / 255.0

    # Normalize whiteness and blackness
    total = whiteness + blackness
    if total > 1:
        whiteness /= total
        blackness /= total

    # Convert via HSB
    brightness = 1 - blackness
    saturation = 1 - whiteness / brightness if brightness > 0 else 0

    c = brightness * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = brightness - c

    if hue < 60:
        r, g, b_val = c, x, 0
    elif hue < 120:
        r, g, b_val = x, c, 0
    elif hue < 180:
        r, g, b_val = 0, c, x
    elif hue < 240:
        r, g, b_val = 0, x, c
    elif hue < 300:
        r, g, b_val = x, 0, c
    else:
        r, g, b_val = c, 0, x

    return (
        clamp_int((r + m) * 255),
        clamp_int((g + m) * 255),
        clamp_int((b_val + m) * 255)
    )


# ============ OHTA ============

def rgb_to_ohta(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to OHTA color space (I1, I2, I3)."""
    i1 = (r + g + b) / 3.0
    i2 = (r - b) / 2.0 + 128
    i3 = (2 * g - r - b) / 4.0 + 128

    return (clamp_int(i1), clamp_int(i2), clamp_int(i3))


def ohta_to_rgb(i1: int, i2: int, i3: int) -> Tuple[int, int, int]:
    """Convert OHTA to RGB."""
    i2_norm = i2 - 128
    i3_norm = i3 - 128

    r = i1 + i2_norm - 2 * i3_norm / 3.0
    g = i1 + 4 * i3_norm / 3.0
    b = i1 - i2_norm - 2 * i3_norm / 3.0

    return (clamp_int(r), clamp_int(g), clamp_int(b))


# ============ CMY ============

def rgb_to_cmy(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to CMY."""
    return (255 - r, 255 - g, 255 - b)


def cmy_to_rgb(c: int, m: int, y: int) -> Tuple[int, int, int]:
    """Convert CMY to RGB."""
    return (255 - c, 255 - m, 255 - y)


# ============ XYZ ============

def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to CIE XYZ."""
    # Gamma correction
    def gamma(v):
        v = v / 255.0
        if v > 0.04045:
            return ((v + 0.055) / 1.055) ** 2.4
        return v / 12.92

    r_lin = gamma(r)
    g_lin = gamma(g)
    b_lin = gamma(b)

    # RGB to XYZ matrix
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    return (
        clamp_int(x * 255),
        clamp_int(y * 255),
        clamp_int(z * 255)
    )


def xyz_to_rgb(x: int, y: int, z: int) -> Tuple[int, int, int]:
    """Convert CIE XYZ to RGB."""
    x_norm = x / 255.0
    y_norm = y / 255.0
    z_norm = z / 255.0

    # XYZ to RGB matrix
    r_lin = x_norm * 3.2404542 + y_norm * -1.5371385 + z_norm * -0.4985314
    g_lin = x_norm * -0.9692660 + y_norm * 1.8760108 + z_norm * 0.0415560
    b_lin = x_norm * 0.0556434 + y_norm * -0.2040259 + z_norm * 1.0572252

    # Inverse gamma
    def inv_gamma(v):
        if v > 0.0031308:
            return 1.055 * (v ** (1 / 2.4)) - 0.055
        return 12.92 * v

    return (
        clamp_int(inv_gamma(r_lin) * 255),
        clamp_int(inv_gamma(g_lin) * 255),
        clamp_int(inv_gamma(b_lin) * 255)
    )


# ============ YXY ============

def rgb_to_yxy(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to CIE Yxy."""
    x, y, z = rgb_to_xyz(r, g, b)
    x_norm = x / 255.0
    y_norm = y / 255.0
    z_norm = z / 255.0

    total = x_norm + y_norm + z_norm
    if total == 0:
        return (0, 128, 128)

    yy = y_norm
    xx = x_norm / total
    yy2 = y_norm / total

    return (
        clamp_int(yy * 255),
        clamp_int(xx * 255),
        clamp_int(yy2 * 255)
    )


def yxy_to_rgb(yy: int, x: int, y: int) -> Tuple[int, int, int]:
    """Convert CIE Yxy to RGB."""
    y_big = yy / 255.0
    x_small = x / 255.0
    y_small = y / 255.0

    if y_small == 0:
        return (0, 0, 0)

    x_norm = x_small * y_big / y_small
    z_norm = (1 - x_small - y_small) * y_big / y_small

    return xyz_to_rgb(
        clamp_int(x_norm * 255),
        clamp_int(y_big * 255),
        clamp_int(z_norm * 255)
    )


# ============ LAB ============

def rgb_to_lab(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to CIE LAB."""
    # First convert to XYZ
    x, y, z = rgb_to_xyz(r, g, b)

    # Reference white (D65)
    x_ref = 95.047 / 100.0
    y_ref = 100.0 / 100.0
    z_ref = 108.883 / 100.0

    x_norm = (x / 255.0) / x_ref
    y_norm = (y / 255.0) / y_ref
    z_norm = (z / 255.0) / z_ref

    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        return (903.3 * t + 16) / 116

    l = 116 * f(y_norm) - 16
    a = 500 * (f(x_norm) - f(y_norm))
    b_val = 200 * (f(y_norm) - f(z_norm))

    return (
        clamp_int(l * 255 / 100),
        clamp_int(a + 128),
        clamp_int(b_val + 128)
    )


def lab_to_rgb(l: int, a: int, b: int) -> Tuple[int, int, int]:
    """Convert CIE LAB to RGB."""
    l_norm = l * 100 / 255.0
    a_norm = a - 128
    b_norm = b - 128

    # Reference white (D65)
    x_ref = 95.047 / 100.0
    y_ref = 100.0 / 100.0
    z_ref = 108.883 / 100.0

    fy = (l_norm + 16) / 116
    fx = a_norm / 500 + fy
    fz = fy - b_norm / 200

    def f_inv(t):
        if t ** 3 > 0.008856:
            return t ** 3
        return (116 * t - 16) / 903.3

    x = f_inv(fx) * x_ref
    y = f_inv(fy) * y_ref
    z = f_inv(fz) * z_ref

    return xyz_to_rgb(
        clamp_int(x * 255),
        clamp_int(y * 255),
        clamp_int(z * 255)
    )


# ============ LUV ============

def rgb_to_luv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to CIE LUV."""
    x, y, z = rgb_to_xyz(r, g, b)
    x_norm = x / 255.0
    y_norm = y / 255.0
    z_norm = z / 255.0

    # Reference white
    x_ref = 95.047 / 100.0
    y_ref = 100.0 / 100.0
    z_ref = 108.883 / 100.0

    u_ref = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
    v_ref = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

    denom = x_norm + 15 * y_norm + 3 * z_norm
    if denom == 0:
        u_prime = 0
        v_prime = 0
    else:
        u_prime = 4 * x_norm / denom
        v_prime = 9 * y_norm / denom

    yr = y_norm / y_ref
    if yr > 0.008856:
        l = 116 * (yr ** (1/3)) - 16
    else:
        l = 903.3 * yr

    u = 13 * l * (u_prime - u_ref)
    v = 13 * l * (v_prime - v_ref)

    return (
        clamp_int(l * 255 / 100),
        clamp_int(u + 128),
        clamp_int(v + 128)
    )


def luv_to_rgb(l: int, u: int, v: int) -> Tuple[int, int, int]:
    """Convert CIE LUV to RGB."""
    l_norm = l * 100 / 255.0
    u_norm = u - 128
    v_norm = v - 128

    # Reference white
    x_ref = 95.047 / 100.0
    y_ref = 100.0 / 100.0
    z_ref = 108.883 / 100.0

    u_ref = 4 * x_ref / (x_ref + 15 * y_ref + 3 * z_ref)
    v_ref = 9 * y_ref / (x_ref + 15 * y_ref + 3 * z_ref)

    if l_norm == 0:
        return (0, 0, 0)

    u_prime = u_norm / (13 * l_norm) + u_ref
    v_prime = v_norm / (13 * l_norm) + v_ref

    if l_norm > 8:
        y = (((l_norm + 16) / 116) ** 3) * y_ref
    else:
        y = l_norm * y_ref / 903.3

    if v_prime == 0:
        return (0, 0, 0)

    x = y * 9 * u_prime / (4 * v_prime)
    z = y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

    return xyz_to_rgb(
        clamp_int(x * 255),
        clamp_int(y * 255),
        clamp_int(z * 255)
    )


# ============ HCL ============

def rgb_to_hcl(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to HCL (Hue, Chroma, Luminance)."""
    l, a, b_lab = rgb_to_lab(r, g, b)
    a_norm = a - 128
    b_norm = b_lab - 128

    c = math.sqrt(a_norm ** 2 + b_norm ** 2)
    h = math.atan2(b_norm, a_norm)
    if h < 0:
        h += 2 * math.pi

    return (
        clamp_int(h * 255 / (2 * math.pi)),
        clamp_int(c),
        l
    )


def hcl_to_rgb(h: int, c: int, l: int) -> Tuple[int, int, int]:
    """Convert HCL to RGB."""
    h_rad = h * 2 * math.pi / 255.0

    a = c * math.cos(h_rad)
    b = c * math.sin(h_rad)

    return lab_to_rgb(l, clamp_int(a + 128), clamp_int(b + 128))


# ============ YUV ============

def rgb_to_yuv(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to YUV."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b + 128
    v = 0.615 * r - 0.515 * g - 0.100 * b + 128

    return (clamp_int(y), clamp_int(u), clamp_int(v))


def yuv_to_rgb(y: int, u: int, v: int) -> Tuple[int, int, int]:
    """Convert YUV to RGB."""
    u_norm = u - 128
    v_norm = v - 128

    r = y + 1.140 * v_norm
    g = y - 0.395 * u_norm - 0.581 * v_norm
    b = y + 2.032 * u_norm

    return (clamp_int(r), clamp_int(g), clamp_int(b))


# ============ YPbPr ============

def rgb_to_ypbpr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to YPbPr."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    pb = -0.169 * r - 0.331 * g + 0.500 * b + 128
    pr = 0.500 * r - 0.419 * g - 0.081 * b + 128

    return (clamp_int(y), clamp_int(pb), clamp_int(pr))


def ypbpr_to_rgb(y: int, pb: int, pr: int) -> Tuple[int, int, int]:
    """Convert YPbPr to RGB."""
    pb_norm = pb - 128
    pr_norm = pr - 128

    r = y + 1.402 * pr_norm
    g = y - 0.344 * pb_norm - 0.714 * pr_norm
    b = y + 1.772 * pb_norm

    return (clamp_int(r), clamp_int(g), clamp_int(b))


# ============ YCbCr ============

def rgb_to_ycbcr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to YCbCr."""
    y = 16 + 65.481 * r / 255 + 128.553 * g / 255 + 24.966 * b / 255
    cb = 128 - 37.797 * r / 255 - 74.203 * g / 255 + 112.0 * b / 255
    cr = 128 + 112.0 * r / 255 - 93.786 * g / 255 - 18.214 * b / 255

    return (clamp_int(y), clamp_int(cb), clamp_int(cr))


def ycbcr_to_rgb(y: int, cb: int, cr: int) -> Tuple[int, int, int]:
    """Convert YCbCr to RGB."""
    y_norm = y - 16
    cb_norm = cb - 128
    cr_norm = cr - 128

    r = 298.082 * y_norm / 256 + 408.583 * cr_norm / 256
    g = 298.082 * y_norm / 256 - 100.291 * cb_norm / 256 - 208.120 * cr_norm / 256
    b = 298.082 * y_norm / 256 + 516.412 * cb_norm / 256

    return (clamp_int(r), clamp_int(g), clamp_int(b))


# ============ YDbDr ============

def rgb_to_ydbdr(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to YDbDr."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    db = -0.450 * r - 0.883 * g + 1.333 * b + 128
    dr = -1.333 * r + 1.116 * g + 0.217 * b + 128

    return (clamp_int(y), clamp_int(db), clamp_int(dr))


def ydbdr_to_rgb(y: int, db: int, dr: int) -> Tuple[int, int, int]:
    """Convert YDbDr to RGB."""
    db_norm = db - 128
    dr_norm = dr - 128

    r = y + 0.000092 * db_norm - 0.525912 * dr_norm
    g = y - 0.129132 * db_norm + 0.267899 * dr_norm
    b = y + 0.664679 * db_norm - 0.000079 * dr_norm

    return (clamp_int(r), clamp_int(g), clamp_int(b))


# ============ GS (Grayscale) ============

def rgb_to_gs(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to Grayscale (luminance in all channels)."""
    gray = int(0.299 * r + 0.587 * g + 0.114 * b)
    return (gray, gray, gray)


def gs_to_rgb(g1: int, g2: int, g3: int) -> Tuple[int, int, int]:
    """Convert Grayscale to RGB (average of channels)."""
    gray = (g1 + g2 + g3) // 3
    return (gray, gray, gray)


# ============ RGGBG (R-G, G, B-G) ============

def rgb_to_rggbg(r: int, g: int, b: int) -> Tuple[int, int, int]:
    """Convert RGB to R-G, G, B-G."""
    rg = r - g + 128
    bg = b - g + 128
    return (clamp_int(rg), g, clamp_int(bg))


def rggbg_to_rgb(rg: int, g: int, bg: int) -> Tuple[int, int, int]:
    """Convert R-G, G, B-G to RGB."""
    r = rg - 128 + g
    b = bg - 128 + g
    return (clamp_int(r), g, clamp_int(b))


# ============ Conversion dispatcher ============

_TO_CONVERTERS = {
    ColorSpace.RGB: rgb_to_rgb,
    ColorSpace.HSB: rgb_to_hsb,
    ColorSpace.HWB: rgb_to_hwb,
    ColorSpace.OHTA: rgb_to_ohta,
    ColorSpace.CMY: rgb_to_cmy,
    ColorSpace.XYZ: rgb_to_xyz,
    ColorSpace.YXY: rgb_to_yxy,
    ColorSpace.LAB: rgb_to_lab,
    ColorSpace.LUV: rgb_to_luv,
    ColorSpace.HCL: rgb_to_hcl,
    ColorSpace.YUV: rgb_to_yuv,
    ColorSpace.YPbPr: rgb_to_ypbpr,
    ColorSpace.YCbCr: rgb_to_ycbcr,
    ColorSpace.YDbDr: rgb_to_ydbdr,
    ColorSpace.GS: rgb_to_gs,
    ColorSpace.RGGBG: rgb_to_rggbg,
}

_FROM_CONVERTERS = {
    ColorSpace.RGB: rgb_from_rgb,
    ColorSpace.HSB: hsb_to_rgb,
    ColorSpace.HWB: hwb_to_rgb,
    ColorSpace.OHTA: ohta_to_rgb,
    ColorSpace.CMY: cmy_to_rgb,
    ColorSpace.XYZ: xyz_to_rgb,
    ColorSpace.YXY: yxy_to_rgb,
    ColorSpace.LAB: lab_to_rgb,
    ColorSpace.LUV: luv_to_rgb,
    ColorSpace.HCL: hcl_to_rgb,
    ColorSpace.YUV: yuv_to_rgb,
    ColorSpace.YPbPr: ypbpr_to_rgb,
    ColorSpace.YCbCr: ycbcr_to_rgb,
    ColorSpace.YDbDr: ydbdr_to_rgb,
    ColorSpace.GS: gs_to_rgb,
    ColorSpace.RGGBG: rggbg_to_rgb,
}


def convert_to_colorspace(r: int, g: int, b: int, colorspace: ColorSpace) -> Tuple[int, int, int]:
    """Convert RGB to the specified color space."""
    converter = _TO_CONVERTERS.get(colorspace)
    if converter is None:
        raise ValueError(f"Unknown color space: {colorspace}")
    return converter(r, g, b)


def convert_from_colorspace(c0: int, c1: int, c2: int, colorspace: ColorSpace) -> Tuple[int, int, int]:
    """Convert from the specified color space back to RGB."""
    converter = _FROM_CONVERTERS.get(colorspace)
    if converter is None:
        raise ValueError(f"Unknown color space: {colorspace}")
    return converter(c0, c1, c2)


def convert_image_to_colorspace(image: np.ndarray, colorspace: ColorSpace) -> np.ndarray:
    """Convert an entire image from RGB to the specified color space."""
    if colorspace == ColorSpace.RGB:
        return image.copy()

    result = np.zeros_like(image)
    converter = _TO_CONVERTERS[colorspace]

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r, g, b = image[y, x, :3]
            result[y, x, :3] = converter(int(r), int(g), int(b))

    return result


def convert_image_from_colorspace(image: np.ndarray, colorspace: ColorSpace) -> np.ndarray:
    """Convert an entire image from the specified color space back to RGB."""
    if colorspace == ColorSpace.RGB:
        return image.copy()

    result = np.zeros_like(image)
    converter = _FROM_CONVERTERS[colorspace]

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            c0, c1, c2 = image[y, x, :3]
            result[y, x, :3] = converter(int(c0), int(c1), int(c2))

    return result
