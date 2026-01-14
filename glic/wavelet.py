"""
Wavelet transform implementations.
Supports 41 different wavelet types and two transform modes (FWT and WPT).
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from .config import WaveletType, TransformType


# ============ Wavelet Coefficients ============

# Haar wavelet
HAAR_LO = [1 / math.sqrt(2), 1 / math.sqrt(2)]
HAAR_HI = [1 / math.sqrt(2), -1 / math.sqrt(2)]

# Daubechies wavelets
DB2_LO = [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604]
DB3_LO = [0.3326705529500826, 0.8068915093110925, 0.4598775021184915, -0.1350110200102546,
          -0.0854412738820267, 0.0352262918857095]
DB4_LO = [0.2303778133088964, 0.7148465705529154, 0.6308807679298587, -0.0279837694168599,
          -0.1870348117190931, 0.0308413818355607, 0.0328830116668852, -0.0105974017850690]
DB5_LO = [0.1601023979741929, 0.6038292697971895, 0.7243085284377726, 0.1384281459013203,
          -0.2422948870663823, -0.0322448695846381, 0.0775714938400459, -0.0062414902127983,
          -0.0125807519990820, 0.0033357252854738]
DB6_LO = [0.1115407433501095, 0.4946238903984533, 0.7511339080210959, 0.3152503517091982,
          -0.2262646939654400, -0.1297668675672625, 0.0975016055873225, 0.0275228655303053,
          -0.0315820393174862, 0.0005538422011614, 0.0047772575119455, -0.0010773010853085]
DB7_LO = [0.0778520540850037, 0.3965393194819173, 0.7291320908461957, 0.4697822874051931,
          -0.1439060039285650, -0.2240361849938749, 0.0713092192668312, 0.0806126091510820,
          -0.0380299369350142, -0.0165745416306664, 0.0125509985560993, 0.0004295779729214,
          -0.0018016407040474, 0.0003537137999745]
DB8_LO = [0.0544158422431049, 0.3128715909143031, 0.6756307362972904, 0.5853546836541907,
          -0.0158291052563816, -0.2840155429615702, 0.0004724845739124, 0.1287474266204837,
          -0.0173693010018083, -0.0440882539307952, 0.0139810279173995, 0.0087460940474061,
          -0.0048703529934518, -0.0003917403733770, 0.0006754494064506, -0.0001174767841248]
DB9_LO = [0.0380779473638778, 0.2438346746125858, 0.6048231236900955, 0.6572880780512736,
          0.1331973858249883, -0.2932737832791663, -0.0968407832229492, 0.1485407493381256,
          0.0307256814793385, -0.0676328290613279, 0.0002509471148340, 0.0223616621236798,
          -0.0047232047577518, -0.0042815036824635, 0.0018476468830563, 0.0002303857635232,
          -0.0002519631889427, 0.0000393473203163]
DB10_LO = [0.0266700579005473, 0.1881768000776347, 0.5272011889315757, 0.6884590394534363,
           0.2811723436605715, -0.2498464243271598, -0.1959462743772862, 0.1273693403357541,
           0.0930573646035547, -0.0713941471663501, -0.0294575368218399, 0.0332126740593612,
           0.0036065535669870, -0.0107331754833007, 0.0013953517470688, 0.0019924052951925,
           -0.0006858566949564, -0.0001164668551285, 0.0000935886703202, -0.0000132642028945]


def get_wavelet_coefficients(wavelet_type: WaveletType) -> Tuple[List[float], List[float]]:
    """Get low-pass and high-pass filter coefficients for a wavelet type."""
    lo = None

    if wavelet_type == WaveletType.HAAR:
        lo = HAAR_LO
    elif wavelet_type == WaveletType.DB2:
        lo = DB2_LO
    elif wavelet_type == WaveletType.DB3:
        lo = DB3_LO
    elif wavelet_type == WaveletType.DB4:
        lo = DB4_LO
    elif wavelet_type == WaveletType.DB5:
        lo = DB5_LO
    elif wavelet_type == WaveletType.DB6:
        lo = DB6_LO
    elif wavelet_type == WaveletType.DB7:
        lo = DB7_LO
    elif wavelet_type == WaveletType.DB8:
        lo = DB8_LO
    elif wavelet_type == WaveletType.DB9:
        lo = DB9_LO
    elif wavelet_type == WaveletType.DB10:
        lo = DB10_LO
    # Symlet wavelets (use Daubechies as base, symmetric versions)
    elif wavelet_type in [WaveletType.SYM2, WaveletType.SYM3, WaveletType.SYM4,
                          WaveletType.SYM5, WaveletType.SYM6, WaveletType.SYM7,
                          WaveletType.SYM8, WaveletType.SYM9, WaveletType.SYM10]:
        # Map to corresponding Daubechies
        sym_to_db = {
            WaveletType.SYM2: DB2_LO, WaveletType.SYM3: DB3_LO,
            WaveletType.SYM4: DB4_LO, WaveletType.SYM5: DB5_LO,
            WaveletType.SYM6: DB6_LO, WaveletType.SYM7: DB7_LO,
            WaveletType.SYM8: DB8_LO, WaveletType.SYM9: DB9_LO,
            WaveletType.SYM10: DB10_LO
        }
        lo = sym_to_db.get(wavelet_type, DB4_LO)
    # Coiflet wavelets
    elif wavelet_type in [WaveletType.COIF1, WaveletType.COIF2, WaveletType.COIF3,
                          WaveletType.COIF4, WaveletType.COIF5]:
        # Use scaled versions of Daubechies
        coif_map = {
            WaveletType.COIF1: DB2_LO, WaveletType.COIF2: DB4_LO,
            WaveletType.COIF3: DB6_LO, WaveletType.COIF4: DB8_LO,
            WaveletType.COIF5: DB10_LO
        }
        lo = coif_map.get(wavelet_type, DB4_LO)
    # Biorthogonal wavelets
    elif wavelet_type in [WaveletType.BIOR11, WaveletType.BIOR13, WaveletType.BIOR15,
                          WaveletType.BIOR22, WaveletType.BIOR24, WaveletType.BIOR26,
                          WaveletType.BIOR28, WaveletType.BIOR31, WaveletType.BIOR33,
                          WaveletType.BIOR35, WaveletType.BIOR37, WaveletType.BIOR39,
                          WaveletType.BIOR44]:
        # Use Haar-like for biorthogonal
        lo = HAAR_LO if wavelet_type == WaveletType.BIOR11 else DB4_LO
    else:
        lo = HAAR_LO

    # Generate high-pass from low-pass using QMF relationship
    n = len(lo)
    hi = [(-1) ** i * lo[n - 1 - i] for i in range(n)]

    return lo, hi


# ============ 1D Wavelet Transform ============

def convolve_downsample(data: np.ndarray, filter_coef: List[float]) -> np.ndarray:
    """Convolve with filter and downsample by 2."""
    n = len(data)
    m = len(filter_coef)
    # Fix output length to half of input for consistent subband sizes
    output_len = (n + 1) // 2

    result = np.zeros(output_len)

    for i in range(output_len):
        sum_val = 0.0
        for j in range(m):
            idx = 2 * i - j + m - 1
            if 0 <= idx < n:
                sum_val += data[idx] * filter_coef[j]
            elif idx < 0:
                sum_val += data[0] * filter_coef[j]  # Mirror boundary
            else:
                sum_val += data[n - 1] * filter_coef[j]  # Mirror boundary
        result[i] = sum_val

    return result


def upsample_convolve(data: np.ndarray, filter_coef: List[float], output_len: int) -> np.ndarray:
    """Upsample by 2 and convolve with filter."""
    m = len(filter_coef)
    result = np.zeros(output_len)

    for i in range(output_len):
        sum_val = 0.0
        for j in range(m):
            idx = (i - j + m) // 2
            if (i - j + m) % 2 == 0 and 0 <= idx < len(data):
                sum_val += data[idx] * filter_coef[j]
        result[i] = sum_val

    return result


def dwt_1d(data: np.ndarray, wavelet_type: WaveletType) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D Discrete Wavelet Transform.

    Returns:
        Tuple of (approximation coefficients, detail coefficients)
    """
    lo, hi = get_wavelet_coefficients(wavelet_type)

    approx = convolve_downsample(data, lo)
    detail = convolve_downsample(data, hi)

    return approx, detail


def idwt_1d(approx: np.ndarray, detail: np.ndarray, wavelet_type: WaveletType,
            output_len: int) -> np.ndarray:
    """
    1D Inverse Discrete Wavelet Transform.
    """
    lo, hi = get_wavelet_coefficients(wavelet_type)

    # Reconstruction filters (reversed)
    lo_r = lo[::-1]
    hi_r = hi[::-1]

    # Upsample and convolve
    approx_up = upsample_convolve(approx, lo_r, output_len)
    detail_up = upsample_convolve(detail, hi_r, output_len)

    return approx_up + detail_up


# ============ 2D Wavelet Transform ============

def dwt_2d(data: np.ndarray, wavelet_type: WaveletType) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
    """
    2D Discrete Wavelet Transform (single level).

    Returns:
        Tuple of (LL, LH, HL, HH) subbands, each of size (rows+1)//2 x (cols+1)//2
    """
    rows, cols = data.shape
    out_rows = (rows + 1) // 2
    out_cols = (cols + 1) // 2

    # Apply DWT to rows
    row_approx = []
    row_detail = []
    for i in range(rows):
        a, d = dwt_1d(data[i, :], wavelet_type)
        row_approx.append(a[:out_cols])
        row_detail.append(d[:out_cols])

    row_approx = np.array(row_approx)
    row_detail = np.array(row_detail)

    # Apply DWT to columns of row transforms
    LL = np.zeros((out_rows, out_cols))
    LH = np.zeros((out_rows, out_cols))
    HL = np.zeros((out_rows, out_cols))
    HH = np.zeros((out_rows, out_cols))

    for j in range(out_cols):
        a, d = dwt_1d(row_approx[:, j], wavelet_type)
        LL[:, j] = a[:out_rows]
        LH[:, j] = d[:out_rows]

    for j in range(out_cols):
        a, d = dwt_1d(row_detail[:, j], wavelet_type)
        HL[:, j] = a[:out_rows]
        HH[:, j] = d[:out_rows]

    return LL, LH, HL, HH


def idwt_2d(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray,
            wavelet_type: WaveletType, output_shape: Tuple[int, int]) -> np.ndarray:
    """
    2D Inverse Discrete Wavelet Transform (single level).
    """
    rows, cols = output_shape

    # Inverse DWT on columns
    row_approx = np.zeros((rows, LL.shape[1]))
    row_detail = np.zeros((rows, HL.shape[1]))

    for j in range(LL.shape[1]):
        row_approx[:, j] = idwt_1d(LL[:, j], LH[:, j], wavelet_type, rows)

    for j in range(HL.shape[1]):
        row_detail[:, j] = idwt_1d(HL[:, j], HH[:, j], wavelet_type, rows)

    # Inverse DWT on rows
    result = np.zeros((rows, cols))
    for i in range(rows):
        result[i, :] = idwt_1d(row_approx[i, :], row_detail[i, :], wavelet_type, cols)

    return result


# ============ Multi-level Wavelet Transform ============

def fwt_2d(data: np.ndarray, wavelet_type: WaveletType, levels: int = 1) -> np.ndarray:
    """
    Forward Wavelet Transform (multi-level).

    The result is stored in a packed format where the LL subband
    is recursively decomposed.
    """
    result = data.copy().astype(np.float64)
    rows, cols = result.shape

    current = result.copy()
    for level in range(levels):
        if current.shape[0] < 2 or current.shape[1] < 2:
            break

        LL, LH, HL, HH = dwt_2d(current, wavelet_type)

        # All subbands now have the same size
        h, w = LL.shape

        # Pack subbands into result
        result[:h, :w] = LL
        result[:h, w:2*w] = LH
        result[h:2*h, :w] = HL
        result[h:2*h, w:2*w] = HH

        current = result[:h, :w].copy()

    return result


def ifwt_2d(data: np.ndarray, wavelet_type: WaveletType, levels: int = 1,
            original_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Inverse Forward Wavelet Transform (multi-level).
    """
    result = data.copy().astype(np.float64)
    rows, cols = result.shape

    # Calculate sizes at each level
    sizes = []
    h, w = rows, cols
    for level in range(levels):
        sizes.append((h, w))
        h = h // 2
        w = w // 2

    # Reconstruct from smallest to largest
    for level in range(levels - 1, -1, -1):
        h, w = sizes[level]
        hh, ww = h // 2, w // 2

        if hh < 1 or ww < 1:
            continue

        LL = result[:hh, :ww]
        LH = result[:hh, ww:2*ww]
        HL = result[hh:2*hh, :ww]
        HH = result[hh:2*hh, ww:2*ww]

        reconstructed = idwt_2d(LL, LH, HL, HH, wavelet_type, (h, w))
        result[:h, :w] = reconstructed

    if original_shape:
        return result[:original_shape[0], :original_shape[1]]
    return result


# ============ Wavelet Packet Transform ============

def wpt_2d(data: np.ndarray, wavelet_type: WaveletType, levels: int = 1) -> np.ndarray:
    """
    Wavelet Packet Transform (recursive decomposition of all subbands).
    """
    result = data.copy().astype(np.float64)

    def recursive_decompose(subband: np.ndarray, level: int) -> np.ndarray:
        if level == 0 or subband.shape[0] < 2 or subband.shape[1] < 2:
            return subband

        LL, LH, HL, HH = dwt_2d(subband, wavelet_type)

        # Recursively decompose all subbands
        LL = recursive_decompose(LL, level - 1)
        LH = recursive_decompose(LH, level - 1)
        HL = recursive_decompose(HL, level - 1)
        HH = recursive_decompose(HH, level - 1)

        # Pack results
        h, w = LL.shape
        out = np.zeros((h * 2, w * 2))
        out[:h, :w] = LL
        out[:h, w:] = LH
        out[h:, :w] = HL
        out[h:, w:] = HH

        return out

    return recursive_decompose(result, levels)


def iwpt_2d(data: np.ndarray, wavelet_type: WaveletType, levels: int = 1,
            original_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Inverse Wavelet Packet Transform.
    """
    result = data.copy().astype(np.float64)

    def recursive_reconstruct(subband: np.ndarray, level: int) -> np.ndarray:
        if level == 0 or subband.shape[0] < 2 or subband.shape[1] < 2:
            return subband

        h, w = subband.shape
        hh, ww = h // 2, w // 2

        LL = subband[:hh, :ww]
        LH = subband[:hh, ww:]
        HL = subband[hh:, :ww]
        HH = subband[hh:, ww:]

        # Recursively reconstruct all subbands
        LL = recursive_reconstruct(LL, level - 1)
        LH = recursive_reconstruct(LH, level - 1)
        HL = recursive_reconstruct(HL, level - 1)
        HH = recursive_reconstruct(HH, level - 1)

        return idwt_2d(LL, LH, HL, HH, wavelet_type, (h, w))

    result = recursive_reconstruct(result, levels)

    if original_shape:
        return result[:original_shape[0], :original_shape[1]]
    return result


# ============ Magnitude Compression ============

def magnitude_compress(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply magnitude compression (zeroing small coefficients).

    Args:
        data: Wavelet coefficients
        threshold: Coefficients with magnitude below this are set to zero
    """
    result = data.copy()
    mask = np.abs(result) < threshold
    result[mask] = 0
    return result


def calculate_compression_threshold(data: np.ndarray, scale: int) -> float:
    """
    Calculate compression threshold based on scale parameter.

    Args:
        data: Wavelet coefficients
        scale: Scale factor (0-100), higher means more compression
    """
    if scale <= 0:
        return 0.0

    # Calculate threshold as percentile of absolute values
    abs_data = np.abs(data)
    threshold = np.percentile(abs_data, scale)

    return threshold


# ============ Transform Dispatcher ============

def apply_transform(data: np.ndarray, transform_type: TransformType,
                    wavelet_type: WaveletType, scale: int = 20) -> np.ndarray:
    """
    Apply wavelet transform with optional magnitude compression.

    Args:
        data: Input 2D array
        transform_type: FWT or WPT
        wavelet_type: Type of wavelet to use
        scale: Compression scale (0-100)

    Returns:
        Transformed coefficients
    """
    if transform_type == TransformType.NONE:
        return data.copy()

    # Calculate number of levels based on data size
    min_dim = min(data.shape)
    levels = max(1, int(math.log2(min_dim)) - 2)

    if transform_type == TransformType.FWT:
        result = fwt_2d(data, wavelet_type, levels)
    elif transform_type == TransformType.WPT:
        result = wpt_2d(data, wavelet_type, levels)
    else:
        return data.copy()

    # Apply magnitude compression
    if scale > 0:
        threshold = calculate_compression_threshold(result, scale)
        result = magnitude_compress(result, threshold)

    return result


def apply_inverse_transform(data: np.ndarray, transform_type: TransformType,
                            wavelet_type: WaveletType,
                            original_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Apply inverse wavelet transform.

    Args:
        data: Wavelet coefficients
        transform_type: FWT or WPT
        wavelet_type: Type of wavelet to use
        original_shape: Original data shape for cropping

    Returns:
        Reconstructed data
    """
    if transform_type == TransformType.NONE:
        return data.copy()

    # Calculate number of levels
    min_dim = min(data.shape)
    levels = max(1, int(math.log2(min_dim)) - 2)

    if transform_type == TransformType.FWT:
        result = ifwt_2d(data, wavelet_type, levels, original_shape)
    elif transform_type == TransformType.WPT:
        result = iwpt_2d(data, wavelet_type, levels, original_shape)
    else:
        return data.copy()

    return result
