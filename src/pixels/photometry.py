"""Brightness and contrast: `out = alpha * in + beta`, done wrong and right.

Both knobs are the same one-line machine.  `beta` shifts every pixel (that is
brightness) and `alpha` scales it (that is contrast).  Everything that goes
wrong here goes wrong because that arithmetic is performed *inside* uint8,
where it wraps, or because it is performed correctly and then handed to a
function that takes an absolute value.

Nothing in this module is hard.  All of it is easy to get silently wrong, which
is why each wrong version is kept and tested rather than deleted.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "brighten_wrapping",
    "brighten_clip_too_late",
    "brighten_saturating",
    "contrast_naive",
    "contrast_pivoted",
    "pivot_beta",
    "convert_scale_abs_reflects_below",
]


def brighten_wrapping(img: np.ndarray, beta: int) -> np.ndarray:
    """`img + beta` in uint8. The highlights come out black."""
    return img + np.full_like(img, np.uint8(beta % 256))


def brighten_clip_too_late(img: np.ndarray, beta: int) -> np.ndarray:
    """`np.clip(img + beta, 0, 255)` -- the fix that is not a fix.

    Python evaluates the inner expression first, so `img + beta` has already
    wrapped by the time `np.clip` runs.  Clipping 59 into [0, 255] leaves 59.
    `np.clip` cannot undo a wrap because it cannot tell one happened: the
    wrapped value is a perfectly legal uint8.

    This function exists purely so a test can assert it differs from
    `brighten_saturating` on exactly the pixels that wrapped.
    """
    return np.clip(img + np.full_like(img, np.uint8(beta % 256)), 0, 255)


def brighten_saturating(img: np.ndarray, beta: int) -> np.ndarray:
    """Widen, add, clip, narrow -- in that order.

    int16 is the smallest type that holds 255 + 255 and also -255.  float32
    would also work and costs 4 bytes per pixel instead of 2; on a 1080p colour
    frame that is 24.9 MB against 12.4 MB, which is the whole argument for
    picking the narrowest type that fits.
    """
    return np.clip(img.astype(np.int16) + int(beta), 0, 255).astype(np.uint8)


def pivot_beta(alpha: float, pivot: float = 128.0) -> float:
    """The `beta` that makes `alpha` a pure contrast knob.

    Multiplying scales *about zero*, so `1.6 * x` moves every pixel up as well
    as apart -- nothing gets darker, which is not what "more contrast" means.
    Rearranging `alpha*(x - pivot) + pivot` gives `alpha*x + pivot*(1 - alpha)`,
    so the pivot costs no new machinery: it just fixes `beta` instead of
    leaving it free.  At alpha = 1.6 that is beta = -76.8.

    The signature of a correct pivot is that mid-grey does not move.
    """
    return float(pivot * (1.0 - alpha))


def _to_uint8(x: np.ndarray) -> np.ndarray:
    """Clip into range, then round -- in that order, and round rather than truncate.

    `astype(np.uint8)` truncates, which is a systematic half-level darkening of
    every pixel in the image.  It is invisible on one operation and accumulates
    into visible banding over a chain of them.  Adding 0.5 before the floor is
    round-half-up, which is what OpenCV's `saturate_cast` does, so our float
    path and the library's integer path agree exactly instead of drifting by
    one level -- and `tests/test_photometry.py` asserts that agreement.
    """
    return np.floor(np.clip(x, 0, 255) + 0.5).astype(np.uint8)


def contrast_naive(img: np.ndarray, alpha: float) -> np.ndarray:
    """Scale with no pivot: a brightness control wearing a contrast label."""
    return _to_uint8(img.astype(np.float64) * alpha)


def contrast_pivoted(img: np.ndarray, alpha: float, pivot: float = 128.0) -> np.ndarray:
    """Scale about mid-grey: darks go darker, brights go brighter, 128 stays."""
    return _to_uint8(img.astype(np.float64) * alpha + pivot_beta(alpha, pivot))


def convert_scale_abs_reflects_below(alpha: float, beta: float) -> float:
    """The input value below which `cv2.convertScaleAbs` gives the wrong answer.

    `convertScaleAbs` computes `saturate_cast<uchar>(|alpha*x + beta|)`.  The
    absolute value is the point -- a pixel that should clip to 0 is *reflected*
    off zero and comes back bright.  That happens for every x with
    `alpha*x + beta < 0`, i.e. `x < -beta/alpha`.

    Which means it is safe exactly when beta >= 0, and pivoted contrast makes
    beta negative by construction.  Its legitimate job is displaying a signed
    gradient, where the absolute value is the feature you wanted.
    """
    if alpha <= 0 or beta >= 0:
        return 0.0
    return float(-beta / alpha)
