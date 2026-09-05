"""Turning a grayscale image into a binary mask: global, Otsu, adaptive.

Every operation so far produced another grayscale image.  This is where a
decision gets made: each pixel becomes foreground or background and nothing in
between.  Three ways to decide, in increasing order of cleverness and cost:

    global    -- one number you pick.  Fast, and wrong the moment the lamp moves.
    Otsu      -- one number the computer picks, by scoring every candidate.
                 Still one number for the whole image.
    adaptive  -- a different number for every pixel, from its own neighbourhood.

Otsu is implemented here rather than called, because the criterion it optimises
is the interesting part and it is three lines once you see it.
"""

from __future__ import annotations

import numpy as np

from .separable import correlate_shift_separable

__all__ = [
    "histogram256",
    "within_class_variance",
    "between_class_variance_curve",
    "otsu_threshold",
    "global_threshold",
    "adaptive_threshold_mean",
    "iou",
    "foreground_fraction",
]


def histogram256(gray: np.ndarray) -> np.ndarray:
    """Counts of each of the 256 possible values, as int64.

    `np.bincount` with an explicit `minlength` rather than `np.histogram`: the
    values are already integers 0..255, so there is nothing to bin, and
    `np.histogram`'s float edges introduce a rounding question that does not
    need to exist.
    """
    if gray.dtype != np.uint8:
        raise TypeError("histogram256 expects uint8; a float image has no 256 levels")
    return np.bincount(gray.ravel(), minlength=256).astype(np.int64)


def within_class_variance(gray: np.ndarray, t: int) -> float | None:
    """Otsu's actual objective, written the slow honest way.

    Split the pixels at `t` using OpenCV's convention -- class 0 is `<= t`,
    class 1 is `> t` -- and return `w0*var0 + w1*var1`: the average spread
    *inside* the two groups, weighted by how many pixels each holds.

    A low score says both groups are narrow, which is what happens when the cut
    landed in a genuine trough of the histogram instead of slicing a hump apart.

    Returns None when one class is empty, because a variance over no pixels is
    not zero, it is undefined -- and treating it as zero makes t = 255 look
    like a perfect threshold.
    """
    v = gray.astype(np.float64).ravel()
    lo, hi = v[v <= t], v[v > t]
    if lo.size == 0 or hi.size == 0:
        return None
    w0, w1 = lo.size / v.size, hi.size / v.size
    return float(w0 * lo.var() + w1 * hi.var())


def between_class_variance_curve(gray: np.ndarray) -> np.ndarray:
    """`sigma_B^2` for every threshold 0..255, in one vectorised pass.

    The identity that makes this worth doing:

        sigma_total^2  =  sigma_within^2  +  sigma_between^2

    The total is fixed by the image alone; no choice of threshold moves it.
    Driving the within-class variance down and the between-class variance up
    are therefore one operation seen from two sides.  The between-class form
    is the one production code computes, because running sums of the histogram
    are all it needs -- no second sweep over the pixels.

    `sigma_B^2 = w0*w1*(mu0 - mu1)^2`.  Entries where a class is empty are set
    to -1 so they can never win the argmax.
    """
    hist = histogram256(gray).astype(np.float64)
    total = hist.sum()
    levels = np.arange(256, dtype=np.float64)

    n0 = np.cumsum(hist)                       # pixels with value <= t
    s0 = np.cumsum(hist * levels)              # their summed value
    n1 = total - n0
    s1 = s0[-1] - s0

    valid = (n0 > 0) & (n1 > 0)
    w0 = n0 / total
    w1 = n1 / total
    mu0 = np.divide(s0, n0, out=np.zeros(256), where=n0 > 0)
    mu1 = np.divide(s1, n1, out=np.zeros(256), where=n1 > 0)

    curve = w0 * w1 * (mu0 - mu1) ** 2
    return np.where(valid, curve, -1.0)


def otsu_threshold(gray: np.ndarray) -> int:
    """The threshold OpenCV's THRESH_OTSU returns, computed from scratch.

    `argmax` returns the FIRST index achieving the maximum, and that tie-break
    is not a detail.  On the classic worked example -- ink at 40 and 50, paper
    at 190 and 200 -- every threshold from 50 to 189 produces the identical
    split and therefore the identical score.  All of them are correct answers;
    OpenCV reports the lowest, so we must too or the tests comparing the two
    will fail on any image with a wide empty valley, which is every image Otsu
    is good at.
    """
    return int(np.argmax(between_class_variance_curve(gray)))


def global_threshold(gray: np.ndarray, t: int, invert: bool = False) -> np.ndarray:
    """`gray > t` as a 0/255 mask, or `gray <= t` when inverted.

    `invert=True` is the one you want for dark ink on bright paper: foreground
    is what is *below* the threshold.  OpenCV spells it THRESH_BINARY_INV, and
    getting it backwards is what makes morphology later appear to do the exact
    opposite of what it should -- erode grows your object because it is
    actually shrinking the white background around it.
    """
    mask = (gray <= t) if invert else (gray > t)
    return mask.astype(np.uint8) * 255


def adaptive_threshold_mean(gray: np.ndarray, block_size: int, c: int,
                            invert: bool = False) -> np.ndarray:
    """Per-pixel threshold = the mean of a `block_size` neighbourhood, minus `c`.

    The question changes from "is this pixel dark?" to "is this pixel darker
    than its surroundings?", and the second question survives a shadow because
    a shadow moves a pixel and its surroundings together.

    Three constraints, all of which OpenCV enforces with an assertion and all
    of which are really about meaning:

    * `block_size` must be odd, so the window has a centre pixel to belong to.
    * `block_size` should be a few times the width of the strokes you want to
      keep.  Too small and the window sits entirely inside the ink, the local
      mean *is* the ink, and the ink vanishes.
    * `c` must not be 0.  With `c = 0` every pixel is compared against its own
      neighbourhood mean, so a blank sheet of paper -- where the only variation
      is sensor noise -- comes back about half foreground.  `c` is the margin
      that says "darker by at least this much, not merely darker".

    The local mean is a box blur, and a box blur is separable, so this reuses
    the machinery from `pixels.separable`.  BORDER_REPLICATE matches what
    OpenCV's `boxFilter` does inside `adaptiveThreshold`.
    """
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size must be odd and >= 3")
    ones = np.ones(block_size, np.float64) / block_size
    local_mean = correlate_shift_separable(gray.astype(np.float64), ones, ones,
                                           border="replicate")
    if invert:
        mask = gray.astype(np.float64) <= local_mean - c
    else:
        mask = gray.astype(np.float64) > local_mean - c
    return mask.astype(np.uint8) * 255


def iou(mask: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two binary masks."""
    a, b = mask > 0, truth > 0
    union = int((a | b).sum())
    return 0.0 if union == 0 else float((a & b).sum()) / union


def foreground_fraction(mask: np.ndarray) -> float:
    """Fraction of the mask that is foreground -- the cheapest sanity guard.

    Otsu never refuses.  Hand it a histogram with no valley at all -- flat
    noise, a blank wall -- and it still returns a number, and that number
    splits the noise straight down the middle.  If this comes back near 0.5
    when you expected a small object, the threshold is meaningless and no
    amount of morphology downstream will save it.
    """
    return float((mask > 0).mean())
