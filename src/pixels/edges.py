"""Derivatives and edges: Sobel -> magnitude and direction -> NMS -> hysteresis.

An edge is a place where brightness changes fast, and "changes fast" is the
language of the derivative.  An image is a grid rather than a smooth function,
so the derivative is approximated with a convolution -- which is why this module
sits on top of `pixels.convolve` and does not import OpenCV for anything except
the comparison in `agreement`.

The whole of Canny is here in five separately testable pieces, because the
usual way this is taught -- one function called `canny` -- hides the two stages
that actually carry the idea.  Non-maximum suppression is why the output is one
pixel wide.  Hysteresis is why it is connected.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .convolve import correlate2d
from .separable import gaussian_kernel_1d, correlate_shift_separable

__all__ = [
    "SOBEL_X",
    "SOBEL_Y",
    "sobel",
    "magnitude",
    "direction",
    "direction_bins",
    "non_maximum_suppression",
    "hysteresis",
    "canny",
    "gaussian_blur",
    "agreement",
]

# Sobel-x differentiates ACROSS the columns and smooths DOWN the rows.  The 2
# in the middle row is the smoothing: without it a single noisy pixel owns the
# answer.  It is exactly [1, 2, 1] (a blur) outer-product [-1, 0, +1] (a
# difference), which is why `separable.is_separable` returns True for it.
#
# The naming is the reverse of what the picture suggests, and it is worth
# stating both halves every time: Sobel-x asks how much brightness changes as
# you step sideways, which is why it lights up on VERTICAL edges.  To measure how tall a fence is, you
# walk across it.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], np.float64)


def sobel(gray: np.ndarray, border: str = "reflect101") -> tuple[np.ndarray, np.ndarray]:
    """Return (gx, gy) as float64.

    float64, never uint8, and this is the single most common Sobel bug.  A
    bright-to-dark edge has a *negative* gradient; in uint8 every negative
    clips to 0 and that entire half of your edges disappears with no warning.
    On a test strip that steps from 200 down to 40, CV_64F reports -640 and
    CV_8U reports 0 -- ten edge pixels found versus zero.
    """
    gx = correlate2d(gray, SOBEL_X, border=border)
    gy = correlate2d(gray, SOBEL_Y, border=border)
    return gx, gy


def magnitude(gx: np.ndarray, gy: np.ndarray, norm: str = "L1") -> np.ndarray:
    """Edge strength.

    L2 is `sqrt(gx^2 + gy^2)` -- the true length of the gradient vector.
    L1 is `|gx| + |gy|`, which is cheaper and is what `cv2.Canny` uses unless
    you pass `L2gradient=True`.  We default to L1 so that `canny` below is
    directly comparable to OpenCV's.  The two genuinely differ, and in practice
    the choice between them is very rarely what is wrong with an edge map.
    """
    if norm == "L2":
        return np.hypot(gx, gy)
    if norm == "L1":
        return np.abs(gx) + np.abs(gy)
    raise ValueError("norm must be 'L1' or 'L2'")


def direction(gx: np.ndarray, gy: np.ndarray, fold: bool = True) -> np.ndarray:
    """Gradient direction in degrees, folded to [0, 180) by default.

    What this returns is the compass heading of steepest brightness increase.
    That heading is at right angles to the line you would draw along the edge,
    so the edge itself sits at `direction + 90`.  Mixing the two up is the
    standard screening question, and worse, it is what turns non-maximum
    suppression from a thinning step into a deletion step.

    The fold matters mechanically, not just conceptually.  atan2 returns
    (-180, 180]; a gradient of 20 degrees and one of 200 describe the same edge
    line seen from its two sides.  Without `% 180` the bin assignment flips as
    you cross the edge, NMS compares the wrong neighbour pair, and you get a
    broken dotted line that reads as a threshold problem.
    """
    ang = np.degrees(np.arctan2(gy, gx))
    return ang % 180.0 if fold else ang


def direction_bins(angle_deg: np.ndarray) -> np.ndarray:
    """Quantise a folded gradient angle into the four neighbour pairs.

        bin 0  gradient points right      -> compare left/right    -> edge is vertical
        bin 1  gradient points down-right -> compare the / diagonal -> edge at 135
        bin 2  gradient points down       -> compare up/down       -> edge is horizontal
        bin 3  gradient points down-left  -> compare the \\ diagonal -> edge at 45

    Four bins because a pixel has eight neighbours in four opposing pairs.
    Bin 0 wraps: angles below 22.5 and at or above 157.5 are both "rightish".
    """
    b = np.zeros(angle_deg.shape, np.uint8)
    b[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1
    b[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2
    b[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3
    return b


# (row, col) step ALONG the gradient for each bin -- i.e. across the edge.
_STEP = np.array([(0, 1), (1, 1), (1, 0), (1, -1)], np.int64)


def non_maximum_suppression(mag: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Thin fat gradient ridges down to their single brightest crest.

    A soft edge produces a magnitude ridge several pixels wide -- on the
    standard ramp drill, one edge becomes the five values 120, 320, 400, 240,
    40.  NMS keeps only the 400.

    The comparison is against the two neighbours ALONG THE GRADIENT, which is
    ACROSS the edge.  Compare along the edge instead and every pixel ties with
    its neighbours, nothing is ever strictly largest, and the output is
    completely empty -- with no error.  That is the difference between 3 pixels
    kept and 0 on the ramp drill.

    `>=` on one side and `>` on the other is the tie-break.  Two adjacent
    pixels of exactly equal magnitude cannot both be crests; this rule keeps
    the first and is what reproduces OpenCV's choice.  Flipping the two gives
    an equally valid Canny whose lines sit one pixel over.
    """
    bins = direction_bins(angle_deg)
    h, w = mag.shape
    out = np.zeros_like(mag)

    # Vectorised over the four bins rather than over pixels: the shifted
    # comparison is the same array operation for every pixel in a bin, so four
    # passes replace H*W interpreter round trips.  The interior-only slice is
    # why the border row and column are never kept, which matches OpenCV.
    core = (slice(1, h - 1), slice(1, w - 1))
    centre = mag[core]
    keep = np.zeros(centre.shape, bool)
    for b in range(4):
        di, dj = _STEP[b]
        fwd = mag[1 + di:h - 1 + di, 1 + dj:w - 1 + dj]
        bwd = mag[1 - di:h - 1 - di, 1 - dj:w - 1 - dj]
        keep |= (bins[core] == b) & (centre >= fwd) & (centre > bwd)
    out[core] = np.where(keep, centre, 0.0)
    return out


def hysteresis(thin: np.ndarray, low: float, high: float) -> np.ndarray:
    """Double threshold, then keep weak pixels that chain back to a strong one.

    A high bar to *begin* an edge and a low one to keep following it.  That is
    the entire answer to "why two thresholds":

        one threshold at `high` -> real edges break into dashes
        one threshold at `low`  -> isolated noise specks get in
        both                    -> the whole edge, and only the edge

    Implemented as a breadth-first flood from the strong pixels through
    8-connected weak ones.  A deque and an explicit queue rather than recursion
    because a long contour in a 4K frame is tens of thousands of pixels deep
    and Python's recursion limit is 1000.
    """
    if low > high:
        low, high = high, low   # OpenCV sorts them too; argument order is not intent
    strong = thin >= high
    weak = (thin >= low) & (thin < high)

    h, w = thin.shape
    out = np.zeros(thin.shape, np.uint8)
    out[strong] = 255

    q = deque(zip(*np.nonzero(strong)))
    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                y, x = i + di, j + dj
                if 0 <= y < h and 0 <= x < w and weak[y, x] and out[y, x] == 0:
                    out[y, x] = 255
                    q.append((y, x))
    return out


def gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur sized from sigma, returned as float64.

    Kernel size 2*ceil(3*sigma)+1 keeps 99.7% of the Gaussian's mass, which is
    the same rule `cv2.GaussianBlur(img, (0, 0), sigma)` applies when you let
    it pick.  Picking the size yourself and getting it too small for the sigma
    silently truncates the tails, and the filter you ran is not the filter you
    asked for.
    """
    k = int(2 * np.ceil(3.0 * sigma) + 1)
    g = gaussian_kernel_1d(k, sigma)
    return correlate_shift_separable(gray.astype(np.float64), g, g)


def canny(gray: np.ndarray, low: float, high: float,
          sigma: float | None = None, norm: str = "L1") -> np.ndarray:
    """The five stages, in order, returning a uint8 mask of 0 and 255.

        1. Gaussian blur      (only if `sigma` is given -- see below)
        2. Sobel gradients
        3. non-maximum suppression
        4. double threshold
        5. hysteresis

    `sigma=None` by default, which means **no blur**, because that is what
    `cv2.Canny` does and this function is written to be compared against it.
    OpenCV gives you stages 2 to 5 and leaves stage 1 to you.  That is the most
    expensive default in the library: on a clean synthetic image it costs you
    nothing, and at a realistic sensor noise of sigma 20 it is the difference
    between about 800 edge pixels and about 27,000.
    """
    img = gray.astype(np.float64)
    if sigma is not None:
        img = gaussian_blur(img, sigma)
    gx, gy = sobel(img)
    mag = magnitude(gx, gy, norm=norm)
    thin = non_maximum_suppression(mag, direction(gx, gy))
    return hysteresis(thin, low, high)


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of pixels on which two binary edge maps agree.

    Deliberately counts agreement on *both* labels, not IoU of the edge
    pixels, because that is the number the comparison against OpenCV is
    quoted in and it should be read with its own weakness in mind: edges are a
    small fraction of the image, so 98% agreement is a weaker statement than
    it sounds.  `examples/07_edges.py` prints the edge-pixel counts beside it
    so the reader can see both.
    """
    return float(100.0 * ((a > 0) == (b > 0)).mean())
