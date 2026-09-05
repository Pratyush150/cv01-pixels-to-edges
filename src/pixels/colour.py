"""Colour spaces: BGR, grayscale, HSV -- and one threshold that only HSV wins.

The chapter this follows makes two claims that are easy to state and easy to
disbelieve:

    1. Grayscale is a weighted sum, not an average, and the weights are a fact
       about human eyes rather than a convention.
    2. Under changing light, the *hue* of a surface barely moves while all
       three of its B, G, R values move together.  So a colour threshold that
       has to survive a shadow belongs in HSV, not in BGR.

Claim 2 gets measured in `examples/04_colour_spaces.py`, with the ground truth
known because `pixels.images` drew the object.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "BT601_WEIGHTS",
    "to_gray",
    "bgr_to_hsv",
    "hue_distance",
    "mask_by_bgr_box",
    "mask_by_hue",
    "iou",
    "best_bgr_rule",
    "best_hue_rule",
]

# OpenCV's COLOR_BGR2GRAY uses the ITU-R BT.601 luma weights, in B, G, R order.
# Summing to 1 is what keeps the output inside 0..255 without a rescale. Being
# unequal is a fact about eyes: the green-sensitive cones carry most of our
# sense of brightness, so green weighs about five times as heavily as blue.
# Average the three channels instead and pure red, pure green and pure blue all
# land on 85 -- the one thing that told them apart is gone.
BT601_WEIGHTS = (0.114, 0.587, 0.299)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Luma by hand, rounded the way OpenCV rounds.

    The `+ 0.5` then floor is round-half-up, which is what OpenCV's fixed-point
    integer path does.  np.round is round-half-to-EVEN, and using it here makes
    this disagree with cv2.cvtColor on roughly one pixel in two hundred -- a
    difference small enough to look like "floating point" and large enough to
    fail an exact test.  The tests assert we match cv2 to within 1 level.
    """
    b, g, r = (bgr[:, :, i].astype(np.float64) for i in range(3))
    wb, wg, wr = BT601_WEIGHTS
    return np.floor(wb * b + wg * g + wr * r + 0.5).astype(np.uint8)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> HSV uint8 in OpenCV's conventions, written out longhand.

    The conventions are the whole reason this is worth writing:

    * **H is 0..179, not 0..359.**  Hue is an angle on a colour wheel, so it
      needs 360 values; one byte holds 256.  OpenCV halves it.  Every hue
      number you copy off a web colour picker is therefore twice too big.
    * **H wraps around.**  The wheel closes, so 179 sits immediately before 0,
      and both of those are red.  Red is the single hue that straddles the join,
      which is why the usual recipe for detecting it needs two ranges -- or one
      circular distance, as in `hue_distance` below.
    * **S and V are 0..255**, not percentages.

    V is simply the largest channel, S is how far the smallest falls below it
    as a fraction of V, and H says *which* channel is largest and by how much.
    Note what that means: scaling all three channels by the same factor -- which
    is exactly what dimming the light does -- divides V, leaves S unchanged
    because it is a ratio, and leaves H unchanged for the same reason.
    """
    f = bgr.astype(np.float64)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    v = f.max(axis=2)
    mn = f.min(axis=2)
    delta = v - mn

    # Guard the divisions: a grey pixel has delta == 0 and no defined hue at
    # all.  OpenCV reports H = 0 for those, and so must we, but dividing first
    # and fixing afterwards would emit a RuntimeWarning on every grey image.
    safe = np.where(delta == 0, 1.0, delta)

    h = np.zeros_like(v)
    h = np.where(v == r, 60.0 * (g - b) / safe, h)
    h = np.where((v == g) & (v != r), 120.0 + 60.0 * (b - r) / safe, h)
    h = np.where((v == b) & (v != r) & (v != g), 240.0 + 60.0 * (r - g) / safe, h)
    h = np.where(delta == 0, 0.0, h % 360.0)

    s = np.where(v == 0, 0.0, 255.0 * delta / np.where(v == 0, 1.0, v))

    out = np.empty(bgr.shape, np.uint8)
    out[:, :, 0] = np.floor(h * 0.5 + 0.5).astype(np.uint8) % 180  # the /2 that costs a byte
    out[:, :, 1] = np.floor(s + 0.5).astype(np.uint8)
    out[:, :, 2] = v.astype(np.uint8)
    return out


def hue_distance(hue: np.ndarray, centre: int) -> np.ndarray:
    """Shortest distance round the 180-step hue wheel.

    Straight subtraction says hue 179 is 178 away from hue 1.  They are two
    apart, and both are red.  Getting this wrong is what forces the usual
    two-range `inRange` dance for red; getting it right makes red no harder
    than green.
    """
    d = np.abs(hue.astype(np.int16) - int(centre))
    return np.minimum(d, 180 - d)


def mask_by_bgr_box(bgr: np.ndarray, r_lo: int, g_hi: int, b_hi: int) -> np.ndarray:
    """"Reddish" as an axis-aligned box in BGR: lots of red, little else."""
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return ((r >= r_lo) & (g <= g_hi) & (b <= b_hi)).astype(np.uint8) * 255


def mask_by_hue(hsv: np.ndarray, centre: int, half_width: int,
                s_min: int = 90, v_min: int = 25) -> np.ndarray:
    """"This colour" as a wedge of the hue wheel, with floors on S and V.

    The two floors are not decoration.  Hue is computed from the *differences*
    between three channels, so on a nearly-grey pixel (small S) or a nearly
    black one (small V) it is the arctangent of noise -- it takes a definite,
    meaningless value.  Without the floors, every shadow in the frame joins
    your mask, which is exactly what "my threshold works indoors and lights up
    the whole car park at night" looks like.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    ok = (hue_distance(h, centre) <= half_width) & (s >= s_min) & (v >= v_min)
    return ok.astype(np.uint8) * 255


def iou(mask: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union: overlap divided by combined area.

    Accuracy is useless here.  The red discs are 11% of the frame, so a mask
    that says "nothing is red" scores 89% accurate and 0.0 IoU.  IoU is the
    number that cannot be gamed by a mostly-empty answer.
    """
    a, b = mask > 0, truth > 0
    union = int((a | b).sum())
    return 0.0 if union == 0 else float((a & b).sum()) / union


def _iou_from_counts(inter, sel, truth_count):
    """IoU for every candidate at once, given intersection and selected counts."""
    union = sel + truth_count - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def best_bgr_rule(bgr: np.ndarray, truth: np.ndarray, step: int = 4):
    """Exhaustively search the family of BGR boxes above; return (iou, params).

    Why search at all instead of tuning one rule by eye: a single badly chosen
    threshold proves nothing about the colour space.  Quoting "BGR scored 0.57"
    should mean *no rule of this shape does better*, which is a claim only a
    sweep can support.

    Why it is not three nested loops.  There are (256/step)^3 candidates -- 32768
    at step 8, 262,144 at step 4 -- and testing each against 172,800 pixels is
    tens of billions of comparisons, well over a minute.  Instead: quantise into
    256/step bins per channel, histogram the image once into that cube, and take
    a running sum along each axis in the direction its bound moves.  After
    that the pixel count for *every* candidate box is a single array lookup, and
    the whole search is two bincounts and three cumsums.

    The quantisation is exact, not approximate, and that is the part worth
    checking: with `q = v // step`, the test `v >= i*step` is exactly `q >= i`,
    and `v <= j*step + step - 1` is exactly `q <= j`.  So this covers a coarse
    grid of thresholds with no error, rather than a fine grid with rounding.
    """
    n = 256 // step
    q = (bgr // step).astype(np.int64)
    idx = (q[:, :, 2] * n + q[:, :, 1]) * n + q[:, :, 0]     # (R, G, B) order
    t = truth > 0
    truth_count = int(t.sum())

    hist_all = np.bincount(idx.ravel(), minlength=n ** 3).reshape(n, n, n)
    hist_hit = np.bincount(idx[t].ravel(), minlength=n ** 3).reshape(n, n, n)

    def cumulate(h):
        c = np.cumsum(h[::-1], axis=0)[::-1]   # r >= r_lo -> suffix sum
        c = np.cumsum(c, axis=1)               # g <= g_hi -> prefix sum
        return np.cumsum(c, axis=2)            # b <= b_hi -> prefix sum

    scores = _iou_from_counts(cumulate(hist_hit), cumulate(hist_all), truth_count)
    i, j, k = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return float(scores[i, j, k]), (int(i * step), int(j * step + step - 1),
                                    int(k * step + step - 1))


def best_hue_rule(hsv: np.ndarray, truth: np.ndarray, v_min: int = 25,
                  s_step: int = 8, centre_step: int = 2):
    """The same exhaustive search over hue wedges, so the comparison is fair.

    Three free parameters on each side -- (r_lo, g_hi, b_hi) against
    (centre, half_width, s_min) -- and the same objective, so a difference in
    the two scores is a difference between the colour spaces rather than
    between how hard somebody tried.

    Same cumulative-histogram trick, one hue centre at a time: for a fixed
    centre the distance array is fixed, so a 2-D histogram over (distance,
    saturation bin) answers every (half_width, s_min) pair at once.

    Pixels below `v_min` are dropped before the histogram rather than filtered
    afterwards.  Same result, clearer intent: on a near-black pixel the hue is
    computed from three tiny numbers and is noise, so it is not a candidate at
    any setting.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    t = truth > 0
    truth_count = int(t.sum())
    ns = 256 // s_step
    qs = (s // s_step).astype(np.int64)
    bright = v >= v_min

    def cumulate(hh):
        c = np.cumsum(hh, axis=0)                          # d <= half_width
        return np.cumsum(c[:, ::-1], axis=1)[:, ::-1]      # s >= s_min

    best = (0.0, (0, 0, 0))
    for centre in range(0, 180, centre_step):
        idx = hue_distance(h, centre).astype(np.int64) * ns + qs   # d is 0..90
        hist_all = np.bincount(idx[bright].ravel(), minlength=91 * ns).reshape(91, ns)
        hist_hit = np.bincount(idx[bright & t].ravel(), minlength=91 * ns).reshape(91, ns)
        scores = _iou_from_counts(cumulate(hist_hit), cumulate(hist_all), truth_count)
        half, m = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if scores[half, m] > best[0]:
            best = (float(scores[half, m]), (centre, int(half), int(m * s_step)))
    return best
