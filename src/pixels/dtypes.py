"""The two bugs in image code that never raise an exception.

Both are here because they are silent.  A crash is a gift: it hands you a
traceback and a line number.  These two hand you a plausible-looking image and
let you ship it.

    1. `uint8` arithmetic wraps.  250 + 10 is 4, not 260 and not 255.
    2. A NumPy slice is a *view*.  Writing into it edits the array you sliced.

Everything in this module either reproduces one of those two bugs on purpose or
provides the one-line fix, and the tests assert the exact value at which each
one begins.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_threshold",
    "add_wrapping",
    "add_saturating",
    "wrapped_pixel_count",
    "is_view_of",
    "mutate_through_slice",
    "mutate_through_copy",
    "dtype_report",
]


def wrap_threshold(delta: int) -> int:
    """The smallest uint8 value for which `value + delta` wraps.

    `uint8` holds 0..255 and counts round like a clock, so `v + delta` wraps
    exactly when `v + delta > 255`, i.e. from `v = 256 - delta` upwards.  For
    delta = 10 that is 246: 245 + 10 = 255 is the last honest answer, and
    246 + 10 comes back as 0.

    This function exists so the claim can be *tested* rather than asserted in a
    comment.  Returns 256 when no value wraps (delta <= 0).
    """
    if delta <= 0:
        return 256
    return max(0, 256 - int(delta))


def add_wrapping(img: np.ndarray, delta: int) -> np.ndarray:
    """`img + delta` with NumPy's native uint8 semantics -- i.e. the bug.

    NumPy 2 refuses `np.uint8(250) + 10` as a *Python int* overflow, but it
    happily wraps when both operands are arrays of the same dtype, which is
    what every real pipeline does.  So we build the delta as a uint8 array to
    reproduce what actually happens to people.
    """
    d = np.full_like(img, np.uint8(delta % 256))
    return img + d  # no error, no warning: the bright pixels come out black


def add_saturating(img: np.ndarray, delta: int) -> np.ndarray:
    """The fix: promote out of uint8, clip, and come back.

    int16 is the smallest type that can hold 255 + 255 and also -255, so it is
    the cheapest correct choice for an 8-bit add.  `cv2.add` does exactly this
    internally and is faster; we spell it out once so the mechanism is visible,
    and `tests/test_dtypes.py` asserts the two agree on every possible value.
    """
    wide = img.astype(np.int16) + int(delta)
    return np.clip(wide, 0, 255).astype(np.uint8)


def wrapped_pixel_count(img: np.ndarray, delta: int) -> int:
    """How many pixels the naive add would corrupt.

    Useful as a guard in real code: if this is nonzero, `img + delta` is lying
    to you.  On a low-key photograph it is zero and the bug hides for months.
    """
    return int((img.astype(np.int16) + int(delta) > 255).sum())


def is_view_of(child: np.ndarray, parent: np.ndarray) -> bool:
    """True when `child` addresses `parent`'s memory rather than a duplicate.

    `np.shares_memory` is the honest test.  Comparing `.base` is the trick
    people reach for first and it is wrong in both directions: a chain of
    slices has a `.base` that is not the array you sliced, and `.copy()` of a
    view can still report a base under some NumPy versions.
    """
    return bool(np.shares_memory(child, parent))


def mutate_through_slice(img: np.ndarray, box: tuple[int, int, int, int], value: int):
    """Zero a rectangle *through a view*, and return (original, region).

    This is the bug.  `img[y0:y1, x0:x1]` is a window onto the same bytes, so
    the assignment lands in `img`.  The caller's "untouched original" now has a
    hole in it, and nothing anywhere said so.
    """
    y0, y1, x0, x1 = box
    region = img[y0:y1, x0:x1]      # a VIEW: no bytes were copied here
    region[:] = value               # ... so this writes into `img`
    return img, region


def mutate_through_copy(img: np.ndarray, box: tuple[int, int, int, int], value: int):
    """The same edit done safely, and the whole fix is one method call.

    Copying to *read* is the opposite mistake, and an expensive one -- see the
    measurement in example 02.  The rule turns on what you are about to do:
    duplicate before a write when the source array still matters, never before
    a look.
    """
    y0, y1, x0, x1 = box
    region = img[y0:y1, x0:x1].copy()   # .copy() is the entire difference
    region[:] = value
    return img, region


def dtype_report(arr: np.ndarray) -> dict:
    """The five numbers to print the moment an array surprises you.

    `size` counts numbers and `nbytes` counts bytes; they coincide only for
    uint8, which is why a memory budget that was right for 8-bit frames is
    silently 4x wrong the day someone converts to float32.
    """
    return {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "itemsize": int(arr.itemsize),
        "nbytes": int(arr.nbytes),
    }
