"""2-D correlation and convolution, written three ways.

Three implementations of one operation, kept side by side on purpose:

    `correlate2d_loops`  -- explicit Python loops.  Slow, and the only version
                            in which the index bookkeeping is visible.
    `correlate2d`        -- the same arithmetic, vectorised.  This is the one
                            everything else in the package calls.
    `cv2.filter2D`       -- the library.  We assert agreement rather than
                            assume it.

`tests/test_convolve.py` asserts all three agree to floating-point tolerance on
random images and random kernels.  That assertion is the point of the module:
it is what makes "I understand convolution" a checkable statement instead of a
claim on a CV.

Naming: what everyone calls convolution -- OpenCV's `filter2D`, and every
convolutional layer in every deep-learning framework -- is *cross-correlation*.
True convolution flips the kernel 180 degrees first.  The flip makes the
operation commutative, which matters for the signal-processing theory and does
not matter here, because a symmetric kernel does not notice and an asymmetric
one just changes sign.  So correlation is the primitive and `convolve2d` is the
thin wrapper, not the other way round.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BORDER_MODES",
    "pad",
    "output_size",
    "correlate2d_loops",
    "correlate2d",
    "convolve2d",
    "BOX3",
    "SHARPEN",
    "LAPLACIAN4",
    "EMBOSS",
]

# np.pad's name for each OpenCV border mode.  The mapping matters more than it
# looks: OpenCV's *default* everywhere (filter2D, blur, GaussianBlur, Sobel) is
# BORDER_REFLECT_101, which np.pad calls "reflect".  np.pad's "symmetric" is
# OpenCV's BORDER_REFLECT, a different mode that repeats the edge pixel.  Pick
# the wrong one and your from-scratch filter disagrees with the library only in
# the outermost row and column -- a failure that looks like a rounding problem.
BORDER_MODES = {
    "constant": "constant",   # BORDER_CONSTANT   0 0 0 | a b c d | 0 0 0
    "reflect101": "reflect",  # BORDER_REFLECT_101 d c b | a b c d | c b a  <- OpenCV default
    "reflect": "symmetric",   # BORDER_REFLECT     c b a | a b c d | d c b
    "replicate": "edge",      # BORDER_REPLICATE   a a a | a b c d | d d d
    # BORDER_WRAP: b c d | a b c d | a b c.  Offered here because np.pad
    # supports it and periodic data (a panorama seam, a tiled texture) is a real
    # case -- but note that `cv2.filter2D` REFUSES it outright with
    # "columnBorderType != BORDER_WRAP", so this mode has no library counterpart
    # to check against.  On an ordinary photograph it is the worst choice of the
    # five: it drags the left edge of the scene onto the right.
    "wrap": "wrap",
}


def pad(image: np.ndarray, p: int, border: str = "reflect101") -> np.ndarray:
    """Add `p` invented rows and columns on every side.

    Zero padding is the obvious choice and it is wrong for photographs: zero
    means black, so every border pixel gets averaged against darkness that is
    not in the scene.  A 3x3 box blur on a uniform image of 100 returns 44.4 at
    the corners -- a 56% error on an image that has no variation to blur.
    `examples/05_convolution.py` prints that table.

    Zeros are still the right default for a CNN feature map, where "outside"
    genuinely has no activation.  The rule is about what the number *means*,
    not about which mode is better.
    """
    if p == 0:
        return image
    if border not in BORDER_MODES:
        raise ValueError(f"unknown border {border!r}; expected one of {sorted(BORDER_MODES)}")
    kwargs = {"constant_values": 0} if border == "constant" else {}
    return np.pad(image, p, mode=BORDER_MODES[border], **kwargs)


def output_size(in_size: int, k: int, p: int = 0, s: int = 1) -> int:
    """floor((in + 2p - k) / s) + 1.

    `in` is the size *before* padding, with the `2p` supplied inside.  Decide
    that once.  The classic failure is to measure the width after padding and
    keep the `+2p` anyway: the output array then comes out `2p` oversized on
    each axis, and the loop reads past the end of the input it was given.

    Term by term: `in + 2p` is the padded width; subtract `k` because the last
    legal starting column is `k-1` short of the right-hand end; divide by `s`
    because only every s-th start is visited; take the floor because half a
    stride is not a stride; add one because the first position counts.
    """
    if k % 2 == 0:
        raise ValueError("even kernels have no centre pixel; use an odd k")
    return (in_size + 2 * p - k) // s + 1


def correlate2d_loops(image: np.ndarray, kernel: np.ndarray,
                      padding: int | None = None, stride: int = 1,
                      border: str = "reflect101") -> np.ndarray:
    """Cross-correlation with the sliding window spelled out.

    This is the whiteboard version.  It is roughly 300x slower than the
    vectorised one on a 240x320 image and it exists because the vectorised one
    hides the two things worth learning: which patch each output cell reads,
    and where the padding enters.

    `padding=None` means "same" -- (k-1)//2, the amount that makes the output
    the same size as the input at stride 1.
    """
    img = image.astype(np.float64)          # FIRST line, before anything else:
    # an edge kernel produces negatives and values past 255, and in uint8 both
    # ends are destroyed silently -- clipped, not raised.  A Laplacian on an
    # ordinary photo runs from about -430 to +430; keep it in uint8 and you
    # lose every dark-to-bright transition and never see a warning.
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("kernel must be square and 2-D")

    kh = k.shape[0]
    p = (kh - 1) // 2 if padding is None else int(padding)

    h, w = img.shape                        # measured BEFORE padding -- see output_size
    out_h = output_size(h, kh, p, stride)
    out_w = output_size(w, kh, p, stride)
    padded = pad(img, p, border)

    out = np.zeros((out_h, out_w), np.float64)
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride   # output cell -> top-left of its patch
            # r:r+kh, never r:r+kh-1.  Python's stop is exclusive, so r:r+kh is
            # exactly kh rows; the -1 version broadcasts wrong and, with a 1xK
            # kernel, produces plausible-looking output that is simply not the
            # answer.
            patch = padded[r:r + kh, c:c + kh]
            out[i, j] = float((patch * k).sum())
    return out


def correlate2d(image: np.ndarray, kernel: np.ndarray,
                padding: int | None = None, stride: int = 1,
                border: str = "reflect101") -> np.ndarray:
    """The same operation with the loops pushed into NumPy.

    `sliding_window_view` builds a (out_h, out_w, kh, kw) *view* -- no data is
    copied, it is the same buffer addressed with clever strides -- and then one
    `tensordot` contracts the last two axes against the kernel.  That is the
    entire vectorisation: the arithmetic is identical, but the loop runs in
    compiled code instead of the interpreter.

    Why the naive version is slow is worth being precise about, because the
    usual answer is wrong.  It is not the multiplies: a 3x3 correlation on
    240x320 is 700k multiply-adds, microseconds of actual arithmetic.  It is
    that the interpreter takes 76,800 round trips to schedule them.
    """
    img = image.astype(np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    kh = k.shape[0]
    p = (kh - 1) // 2 if padding is None else int(padding)

    h, w = img.shape
    out_h = output_size(h, kh, p, stride)
    out_w = output_size(w, kh, p, stride)
    padded = pad(img, p, border)

    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kh))
    if stride != 1:
        windows = windows[::stride, ::stride]
    windows = windows[:out_h, :out_w]
    return np.tensordot(windows, k, axes=((2, 3), (0, 1)))


def convolve2d(image: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
    """True convolution: flip the kernel 180 degrees, then correlate.

    On a symmetric kernel (box, Gaussian, Laplacian) this is bit-identical to
    correlation and the distinction is invisible.  On an asymmetric one (Sobel,
    emboss) it negates the result.  That is exactly why the bug is annoying:
    it appears and disappears depending on which filter you happened to test
    with.
    """
    return correlate2d(image, np.flipud(np.fliplr(np.asarray(kernel))), **kwargs)


# ---------------------------------------------------------------------------
# A small kernel zoo.  The teaching point is that the sliding never changes --
# only these numbers do -- and that the SUM of the numbers predicts what
# happens to the image's average brightness:
#
#   sum == 1  ->  mean brightness untouched; the result still reads as a photo.
#   sum == 0  ->  mean pulled to zero; the result is a map of differences.
#   sum  > 1  ->  the whole frame lifts.  Usually a normalisation you forgot.
# ---------------------------------------------------------------------------

BOX3 = np.ones((3, 3), np.float64) / 9.0

# Centre 5 = the identity (a 1 in the middle) PLUS the centre-4 Laplacian.
# So this returns the original image plus its own edge map: a sharpen.
SHARPEN = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], np.float64)

# Centre 4 instead of 5, and the weights now sum to zero.  The copy of the
# original image is gone and only the differences survive.  This is the single
# most common "my convolution code is broken" report: it is not, the kernel
# just stopped being a photograph filter and became an edge detector.
LAPLACIAN4 = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], np.float64)

# Sum 1, but asymmetric -- the kernel where correlation and convolution differ
# visibly, which makes it the honest test case for `convolve2d`.
EMBOSS = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]], np.float64)
