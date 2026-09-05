"""Separable kernels: when a K x K pass is secretly two 1-D passes.

Some 2-D kernels factor into a column times a row.  When they do, you can run a
1 x K pass and then a K x 1 pass and get the *identical* answer -- not an
approximation -- for 2K multiplies per pixel instead of K squared.  At K = 31
that is 62 instead of 961.

A kernel factors exactly when its matrix rank is 1, which is a testable
property, not a judgement call.  Box blurs and Gaussians are rank 1.  Sobel is
rank 1 (it is a blur times a difference, which is what the 2 in its middle row
is doing).  The Laplacian is rank 2 and does not factor at all; forcing it
through the largest singular vector gives you a different kernel and a quietly
wrong answer.

The benchmark in this module is written so the two sides differ *only* in how
many multiply-adds they do.  Timing a Python loop against a library call would
measure interpreter overhead and report it as an algorithmic win.
"""

from __future__ import annotations

import time

import numpy as np

from .convolve import pad

__all__ = [
    "is_separable",
    "separate",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
    "correlate_shift_2d",
    "correlate_shift_separable",
    "benchmark",
]


def is_separable(kernel: np.ndarray, tol: float | None = None) -> bool:
    """True when the kernel is an outer product of two vectors.

    `np.linalg.matrix_rank` compares singular values against a tolerance
    derived from the matrix size and machine epsilon, which is the right test:
    a kernel built as an outer product and then rounded to float32 has a second
    singular value of about 1e-8 rather than exactly zero.
    """
    k = np.asarray(kernel, np.float64)
    rank = np.linalg.matrix_rank(k) if tol is None else np.linalg.matrix_rank(k, tol=tol)
    return int(rank) == 1


def separate(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor a rank-1 kernel into (column, row) with `col[:, None] * row == kernel`.

    The SVD gives it directly: for a rank-1 matrix the first singular triple
    reconstructs the whole thing, so `col = U[:,0] * S[0]` and `row = Vt[0]`.

    SVD is free to negate both vectors at once -- the outer product is
    unchanged -- so do not be alarmed when Sobel-x factors as `[-1.41, -2.83,
    -1.41]` and `[0.71, 0, -0.71]` instead of the textbook `[1,2,1]` and
    `[-1,0,1]`.  Same product, opposite signs, and a shared scale absorbed into
    the two vectors.  Raises on a kernel that does not factor, because
    returning a silent approximation is exactly the failure this guards.
    """
    k = np.asarray(kernel, np.float64)
    if not is_separable(k):
        raise ValueError(
            f"kernel has rank {int(np.linalg.matrix_rank(k))}, so no exact factorisation "
            "exists; two 1-D passes would compute a different filter"
        )
    u, s, vt = np.linalg.svd(k)
    return u[:, 0] * s[0], vt[0, :]


def gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """A normalised 1-D Gaussian, matching `cv2.getGaussianKernel`.

    Normalising to sum 1 is what makes it a *blur* rather than a brightness
    change: a kernel whose weights sum to 1 rearranges brightness, one that
    sums to 1.2 adds 20% to every pixel.
    """
    if ksize % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """The 2-D Gaussian as the outer product of two 1-D ones.

    Building it this way is the proof of separability rather than an appeal to
    it: exp(-(x^2+y^2)/2s^2) factors as exp(-x^2/2s^2) * exp(-y^2/2s^2) because
    the exponent is a sum.  That is the *only* reason large Gaussian blurs are
    affordable, and it is why `cv2.GaussianBlur` never runs a 2-D pass.
    """
    g = gaussian_kernel_1d(ksize, sigma)
    return np.outer(g, g)


def correlate_shift_2d(img: np.ndarray, kernel: np.ndarray,
                       border: str = "reflect101") -> np.ndarray:
    """K x K correlation by accumulating K*K shifted copies of the image.

    This is a strange way to write a convolution and a deliberate one.  Every
    multiply-add is a whole-array NumPy operation, so the Python-level loop
    runs K*K times rather than once per pixel.  Its partner below runs 2*K
    times over the same machinery.  The ratio between their run times is
    therefore the ratio of their *arithmetic*, which is the thing separability
    actually changes -- and not a measurement of interpreter overhead wearing
    an algorithmic disguise.
    """
    k = np.asarray(kernel, np.float64)
    ks = k.shape[0]
    p = ks // 2
    im = pad(img.astype(np.float64), p, border)
    h, w = img.shape
    out = np.zeros((h, w), np.float64)
    for di in range(ks):
        for dj in range(ks):
            out += k[di, dj] * im[di:di + h, dj:dj + w]
    return out


def correlate_shift_separable(img: np.ndarray, col: np.ndarray, row: np.ndarray,
                              border: str = "reflect101") -> np.ndarray:
    """The same filter as two 1-D passes: horizontal first, then vertical.

    The border is applied on each pass in the direction that pass moves, which
    is what OpenCV does.  Padding both axes up front and then running two
    passes over the doubly-padded array gives a different -- and wrong -- answer
    in the corners, and that is a real bug people ship: it disagrees with the
    2-D version by a few units in a 1-pixel frame, which reads as noise.
    """
    col = np.asarray(col, np.float64)
    row = np.asarray(row, np.float64)
    h, w = img.shape
    ph, pw = len(row) // 2, len(col) // 2

    im = np.pad(img.astype(np.float64), ((0, 0), (ph, ph)),
                mode="reflect" if border == "reflect101" else "edge")
    tmp = np.zeros((h, w), np.float64)
    for dj in range(len(row)):
        tmp += row[dj] * im[:, dj:dj + w]

    tm = np.pad(tmp, ((pw, pw), (0, 0)),
                mode="reflect" if border == "reflect101" else "edge")
    out = np.zeros((h, w), np.float64)
    for di in range(len(col)):
        out += col[di] * tm[di:di + h, :]
    return out


def benchmark(img: np.ndarray, sizes=(3, 7, 15, 31), repeats: int = 5, sigma_ratio: float = 6.0):
    """Time both routes on real Gaussians and return one row per kernel size.

    Rows are dicts with `k`, `mults_2d` (k*k), `mults_sep` (2k), `predicted`
    (their ratio), `t_2d`, `t_sep`, `measured` and `max_abs_diff`.

    Best-of-N, not mean-of-N: the fastest run is the one least polluted by the
    scheduler putting something else on the core.  A mean over five runs on a
    shared machine measures the noise as much as the code.

    Expect `measured` to sit near `predicted` but not on it.  The 2-D version
    also touches K^2 times as much memory, so at large K it tends to lose by
    *more* than the multiply count predicts.  The defensible statement is
    "same order of magnitude, rising with K".  `max_abs_diff` is the column
    that has to hold exactly, because separability is a factorisation and not
    an approximation.
    """
    rows = []
    for k in sizes:
        sigma = k / sigma_ratio
        k2d = gaussian_kernel_2d(k, sigma)
        g1d = gaussian_kernel_1d(k, sigma)

        # One untimed call each, to fault in the output buffers and warm the
        # cache.  Without it the first timed iteration of the first kernel size
        # carries the cost of everything NumPy allocates on the way, and
        # best-of-N cannot help because it is the *minimum* that is wanted and
        # every later run is already warm.
        correlate_shift_2d(img, k2d)
        correlate_shift_separable(img, g1d, g1d)

        t2d = tsep = float("inf")
        a = b = None
        for _ in range(repeats):
            t = time.perf_counter()
            a = correlate_shift_2d(img, k2d)
            t2d = min(t2d, time.perf_counter() - t)

            t = time.perf_counter()
            b = correlate_shift_separable(img, g1d, g1d)
            tsep = min(tsep, time.perf_counter() - t)

        rows.append({
            "k": k,
            "mults_2d": k * k,
            "mults_sep": 2 * k,
            "predicted": (k * k) / (2.0 * k),
            "t_2d": t2d,
            "t_sep": tsep,
            "measured": t2d / tsep,
            "max_abs_diff": float(np.abs(a - b).max()),
        })
    return rows
