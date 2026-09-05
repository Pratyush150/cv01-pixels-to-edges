"""Separability: an exact factorisation, not an approximation."""

import cv2
import numpy as np
import pytest

from pixels import convolve as K, images, separable as S


def test_rank_one_means_separable_and_rank_two_does_not():
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    assert S.is_separable(K.BOX3)
    assert S.is_separable(sobel_x)
    assert S.is_separable(S.gaussian_kernel_2d(9, 1.5))
    assert not S.is_separable(K.LAPLACIAN4)
    assert not S.is_separable(K.EMBOSS)


def test_factorisation_reproduces_the_kernel_exactly():
    for kernel in (K.BOX3, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64),
                   S.gaussian_kernel_2d(7, 1.2)):
        col, row = S.separate(kernel)
        assert np.allclose(np.outer(col, row), kernel, atol=1e-12)


def test_separating_a_non_separable_kernel_raises_instead_of_approximating():
    """The failure mode this guard exists for is silence: taking the largest
    singular vector of a rank-2 kernel gives two passes that compute a
    DIFFERENT filter, correctly, with no warning."""
    with pytest.raises(ValueError, match="rank"):
        S.separate(K.LAPLACIAN4)


def test_our_gaussian_matches_opencv():
    for k in (3, 5, 9, 15, 31):
        sigma = k / 6.0
        ours = S.gaussian_kernel_1d(k, sigma)
        theirs = cv2.getGaussianKernel(k, sigma).ravel()
        assert np.allclose(ours, theirs, atol=1e-12)
        assert ours.sum() == pytest.approx(1.0)


def test_even_sized_gaussian_is_refused():
    with pytest.raises(ValueError):
        S.gaussian_kernel_1d(4, 1.0)


@pytest.mark.parametrize("k", [3, 5, 9, 15])
def test_two_one_dimensional_passes_equal_one_two_dimensional_pass(k):
    """The claim that makes the speed-up free rather than a trade-off."""
    img = images.shapes_gray().astype(np.float64)
    sigma = k / 6.0
    g1 = S.gaussian_kernel_1d(k, sigma)
    two_d = S.correlate_shift_2d(img, S.gaussian_kernel_2d(k, sigma))
    sep = S.correlate_shift_separable(img, g1, g1)
    assert np.abs(two_d - sep).max() < 1e-9


def test_the_shift_based_two_d_pass_agrees_with_the_windowed_one():
    """Two independent implementations of the same convolution, so the timing
    comparison is not measuring a difference in correctness."""
    img = images.shapes_gray()
    kernel = S.gaussian_kernel_2d(7, 1.2)
    assert np.allclose(S.correlate_shift_2d(img, kernel),
                       K.correlate2d(img, kernel), atol=1e-9)


def test_separable_gaussian_agrees_with_cv2_gaussianblur():
    img = images.shapes_gray().astype(np.float64)
    k, sigma = 15, 2.5
    g1 = S.gaussian_kernel_1d(k, sigma)
    ours = S.correlate_shift_separable(img, g1, g1)
    theirs = cv2.GaussianBlur(img, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - theirs).max() < 1e-9


def test_the_benchmark_reports_an_exact_match_at_every_size():
    """The timing numbers are allowed to wobble; this column is not."""
    rng = np.random.default_rng(0)
    img = rng.random((128, 128)) * 255
    for row in S.benchmark(img, sizes=(3, 9), repeats=1):
        assert row["max_abs_diff"] < 1e-9
        assert row["mults_2d"] == row["k"] ** 2
        assert row["mults_sep"] == 2 * row["k"]
