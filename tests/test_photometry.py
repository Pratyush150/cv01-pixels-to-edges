"""Brightness and contrast: the pivot, and the absolute value that eats shadows."""

import cv2
import numpy as np
import pytest

from pixels import photometry as P


def test_pivoted_contrast_leaves_mid_grey_alone():
    """The signature of a correct contrast control, at every alpha."""
    grey = np.full((4, 4), 128, np.uint8)
    for alpha in (0.5, 0.8, 1.0, 1.6, 2.5):
        assert int(P.contrast_pivoted(grey, alpha)[0, 0]) == 128


def test_unpivoted_scaling_never_darkens_anything():
    """Which is why it is a brightness control wearing a contrast label."""
    values = np.arange(256, dtype=np.uint8).reshape(1, -1)
    out = P.contrast_naive(values, 1.6).astype(int)
    assert np.all(out >= values.astype(int))


def test_pivot_beta_matches_the_algebra():
    for alpha in (0.5, 1.6, 2.0):
        assert P.pivot_beta(alpha) == pytest.approx(128.0 * (1.0 - alpha))


def test_convert_scale_abs_reflects_shadows_upward():
    """The failure is a reflection off zero, and we can name where it starts."""
    alpha = 1.6
    beta = P.pivot_beta(alpha)
    cut = P.convert_scale_abs_reflects_below(alpha, beta)
    assert cut == pytest.approx(48.0)

    values = np.arange(256, dtype=np.uint8).reshape(1, -1)
    correct = P.contrast_pivoted(values, alpha).astype(int)
    csa = cv2.convertScaleAbs(values, alpha=alpha, beta=beta).astype(int)

    below = values[0] < cut
    # Above the cut the two agree exactly; below it, convertScaleAbs is BRIGHTER
    # than the correct answer wherever the correct answer clipped to 0.
    assert np.array_equal(correct[0][~below], csa[0][~below])
    assert np.all(csa[0][below] >= correct[0][below])
    assert csa[0][5] == 69 and correct[0][5] == 0


def test_convert_scale_abs_is_safe_when_beta_is_non_negative():
    values = np.arange(256, dtype=np.uint8).reshape(1, -1)
    for beta in (0.0, 10.0, 40.0):
        expected = np.clip(values.astype(np.float64) * 1.2 + beta, 0, 255)
        got = cv2.convertScaleAbs(values, alpha=1.2, beta=beta).astype(np.float64)
        assert np.abs(got - expected).max() <= 1.0   # rounding only
