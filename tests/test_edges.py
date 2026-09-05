"""Sobel, non-maximum suppression, hysteresis, and our Canny against OpenCV's."""

import cv2
import numpy as np
import pytest

from pixels import edges as E, images


# The floor our Canny must clear against cv2.Canny on a clean image. Set from
# measured values (99.98% on shapes, 99.99% on a single diagonal) with room for
# the tie-breaking differences described in `non_maximum_suppression`. If this
# ever fails it is a real regression, not noise: the inputs are deterministic.
CLEAN_AGREEMENT_FLOOR = 99.5
NOISE_AGREEMENT_FLOOR = 97.5


def test_sobel_is_bit_identical_to_opencv():
    """Because our correlate2d defaults to the same BORDER_REFLECT_101."""
    img = images.shapes_gray()
    gx, gy = E.sobel(img)
    assert np.array_equal(gx, cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    assert np.array_equal(gy, cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))


def test_the_hand_worked_patch():
    patch = np.array([[10, 10, 10], [10, 10, 80], [10, 10, 80]], np.float64)
    gx = float((patch * E.SOBEL_X).sum())
    gy = float((patch * E.SOBEL_Y).sum())
    assert (gx, gy) == (210.0, 70.0)
    assert np.hypot(gx, gy) == pytest.approx(221.36, abs=0.01)
    assert np.degrees(np.arctan2(gy, gx)) == pytest.approx(18.43, abs=0.01)


def test_sobel_x_finds_vertical_edges_and_sobel_y_finds_horizontal_ones():
    """The naming is the reverse of what the kernel's shape suggests."""
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    vertical = cv2.GaussianBlur(np.where(xx >= n // 2, 200, 0).astype(np.uint8),
                                (0, 0), 2.0)
    horizontal = cv2.GaussianBlur(np.where(yy >= n // 2, 200, 0).astype(np.uint8),
                                  (0, 0), 2.0)
    gx, gy = E.sobel(vertical)
    assert np.abs(gx).max() > 100 * np.abs(gy).max() + 1
    gx, gy = E.sobel(horizontal)
    assert np.abs(gy).max() > 100 * np.abs(gx).max() + 1


def test_edge_orientation_is_the_gradient_plus_ninety():
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    for image, want_gradient in [
        (np.where(xx >= n // 2, 200, 0), 0.0),      # vertical edge
        (np.where(yy >= n // 2, 200, 0), 90.0),     # horizontal edge
        (np.where(xx + yy >= n, 200, 0), 45.0),     # diagonal
    ]:
        blurred = cv2.GaussianBlur(image.astype(np.uint8), (0, 0), 2.0)
        gx, gy = E.sobel(blurred)
        mag = E.magnitude(gx, gy, norm="L2")
        strong = mag > 0.6 * mag.max()
        grad = float(np.median(E.direction(gx, gy)[strong]))
        assert grad == pytest.approx(want_gradient, abs=1.0)


def test_direction_is_folded_to_a_half_turn():
    """A gradient of 20 degrees and one of 200 describe the same edge line."""
    gx = np.array([[1.0, -1.0]])
    gy = np.array([[0.0, 0.0]])
    folded = E.direction(gx, gy)
    assert folded[0, 0] == pytest.approx(0.0)
    assert folded[0, 1] == pytest.approx(0.0)     # 180 folds onto 0
    assert np.all((folded >= 0) & (folded < 180))


def test_nms_thins_a_five_pixel_ridge_to_one():
    row = np.array([10, 10, 40, 90, 140, 150, 150], np.uint8)
    img = np.tile(row, (5, 1))
    gx, gy = E.sobel(img)
    mag = E.magnitude(gx, gy, norm="L2")
    assert np.array_equal(mag[2].astype(int), [0, 120, 320, 400, 240, 40, 0])

    thin = E.non_maximum_suppression(mag, E.direction(gx, gy))
    assert np.array_equal(thin[2].astype(int), [0, 0, 0, 400, 0, 0, 0])
    assert int((thin > 0).sum()) == 3   # rows 1..3; row 0 and 4 are border


def test_comparing_along_the_edge_deletes_the_edge():
    """Not 'thins it badly' -- deletes it, silently, to nothing."""
    row = np.array([10, 10, 40, 90, 140, 150, 150], np.uint8)
    img = np.tile(row, (5, 1))
    gx, gy = E.sobel(img)
    mag = E.magnitude(gx, gy, norm="L2")

    # The gradient here points at 0 degrees, so bin 0 is correct and bin 2 --
    # up and down, ALONG the edge -- is the classic mistake.
    wrong_bins = np.full(mag.shape, 90.0)          # forces bin 2 everywhere
    assert int((E.non_maximum_suppression(mag, wrong_bins) > 0).sum()) == 0


def test_hysteresis_keeps_the_chain_and_drops_the_orphan():
    strip = np.array([[0, 0, 160, 90, 80, 70, 0, 60, 0]], np.float64)
    strip = np.repeat(strip, 3, axis=0)
    out = E.hysteresis(strip, 50, 150)
    assert np.array_equal((out[1] > 0).astype(int), [0, 0, 1, 1, 1, 1, 0, 0, 0])
    # index 7 is above `low` but touches nothing strong, so it is noise
    assert out[1, 7] == 0
    # and each single threshold gets a different, worse answer
    assert int((strip[1] >= 150).sum()) == 1
    assert int((strip[1] >= 50).sum()) == 5


def test_hysteresis_sorts_its_thresholds_like_opencv_does():
    strip = np.repeat(np.array([[0, 160, 90, 0]], np.float64), 3, axis=0)
    assert np.array_equal(E.hysteresis(strip, 50, 150), E.hysteresis(strip, 150, 50))


@pytest.mark.parametrize("name", ["shapes", "blurred", "checkerboard", "diagonal"])
def test_our_canny_agrees_with_opencv_on_clean_images(name):
    sources = {
        "shapes": images.shapes_gray(),
        "blurred": cv2.GaussianBlur(images.shapes_gray(), (0, 0), 1.4),
        "checkerboard": images.checkerboard(),
        "diagonal": cv2.GaussianBlur(
            np.where(np.add(*np.mgrid[0:160, 0:160]) >= 150, 200, 40).astype(np.uint8),
            (0, 0), 1.4),
    }
    img = sources[name]
    agree = E.agreement(E.canny(img, 50, 150), cv2.Canny(img, 50, 150))
    assert agree >= CLEAN_AGREEMENT_FLOOR, f"{name}: {agree:.3f}%"


def test_our_canny_still_agrees_on_pure_noise_where_almost_everything_ties():
    noise = (np.random.default_rng(7).random((160, 160)) * 255).astype(np.uint8)
    agree = E.agreement(E.canny(noise, 50, 150), cv2.Canny(noise, 50, 150))
    assert agree >= NOISE_AGREEMENT_FLOOR
    # Worse than the clean images, and that IS the lesson: on noise nearly every
    # pair of neighbouring magnitudes is a near-tie, so which of the two is the
    # crest is genuinely undetermined.
    assert agree < 99.5


def test_canny_output_is_a_binary_mask_and_the_border_is_never_kept():
    out = E.canny(images.shapes_gray(), 50, 150)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
    assert out[0].sum() == 0 and out[-1].sum() == 0
    assert out[:, 0].sum() == 0 and out[:, -1].sum() == 0


def test_the_pre_blur_is_not_optional_on_noisy_input():
    """cv2.Canny gives you stages 2..5; stage 1 is yours, and it is not cosmetic."""
    clean = images.shapes_gray()
    noisy = images.add_gaussian_noise(clean, 20, seed=1)
    raw = int((E.canny(noisy, 50, 150) > 0).sum())
    pre = int((E.canny(noisy, 50, 150, sigma=1.4) > 0).sum())
    truth = int((E.canny(clean, 50, 150, sigma=1.4) > 0).sum())
    assert raw > 8 * pre, "skipping the blur should flood the map with noise"
    assert 0.7 * truth < pre < 1.4 * truth, "with the blur, close to the real count"


def test_l1_and_l2_magnitudes_are_different_and_both_are_offered():
    gx = np.array([[3.0]])
    gy = np.array([[4.0]])
    assert E.magnitude(gx, gy, "L1")[0, 0] == 7.0
    assert E.magnitude(gx, gy, "L2")[0, 0] == 5.0
    with pytest.raises(ValueError):
        E.magnitude(gx, gy, "L3")
