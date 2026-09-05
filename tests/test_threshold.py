"""Otsu from scratch, and the case where no global threshold can win."""

import cv2
import numpy as np
import pytest

from pixels import images, threshold as T


TINY = np.array([[40, 40, 40, 40, 50],
                 [50, 50, 50, 50, 50],
                 [190, 190, 190, 190, 190],
                 [190, 200, 200, 200, 200]], np.uint8)


def sample_images():
    scan, _ = images.lit_document()
    return {
        "tiny": TINY,
        "shapes": images.shapes_gray(),
        "ramp": images.gray_ramp(),
        "scan": scan,
        "checkerboard": images.checkerboard(),
        "green channel": images.tabletop_scene()[:, :, 1],
        "noisy shapes": images.add_gaussian_noise(images.shapes_gray(), 25, seed=4),
        "flat noise": np.clip(
            np.random.default_rng(3).normal(128, 10, (100, 100)), 0, 255).astype(np.uint8),
    }


@pytest.mark.parametrize("name", list(sample_images()))
def test_our_otsu_returns_exactly_what_opencv_returns(name):
    """Including the tie-break: argmax takes the first maximum, and so does OpenCV.

    That detail is not pedantry. Every histogram Otsu is good at has a wide
    empty valley, so a different tie-break disagrees on essentially every image
    the method is meant for.
    """
    img = sample_images()[name]
    theirs, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert T.otsu_threshold(img) == int(theirs)


def test_our_mask_matches_opencvs_mask_too():
    for img in sample_images().values():
        _, theirs = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ours = T.global_threshold(img, T.otsu_threshold(img))
        assert np.array_equal(ours, theirs)


def test_the_hand_worked_otsu_scores():
    assert T.within_class_variance(TINY, 40) == pytest.approx(3900.0)
    assert T.within_class_variance(TINY, 50) == pytest.approx(24.0)
    assert T.within_class_variance(TINY, 190) == pytest.approx(3900.0)
    assert T.otsu_threshold(TINY) == 50


def test_within_plus_between_equals_the_image_variance():
    """The identity that makes minimising one the same act as maximising the other."""
    for img in sample_images().values():
        total = float(img.astype(np.float64).var())
        curve = T.between_class_variance_curve(img)
        for t in range(0, 256, 7):
            within = T.within_class_variance(img, t)
            if within is None:
                assert curve[t] == -1.0, "an empty class must never win the argmax"
                continue
            assert within + curve[t] == pytest.approx(total, rel=1e-9, abs=1e-6)


def test_the_tie_range_is_real_and_we_pick_the_bottom_of_it():
    curve = T.between_class_variance_curve(TINY)
    winners = np.nonzero(np.abs(curve - curve.max()) < 1e-9)[0]
    assert winners.min() == 50 and winners.max() == 189
    assert T.otsu_threshold(TINY) == winners.min()


def test_histogram_refuses_a_float_image():
    with pytest.raises(TypeError):
        T.histogram256(np.zeros((4, 4), np.float32))


def test_our_adaptive_threshold_matches_opencv_to_a_stated_tolerance():
    """Not exactly: OpenCV computes the local mean in integers and rounds, we
    accumulated in float64. The pixels that differ are those within half a level
    of their own neighbourhood mean, where the answer was a coin toss."""
    scan, _ = images.lit_document()
    ours = T.adaptive_threshold_mean(scan, 31, 10, invert=True)
    theirs = cv2.adaptiveThreshold(scan, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    agree = 100.0 * ((ours > 0) == (theirs > 0)).mean()
    assert agree > 99.5, f"{agree:.4f}%"


def test_adaptive_rejects_an_even_block_size():
    with pytest.raises(ValueError):
        T.adaptive_threshold_mean(np.zeros((32, 32), np.uint8), 30, 5)


def test_on_a_lit_page_only_the_adaptive_threshold_is_usable():
    """The ranking is the claim; the exact digits are printed by example 08."""
    scan, truth = images.lit_document()
    t = T.otsu_threshold(scan)
    global_iou = T.iou(T.global_threshold(scan, 127, invert=True), truth)
    otsu_iou = T.iou(T.global_threshold(scan, t, invert=True), truth)
    adaptive_iou = T.iou(T.adaptive_threshold_mean(scan, 31, 10, invert=True), truth)

    assert global_iou < 0.4
    assert otsu_iou < 0.5
    assert adaptive_iou > 0.9
    assert global_iou < otsu_iou < adaptive_iou


def test_otsu_never_refuses_even_when_there_is_no_valley():
    """There is no 'this histogram is not bimodal' error, and there never will be."""
    flat = np.clip(np.random.default_rng(3).normal(128, 10, (200, 200)),
                   0, 255).astype(np.uint8)
    t = T.otsu_threshold(flat)
    mask = T.global_threshold(flat, t)
    assert 0 < t < 255
    # It splits the noise straight down the middle, which is the symptom to
    # guard for: a foreground fraction near 0.5 when you expected a small object.
    assert 0.4 < T.foreground_fraction(mask) < 0.6


def test_blur_before_otsu_not_after():
    clean = np.zeros((200, 200), np.uint8)
    cv2.rectangle(clean, (50, 50), (150, 150), 200, -1)
    truth = clean > 100
    noisy = images.add_gaussian_noise(clean, 45, seed=2)

    wrong_raw = int(((noisy > T.otsu_threshold(noisy)) != truth).sum())
    blurred = cv2.GaussianBlur(noisy, (0, 0), 2.0)
    wrong_blurred = int(((blurred > T.otsu_threshold(blurred)) != truth).sum())
    assert wrong_blurred < wrong_raw / 5


def test_invert_flips_the_mask_and_nothing_else():
    img = images.shapes_gray()
    t = T.otsu_threshold(img)
    a = T.global_threshold(img, t)
    b = T.global_threshold(img, t, invert=True)
    assert np.array_equal(a, 255 - b)


def test_iou_is_not_gameable_by_an_empty_answer():
    truth = np.zeros((100, 100), np.uint8)
    truth[10:20, 10:20] = 255
    empty = np.zeros((100, 100), np.uint8)
    assert T.iou(empty, truth) == 0.0
    # ...while plain accuracy would score this 99%
    assert 100.0 * (empty == truth).mean() == pytest.approx(99.0)
    assert T.iou(truth, truth) == 1.0
