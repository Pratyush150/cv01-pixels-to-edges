"""Grayscale, HSV, and the circular hue distance."""

import cv2
import numpy as np
import pytest

from pixels import colour, images


def test_grayscale_weights_are_not_an_average():
    """Pure red, green and blue must map to three different values."""
    blue = colour.to_gray(np.array([[(255, 0, 0)]], np.uint8))[0, 0]
    green = colour.to_gray(np.array([[(0, 255, 0)]], np.uint8))[0, 0]
    red = colour.to_gray(np.array([[(0, 0, 255)]], np.uint8))[0, 0]
    assert (int(blue), int(green), int(red)) == (29, 150, 76)
    assert sum(colour.BT601_WEIGHTS) == pytest.approx(1.0)
    # A plain mean would map all three to 85 and delete the only thing that
    # distinguished them, which is why "convert to grayscale" is a lossy,
    # opinionated step and not a neutral one.


def test_our_luma_matches_opencv_to_within_one_level():
    scene = images.tabletop_scene()
    ours = colour.to_gray(scene).astype(int)
    theirs = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(int)
    assert np.abs(ours - theirs).max() <= 1
    assert (ours == theirs).mean() > 0.99


def test_our_hsv_matches_opencv_to_within_one_level():
    scene = images.tabletop_scene()
    ours = colour.bgr_to_hsv(scene).astype(int)
    theirs = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV).astype(int)

    dh = np.abs(ours[:, :, 0] - theirs[:, :, 0])
    dh = np.minimum(dh, 180 - dh)            # hue is cyclic
    assert dh.max() <= 1
    assert np.abs(ours[:, :, 1] - theirs[:, :, 1]).max() <= 1
    assert np.array_equal(ours[:, :, 2], theirs[:, :, 2])   # V is just the max


def test_hsv_ranges_are_opencvs_not_the_internets():
    scene = images.tabletop_scene()
    hsv = colour.bgr_to_hsv(scene)
    assert hsv[:, :, 0].max() <= 179, "hue is halved to fit a byte"
    assert hsv[:, :, 1].max() <= 255 and hsv[:, :, 2].max() <= 255


def test_a_grey_pixel_gets_hue_zero_and_saturation_zero():
    grey = np.full((2, 2, 3), 100, np.uint8)
    hsv = colour.bgr_to_hsv(grey)
    assert np.all(hsv[:, :, 0] == 0) and np.all(hsv[:, :, 1] == 0)
    assert np.all(hsv[:, :, 2] == 100)


def test_hue_distance_wraps_round_the_wheel():
    """179 and 1 are two apart, not 178, and both are red."""
    assert int(colour.hue_distance(np.array([179]), 1)[0]) == 2
    assert int(colour.hue_distance(np.array([0]), 179)[0]) == 1
    assert int(colour.hue_distance(np.array([90]), 0)[0]) == 90
    assert np.all(colour.hue_distance(np.arange(180), 0) <= 90)


def test_illumination_moves_bgr_and_leaves_hue_alone():
    """The mechanism behind the whole HSV example, on synthetic pixels so the
    only thing changing is the illumination."""
    base = np.array([[(40, 60, 200)]], np.uint8)
    dim = (base.astype(np.float64) * 0.35).astype(np.uint8)

    h_base = colour.bgr_to_hsv(base)[0, 0]
    h_dim = colour.bgr_to_hsv(dim)[0, 0]
    assert int(colour.hue_distance(np.array([h_dim[0]]), int(h_base[0]))[0]) <= 1
    assert abs(int(h_dim[1]) - int(h_base[1])) <= 2         # saturation is a ratio
    assert int(h_dim[2]) < int(h_base[2]) / 2               # value collapsed
    assert np.all(dim[0, 0].astype(int) < base[0, 0].astype(int) / 2)


def test_hue_beats_the_best_possible_bgr_box_on_the_shadowed_scene():
    """The headline result of example 04, asserted rather than shown."""
    scene = images.tabletop_scene()
    truth = images.tabletop_truth()
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

    bgr_iou, _ = colour.best_bgr_rule(scene, truth, step=8)
    hue_iou, _ = colour.best_hue_rule(hsv, truth, centre_step=6)
    assert bgr_iou < 0.7, "no axis-aligned BGR box can hold both discs cleanly"
    assert hue_iou > 0.95
    assert hue_iou > bgr_iou + 0.25


def test_the_naive_non_circular_hue_window_misses_the_shadowed_disc():
    scene = images.tabletop_scene()
    truth = images.tabletop_truth()
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

    naive = ((np.abs(hsv[:, :, 0].astype(int) - 1) <= 7)
             & (hsv[:, :, 1] >= 88)).astype(np.uint8) * 255
    circular = colour.mask_by_hue(hsv, 0, 7, s_min=88)
    assert colour.iou(naive, truth) < 0.7
    assert colour.iou(circular, truth) > 0.95


def test_iou_of_disjoint_masks_is_zero():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[0:3, 0:3] = 255
    b[7:10, 7:10] = 255
    assert colour.iou(a, b) == 0.0
    assert colour.iou(a, a) == 1.0
