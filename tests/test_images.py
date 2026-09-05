"""The image generators: deterministic, self-contained, and shaped as claimed."""

import numpy as np

from pixels import images


def test_every_generator_is_deterministic():
    """A number quoted in the README must be re-derivable on any machine."""
    assert np.array_equal(images.tabletop_scene(0), images.tabletop_scene(0))
    assert np.array_equal(images.shapes_gray(0), images.shapes_gray(0))
    scan_a, truth_a = images.lit_document(0)
    scan_b, truth_b = images.lit_document(0)
    assert np.array_equal(scan_a, scan_b) and np.array_equal(truth_a, truth_b)
    assert np.array_equal(images.add_gaussian_noise(images.shapes_gray(), 10, seed=3),
                          images.add_gaussian_noise(images.shapes_gray(), 10, seed=3))


def test_shapes_and_dtypes():
    assert images.gray_ramp().shape == (100, 256)
    assert images.tabletop_scene().shape == (360, 480, 3)
    assert images.shapes_gray().shape == (240, 320)
    scan, truth = images.lit_document()
    assert scan.shape == truth.shape == (240, 400)
    for arr in (images.gray_ramp(), images.tabletop_scene(), images.shapes_gray(),
                scan, truth, images.checkerboard()):
        assert arr.dtype == np.uint8


def test_the_ramp_holds_its_own_column_index():
    ramp = images.gray_ramp()
    for col in (0, 1, 128, 200, 255):
        assert np.all(ramp[:, col] == col)


def test_the_scene_reaches_the_top_of_the_uint8_range():
    """Without a specular highlight there is nothing for an overflow to overflow."""
    assert images.tabletop_scene().max() == 255


def test_the_scene_spans_the_illumination_range_the_colour_example_needs():
    """A BGR box can hold a lit and a shadowed sample of one colour only while
    the illumination ratio between them stays under about 3. Above that the
    lit background necessarily leaks in, which is what example 04 measures."""
    scene = images.tabletop_scene().astype(np.float64)
    lit = scene[80:130, 55:105, 2].mean()
    shadow = scene[245:290, 390:435, 2].mean()
    assert lit / shadow > 3.0


def test_the_truth_mask_covers_only_the_red_discs():
    scene = images.tabletop_scene()
    truth = images.tabletop_truth()
    assert set(np.unique(truth)).issubset({0, 255})
    inside = truth > 0
    b, g, r = (scene[:, :, i][inside].astype(int) for i in range(3))
    assert (r > g).mean() > 0.99 and (r > b).mean() > 0.99
    assert 0.05 < inside.mean() < 0.20     # two discs, about 11% of the frame


def test_the_document_really_is_lit_from_one_side():
    scan, truth = images.lit_document()
    assert scan[:, :50].mean() > 3 * scan[:, -50:].mean()
    # ...while the ink itself is evenly spread, so any left/right asymmetry in a
    # mask is the threshold's fault and not the image's.
    left = (truth[:, :200] > 0).mean()
    right = (truth[:, 200:] > 0).mean()
    assert abs(left - right) < 0.03


def test_noise_is_clipped_rather_than_wrapped():
    """Adding noise in uint8 without clipping would turn bright pixels black."""
    bright = np.full((64, 64), 250, np.uint8)
    out = images.add_gaussian_noise(bright, 30, seed=0)
    assert out.min() > 100, "no pixel wrapped round to near zero"
