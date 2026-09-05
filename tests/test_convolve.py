"""Convolution: three implementations, one answer -- and the border modes."""

import cv2
import numpy as np
import pytest

from pixels import convolve as K, images


RNG = np.random.default_rng(11)


def random_kernels():
    """A mix: symmetric, asymmetric, zero-sum, and larger than 3x3.

    Asymmetric kernels matter here because a flip bug is INVISIBLE on symmetric
    ones -- testing only with a box blur would pass a broken implementation.
    """
    return [
        K.BOX3,
        K.SHARPEN,
        K.LAPLACIAN4,
        K.EMBOSS,
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64),
        RNG.normal(size=(3, 3)),
        RNG.normal(size=(5, 5)),
        RNG.normal(size=(7, 7)),
    ]


def test_the_hand_worked_two_by_two():
    """The drill everybody does on paper, checked against the machine."""
    img = np.array([[10, 10, 10, 10],
                    [10, 10, 80, 80],
                    [10, 10, 80, 80],
                    [10, 10, 80, 80]], np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    expected = np.array([[210.0, 210.0], [280.0, 280.0]])
    assert np.array_equal(K.correlate2d_loops(img, kx, padding=0), expected)
    assert np.array_equal(K.correlate2d(img, kx, padding=0), expected)


@pytest.mark.parametrize("kernel", random_kernels())
def test_loops_equal_vectorised_equal_opencv(kernel):
    """The claim the whole module exists to support."""
    img = images.shapes_gray()
    loops = K.correlate2d_loops(img, kernel)
    vector = K.correlate2d(img, kernel)
    library = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel,
                           borderType=cv2.BORDER_REFLECT_101)
    assert np.allclose(loops, vector, atol=1e-9)
    assert np.allclose(vector, library, atol=1e-9)


@pytest.mark.parametrize("mode,cv_border", [
    ("constant", cv2.BORDER_CONSTANT),
    ("reflect101", cv2.BORDER_REFLECT_101),
    ("reflect", cv2.BORDER_REFLECT),
    ("replicate", cv2.BORDER_REPLICATE),
])
def test_every_border_mode_matches_its_opencv_counterpart(mode, cv_border):
    """The name mapping is the bug-prone part: np.pad's 'reflect' is OpenCV's
    REFLECT_101, and np.pad's 'symmetric' is OpenCV's REFLECT. Swapping them
    fails only in the outermost row and column, which reads as rounding."""
    img = images.shapes_gray()
    kernel = RNG.normal(size=(5, 5))
    ours = K.correlate2d(img, kernel, border=mode)
    theirs = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel,
                          borderType=cv_border)
    assert np.allclose(ours, theirs, atol=1e-9)


def test_wrap_is_ours_alone():
    """`cv2.filter2D` refuses BORDER_WRAP outright:

        (-215:Assertion failed) columnBorderType != BORDER_WRAP in function 'init'

    np.pad has no such restriction, so `border="wrap"` works here and has no
    library counterpart to check against. It is the right mode for genuinely
    periodic data (a panorama seam, a tiled texture) and the wrong one for a
    photograph, where it drags the left edge of the scene onto the right.
    """
    img = images.shapes_gray()
    ours = K.correlate2d(img, K.BOX3, border="wrap")
    assert ours.shape == img.shape
    with pytest.raises(cv2.error):
        cv2.filter2D(img.astype(np.float64), cv2.CV_64F, K.BOX3,
                     borderType=cv2.BORDER_WRAP)


def test_zero_padding_darkens_a_uniform_image_by_the_amount_we_claim():
    flat = np.full((5, 5), 100.0)
    zeros = K.correlate2d(flat, K.BOX3, border="constant")
    mirror = K.correlate2d(flat, K.BOX3, border="reflect101")

    assert zeros[0, 0] == pytest.approx(400 / 9)     # 4 real cells of 9
    assert zeros[0, 2] == pytest.approx(600 / 9)     # 6 real cells of 9
    assert zeros[2, 2] == pytest.approx(100.0)       # all 9 real
    assert np.allclose(mirror, 100.0), "a blur cannot change an image with no variation"
    assert np.allclose(mirror, cv2.blur(flat, (3, 3)))


@pytest.mark.parametrize("n,k,p,s", [
    (224, 3, 1, 1), (224, 3, 1, 2), (4, 3, 0, 1), (32, 5, 2, 2),
    (17, 7, 3, 1), (100, 3, 0, 3),
])
def test_output_size_formula_matches_the_array_it_predicts(n, k, p, s):
    got = K.correlate2d(np.zeros((n, n)), np.ones((k, k)), padding=p, stride=s)
    assert got.shape == (K.output_size(n, k, p, s), K.output_size(n, k, p, s))


def test_stride_paths_agree_between_the_two_implementations():
    """At stride 1 an index bug is invisible; it only shows up at stride 2."""
    img = images.shapes_gray()
    for stride in (1, 2, 3):
        a = K.correlate2d_loops(img, K.BOX3, stride=stride)
        b = K.correlate2d(img, K.BOX3, stride=stride)
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-9)


def test_convolution_flips_and_correlation_does_not():
    img = images.shapes_gray()
    # Symmetric kernel: the flip is a no-op and the bug would hide.
    assert np.allclose(K.convolve2d(img, K.BOX3), K.correlate2d(img, K.BOX3))
    # Asymmetric kernel: the flip is exactly a 180-degree rotation.
    assert np.allclose(K.convolve2d(img, K.EMBOSS),
                       K.correlate2d(img, np.rot90(K.EMBOSS, 2)))
    assert not np.allclose(K.convolve2d(img, K.EMBOSS), K.correlate2d(img, K.EMBOSS))


def test_kernel_sum_predicts_what_happens_to_the_mean():
    img = images.shapes_gray().astype(np.float64)
    mean = img.mean()
    for kernel in (K.BOX3, K.SHARPEN):
        assert kernel.sum() == pytest.approx(1.0)
        assert K.correlate2d(img, kernel).mean() == pytest.approx(mean, abs=0.05)
    assert K.LAPLACIAN4.sum() == pytest.approx(0.0)
    assert K.correlate2d(img, K.LAPLACIAN4).mean() == pytest.approx(0.0, abs=0.05)


def test_uint8_would_have_destroyed_the_laplacian_output():
    """The reason correlate2d's first line is a cast to float."""
    out = K.correlate2d(images.shapes_gray(), K.LAPLACIAN4)
    assert out.min() < 0, "an edge kernel produces negatives"
    assert out.max() > 255, "and values past the top of uint8"
    lost = int(((out < 0) | (out > 255)).sum())
    assert lost > 0
    # In uint8 both ends clip silently, so half the edge information -- always
    # the same half, the bright-to-dark transitions -- disappears with no error.


def test_even_kernels_are_refused_rather_than_silently_shifting_the_image():
    with pytest.raises(ValueError):
        K.output_size(100, 4, 1, 1)
    with pytest.raises(ValueError):
        K.correlate2d(np.zeros((10, 10)), np.ones((4, 4)))


def test_unknown_border_is_refused():
    with pytest.raises(ValueError):
        K.pad(np.zeros((4, 4)), 1, border="mirror-ish")
