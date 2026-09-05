"""The two silent bugs, asserted rather than described."""

import cv2
import numpy as np
import pytest

from pixels import dtypes


ALL_VALUES = np.arange(256, dtype=np.uint8).reshape(1, -1)


@pytest.mark.parametrize("delta", [1, 10, 60, 100, 200, 255])
def test_wrap_begins_exactly_where_we_say(delta):
    """The claim `250 + 10 == 4` is a special case of: wrapping starts at 256-delta."""
    cliff = dtypes.wrap_threshold(delta)
    wrapped = dtypes.add_wrapping(ALL_VALUES, delta)[0].astype(int)
    values = np.arange(256)

    # Below the cliff the naive add is simply correct...
    assert np.array_equal(wrapped[:cliff], values[:cliff] + delta)
    # ...and at or above it, every single value has gone round the clock.
    assert np.all(wrapped[cliff:] == (values[cliff:] + delta) - 256)
    # Which means the wrapped result is SMALLER than the input -- the guard any
    # pipeline can afford: no honest brightening can make a pixel darker.
    assert np.all(wrapped[cliff:] < values[cliff:])


def test_saturating_add_matches_opencv_on_every_possible_value():
    """Our four-step widen/add/clip/narrow is exactly what cv2.add does."""
    for delta in (1, 10, 60, 128, 255):
        ours = dtypes.add_saturating(ALL_VALUES, delta)
        theirs = cv2.add(ALL_VALUES, delta)
        assert np.array_equal(ours, theirs), f"disagreement at delta={delta}"


def test_clip_after_the_add_does_not_help():
    """np.clip cannot undo a wrap, because a wrapped value is a legal uint8."""
    from pixels import photometry

    delta = 60
    late = photometry.brighten_clip_too_late(ALL_VALUES, delta)
    correct = dtypes.add_saturating(ALL_VALUES, delta)
    differing = int((late != correct).sum())
    assert differing == delta, "exactly the values from 256-delta up should be wrong"


def test_wrapped_pixel_count_is_zero_on_a_dark_image():
    """Why the bug survives code review: on low-key footage it does not fire."""
    dark = np.full((32, 32), 40, np.uint8)
    assert dtypes.wrapped_pixel_count(dark, 60) == 0
    bright = np.full((32, 32), 220, np.uint8)
    assert dtypes.wrapped_pixel_count(bright, 60) == bright.size


def test_a_slice_is_a_view_and_writing_through_it_propagates():
    original = np.arange(80, dtype=np.uint8).reshape(8, 10)
    before = original.copy()

    img, region = dtypes.mutate_through_slice(original, (2, 5, 3, 7), 0)
    assert dtypes.is_view_of(region, img)
    assert np.all(img[2:5, 3:7] == 0), "the write landed in the region"
    assert not np.array_equal(img, before), "and therefore also in the original"
    assert np.array_equal(img[0:2], before[0:2]), "outside the slice, untouched"


def test_a_copy_is_not_a_view_and_writing_through_it_does_not_propagate():
    original = np.arange(80, dtype=np.uint8).reshape(8, 10)
    before = original.copy()

    img, region = dtypes.mutate_through_copy(original, (2, 5, 3, 7), 0)
    assert not dtypes.is_view_of(region, img)
    assert np.all(region == 0), "the write landed in the copy"
    assert np.array_equal(img, before), "and nowhere else"


def test_the_two_crops_are_indistinguishable_until_you_write():
    """The whole reason this bug is hard: reading tells you nothing."""
    a = np.array([10, 20, 30, 40, 50, 60, 70, 80], np.uint8)
    view, copy = a[2:5], a[2:5].copy()
    assert np.array_equal(view, copy)
    assert view.shape == copy.shape and view.dtype == copy.dtype


def test_size_is_not_nbytes_except_for_uint8():
    u8 = np.zeros((4, 4), np.uint8)
    f32 = np.zeros((4, 4), np.float32)
    assert dtypes.dtype_report(u8)["size"] == dtypes.dtype_report(u8)["nbytes"]
    assert dtypes.dtype_report(f32)["nbytes"] == 4 * dtypes.dtype_report(f32)["size"]
