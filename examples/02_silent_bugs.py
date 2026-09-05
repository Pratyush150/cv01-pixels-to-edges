"""02 -- The two bugs that never raise.

`uint8` arithmetic wraps, and a NumPy slice is a view.  Both are reproduced
here on purpose, shown failing, then fixed.

    python3 examples/02_silent_bugs.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from pixels import dtypes, images, photometry
from pixels.figures import new_figure, save, show_bgr, show_gray


def main() -> None:
    scene = images.tabletop_scene()
    ramp = images.gray_ramp()
    delta = 60

    # ================================================== BUG 1: the uint8 wrap
    print("BUG 1 -- uint8 counts round like a clock\n")
    probe = np.array([[0, 100, 190, 194, 195, 196, 240, 255]], np.uint8)
    wrapped = dtypes.add_wrapping(probe, delta)
    safe = dtypes.add_saturating(probe, delta)
    print(f"    input      : {probe[0]}")
    print(f"    img + {delta}   : {wrapped[0]}   <- NumPy: modulo 256")
    print(f"    saturating : {safe[0]}   <- clipped at 255")
    cliff = dtypes.wrap_threshold(delta)
    print(f"\n    {cliff - 1} + {delta} = 255 is the last honest answer.")
    print(f"    {cliff} + {delta} = 256, and 256 mod 256 is 0.")
    print(f"    So every pixel at or above {cliff} is destroyed, and the damage is")
    print("    concentrated in the highlights -- sky, glints, white paint.\n")

    # The ramp first, because its answer is arithmetic rather than a measurement:
    # it holds every value 0..255 once per row, so exactly the 60 columns from
    # 196 to 255 must wrap -- 60/256 = 23.4% of the image.
    n_ramp = dtypes.wrapped_pixel_count(ramp, delta)
    print(f"    on gray_ramp(): {n_ramp:,} of {ramp.size:,} pixels wrap "
          f"({100 * n_ramp / ramp.size:.1f}%), predicted {100 * delta / 256:.1f}%")

    n_wrap = dtypes.wrapped_pixel_count(scene, delta)
    print(f"    on the photograph: {n_wrap:,} of {scene.size:,} numbers "
          f"({100 * n_wrap / scene.size:.2f}%) wrap -- and that is the dangerous case,")
    print("    because a fraction of a percent will not show up in any summary you check.")
    per_channel = [int((scene[:, :, i] >= 256 - delta).sum()) for i in range(3)]
    print(f"    by channel [B, G, R]: {per_channel}  -- they are the specular")
    print("    highlights and the lit red disc, i.e. exactly the pixels a viewer looks at.\n")

    buggy = dtypes.add_wrapping(scene, delta)
    fixed = dtypes.add_saturating(scene, delta)
    print(f"    whole-image mean -- before {scene.mean():6.2f} | "
          f"buggy {buggy.mean():6.2f} | fixed {fixed.mean():6.2f}")
    print("    Those last two are nearly equal, so a mean-based sanity check passes")
    print("    on the ruined image. A whole-frame average hides a local disaster.")
    hot = np.s_[95:120, 55:85]      # the specular on the lit red disc
    print(f"    the specular alone -- before {scene[hot].mean():6.2f} | "
          f"buggy {buggy[hot].mean():6.2f} | fixed {fixed[hot].mean():6.2f}")
    print("    Inspect a patch where the damage ought to be worst, not the frame.\n")

    print("    And the fix that is not a fix:")
    late = photometry.brighten_clip_too_late(scene, delta)
    print(f"      np.clip(img + {delta}, 0, 255) disagrees with the correct answer on")
    print(f"      {int((late != fixed).sum()):,} numbers -- exactly the ones that wrapped.")
    print("      Python evaluates the inside first, so the wrap already happened;")
    print("      clip cannot undo it, because a wrapped value is a legal uint8.\n")

    # =============================================== BUG 2: views versus copies
    print("BUG 2 -- a slice is a window onto the same memory\n")
    photo = np.array([10, 20, 30, 40, 50, 60, 70, 80], np.uint8)
    view = photo[2:5]
    copy = photo[2:5].copy()
    print(f"    photo           = {photo}")
    print(f"    photo[2:5]      = {view}   (a view)")
    print(f"    photo[2:5].copy() = {copy} (a copy)")
    print("    Identical values. Nothing distinguishes them until you write.\n")
    print(f"    np.shares_memory(photo, view) = {dtypes.is_view_of(view, photo)}")
    print(f"    np.shares_memory(photo, copy) = {dtypes.is_view_of(copy, photo)}")
    copy[:] = 0
    print(f"    after copy[:] = 0 -> photo = {photo}   survived")
    view[:] = 0
    print(f"    after view[:] = 0 -> photo = {photo}   damaged\n")
    print("    Both crops ended up [0 0 0]; the entire difference is what happened")
    print("    to `photo`. And `view[:] = 0` is the same instruction as")
    print("    `photo[2:5] = 0` -- the line that damages `photo` never mentions it.\n")

    # Why NumPy chose the dangerous default: measure it.
    # A real 4K buffer, touched once so the pages are actually resident -- timing
    # a freshly allocated np.zeros measures the kernel faulting pages in, not
    # the copy.
    big = np.zeros((2160, 3840, 3), np.uint8)
    big[::7] = 1
    t = time.perf_counter()
    for _ in range(200):
        _ = big[400:1400, 600:2200]
    t_view = (time.perf_counter() - t) / 200
    big[400:1400, 600:2200].copy()          # warm up
    t = time.perf_counter()
    for _ in range(20):
        _ = big[400:1400, 600:2200].copy()
    t_copy = (time.perf_counter() - t) / 20
    print(f"    one 4K frame is {big.nbytes / 1e6:.1f} MB. Cropping a 1000x1600 region:")
    print(f"      as a view : {t_view * 1e6:8.2f} us")
    print(f"      with copy : {t_copy * 1e6:8.2f} us   ({t_copy / t_view:,.0f}x slower, "
          f"{1000 * 1600 * 3 / 1e6:.1f} MB moved)")
    print("    So the default is right and the rule is about intent:")
    print("      about to write, and the source array still matters -> .copy()")
    print("      about to read -> take the view; copying buys you nothing at all\n")

    # -------------------------------------------------------------- the figure
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    damaged, _ = dtypes.mutate_through_slice(gray.copy(), (60, 180, 120, 300), 0)
    intact, _ = dtypes.mutate_through_copy(gray.copy(), (60, 180, 120, 300), 0)

    fig, ax = new_figure(2, 4, width=14.0, height=6.0)
    show_gray(ax[0], ramp, "gray_ramp(): every value 0..255")
    show_gray(ax[1], dtypes.add_wrapping(ramp, delta), f"BUG: ramp + {delta} in uint8")
    show_gray(ax[2], dtypes.add_saturating(ramp, delta), f"FIX: saturating + {delta}")
    show_gray(ax[3], (ramp >= 256 - delta).astype(np.uint8) * 255,
              f"the {100 * delta / 256:.1f}% at or above {256 - delta}")

    show_bgr(ax[4], scene, "the photograph")
    show_bgr(ax[5], buggy, f"BUG: scene + {delta}  "
                           f"({100 * n_wrap / scene.size:.1f}% of numbers wrap)")
    show_gray(ax[6], damaged, "BUG: wrote through img[60:180, 120:300]")
    show_gray(ax[7], intact, "FIX: wrote through ...copy()")
    save(fig, "02_silent_bugs.png")

    print("Panel 2: the ramp climbs to white, hits 196, and falls off a cliff back")
    print("to black -- that is modulo 256, drawn. Panel 6: on the photograph the")
    print(f"same bug costs {100 * n_wrap / scene.size:.1f}% of the numbers, and every one")
    print("of them is a highlight. No error, no warning, and a mean that checks out.")


if __name__ == "__main__":
    main()
