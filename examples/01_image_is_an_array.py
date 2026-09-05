"""01 -- An image is an array.

Reads real pixel values off a synthetic photograph, shows that a channel is a
whole grayscale image in its own right, and prices the array in bytes.

    python3 examples/01_image_is_an_array.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import colour, dtypes, images
from pixels.figures import new_figure, save, show_bgr, show_gray


def main() -> None:
    print(f"numpy {np.__version__} | opencv {cv2.__version__}\n")

    # ---------------------------------------------------------------- the ramp
    ramp = images.gray_ramp()
    print("A. The smallest image whose every pixel you can predict")
    print("   ", dtypes.dtype_report(ramp))
    print(f"    ramp[0, 0]   = {ramp[0, 0]}   (top-left, black)")
    print(f"    ramp[50, 200]= {ramp[50, 200]}   column 200 holds the value 200")
    print(f"    ramp[99, 255]= {ramp[99, 255]}   (bottom-right, white)")
    print("    the picture IS the numbers; nothing else is going on\n")

    # ------------------------------------------------------------ a photograph
    scene = images.tabletop_scene()
    print("B. A photograph, read as numbers")
    print("   ", dtypes.dtype_report(scene))
    b, g, r = scene[105, 80]
    print(f"    scene[105, 80] = {scene[105, 80]}  <- [B, G, R], OpenCV's order")
    print(f"      so this pixel is R={r} G={g} B={b}: strongly red, on the lit disc")
    b, g, r = scene[268, 412]
    print(f"    scene[268, 412]= {scene[268, 412]}  same paint, in the shadow: "
          f"R={r} G={g} B={b}")
    print("      all three channels fell together. That is what a shadow does,")
    print("      and it is why example 04 needs a colour space that separates")
    print("      'which colour' from 'how bright'.\n")

    print("C. A channel is a full 2-D grid, not a property of a pixel")
    for i, name in enumerate("BGR"):
        chan = scene[:, :, i]
        print(f"    scene[:, :, {i}] -> shape {chan.shape}, "
              f"mean {chan.mean():6.2f}  ({name})")
    print("    Lift one out on its own and what you have is a grayscale image.\n")

    print("D. Grayscale is a weighted sum, not an average")
    gray = colour.to_gray(scene)
    for name, bgr in [("pure blue", (255, 0, 0)), ("pure green", (0, 255, 0)),
                      ("pure red", (0, 0, 255))]:
        px = np.array([[bgr]], np.uint8)
        print(f"    {name:11s} BGR{bgr} -> luma {int(colour.to_gray(px)[0, 0]):3d} "
              f"(a plain mean would say 85 for all three)")
    same = 100.0 * (gray == cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)).mean()
    print(f"    our hand-written luma matches cv2.cvtColor on {same:.2f}% of pixels,")
    print("    and never by more than one level (see tests/test_colour.py)\n")

    print("E. size counts numbers, nbytes counts bytes")
    print(f"    scene as uint8   : {scene.size:>9,} numbers  {scene.nbytes:>10,} bytes")
    f32 = scene.astype(np.float32)
    print(f"    scene as float32 : {f32.size:>9,} numbers  {f32.nbytes:>10,} bytes"
          f"  ({f32.nbytes / scene.nbytes:.0f}x)")
    frame = 1920 * 1080 * 3
    print(f"    one 1080p frame  : {frame:>9,} bytes = {frame / 1e6:.2f} MB uint8, "
          f"{frame * 4 / 1e6:.1f} MB float32")
    print(f"    at 30 fps that is {frame * 30 / 1e6:.1f} MB/s of memory traffic, "
          "whatever the file on disk says\n")

    # -------------------------------------------------------------- the figure
    fig, ax = new_figure(2, 3, width=12.0, height=6.6)

    show_gray(ax[0], ramp, "gray_ramp(): column c holds the value c")

    # An 8x8 patch with the actual integers written on it -- the moment the
    # abstraction stops being abstract.  Ticks go back on here because the row
    # and column indices are the entire point of the panel.
    patch = scene[100:108, 134:142, 2]
    ax[1].imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for rr in range(8):
        for cc in range(8):
            v = int(patch[rr, cc])
            ax[1].text(cc, rr, str(v), ha="center", va="center", fontsize=6,
                       color="black" if v > 127 else "white")
    ax[1].set_title("scene[100:108, 134:142, 2]  -- the rim of the red disc")
    ax[1].set_xticks(range(8), [str(134 + i) for i in range(8)], fontsize=6)
    ax[1].set_yticks(range(8), [str(100 + i) for i in range(8)], fontsize=6)

    show_bgr(ax[2], scene, "the whole scene, converted BGR->RGB for display")
    for i, name in enumerate(("B", "G", "R")):
        show_gray(ax[3 + i], scene[:, :, i], f"channel {i} = {name}, as its own image")

    save(fig, "01_image_is_an_array.png")

    # The BGR bug on its own, because it deserves its own picture.
    fig, ax = new_figure(1, 2, width=9.0, height=3.6)
    ax[0].imshow(scene, interpolation="nearest")   # deliberately NOT converted
    ax[0].set_title("BUG: BGR array handed straight to matplotlib")
    show_bgr(ax[1], scene, "FIX: cv2.cvtColor(img, COLOR_BGR2RGB) first")
    save(fig, "01_bgr_vs_rgb.png")

    print("The two figures are in docs/figures/. Open 01_bgr_vs_rgb.png: the red")
    print("discs are blue on the left. Same array, same shape, same dtype, every")
    print("value in range -- and no exception anywhere. That is the shape of")
    print("nearly every colour bug you will meet.")


if __name__ == "__main__":
    main()
