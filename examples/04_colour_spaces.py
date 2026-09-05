"""04 -- Colour spaces, and one threshold that only HSV can win.

Same photograph, same object, same ground truth. A box in BGR versus a wedge
of the hue wheel, both searched exhaustively so the comparison is fair.

    python3 examples/04_colour_spaces.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import colour, images
from pixels.figures import new_figure, save, show_bgr, show_gray


LIT = np.s_[80:130, 55:105]        # inside the red disc under the lamp
SHADOW = np.s_[245:290, 390:435]   # inside the red disc in the shadow


def main() -> None:
    scene = images.tabletop_scene()
    truth = images.tabletop_truth()
    hsv_mine = colour.bgr_to_hsv(scene)
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)

    print("A. Our hand-written BGR->HSV against cv2.cvtColor")
    for i, name in enumerate("HSV"):
        d = np.abs(hsv_mine[:, :, i].astype(int) - hsv[:, :, i].astype(int))
        if i == 0:
            d = np.minimum(d, 180 - d)     # hue is cyclic: 179 and 0 are 1 apart
        print(f"    {name}: identical on {100 * (d == 0).mean():6.2f}% of pixels, "
              f"never off by more than {int(d.max())}")
    print("    The remainder is rounding: OpenCV runs a fixed-point integer path")
    print("    and we ran float64. One level, and it is not a bug in either.\n")

    print("B. What a shadow does to the same paint")
    print(f"    {'':10s} {'B':>5s} {'G':>5s} {'R':>5s}   |{'H':>5s} {'S':>5s} {'V':>5s}")
    for name, sl in [("lit    ", LIT), ("shadow ", SHADOW)]:
        b, g, r = [scene[sl][:, :, i].mean() for i in range(3)]
        h, s, v = [hsv[sl][:, :, i].mean() for i in range(3)]
        print(f"    {name:10s} {b:5.0f} {g:5.0f} {r:5.0f}   |{h:5.0f} {s:5.0f} {v:5.0f}")
    ratios = [scene[LIT][:, :, i].mean() / max(scene[SHADOW][:, :, i].mean(), 1e-9)
              for i in range(3)]
    print(f"    lit / shadow per channel: B {ratios[0]:.2f}x  G {ratios[1]:.2f}x  "
          f"R {ratios[2]:.2f}x")
    print("    All three fell together, by roughly a factor of three, because light")
    print("    is MULTIPLICATIVE: illumination slides a colour along a ray through")
    print("    the origin of BGR space. Hue is the angle of that ray and V is the")
    print("    distance along it, so V collapses and hue barely moves. Saturation is")
    print("    a ratio of channel differences, so it survives too.")
    print("    The three factors are not identical -- 2.4x, 3.1x, 3.6x -- because the")
    print("    lamp is warm and the shadow is lit by cooler bounced light, so the ray")
    print("    tilts slightly as well as shortening. That tilt is the whole reason")
    print("    hue moves at all here, and it moves by 2 out of 180 while R falls by")
    print("    154 out of 255.\n")

    print("C. Red sits on the seam of the hue wheel, and that is a trap")
    h_lit = int(np.median(hsv[LIT][:, :, 0]))
    h_shadow = int(np.median(hsv[SHADOW][:, :, 0]))
    print(f"    median hue, lit disc    : {h_lit}")
    print(f"    median hue, shadow disc : {h_shadow}")
    print(f"    plain subtraction says they are {abs(h_lit - h_shadow)} apart.")
    print(f"    They are {int(colour.hue_distance(np.array([h_shadow]), h_lit)[0])} apart:")
    print("    hue is an ANGLE, OpenCV stores it 0..179 (the wheel halved to fit a")
    print("    byte), and 179 and 0 are neighbours. Both are red. This is why every")
    print("    tutorial detects red with two inRange calls -- or, as here, with one")
    print("    circular distance, which makes red no harder than green.\n")

    naive = ((np.abs(hsv[:, :, 0].astype(int) - h_lit) <= 7)
             & (hsv[:, :, 1] >= 88)).astype(np.uint8) * 255
    print(f"    a hue window using plain |H - {h_lit}| <= 7 : IoU "
          f"{colour.iou(naive, truth):.3f}")
    print("      -- it finds the lit disc and misses the shadowed one completely,")
    print("      because that one's hue reads 179 and 179 - 1 = 178.\n")

    print("D. Now threshold for 'the red discs', both ways.")
    print("   Each family has three free parameters and both are searched")
    print("   exhaustively against the same ground truth, so what is being compared")
    print("   is the colour space and not how hard somebody tried.\n")

    bgr_iou, (r_lo, g_hi, b_hi) = colour.best_bgr_rule(scene, truth)
    hue_iou, (centre, half, s_min) = colour.best_hue_rule(hsv, truth)
    bgr_mask = colour.mask_by_bgr_box(scene, r_lo, g_hi, b_hi)
    hue_mask = colour.mask_by_hue(hsv, centre, half, s_min)

    print(f"    best BGR box   R>={r_lo:3d}, G<={g_hi:3d}, B<={b_hi:3d}"
          f"        -> IoU {bgr_iou:.4f}")
    print(f"    best hue wedge |H-{centre}|<={half}, S>={s_min}, V>=25"
          f"   -> IoU {hue_iou:.4f}\n")

    # A rule tuned only on the lit disc, to show what actually goes wrong.
    lit_only = colour.mask_by_bgr_box(scene, 150, 90, 90)
    print("    And the rule you would have written by eye, tuned on the lit disc")
    print(f"    (R>=150, G<=90, B<=90): IoU {colour.iou(lit_only, truth):.3f}")
    lit_truth = truth[:, :240] > 0
    shadow_truth = truth[:, 240:] > 0
    print(f"      it recovers {100 * (lit_only[:, :240] > 0)[lit_truth].mean():5.1f}% "
          "of the lit disc")
    print(f"      and         {100 * (lit_only[:, 240:] > 0)[shadow_truth].mean():5.1f}% "
          "of the shadowed one.")
    print("    Widen it until the shadowed disc gets in and the lit wood arrives")
    print("    with it -- which is what the exhaustive search discovered, and why")
    print(f"    the best BGR box of any tuning tops out at {bgr_iou:.3f}.\n")

    for name, mask in [("best BGR box", bgr_mask), ("best hue wedge", hue_mask)]:
        fp = int(((mask > 0) & (truth == 0)).sum())
        fn = int(((mask == 0) & (truth > 0)).sum())
        print(f"    {name:15s} false positives {fp:6,}  false negatives {fn:6,}")
    print("\n    Be honest about that hue score. This is a synthetic scene, so the")
    print("    discs have exactly one reflectance and the only thing changing across")
    print("    the frame is the illumination -- which is precisely the case hue is")
    print("    invariant to. On a real photograph, inter-reflections, a coloured")
    print("    bounce and sensor noise all move hue as well, and the same rule would")
    print("    score in the low nineties rather than at 0.999. The claim that")
    print("    transfers is the RANKING and its mechanism, not the digit.\n")

    # -------------------------------------------------------------- the figure
    fig, ax = new_figure(2, 4, width=14.0, height=6.4)
    show_bgr(ax[0], scene, "the scene: two red discs, one in shadow")
    show_gray(ax[1], colour.to_gray(scene), "grayscale (0.299R + 0.587G + 0.114B)")
    show_gray(ax[2], hsv[:, :, 0], "H: which colour (0..179)", vmax=179, cmap="hsv")
    show_gray(ax[3], hsv[:, :, 1], "S: how vivid (0..255)")

    show_gray(ax[4], truth, "ground truth: the red discs")
    show_gray(ax[5], lit_only, f"BGR box tuned on the lit disc\nIoU {colour.iou(lit_only, truth):.3f}")
    show_gray(ax[6], bgr_mask, f"best BGR box of any tuning\nIoU {bgr_iou:.3f}")
    show_gray(ax[7], hue_mask, f"best hue wedge\nIoU {hue_iou:.4f}")
    save(fig, "04_colour_spaces.png")

    print("Panel 6 is the point. That is not a badly chosen threshold -- it is the")
    print("best axis-aligned box in BGR that exists for this image. The shadowed")
    print("disc and the lit wood overlap in every channel, and no box can separate")
    print("two sets that overlap. Rotating to a space where 'which colour' is one")
    print("axis makes the same problem trivial.")


if __name__ == "__main__":
    main()
