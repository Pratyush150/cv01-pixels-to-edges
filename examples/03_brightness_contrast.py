"""03 -- Brightness and contrast: out = alpha * in + beta.

One line of arithmetic and three ways to get it wrong: doing it inside uint8,
scaling without a pivot, and handing a negative beta to `convertScaleAbs`.

    python3 examples/03_brightness_contrast.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import images, photometry as P
from pixels.figures import new_figure, save, show_bgr, show_gray


def main() -> None:
    scene = images.tabletop_scene()
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    print("A. Brightness is beta: a shift. Do it in the wrong dtype and the")
    print("   highlights go black.\n")
    beta = 60
    for name, fn in [("img + beta          ", P.brighten_wrapping),
                     ("np.clip(img + beta) ", P.brighten_clip_too_late),
                     ("int16 -> clip -> u8 ", P.brighten_saturating)]:
        out = fn(np.array([[190, 195, 196, 240, 255]], np.uint8), beta)
        print(f"    {name}: {out[0]}")
    print("    The middle row is the trap: clipping AFTER the add cannot help,")
    print("    because the wrap has already produced a legal uint8.\n")

    print("B. Contrast is alpha: a scale. Scaling is about ZERO, so `alpha`")
    print("   alone brightens as well as spreads -- nothing gets darker.\n")
    alpha = 1.6
    b = P.pivot_beta(alpha)
    print(f"    alpha = {alpha}  ->  pivot beta = 128 * (1 - alpha) = {b:.1f}\n")
    print(f"    {'x':>5} | {'pivoted':>8} | {'no pivot':>9} | {'convertScaleAbs':>16}")
    print("    " + "-" * 48)
    for x in (5, 50, 128, 200):
        px = np.array([[x]], np.uint8)
        piv = int(P.contrast_pivoted(px, alpha)[0, 0])
        nop = int(P.contrast_naive(px, alpha)[0, 0])
        csa = int(cv2.convertScaleAbs(px, alpha=alpha, beta=b)[0, 0])
        print(f"    {x:5d} | {piv:8d} | {nop:9d} | {csa:16d}")
    print("\n    Read the middle row: mid-grey stays at exactly 128 under the pivot.")
    print("    That is the signature of a correct contrast control.")
    print("    Read the top row: convertScaleAbs turned 5 into 69 -- it took the")
    print("    ABSOLUTE VALUE of -68.8 instead of clipping it to 0. The pixel")
    print("    reflected off zero instead of stopping there.\n")

    cut = P.convert_scale_abs_reflects_below(alpha, b)
    n_bad = int((gray < cut).sum())
    print(f"    convertScaleAbs is wrong for every x below {cut:.0f}, which on this")
    print(f"    image is {n_bad:,} of {gray.size:,} pixels ({100 * n_bad / gray.size:.1f}%).")
    print("    It agrees with the correct answer everywhere else, so the bug shows")
    print("    up only in the shadows -- where you look least.")
    print("    Rule: convertScaleAbs is safe only when beta >= 0. Pivoted contrast")
    print("    makes beta negative by construction. Its real job is displaying a")
    print("    signed gradient, where the absolute value is the feature.\n")

    # -------------------------------------------------------------- the figure
    fig, ax = new_figure(2, 4, width=14.0, height=6.4)

    show_bgr(ax[0], scene, "original")
    show_bgr(ax[1], P.brighten_wrapping(scene, beta), f"BUG: scene + {beta} in uint8")
    show_bgr(ax[2], P.brighten_saturating(scene, beta), f"FIX: saturating + {beta}")
    show_bgr(ax[3], P.contrast_pivoted(scene, 1.6), "contrast alpha=1.6, pivoted at 128")

    show_gray(ax[4], gray, "grayscale original")
    show_gray(ax[5], P.contrast_naive(gray, alpha), f"BUG: gray * {alpha}, no pivot")
    show_gray(ax[6], P.contrast_pivoted(gray, alpha), f"FIX: gray * {alpha} about 128")
    show_gray(ax[7], cv2.convertScaleAbs(gray, alpha=alpha, beta=b),
              f"BUG: convertScaleAbs, beta={b:.1f}")
    save(fig, "03_brightness_contrast.png")

    # The transfer curves make the arithmetic visible in a way images cannot.
    import matplotlib.pyplot as plt
    x = np.arange(256)
    px = x.astype(np.uint8).reshape(1, -1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(x, P.brighten_wrapping(px, beta)[0], lw=1.6, label=f"img + {beta} (uint8)")
    axes[0].plot(x, P.brighten_saturating(px, beta)[0], lw=1.6, ls="--",
                 label="int16 -> clip -> uint8")
    axes[0].axvline(256 - beta, color="0.4", lw=0.8)
    axes[0].annotate(f"wraps from {256 - beta}", (256 - beta, 15), (40, 35),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    axes[0].set_title("brightness: what uint8 does to a +60 shift")
    axes[1].plot(x, P.contrast_pivoted(px, alpha)[0], lw=1.6, label="pivoted at 128")
    axes[1].plot(x, P.contrast_naive(px, alpha)[0], lw=1.6, ls="--", label="no pivot")
    axes[1].plot(x, cv2.convertScaleAbs(px, alpha=alpha, beta=b)[0], lw=1.6, ls=":",
                 label="convertScaleAbs")
    axes[1].axvline(cut, color="0.4", lw=0.8)
    axes[1].annotate("reflects off 0\nbelow x=48", (cut, 120), (70, 180),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    axes[1].set_title(f"contrast: alpha={alpha}, three transfer curves")
    for a in axes:
        a.set_xlabel("input value")
        a.set_ylabel("output value")
        a.set_xlim(0, 255)
        a.set_ylim(0, 260)
        a.grid(alpha=0.25)
        a.legend(fontsize=8, loc="upper left" if a is axes[0] else "lower right")
    fig.tight_layout()
    save(fig, "03_transfer_curves.png")

    print("The transfer-curve figure is the one to keep. Every bug in this example")
    print("is one line on it: the cliff at 196, the curve that never goes below its")
    print("input, and the V-shape where convertScaleAbs bounces off zero.")


if __name__ == "__main__":
    main()
