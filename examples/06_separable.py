"""06 -- Separable kernels: K*K multiplies become 2*K, measured on this CPU.

Shows which kernels factor and which do not, proves the factorisation is exact
rather than approximate, and times both routes.

    python3 examples/06_separable.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import convolve as K, images, separable as S
from pixels.figures import new_figure, save, show_gray


def main() -> None:
    print("A. Which kernels factor? Rank 1 means yes, anything higher means no.\n")
    zoo = {
        "3x3 box": K.BOX3,
        "Sobel-x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "5x5 Gaussian": S.gaussian_kernel_2d(5, 5 / 6.0),
        "Laplacian": K.LAPLACIAN4,
        "emboss": K.EMBOSS,
    }
    for name, kern in zoo.items():
        rank = int(np.linalg.matrix_rank(kern))
        sep = S.is_separable(kern)
        print(f"    {name:14s} rank {rank}  ->  separable: {sep}")
        if sep:
            col, row = S.separate(kern)
            ok = np.allclose(np.outer(col, row), kern)
            print(f"      col {np.round(col, 3)}  x  row {np.round(row, 3)}  "
                  f"-> reproduces the kernel: {ok}")
    print("\n    Sobel-x factors into a blur [1,2,1] and a difference [-1,0,1], with")
    print("    a shared scale and both signs flipped -- SVD is free to negate both")
    print("    vectors at once because the outer product does not notice. So the 2")
    print("    in Sobel's middle row is literally a smoothing pass glued to a")
    print("    differencing pass, not a magic constant.")
    print("    The Laplacian is rank 2. Forcing it through its largest singular")
    print("    vector would give a different filter and no error message.\n")

    print("B. Our Gaussian against cv2.getGaussianKernel\n")
    for k in (5, 15, 31):
        sigma = k / 6.0
        d = np.abs(S.gaussian_kernel_1d(k, sigma) - cv2.getGaussianKernel(k, sigma).ravel())
        print(f"    k={k:2d}, sigma={sigma:5.2f}: max |difference| = {d.max():.2e}")
    print()

    print("C. Timing. Both routes accumulate whole shifted arrays, so the only")
    print("   difference between them is the number of multiply-adds per pixel --")
    print("   which is what separability actually changes. Timing a Python loop")
    print("   against a library call would measure interpreter overhead instead\n")
    rng = np.random.default_rng(0)
    img = rng.random((512, 512)) * 255
    rows = S.benchmark(img, sizes=(3, 7, 15, 31, 63), repeats=7)
    print(f"    {'k':>3} {'mults 2D':>9} {'mults sep':>10} {'predicted':>10} "
          f"{'2D (ms)':>9} {'sep (ms)':>9} {'measured':>9} {'max diff':>10}")
    for r in rows:
        print(f"    {r['k']:3d} {r['mults_2d']:9d} {r['mults_sep']:10d} "
              f"{r['predicted']:9.1f}x {r['t_2d'] * 1e3:9.1f} {r['t_sep'] * 1e3:9.2f} "
              f"{r['measured']:8.2f}x {r['max_abs_diff']:10.1e}")

    print("\n    Two separate claims, and they must not be blurred together.")
    print("    The `max diff` column is EXACT: separability is a factorisation, not")
    print("    an approximation. Anything above floating-point dust there is a")
    print("    defect, and the usual culprit is the two 1-D passes padding")
    print("    differently at the border.")
    print("    The `measured` column is a wall-clock timing on one CPU and it")
    print(f"    wobbles. At k=3 it lands near 1 (this run: {rows[0]['measured']:.2f}x),")
    print("    because dropping 9 multiplies to 6 is swamped by the fixed cost of")
    print("    each whole-array operation. The saving becomes real from k=7 and then")
    print("    OVERTAKES the prediction, because the 2-D pass also touches k^2 times")
    print("    as much memory and at large k the cache, not the multiplier, is the")
    print("    bottleneck. The claim that survives is 'same order, growing with k'.\n")

    print("D. And the same thing through the library, for scale\n")
    photo = images.shapes_gray().astype(np.float64)
    for k in (15, 31):
        sigma = k / 6.0
        g2 = S.gaussian_kernel_2d(k, sigma)
        a = cv2.filter2D(photo, cv2.CV_64F, g2, borderType=cv2.BORDER_REFLECT_101)
        b = cv2.GaussianBlur(photo, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)
        print(f"    k={k}: filter2D (one 2-D pass) vs GaussianBlur (two 1-D passes) "
              f"agree to {np.abs(a - b).max():.2e}")
    print("    That is why `cv2.GaussianBlur` stays affordable at large sigma: it")
    print("    never runs a 2-D pass. `cv2.filter2D` cannot know your kernel factors")
    print("    unless you tell it, which is what `cv2.sepFilter2D` is for.\n")

    # -------------------------------------------------------------- the figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ks = [r["k"] for r in rows]
    axes[0].plot(ks, [r["predicted"] for r in rows], "o-", lw=1.6,
                 label="predicted: k*k / 2k")
    axes[0].plot(ks, [r["measured"] for r in rows], "s--", lw=1.6,
                 label="measured on this CPU")
    axes[0].axhline(1.0, color="0.5", lw=0.8)
    axes[0].set_xlabel("kernel size k")
    axes[0].set_ylabel("speed-up, 2-D pass / separable")
    axes[0].set_title("separable speed-up: multiply count vs the clock")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(ks, [r["t_2d"] * 1e3 for r in rows], "o-", lw=1.6, label="one k x k pass")
    axes[1].plot(ks, [r["t_sep"] * 1e3 for r in rows], "s--", lw=1.6,
                 label="two 1-D passes")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("kernel size k")
    axes[1].set_ylabel("time per 512x512 image (ms, log scale)")
    axes[1].set_title("cost grows as k^2 one way and as k the other")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "06_separable_timing.png")

    src = images.shapes_gray()
    k = 31
    g1 = S.gaussian_kernel_1d(k, k / 6.0)
    two_d = S.correlate_shift_2d(src, S.gaussian_kernel_2d(k, k / 6.0))
    sep = S.correlate_shift_separable(src, g1, g1)
    fig, ax = new_figure(1, 4, width=14.0, height=3.6)
    show_gray(ax[0], src, "input")
    show_gray(ax[1], np.clip(two_d, 0, 255), f"one {k}x{k} pass ({k * k} mults/px)")
    show_gray(ax[2], np.clip(sep, 0, 255), f"two 1-D passes ({2 * k} mults/px)")
    # Pinned to 0..1, NOT autoscaled. Autoscaling would stretch 1e-13 of
    # floating-point dust across the full black-to-white range and draw a
    # picture of a difference that is not there -- which is exactly the trap
    # `show_gray`'s explicit vmin/vmax exists to prevent.
    show_gray(ax[3], np.abs(two_d - sep), f"|difference| on a 0..1 scale\nmax "
                                          f"{np.abs(two_d - sep).max():.1e}",
              vmin=0, vmax=1.0)
    save(fig, "06_separable_identical.png")

    print("The right-hand panel of the second figure is drawn on a fixed 0..1 grey")
    print("scale, so it is black. Autoscale it instead and matplotlib will stretch")
    print("1e-13 of floating-point dust across the full range and hand you a")
    print("convincing picture of a difference that is not there.")


if __name__ == "__main__":
    main()
