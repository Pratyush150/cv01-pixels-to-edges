"""08 -- Thresholding: global, Otsu from scratch, and adaptive.

Works Otsu's criterion by hand on a twenty-pixel image, checks it against
OpenCV, then puts all three methods on a page photographed under a lamp where
the ground truth is known.

    python3 examples/08_thresholding.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import images, threshold as T
from pixels.figures import new_figure, save, show_gray


def main() -> None:
    # ------------------------------------------------ A. Otsu on 20 pixels
    print("A. Otsu by hand on a 4x5 crop of a document scan\n")
    tiny = np.array([[40, 40, 40, 40, 50],
                     [50, 50, 50, 50, 50],
                     [190, 190, 190, 190, 190],
                     [190, 200, 200, 200, 200]], np.uint8)
    print("    the crop:")
    print("   ", str(tiny).replace("\n", "\n    "))
    hist = T.histogram256(tiny)
    present = np.nonzero(hist)[0]
    print(f"\n    histogram: " + "  ".join(f"{v}x{hist[v]}" for v in present))
    print("    Two humps -- ink near 40-50, paper near 190-200 -- and a wide empty")
    print("    valley between them. You can see the answer without arithmetic; the")
    print("    method has to reach it by scoring all 256 candidates.\n")

    var_total = float(tiny.astype(np.float64).var())
    print(f"    {'T':>5} {'within-class':>14} {'between-class':>15} {'sum':>9}")
    for t in (40, 50, 190):
        w = T.within_class_variance(tiny, t)
        b = float(T.between_class_variance_curve(tiny)[t])
        print(f"    {t:5d} {w:14.2f} {b:15.2f} {w + b:9.2f}")
    print(f"    {'':5} {'':14} {'image variance':>15} {var_total:9.2f}")
    print("\n    The three rows sum to the same total, and that total is fixed by")
    print("    the image alone -- no choice of T moves it. Pushing the within-class")
    print("    spread down and pulling the between-class separation up are the same")
    print("    operation seen from two sides. Production code computes the")
    print("    between-class form: running sums of the histogram are all it needs.\n")

    curve = T.between_class_variance_curve(tiny)
    best = curve.max()
    winners = np.nonzero(np.abs(curve - best) < 1e-9)[0]
    t_mine = T.otsu_threshold(tiny)
    t_cv2, mask_cv2 = cv2.threshold(tiny, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"    the maximum is achieved for EVERY T from {winners.min()} to "
          f"{winners.max()},")
    print("    since the image holds no values in that gap, every one of those")
    print("    thresholds cuts the pixels into the same two groups.")
    print(f"    ours returns {t_mine} (argmax takes the first), cv2 returns "
          f"{int(t_cv2)}.")
    print("    Matching that tie-break is not pedantry: every image Otsu is good at")
    print("    has a wide empty valley, so a different tie-break disagrees with")
    print("    OpenCV on essentially every real input.\n")

    # ------------------------------------------------ B. the same page, lit badly
    print("B. A page under a lamp: where a single number cannot win\n")
    scan, truth = images.lit_document()
    w = scan.shape[1]
    print(f"    leftmost 50 columns mean {scan[:, :50].mean():6.1f}   "
          f"rightmost 50 columns mean {scan[:, -50:].mean():6.1f}")
    print(f"    true ink coverage: {100 * (truth > 0).mean():.1f}% overall, "
          f"{100 * (truth[:, :w // 2] > 0).mean():.1f}% left half, "
          f"{100 * (truth[:, w // 2:] > 0).mean():.1f}% right half\n")

    t_otsu = T.otsu_threshold(scan)
    methods = {
        "global T=127": T.global_threshold(scan, 127, invert=True),
        f"otsu T={t_otsu}": T.global_threshold(scan, t_otsu, invert=True),
        "adaptive 31/10": T.adaptive_threshold_mean(scan, 31, 10, invert=True),
    }
    print(f"    {'method':16s} {'IoU':>6s} {'ink found, left':>16s} "
          f"{'ink found, right':>17s}")
    for name, m in methods.items():
        print(f"    {name:16s} {T.iou(m, truth):6.3f} "
              f"{100 * (m[:, :w // 2] > 0).mean():15.1f}% "
              f"{100 * (m[:, w // 2:] > 0).mean():16.1f}%")
    print(f"    {'(truth)':16s} {1.0:6.3f} "
          f"{100 * (truth[:, :w // 2] > 0).mean():15.1f}% "
          f"{100 * (truth[:, w // 2:] > 0).mean():16.1f}%")
    print("\n    Read the last column. The truth is about 18% ink in the right half.")
    print("    The global threshold claims nearly 100%: the shadowed paper has fallen")
    print("    below 127, so as far as one fixed number is concerned the whole")
    print("    right-hand side of the sheet is writing. Otsu improves on that and is")
    print("    still wrong, for the same structural reason -- one number for the")
    print("    entire frame cannot straddle that ramp. Adaptive")
    print("    asks a different question -- 'darker than your own neighbourhood?' --")
    print("    and a shadow moves a pixel and its neighbourhood together.\n")

    print("C. Our adaptive threshold against cv2.adaptiveThreshold")
    theirs = cv2.adaptiveThreshold(scan, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    ours = methods["adaptive 31/10"]
    agree = 100.0 * ((ours > 0) == (theirs > 0)).mean()
    print(f"    agreement: {agree:.4f}% of pixels "
          f"({int(((ours > 0) != (theirs > 0)).sum())} differ)")
    print("    Not 100%, and the reason is worth knowing: OpenCV computes the local")
    print("    mean in integer arithmetic and rounds, we accumulated in float64.")
    print("    The pixels that differ are the ones sitting within half a level of")
    print("    their own local mean, which is exactly the set where the answer was")
    print("    a coin toss anyway.\n")

    print("D. Two ways Otsu lies to you\n")
    rng = np.random.default_rng(3)
    flat = np.clip(rng.normal(128, 10, (200, 200)), 0, 255).astype(np.uint8)
    t_flat = T.otsu_threshold(flat)
    m_flat = T.global_threshold(flat, t_flat)
    print(f"    unimodal noise, no valley at all -> Otsu returns T = {t_flat}, "
          f"and {100 * T.foreground_fraction(m_flat):.1f}% of the")
    print("    image comes back foreground. No implementation anywhere reports")
    print("    'your histogram has only one hump'. The affordable guard is to check")
    print("    the foreground fraction: near 50% when you expected a small object")
    print("    means the threshold is meaningless.\n")

    clean = np.zeros((200, 200), np.uint8)
    cv2.rectangle(clean, (50, 50), (150, 150), 200, -1)
    truth_sq = clean > 100
    noisy = images.add_gaussian_noise(clean, 45, seed=2)
    t_noisy = T.otsu_threshold(noisy)
    wrong_after = int(((noisy > t_noisy) != truth_sq).sum())
    blurred = cv2.GaussianBlur(noisy, (0, 0), 2.0)
    t_blur = T.otsu_threshold(blurred)
    wrong_before = int(((blurred > t_blur) != truth_sq).sum())
    print(f"    noise fattens both humps until they run into each other:")
    print(f"      Otsu on the noisy image      -> T={t_noisy}, "
          f"{wrong_after:,} wrong pixels of {noisy.size:,}")
    print(f"      one GaussianBlur, then Otsu  -> T={t_blur}, "
          f"{wrong_before:,} wrong pixels")
    print("    The blur belongs upstream of Otsu. Otsu only ever looks at the")
    print("    histogram, so tidying the mask afterwards arrives too late -- the")
    print("    misclassified pixels are already decided.\n")

    # -------------------------------------------------------------- the figures
    fig, ax = new_figure(2, 3, width=12.0, height=6.6)
    show_gray(ax[0], scan, "a page under a lamp (lit_document())")
    show_gray(ax[1], truth, "ground truth: the ink we drew")
    # A line, not an image: this panel exists to show that one horizontal
    # threshold cannot sit above the ink everywhere and below the paper
    # everywhere at once, and an image of the column means cannot show that.
    cols = np.arange(w)
    paper_only = np.where(truth > 0, np.nan, scan.astype(float))
    ink_only = np.where(truth > 0, scan.astype(float), np.nan)
    ax[2].plot(cols, np.nanmean(paper_only, axis=0), lw=1.4, label="paper")
    ax[2].plot(cols, np.nanmean(ink_only, axis=0), lw=1.4, label="ink")
    ax[2].axhline(127, color="steelblue", ls="--", lw=1.2, label="global T=127")
    ax[2].axhline(t_otsu, color="crimson", ls=":", lw=1.4, label=f"Otsu T={t_otsu}")
    ax[2].set_title("column means: the paper crosses below T")
    ax[2].set_xlabel("column")
    ax[2].set_ylabel("pixel value")
    ax[2].set_xticks([0, 200, 399])
    ax[2].set_yticks([0, 64, 128, 192, 255])
    ax[2].legend(fontsize=7, loc="upper right")
    ax[2].grid(alpha=0.25)
    for a, (name, m) in zip(ax[3:], methods.items()):
        show_gray(a, m, f"{name}\nIoU {T.iou(m, truth):.3f}")
    save(fig, "08_thresholding.png")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    h = T.histogram256(scan)
    axes[0].bar(np.arange(256), h, width=1.0, color="0.35")
    axes[0].axvline(t_otsu, color="crimson", lw=1.6, label=f"Otsu T = {t_otsu}")
    axes[0].axvline(127, color="steelblue", lw=1.6, ls="--", label="global T = 127")
    axes[0].set_title("histogram of the lit page: no clean valley")
    axes[0].set_xlabel("pixel value")
    axes[0].set_ylabel("pixel count")
    axes[0].legend(fontsize=8)

    c = T.between_class_variance_curve(scan)
    axes[1].plot(np.arange(256), np.where(c < 0, np.nan, c), lw=1.6)
    axes[1].axvline(t_otsu, color="crimson", lw=1.6, label=f"argmax at T = {t_otsu}")
    axes[1].set_title("Otsu's criterion: between-class variance vs T")
    axes[1].set_xlabel("candidate threshold T")
    axes[1].set_ylabel(r"$\sigma_B^2 = w_0 w_1 (\mu_0 - \mu_1)^2$")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "08_otsu_criterion.png")

    print("The right-hand plot is Otsu in one picture: a single smooth curve over")
    print("256 candidates, and the threshold is wherever it peaks. The left-hand")
    print("plot is why that peak is not good enough here -- the lamp has smeared")
    print("the paper across the whole range, so there is no valley to find.")


if __name__ == "__main__":
    main()
