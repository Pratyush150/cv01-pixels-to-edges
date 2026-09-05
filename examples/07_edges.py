"""07 -- Derivatives and edges: Sobel to a from-scratch Canny.

Checks the hand-worked gradient, shows what non-maximum suppression does and
what happens when you compare along the edge instead of across it, then runs
our Canny against OpenCV's and quantifies the disagreement.

    python3 examples/07_edges.py
"""

from __future__ import annotations

import cv2
import numpy as np

from pixels import edges as E, images
from pixels.figures import new_figure, save, show_gray


def main() -> None:
    # ------------------------------------------------ A. one patch, by hand
    print("A. One 3x3 patch, finished to the last decimal\n")
    patch = np.array([[10, 10, 10],
                      [10, 10, 80],
                      [10, 10, 80]], np.float64)
    gx = float((patch * E.SOBEL_X).sum())
    gy = float((patch * E.SOBEL_Y).sum())
    print(f"    Gx = {gx:.0f}   Gy = {gy:.0f}")
    print(f"    magnitude (L2)     = {np.hypot(gx, gy):.2f}")
    print(f"    gradient direction = {np.degrees(np.arctan2(gy, gx)):.2f} deg  "
          "(across the edge)")
    print(f"    edge orientation   = "
          f"{(np.degrees(np.arctan2(gy, gx)) + 90) % 180:.2f} deg  (along the edge)")
    print("    Gy is 70, not zero -- the bottom-right 80 breaks the patch's")
    print("    top-to-bottom symmetry. What makes this a 'vertical edge' is Gx")
    print("    DOMINATING Gy, not Gy vanishing, which it almost never does.\n")

    print("B. Our Sobel against cv2.Sobel")
    img = images.shapes_gray()
    mx, my = E.sobel(img)
    cx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    cy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    print(f"    max |difference|: Gx {np.abs(mx - cx).max():.1e}, "
          f"Gy {np.abs(my - cy).max():.1e}")
    print("    Exactly zero, because our correlate2d defaults to the same")
    print("    BORDER_REFLECT_101 that every OpenCV filter defaults to.\n")

    print("C. The gradient points ACROSS the edge, never along it\n")
    print(f"    {'test image':18s} {'|Gx|':>7s} {'|Gy|':>7s} {'gradient':>10s} "
          f"{'edge line':>10s}")
    for kind in ("vertical edge", "horizontal edge", "diagonal edge"):
        n = 101
        yy, xx = np.mgrid[0:n, 0:n]
        base = np.zeros((n, n), np.uint8)
        if kind == "vertical edge":
            base[xx >= n // 2] = 200
        elif kind == "horizontal edge":
            base[yy >= n // 2] = 200
        else:
            base[xx + yy >= n] = 200
        blurred = cv2.GaussianBlur(base, (0, 0), 2.0)
        a, b = E.sobel(blurred)
        mag = E.magnitude(a, b, norm="L2")
        sel = mag > 0.6 * mag.max()
        grad = float(np.median(E.direction(a, b)[sel]))
        print(f"    {kind:18s} {np.abs(a[sel]).mean():7.1f} {np.abs(b[sel]).mean():7.1f} "
              f"{grad:9.1f} {(grad + 90) % 180:10.1f}")
    print("    Row 1: large Gx, zero Gy, gradient at 0 deg, edge line at 90 deg.")
    print("    A large Gx puts the arrow along the HORIZONTAL and the edge line")
    print("    along the VERTICAL. Two statements about one pixel; say both.\n")

    # ------------------------------ D. NMS, and the bug that erases your edges
    print("D. Non-maximum suppression, and what happens if you get it backwards\n")
    row = np.array([10, 10, 40, 90, 140, 150, 150], np.uint8)
    ramp = np.tile(row, (5, 1))
    a, b = E.sobel(ramp)
    mag = E.magnitude(a, b, norm="L2")
    print(f"    image row       : {row}")
    print(f"    magnitude row   : {mag[2].astype(int)}")
    print("    One soft edge, five pixels of ridge. That is the problem NMS solves.\n")

    def thin(m, di, dj):
        out = np.zeros_like(m)
        for i in range(1, m.shape[0] - 1):
            for j in range(1, m.shape[1] - 1):
                if m[i, j] >= m[i + di, j + dj] and m[i, j] > m[i - di, j - dj]:
                    out[i, j] = m[i, j]
        return out

    across = thin(mag, 0, 1)
    along = thin(mag, 1, 0)
    print(f"    compare ACROSS the edge (correct): {across[2].astype(int)}"
          f"  -> {int((across > 0).sum())} pixels kept")
    print(f"    compare ALONG  the edge (bug)    : {along[2].astype(int)}"
          f"  -> {int((along > 0).sum())} pixels kept")
    print("    Every row of this image is identical, so along the edge every pixel")
    print("    ties with its neighbours, nothing is ever strictly largest, and the")
    print("    edge map comes back EMPTY, with no exception raised. Stepping along")
    print("    the ridge does not narrow it -- it wipes it out.\n")

    print("E. Hysteresis: why two thresholds beat one\n")
    strip = np.array([[0, 0, 160, 90, 80, 70, 0, 60, 0]], np.float64)
    strip = np.repeat(strip, 3, axis=0)
    low, high = 50, 150
    h = E.hysteresis(strip, low, high)
    print(f"    thinned magnitudes : {strip[1].astype(int)}")
    print(f"    single threshold at {high}: "
          f"{(strip[1] >= high).astype(int)}  -> the edge broke into a dot")
    print(f"    single threshold at  {low}: "
          f"{(strip[1] >= low).astype(int)}  -> the speck at index 7 got in")
    print(f"    hysteresis {low}/{high}      : "
          f"{(h[1] > 0).astype(int)}  -> the whole edge, and only the edge")
    print("    A high bar to begin an edge, a low one to keep following it.\n")

    # ------------------------------------------- F. our Canny against OpenCV's
    print("F. Our Canny against cv2.Canny, pixel for pixel\n")
    cases = {
        "shapes": images.shapes_gray(),
        "shapes, pre-blurred": cv2.GaussianBlur(images.shapes_gray(), (0, 0), 1.4),
        "checkerboard": images.checkerboard(),
        "diagonal ramp": cv2.GaussianBlur(
            np.where(np.add(*np.mgrid[0:160, 0:160]) >= 150, 200, 40).astype(np.uint8),
            (0, 0), 1.4),
        "pure noise": (np.random.default_rng(7).random((160, 160)) * 255).astype(np.uint8),
    }
    print(f"    {'image':22s} {'ours':>7s} {'cv2':>7s} {'differ':>7s} {'agreement':>10s}")
    scores = {}
    for name, im in cases.items():
        mine = E.canny(im, 50, 150)
        theirs = cv2.Canny(im, 50, 150)
        differ = int(((mine > 0) != (theirs > 0)).sum())
        agree = E.agreement(mine, theirs)
        scores[name] = agree
        print(f"    {name:22s} {int((mine > 0).sum()):7d} {int((theirs > 0).sum()):7d} "
              f"{differ:7d} {agree:9.3f}%")
    print("\n    Where the remaining pixels go, and why it is a lesson rather than")
    print("    a defect: OpenCV does its magnitude arithmetic in 16-bit integers")
    print("    and we did it in float64, so wherever two neighbouring magnitudes")
    print("    are exactly equal the two implementations break the tie differently")
    print("    and the crest lands one pixel over. A ridge that is genuinely flat")
    print("    has no single crest, so both answers are correct.")
    print(f"    Pure noise is nearly all ties, which is why it is the worst row")
    print(f"    ({scores['pure noise']:.2f}%). The single smooth diagonal is the best")
    print(f"    ({scores['diagonal ramp']:.3f}%) -- one unambiguous ridge, no ties.")
    print(f"    The checkerboard sits between them ({scores['checkerboard']:.3f}%)")
    print("    despite having the sharpest edges in the set, because its right-angle")
    print("    corners are exactly where the four direction bins are ambiguous.")
    print("    Chasing the last 0.1% would mean reimplementing OpenCV's integer")
    print("    arithmetic, which teaches nothing about Canny.\n")

    print("G. The blur cv2.Canny does not do for you\n")
    clean = images.shapes_gray()
    truth = int((cv2.Canny(cv2.GaussianBlur(clean, (0, 0), 1.4), 50, 150) > 0).sum())
    print(f"    the scene has about {truth} real edge pixels\n")
    print(f"    {'noise sigma':>11s} {'no pre-blur':>12s} {'pre-blurred':>12s} {'excess':>8s}")
    for s in (0, 5, 10, 20, 30):
        noisy = images.add_gaussian_noise(clean, s, seed=1)
        raw = int((cv2.Canny(noisy, 50, 150) > 0).sum())
        pre = int((cv2.Canny(cv2.GaussianBlur(noisy, (0, 0), 1.4), 50, 150) > 0).sum())
        print(f"    {s:11d} {raw:12d} {pre:12d} {raw / max(pre, 1):7.1f}x")
    print("    Below sigma 10 the blur changes essentially nothing, which is exactly")
    print("    why a clean synthetic frame hides this lesson. At sigma 20 -- ordinary")
    print("    sensor noise on a real camera -- leaving it out multiplies the edge")
    print("    count by an order of magnitude, and nearly all of it is grain.\n")

    # -------------------------------------------------------------- the figures
    src = cv2.GaussianBlur(images.shapes_gray(), (0, 0), 1.4)
    gx, gy = E.sobel(src)
    mag = E.magnitude(gx, gy, norm="L2")
    ang = E.direction(gx, gy)
    thinned = E.non_maximum_suppression(E.magnitude(gx, gy), ang)
    mine = E.canny(src, 50, 150)
    theirs = cv2.Canny(src, 50, 150)

    fig, ax = new_figure(2, 4, width=14.0, height=6.6)
    show_gray(ax[0], src, "input (Gaussian sigma 1.4)")
    show_gray(ax[1], gx, "Gx: fires on VERTICAL edges", vmin=-500, vmax=500,
              cmap="coolwarm")
    show_gray(ax[2], gy, "Gy: fires on HORIZONTAL edges", vmin=-500, vmax=500,
              cmap="coolwarm")
    show_gray(ax[3], mag, "magnitude sqrt(Gx^2 + Gy^2)", vmax=None)

    hsv = np.zeros(src.shape + (3,), np.uint8)
    hsv[:, :, 0] = (ang % 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag / max(mag.max(), 1) * 255 * 3, 0, 255).astype(np.uint8)
    ax[4].imshow(cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))
    ax[4].set_title("gradient direction (hue), strength (value)")

    show_gray(ax[5], thinned, "after non-maximum suppression\n(one pixel wide)",
              vmax=None)
    show_gray(ax[6], mine, "our Canny, thresholds 50/150")
    diff = ((mine > 0) != (theirs > 0)).astype(np.uint8) * 255
    show_gray(ax[7], diff, f"where we differ from cv2.Canny\n"
                           f"{int((diff > 0).sum())} px, "
                           f"{E.agreement(mine, theirs):.3f}% agreement")
    save(fig, "07_edges.png")

    fig, ax = new_figure(1, 4, width=14.0, height=3.7)
    noisy = images.add_gaussian_noise(images.shapes_gray(), 20, seed=1)
    show_gray(ax[0], noisy, "input, sensor noise sigma 20")
    show_gray(ax[1], cv2.Canny(noisy, 50, 150),
              f"cv2.Canny straight on it\n"
              f"{int((cv2.Canny(noisy, 50, 150) > 0).sum())} edge pixels")
    pre = cv2.GaussianBlur(noisy, (0, 0), 1.4)
    show_gray(ax[2], cv2.Canny(pre, 50, 150),
              f"one GaussianBlur first\n"
              f"{int((cv2.Canny(pre, 50, 150) > 0).sum())} edge pixels")
    show_gray(ax[3], E.canny(noisy, 50, 150, sigma=1.4),
              "our canny(..., sigma=1.4)\nstage 1 done for you")
    save(fig, "07_canny_needs_a_blur.png")

    print("The last panel of the first figure is the honest one. It is not blank")
    print("and it is not supposed to be: it is every pixel where two defensible")
    print("implementations of the same algorithm disagreed about which of two")
    print("equal neighbours was the crest.")


if __name__ == "__main__":
    main()
