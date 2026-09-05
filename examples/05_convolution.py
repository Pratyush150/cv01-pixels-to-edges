"""05 -- Convolution from scratch: loops, vectorised, and the library.

Checks the hand-worked 2x2 answer, shows the three implementations agreeing,
prices the border modes, and demonstrates that the sum of a kernel's weights
predicts what it does to the image.

    python3 examples/05_convolution.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from pixels import convolve as K, images
from pixels.figures import new_figure, save, show_gray


def main() -> None:
    # ------------------------------------------------ A. the hand-worked case
    print("A. The 4x4 hand drill, checked against the machine\n")
    I = np.array([[10, 10, 10, 10],
                  [10, 10, 80, 80],
                  [10, 10, 80, 80],
                  [10, 10, 80, 80]], np.float64)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    valid = K.correlate2d_loops(I, Kx, padding=0)
    print("    a dark left half, a bright right half, and Sobel-x, valid padding:")
    print("   ", str(valid).replace("\n", "\n    "))
    print("    Top-left by hand:  (-1)(10)+(1)(10) = 0")
    print("                     + (-2)(10)+(2)(80) = 140")
    print("                     + (-1)(10)+(1)(80) = 70   -> 210")
    print("    The bottom row is larger (280) because those windows sit entirely")
    print("    on the step; the top row averages in a flat row that contributes 0.\n")

    print("B. Output sizes, from the formula, checked against the array\n")
    print(f"    {'in':>4} {'k':>3} {'p':>3} {'s':>3} | {'formula':>8} | actual")
    for (n, k, p, s) in [(224, 3, 1, 1), (224, 3, 1, 2), (4, 3, 0, 1), (32, 5, 2, 2)]:
        got = K.correlate2d(np.zeros((n, n)), np.ones((k, k)), padding=p, stride=s).shape
        print(f"    {n:4d} {k:3d} {p:3d} {s:3d} | {K.output_size(n, k, p, s):8d} | {got}")
    print("    3x3 with pad 1 stride 1 preserves resolution exactly, which is why")
    print("    it is the default in every modern architecture; stride 2 halves it.\n")

    # -------------------------------------- C. three implementations, one answer
    print("C. Loops == vectorised == cv2.filter2D\n")
    img = images.shapes_gray()
    print(f"    {'kernel':12s} {'loops vs vectorised':>22s} {'vectorised vs cv2':>20s}")
    for name, kern in [("box 3x3", K.BOX3), ("laplacian", K.LAPLACIAN4),
                       ("sharpen", K.SHARPEN), ("emboss", K.EMBOSS)]:
        a = K.correlate2d_loops(img, kern)
        b = K.correlate2d(img, kern)
        c = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kern,
                         borderType=cv2.BORDER_REFLECT_101)
        print(f"    {name:12s} {np.abs(a - b).max():22.3e} {np.abs(b - c).max():20.3e}")
    print("    Zero, or a few times 1e-14 -- the order NumPy happens to add nine")
    print("    floats in. `cv2.filter2D` does cross-correlation despite its name,")
    print("    so no flip is needed to make these agree.\n")

    flipped = K.convolve2d(img, K.EMBOSS)
    print(f"    true convolution vs correlation, emboss: max |difference| = "
          f"{np.abs(flipped - K.correlate2d(img, K.EMBOSS)).max():.1f}")
    print(f"    same comparison with the symmetric box kernel:               "
          f"{np.abs(K.convolve2d(img, K.BOX3) - K.correlate2d(img, K.BOX3)).max():.1e}")
    print("    That is why the flip bug is so slippery: it is invisible on every")
    print("    symmetric kernel and only appears the day you test an asymmetric one.\n")

    # Warm up once, then take the best of three. The first call to either path
    # pays for buffer allocation and, in the vectorised case, for NumPy
    # resolving the tensordot; timing that would inflate the ratio by 3x on a
    # loaded machine and make the number worthless.
    K.correlate2d_loops(img, K.BOX3)
    K.correlate2d(img, K.BOX3)
    t_loops = t_vec = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        K.correlate2d_loops(img, K.BOX3)
        t_loops = min(t_loops, time.perf_counter() - t)
        t = time.perf_counter()
        K.correlate2d(img, K.BOX3)
        t_vec = min(t_vec, time.perf_counter() - t)
    print(f"    on this {img.shape[0]}x{img.shape[1]} image: loops {t_loops * 1e3:7.1f} ms, "
          f"vectorised {t_vec * 1e3:6.2f} ms  ({t_loops / t_vec:.0f}x)")
    print("    The multiplies are identical. What the loop costs is 76,800 trips")
    print("    round the interpreter to schedule nine of them at a time.\n")

    # ------------------------------------------------------ D. the border modes
    print("D. What you fill the padding with, on a uniform image of 100\n")
    flat = np.full((5, 5), 100.0)
    print(f"    {'border':12s} {'corner':>8s} {'edge':>8s} {'centre':>8s}")
    for mode in ("constant", "reflect101", "replicate", "reflect", "wrap"):
        o = K.correlate2d(flat, K.BOX3, border=mode)
        print(f"    {mode:12s} {o[0, 0]:8.1f} {o[0, 2]:8.1f} {o[2, 2]:8.1f}")
    lib = cv2.blur(flat, (3, 3))
    print(f"    {'cv2.blur':12s} {lib[0, 0]:8.1f} {lib[0, 2]:8.1f} {lib[2, 2]:8.1f}")
    print(f"    reflect101 matches cv2.blur exactly: "
          f"{np.allclose(K.correlate2d(flat, K.BOX3), lib)}\n")
    print("    Every input pixel was 100. Blurring an image with no variation must")
    print("    return it unchanged -- and zero padding returns 44.4 at the corners,")
    print("    a 56% error, because 5 of the 9 window cells are invented black.")
    print("    (4*100 + 5*0)/9 = 44.4, and at an edge (6*100)/9 = 66.7.")
    print("    Symptom to memorise: a hand-rolled filter with a dark rim that the")
    print("    library version does not have. Check the padding first, every time.\n")

    # ------------------------------------------------- E. the kernel sum rule
    print("E. The sum of the weights is the brightness knob\n")
    f = img.astype(np.float64)
    print(f"    {'kernel':12s} {'sum':>5s} {'out mean':>10s} {'out min':>9s} {'out max':>9s}")
    print(f"    {'(original)':12s} {'-':>5s} {f.mean():10.2f} {f.min():9.1f} {f.max():9.1f}")
    for name, kern in [("box 3x3", K.BOX3), ("laplacian", K.LAPLACIAN4),
                       ("sharpen", K.SHARPEN)]:
        o = K.correlate2d(img, kern)
        print(f"    {name:12s} {kern.sum():5.0f} {o.mean():10.2f} {o.min():9.1f} "
              f"{o.max():9.1f}")
    print("    Weights summing to 1 leave the mean untouched: still a photograph.")
    print("    Weights summing to 0 pull it to zero: a map of differences on black.")
    print("    Now read the min and max columns: the Laplacian and the sharpen both")
    print("    leave 0..255 in both directions. Compute them in uint8 and every")
    print("    negative clips to 0 and every overflow to 255, silently, and you lose")
    print("    half your edge information without a warning. That is why the first")
    print("    line of `correlate2d_loops` is a cast to float.\n")

    # -------------------------------------------------------------- the figure
    fig, ax = new_figure(2, 4, width=14.0, height=6.6)
    show_gray(ax[0], img, "input: shapes_gray()")
    show_gray(ax[1], np.clip(K.correlate2d(img, K.BOX3), 0, 255),
              "box 3x3 (sum 1): blur")
    show_gray(ax[2], np.clip(K.correlate2d(img, K.SHARPEN), 0, 255),
              "centre 5 (sum 1): sharpen")
    lap = K.correlate2d(img, K.LAPLACIAN4)
    show_gray(ax[3], np.abs(lap), f"centre 4 (sum 0): edge map\nrange {lap.min():.0f} "
                                  f"to {lap.max():.0f}", vmax=None)

    big = np.ones((15, 15)) / 225.0
    z = K.correlate2d(img, big, border="constant")
    r = K.correlate2d(img, big, border="reflect101")
    show_gray(ax[4], np.clip(z, 0, 255), "15x15 blur, zero padding")
    show_gray(ax[5], np.clip(r, 0, 255), "15x15 blur, mirrored padding")
    # Autoscaled on purpose here: unlike the separable check in example 06, this
    # difference is real and its whole point is where it is, not how big.
    show_gray(ax[6], np.abs(r - z), f"the difference: a dark frame 7 px deep\n"
                                    f"max {np.abs(r - z).max():.0f} levels", vmax=None)
    show_gray(ax[7], np.clip(K.correlate2d(img, K.EMBOSS) + 128, 0, 255),
              "emboss (asymmetric):\nthe kernel where the flip shows")
    save(fig, "05_convolution.png")

    print("Panel 7 is the border bug on its own. Nothing in the scene changed; the")
    print("difference between the two blurs is a frame of exactly the kernel radius,")
    print("and it is brightest where the image was brightest -- because that is where")
    print("averaging against invented black costs the most.")


if __name__ == "__main__":
    main()
