"""Matplotlib helpers, so the eight examples look like one document.

Two conventions, both deliberate:

* **Light background.**  These are teaching figures.  They get pasted into a
  README, printed, and looked at on a projector in a bright room.
* **`Agg` backend, chosen at import.**  Every example must run headless -- in
  CI, over SSH, in a container with no display -- and the default backend
  raises or hangs there.  Selecting it here rather than in each example means
  it cannot be forgotten in one of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

__all__ = ["FIGURE_DIR", "new_figure", "show_gray", "show_bgr", "save"]

# docs/figures relative to the repository root, resolved from this file's own
# location so an example works from any working directory.  Overridable for
# tests, which must never write into the committed figure set.
FIGURE_DIR = Path(os.environ.get(
    "PIXELS_FIGURE_DIR",
    Path(__file__).resolve().parents[2] / "docs" / "figures",
))

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
})


def new_figure(rows: int, cols: int, width: float = 11.0, height: float | None = None):
    """A grid of axes with the ticks already gone.

    Pixel indices on the axis of an image plot are noise nine times out of ten:
    the reader is looking at the picture, not measuring it.  The one example
    that genuinely needs coordinates (the pixel-grid inset in example 01) turns
    them back on locally.
    """
    if height is None:
        height = 3.1 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(width, height))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def show_gray(ax, img, title: str, vmin=0, vmax=255, cmap="gray"):
    """Draw a single-channel image with an EXPLICIT value range.

    Leaving vmin/vmax to matplotlib autoscales every panel independently, so a
    nearly-black gradient image and a nearly-white one look identical and the
    figure quietly contradicts the text beside it.  Pass `vmin=None` on purpose
    when autoscaling is what you want -- for a signed gradient, say.
    """
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    return ax


def show_bgr(ax, img, title: str):
    """Draw an OpenCV BGR image in matplotlib, which expects RGB.

    This one call is the boundary conversion that Gotcha #1 of every OpenCV
    tutorial is about.  Skip it and reds and blues swap: no error, no warning,
    every face turns blue, because the array was perfectly valid all along.
    """
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax.set_title(title)
    return ax


def save(fig, name: str) -> Path:
    """Write into docs/figures and print the path, so the example logs it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path
