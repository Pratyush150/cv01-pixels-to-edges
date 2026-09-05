"""Every test image this repository uses, generated from a seed.

Why generate instead of committing photographs: a teaching repo that ships a
JPEG is a teaching repo whose results you cannot reproduce if the JPEG is
resized, recompressed or replaced.  Everything here is deterministic given the
seed, so a number quoted in the README can be re-derived on any machine by
running the same function.  It also keeps the repository self-contained -- no
downloads, no licence questions about somebody else's photograph.

The images are *synthetic photographs*: gradients, grain, vignetting and
overlapping objects, chosen so that each teaching point actually shows up.
A flat colour block hides overflow, hides interpolation and hides histograms,
which is why the curriculum this repo follows insists on texture.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "gray_ramp",
    "tabletop_scene",
    "tabletop_truth",
    "shapes_gray",
    "lit_document",
    "add_gaussian_noise",
    "checkerboard",
]


# The disc palette, in OpenCV's B, G, R order.  Shared between the scene and
# its ground truth so the two can never drift apart -- if you edit a radius
# here, the truth mask moves with it.
_RED = (48.0, 52.0, 205.0)
_DISCS = [
    (80, 105, 58, _RED),                 # red, under the lamp
    (412, 268, 54, _RED),                # red, in the shadow -- same paint
    (255, 90, 52, (70.0, 165.0, 92.0)),  # green distractor
    (215, 265, 56, (190.0, 118.0, 60.0)),  # blue distractor
]


def gray_ramp(height: int = 100, width: int = 256) -> np.ndarray:
    """A black-to-white ramp: the smallest image whose pixels you can predict.

    Column c holds the value c, so `img[any_row, 200] == 200`.  The picture *is*
    the numbers, and that is the only claim this image exists to make.
    """
    # np.tile broadcasts one row down the image.  The obvious `for col in
    # range(width)` loop does the same thing and is the version most people
    # write first; it is 250x slower and teaches nothing extra.
    row = np.arange(width, dtype=np.uint8)
    return np.tile(row, (height, 1))


def _vignette(height: int, width: int, strength: float = 0.35) -> np.ndarray:
    """Radial falloff in [1-strength, 1], as float32.

    Real lenses drop light off towards the corners.  Including it here is not
    decoration: it is what makes the global threshold in example 08 fail in a
    way that looks like a photograph rather than like a bug.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    return (1.0 - strength * np.clip(r / np.sqrt(2.0), 0.0, 1.0)).astype(np.float32)


def tabletop_scene(seed: int = 0) -> np.ndarray:
    """A 360x480 BGR 'photograph': four coloured discs on a wooden surface, lit
    hard from the left so the right-hand side falls into deep shadow.

    Three things are deliberate and load-bearing for example 04:

    * **The background is wood**, which in BGR has R > G > B -- it is *reddish*.
      A red object on a grey background is a toy problem; a red object on a
      warm brown background is the reason that example needs HSV.
    * **There are two red discs**, one under the lamp and one in the shadow.
      Together they span a 3x range of illumination, which is the threshold
      past which no axis-aligned box in BGR can hold both without also holding
      lit wood.  The arithmetic for that 3x is in docs/WALKTHROUGH.md.
    * **The shading is multiplicative**, because that is what light does:
      halving the light halves all three channels together.  That single fact
      is why hue survives a shadow and raw B, G, R values do not.
    """
    rng = np.random.default_rng(seed)
    h, w = 360, 480

    # ---- the wooden surface: a warm base, a slow grain, fine sensor noise
    base = np.zeros((h, w, 3), np.float32)
    base[:, :, 0] = 78.0    # B
    base[:, :, 1] = 122.0   # G
    base[:, :, 2] = 165.0   # R  -> a tan/oak colour, and clearly R > G > B

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = 13.0 * np.sin(yy / 7.0 + 2.0 * np.sin(xx / 130.0))
    base += grain[:, :, None]
    base += rng.normal(0.0, 4.0, base.shape).astype(np.float32)

    # ---- the objects, drawn in BGR because that is what OpenCV stores.
    # LINE_AA anti-aliases the rims; a hard-edged disc makes every gradient
    # example look artificially clean and hides what NMS is actually doing.
    disc = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    for (cx, cy, radius, bgr) in _DISCS:
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (cx, cy), radius, 1.0, -1, lineType=cv2.LINE_AA)
        disc += m[:, :, None] * np.array(bgr, np.float32)
        alpha = np.maximum(alpha, m)

    img = base * (1.0 - alpha[:, :, None]) + disc

    # ---- specular highlights: the glossy spot where the lamp reflects off a
    # curved surface.  Not decoration.  A specular is where a real photograph
    # gets within a few levels of 255, and without one nothing in this scene is
    # bright enough for example 02's uint8 overflow to have anything to
    # overflow.  It is also honestly hard for colour thresholding: at a
    # specular the surface reflects the *lamp's* colour, not its own, so its
    # saturation collapses and its hue becomes meaningless.  Example 04 loses
    # those pixels and says so.
    spec = np.zeros((h, w), np.float32)
    for (cx, cy, radius, _bgr) in _DISCS:
        cv2.circle(spec, (cx - radius // 3, cy - radius // 3), max(6, radius // 6),
                   1.0, -1, lineType=cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 4.0)
    spec /= spec.max()          # the blur cost the blob its peak; put it back,
    img += (150.0 * spec)[:, :, None]   # so the brightest pixels really do reach ~255

    # ---- shading: a lamp at the left edge, plus the lens vignette. The lamp
    # brightens as well as darkens (1.35 at the left) so that the lit disc sits
    # near the top of the uint8 range -- that is where example 03's overflow
    # has something to overflow.
    lamp = np.linspace(1.35, 0.22, w, dtype=np.float32)[None, :]

    # The lamp also has a COLOUR temperature, and it changes across the frame:
    # the tungsten bulb on the left is warm, while the shadowed right is lit by
    # bounced daylight and is cooler.  This is what stops the HSV example from
    # being a rigged demo.  A purely grey illumination ramp leaves hue exactly
    # constant and hue thresholding scores a perfect 1.000, which is true of
    # this synthetic scene and not of any photograph.  With a colour shift, hue
    # drifts a little too -- it is *more* stable than B, G and R, not immune.
    tint = np.stack([
        np.linspace(0.86, 1.20, w, dtype=np.float32),   # B: cooler to the right
        np.ones(w, np.float32),                         # G: the reference
        np.linspace(1.12, 0.88, w, dtype=np.float32),   # R: warmer to the left
    ], axis=-1)[None, :, :]

    img *= lamp[:, :, None] * _vignette(h, w)[:, :, None] * tint

    return np.clip(img, 0, 255).astype(np.uint8)


def tabletop_truth(seed: int = 0) -> np.ndarray:
    """The exact pixels belonging to the two red discs in `tabletop_scene`.

    We know the answer because we drew it, which is the only reason example 04
    can quote an IoU instead of an opinion.  Thresholding the same anti-aliased
    disc at 0.5 excludes half-covered rim pixels, whose colour is a blend of
    disc and wood and therefore belongs to neither.
    """
    h, w = 360, 480
    m = np.zeros((h, w), np.float32)
    for (cx, cy, radius, bgr) in _DISCS:
        if bgr is _RED:
            cv2.circle(m, (cx, cy), radius, 1.0, -1, lineType=cv2.LINE_AA)
    return (m > 0.5).astype(np.uint8) * 255


def shapes_gray(seed: int = 0) -> np.ndarray:
    """A grayscale test card for the convolution and edge chapters.

    It carries, on purpose: straight vertical and horizontal edges (a
    rectangle), a curved edge (a disc), a 45 degree edge (a triangle), a
    3-pixel wire that opening will destroy, and faint grain so that a
    'do nothing' filter is visibly different from a blur.
    """
    rng = np.random.default_rng(seed)
    img = np.full((240, 320), 110.0, np.float32)
    img += rng.normal(0.0, 6.0, img.shape).astype(np.float32)

    cv2.rectangle(img, (40, 40), (150, 160), 215.0, -1)
    cv2.circle(img, (235, 90), 42, 40.0, -1)
    cv2.line(img, (0, 205), (319, 213), 250.0, 3)
    pts = np.array([[190, 230], [265, 155], [300, 230]], np.int32)
    cv2.fillPoly(img, [pts], 175.0)

    return np.clip(img, 0, 255).astype(np.uint8)


def lit_document(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A page of 'text' under a lamp, with the ground-truth ink mask.

    Returns `(scan, truth)`.  The lamp is a multiplicative ramp from 1.05 down
    to 0.20 across the page, which is the standard photograph-of-a-page
    failure: on the shadowed side the *paper* is darker than the ink on the lit
    side, so no single global threshold can separate them.  Example 08 measures
    exactly that.
    """
    rng = np.random.default_rng(seed)
    h, w = 240, 400
    paper = np.full((h, w), 205.0, np.float32)
    truth = np.zeros((h, w), np.uint8)

    for r in range(28, h - 24, 38):
        for c in range(18, w - 46, 88):
            # Rows of blocks stand in for lines of text.  Their width and gap
            # set the block size adaptive thresholding needs (gotcha: blockSize
            # must be a few times the stroke width, or the local mean is
            # computed over the ink itself and the ink vanishes).
            cv2.rectangle(paper, (c, r), (c + 62, r + 13), 38.0, -1)
            cv2.rectangle(truth, (c, r), (c + 62, r + 13), 255, -1)

    paper += rng.normal(0.0, 8.0, paper.shape).astype(np.float32)
    lamp = np.linspace(1.05, 0.20, w, dtype=np.float32)[None, :]
    scan = np.clip(paper * lamp, 0, 255).astype(np.uint8)
    return scan, truth


def add_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise and clip back into uint8.

    The clip is the point: without it the noise would wrap, and a bright pixel
    pushed to 258 would come back as 2 -- a black speck in a white region.
    That is the uint8 bug of module `dtypes`, arriving through the back door.
    """
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def checkerboard(size: int = 240, square: int = 30) -> np.ndarray:
    """Hard black/white squares -- the worst case for any smoothing filter and
    the easiest image on which to see what a border mode does at the rim."""
    yy, xx = np.mgrid[0:size, 0:size]
    return np.where(((yy // square) + (xx // square)) % 2 == 0, 235, 25).astype(np.uint8)
