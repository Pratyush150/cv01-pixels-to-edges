# Decisions

One record per real choice: what was decided, what else was on the table, why
this one, and what it costs. If a decision here has no cost listed, it has not
been thought about hard enough.

---

## 1. Cross-correlation is the primitive; convolution is the thin wrapper

**Decision.** `correlate2d` is the function everything else calls.
`convolve2d` flips the kernel and delegates to it.

**Alternatives.** Make true convolution the primitive and correlation the
special case, which is the order a signal-processing textbook uses.

**Why.** Everything a reader will meet in practice does correlation:
`cv2.filter2D`, `scipy.ndimage.correlate`, and every convolutional layer in
every deep-learning framework. Making the mathematically-named operation the
default would mean our Sobel disagreed in sign with OpenCV's, and the reader
would spend their attention on our naming rather than on the sliding window.

**Cost.** The name `convolve.py` is slightly a lie about what its main function
does, and someone arriving from a DSP background has to read one paragraph to
find out. That paragraph is the module docstring.

---

## 2. Three implementations of convolution are kept, not one

**Decision.** `correlate2d_loops` (explicit Python loops), `correlate2d`
(vectorised), and `cv2.filter2D` all survive, and the tests assert all three
agree.

**Alternatives.** Ship only the vectorised one and mention that a loop version
exists; or ship only the loop version because it is "the teaching one".

**Why.** The loop version is where the index bookkeeping is visible -- `r =
i*stride`, the slice `r:r+K`, the padding entering exactly once -- and every one
of those is a place people get it wrong under pressure. The vectorised version
is what the rest of the package can afford to call. And the agreement between
them, asserted in a test, is what turns "I understand convolution" from a claim
into a check. Deleting any one of the three removes something.

**Cost.** Roughly 40 extra lines and a slower test suite. The loop version is
about 30x slower on a 240x320 image, so it is never used internally.

---

## 3. float64 through the pipeline, uint8 only at the boundary

**Decision.** Every filter casts to float on its first line and returns float.
Conversion back to `uint8` happens at display or save time, explicitly, with a
`np.clip`.

**Alternatives.** Stay in `uint8` and rely on OpenCV's saturating arithmetic;
or use `float32`, which is half the memory.

**Why.** An edge kernel produces negative numbers and numbers above 255. On our
own test image the Laplacian output runs from -333 to +297 and the sharpen from
-235 to +547 (`examples/05_convolution.py` prints both). In `uint8` the
negatives clip to 0 and the overflows to 255 -- silently, with no exception --
and what you lose is always the same half of your edges: the bright-to-dark
transitions. Saturating arithmetic fixes the wrap but not the clipping, because
the information genuinely does not fit.

`float64` over `float32` because these are teaching materials where an exact
comparison against OpenCV is the point, and `cv2.CV_64F` is what makes the
`np.array_equal` assertions in `tests/test_edges.py` pass rather than merely
`allclose`.

**Cost.** Four bytes per pixel becomes eight. A 1080p colour frame is 6.2 MB as
`uint8` and 49.8 MB as `float64`. For a real-time pipeline that is the wrong
trade and `float32` (or fixed-point) is the right one; this repo is optimised
for being checkable, not for throughput.

---

## 4. `BORDER_REFLECT_101` is the default border mode

**Decision.** `pad(..., border="reflect101")` unless asked otherwise, which is
`np.pad`'s `mode="reflect"`.

**Alternatives.** Zero padding, which is what almost every from-scratch
implementation does because `np.pad`'s default is `constant`.

**Why.** Zero means black, and black is a claim about the scene that is not
true. Take an image where every pixel is 100 and blur it 3x3 with zero padding: the
corners come back as 44.4 and the edges as 66.7. That is a 56% error on a
picture with no variation in it for a blur to act on. Mirroring returns 100 everywhere, and matches
what every OpenCV filter does by default, which is why our output can be
compared against theirs at all.

**Cost.** Mirroring is the wrong model for a genuinely periodic signal, and it
is the wrong model for a CNN feature map, where "outside" really does have no
activation and zero is the honest answer. Both other modes are available; the
default just reflects which case comes up more often on photographs.

---

## 5. Every test image is generated, none is committed

**Decision.** `pixels.images` builds every picture this repo uses from a seed.
No JPEGs, no PNGs of photographs, no downloads.

**Alternatives.** Commit a handful of real photographs, or fetch a standard
test image at run time.

**Why.** A number quoted in this README has to be re-derivable. A committed
JPEG can be resized, recompressed or replaced and the number silently stops
matching; a fetched image can 404 or change. Generation also means every
teaching point can be *arranged*: the scene has a specular highlight because
without one nothing is bright enough for the uint8 overflow to overflow, and it
has two red discs three illumination stops apart because that is the threshold
past which no axis-aligned BGR box can hold both (the arithmetic is in
[WALKTHROUGH.md](WALKTHROUGH.md)). And the ground-truth masks exist at all only
because we drew the objects, which is what lets example 04 and example 08 quote
an IoU instead of an opinion.

**Cost.** This is the real one, and it is not small. **A synthetic scene is
easier than a photograph**, and the hue-thresholding result in example 04 --
IoU 0.999 -- is higher than any real image would give, because the discs have
exactly one reflectance and the only thing changing across the frame is the
light. The example says so in its own output. What transfers is the *ranking*
and its mechanism, not the digit. See "Limitations" in the README.

---

## 6. Otsu is computed from the between-class curve, and ties break low

**Decision.** `otsu_threshold` maximises `sigma_B^2 = w0*w1*(mu0 - mu1)^2` over
a 256-entry curve built from cumulative sums of the histogram, and returns
`np.argmax`, which is the *first* maximum.

**Alternatives.** Minimise the within-class variance directly, which is the
form the method is usually explained in; or break ties at the midpoint or the
top of the tie range.

**Why.** Both forms give the same answer, because `sigma_total^2 = sigma_W^2 +
sigma_B^2` and the total does not depend on the threshold -- an identity
`tests/test_threshold.py` checks at 37 thresholds on 8 images. The
between-class form needs only running sums, so it is one vectorised pass
instead of 256 passes over the pixels. `within_class_variance` is kept anyway,
because it is the definition and the worked example scores it directly.

The tie-break is the part that looks like pedantry and is not. On the classic
worked example, every threshold from 50 to 189 produces the identical split and
therefore the identical score. OpenCV returns 50. *Every* histogram Otsu is
good at has a wide empty valley, so a different tie-break disagrees with OpenCV
on essentially every image the method is meant for -- and the test that asserts
we match it would fail on all of them.

**Cost.** Nothing measurable. The curve is 256 floats.

---

## 7. The separability benchmark times shift-accumulation against shift-accumulation

**Decision.** `separable.correlate_shift_2d` and
`correlate_shift_separable` both work by accumulating whole shifted copies of
the image. Neither is the fastest way to write the operation.

**Alternatives.** Time a naive Python double loop against `cv2.sepFilter2D`,
which is what most speed comparisons do.

**Why.** That comparison would be dishonest in a specific way: it measures
Python interpreter overhead against optimised C and reports the result as an
algorithmic win. Here the two functions use identical machinery and differ only
in how many multiply-adds happen per pixel -- `K*K` against `2*K` -- so the
ratio between their run times is the ratio of their arithmetic.

**Cost.** Neither function is a good general-purpose convolution, and the
absolute times are much slower than the library. `correlate2d` is what the rest
of the package uses. The benchmark is a measuring instrument, not a filter.

---

## 8. Our Canny does not blur by default

**Decision.** `edges.canny(gray, low, high)` runs stages 2 to 5. Stage 1, the
Gaussian, happens only if you pass `sigma=`.

**Alternatives.** Blur by default, which is what the algorithm as published
does and what a reader expects.

**Why.** This function exists to be compared against `cv2.Canny`, pixel for
pixel, and `cv2.Canny` does not blur. Making our default differ from OpenCV's
would mean the comparison could never be exact, and the exactness is what makes
the implementation credible.

**Cost.** The default is the wrong one for actual use, and a reader who calls
it on a noisy frame gets a speckled mess. The docstring says so, and
`examples/07_edges.py` measures the damage: at a sensor noise of sigma 20,
skipping the blur multiplies the edge count by about 17. The mitigation is that
`sigma=` exists and is one keyword away.

---

## 9. The Canny test asserts an agreement floor, not equality

**Decision.** `tests/test_edges.py` requires >= 99.5% agreement with
`cv2.Canny` on clean images and >= 97.5% on pure noise, rather than
`np.array_equal`.

**Alternatives.** Demand bit-exactness, and reimplement OpenCV's integer
arithmetic until we get it.

**Why.** OpenCV computes gradient magnitudes in 16-bit integers; we compute
them in float64. Wherever two neighbouring magnitudes are exactly equal, the
two implementations break the tie differently and the crest lands one pixel
over. Both answers are correct -- a ridge that is genuinely flat has no single
crest. Chasing that last fraction of a percent would mean porting OpenCV's
fixed-point path, which teaches nothing about Canny and makes the code worse.

The disagreement is also a result worth reporting rather than hiding: it is
highest on pure noise (98.27%), where nearly every pair of neighbours is a
near-tie, and lowest on a single smooth diagonal (99.99%), where there are no
ties at all. That ordering is the lesson.

**Cost.** The test cannot catch a bug that moves fewer than 0.5% of the edge
pixels. Mitigated by the other tests in that file, which pin the NMS output,
the hysteresis output and the Sobel values exactly.

---

## 10. `pixels.figures` is not imported by `pixels/__init__.py`

**Decision.** The package imports `images`, `dtypes`, `photometry`, `colour`,
`convolve`, `separable`, `edges`, `threshold` -- and not `figures`.

**Alternatives.** Import everything, for convenience.

**Why.** `figures` imports matplotlib and calls `matplotlib.use("Agg")` at
import time. Neither belongs in a library import: matplotlib is a heavy
optional dependency, and selecting a backend is a global side effect that a
library has no business imposing on the program that imported it. The examples
import it explicitly because they are the things that draw.

**Cost.** `from pixels import figures` is one extra line in each example.

---

## 11. The colour-space comparison is an exhaustive search, not two tuned rules

**Decision.** `colour.best_bgr_rule` and `colour.best_hue_rule` sweep every
threshold in their respective three-parameter families and report the best IoU
either can reach.

**Alternatives.** Hand-tune one rule per colour space and compare those.

**Why.** A hand-tuned comparison proves nothing about the colour spaces; it
proves something about the person who tuned them. Quoting "BGR reached 0.575"
should mean *no rule of that shape does better*, and only a sweep supports
that. Both families get three free parameters and the same objective, so the
difference in scores is a property of the space.

The searches use cumulative histograms rather than nested loops: quantise,
histogram once, then take a running sum along each axis in the direction that
bound moves, and every candidate becomes a single lookup. The naive version
takes over a minute; this takes under half a second. The quantisation is exact
rather than approximate -- with `q = v // step`, `v >= i*step` is exactly
`q >= i` -- so this is a coarse grid evaluated perfectly, not a fine grid
evaluated sloppily.

**Cost.** The families are still restricted: three parameters, axis-aligned in
BGR, a wedge in hue. A general six-parameter BGR box, or a linear discriminant,
would score higher than 0.575. The claim is bounded accordingly -- it is about
this family of rules, which is the family a person actually writes.

---

## 12. Hysteresis uses an explicit queue; NMS is vectorised over four bins

**Decision.** Non-maximum suppression runs four whole-array passes, one per
gradient bin. Hysteresis runs a breadth-first flood with a `collections.deque`.

**Alternatives.** Write both as per-pixel Python loops (clearer, unusably
slow), or both vectorised (NMS fine, hysteresis awkward).

**Why.** They are different shapes of problem. NMS asks the same question of
every pixel independently, so it vectorises perfectly: four passes replace
`H*W` interpreter round trips. Hysteresis is a connectivity problem -- whether
to keep a pixel depends on what has already been kept -- so it is inherently
sequential, and the honest implementation is a flood fill.

A queue rather than recursion because a single contour in a 4K frame can be
tens of thousands of pixels long and Python's recursion limit is 1000.

**Cost.** The two functions read at different levels of abstraction, and the
NMS one takes a moment to see as "the same comparison, four times". A comment
covers it. Hysteresis is the slowest part of `canny` on a noisy image, because
that is where the weak-pixel set is large.
