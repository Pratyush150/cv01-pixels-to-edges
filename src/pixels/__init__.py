"""pixels -- from raw arrays to a hand-written Canny, one testable step at a time.

Import order below is also the order the examples run and the order the
README teaches, which is not a coincidence: each module only uses the ones
above it.  `edges` builds on `separable`, which builds on `convolve`, which
builds on nothing but NumPy.  If that ever stops being true the dependency has
gone the wrong way.

`figures` is deliberately NOT imported here.  It pulls in matplotlib and calls
`matplotlib.use("Agg")` at import time, and a library has no business either
requiring a heavy optional dependency or silently choosing a plotting backend
for the program that imported it.  The examples import it explicitly, because
the examples are the things that draw.
"""

from . import (  # noqa: F401
    images,
    dtypes,
    photometry,
    colour,
    convolve,
    separable,
    edges,
    threshold,
)

__version__ = "0.1.0"

__all__ = [
    "images",
    "dtypes",
    "photometry",
    "colour",
    "convolve",
    "separable",
    "edges",
    "threshold",
    "__version__",
]
