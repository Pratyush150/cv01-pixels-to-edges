"""The figure helpers: headless, light-background, and writing where we say."""

import matplotlib
import numpy as np

from pixels import figures, images


def test_the_backend_is_headless():
    """Every example must run in CI and over SSH, where there is no display."""
    assert matplotlib.get_backend().lower() == "agg"


def test_figures_are_light_not_dark():
    """These are teaching materials, printed and projected, not portfolio art."""
    import matplotlib.pyplot as plt
    assert plt.rcParams["savefig.facecolor"] == "white"
    assert plt.rcParams["figure.facecolor"] == "white"


def test_save_writes_a_png_where_it_was_told(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIGURE_DIR", tmp_path)
    fig, ax = figures.new_figure(1, 2, width=4.0, height=2.0)
    figures.show_gray(ax[0], images.gray_ramp(), "ramp")
    figures.show_bgr(ax[1], images.tabletop_scene(), "scene")
    path = figures.save(fig, "probe.png")
    assert path.exists() and path.suffix == ".png"
    assert path.stat().st_size > 1000


def test_show_gray_pins_the_value_range_by_default(tmp_path, monkeypatch):
    """Autoscaling every panel independently makes a dark image and a bright one
    look identical, so a figure can quietly contradict the text beside it."""
    monkeypatch.setattr(figures, "FIGURE_DIR", tmp_path)
    fig, ax = figures.new_figure(1, 1, width=2.0, height=2.0)
    im = figures.show_gray(ax[0], np.full((8, 8), 40, np.uint8), "dark").images[0]
    assert im.get_clim() == (0, 255)
