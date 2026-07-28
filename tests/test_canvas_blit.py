"""Smoke tests for CDF canvas hover blitting (offscreen Qt)."""

from __future__ import annotations

import os
import sys
import unittest

# Must set before QApplication / QtAgg import path is exercised.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import numpy as np

from cdf_app.canvas import CDFCanvas
from cdf_app.ecdf import compute_ecdf


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


class CanvasBlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_motion_uses_cached_background_without_full_redraw(self):
        canvas = CDFCanvas()
        canvas.show()
        rng = np.random.default_rng(0)
        group = compute_ecdf(rng.normal(size=50_000), name="big")
        self.assertIsNotNone(group)
        assert group is not None
        canvas.plot_groups([group])
        # Force a synchronous draw so draw_event caches the background.
        canvas.draw()
        self.app.processEvents()
        self.assertIsNotNone(canvas._background)

        class Ev:
            inaxes = canvas.ax
            xdata = 0.0
            ydata = 0.5

        # Replace draw_idle to ensure motion path does not fall back to it.
        calls = {"n": 0}
        original = canvas.draw_idle

        def counting_draw_idle(*_a, **_k):
            calls["n"] += 1
            return original()

        canvas.draw_idle = counting_draw_idle  # type: ignore[method-assign]
        canvas._last_motion_ts = 0.0
        canvas._on_motion(Ev())
        self.assertEqual(calls["n"], 0)
        vline, hline, annot = canvas._crosshairs[canvas.ax]
        self.assertTrue(vline.get_visible())
        self.assertTrue(hline.get_visible())
        self.assertTrue(annot.get_visible())

        canvas._on_leave(None)
        self.assertFalse(vline.get_visible())
        self.assertEqual(calls["n"], 0)

    def test_large_series_skips_markers(self):
        canvas = CDFCanvas()
        rng = np.random.default_rng(1)
        group = compute_ecdf(rng.normal(size=5_000), name="big")
        assert group is not None
        canvas.plot_groups([group])
        canvas.draw()
        # Only the step line (+ mean vline), no PathCollection of markers.
        ax = canvas.ax
        # Line2D artists: step + mean guide; scatter would add another Line2D with marker.
        marker_lines = [
            line
            for line in ax.lines
            if line.get_marker() not in (None, "None", "") and line.get_marker() != ","
        ]
        # Mean guide is a plain line; markers use 'o'. Ensure no 'o' markers were added.
        o_markers = [line for line in marker_lines if line.get_marker() == "o"]
        self.assertEqual(o_markers, [])


if __name__ == "__main__":
    unittest.main()
