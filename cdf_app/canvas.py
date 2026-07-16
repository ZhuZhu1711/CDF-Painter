"""Matplotlib canvas for Empirical CDF with crosshair and reference lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtCore import pyqtSignal

from .ecdf import GroupECDF, proportions_at, stats_legend_text


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


@dataclass
class ReferenceLine:
    """User-defined reference line on the CDF plot."""

    orientation: str  # "vertical" | "horizontal"
    value: float
    label: str = ""
    color: str = "#555555"
    linestyle: str = "--"


class CDFCanvas(FigureCanvas):
    """Interactive Empirical CDF plot."""

    cursor_info = pyqtSignal(str)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._groups: List[GroupECDF] = []
        self._ref_lines: List[ReferenceLine] = []
        self._x_limits: Optional[Tuple[float, float]] = None
        self._title = "经验累积分布函数（CDF）"
        self._xlabel = "数值"
        self._ylabel = "累积比例"

        self._vline: Optional[Line2D] = None
        self._hline: Optional[Line2D] = None
        self._annot = None
        self._crosshair_enabled = True

        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("axes_leave_event", self._on_leave)
        self._style_axes()

    def _style_axes(self) -> None:
        self.ax.set_facecolor("#fafafa")
        self.ax.grid(True, which="both", linestyle=":", alpha=0.6)
        self.ax.set_ylim(0.0, 1.05)
        self.ax.set_ylabel(self._ylabel)
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_title(self._title)

    def set_crosshair_enabled(self, enabled: bool) -> None:
        self._crosshair_enabled = enabled
        if not enabled:
            self._hide_crosshair()
            self.draw_idle()

    def set_axis_labels(self, xlabel: str, ylabel: str, title: str) -> None:
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._title = title

    def set_x_limits(self, xmin: Optional[float], xmax: Optional[float], *, redraw: bool = True) -> None:
        if xmin is None or xmax is None or xmin >= xmax:
            self._x_limits = None
        else:
            self._x_limits = (float(xmin), float(xmax))
        if redraw:
            self.redraw()

    def set_reference_lines(self, lines: Sequence[ReferenceLine], *, redraw: bool = True) -> None:
        self._ref_lines = list(lines)
        if redraw:
            self.redraw()

    def plot_groups(self, groups: List[GroupECDF]) -> None:
        self._groups = list(groups)
        self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self._style_axes()
        self._vline = None
        self._hline = None
        self._annot = None

        for i, group in enumerate(self._groups):
            color = COLORS[i % len(COLORS)]
            # Extend step to show F=0 before first point and F=1 after last
            x_plot = group.x
            y_plot = group.y
            self.ax.step(
                x_plot,
                y_plot,
                where="post",
                color=color,
                linewidth=1.8,
                label=stats_legend_text(group),
            )
            self.ax.plot(group.x, group.y, "o", color=color, markersize=3, alpha=0.55)
            self.ax.axvline(group.mean, color=color, linestyle=":", alpha=0.35, linewidth=1.0)

        for ref in self._ref_lines:
            label = ref.label or (
                f"x = {ref.value:g}" if ref.orientation == "vertical" else f"p = {ref.value:g}"
            )
            if ref.orientation == "vertical":
                self.ax.axvline(
                    ref.value,
                    color=ref.color,
                    linestyle=ref.linestyle,
                    linewidth=1.5,
                    label=f"参考线: {label}",
                )
            else:
                self.ax.axhline(
                    ref.value,
                    color=ref.color,
                    linestyle=ref.linestyle,
                    linewidth=1.5,
                    label=f"参考线: {label}",
                )

        if self._x_limits is not None:
            self.ax.set_xlim(*self._x_limits)

        if self._groups or self._ref_lines:
            legend = self.ax.legend(
                loc="lower right",
                fontsize=8,
                framealpha=0.92,
                fancybox=False,
                edgecolor="#cccccc",
            )
            legend.get_frame().set_linewidth(0.8)

        self.draw_idle()

    def _ensure_crosshair(self) -> None:
        if self._vline is None:
            self._vline = self.ax.axvline(
                0, color="#333333", linewidth=0.9, alpha=0.75, visible=False, zorder=20
            )
        if self._hline is None:
            self._hline = self.ax.axhline(
                0, color="#333333", linewidth=0.9, alpha=0.75, visible=False, zorder=20
            )
        if self._annot is None:
            self._annot = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fffef5", ec="#888888", alpha=0.95),
                fontsize=8,
                visible=False,
                zorder=30,
            )

    def _hide_crosshair(self) -> None:
        if self._vline is not None:
            self._vline.set_visible(False)
        if self._hline is not None:
            self._hline.set_visible(False)
        if self._annot is not None:
            self._annot.set_visible(False)
        self.cursor_info.emit("")

    def _on_leave(self, _event) -> None:
        self._hide_crosshair()
        self.draw_idle()

    def _on_motion(self, event) -> None:
        if not self._crosshair_enabled or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        self._ensure_crosshair()
        assert self._vline is not None and self._hline is not None and self._annot is not None

        self._vline.set_xdata([x, x])
        self._hline.set_ydata([y, y])
        self._vline.set_visible(True)
        self._hline.set_visible(True)

        lines = [f"x = {x:.6g}", f"y = {y:.4f}"]
        if self._groups:
            props = proportions_at(self._groups, x)
            for name, p in props.items():
                lines.append(f"{name}: F(x) = {p:.4f}（{p * 100:.2f}%）")

        text = "\n".join(lines)
        self._annot.xy = (x, y)
        self._annot.set_text(text)
        self._annot.set_visible(True)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_mid = (xlim[0] + xlim[1]) / 2
        y_mid = (ylim[0] + ylim[1]) / 2
        offset_x = -110 if x > x_mid else 12
        offset_y = -40 if y > y_mid else 12
        self._annot.set_position((offset_x, offset_y))

        self.cursor_info.emit(text.replace("\n", "  |  "))
        self.draw_idle()
