"""Matplotlib canvas for Empirical CDF with crosshair and reference lines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtCore import pyqtSignal

from .ecdf import GroupECDF, proportions_at, stats_legend_text
from .fonts import configure_matplotlib_fonts

# Ensure CJK glyphs render on titles / legends / annotations.
configure_matplotlib_fonts()


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

# How multiple groups are arranged when a grouping column is used.
LAYOUT_OVERLAY = "overlay"  # all series on one axes
LAYOUT_GRID = "grid"  # subplot grid inside one figure
LAYOUT_MULTI = "multi"  # one axes per group (stacked / separate panels)


def grid_shape(n: int) -> Tuple[int, int]:
    """Choose a compact nrow x ncol layout for n panels."""
    if n <= 0:
        return 1, 1
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


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
        self._axes: List = [self.ax]
        self._groups: List[GroupECDF] = []
        self._ref_lines: List[ReferenceLine] = []
        self._x_limits: Optional[Tuple[float, float]] = None
        self._title = "经验累积分布函数（CDF）"
        self._xlabel = "数值"
        self._ylabel = "累积比例"
        self._layout_mode = LAYOUT_OVERLAY

        # Crosshair artists are per-axes (ax -> (vline, hline, annot))
        self._crosshairs: dict = {}
        self._crosshair_enabled = True

        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("axes_leave_event", self._on_leave)
        self._style_axes(self.ax)

    def _style_axes(self, ax, *, title: Optional[str] = None) -> None:
        ax.set_facecolor("#fafafa")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel)
        ax.set_title(title if title is not None else self._title)

    def set_layout_mode(self, mode: str, *, redraw: bool = True) -> None:
        if mode not in (LAYOUT_OVERLAY, LAYOUT_GRID, LAYOUT_MULTI):
            mode = LAYOUT_OVERLAY
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        if redraw:
            self.redraw()

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

    def _clear_figure(self) -> None:
        self.fig.clear()
        self._axes = []
        self._crosshairs.clear()
        self.ax = None

    def _make_axes(self, n_panels: int) -> List:
        """Create axes according to layout mode. n_panels is number of groups (or 1)."""
        self._clear_figure()
        mode = self._layout_mode
        n = max(n_panels, 1)

        if mode == LAYOUT_OVERLAY or n == 1:
            self.fig.set_size_inches(8, 5, forward=True)
            ax = self.fig.add_subplot(111)
            self._axes = [ax]
            self.ax = ax
            return self._axes

        if mode == LAYOUT_GRID:
            self.fig.set_size_inches(8, 5, forward=True)
            nrows, ncols = grid_shape(n)
            axes = self.fig.subplots(nrows, ncols, squeeze=False)
            flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            # Hide unused panels
            for i, ax in enumerate(flat):
                if i >= n:
                    ax.set_visible(False)
            self._axes = flat[:n]
            self.ax = self._axes[0]
            return self._axes

        # LAYOUT_MULTI: one panel per group, stacked vertically (separate charts)
        self.fig.set_size_inches(8, max(3.2 * n, 5), forward=True)
        axes = self.fig.subplots(n, 1, squeeze=False)
        self._axes = [axes[r][0] for r in range(n)]
        self.ax = self._axes[0]
        return self._axes

    def _draw_group_on_ax(self, ax, group: GroupECDF, color: str, *, with_legend: bool) -> None:
        ax.step(
            group.x,
            group.y,
            where="post",
            color=color,
            linewidth=1.8,
            label=stats_legend_text(group),
        )
        ax.plot(group.x, group.y, "o", color=color, markersize=3, alpha=0.55)
        ax.axvline(group.mean, color=color, linestyle=":", alpha=0.35, linewidth=1.0)
        if with_legend:
            legend = ax.legend(
                loc="lower right",
                fontsize=8,
                framealpha=0.92,
                fancybox=False,
                edgecolor="#cccccc",
            )
            legend.get_frame().set_linewidth(0.8)

    def _draw_reference_lines(self, ax, *, include_in_legend: bool) -> None:
        for ref in self._ref_lines:
            label = ref.label or (
                f"x = {ref.value:g}" if ref.orientation == "vertical" else f"p = {ref.value:g}"
            )
            legend_label = f"参考线: {label}" if include_in_legend else None
            if ref.orientation == "vertical":
                ax.axvline(
                    ref.value,
                    color=ref.color,
                    linestyle=ref.linestyle,
                    linewidth=1.5,
                    label=legend_label,
                )
            else:
                ax.axhline(
                    ref.value,
                    color=ref.color,
                    linestyle=ref.linestyle,
                    linewidth=1.5,
                    label=legend_label,
                )

    def redraw(self) -> None:
        groups = self._groups
        n = len(groups)
        mode = self._layout_mode

        # Single empty axes when nothing to plot
        if n == 0:
            self._make_axes(1)
            self._style_axes(self._axes[0])
            self._draw_reference_lines(self._axes[0], include_in_legend=bool(self._ref_lines))
            if self._x_limits is not None:
                self._axes[0].set_xlim(*self._x_limits)
            if self._ref_lines:
                legend = self._axes[0].legend(
                    loc="lower right",
                    fontsize=8,
                    framealpha=0.92,
                    fancybox=False,
                    edgecolor="#cccccc",
                )
                legend.get_frame().set_linewidth(0.8)
            self.fig.tight_layout()
            self.draw_idle()
            return

        if mode == LAYOUT_OVERLAY or n == 1:
            self._make_axes(1)
            ax = self._axes[0]
            self._style_axes(ax)
            for i, group in enumerate(groups):
                color = COLORS[i % len(COLORS)]
                self._draw_group_on_ax(ax, group, color, with_legend=False)
            self._draw_reference_lines(ax, include_in_legend=True)
            if self._x_limits is not None:
                ax.set_xlim(*self._x_limits)
            legend = ax.legend(
                loc="lower right",
                fontsize=8,
                framealpha=0.92,
                fancybox=False,
                edgecolor="#cccccc",
            )
            legend.get_frame().set_linewidth(0.8)
        else:
            # Grid or multi: one group per axes
            self._make_axes(n)
            for i, (ax, group) in enumerate(zip(self._axes, groups)):
                color = COLORS[i % len(COLORS)]
                panel_title = group.name if mode == LAYOUT_MULTI else f"{self._title} — {group.name}"
                if mode == LAYOUT_GRID:
                    panel_title = group.name
                self._style_axes(ax, title=panel_title)
                self._draw_group_on_ax(ax, group, color, with_legend=True)
                self._draw_reference_lines(ax, include_in_legend=False)
                if self._x_limits is not None:
                    ax.set_xlim(*self._x_limits)
            # Shared figure title for grid mode
            if mode == LAYOUT_GRID:
                self.fig.suptitle(self._title, fontsize=12, y=1.02)

        try:
            self.fig.tight_layout()
        except Exception:  # noqa: BLE001 — layout can fail with many panels
            pass
        self.draw_idle()

    def _ensure_crosshair(self, ax) -> Tuple[Line2D, Line2D, object]:
        if ax in self._crosshairs:
            return self._crosshairs[ax]
        vline = ax.axvline(0, color="#333333", linewidth=0.9, alpha=0.75, visible=False, zorder=20)
        hline = ax.axhline(0, color="#333333", linewidth=0.9, alpha=0.75, visible=False, zorder=20)
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fffef5", ec="#888888", alpha=0.95),
            fontsize=8,
            visible=False,
            zorder=30,
        )
        self._crosshairs[ax] = (vline, hline, annot)
        return vline, hline, annot

    def _hide_crosshair(self) -> None:
        for vline, hline, annot in self._crosshairs.values():
            vline.set_visible(False)
            hline.set_visible(False)
            annot.set_visible(False)
        self.cursor_info.emit("")

    def _on_leave(self, _event) -> None:
        self._hide_crosshair()
        self.draw_idle()

    def _groups_for_axes(self, ax) -> List[GroupECDF]:
        """Groups shown on a given axes (all in overlay; one in panel modes)."""
        if not self._groups:
            return []
        if self._layout_mode == LAYOUT_OVERLAY or len(self._groups) == 1:
            return self._groups
        try:
            idx = self._axes.index(ax)
        except ValueError:
            return self._groups
        if 0 <= idx < len(self._groups):
            return [self._groups[idx]]
        return []

    def _on_motion(self, event) -> None:
        if not self._crosshair_enabled or event.inaxes is None:
            return
        if event.inaxes not in self._axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes
        x, y = float(event.xdata), float(event.ydata)

        # Hide crosshair on other axes
        for other_ax, (vline, hline, annot) in self._crosshairs.items():
            if other_ax is not ax:
                vline.set_visible(False)
                hline.set_visible(False)
                annot.set_visible(False)

        vline, hline, annot = self._ensure_crosshair(ax)
        vline.set_xdata([x, x])
        hline.set_ydata([y, y])
        vline.set_visible(True)
        hline.set_visible(True)

        lines = [f"x = {x:.6g}", f"y = {y:.4f}"]
        groups = self._groups_for_axes(ax)
        if groups:
            props = proportions_at(groups, x)
            for name, p in props.items():
                lines.append(f"{name}: F(x) = {p:.4f}（{p * 100:.2f}%）")

        text = "\n".join(lines)
        annot.xy = (x, y)
        annot.set_text(text)
        annot.set_visible(True)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_mid = (xlim[0] + xlim[1]) / 2
        y_mid = (ylim[0] + ylim[1]) / 2
        offset_x = -110 if x > x_mid else 12
        offset_y = -40 if y > y_mid else 12
        annot.set_position((offset_x, offset_y))

        self.cursor_info.emit(text.replace("\n", "  |  "))
        self.draw_idle()
