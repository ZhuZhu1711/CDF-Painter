"""Main application window for CDF plotting."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import (
    CDFCanvas,
    LAYOUT_GRID,
    LAYOUT_MULTI,
    LAYOUT_OVERLAY,
    ReferenceLine,
)
from .ecdf import compute_grouped_ecdfs, data_x_range


class ReferenceLineDialog(QDialog):
    """Dialog to add a vertical or horizontal reference line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加参考线")
        self.setMinimumWidth(320)

        self.orientation = QComboBox()
        self.orientation.addItem("垂直线（数值 x）", "vertical")
        self.orientation.addItem("水平线（比例）", "horizontal")

        self.value_edit = QLineEdit()
        self.value_edit.setValidator(QDoubleValidator())
        self.value_edit.setPlaceholderText("例如 10 或 0.95")

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("可选标签")

        self.color = QComboBox()
        for name, hex_color in [
            ("灰色", "#555555"),
            ("红色", "#c0392b"),
            ("蓝色", "#2980b9"),
            ("绿色", "#27ae60"),
            ("橙色", "#e67e22"),
        ]:
            self.color.addItem(name, hex_color)

        form = QFormLayout()
        form.addRow("类型:", self.orientation)
        form.addRow("数值:", self.value_edit)
        form.addRow("标签:", self.label_edit)
        form.addRow("颜色:", self.color)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def get_line(self) -> Optional[ReferenceLine]:
        text = self.value_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        orientation = self.orientation.currentData()
        if orientation == "horizontal" and not (0.0 <= value <= 1.0):
            QMessageBox.warning(self, "数值无效", "水平参考线的比例必须在 [0, 1] 范围内。")
            return None
        return ReferenceLine(
            orientation=orientation,
            value=value,
            label=self.label_edit.text().strip(),
            color=self.color.currentData(),
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CDF 图 — 经验累积分布")
        self.resize(1180, 720)

        self.df: Optional[pd.DataFrame] = None
        self.ref_lines: List[ReferenceLine] = []
        self._updating_limits = False

        self._build_ui()
        self._connect_signals()
        self._set_controls_enabled(False)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left control panel ----
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(360)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)

        file_box = QGroupBox("数据")
        file_layout = QVBoxLayout(file_box)
        self.btn_load = QPushButton("加载 CSV / Excel…")
        self.lbl_file = QLabel("未加载文件")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #666;")
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.lbl_file)
        panel_layout.addWidget(file_box)

        col_box = QGroupBox("列设置")
        col_form = QFormLayout(col_box)
        self.cmb_value = QComboBox()
        self.cmb_group = QComboBox()
        col_form.addRow("数值列:", self.cmb_value)
        col_form.addRow("分组列:", self.cmb_group)
        panel_layout.addWidget(col_box)

        axis_box = QGroupBox("横坐标范围")
        axis_form = QFormLayout(axis_box)
        self.edit_xmin = QLineEdit()
        self.edit_xmax = QLineEdit()
        for w in (self.edit_xmin, self.edit_xmax):
            w.setValidator(QDoubleValidator())
        self.chk_auto_x = QCheckBox("自动（根据数据）")
        self.chk_auto_x.setChecked(True)
        self.btn_apply_x = QPushButton("应用横坐标范围")
        axis_form.addRow("X 最小:", self.edit_xmin)
        axis_form.addRow("X 最大:", self.edit_xmax)
        axis_form.addRow(self.chk_auto_x)
        axis_form.addRow(self.btn_apply_x)
        panel_layout.addWidget(axis_box)

        ref_box = QGroupBox("参考线")
        ref_layout = QVBoxLayout(ref_box)
        self.list_refs = QListWidget()
        self.list_refs.setMaximumHeight(120)
        ref_btns = QHBoxLayout()
        self.btn_add_ref = QPushButton("添加…")
        self.btn_del_ref = QPushButton("删除")
        self.btn_clear_refs = QPushButton("清空")
        ref_btns.addWidget(self.btn_add_ref)
        ref_btns.addWidget(self.btn_del_ref)
        ref_btns.addWidget(self.btn_clear_refs)
        ref_layout.addWidget(self.list_refs)
        ref_layout.addLayout(ref_btns)
        panel_layout.addWidget(ref_box)

        opt_box = QGroupBox("显示")
        opt_layout = QVBoxLayout(opt_box)
        self.chk_crosshair = QCheckBox("十字定位线（数值 + 比例）")
        self.chk_crosshair.setChecked(True)

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("多组布局:"))
        self.cmb_layout = QComboBox()
        self.cmb_layout.addItem("叠加同一图", LAYOUT_OVERLAY)
        self.cmb_layout.addItem("网格分布（一张图）", LAYOUT_GRID)
        self.cmb_layout.addItem("多图显示（每组一图）", LAYOUT_MULTI)
        self.cmb_layout.setToolTip(
            "分组后有多组数据时：\n"
            "· 叠加同一图：所有组画在同一坐标系\n"
            "· 网格分布：在一张图内按网格分面\n"
            "· 多图显示：每组单独一张图（纵向排列）"
        )
        layout_row.addWidget(self.cmb_layout, 1)

        self.btn_plot = QPushButton("更新图形")
        self.btn_plot.setDefault(True)
        opt_layout.addWidget(self.chk_crosshair)
        opt_layout.addLayout(layout_row)
        opt_layout.addWidget(self.btn_plot)
        panel_layout.addWidget(opt_box)

        panel_layout.addStretch(1)
        hint = QLabel("提示：在图上移动鼠标，可查看数值 x 与累积比例 F(x)。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        panel_layout.addWidget(hint)

        splitter.addWidget(panel)

        # ---- Right plot area ----
        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = CDFCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("请加载数据文件开始使用。")

    def _connect_signals(self) -> None:
        self.btn_load.clicked.connect(self.load_file)
        self.btn_plot.clicked.connect(self.refresh_plot)
        self.cmb_value.currentIndexChanged.connect(self.refresh_plot)
        self.cmb_group.currentIndexChanged.connect(self.refresh_plot)
        self.cmb_layout.currentIndexChanged.connect(self.refresh_plot)
        self.btn_apply_x.clicked.connect(self.apply_x_range)
        self.chk_auto_x.toggled.connect(self._on_auto_x_toggled)
        self.btn_add_ref.clicked.connect(self.add_reference_line)
        self.btn_del_ref.clicked.connect(self.remove_reference_line)
        self.btn_clear_refs.clicked.connect(self.clear_reference_lines)
        self.chk_crosshair.toggled.connect(self.canvas.set_crosshair_enabled)
        self.canvas.cursor_info.connect(self._on_cursor_info)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.cmb_value,
            self.cmb_group,
            self.cmb_layout,
            self.edit_xmin,
            self.edit_xmax,
            self.chk_auto_x,
            self.btn_apply_x,
            self.btn_add_ref,
            self.btn_del_ref,
            self.btn_clear_refs,
            self.btn_plot,
            self.chk_crosshair,
        ):
            w.setEnabled(enabled)
        if enabled and self.chk_auto_x.isChecked():
            self.edit_xmin.setEnabled(False)
            self.edit_xmax.setEnabled(False)

    def _on_cursor_info(self, text: str) -> None:
        if text:
            self.status.showMessage(text)
        elif self.df is not None:
            self.status.showMessage(f"就绪 — {len(self.df)} 行")

    def _on_auto_x_toggled(self, checked: bool) -> None:
        self.edit_xmin.setEnabled(not checked)
        self.edit_xmax.setEnabled(not checked)
        if checked:
            self.apply_x_range()

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开数据文件",
            "",
            "数据文件 (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)",
        )
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 — show to user
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        if df.empty:
            QMessageBox.warning(self, "数据为空", "文件中没有数据行。")
            return

        self.df = df
        self.lbl_file.setText(Path(path).name)
        self._populate_columns()
        self._set_controls_enabled(True)
        self.refresh_plot()

    def _populate_columns(self) -> None:
        assert self.df is not None
        numeric_cols = [
            c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])
        ]
        all_cols = [str(c) for c in self.df.columns]

        self.cmb_value.blockSignals(True)
        self.cmb_group.blockSignals(True)
        self.cmb_value.clear()
        self.cmb_group.clear()
        self.cmb_group.addItem("（无）", None)
        for c in numeric_cols:
            self.cmb_value.addItem(str(c), c)
        for c in all_cols:
            self.cmb_group.addItem(c, c)
        self.cmb_value.blockSignals(False)
        self.cmb_group.blockSignals(False)

        if self.cmb_value.count() == 0:
            QMessageBox.warning(self, "无数值列", "未找到可用的数值列。")
            self._set_controls_enabled(False)
            return

        # Heuristic: first numeric as value; first non-numeric as group if present
        non_numeric = [c for c in all_cols if c not in numeric_cols]
        if non_numeric:
            idx = self.cmb_group.findData(non_numeric[0])
            if idx >= 0:
                self.cmb_group.setCurrentIndex(idx)

    def _current_groups(self):
        if self.df is None or self.cmb_value.count() == 0:
            return []
        value_col = self.cmb_value.currentData()
        group_col = self.cmb_group.currentData()
        try:
            return compute_grouped_ecdfs(self.df, value_col, group_col)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "绘图错误", str(exc))
            return []

    def refresh_plot(self) -> None:
        groups = self._current_groups()
        value_col = self.cmb_value.currentData() or "数值"
        layout_mode = self.cmb_layout.currentData() or LAYOUT_OVERLAY
        # Layout choice only matters when there are multiple groups
        self.cmb_layout.setEnabled(self.df is not None and len(groups) > 1)

        self.canvas.set_layout_mode(layout_mode, redraw=False)
        self.canvas.set_axis_labels(
            xlabel=str(value_col),
            ylabel="ratio",
            title="CDF",
        )
        self.canvas.set_reference_lines(self.ref_lines, redraw=False)

        if groups:
            lo, hi = data_x_range(groups)
            pad = (hi - lo) * 0.03
            auto_lo, auto_hi = lo - pad, hi + pad
            if self.chk_auto_x.isChecked():
                self._updating_limits = True
                self.edit_xmin.setText(f"{auto_lo:.6g}")
                self.edit_xmax.setText(f"{auto_hi:.6g}")
                self._updating_limits = False
                self.canvas.set_x_limits(auto_lo, auto_hi, redraw=False)
            else:
                self._apply_x_limits_from_edits(redraw=False)

        self.canvas.plot_groups(groups)
        n_groups = len(groups)
        n_rows = 0 if self.df is None else len(self.df)
        layout_note = ""
        if n_groups > 1:
            labels = {
                LAYOUT_OVERLAY: "叠加",
                LAYOUT_GRID: "网格",
                LAYOUT_MULTI: "多图",
            }
            layout_note = f"，布局={labels.get(layout_mode, layout_mode)}"
        self.status.showMessage(f"就绪 — {n_rows} 行，{n_groups} 组{layout_note}")

    def _apply_x_limits_from_edits(self, *, redraw: bool) -> bool:
        try:
            xmin = float(self.edit_xmin.text())
            xmax = float(self.edit_xmax.text())
        except ValueError:
            if redraw:
                QMessageBox.warning(self, "范围无效", "请输入有效的 X 最小值 / 最大值。")
            return False
        if xmin >= xmax:
            if redraw:
                QMessageBox.warning(self, "范围无效", "X 最小值必须小于 X 最大值。")
            return False
        self.canvas.set_x_limits(xmin, xmax, redraw=redraw)
        return True

    def apply_x_range(self) -> None:
        if self._updating_limits:
            return
        if not self._apply_x_limits_from_edits(redraw=True):
            return
        self.chk_auto_x.blockSignals(True)
        self.chk_auto_x.setChecked(False)
        self.chk_auto_x.blockSignals(False)
        self.edit_xmin.setEnabled(True)
        self.edit_xmax.setEnabled(True)

    def add_reference_line(self) -> None:
        dlg = ReferenceLineDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        line = dlg.get_line()
        if line is None:
            QMessageBox.warning(self, "输入无效", "请输入有效数值。")
            return
        self.ref_lines.append(line)
        orient = "竖" if line.orientation == "vertical" else "横"
        label = line.label or f"{line.value:g}"
        self.list_refs.addItem(QListWidgetItem(f"[{orient}] {label} = {line.value:g}"))
        self.canvas.set_reference_lines(self.ref_lines)

    def remove_reference_line(self) -> None:
        row = self.list_refs.currentRow()
        if row < 0:
            return
        self.list_refs.takeItem(row)
        del self.ref_lines[row]
        self.canvas.set_reference_lines(self.ref_lines)

    def clear_reference_lines(self) -> None:
        self.list_refs.clear()
        self.ref_lines.clear()
        self.canvas.set_reference_lines(self.ref_lines)


def generate_demo_dataframe(seed: int = 42) -> pd.DataFrame:
    """Synthetic grouped measurements for first-run demo."""
    rng = np.random.default_rng(seed)
    n = 80
    lines = ["Line-A"] * n + ["Line-B"] * n + ["Line-C"] * n
    values = np.concatenate(
        [
            rng.normal(10.0, 1.2, n),
            rng.normal(11.5, 1.5, n),
            rng.normal(9.5, 0.9, n),
        ]
    )
    return pd.DataFrame({"Measurement": values, "Line": lines, "Shift": (["Day", "Night"] * (3 * n // 2))[: 3 * n]})
