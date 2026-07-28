"""Lazy-loading data preview dialog with background workers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd
from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    QRunnable,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from .outliers import (
    RULE_LABELS,
    OutlierResult,
    OutlierRule,
    preview_outlier_counts,
    remove_outliers,
)


# Rows fetched per background chunk (lazy page size).
PAGE_SIZE = 200


class _ChunkSignals(QObject):
    """Signals for a background chunk-format worker."""

    finished = pyqtSignal(int, object)  # start_row, rows as list[list[str]]
    failed = pyqtSignal(str)


class ChunkFormatWorker(QRunnable):
    """Format a slice of a DataFrame to display strings off the UI thread."""

    def __init__(self, df: pd.DataFrame, start: int, end: int, signals: _ChunkSignals):
        super().__init__()
        self.df = df
        self.start = start
        self.end = end
        # signals must outlive this runnable (owned by the model / dialog).
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            slice_df = self.df.iloc[self.start : self.end]
            rows: List[List[str]] = []
            for tup in slice_df.itertuples(index=False, name=None):
                rows.append(["" if pd.isna(v) else str(v) for v in tup])
            self.signals.finished.emit(self.start, rows)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))


class LazyDataFrameModel(QAbstractTableModel):
    """Table model that materializes row text in background pages on demand.

    ``canFetchMore`` / ``fetchMore`` drive lazy loading: only when the view
    scrolls near uncached rows does a worker format the next page.
    """

    loadingChanged = pyqtSignal(bool)
    loadError = pyqtSignal(str)

    def __init__(
        self,
        df: pd.DataFrame,
        parent=None,
        *,
        page_size: int = PAGE_SIZE,
        pool: Optional[QThreadPool] = None,
    ):
        super().__init__(parent)
        self._df = df
        self._page_size = max(50, int(page_size))
        self._columns = [str(c) for c in df.columns]
        self._n_rows = len(df)
        # Cache of formatted rows; None means not yet loaded.
        self._cache: List[Optional[List[str]]] = [None] * self._n_rows
        self._loaded_upto = 0
        self._fetching = False
        self._pool = pool or QThreadPool.globalInstance()
        self._pending_workers = 0
        # Keep signal emitters alive for the lifetime of in-flight runnables.
        self._chunk_signals: List[_ChunkSignals] = []

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._loaded_upto

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section]
            return None
        # 1-based row numbers for the currently materialized range
        return str(section + 1)

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row < 0 or row >= self._loaded_upto or col < 0 or col >= len(self._columns):
            return None
        if role == Qt.DisplayRole:
            cached = self._cache[row]
            if cached is None:
                return "…"
            return cached[col]
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self._cache[row] is None:
            return QColor("#f5f5f5")
        return None

    def canFetchMore(self, parent=QModelIndex()) -> bool:  # noqa: N802
        if parent.isValid():
            return False
        return self._loaded_upto < self._n_rows and not self._fetching

    def fetchMore(self, parent=QModelIndex()) -> None:  # noqa: N802
        if parent.isValid() or self._fetching or self._loaded_upto >= self._n_rows:
            return
        start = self._loaded_upto
        end = min(start + self._page_size, self._n_rows)
        self._fetching = True
        self.loadingChanged.emit(True)

        # Reserve rows in the model immediately so the view can scroll,
        # then fill cell text when the worker finishes.
        self.beginInsertRows(QModelIndex(), start, end - 1)
        self._loaded_upto = end
        self.endInsertRows()

        signals = _ChunkSignals(self)
        self._chunk_signals.append(signals)
        signals.finished.connect(self._on_chunk_ready)
        signals.failed.connect(self._on_chunk_failed)
        # Drop emitter after it fires so the list does not grow forever.
        signals.finished.connect(lambda *_a, s=signals: self._release_signals(s))
        signals.failed.connect(lambda *_a, s=signals: self._release_signals(s))

        worker = ChunkFormatWorker(self._df, start, end, signals)
        self._pending_workers += 1
        self._pool.start(worker)

    def _release_signals(self, signals: _ChunkSignals) -> None:
        try:
            self._chunk_signals.remove(signals)
        except ValueError:
            pass

    @pyqtSlot(int, object)
    def _on_chunk_ready(self, start: int, rows: object) -> None:
        if not isinstance(rows, list):
            self._pending_workers = max(0, self._pending_workers - 1)
            self._fetching = False
            self.loadingChanged.emit(self._pending_workers > 0)
            return
        for i, row_data in enumerate(rows):
            idx = start + i
            if 0 <= idx < self._n_rows:
                self._cache[idx] = row_data
        if rows:
            top_left = self.index(start, 0)
            bottom_right = self.index(start + len(rows) - 1, len(self._columns) - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.DisplayRole, Qt.BackgroundRole])
        self._pending_workers = max(0, self._pending_workers - 1)
        self._fetching = False
        self.loadingChanged.emit(self._pending_workers > 0)

    @pyqtSlot(str)
    def _on_chunk_failed(self, message: str) -> None:
        self._pending_workers = max(0, self._pending_workers - 1)
        self._fetching = False
        self.loadingChanged.emit(self._pending_workers > 0)
        self.loadError.emit(message)

    def total_rows(self) -> int:
        return self._n_rows

    def loaded_rows(self) -> int:
        return self._loaded_upto


class DataPreviewDialog(QDialog):
    """Scrollable preview of a DataFrame with multi-threaded lazy row loading."""

    def __init__(self, df: pd.DataFrame, parent=None, *, title: str = "数据预览"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)
        self._df = df
        self._pool = QThreadPool(self)
        # Cap workers so formatting does not starve the UI / plot threads.
        self._pool.setMaxThreadCount(max(2, min(4, QThreadPool.globalInstance().maxThreadCount())))

        root = QVBoxLayout(self)

        info = QLabel(
            f"共 {len(df)} 行 × {len(df.columns)} 列。"
            "表格按需分页加载（后台线程格式化），滚动时自动拉取下一批。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        root.addWidget(info)

        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("准备加载…")
        self.progress = QProgressBar()
        self.progress.setRange(0, max(1, len(df)))
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("已加载 %v / %m 行")
        status_row.addWidget(self.lbl_status, 1)
        status_row.addWidget(self.progress, 1)
        root.addLayout(status_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        root.addWidget(buttons)

        self._model = LazyDataFrameModel(df, self, pool=self._pool)
        self.table.setModel(self._model)
        self._model.loadingChanged.connect(self._on_loading_changed)
        self._model.loadError.connect(self._on_load_error)

        # Kick off the first page so the dialog is not empty.
        if self._model.canFetchMore():
            self._model.fetchMore()
        self._refresh_status()

    def _on_loading_changed(self, loading: bool) -> None:
        self._refresh_status(loading=loading)
        # If viewport still has room and more data exists, fetch again.
        if not loading and self._model.canFetchMore():
            # Ensure first screenful is filled even before user scrolls.
            viewport_rows = max(1, self.table.viewport().height() // 22)
            if self._model.loaded_rows() < viewport_rows + PAGE_SIZE:
                self._model.fetchMore()

    def _on_load_error(self, message: str) -> None:
        QMessageBox.warning(self, "预览加载失败", message)

    def _refresh_status(self, *, loading: bool = False) -> None:
        loaded = self._model.loaded_rows()
        total = self._model.total_rows()
        self.progress.setValue(loaded)
        state = "加载中…" if loading else ("已全部加载" if loaded >= total else "等待滚动继续加载")
        self.lbl_status.setText(f"{state}（已显示 {loaded} / {total} 行）")


class OutlierFilterDialog(QDialog):
    """Let the user pick an outlier rule and columns, then apply removal."""

    def __init__(self, df: pd.DataFrame, parent=None, *, preferred_cols: Optional[Sequence] = None):
        super().__init__(parent)
        self.setWindowTitle("异常数据剔除")
        self.setMinimumWidth(420)
        self.df = df
        self.result: Optional[OutlierResult] = None

        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        preferred = set(preferred_cols or [])

        root = QVBoxLayout(self)

        rule_box = QGroupBox("剔除规则")
        rule_form = QFormLayout(rule_box)
        self.cmb_rule = QComboBox()
        for rule, label in RULE_LABELS.items():
            self.cmb_rule.addItem(label, rule)
        rule_form.addRow("方法:", self.cmb_rule)
        hint = QLabel(
            "· IQR：经典箱线图规则，适合偏态不明显的数据\n"
            "· Z-Score：按标准差倍数剔除极端值（|z|>3）\n"
            "· 分位数：去掉两端各约 1% 的极端观测"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        rule_form.addRow(hint)
        root.addWidget(rule_box)

        col_box = QGroupBox("检测列（数值列）")
        col_layout = QVBoxLayout(col_box)
        self.list_cols = QListWidget()
        self.list_cols.setMinimumHeight(140)
        for c in numeric_cols:
            item = QListWidgetItem(str(c))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if (not preferred or c in preferred) else Qt.Unchecked)
            item.setData(Qt.UserRole, c)
            self.list_cols.addItem(item)
        col_layout.addWidget(self.list_cols)
        col_btns = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("清空")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        col_btns.addWidget(btn_all)
        col_btns.addWidget(btn_none)
        col_btns.addStretch(1)
        col_layout.addLayout(col_btns)
        root.addWidget(col_box)

        self.lbl_preview = QLabel("选择规则与列后，将显示预计剔除行数。")
        self.lbl_preview.setWordWrap(True)
        root.addWidget(self.lbl_preview)

        self.chk_confirm = QCheckBox("确认剔除（不可自动撤销，请先预览确认）")
        root.addWidget(self.chk_confirm)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_preview = QPushButton("预估影响")
        buttons.addButton(self.btn_preview, QDialogButtonBox.ActionRole)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.cmb_rule.currentIndexChanged.connect(self._update_estimate)
        self.list_cols.itemChanged.connect(lambda _i: self._update_estimate())
        self.btn_preview.clicked.connect(self._update_estimate)

        if not numeric_cols:
            self.lbl_preview.setText("当前数据没有数值列，无法剔除异常值。")
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        else:
            self._update_estimate()

    def _selected_columns(self) -> List:
        cols = []
        for i in range(self.list_cols.count()):
            item = self.list_cols.item(i)
            if item.checkState() == Qt.Checked:
                cols.append(item.data(Qt.UserRole))
        return cols

    def _select_all(self) -> None:
        for i in range(self.list_cols.count()):
            self.list_cols.item(i).setCheckState(Qt.Checked)

    def _select_none(self) -> None:
        for i in range(self.list_cols.count()):
            self.list_cols.item(i).setCheckState(Qt.Unchecked)

    def _current_rule(self) -> OutlierRule:
        return self.cmb_rule.currentData()

    def _update_estimate(self) -> None:
        cols = self._selected_columns()
        if not cols:
            self.lbl_preview.setText("请至少勾选一个数值列。")
            return
        try:
            n_removed, n_total = preview_outlier_counts(self.df, cols, self._current_rule())
        except Exception as exc:  # noqa: BLE001
            self.lbl_preview.setText(f"预估失败：{exc}")
            return
        pct = (100.0 * n_removed / n_total) if n_total else 0.0
        self.lbl_preview.setText(
            f"预计剔除 {n_removed} / {n_total} 行（{pct:.1f}%），"
            f"保留 {n_total - n_removed} 行。"
        )

    def _on_accept(self) -> None:
        cols = self._selected_columns()
        if not cols:
            QMessageBox.information(self, "未选择列", "请至少勾选一个用于检测的数值列。")
            return
        if not self.chk_confirm.isChecked():
            QMessageBox.information(self, "请确认", "请勾选「确认剔除」后再执行。")
            return
        try:
            self.result = remove_outliers(self.df, cols, self._current_rule())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "剔除失败", str(exc))
            return
        if self.result.n_removed == 0:
            QMessageBox.information(self, "无异常", "按当前规则未检测到需要剔除的行。")
            self.reject()
            return
        self.accept()
