"""Searchable column pickers (single-select and multi-select)."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from PyQt5.QtCore import Qt, QSortFilterProxyModel, pyqtSignal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _fuzzy_match(query: str, text: str) -> bool:
    """Case-insensitive match: substring, or characters in order (fuzzy)."""
    q = query.strip().lower()
    if not q:
        return True
    t = text.lower()
    if q in t:
        return True
    # subsequence: e.g. "msr" matches "Measurement"
    i = 0
    for ch in t:
        if ch == q[i]:
            i += 1
            if i == len(q):
                return True
    return False


class FuzzyFilterProxy(QSortFilterProxyModel):
    """Proxy that filters with substring / fuzzy matching."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._query = ""

    def set_query(self, query: str) -> None:
        self._query = query or ""
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:  # noqa: N802
        if not self._query.strip():
            return True
        index = self.sourceModel().index(source_row, 0, source_parent)
        text = str(self.sourceModel().data(index, Qt.DisplayRole) or "")
        return _fuzzy_match(self._query, text)


class SearchableComboBox(QComboBox):
    """Editable single-select combo with fuzzy filter-as-you-type."""

    def __init__(self, parent=None, *, placeholder: str = "搜索并选择…"):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setMaxVisibleItems(16)
        self.lineEdit().setPlaceholderText(placeholder)

        self._model = QStandardItemModel(self)
        self._proxy = FuzzyFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self.setModel(self._proxy)

        self.setView(QListView(self))
        self.view().setUniformItemSizes(True)

        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.activated.connect(self._on_activated)

        # Prefer our fuzzy filter over the default completer popup
        self.setCompleter(None)

    def _on_text_edited(self, text: str) -> None:
        self._proxy.set_query(text)
        self.showPopup()

    def _on_activated(self, index: int) -> None:
        # Keep the chosen label in the line edit, then reset filter.
        # Resolve via source model so clearing the filter cannot remap the
        # selection to a different row (e.g. filtered index 0 →「无」).
        if 0 <= index < self._proxy.rowCount():
            proxy_index = self._proxy.index(index, 0)
            src = self._proxy.mapToSource(proxy_index)
            label = self._model.data(src, Qt.DisplayRole)
            data = self._model.data(src, Qt.UserRole)
            self._proxy.set_query("")
            if data is not None or src.isValid():
                self.setCurrentData(data)
            elif self.lineEdit() and label is not None:
                self.lineEdit().setText(str(label))
            return
        self._proxy.set_query("")

    def clear_items(self) -> None:
        self._model.clear()
        self._proxy.set_query("")
        self.setCurrentIndex(-1)
        if self.lineEdit():
            self.lineEdit().clear()

    def add_option(self, label: str, data=None) -> None:
        item = QStandardItem(label)
        item.setData(data, Qt.UserRole)
        item.setEditable(False)
        self._model.appendRow(item)

    def set_options(self, options: Sequence[tuple]) -> None:
        """options: sequence of (label, data)."""
        self.blockSignals(True)
        self.clear_items()
        for label, data in options:
            self.add_option(str(label), data)
        if self._model.rowCount():
            self.setCurrentIndex(0)
            first = self._model.item(0).text()
            if self.lineEdit():
                self.lineEdit().setText(first)
        self.blockSignals(False)

    def currentData(self, role=Qt.UserRole):  # noqa: N802 — match QComboBox API
        idx = self.currentIndex()
        if idx >= 0:
            src = self._proxy.mapToSource(self._proxy.index(idx, 0))
            if src.isValid():
                return self._model.data(src, role)

        # Fallback: match the visible line-edit text to an option label.
        # Covers typing a full column name without clicking the popup item.
        if role == Qt.UserRole and self.lineEdit() is not None:
            text = self.lineEdit().text().strip()
            if text:
                for row in range(self._model.rowCount()):
                    item = self._model.item(row)
                    if item is not None and item.text() == text:
                        return item.data(Qt.UserRole)
        return None

    def findData(self, data) -> int:  # noqa: N802
        self._proxy.set_query("")
        for row in range(self._model.rowCount()):
            item_data = self._model.item(row).data(Qt.UserRole)
            if item_data == data or (item_data is None and data is None):
                proxy_idx = self._proxy.mapFromSource(self._model.index(row, 0))
                return proxy_idx.row()
        return -1

    def setCurrentData(self, data) -> None:  # noqa: N802
        idx = self.findData(data)
        if idx >= 0:
            self.setCurrentIndex(idx)
            label = self.itemText(idx)
            if self.lineEdit():
                self.lineEdit().setText(label)
        else:
            self.setCurrentIndex(-1)
            if self.lineEdit():
                self.lineEdit().clear()


class SearchableMultiSelect(QWidget):
    """Searchable checklist for selecting multiple columns."""

    selectionChanged = pyqtSignal()

    def __init__(self, parent=None, *, placeholder: str = "搜索数值列…", empty_summary: str = "未选择"):
        super().__init__(parent)
        self._items: List[tuple] = []  # (label, data)
        self._block = False
        self._empty_summary = empty_summary

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        search_row = QHBoxLayout()
        search_row.setSpacing(4)
        self.search = QLineEdit()
        self.search.setPlaceholderText(placeholder)
        self.search.setClearButtonEnabled(True)
        self.btn_all = QToolButton()
        self.btn_all.setText("全选")
        self.btn_none = QToolButton()
        self.btn_none.setText("清空")
        search_row.addWidget(self.search, 1)
        search_row.addWidget(self.btn_all)
        search_row.addWidget(self.btn_none)
        root.addLayout(search_row)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setMinimumHeight(120)
        self.list.setMaximumHeight(180)
        self.list.setAlternatingRowColors(True)
        root.addWidget(self.list)

        self.lbl_summary = QLineEdit()
        self.lbl_summary.setReadOnly(True)
        self.lbl_summary.setPlaceholderText(empty_summary)
        self.lbl_summary.setFocusPolicy(Qt.NoFocus)
        root.addWidget(self.lbl_summary)

        self.search.textChanged.connect(self._apply_filter)
        self.list.itemChanged.connect(self._on_item_changed)
        self.btn_all.clicked.connect(self.select_all_visible)
        self.btn_none.clicked.connect(self.clear_selection)

    def set_options(self, options: Sequence[tuple]) -> None:
        """options: sequence of (label, data)."""
        self._block = True
        self._items = [(str(label), data) for label, data in options]
        self.list.clear()
        self.search.clear()
        for label, data in self._items:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, data)
            self.list.addItem(item)
        self._block = False
        self._update_summary()

    def _apply_filter(self, query: str) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(not _fuzzy_match(query, item.text()))

    def _on_item_changed(self, _item: QListWidgetItem) -> None:
        if self._block:
            return
        self._update_summary()
        self.selectionChanged.emit()

    def _update_summary(self) -> None:
        selected = self.selected_labels()
        if not selected:
            self.lbl_summary.setText("")
            self.lbl_summary.setPlaceholderText(self._empty_summary)
        elif len(selected) <= 3:
            self.lbl_summary.setText("、".join(selected))
        else:
            self.lbl_summary.setText(f"已选 {len(selected)} 项：" + "、".join(selected[:3]) + "…")

    def selected_data(self) -> List:
        result = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.data(Qt.UserRole))
        return result

    def selected_labels(self) -> List[str]:
        result = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    def set_selected_data(self, values: Iterable) -> None:
        wanted = set(values)
        self._block = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted else Qt.Unchecked)
        self._block = False
        self._update_summary()
        self.selectionChanged.emit()

    def select_all_visible(self) -> None:
        self._block = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)
        self._block = False
        self._update_summary()
        self.selectionChanged.emit()

    def clear_selection(self) -> None:
        self._block = True
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Unchecked)
        self._block = False
        self._update_summary()
        self.selectionChanged.emit()

    def clear_options(self) -> None:
        self._block = True
        self._items.clear()
        self.list.clear()
        self.search.clear()
        self._block = False
        self._update_summary()
