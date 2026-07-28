"""Background workers for non-blocking file I/O."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class FileLoadSignals(QObject):
    finished = pyqtSignal(object, str)  # DataFrame, path
    failed = pyqtSignal(str, str)  # message, path
    started = pyqtSignal(str)  # path


class FileLoadWorker(QRunnable):
    """Load CSV / Excel on a worker thread so the UI stays responsive."""

    def __init__(self, path: str, signals: FileLoadSignals):
        super().__init__()
        self.path = path
        # Owned by the caller (MainWindow) so it outlives this runnable.
        self.signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:
        path = self.path
        self.signals.started.emit(path)
        try:
            lower = path.lower()
            if lower.endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            self.signals.finished.emit(df, path)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc), path)


def file_basename(path: str) -> str:
    return Path(path).name
