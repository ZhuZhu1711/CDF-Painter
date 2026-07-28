"""CDF Plot application entry point."""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from cdf_app.fonts import configure_matplotlib_fonts, preferred_ui_font_families
from cdf_app.main_window import MainWindow, generate_demo_dataframe


def main() -> int:
    # Must run before any Figure/canvas is created.
    configure_matplotlib_fonts()

    app = QApplication(sys.argv)
    app.setApplicationName("CDF 图")
    app.setStyle("Fusion")

    font = QFont()
    font.setFamilies(preferred_ui_font_families())
    font.setPointSize(9)
    app.setFont(font)

    window = MainWindow()
    # Preload demo data for convenience, but do not plot until the user
    # selects columns and clicks「生成图形」.
    window.df = generate_demo_dataframe()
    window.lbl_file.setText("演示数据（内存）")
    window._populate_columns()
    window._set_controls_enabled(True)
    window._clear_plot()
    window.status.showMessage(
        f"已加载演示数据 — {len(window.df)} 行。请选择数值列 / 分组列，然后点击「生成图形」。"
    )
    window._on_column_selection_changed()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
