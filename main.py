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
    # Load demo data so the app is usable immediately
    window.df = generate_demo_dataframe()
    window.lbl_file.setText("演示数据（内存）")
    window._populate_columns()
    window._set_controls_enabled(True)
    window.refresh_plot()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
