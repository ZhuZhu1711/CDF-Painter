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
    demo = generate_demo_dataframe()
    window._apply_dataframe(
        demo,
        label="演示数据（内存）",
        reset_original=True,
        status_extra="请选择数值列 / 分组列，然后点击「生成图形」。也可先「数据预览」或「异常剔除」。",
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
