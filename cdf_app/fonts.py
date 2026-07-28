"""Configure fonts so Chinese labels render correctly in Qt and Matplotlib."""

from __future__ import annotations

from typing import List, Optional

from matplotlib import font_manager, rcParams

# Prefer common CJK fonts across Windows / macOS / Linux.
_CJK_CANDIDATES: List[str] = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
]

_configured = False
_selected_font: Optional[str] = None


def available_cjk_font() -> Optional[str]:
    """Return the first installed CJK-capable font name, or None."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    # Fallback: probe via findfont (handles some alias cases)
    for name in _CJK_CANDIDATES:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        # Reject matplotlib's default if findfont silently substituted
        if path and "DejaVu" not in path:
            return name
    return None


def configure_matplotlib_fonts() -> Optional[str]:
    """Set Matplotlib sans-serif to a CJK font so plot Chinese is not tofu/garbled.

    Returns the selected font family name, or None if none was found.
    """
    global _configured, _selected_font
    if _configured:
        return _selected_font

    font_name = available_cjk_font()
    if font_name:
        # Put chosen font first; keep DejaVu as Latin fallback.
        rcParams["font.family"] = "sans-serif"
        existing = list(rcParams.get("font.sans-serif", []))
        rcParams["font.sans-serif"] = [font_name] + [
            f for f in existing if f != font_name
        ]
        _selected_font = font_name

    # Avoid squares when using Unicode minus with CJK fonts
    rcParams["axes.unicode_minus"] = False
    _configured = True
    return _selected_font


def preferred_ui_font_families() -> List[str]:
    """Font family list for QFont, best-first."""
    chosen = configure_matplotlib_fonts()
    families: List[str] = []
    if chosen:
        families.append(chosen)
    for name in _CJK_CANDIDATES:
        if name not in families:
            families.append(name)
    families.append("sans-serif")
    return families
