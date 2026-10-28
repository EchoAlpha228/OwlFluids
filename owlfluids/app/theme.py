"""Colours, fonts and the Qt style sheet for the desktop app, from the style guide.

Every colour the app draws comes from here. Gray ramp for chrome and text on a
dark steel base; the bright colours carry meaning (red emergency, amber caution,
olive stale data) and are not used for decoration.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

# gray ramp (steel hue, for dark backgrounds)
GRAY_050 = "#DBDBDC"
GRAY_100 = "#B9BABC"
GRAY_200 = "#999EA3"
GRAY_300 = "#818A92"
GRAY_400 = "#687582"
GRAY_500 = "#45525F"
GRAY_600 = "#34404B"
GRAY_700 = "#242B33"
GRAY_800 = "#151C23"
GRAY_900 = "#0E1216"
GRAY_950 = "#060809"
# telemetry
WHITE = "#FFFFFF"
BLACK = "#000000"
# emergency, caution, stale data
RED = "#FF0000"
DARK_RED = "#B30000"
AMBER = "#FFA200"
DARK_AMBER = "#B37100"
OLIVE = "#B3B000"
DARK_OLIVE = "#666400"
# multi-purpose
GREEN = "#00FF4D"
DARK_GREEN = "#00B336"
CYAN = "#00D0FF"
DARK_CYAN = "#0092B3"
BLUE = "#0088FF"
DARK_BLUE = "#005FB3"
MAGENTA = "#FF00EE"
DARK_MAGENTA = "#B300A7"

# roles
BACKGROUND = GRAY_900
PANEL = GRAY_800
FIELD = GRAY_950
RAISED = GRAY_700
BORDER = GRAY_600
TEXT = GRAY_050
TEXT_DIM = GRAY_200
TEXT_OFF = GRAY_400
ACCENT = CYAN
SELECTION = DARK_BLUE

VALVE_OPEN = DARK_GREEN
VALVE_CLOSED = DARK_RED  # bright Red stays for emergencies

# fluid material colours: desaturated so they sit below the UI colours. Liquid is
# drawn solid; gas is the same hue, see-through (GAS_ALPHA) over the dark background.
FLUID_COLOURS = {
    "oxygen": "#7FB0D6",  # light blue
    "methane": "#C98A4E",  # orange
    "helium": "#C7B65E",  # yellow
    "nitrogen": "#6FA27A",  # green
    "air": "#A2A7AC",  # gray
    "nitrous": "#4D6C9E",  # dark blue
    "ethanol": "#B4625F",  # red
    "water": "#5E9C9A",  # teal
}
FLUID_UNKNOWN = GRAY_400
GAS_ALPHA = 0.6

# plot traces, in order; no red, amber or olive, which carry state and alarm meanings
TRACES = [CYAN, GREEN, BLUE, MAGENTA, GRAY_050, DARK_CYAN, DARK_GREEN, DARK_BLUE, DARK_MAGENTA, GRAY_300]

UI_FONT = "Lato"
MONO_FONT = "Roboto Mono"
LABEL_FONT = "Barlow Semi Condensed SemiBold"  # free condensed stand-in for DIN: labels and navigation
_FONTS = Path(__file__).with_name("fonts")


def qc(hex_colour: str) -> QColor:
    return QColor(hex_colour)


def load_fonts() -> None:
    """Register the bundled fonts (Lato, Roboto Mono, Barlow Semi Condensed); Qt falls back to system fonts."""
    for f in sorted(_FONTS.glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(f))


def ui_font(size: float = 9, bold: bool = False) -> QFont:
    f = QFont(UI_FONT)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def mono_font(size: float = 9) -> QFont:
    f = QFont(MONO_FONT)
    f.setStyleHint(QFont.Monospace)
    f.setPointSizeF(size)
    return f


def monospace(widget):
    """Show a widget in Roboto Mono. The style sheet's font wins over setFont, so this goes through it."""
    widget.setProperty("mono", True)
    return widget


STYLE_SHEET = f"""
* {{ font-family: "{UI_FONT}"; font-size: 9pt; color: {TEXT}; }}
QMainWindow, QDialog, QMessageBox {{ background: {BACKGROUND}; }}
QWidget {{ background: {BACKGROUND}; selection-background-color: {SELECTION}; selection-color: {WHITE}; }}
QMenuBar, QToolBar, QStatusBar {{ background: {GRAY_950}; border: none; }}
QToolBar {{ padding: 3px; spacing: 2px; border-bottom: 1px solid {BORDER}; }}
QStatusBar {{ border-top: 1px solid {BORDER}; }}
QLabel {{ background: transparent; }}
QMenuBar::item {{ background: transparent; padding: 4px 9px; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {GRAY_600}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 14px; }}
QMenu::item:disabled {{ color: {TEXT_OFF}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 4px 9px; }}
QToolButton:hover {{ background: {GRAY_600}; border-color: {GRAY_500}; }}
QToolButton:checked {{ background: {DARK_RED}; color: {WHITE}; }}
QToolButton:disabled {{ color: {TEXT_OFF}; }}
QDockWidget {{ color: {TEXT_DIM}; titlebar-close-icon: none; }}
QDockWidget::title {{ background: {GRAY_950}; padding: 5px 8px; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; background: {PANEL}; }}
QTabBar::tab {{ background: transparent; color: {TEXT_DIM}; padding: 6px 14px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {PANEL}; border: none; }}
QTreeWidget, QListWidget {{ background: {PANEL}; border: none; outline: none; }}
QTreeWidget::item, QListWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {GRAY_700}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{ background: {SELECTION}; }}
QLineEdit, QComboBox, QAbstractSpinBox {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 3px;
                                         padding: 4px 6px; }}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:read-only {{ color: {TEXT_DIM}; border-color: {GRAY_700}; }}
QLineEdit[error="true"] {{ background: {DARK_RED}; border-color: {RED}; color: {WHITE}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 3px; padding: 5px 12px; }}
QPushButton:hover {{ background: {GRAY_600}; border-color: {GRAY_500}; }}
QPushButton:pressed {{ background: {GRAY_500}; }}
QPushButton:disabled {{ color: {TEXT_OFF}; background: {GRAY_800}; border-color: {GRAY_700}; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {GRAY_400}; background: {FIELD}; }}
QListWidget::indicator {{ width: 11px; height: 11px; border: 1px solid {GRAY_400}; background: {FIELD}; }}
QCheckBox::indicator:checked, QListWidget::indicator:checked {{ background: {GRAY_100}; border-color: {GRAY_100}; }}
QProgressBar {{ background: {FIELD}; border: 1px solid {BORDER}; border-radius: 2px; text-align: center;
               font-family: "{MONO_FONT}"; font-size: 8pt; }}
QProgressBar::chunk {{ background: {GRAY_500}; }}
QSlider::groove:horizontal {{ height: 4px; background: {GRAY_600}; }}
QSlider::handle:horizontal {{ background: {GRAY_100}; width: 10px; margin: -5px 0; border-radius: 2px; }}
QScrollBar {{ background: {PANEL}; border: none; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {GRAY_600}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {GRAY_500}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle, QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
QToolTip {{ background: {GRAY_700}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QLabel#hint {{ color: {TEXT_DIM}; }}
QLabel#heading {{ color: {TEXT_DIM}; font-weight: bold; letter-spacing: 1px; }}
QPushButton#live {{ font-family: "{LABEL_FONT}"; letter-spacing: 1px; padding: 2px 10px;
                   background: {DARK_OLIVE}; border-color: {DARK_OLIVE}; color: {WHITE}; }}
QPushButton#live:checked {{ background: {DARK_GREEN}; border-color: {DARK_GREEN}; }}
*[mono="true"] {{ font-family: "{MONO_FONT}"; }}
QMenuBar, QToolButton, QDockWidget, QTabBar::tab, QLabel#heading {{ font-family: "{LABEL_FONT}"; font-size: 10pt; }}
"""


def apply(app) -> None:
    """Fusion style, the bundled fonts and the style sheet, once per application."""
    if app.property("owlfluids_theme"):
        return
    load_fonts()
    app.setStyle("Fusion")
    app.setFont(ui_font())
    app.setStyleSheet(STYLE_SHEET)
    app.setProperty("owlfluids_theme", True)


BADGES = {"stopped": (GRAY_600, TEXT), "building": (DARK_CYAN, WHITE), "running": (DARK_GREEN, WHITE),
          "paused": (DARK_OLIVE, WHITE)}
