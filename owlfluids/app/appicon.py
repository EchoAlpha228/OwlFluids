"""The app's icon and its Windows taskbar identity.

The logo lives in ``owlfluids/app/icons/owlfluids.png``; ``packaging/make_icon.py``
turns it into the multi-size ``.ico`` that the exe, installer and shortcuts use.
"""

from __future__ import annotations

import sys
from pathlib import Path

ICON_DIR = Path(__file__).with_name("icons")
ICON_PNG = ICON_DIR / "owlfluids.png"

# Windows groups taskbar buttons and pins by this id. Without one the app is grouped
# under python.exe / pythonw.exe and shows the Python logo. The installer's Start menu
# shortcut carries the same id (packaging/owlfluids.iss) so a pinned shortcut and the
# running window are one taskbar button.
APP_USER_MODEL_ID = "FRL.OwlFluids"


def set_windows_app_id() -> None:
    """Give this process its own taskbar identity. Call before any window is shown."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:  # cosmetic only: never stop the app over it
        pass


def app_icon():
    """The window icon, or an empty QIcon if the logo file is missing."""
    from PySide6.QtGui import QIcon

    return QIcon(str(ICON_PNG)) if ICON_PNG.exists() else QIcon()
