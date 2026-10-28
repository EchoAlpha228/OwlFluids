"""Per-user app preferences: window layout, last folder and recent systems (Qt settings, not the repo)."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings

RECENT_MAX = 8


def settings() -> QSettings:
    """The user's settings; OWLFLUIDS_SETTINGS (a file path) points them elsewhere, as the tests do."""
    path = os.environ.get("OWLFLUIDS_SETTINGS")
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings(QSettings.IniFormat, QSettings.UserScope, "OwlFluids", "OwlFluids")


def recent() -> list[str]:
    """Recently opened or saved systems, newest first, that still exist."""
    files = settings().value("recent", []) or []
    if isinstance(files, str):  # a one-item list comes back as a plain string
        files = [files]
    return [f for f in files if Path(f).is_file()]


def add_recent(path: str | Path) -> None:
    path = str(Path(path).resolve())
    s = settings()
    files = [path] + [f for f in recent() if f != path]
    s.setValue("recent", files[:RECENT_MAX])
    s.setValue("last_dir", str(Path(path).parent))


def last_dir() -> str:
    return str(settings().value("last_dir", "") or "")
