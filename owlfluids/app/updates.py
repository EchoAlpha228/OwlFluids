"""Help > Check for updates and Help > Update settings: the app side of ``owlfluids.update``."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
)

from .. import __version__, update


class UpdateSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update settings")
        cfg = update.load_config()
        form = QFormLayout(self)
        note = QLabel("OwlFluids checks its GitHub releases for new versions. A GitHub token is optional: it "
                      "lifts GitHub's limit on anonymous requests, and is needed only if the repository is "
                      f"private (a fine-grained token limited to {update.REPO} with <i>Contents: read-only</i>).")
        note.setWordWrap(True)
        form.addRow(note)
        self.token = QLineEdit(str(cfg.get("token") or ""))
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("optional: github_pat_…")
        form.addRow("GitHub token", self.token)
        self.on_start = QCheckBox("check when OwlFluids starts")
        self.on_start.setChecked(bool(cfg.get("check_on_start", True)))
        form.addRow("", self.on_start)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(460, self.sizeHint().height())

    def _ok(self) -> None:
        cfg = update.load_config()
        cfg["token"] = self.token.text().strip()
        cfg["check_on_start"] = self.on_start.isChecked()
        try:
            update.save_config(cfg)
        except OSError as e:
            QMessageBox.critical(self, "Update settings", f"Could not save {update.CONFIG}: {e}")
            return
        self.accept()


class Updater(QObject):
    """Checks and downloads on worker threads; results come back to the GUI thread through signals."""

    _checked = Signal(object, bool)  # Release | None | UpdateError, quiet
    _progress = Signal(int, int)
    _downloaded = Signal(object)  # Path | UpdateError

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._busy = False
        self._dialog: QProgressDialog | None = None
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._downloaded.connect(self._on_downloaded)

    def check_on_start(self) -> None:
        cfg = update.load_config()
        if update.frozen() and cfg.get("check_on_start", True):
            self.check(quiet=True)

    def check(self, quiet: bool = False) -> None:
        if self._busy:
            return
        self._busy = True

        def work():
            try:
                res = update.latest_release()
            except update.UpdateError as e:
                res = e
            self._checked.emit(res, quiet)

        threading.Thread(target=work, name="update-check", daemon=True).start()

    def settings(self) -> None:
        UpdateSettings(self.window).exec()

    def _on_checked(self, res, quiet: bool) -> None:
        self._busy = False
        if isinstance(res, update.UpdateError):
            if not quiet:
                r = QMessageBox.warning(self.window, "Check for updates", str(res),
                                        QMessageBox.Ok | QMessageBox.Open)
                if r == QMessageBox.Open:
                    self.settings()
            return
        if res is None or not res.newer:
            if not quiet:
                QMessageBox.information(self.window, "Check for updates", f"OwlFluids {__version__} is up to date.")
            return
        if not update.frozen():
            if not quiet:
                QMessageBox.information(self.window, "Check for updates",
                                        f"OwlFluids {res.version} is available (you have {__version__}).\n\n"
                                        "This copy runs from a Python install: update it with git pull and "
                                        f"pip install -e ., or use the installer from\n{res.url}")
            return
        msg = f"OwlFluids {res.version} is available (you have {__version__}). Install it now?"
        if self.window.runner is not None:
            msg += "\n\nThe running simulation will be stopped."
        box = QMessageBox(QMessageBox.Question, "Update available", msg, QMessageBox.Yes | QMessageBox.No,
                          self.window)
        if res.notes:
            box.setDetailedText(res.notes)
        if box.exec() != QMessageBox.Yes:
            return
        self._download(res)

    def _download(self, rel: update.Release) -> None:
        self._busy = True
        self._dialog = QProgressDialog(f"Downloading OwlFluids {rel.version}…", None, 0, 100, self.window)
        self._dialog.setWindowTitle("Update")
        self._dialog.setWindowModality(Qt.WindowModal)
        self._dialog.setMinimumDuration(0)
        self._dialog.setValue(0)

        def work():
            try:
                res = update.download(rel, progress=self._progress.emit)
            except update.UpdateError as e:
                res = e
            self._downloaded.emit(res)

        threading.Thread(target=work, name="update-download", daemon=True).start()

    def _on_progress(self, done: int, total: int) -> None:
        if self._dialog and total:
            self._dialog.setValue(int(100 * done / total))

    def _on_downloaded(self, res) -> None:
        self._busy = False
        if self._dialog:
            self._dialog.close()
            self._dialog = None
        if isinstance(res, update.UpdateError):
            QMessageBox.critical(self.window, "Update", str(res))
            return
        if not self.window.close():  # unsaved changes: the user cancelled
            return
        try:
            update.run_installer(res)
        except (update.UpdateError, OSError) as e:
            QMessageBox.critical(None, "Update", f"Could not start the installer {res}: {e}")
