"""Desktop app entry point: ``owlfluids app [system.yaml|example]``."""

from __future__ import annotations

import sys


def main(path: str | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The desktop app needs PySide6 and pyqtgraph: pip install owlfluids[app]", file=sys.stderr)
        return 1
    from ..examples import EXAMPLES
    from .window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OwlFluids")
    app.setStyle("Fusion")
    w = MainWindow()
    if path:
        if path in EXAMPLES:
            w.load_example(path)
        else:
            w.open_file(path)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
