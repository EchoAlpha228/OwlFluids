"""Entry point of both installed executables.

``owlfluids.exe`` (console) is the command line; ``owlfluids-app.exe`` (no console)
opens the desktop app. They are the same program, picked by the executable's name,
so the bundle holds one copy of the Python code.
"""

import os
import sys
from pathlib import Path


def _main() -> int:
    if Path(sys.executable).stem.lower().endswith("-app"):
        if sys.stdout is None:  # no console: give libraries somewhere harmless to write
            sys.stdout = sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        from owlfluids.app.main import gui_main

        return gui_main()
    from owlfluids.cli import main

    return main()


if __name__ == "__main__":
    sys.exit(_main())
