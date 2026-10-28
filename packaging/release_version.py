"""The version a release build gets, for .github/workflows/release.yml.

Every merge to main is released. ``__version__`` in owlfluids/__init__.py holds the
major.minor line (and the first release of it); each later merge on that line takes
the next patch number after the newest ``v<major>.<minor>.*`` tag. The build's copy of
owlfluids/__init__.py is then rewritten to that version, so the installed app knows it
and its updater compares against the right number. Nothing is committed.

    python packaging/release_version.py            # print the next release version
    python packaging/release_version.py --apply    # ...and write it into owlfluids/__init__.py
    python packaging/release_version.py --base     # print __version__ as committed
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "owlfluids" / "__init__.py"
VERSION = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def next_version(base: str, tags: list[str]) -> str:
    """``base`` if it has not been released yet, else one patch past the newest release on its line."""
    major, minor, _ = (int(x) for x in base.split("."))
    if f"v{base}" not in tags:
        return base
    patches = [int(m.group(1)) for t in tags if (m := re.fullmatch(rf"v{major}\.{minor}\.(\d+)", t))]
    return f"{major}.{minor}.{max(patches) + 1}"


def main(argv: list[str]) -> int:
    text = INIT.read_text(encoding="utf-8")
    m = VERSION.search(text)
    if not m:
        print(f"no X.Y.Z __version__ in {INIT}", file=sys.stderr)
        return 1
    base = ".".join(m.groups())
    if "--base" in argv:
        print(base)
        return 0
    tags = subprocess.run(["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True).stdout.split()
    version = next_version(base, tags)
    if "--apply" in argv and version != base:
        INIT.write_text(VERSION.sub(f'__version__ = "{version}"', text, count=1), encoding="utf-8")
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
