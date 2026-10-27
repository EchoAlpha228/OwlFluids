"""Updates for the Windows install: find a newer GitHub release and run its installer.

Releases are published by ``.github/workflows/release.yml`` with an
``OwlFluids-Setup-<version>.exe`` asset. The repository is private, so reading
releases needs a GitHub token with read access to its contents (a fine-grained
token limited to this repository is enough). The token comes from the
``OWLFLUIDS_GITHUB_TOKEN`` environment variable or from ``~/.owlfluids/update.json``,
which the app's Help > Update settings writes.

    rel = latest_release()
    if rel and rel.newer:
        run_installer(download(rel))   # the installer closes OwlFluids and reopens it

Only the standard library is used, so the check works from the command line too
(``owlfluids update``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__

REPO = "EchoAlpha228/OwlFluids"
API = "https://api.github.com"
ASSET = re.compile(r"OwlFluids-Setup-.*\.exe$", re.IGNORECASE)
CONFIG = Path.home() / ".owlfluids" / "update.json"
TIMEOUT = 10.0


class UpdateError(RuntimeError):
    pass


@dataclass
class Release:
    version: str
    tag: str
    url: str  # release page
    asset_name: str
    asset_api_url: str  # download with Accept: application/octet-stream
    size: int
    notes: str = ""

    @property
    def newer(self) -> bool:
        return parse_version(self.version) > parse_version(__version__)


def parse_version(v: str) -> tuple[int, ...]:
    """``v0.2.10`` -> (0, 2, 10); anything after the numbers (``-rc1``) is ignored."""
    m = re.match(r"v?(\d+(?:\.\d+)*)", v.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def frozen() -> bool:
    """True when running from the installed EXE rather than a Python install."""
    return bool(getattr(sys, "frozen", False))


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def token() -> str:
    return os.environ.get("OWLFLUIDS_GITHUB_TOKEN") or str(load_config().get("token") or "")


def _request(url: str, tok: str, accept: str = "application/vnd.github+json") -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": f"OwlFluids/{__version__}",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    if tok:
        # not carried over redirects: asset downloads redirect to storage that rejects a second credential
        req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return req


def latest_release(tok: str | None = None, repo: str = REPO) -> Release | None:
    """The newest published release that has an installer, or None if there is no release yet."""
    tok = token() if tok is None else tok
    try:
        with urllib.request.urlopen(_request(f"{API}/repos/{repo}/releases/latest", tok), timeout=TIMEOUT) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404 and tok and _can_read(repo, tok):
            return None  # GitHub also answers 404 to a token that can't see the repository
        if e.code in (401, 403, 404):
            raise UpdateError("GitHub refused the update check. The OwlFluids repository is private, so set a "
                              "GitHub token with read access to it (Help > Update settings, or the "
                              "OWLFLUIDS_GITHUB_TOKEN environment variable).") from None
        raise UpdateError(f"update check failed: GitHub answered {e.code} {e.reason}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise UpdateError(f"update check failed: cannot reach GitHub ({getattr(e, 'reason', e)})") from None
    return _parse_release(data)


def _can_read(repo: str, tok: str) -> bool:
    try:
        with urllib.request.urlopen(_request(f"{API}/repos/{repo}", tok), timeout=TIMEOUT):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _parse_release(data: dict) -> Release | None:
    for a in data.get("assets") or []:
        if ASSET.search(a.get("name", "")):
            tag = data.get("tag_name", "")
            return Release(version=tag.lstrip("v"), tag=tag, url=data.get("html_url", ""), asset_name=a["name"],
                           asset_api_url=a["url"], size=int(a.get("size") or 0), notes=data.get("body") or "")
    return None


def download(rel: Release, tok: str | None = None, progress: Callable[[int, int], None] | None = None,
             folder: Path | None = None) -> Path:
    """Download the release's installer; returns its path."""
    tok = token() if tok is None else tok
    folder = folder or Path(tempfile.gettempdir()) / "owlfluids-update"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / rel.asset_name
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(_request(rel.asset_api_url, tok, "application/octet-stream"),
                                    timeout=60) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or rel.size or 0)
            done = 0
            while chunk := r.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise UpdateError(f"download failed: {getattr(e, 'reason', e)}") from None
    if rel.size and tmp.stat().st_size != rel.size:
        tmp.unlink(missing_ok=True)
        raise UpdateError("download was incomplete; try again")
    os.replace(tmp, path)
    return path


def run_installer(path: Path) -> None:
    """Start the installer and return; the caller should exit so the installer can replace the files.

    ``/SILENT`` shows only a progress bar, and the installer reopens the app when it finishes.
    """
    if sys.platform != "win32":
        raise UpdateError("the installer runs on Windows only")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([str(path), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
                     creationflags=flags, close_fds=True)
