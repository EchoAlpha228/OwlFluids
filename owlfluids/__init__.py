"""OwlFluids: a real-time 1-D fluid network simulator for test automation."""

import hashlib as _hashlib
import os as _os
from pathlib import Path as _Path

__version__ = "0.1.0"


def _numba_cache_dir() -> None:
    # Numba's on-disk cache only notices edits to the file that holds a compiled
    # function, not to the functions it calls in other files. Keying the cache
    # directory on a hash of all kernel sources keeps a stale kernel from loading.
    if "NUMBA_CACHE_DIR" in _os.environ:
        return
    here = _Path(__file__).parent
    h = _hashlib.sha1(__version__.encode())
    for rel in ("core/kernel.py", "core/layout.py", "fluids/props.py", "fluids/registry.py", "fluids/tables.py"):
        try:
            h.update((here / rel).read_bytes())
        except OSError:
            pass
    _os.environ["NUMBA_CACHE_DIR"] = str(_Path.home() / ".owlfluids" / "numba-cache" / h.hexdigest()[:12])


_numba_cache_dir()
