"""The version each automatic release takes (packaging/release_version.py)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "release_version", Path(__file__).resolve().parents[1] / "packaging" / "release_version.py")
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)


def test_first_release_of_a_line_is_the_committed_version():
    assert rv.next_version("0.2.0", ["v0.1.0"]) == "0.2.0"
    assert rv.next_version("0.1.0", []) == "0.1.0"


def test_later_merges_take_the_next_patch_on_that_line():
    assert rv.next_version("0.2.0", ["v0.1.0", "v0.2.0"]) == "0.2.1"
    assert rv.next_version("0.2.0", ["v0.2.0", "v0.2.1", "v0.2.10", "v0.3.0", "v0.2.3-rc1"]) == "0.2.11"
