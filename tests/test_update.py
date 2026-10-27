"""Update check (GitHub releases) and bundled property tables, without the network."""
import io
import json

import pytest

from owlfluids import __version__, update
from owlfluids.fluids import tables


def test_parse_version():
    assert update.parse_version("v0.2.10") == (0, 2, 10)
    assert update.parse_version("1.0.0-rc1") == (1, 0, 0)
    assert update.parse_version("0.10.0") > update.parse_version("0.9.9")
    assert update.parse_version("junk") == (0,)


def _release(tag, assets):
    return {"tag_name": tag, "html_url": f"https://github.com/{update.REPO}/releases/tag/{tag}", "body": "notes",
            "assets": assets}


def test_parse_release_picks_installer():
    exe = {"name": "OwlFluids-Setup-9.0.0.exe", "url": "https://api.github.com/x/assets/1", "size": 123}
    rel = update._parse_release(_release("v9.0.0", [{"name": "source.zip", "url": "u", "size": 1}, exe]))
    assert rel.version == "9.0.0" and rel.asset_api_url == exe["url"] and rel.size == 123
    assert rel.newer
    assert update._parse_release(_release("v9.0.0", [{"name": "source.zip", "url": "u"}])) is None
    same = update._parse_release(_release(f"v{__version__}", [exe]))
    assert not same.newer


def test_latest_release_sends_token_only_to_github(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.unredirected_hdrs.get("Authorization")
        seen["redirected"] = req.headers.get("Authorization")
        body = _release("v9.0.0", [{"name": "OwlFluids-Setup-9.0.0.exe", "url": "a", "size": 1}])
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)
    rel = update.latest_release("tok")
    assert rel.version == "9.0.0"
    assert seen["url"].endswith(f"/repos/{update.REPO}/releases/latest")
    assert seen["auth"] == "Bearer tok"
    assert seen["redirected"] is None  # not forwarded to the storage host the asset redirects to


def test_latest_release_without_access(monkeypatch):
    def fake_urlopen(req, timeout):
        raise update.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(update.UpdateError, match="token"):
        update.latest_release("")
    with pytest.raises(update.UpdateError, match="token"):
        update.latest_release("bad")  # token can't see the repository either


def test_token_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "CONFIG", tmp_path / "update.json")
    monkeypatch.delenv("OWLFLUIDS_GITHUB_TOKEN", raising=False)
    assert update.token() == ""
    update.save_config({"token": "abc"})
    assert update.token() == "abc"
    monkeypatch.setenv("OWLFLUIDS_GITHUB_TOKEN", "env")
    assert update.token() == "env"


def test_bundled_tables_are_a_fallback(tmp_path, monkeypatch):
    user, bundled = tmp_path / "user", tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    monkeypatch.setenv("OWLFLUIDS_TABLE_DIR", str(user))
    monkeypatch.setattr(tables, "bundled_table_dir", lambda: bundled)
    name = tables.table_path("Nitrogen", "HEOS").name
    assert tables.find_table("Nitrogen", "HEOS") is None
    (bundled / name).write_bytes(b"")
    assert tables.find_table("Nitrogen", "HEOS") == bundled / name
    (user / name).write_bytes(b"")
    assert tables.find_table("Nitrogen", "HEOS") == user / name


def test_no_release_yet(monkeypatch):
    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/releases/latest"):
            raise update.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return io.BytesIO(b"{}")  # the repository itself is readable

    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)
    assert update.latest_release("") is None
