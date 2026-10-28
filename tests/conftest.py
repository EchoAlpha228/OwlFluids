import pytest

from owlfluids.io.system import System


@pytest.fixture
def gas_blowdown():
    def make(model="table", cda="5 mm2", p="200 bar", t="300 K", volume="50 L"):
        s = System("blowdown")
        s.settings["fluid_models"] = {"nitrogen": model}
        s.add("TK", "tank", fluid="nitrogen", volume=volume, pressure=p, temperature=t)
        s.add("O1", "orifice", cda=cda)
        s.add("AMB", "boundary", fluid="nitrogen", pressure="1 bar")
        s.connect("TK.port", "O1.in")
        s.connect("O1.out", "AMB.port")
        return s
    return make


@pytest.fixture(autouse=True)
def _app_settings(tmp_path, monkeypatch):
    """Keep the desktop app's preferences (layout, recent files) out of the user's own."""
    monkeypatch.setenv("OWLFLUIDS_SETTINGS", str(tmp_path / "owlfluids.ini"))
