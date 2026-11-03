"""Synnax driver logic against an in-memory stand-in for the Synnax client."""
import threading
import time
from types import SimpleNamespace

import pytest

from owlfluids.drivers.synnax import SynnaxConfig, SynnaxDriver, SynnaxLink
from owlfluids.examples import lox_feed
from owlfluids.sim.realtime import Runner
from owlfluids.sim.simulation import Simulation


class FakeLink:
    instances = []

    def __init__(self, cfg):
        self.channels = {}
        self.written = []
        self.inbox = []
        self.lock = threading.Lock()
        FakeLink.instances.append(self)

    def ensure_channels(self, index, data, commands, create):
        self.channels = {"index": index, "data": data, "commands": commands}
        return [index]

    def open(self, write, read):
        self.write_channels, self.read_channels = write, read

    def write(self, values, indexes):
        with self.lock:
            self.written.append(dict(values))

    def read(self, timeout):
        with self.lock:
            if self.inbox:
                return self.inbox.pop(0)
        time.sleep(timeout / 10)
        return {}

    def close(self):
        self.closed = True


def test_default_channel_names_and_units():
    sim = Simulation(lox_feed())
    sensors, valves = SynnaxConfig().resolve(sim)
    assert sensors["PT-TK"] == "pt_tk"
    assert valves["V-M"] == ("v_m_cmd", "v_m_state")
    with pytest.raises(KeyError):
        SynnaxConfig(sensors={"NOPE": "x"}).resolve(sim)
    with pytest.raises(ValueError):
        SynnaxConfig.from_dict({"hots": "typo"})


def test_modes_pick_valve_channels():
    sim = Simulation(lox_feed())
    spec = {"V-M": {"cmd": "mpv_cmd", "state": "mpv_state"}, "V-P": {"read": "a", "write": "b"}}
    _, hootl = SynnaxConfig(valves=spec).resolve(sim)
    assert hootl == {"V-M": ("mpv_cmd", "mpv_state"), "V-P": ("a", "b")}  # read/write still override
    _, hitl = SynnaxConfig.from_dict({"mode": "HITL", "valves": spec}).resolve(sim)
    assert hitl == {"V-M": ("mpv_state", None), "V-P": ("a", None)}  # hardware owns the states
    _, hitl = SynnaxConfig(mode="hitl").resolve(sim)
    assert hitl["V-M"] == ("v_m_state", None)
    with pytest.raises(ValueError):
        SynnaxConfig(mode="hils")
    with pytest.raises(ValueError):
        SynnaxConfig(valves={"V-M": {"command": "x"}})
    with pytest.raises(ValueError):  # a state channel the driver would both write and read
        SynnaxDriver(Runner(sim, rate=0), SynnaxConfig(valves={"V-M": {"cmd": "x", "state": "x"}}),
                     link_factory=FakeLink)


def test_hitl_reads_hardware_states_and_writes_none():
    runner = Runner(Simulation(lox_feed()), rate=0)
    drv = SynnaxDriver(runner, SynnaxConfig(mode="hitl", rate=50), link_factory=FakeLink)
    link = FakeLink.instances[-1]
    assert link.read_channels == ["v_p_state", "v_m_state"]
    assert not any(c.endswith(("_state", "_cmd")) for c in link.write_channels)
    assert "pt_tk" in link.write_channels
    with link.lock:
        link.inbox.append({"v_m_state": 1.0})
    drv.start()  # runner not started: commands apply at once
    try:
        deadline = time.perf_counter() + 2.0
        while runner.sim.valve_command("V-M") != 1.0 and time.perf_counter() < deadline:
            time.sleep(0.01)
    finally:
        drv.stop()
    assert runner.sim.valve_command("V-M") == 1.0
    assert not any("v_m_state" in w for w in link.written)


def test_hootl_fakes_states_from_command_or_limit_switches():
    runner = Runner(Simulation(lox_feed()), rate=0)
    frame = runner.frame
    echo = SynnaxDriver(runner, SynnaxConfig(), link_factory=FakeLink)
    frame.valves["V-M"] = 0.0
    frame.commands["V-M"] = 1.0
    assert echo.values(frame)["v_m_state"] == 1.0  # a DO task's state echoes the command at once
    switches = SynnaxDriver(runner, SynnaxConfig(valve_state="position"), link_factory=FakeLink)
    states = []
    for pos in (0.0, 0.5, 0.98, 1.0, 0.5, 0.02, 0.0):  # open stroke, then close stroke
        frame.valves["V-M"] = pos
        states.append(switches.values(frame)["v_m_state"])
    assert states == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]  # held mid-stroke, flips only at the ends
    with pytest.raises(ValueError):
        SynnaxConfig(valve_state="switch")


def test_hitl_position_states_skip_the_stroke():
    runner = Runner(Simulation(lox_feed()), rate=0)
    drv = SynnaxDriver(runner, SynnaxConfig(mode="hitl", valve_state="position"), link_factory=FakeLink)
    drv.apply("V-M", 1.0)  # runner not started: applied at once
    assert runner.sim.valve_position("V-M") == 1.0 and runner.sim.valve_command("V-M") == 1.0
    runner = Runner(Simulation(lox_feed()), rate=0)
    cmd = SynnaxDriver(runner, SynnaxConfig(mode="hitl"), link_factory=FakeLink)
    cmd.apply("V-M", 1.0)  # a command echo: the sim strokes the valve
    assert runner.sim.valve_command("V-M") == 1.0 and runner.sim.valve_position("V-M") == 0.0


class FakeClient:
    """Just enough of synnax.Synnax.channels for SynnaxLink.ensure_channels."""

    def __init__(self, channels):
        self.store = {c.name: c for c in channels}
        self.channels = self
        self.created = []

    def retrieve(self, names):
        return [c for c in self.store.values() if c.name in names or c.key in names]

    def create(self, name, is_index=False, index=0, virtual=False, data_type=None):
        c = SimpleNamespace(name=name, key=100 + len(self.store), is_index=is_index, index=index, virtual=virtual)
        self.store[name] = c
        self.created.append(name)
        return c


def _channel(name, key, index=0, is_index=False, virtual=False):
    return SimpleNamespace(name=name, key=key, index=index, is_index=is_index, virtual=virtual)


def test_link_keeps_existing_indexes_and_creates_missing_on_its_own():
    link = object.__new__(SynnaxLink)
    link.sy = SimpleNamespace(DataType=SimpleNamespace(TIMESTAMP="ts", FLOAT32="f32"))
    link.client = FakeClient([_channel("ai_time", 1, is_index=True), _channel("pt_tk", 2, index=1),
                              _channel("do_time", 3, is_index=True), _channel("mpv_state", 4, index=3),
                              _channel("mpv_cmd", 5, virtual=True)])
    # every channel already on the stand: write its indexes, create nothing
    assert link.ensure_channels("owl_time", ["pt_tk", "mpv_state"], ["mpv_cmd"], False) == ["ai_time", "do_time"]
    assert link.client.created == []
    with pytest.raises(RuntimeError, match="owl_time, pt_inj"):
        link.ensure_channels("owl_time", ["pt_tk", "pt_inj"], [], False)
    indexes = link.ensure_channels("owl_time", ["pt_tk", "pt_inj"], ["press_cmd"], True)
    assert indexes == ["ai_time", "owl_time"]
    assert link.client.created == ["owl_time", "pt_inj", "press_cmd"]
    assert link.client.store["pt_inj"].index == link.client.store["owl_time"].key
    assert link.client.store["press_cmd"].virtual
    with pytest.raises(RuntimeError, match="index channel"):
        link.ensure_channels("owl_time", ["ai_time"], [], True)


def test_link_refuses_an_index_the_hardware_also_writes():
    link = object.__new__(SynnaxLink)
    link.sy = SimpleNamespace(DataType=SimpleNamespace(TIMESTAMP="ts", FLOAT32="f32"))
    # one DAQ task stamps both the sensors and the valve states on daq_time
    link.client = FakeClient([_channel("daq_time", 1, is_index=True), _channel("pt_tk", 2, index=1),
                              _channel("mpv_state", 3, index=1)])
    with pytest.raises(RuntimeError, match="daq_time.*mpv_state"):
        link.ensure_channels("owl_time", ["pt_tk"], ["mpv_state"], True)


@pytest.mark.slow
def test_driver_publishes_sensors_and_applies_valve_commands():
    runner = Runner(Simulation(lox_feed()), rate=0)  # as fast as possible
    cfg = SynnaxConfig(rate=200)
    drv = SynnaxDriver(runner, cfg, link_factory=FakeLink)
    link = FakeLink.instances[-1]
    assert "pt_tk" in link.channels["data"] and "v_m_state" in link.channels["data"]
    assert link.channels["commands"] == ["v_p_cmd", "v_m_cmd"]
    runner.start()
    drv.start()
    try:
        runner.wait(0.2)
        with link.lock:
            link.inbox.append({"v_m_cmd": 1.0})
        runner.wait_until(lambda f: f.valves["V-M"] > 0.99, timeout=5.0)
        runner.wait(0.5)
        time.sleep(0.1)
    finally:
        drv.stop()
        runner.stop()
    assert drv.error is None
    assert drv.commands == 1
    last = link.written[-1]
    assert last["v_m_state"] == pytest.approx(1.0)
    assert 100 < last["pt_tk"] < 400  # psi, the sensor's display unit
    assert last["fm_lox"] > 0.5


class StoredLink(FakeLink):
    """A link whose valve channel already holds a value, and whose writer fails on close."""

    def latest(self, names):
        return {n: 1.0 for n in names if n == "v_m_cmd"}

    def close(self):
        raise RuntimeError("writer lost its connection")


def test_driver_syncs_valves_at_connect_and_stops_cleanly():
    runner = Runner(Simulation(lox_feed()), rate=0)
    vp = runner.sim.valve_command("V-P")
    assert runner.sim.valve_command("V-M") == 0.0
    drv = SynnaxDriver(runner, SynnaxConfig(rate=50), link_factory=StoredLink)
    drv.start()  # runner not started: commands apply at once
    assert runner.sim.valve_command("V-M") == 1.0
    assert runner.sim.valve_command("V-P") == vp  # no stored value: left alone
    drv.stop()  # must not raise
    assert isinstance(drv.error, RuntimeError)


def test_version_mismatch():
    from owlfluids.drivers.synnax import version_mismatch

    assert version_mismatch("0.60.0", "0.60.3") is None
    assert version_mismatch("0.60.0", "v0.60.1") is None
    assert "upgrade the Core" in version_mismatch("0.60.0", "0.59.2")
    assert "upgrade the OwlFluids build" in version_mismatch("0.59.0", "0.60.0")
    assert "did not report" in version_mismatch("0.60.0", "")
    assert version_mismatch("", "0.60.0") is None  # client version unknown: nothing to compare
    assert version_mismatch("0.60.0", "0.0.0") is None  # development Core


def test_check_versions_modes():
    from owlfluids.drivers.synnax import check_versions

    def client(core):
        state = SimpleNamespace(client_version="0.60.0", node_version=core)
        return SimpleNamespace(connectivity=SimpleNamespace(check=lambda: state))

    assert check_versions(client("0.60.2"), "warn") is None
    assert "0.58.2" in check_versions(client("0.58.2"), "warn")  # warns, still connects
    with pytest.raises(RuntimeError, match="strict"):
        check_versions(client("0.58.2"), "strict")
    assert check_versions(client("0.58.2"), "off") is None
    assert check_versions(SimpleNamespace(), "strict") is None  # an old client can't report versions

    def broken():
        raise AttributeError("no checker in this client")

    assert check_versions(SimpleNamespace(connectivity=SimpleNamespace(check=broken)), "strict") is None
    failed = SimpleNamespace(status="failed", message="connection refused", node_version="")
    with pytest.raises(RuntimeError, match="cannot reach the Synnax Core: connection refused"):
        check_versions(SimpleNamespace(connectivity=SimpleNamespace(check=lambda: failed)), "warn")
    ok = SimpleNamespace(status="connected", client_server_compatible=True, client_version="0.60.0",
                         node_version="0.0.0")
    assert check_versions(SimpleNamespace(connectivity=SimpleNamespace(check=lambda: ok)), "strict") is None
    assert SynnaxConfig.from_dict({"check_version": False}).check_version == "off"
    assert SynnaxConfig.from_dict({"check_version": True}).check_version == "warn"
    with pytest.raises(ValueError, match="check_version"):
        SynnaxConfig.from_dict({"check_version": "maybe"})


def test_link_warnings_reach_the_runner():
    class WarningLink(FakeLink):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.warnings = ["Synnax: versions differ"]

    runner = Runner(Simulation(lox_feed()))
    SynnaxDriver(runner, SynnaxConfig(), link_factory=WarningLink)
    assert "Synnax: versions differ" in runner.warnings


def test_exclude_leaves_sensors_and_valves_out():
    sim = Simulation(lox_feed())
    sensors, valves = SynnaxConfig(exclude=["PT-TK", "V-M"]).resolve(sim)
    assert "PT-TK" not in sensors and "PT-INJ" in sensors
    assert "V-M" not in valves and "V-P" in valves
    sensors, _ = SynnaxConfig(sensors={"PT-TK": "a", "PT-INJ": "b"}, exclude=["PT-INJ"]).resolve(sim)
    assert sensors == {"PT-TK": "a"}


def test_split_keeps_the_login_out_of_the_system_section():
    cfg = SynnaxConfig(host="stand", password="pw", mode="hitl", sensors={"PT-TK": "pt"})
    conn, rest = cfg.split()
    assert conn == {"host": "stand", "port": 9090, "username": "synnax", "password": "pw", "secure": False,
                    "desktop": False}
    assert rest == {"mode": "hitl", "sensors": {"PT-TK": "pt"}}
    assert SynnaxConfig.from_dict({**conn, **rest}) == cfg


def test_system_file_keeps_its_synnax_section_and_renames_follow(tmp_path):
    from owlfluids.io.system import System

    s = lox_feed()
    s.synnax = {"mode": "hitl", "sensors": {"PT-TK": "pt_tank"}, "valves": {"V-M": {"cmd": "m", "state": "ms"}},
                "exclude": ["PT-INJ"]}
    s.renamed("PT-TK", "PT-T1")
    s.renamed("V-M", "V-MAIN")
    s.renamed("PT-INJ", "PT-I")
    assert s.synnax["sensors"] == {"PT-T1": "pt_tank"}
    assert s.synnax["valves"] == {"V-MAIN": {"cmd": "m", "state": "ms"}}
    assert s.synnax["exclude"] == ["PT-I"]
    s.save(tmp_path / "s.yaml")
    assert System.load(tmp_path / "s.yaml").synnax == s.synnax


class ProbeLink:
    def __init__(self, cfg):
        self.warnings = ["Synnax: versions differ"]
        self.closed = False
        ProbeLink.last = self

    def existing(self, names):
        return [n for n in names if n.startswith("pt")]

    def core_version(self):
        return "0.60.1"

    def disconnect(self):
        self.closed = True


def test_probe_reports_channels_and_disconnects():
    from owlfluids.drivers.synnax import probe

    res = probe(SynnaxConfig(), ["pt_tk", "v_m_cmd", "pt_tk"], ProbeLink)
    assert res.found == ["pt_tk"] and res.missing == ["v_m_cmd"]
    assert res.core == "0.60.1" and res.warnings == ["Synnax: versions differ"]
    assert ProbeLink.last.closed
    assert "1 of 2 channels exist, the rest will be created" in res.summary(True)
    assert "missing: v_m_cmd" in res.summary(False)


@pytest.fixture
def old_core():
    """A stand-in for a Synnax Core older than 0.60: no session login, an open connectivity check."""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.rstrip("/").endswith("/connectivity/check"):
                body, code, kind = json.dumps({"node_version": "0.59.2"}).encode(), 200, "application/json"
            else:
                body, code, kind = b"Method Not Allowed", 405, "text/plain"
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_connect_error_names_the_core_version_and_the_fix(old_core):
    from owlfluids.drivers.synnax import core_version, explain_connect_error

    assert core_version("127.0.0.1", old_core) == "0.59.2"
    cfg = SynnaxConfig(host="127.0.0.1", port=old_core)
    msg = explain_connect_error(ValueError("failed to decode error response: b'Method Not Allowed'"), cfg, "0.60.1")
    assert "Synnax Core 0.59.2" in msg and "client 0.60.1" in msg
    assert "Upgrade the Core to 0.60" in msg and "synnax_version 0.59.*" in msg
    newer = explain_connect_error(ValueError("x"), cfg, "0.58.0")
    assert "Update OwlFluids" in newer


def test_connect_error_without_a_core_keeps_the_clients_message():
    from owlfluids.drivers.synnax import explain_connect_error

    cfg = SynnaxConfig(host="127.0.0.1", port=1)  # nothing listens there
    assert explain_connect_error(ConnectionError("refused"), cfg, "0.60.1") == "refused"
    msg = explain_connect_error(ValueError("Method Not Allowed"), cfg, "0.60.1")
    assert "may not be a Synnax Core" in msg and "older than 0.60" in msg


def test_real_client_against_an_old_core_says_why(old_core):
    pytest.importorskip("synnax")
    from owlfluids.drivers.synnax import client_version

    if not client_version().startswith("0.60"):
        pytest.skip("checks the 0.60 client's session login")
    with pytest.raises(RuntimeError, match="Synnax Core 0.59.2"):
        SynnaxLink(SynnaxConfig(host="127.0.0.1", port=old_core))


def _desktop_file(port, key="syk_test_key"):
    """Write a ~/.synnax/desktop.json like Synnax Desktop's for a Core on this port."""
    import json

    from owlfluids.drivers.synnax import desktop_file

    desktop_file().write_text(json.dumps({"host": "127.0.0.1", "port": port, "secure": False, "api_key": key}))


def test_desktop_reads_its_connection_file_at_each_connect(old_core):
    from owlfluids.drivers.synnax import desktop_options

    with pytest.raises(RuntimeError, match="0.60.1 and older don't"):
        desktop_options()
    _desktop_file(old_core)
    cfg = SynnaxConfig(desktop=True, host="ignored", port=1, secure=True)
    assert cfg.address() == ("127.0.0.1", old_core, False, "syk_test_key")
    _desktop_file(1)  # Desktop closed: its port answers nothing
    with pytest.raises(RuntimeError, match="isn't running"):
        cfg.address()
    assert SynnaxConfig(host="h", port=5).address() == ("h", 5, False, "")


def test_desktop_key_refused_says_to_reopen_desktop(old_core):
    from owlfluids.drivers.synnax import explain_connect_error

    msg = explain_connect_error(ValueError("AuthError: invalid credentials"), SynnaxConfig(desktop=True), "0.60.1",
                                "127.0.0.1", old_core)
    assert "API key" in msg and "Open Synnax Desktop" in msg


@pytest.fixture
def desktop_core():
    """A stand-in for Synnax Desktop's Core: logs in only with its API key, as Synnax main does."""
    import json
    import uuid
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            path = self.path.rstrip("/")
            seen.append((path, body))
            code, out = 200, {}
            if path.endswith("/connectivity/check"):
                out = {"cluster_key": "k", "node_version": "0.60.1", "node_time": 0}
            elif path.endswith("/auth/session/create"):
                if body.get("proof") != {"method": "api_key", "key": "syk_test_key"}:
                    code, out = 401, {"type": "sy.auth.invalid_credentials", "data": "invalid credentials"}
                else:
                    out = {"token": "tok", "session": {"key": str(uuid.uuid4()),
                                                       "subject": {"type": "user", "key": str(uuid.uuid4())},
                                                       "credential": str(uuid.uuid4()), "last_active": 0}}
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], seen
    server.shutdown()


def test_real_client_logs_in_to_synnax_desktop_with_its_api_key(desktop_core):
    pytest.importorskip("synnax")
    from owlfluids.drivers.synnax import client_version

    if tuple(int(x) for x in client_version().split(".")[:2]) < (0, 60):
        pytest.skip("API keys and session logins came with the 0.60 client")
    port, seen = desktop_core
    _desktop_file(port)
    link = SynnaxLink(SynnaxConfig(desktop=True))
    try:
        logins = [b for p, b in seen if p.endswith("/auth/session/create")]
        assert logins == [{"proof": {"method": "api_key", "key": "syk_test_key"}}]
        assert link.client.auth.token == "tok"
    finally:
        link.disconnect()
    _desktop_file(port, key="stale")
    with pytest.raises(RuntimeError, match="API key"):
        SynnaxLink(SynnaxConfig(desktop=True))
