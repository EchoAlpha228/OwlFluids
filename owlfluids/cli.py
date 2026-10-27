"""Command line: ``owlfluids app|run|console|check|bench|example|update|tables``."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from . import units
from .core.network import BuildError


def _load(path: str):
    from .examples import EXAMPLES
    from .io.system import System

    if path in EXAMPLES:
        return EXAMPLES[path]()
    if not Path(path).exists():
        raise SystemExit(f"error: {path} not found; give a .yaml file or one of: {', '.join(EXAMPLES)}")
    return System.load(path)


def _fmt(sim, frame) -> str:
    parts = [f"t={frame.t:8.3f}s"]
    for s in sim.system.sensors:
        unit = s.units or units.DEFAULT_UNITS.get(s.type, "")
        parts.append(f"{s.id}={units.convert(frame.sensors[s.id], unit):.4g}{(' ' + unit) if unit else ''}")
    return "  ".join(parts)


def cmd_check(args) -> int:
    from .core.network import build_network

    net = build_network(_load(args.system))
    print(f"{net.system.name}: {len(net.system.components)} components, {net.NI.shape[0]} nodes, "
          f"{net.LI.shape[0]} flow elements, {net.n_states} states, band {net.kl}, {net.n_colors} colours")
    print("fluids:", ", ".join(f"{n} ({m})" for n, m in zip(net.fluids.names, net.fluids.models, strict=True)))
    print("valves:", ", ".join(net.valves) or "none")
    for w in net.warnings:
        print("warning:", w)
    return 0


def cmd_run(args) -> int:
    from .io.logger import open_logger
    from .sim.realtime import Runner
    from .sim.simulation import Simulation

    sim = Simulation(_load(args.system))
    runner = Runner(sim, rate=args.rate)
    log = open_logger(args.log, runner, args.log_interval) if args.log else None
    last = [-1e9]

    def show(frame):
        if args.print_every and frame.t - last[0] >= args.print_every - 1e-9:
            last[0] = frame.t
            print(_fmt(sim, frame), flush=True)

    runner.add_listener(show)
    driver = _start_driver(args, runner)
    runner.start()
    if driver:
        driver.start()
    try:
        if args.script:
            ns: dict = {"__name__": "__owlfluids__", "units": units}
            code = Path(args.script).read_text(encoding="utf-8")
            exec(compile(code, args.script, "exec"), ns)
            if "main" in ns:
                ns["main"](runner)
        # with a duration, run to it; without one (and no script), run until Ctrl-C
        warned = 0
        while args.duration or not args.script:
            if args.duration and runner.frame.t >= args.duration - 1e-9:
                break
            if runner.error or (driver and driver.error) or not runner.running:
                break
            warned = _print_warnings(runner, warned)
            nap = 0.05
            if args.duration:  # don't overshoot a short run by a whole poll
                left = args.duration - runner.frame.t
                nap = min(nap, left / args.rate if args.rate > 0 else 0.002)
            time.sleep(max(nap, 0.001))
    except KeyboardInterrupt:
        pass
    finally:
        if driver:
            driver.stop()
        runner.stop()
        if log:
            log.close()
    _print_warnings(runner, 0)
    st = sim.solver_stats()
    print(f"done: t={sim.t:.3f}s, {st['steps']} steps, {runner.slips} clock slips, {st['forced_steps']} forced steps")
    return _report_errors(runner, driver)


def _print_warnings(runner, shown: int) -> int:
    """Print runner warnings not shown yet; returns how many there are now."""
    for w in runner.warnings[shown:]:
        print("warning:", w, file=sys.stderr)
    return len(runner.warnings)


def _report_errors(runner, driver) -> int:
    code = 0
    if runner.error:
        print("simulation error:", repr(runner.error), file=sys.stderr)
        code = 1
    if driver and driver.error:
        print("Synnax driver error:", repr(driver.error), file=sys.stderr)
        code = 1
    return code


def _start_driver(args, runner):
    if not getattr(args, "synnax", None):
        return None
    from .drivers.synnax import SynnaxConfig, SynnaxDriver

    cfg = SynnaxConfig.load(args.synnax)
    if getattr(args, "synnax_mode", None):
        cfg = SynnaxConfig.from_dict({**cfg.__dict__, "mode": args.synnax_mode})
    driver = SynnaxDriver(runner, cfg)
    what = "faking valve states, reading valve commands" if cfg.mode == "hootl" else "reading valve states"
    print(f"Synnax {cfg.mode.upper()}: writing {len(driver.sensors)} sensors at {cfg.rate:g} Hz to "
          f"{cfg.host}:{cfg.port}, {what} for {len(driver.valves)} valves")
    return driver


CONSOLE_HELP = """commands:
  open V [V ...]      close V [V ...]      set V 0.4      (valve position 0-1)
  status              valves               rate 1|0.5|0 (0 = as fast as possible)
  pause  resume  reset  log FILE.csv|FILE.h5  stoplog   quit"""


def cmd_console(args) -> int:
    """Manual valve control in a terminal while the simulation runs in real time."""
    from .io.logger import open_logger
    from .sim.realtime import Runner
    from .sim.simulation import Simulation

    sim = Simulation(_load(args.system))
    runner = Runner(sim)
    driver = _start_driver(args, runner)
    runner.start()
    if driver:
        driver.start()
    log = None
    warned = 0

    def stop_log():
        nonlocal log
        if log:
            old, log = log, None
            runner.remove_listener(old)
            runner.submit(lambda sim: old.close())  # on the runner thread, never mid-write

    print(f"{sim.system.name}: running in real time. Valves: {', '.join(sim.valves()) or 'none'}")
    print(CONSOLE_HELP)
    try:
        while True:
            try:
                line = input("owl> ").strip()
            except EOFError:
                break
            if not line:
                continue
            words = shlex.split(line)
            cmd, rest = words[0].lower(), words[1:]
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "open":
                    runner.open(*rest)
                elif cmd == "close":
                    runner.close(*rest)
                elif cmd == "set":
                    if len(rest) < 2:
                        raise IndexError
                    runner.command(rest[0], float(rest[1]))
                elif cmd in ("status", "s"):
                    print(_fmt(sim, runner.frame))
                    print(f"lag {runner.frame.lag * 1000:.1f} ms, load {runner.frame.load * 100:.0f}%, "
                          f"slips {runner.slips}")
                elif cmd == "valves":
                    f = runner.frame
                    for v in sim.valves():
                        print(f"  {v}: cmd {f.commands[v]:.2f} pos {f.valves[v]:.2f}")
                elif cmd == "rate":
                    if not rest:
                        raise IndexError
                    runner.set_rate(float(rest[0]))
                elif cmd == "pause":
                    runner.pause()
                elif cmd == "resume":
                    runner.resume()
                elif cmd == "reset":
                    runner.reset()
                elif cmd == "log":
                    if not rest:
                        raise IndexError
                    stop_log()
                    log = open_logger(rest[0], runner)
                elif cmd == "stoplog":
                    stop_log()
                elif cmd in ("help", "h", "?"):
                    print(CONSOLE_HELP)
                else:
                    print(f"unknown command {cmd!r}")
                    print(CONSOLE_HELP)
            except IndexError:
                print("error: missing argument (type help for usage)")
            except (KeyError, ValueError) as e:
                print("error:", e)
            warned = _print_warnings(runner, warned)
            if runner.error or (driver and driver.error):
                break
    finally:
        if driver:
            driver.stop()
        runner.stop()
        if log:
            log.close()
    return _report_errors(runner, driver)


def cmd_bench(args) -> int:
    from .examples.generate import engine_test_stand
    from .sim.simulation import Simulation

    t0 = time.perf_counter()
    sim = Simulation(engine_test_stand())
    print(f"build + compile: {time.perf_counter() - t0:.1f} s; {len(sim.system.components)} components, "
          f"{sim.net.n_states} states; step budget {sim.step_budget:.0f} evaluations "
          f"({sim.rhs_seconds * 1e6:.1f} us each)")
    sim.run(0.1)
    phases = [("pressurize", ["V-OXP", "V-FUP"], [], 1.0), ("lock up", [], ["V-OXV", "V-FUV"], 1.0),
              ("main valves open", ["V-OXM", "V-FUM"], [], 0.5), ("steady flow", [], [], 1.5),
              ("shutdown", [], ["V-OXM", "V-FUM"], 1.0), ("purge", ["V-P1", "V-P2", "V-P3", "V-P4"], [], 1.5)]
    worst = 0.0
    for name, op, cl, dur in phases:
        for v in op:
            sim.open(v)
        for v in cl:
            sim.close(v)
        t0 = time.perf_counter()
        for _ in range(int(dur / 0.01)):
            c0 = time.perf_counter()
            sim.run(0.01)
            worst = max(worst, time.perf_counter() - c0)
        el = time.perf_counter() - t0
        print(f"{name:18s} {el / dur * 1000:6.1f} ms per simulated s ({el / (dur / sim.dt) * 1e6:5.1f} us/step)")
    print(f"worst 10 ms chunk: {worst * 1000:.2f} ms;", sim.solver_stats())
    return 0


def cmd_example(args) -> int:
    from .examples import EXAMPLES

    if not args.name:
        print("examples:", ", ".join(EXAMPLES))
        return 0
    if args.name not in EXAMPLES:
        print(f"error: no example {args.name!r}; examples: {', '.join(EXAMPLES)}", file=sys.stderr)
        return 2
    s = EXAMPLES[args.name]()
    out = args.output or f"{args.name}.yaml"
    s.save(out)
    print(f"wrote {out}")
    return 0


def cmd_app(args) -> int:
    from .app.main import main as app_main

    return app_main(args.system)


def cmd_update(args) -> int:
    from . import __version__, update

    if args.token is not None:
        cfg = update.load_config()
        cfg["token"] = args.token.strip()
        update.save_config(cfg)
        print(f"token saved to {update.CONFIG}")
    try:
        rel = update.latest_release()
    except update.UpdateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not rel or not rel.newer:
        print(f"OwlFluids {__version__} is up to date")
        return 0
    print(f"OwlFluids {rel.version} is available (you have {__version__}): {rel.url}")
    if args.check:
        return 0
    if not update.frozen():
        print("this is a Python install: update it with git pull and pip install -e ., or use the installer")
        return 0
    try:
        path = update.download(rel, progress=lambda done, total: print(
            f"\rdownloading {done / 1e6:.0f} of {total / 1e6:.0f} MB", end="", flush=True))
        print()
        update.run_installer(path)
    except update.UpdateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("installer started; it will finish in the background")
    return 0


SYSTEM_HELP = "system .yaml file or example name (see `owlfluids example`)"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["tables"]:  # the table builder has its own options
        from .fluids.tables import main as tables_main

        return tables_main(argv[1:]) or 0
    from . import __version__

    ap = argparse.ArgumentParser(prog="owlfluids", description="Real-time 1-D fluid network simulator")
    ap.add_argument("--version", action="version", version=f"OwlFluids {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("app", help="open the desktop app")
    p.add_argument("system", nargs="?", help=SYSTEM_HELP)
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("run", help="run a system headless")
    p.add_argument("system", help=SYSTEM_HELP)
    p.add_argument("--duration", type=float, default=0.0, help="simulated seconds to run (0 = until Ctrl-C)")
    p.add_argument("--script", help="Python file with main(runner) that commands valves")
    p.add_argument("--rate", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    p.add_argument("--log", help="write a .csv or .h5 log")
    p.add_argument("--log-interval", type=float, default=0.01, help="seconds between log rows (default 0.01)")
    p.add_argument("--print-every", type=float, default=1.0, help="print sensors every N simulated s (0 = never)")
    p.add_argument("--synnax", help="YAML file with a synnax: section; connects the sim to Synnax")
    p.add_argument("--synnax-mode", choices=["hootl", "hitl"],
                   help="override the file's mode: hootl fakes valve states, hitl reads them from hardware")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("console", help="run in real time with typed valve commands")
    p.add_argument("system", help=SYSTEM_HELP)
    p.add_argument("--synnax", help="YAML file with a synnax: section; connects the sim to Synnax")
    p.add_argument("--synnax-mode", choices=["hootl", "hitl"],
                   help="override the file's mode: hootl fakes valve states, hitl reads them from hardware")
    p.set_defaults(fn=cmd_console)

    p = sub.add_parser("check", help="build a system and print a summary")
    p.add_argument("system", help=SYSTEM_HELP)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("bench", help="time the 60-component test stand")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("example", help="list examples or write one to a file")
    p.add_argument("name", nargs="?", help="example to write (leave out to list them)")
    p.add_argument("-o", "--output", help="file to write (default <name>.yaml)")
    p.set_defaults(fn=cmd_example)

    p = sub.add_parser("update", help="check for a newer release and install it (Windows installer)")
    p.add_argument("--check", action="store_true", help="only say whether there is a newer release")
    p.add_argument("--token", help="save a GitHub token with read access to the repository, then check")
    p.set_defaults(fn=cmd_update)

    sub.add_parser("tables", help="prebuild property tables (see `owlfluids tables --help`)")

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        return cmd_app(argparse.Namespace(system=None))
    try:
        return int(args.fn(args) or 0)
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
