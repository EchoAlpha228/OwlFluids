"""Command line: ``owlfluids app|run|console|check|bench|example|tables``."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from . import units


def _load(path: str):
    from .examples import EXAMPLES
    from .io.system import System

    if path in EXAMPLES:
        return EXAMPLES[path]()
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
        if args.duration:
            runner.wait_until(lambda f: f.t >= args.duration - 1e-9)
    except KeyboardInterrupt:
        pass
    finally:
        if driver:
            driver.stop()
        runner.stop()
        if log:
            log.close()
    st = sim.solver_stats()
    print(f"done: t={sim.t:.3f}s, {st['steps']} steps, {runner.slips} clock slips, {st['forced_steps']} forced steps")
    return 1 if runner.error else 0


def _start_driver(args, runner):
    if not getattr(args, "synnax", None):
        return None
    from .drivers.synnax import SynnaxConfig, SynnaxDriver

    cfg = SynnaxConfig.load(args.synnax)
    driver = SynnaxDriver(runner, cfg)
    print(f"Synnax: writing {len(driver.sensors)} sensors at {cfg.rate:g} Hz to {cfg.host}:{cfg.port}, "
          f"reading {len(driver.valves)} valve channels")
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
                    runner.set_rate(float(rest[0]))
                elif cmd == "pause":
                    runner.pause()
                elif cmd == "resume":
                    runner.resume()
                elif cmd == "reset":
                    runner.reset()
                elif cmd == "log":
                    log = open_logger(rest[0], runner)
                elif cmd == "stoplog" and log:
                    log.close()
                    log = None
                else:
                    print(CONSOLE_HELP)
            except (KeyError, ValueError, IndexError) as e:
                print("error:", e)
            if runner.error:
                print("simulation error:", runner.error)
                break
    finally:
        if driver:
            driver.stop()
        runner.stop()
        if log:
            log.close()
    return 0


def cmd_bench(args) -> int:
    from .examples.generate import engine_test_stand
    from .sim.simulation import Simulation

    t0 = time.perf_counter()
    sim = Simulation(engine_test_stand())
    print(f"build + compile: {time.perf_counter() - t0:.1f} s; {len(sim.system.components)} components, "
          f"{sim.net.n_states} states")
    sim.run(0.1)
    phases = [("pressurize", ["V-OXP", "V-FUP"], [], 1.0), ("lock up", [], ["V-OXV", "V-FUV"], 1.0),
              ("main valves open", ["V-OXM", "V-FUM"], [], 0.5), ("steady flow", [], [], 1.5),
              ("shutdown", [], ["V-OXM", "V-FUM"], 1.0)]
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
    s = EXAMPLES[args.name]()
    out = args.output or f"{args.name}.yaml"
    s.save(out)
    print(f"wrote {out}")
    return 0


def cmd_app(args) -> int:
    from .app.main import main as app_main

    return app_main(args.system)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="owlfluids", description="Real-time 1-D fluid network simulator")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("app", help="open the desktop app")
    p.add_argument("system", nargs="?", help="system file (.yaml) or example name")
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("run", help="run a system headless")
    p.add_argument("system")
    p.add_argument("--duration", type=float, default=0.0, help="simulated seconds to run")
    p.add_argument("--script", help="Python file with main(runner) that commands valves")
    p.add_argument("--rate", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    p.add_argument("--log", help="write a .csv or .h5 log")
    p.add_argument("--log-interval", type=float, default=0.01)
    p.add_argument("--print-every", type=float, default=1.0, help="print sensors every N simulated s (0 = never)")
    p.add_argument("--synnax", help="YAML file with a synnax: section; runs hardware-in-the-loop")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("console", help="run in real time with typed valve commands")
    p.add_argument("system")
    p.add_argument("--synnax", help="YAML file with a synnax: section; runs hardware-in-the-loop")
    p.set_defaults(fn=cmd_console)

    p = sub.add_parser("check", help="build a system and print a summary")
    p.add_argument("system")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("bench", help="time the 81-component test stand")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("example", help="list examples or write one to a file")
    p.add_argument("name", nargs="?")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_example)

    p = sub.add_parser("tables", help="prebuild property tables (HEOS or REFPROP)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: __import__("owlfluids.fluids.tables", fromlist=["main"]).main(a.rest) or 0)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        return cmd_app(argparse.Namespace(system=None))
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
