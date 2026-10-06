"""Example sequence for `owlfluids run examples/lox-feed.yaml --script examples/lox_flow_sequence.py`.

``main`` gets the real-time runner: command valves, wait on simulated time or on
conditions, and read sensors (SI units; ``units.convert`` turns them into psi etc.).
"""

PSI = 6894.757


def main(rt):
    rt.wait(0.5)
    rt.open("V-M")
    # hold until injector pressure comes up, at most 2 s
    if not rt.wait_until(lambda f: f.sensors["PT-INJ"] > 100 * PSI, timeout=2.0):
        print("injector pressure never came up; aborting")
        rt.close("V-M")
        return
    rt.wait(2.0)
    rt.close("V-P")  # isolate the pressurant: tank blows down
    rt.wait(1.0)
    rt.close("V-M")
    rt.wait(0.5)
