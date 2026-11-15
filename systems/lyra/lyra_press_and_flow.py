"""Press and flow the Lyra feed system:
`owlfluids run systems/lyra/lyra-feed.yaml --script systems/lyra/lyra_press_and_flow.py --log lyra.csv`

Every timing here is a placeholder, like the numbers in the system file.
"""

PSI = 6894.757


def main(rt):
    rt.open("NI-F", "NTI-O")
    rt.wait(2.0)
    rt.open("FPV", "OPV")
    rt.wait_until(lambda f: f.sensors["FTPT"] > 650 * PSI, timeout=15.0)
    rt.open("IFS", "IOS")
    rt.wait(0.5)
    rt.open("FMV", "OMV")
    rt.wait(4.0)
    rt.close("FMV", "OMV", "IFS", "IOS", "FPV", "OPV")
    rt.open("EPS")  # purge the feeds into the engine
    rt.wait(2.0)
    rt.close("EPS")
    rt.open("FTVV", "OTVV")
    rt.wait(10.0)
