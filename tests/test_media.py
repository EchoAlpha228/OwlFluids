"""What the editor colours lines with before a run (no Qt)."""
from owlfluids.app.media import edit_media
from owlfluids.examples import EXAMPLES


def test_media_spread_from_tanks_through_parts():
    m = edit_media(EXAMPLES["lox-feed"]())
    assert m["TK.bottom"] == ("oxygen", False)  # liquid side of the run tank
    assert m["TK.top"] == ("nitrogen", True)  # its pressurant
    assert m["V-M.out"] == ("oxygen", False)  # carried through the main valve
    assert m["REG.out"] == ("nitrogen", True)
    assert m["AMB.port"] == ("oxygen", False)  # a vent takes what flows into it
