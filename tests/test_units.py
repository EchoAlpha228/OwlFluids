import pytest

from owlfluids import units


@pytest.mark.parametrize("text,dim,si", [
    ("3000 psi", "pressure", 3000 * 6894.757293168),
    ("1 atm", "pressure", 101325.0),
    ("-196 C", "temperature", 77.15),
    ("70 F", "temperature", 294.261),
    ("90 K", "temperature", 90.0),
    ("0.5 in", "length", 0.0127),
    ("49 L", "volume", 0.049),
    ("150 ms", "time", 0.15),
    ("60 mm2", "area", 60e-6),
    (12.5, "pressure", 12.5),
])
def test_parse(text, dim, si):
    assert units.parse(text, dim) == pytest.approx(si, rel=1e-5)


def test_psig_adds_atmosphere():
    assert units.parse("0 psig", "pressure") == pytest.approx(101325.0)


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError):
        units.parse("5 m", "pressure")
