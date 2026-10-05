"""Interpolated real-fluid tables against CoolProp (HEOS) at random states."""
import numpy as np
import pytest

CP = pytest.importorskip("CoolProp.CoolProp")

from owlfluids.fluids import props  # noqa: E402
from owlfluids.fluids.registry import FLUIDS, FluidSet  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("name", ["nitrogen", "oxygen", "methane", "nitrous"])
def test_table_pressure_matches_coolprop(name):
    fs = FluidSet([name], ["table"])
    st = CP.AbstractState("HEOS", FLUIDS[name][0])
    arrs = fs.arrays()[:7]
    rng = np.random.default_rng(0)
    errs = []
    t_low = st.Ttriple() + 3
    while len(errs) < 800:
        p = 10 ** rng.uniform(5, np.log10(250e5))
        t = rng.uniform(t_low, 450)
        try:
            st.update(CP.PT_INPUTS, p, t)
        except ValueError:
            continue
        P, T, *_ = props.state(0, st.rhomass(), st.umass(), *arrs)
        errs.append((abs(P - p) / p, abs(T - t)))
    errs = np.array(errs)
    assert np.percentile(errs[:, 0], 99) < 0.01
    assert errs[:, 0].max() < 0.03
    assert np.percentile(errs[:, 1], 99) < 0.5


def test_ideal_gas_state():
    fs = FluidSet(["nitrogen"], ["ideal"])
    model, par, *rest = fs.arrays()
    R, cv = par[0, 0], par[0, 1]
    P, T, x, g, mu = props.state(0, 10.0, cv * 300.0, model, par, *rest[:5])
    assert T == pytest.approx(300.0)
    assert P == pytest.approx(10.0 * R * 300.0)


def test_liquid_cavitates_smoothly_at_vapor_pressure():
    fs = FluidSet(["ethanol"], ["liquid"])
    arrs = fs.arrays()[:7]
    rho0 = fs.par[0, 3]
    ps = [props.state(0, rho0 * (1 - e), 0.0, *arrs)[0] for e in np.linspace(0, 1e-3, 200)]
    assert min(ps) > 0.99 * fs.par[0, 5]  # never below vapor pressure
    assert np.all(np.diff(ps) <= 0)  # monotonic in density
