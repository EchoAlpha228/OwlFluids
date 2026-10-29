"""One tank type: single fluid, saturated, or liquid under a pressurant; old pressurized tanks load as one."""
import pytest
import yaml

from owlfluids.core.network import BuildError, build_network
from owlfluids.examples import lox_feed
from owlfluids.io.system import System
from owlfluids.sim.simulation import Simulation

OLD = """
name: old LOX feed
components:
- {id: TK, type: pressurized_tank, liquid: oxygen, volume: 60 L, pressure: 300 psi,
   liquid_temperature: 90 K, diameter: 0.3 m, at: [0, 0]}
- {id: K-N2, type: tank, fluid: nitrogen, volume: 49 L, pressure: 4000 psi, temperature: 20 C, at: [0, -200]}
- {id: V-P, type: valve, cv: 0.5, normally: open, at: [150, -200]}
- {id: V-M, type: valve, cv: 6, at: [150, 150]}
- {id: AMB, type: boundary, fluid: oxygen, at: [300, 150]}
connections:
- [K-N2.port, V-P.in]
- [V-P.out, TK.gas]
- [TK.liquid, V-M.in]
- [V-M.out, AMB.port]
sensors:
- {id: PT-TK, type: pressure, at: TK.gas}
- {id: TC-TK, type: temperature, at: TK.liquid}
routes:
  'V-P.out -- TK.gas': {h: true, bends: [[60, -200]]}
"""


def test_old_pressurized_tank_files_load_as_tanks_with_a_pressurant():
    s = System.from_dict(yaml.safe_load(OLD))
    tk = s.component("TK")
    assert tk.type == "tank"
    # the old names and defaults: the liquid is the fluid, nitrogen pressurant, 80 % full
    assert tk.params == {"fluid": "oxygen", "volume": "60 L", "pressure": "300 psi", "temperature": "90 K",
                         "diameter": "0.3 m", "pressurant": "nitrogen", "fill": 0.8}
    assert ("V-P.out", "TK.top") in s.connections and ("TK.bottom", "V-M.in") in s.connections
    assert [x.at for x in s.sensors] == ["TK.top", "TK.bottom"]
    assert list(s.routes) == ["V-P.out -- TK.top"]
    assert any("pressurized tanks are now tanks" in n for n in s.notes)
    sim = Simulation(s)
    assert sim.read("PT-TK") == pytest.approx(300 * 6894.757, rel=0.01)
    assert sim.temperature("TK.bottom") == pytest.approx(90.0, abs=0.1)
    assert sim.pressure("TK.gas") == sim.pressure("TK.top")  # scripts may still use the old port names
    again = System.from_dict(s.to_dict())  # saved, it is a plain tank
    assert again.component("TK").type == "tank" and not again.notes


def test_scripts_may_still_add_pressurized_tanks():
    s = System("script")
    s.add("TK", "pressurized_tank", liquid="water", pressurant="nitrogen", volume="20 L", fill=0.5,
          pressure="100 psi")
    s.add("V", "valve", cv=0.5)
    s.add("A", "boundary", fluid="water")
    s.connect("TK.liquid", "V.in"), s.connect("V.out", "A.port")
    s.sense("PT", "pressure", "TK.gas")
    assert s.component("TK").type == "tank" and s.connections[0] == ("TK.bottom", "V.in")
    sim = Simulation(s)
    assert sim._level("TK") == pytest.approx(0.5, abs=0.01)


def test_one_tank_type_builds_each_kind():
    def tank(**prm):
        s = System("t")
        s.add("TK", "tank", volume="10 L", **prm)
        s.add("O", "orifice", cda="1 mm2")
        s.add("A", "boundary", fluid=prm["fluid"])
        s.connect("TK.bottom", "O.in"), s.connect("O.out", "A.port")
        return build_network(s)

    assert len(tank(fluid="nitrogen", pressure="1000 psi", temperature="20 C").comp_nodes["TK"]) == 1
    assert len(tank(fluid="nitrous", fill=0.8, temperature="20 C").comp_nodes["TK"]) == 1
    two = tank(fluid="ethanol", pressurant="helium", fill=0.7, pressure="300 psi")
    assert [two.node_names[i] for i in two.comp_nodes["TK"]] == ["TK.ullage", "TK.liquid"]
    assert two.port_nodes["TK.top"] == two.port_nodes["TK.port"] == two.comp_nodes["TK"][0]
    assert two.port_nodes["TK.bottom"] == two.comp_nodes["TK"][1]
    with pytest.raises(BuildError, match="fill"):
        tank(fluid="ethanol", pressurant="helium", pressure="300 psi")


def test_mass_flow_on_a_line_reads_at_either_kind_of_end():
    s = lox_feed()
    s.settings["sensor_noise"] = "off"
    s.sense("F-OUT", "mass_flow", "TK.bottom")  # leaving the tank
    s.sense("F-VM", "mass_flow", "V-M.out")  # through the main valve, in to out
    s.sense("F-VENT", "mass_flow", "AMB.port")  # leaving the vent: negative while it takes flow
    sim = Simulation(s)
    sim.open("V-M")
    sim.run(0.5)
    assert sim.read("F-VM") == pytest.approx(sim.mass_flow("V-M"))
    assert sim.read("F-VM") > 1.0
    assert sim.read("F-OUT") == pytest.approx(sim.read("F-VM"), rel=0.05)
    assert sim.read("F-VENT") == pytest.approx(-sim.mass_flow("INJ"))


def test_mass_flow_needs_something_that_passes_flow():
    s = System("tee")
    s.add("K", "tank", fluid="nitrogen", volume="10 L", pressure="500 psi", temperature="20 C")
    s.add("V", "valve", cv=0.1)
    s.add("T", "tee")
    s.add("O", "orifice", cda="1 mm2")
    s.add("A", "boundary", fluid="nitrogen")
    for a, b in [("K.port", "V.in"), ("V.out", "T.a"), ("T.b", "O.in"), ("O.out", "A.port")]:
        s.connect(a, b)
    s.sense("F", "mass_flow", "T.a")
    with pytest.raises(ValueError, match="tee"):
        Simulation(s)
