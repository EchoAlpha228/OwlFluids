import sys; sys.path.insert(0, "tests")  # temporary CI diagnostic
from test_speed import _feed_transient
from owlfluids.examples.generate import engine_test_stand
from owlfluids.sim.simulation import Simulation
print("small", _feed_transient(12)[1])
sim = Simulation(engine_test_stand(), step_budget=200)
sim.run(0.1); sim.open("V-OXP","V-FUP"); sim.run(1.0); sim.close("V-OXV","V-FUV"); sim.run(0.5)
sim.open("V-OXM","V-FUM"); sim.run(1.0); sim.close("V-OXM","V-FUM"); sim.run(0.3)
sim.open("V-P1","V-P2","V-P3","V-P4"); sim.run(1.5); print("purge", sim.solver_stats()["forced_steps"])
