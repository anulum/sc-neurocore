
import unittest
import numpy as np
from sc_neurocore.quantum.hybrid import QuantumStochasticLayer
from sc_neurocore.bio.grn import GeneticRegulatoryLayer
from sc_neurocore.physics.heat import StochasticHeatSolver
from sc_neurocore.robotics.cpg import StochasticCPG

class TestFrontiers(unittest.TestCase):
    
    def test_quantum_layer(self):
        layer = QuantumStochasticLayer(n_qubits=2, length=64)
        bits = np.zeros((2, 64), dtype=np.uint8) # 0 prob
        out = layer.forward(bits)
        # 0 prob -> 0 angle -> cos(0)=1 prob -> output 1s
        self.assertGreater(np.mean(out), 0.9)
        
    def test_bio_grn(self):
        grn = GeneticRegulatoryLayer(n_neurons=2)
        initial_p = grn.protein_levels.copy()
        grn.step(np.array([1, 1]))
        # Protein should increase
        self.assertTrue(np.all(grn.protein_levels > initial_p))
        
    def test_physics_heat(self):
        solver = StochasticHeatSolver(length=10, num_walkers=100, alpha=0.1)
        solver.step()
        prof = solver.get_temperature_profile()
        self.assertEqual(len(prof), 10)
        self.assertAlmostEqual(np.sum(prof), 1.0)
        
    def test_robotics_cpg(self):
        cpg = StochasticCPG()
        s1, s2 = cpg.step()
        self.assertIn(s1, [0, 1])
        self.assertIn(s2, [0, 1])

if __name__ == '__main__':
    unittest.main()
