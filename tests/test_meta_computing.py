
import unittest
import numpy as np
from sc_neurocore.meta.time_crystal import TimeCrystalLayer
from sc_neurocore.meta.vacuum import VacuumNoiseSource
from sc_neurocore.meta.hyper_turing import OracleLayer
from sc_neurocore.meta.black_hole import EventHorizonLayer

class TestMetaComputing(unittest.TestCase):
    
    def test_time_crystal(self):
        tc = TimeCrystalLayer(n_spins=4)
        bits = tc.get_bitstream(cycles=4)
        self.assertEqual(len(bits), 4)
        
    def test_vacuum_source(self):
        vac = VacuumNoiseSource(dimension=2)
        bits = vac.generate_virtual_bits(length=100)
        self.assertEqual(bits.shape, (2, 100))
        
    def test_hyper_turing(self):
        oracle = OracleLayer()
        res = oracle.solve_halting(np.zeros(100))
        self.assertTrue(res)
        
    def test_black_hole(self):
        bh = EventHorizonLayer(n_inputs=2, n_outputs=4)
        bits = np.zeros((2, 10), dtype=np.uint8)
        scrambled = bh.scramble(bits)
        self.assertEqual(scrambled.shape, (4, 10))

if __name__ == '__main__':
    unittest.main()
