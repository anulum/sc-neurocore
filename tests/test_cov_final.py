# SPDX-License-Identifier: AGPL-3.0-or-later
"""Final coverage gap tests — targets all 27 remaining uncovered lines."""

from unittest.mock import patch
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Category 1: No-JAX fallback branches (6 lines)
# ---------------------------------------------------------------------------

class TestJaxBackendFallback:
    def test_to_jax_no_jax(self):
        with patch("sc_neurocore.accel.jax_backend.HAS_JAX", False):
            from sc_neurocore.accel.jax_backend import to_jax
            arr = np.array([1, 2, 3])
            result = to_jax(arr)
            assert result is arr


class TestJaxCompatFallback:
    def test_make_rng_no_jax(self):
        with patch("sc_neurocore.adapters.holonomic._jax_compat.HAS_JAX", False):
            from sc_neurocore.adapters.holonomic._jax_compat import make_rng
            key = make_rng(42)
            assert hasattr(key, "random")

    def test_split_rng_no_jax(self):
        with patch("sc_neurocore.adapters.holonomic._jax_compat.HAS_JAX", False):
            from sc_neurocore.adapters.holonomic._jax_compat import make_rng, split_rng
            key = make_rng(42)
            k1, k2 = split_rng(key)
            assert k1 is key

    def test_uniform_no_jax(self):
        with patch("sc_neurocore.adapters.holonomic._jax_compat.HAS_JAX", False):
            from sc_neurocore.adapters.holonomic._jax_compat import make_rng, uniform
            key = make_rng(42)
            result = uniform(key, (3,))
            assert result.shape == (3,)

    def test_normal_no_jax(self):
        with patch("sc_neurocore.adapters.holonomic._jax_compat.HAS_JAX", False):
            from sc_neurocore.adapters.holonomic._jax_compat import make_rng, normal
            key = make_rng(42)
            result = normal(key, (3,))
            assert result.shape == (3,)

    def test_maybe_jit_no_jax(self):
        with patch("sc_neurocore.adapters.holonomic._jax_compat.HAS_JAX", False):
            from sc_neurocore.adapters.holonomic._jax_compat import maybe_jit
            fn = lambda x: x + 1
            result = maybe_jit(fn)
            assert result is fn


# ---------------------------------------------------------------------------
# Category 2: Holonomic adapter step(inputs=None) + decode()
# ---------------------------------------------------------------------------

class TestHolonomicAdapterGaps:
    def test_l2_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter
        a = L2_NeurochemicalAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l4_decode(self):
        from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter
        a = L4_CellularAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "synchronization_r4" in d

    def test_l5_decode(self):
        from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter
        a = L5_OrganismalAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "organismal_valence" in d

    def test_l6_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter
        a = L6_PlanetaryAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l6_decode(self):
        from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter
        a = L6_PlanetaryAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "global_coherence_index" in d

    def test_l7_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter
        a = L7_SymbolicAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l7_decode(self):
        from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter
        a = L7_SymbolicAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "symbolic_unity_r7" in d

    def test_l8_decode(self):
        from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter
        a = L8_CosmicAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "cosmic_alignment_r8" in d

    def test_l9_decode(self):
        from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter
        a = L9_MemoryAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "memory_retrieval_r9" in d

    def test_l10_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter
        a = L10_FirewallAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l10_decode(self):
        from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter
        a = L10_FirewallAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "firewall_integrity_r10" in d

    def test_l11_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter
        a = L11_NoosphericAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l11_decode(self):
        from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter
        a = L11_NoosphericAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "noospheric_polarization" in d
        assert "collective_coherence_r11" in d

    def test_l12_step_no_inputs(self):
        from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter
        a = L12_GaianAdapter()
        out = a.step_jax(0.01, inputs=None)
        assert out.shape[0] > 0

    def test_l12_decode(self):
        from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter
        a = L12_GaianAdapter()
        bs = a.encode(None)
        d = a.decode(bs)
        assert "gaian_synchrony_index" in d


# ---------------------------------------------------------------------------
# Category 3: DNA mutation, quantum backend guards, sparse layer guard
# ---------------------------------------------------------------------------

class TestDNAMutation:
    def test_decode_triggers_mutation(self):
        from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder
        enc = DNAEncoder(mutation_rate=1.0)
        result = enc.decode("ACGT")
        assert len(result) == 8


class TestQuantumBackendGuards:
    def test_aer_without_qiskit(self):
        with patch("sc_neurocore.quantum.hardware_bridge.HAS_QISKIT", False):
            from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer
            with pytest.raises(RuntimeError, match="Qiskit"):
                QuantumHardwareLayer(n_qubits=2, backend_type="aer_simulator")

    def test_pennylane_without_pennylane(self):
        with patch("sc_neurocore.quantum.hardware_bridge.HAS_PENNYLANE", False):
            from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer
            with pytest.raises(RuntimeError, match="PennyLane"):
                QuantumHardwareLayer(n_qubits=2, backend_type="pennylane.default.qubit")


class TestVectorizedLayerSparseGuard:
    def test_sparse_without_scipy(self):
        with patch("sc_neurocore.layers.vectorized_layer.HAS_SCIPY_SPARSE", False):
            from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
            with pytest.raises(ImportError, match="scipy"):
                VectorizedSCLayer(n_inputs=4, n_neurons=4, sparse=True)
