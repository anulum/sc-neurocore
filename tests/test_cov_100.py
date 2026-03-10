# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests targeting remaining coverage gaps (98% -> 100%)."""

from __future__ import annotations

import os
import sys
import tempfile
import types

from unittest.mock import patch

import numpy as np
import pytest


# ── Holonomic adapter dimension mismatches ───────────────────────────────

class TestHolonomicDimMismatch:
    @pytest.fixture(autouse=True)
    def _skip_no_jax(self):
        pytest.importorskip("jax")

    def test_l1_quantum_with_inputs(self):
        from sc_neurocore.adapters.holonomic.l1_quantum import L1_QuantumAdapter
        a = L1_QuantumAdapter()
        a.step_jax(0.01, inputs=np.ones((a.params.n_qubits, a.params.bitstream_length)))

    def test_l2_chem_mismatch(self):
        from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter
        a = L2_NeurochemicalAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l3_gen_mismatch(self):
        from sc_neurocore.adapters.holonomic.l3_gen import L3_GenomicAdapter
        a = L3_GenomicAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l4_cell_mismatch(self):
        from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter
        a = L4_CellularAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l5_org_mismatch(self):
        from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter
        a = L5_OrganismalAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l6_plan_mismatch(self):
        from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter
        a = L6_PlanetaryAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l7_sym_mismatch(self):
        from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter
        a = L7_SymbolicAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l8_cosm_mismatch(self):
        from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter
        a = L8_CosmicAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l9_mem_mismatch(self):
        from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter
        a = L9_MemoryAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l10_fire_mismatch(self):
        from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter
        a = L10_FirewallAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l11_noos_mismatch(self):
        from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter
        a = L11_NoosphericAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))

    def test_l12_gaian_mismatch(self):
        from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter
        a = L12_GaianAdapter()
        a.step_jax(0.01, inputs=np.ones((999, a.params.bitstream_length)))


# ── JAX backend fallbacks ────────────────────────────────────────────────

class TestJaxBackendPaths:

    def test_to_jax_with_jax(self):
        pytest.importorskip("jax")
        from sc_neurocore.accel.jax_backend import to_jax
        result = to_jax(np.array([1.0, 2.0]))
        assert hasattr(result, "shape")

    def test_to_host_jax_array(self):
        jax = pytest.importorskip("jax")
        from sc_neurocore.accel.jax_backend import to_host
        arr = jax.numpy.array([1.0, 2.0])
        result = to_host(arr)
        assert isinstance(result, np.ndarray)

    def test_to_host_plain_ndarray(self):
        from sc_neurocore.accel.jax_backend import to_host
        result = to_host(np.array([1.0]))
        assert isinstance(result, np.ndarray)

    def test_jax_pack_3d_raises(self):
        pytest.importorskip("jax")
        from sc_neurocore.accel.jax_backend import jax_pack_bitstream
        with pytest.raises(ValueError, match="Expected 1-D or 2-D"):
            jax_pack_bitstream(np.zeros((2, 3, 64), dtype=np.uint8))

    def test_jax_pack_2d(self):
        pytest.importorskip("jax")
        from sc_neurocore.accel.jax_backend import jax_pack_bitstream
        result = jax_pack_bitstream(np.ones((2, 64), dtype=np.uint8))
        assert result.shape[0] == 2

    def test_jax_pack_no_jax(self):
        from sc_neurocore.accel import jax_backend
        orig = jax_backend.HAS_JAX
        try:
            jax_backend.HAS_JAX = False
            with pytest.raises(RuntimeError, match="JAX is not available"):
                jax_backend.jax_pack_bitstream(np.zeros(64, dtype=np.uint8))
        finally:
            jax_backend.HAS_JAX = orig


# ── JAX compat with-JAX branches ────────────────────────────────────────

class TestJaxCompatWithJax:

    @pytest.fixture(autouse=True)
    def _skip_no_jax(self):
        pytest.importorskip("jax")

    def test_make_rng(self):
        from sc_neurocore.adapters.holonomic._jax_compat import make_rng
        result = make_rng(42)
        assert hasattr(result, "shape")

    def test_split_rng(self):
        from sc_neurocore.adapters.holonomic._jax_compat import split_rng, make_rng
        k1, k2 = split_rng(make_rng(42))
        assert k1.shape == k2.shape

    def test_uniform(self):
        from sc_neurocore.adapters.holonomic._jax_compat import uniform, make_rng
        result = uniform(make_rng(42), (4,))
        assert result.shape == (4,)

    def test_normal(self):
        from sc_neurocore.adapters.holonomic._jax_compat import normal, make_rng
        result = normal(make_rng(42), (4,))
        assert result.shape == (4,)

    def test_maybe_jit(self):
        from sc_neurocore.adapters.holonomic._jax_compat import maybe_jit
        assert maybe_jit(lambda x: x + 1)(1) == 2


# ── DNA storage ──────────────────────────────────────────────────────────

class TestDNAStorage:

    def test_odd_length_pads(self):
        from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder
        dna = DNAEncoder(mutation_rate=0.0).encode(np.array([1, 0, 1], dtype=np.uint8))
        assert len(dna) == 2

    def test_decode(self):
        from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder
        result = DNAEncoder(mutation_rate=0.0).decode("ACGT")
        assert len(result) == 8


# ── Error path validation ────────────────────────────────────────────────

class TestErrorPaths:

    def test_fp_lif_data_width_zero(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        with pytest.raises(ValueError, match="data_width"):
            FixedPointLIFNeuron(data_width=0)

    def test_fp_lif_data_width_33(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        with pytest.raises(ValueError, match="data_width"):
            FixedPointLIFNeuron(data_width=33)

    def test_fp_lif_fraction_invalid(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        with pytest.raises(ValueError, match="fraction"):
            FixedPointLIFNeuron(data_width=16, fraction=16)

    def test_fp_lif_negative_refractory(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        with pytest.raises(ValueError, match="refractory_period"):
            FixedPointLIFNeuron(refractory_period=-1)

    def test_stochastic_lif_negative_tau(self):
        from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
        with pytest.raises(ValueError, match="tau_mem"):
            StochasticLIFNeuron(tau_mem=-1.0)

    def test_vectorized_layer_connectivity_zero(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
        with pytest.raises(ValueError, match="connectivity"):
            VectorizedSCLayer(n_inputs=4, n_neurons=4, connectivity=0.0)

    def test_vectorized_layer_connectivity_over_1(self):
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
        with pytest.raises(ValueError, match="connectivity"):
            VectorizedSCLayer(n_inputs=4, n_neurons=4, connectivity=1.5)

    def test_swarm_agent_wrong_weight_size(self):
        from sc_neurocore.swarm.agent import SwarmAgent, AgentConfig
        agent = SwarmAgent(AgentConfig(n_sensory=4, n_hidden=8, n_motor=2))
        with pytest.raises(ValueError, match="Expected"):
            agent.weights = np.zeros(3)


# ── SSGF audio mapping fallbacks ────────────────────────────────────────

class TestSSGFSmallN:

    def test_n2_all_fallbacks(self):
        from sc_neurocore.audio.ssgf_engine import SSGFConfig, SSGFEngine
        engine = SSGFEngine(SSGFConfig(N=2))
        engine.outer_step()
        m = engine.get_audio_mapping()
        assert m["binaural_hz"] == 10.0
        assert m["pulse_rate"] == 8.0
        assert m["spatial_angle"] == 0.0

    def test_n3_binaural_branch(self):
        from sc_neurocore.audio.ssgf_engine import SSGFConfig, SSGFEngine
        engine = SSGFEngine(SSGFConfig(N=3))
        engine.outer_step()
        m = engine.get_audio_mapping()
        assert m["pulse_rate"] == 8.0
        assert m["spatial_angle"] == 0.0

    def test_n5_pulse_branch(self):
        from sc_neurocore.audio.ssgf_engine import SSGFConfig, SSGFEngine
        engine = SSGFEngine(SSGFConfig(N=5))
        engine.outer_step()
        m = engine.get_audio_mapping()
        assert m["spatial_angle"] == 0.0


# ── Sleep edge cases ─────────────────────────────────────────────────────

class TestSleepEdgeCases:

    def test_circadian_non_wrapping(self):
        from sc_neurocore.sleep.circadian_optimizer import CircadianOptimizer, Chronotype
        opt = CircadianOptimizer(Chronotype.WOLF)  # bed=0.5, wake=8.5
        assert opt.is_in_sleep_window(4.0) is True
        assert opt.is_in_sleep_window(12.0) is False

    def test_circadian_wrapping(self):
        from sc_neurocore.sleep.circadian_optimizer import CircadianOptimizer, Chronotype
        opt = CircadianOptimizer(Chronotype.LION)  # bed=21.5, wake=5.5
        assert opt.is_in_sleep_window(23.0) is True
        assert opt.is_in_sleep_window(3.0) is True
        assert opt.is_in_sleep_window(12.0) is False

    def test_protocol_rem_fallback(self):
        from sc_neurocore.sleep.protocol_library import SleepProtocol
        from sc_neurocore.sleep.sleep_stage_detector import SleepStage
        proto = SleepProtocol(stage_targets={})
        assert proto.get_target_stage(0.5) == SleepStage.REM

    def test_sleep_detector_zero_signal(self):
        from sc_neurocore.sleep.sleep_stage_detector import SleepStageDetector, SleepStage
        assert SleepStageDetector._classify(np.zeros(5)) == SleepStage.WAKE


# ── Compiler/MLIR ────────────────────────────────────────────────────────

class TestCompilerCoverage:

    def test_mlir_emit_xor(self):
        from sc_neurocore.compiler.mlir_emitter import MLIREmitter
        e = MLIREmitter("test_xor")
        a = e.emit_and("in1", "in2")
        e.emit_xor(a, "in1")
        assert "comb.xor" in e.generate()

    def test_mlir_emit_mux(self):
        from sc_neurocore.compiler.mlir_emitter import MLIREmitter
        e = MLIREmitter("test_mux")
        e.emit_mux("cond", "t", "f")
        assert "comb.mux" in e.generate()

    def test_pipeline_makedirs(self):
        from sc_neurocore.compiler.pipeline import CompilerPipeline
        with tempfile.TemporaryDirectory() as td:
            new_dir = os.path.join(td, "sub", "dir")
            CompilerPipeline(work_dir=new_dir)
            assert os.path.isdir(new_dir)

    def test_pipeline_sanitize_empty(self):
        from sc_neurocore.compiler.pipeline import CompilerPipeline
        with tempfile.TemporaryDirectory() as td:
            p = CompilerPipeline(work_dir=td)
            with pytest.raises(ValueError, match="Invalid output name"):
                p._sanitize_name("!@#$")

    def test_pipeline_path_escape(self):
        from sc_neurocore.compiler.pipeline import CompilerPipeline
        with tempfile.TemporaryDirectory() as td:
            p = CompilerPipeline(work_dir=td)
            with pytest.raises(ValueError, match="Path escapes"):
                p._validate_path("/etc/passwd")

    def test_pipeline_bad_target(self):
        from sc_neurocore.compiler.pipeline import CompilerPipeline
        with tempfile.TemporaryDirectory() as td:
            p = CompilerPipeline(work_dir=td)
            v_path = os.path.join(td, "dummy.v")
            with open(v_path, "w") as f:
                f.write("module dummy(); endmodule")
            with pytest.raises(ValueError, match="Unknown target"):
                p.run_synthesis(v_path, target_fpga="bad_fpga")


# ── Export/HDL write errors ──────────────────────────────────────────────

class TestFileWriteErrors:

    def test_onnx_export_write_error(self):
        from sc_neurocore.export.onnx_exporter import SCOnnxExporter

        class _Layer:
            n_inputs = 4

        with patch("builtins.open", side_effect=OSError("test")):
            with pytest.raises(OSError):
                SCOnnxExporter.export([_Layer()], "model.json")

    def test_verilog_save_write_error(self):
        from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator
        gen = VerilogGenerator(module_name="test_mod")
        with patch("builtins.open", side_effect=OSError("test")):
            with pytest.raises(OSError):
                gen.save_to_file("output.v")


# ── QEC non-repetition code ─────────────────────────────────────────────

class TestQECNonRepetition:

    def test_surface_passthrough(self):
        from sc_neurocore.quantum.qec import QecShield
        qec = QecShield(code_type="surface", distance=3)
        bits = np.random.randint(0, 2, (4, 3, 64), dtype=np.uint8)
        assert qec.encode(bits) is bits
        assert qec.extract_syndromes(bits).shape == bits.shape
        assert qec.decode(bits) is bits


# ── Quantum hardware bridge ─────────────────────────────────────────────

class TestQuantumHardwareBridge:

    def test_unknown_backend_raises(self):
        from sc_neurocore.quantum.hardware_bridge import QuantumHardwareLayer
        layer = QuantumHardwareLayer.__new__(QuantumHardwareLayer)
        layer.n_qubits = 2
        layer.length = 64
        layer.backend_type = "unknown"
        layer._qiskit_simulator = None
        layer._pennylane_dev = None
        with pytest.raises(ValueError, match="Unknown backend"):
            layer.forward(np.zeros((2, 64), dtype=np.uint8))


# ── Verification ─────────────────────────────────────────────────────────

class TestVerificationCoverage:

    def test_interval_repr(self):
        from sc_neurocore.verification.formal_proofs import Interval
        s = repr(Interval(0.1, 0.9))
        assert "0.1000" in s and "0.9000" in s

    def test_safety_blocked_attr(self):
        from sc_neurocore.verification.safety import CodeSafetyVerifier
        assert CodeSafetyVerifier().verify_code_safety("import os; os.system('rm -rf /')") is False

    def test_safety_blocked_builtin(self):
        from sc_neurocore.verification.safety import CodeSafetyVerifier
        assert CodeSafetyVerifier().verify_code_safety("exec('print(1)')") is False


# ── MDL parser ───────────────────────────────────────────────────────────

class TestMDLParser:

    def test_encode_with_get_state(self):
        from sc_neurocore.core.mdl_parser import MindDescriptionLanguage

        class _Orch:
            modules = {"m": type("M", (), {"get_state": lambda self: {"v": 0.5}, "__module__": __name__})()}

        yaml_str = MindDescriptionLanguage.encode(_Orch(), "agent")
        assert "m" in yaml_str

    def test_encode_with_weights(self):
        from sc_neurocore.core.mdl_parser import MindDescriptionLanguage

        class _Orch:
            modules = {"w": type("W", (), {"weights": np.array([0.5]), "__module__": __name__})()}

        yaml_str = MindDescriptionLanguage.encode(_Orch(), "agent")
        assert "w" in yaml_str


# ── JAX dense layer ─────────────────────────────────────────────────────

class TestJaxDenseLayerCoverage:

    @pytest.fixture(autouse=True)
    def _skip_no_jax(self):
        pytest.importorskip("jax")

    def test_no_jax_raises(self):
        from sc_neurocore.layers import jax_dense_layer
        orig = jax_dense_layer.HAS_JAX
        try:
            jax_dense_layer.HAS_JAX = False
            with pytest.raises(RuntimeError, match="JAX is required"):
                jax_dense_layer.JaxSCDenseLayer(n_neurons=4, n_inputs=4)
        finally:
            jax_dense_layer.HAS_JAX = orig

    def test_reset(self):
        from sc_neurocore.layers.jax_dense_layer import JaxSCDenseLayer
        import jax.numpy as jnp
        layer = JaxSCDenseLayer(n_neurons=4, n_inputs=4)
        layer.v = jnp.ones(4)
        layer.reset()
        assert float(layer.v.sum()) == pytest.approx(0.0, abs=0.01)


# ── MPI driver gather ───────────────────────────────────────────────────

class TestMPIGather:

    def test_gather_non_root_returns_empty(self):
        from sc_neurocore.accel import mpi_driver

        driver = mpi_driver.MPIDriver.__new__(mpi_driver.MPIDriver)
        driver.rank = 1
        driver.size = 2

        class _Comm:
            def Gather(self, local, buf, root=0):
                pass

        driver.comm = _Comm()
        orig = mpi_driver.HAS_MPI
        try:
            mpi_driver.HAS_MPI = True
            result = driver.gather_results(np.array([1.0, 2.0]))
            assert result.size == 0
        finally:
            mpi_driver.HAS_MPI = orig

    def test_gather_root_returns_data(self):
        from sc_neurocore.accel import mpi_driver

        driver = mpi_driver.MPIDriver.__new__(mpi_driver.MPIDriver)
        driver.rank = 0
        driver.size = 2

        class _Comm:
            def Gather(self, local, buf, root=0):
                if buf is not None:
                    buf[:len(local)] = local

        driver.comm = _Comm()
        orig = mpi_driver.HAS_MPI
        try:
            mpi_driver.HAS_MPI = True
            result = driver.gather_results(np.array([1.0, 2.0]))
            assert result.shape == (4,)
        finally:
            mpi_driver.HAS_MPI = orig


# ── CLI info with engine mock ────────────────────────────────────────────

class TestCLIInfo:

    def test_cmd_info_with_engine(self):
        from sc_neurocore.cli import _cmd_info
        fake = types.ModuleType("sc_neurocore_engine")
        fake.__version__ = "0.0.0-test"  # type: ignore[attr-defined]
        fake.simd_tier = lambda: "test"  # type: ignore[attr-defined]
        sys.modules["sc_neurocore_engine"] = fake
        try:
            assert _cmd_info() == 0
        finally:
            del sys.modules["sc_neurocore_engine"]


# ── SCPN layer edge cases ───────────────────────────────────────────────

class TestSCPNLayerEdgeCases:

    def test_l1_quantum_hardware_backend(self):
        pytest.importorskip("qiskit")
        from sc_neurocore.scpn.layers.l1_quantum import L1_QuantumLayer, L1_StochasticParameters
        layer = L1_QuantumLayer(params=L1_StochasticParameters(n_qubits=2, backend="aer_simulator"))
        assert layer.quantum_core is not None

    def test_l10_boundary_short_noise(self):
        from sc_neurocore.scpn.layers.l10_boundary import L10_BoundaryLayer, L10_StochasticParameters
        layer = L10_BoundaryLayer(params=L10_StochasticParameters(n_boundary_nodes=4))
        result = layer.step(0.01, external_noise=np.array([0.1, 0.2]))
        assert "firewall_strength" in result

    def test_l10_boundary_long_noise(self):
        from sc_neurocore.scpn.layers.l10_boundary import L10_BoundaryLayer, L10_StochasticParameters
        layer = L10_BoundaryLayer(params=L10_StochasticParameters(n_boundary_nodes=4))
        result = layer.step(0.01, external_noise=np.ones(10))
        assert "firewall_strength" in result


# ── Swarm evolver with collective fields ─────────────────────────────────

class TestSwarmEvolver:

    def test_evolver_with_fields(self):
        from sc_neurocore.swarm.neuroevolution_swarm import SwarmEvolver, EvolverConfig
        evolver = SwarmEvolver(EvolverConfig(
            pop_size=4, n_elite=2, n_eval_steps=5, use_fields=True, seed=42,
        ))
        assert isinstance(evolver.evolve_generation(), float)
