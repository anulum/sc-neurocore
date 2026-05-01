# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import json

from sc_neurocore.compiler.intelligence import (
    dispatch_omni_paradigm,
    enforce_cognitive_bounds,
    generate_adiabatic_clocks,
    map_wetware_mea,
    route_holographic_interconnects,
    synthesize_morphology,
    synthesize_reversible_logic,
)

class TestMZIWeightEncoding:
    """Photonic MZI phase-shift weight encoding."""

    def test_encode_identity_matrix(self):
        from sc_neurocore.compiler.intelligence import encode_mzi_weights

        weights = [[1.0, 0.0], [0.0, 1.0]]
        enc = encode_mzi_weights(weights)
        assert enc.mesh_size == 2
        assert len(enc.phases_theta) == 2
        assert len(enc.phases_theta[0]) == 2

    def test_negative_weights_use_pi_shift(self):
        import math
        from sc_neurocore.compiler.intelligence import encode_mzi_weights

        weights = [[-1.0, 0.5]]
        enc = encode_mzi_weights(weights)
        # Negative weight → φ = π
        assert enc.phases_phi[0][0] == round(math.pi, 6)
        # Positive weight → φ = 0
        assert enc.phases_phi[0][1] == 0.0

    def test_theta_range(self):
        import math
        from sc_neurocore.compiler.intelligence import encode_mzi_weights

        weights = [[0.0, 0.5, 1.0]]
        enc = encode_mzi_weights(weights)
        for theta in enc.phases_theta[0]:
            assert 0 <= theta <= math.pi + 0.001

    def test_loss_reduces_transmission(self):
        from sc_neurocore.compiler.intelligence import encode_mzi_weights

        w = [[1.0]]
        enc_low = encode_mzi_weights(w, loss_db_per_mzi=0.0)
        enc_high = encode_mzi_weights(w, loss_db_per_mzi=3.0)
        assert enc_low.transmission[0][0] > enc_high.transmission[0][0]

    def test_json_config_output(self):
        from sc_neurocore.compiler.intelligence import (
            encode_mzi_weights,
            generate_mzi_config,
        )

        enc = encode_mzi_weights([[1.0, -0.5], [0.3, 0.8]])
        cfg = generate_mzi_config(enc, output_format="json")
        data = json.loads(cfg)
        assert "mesh_size" in data
        assert "phases_theta" in data

    def test_csv_config_output(self):
        from sc_neurocore.compiler.intelligence import (
            encode_mzi_weights,
            generate_mzi_config,
        )

        enc = encode_mzi_weights([[1.0, -0.5]])
        cfg = generate_mzi_config(enc, output_format="csv")
        assert "row,col,theta,phi,transmission" in cfg
        lines = cfg.strip().split("\n")
        assert len(lines) == 3  # header + 2 entries

    def test_zero_matrix_encoding(self):
        from sc_neurocore.compiler.intelligence import encode_mzi_weights

        enc = encode_mzi_weights([[0.0, 0.0], [0.0, 0.0]])
        # All theta should be 0 (no transmission)
        for row in enc.phases_theta:
            for t in row:
                assert t == 0.0


def test_omni_paradigm_dispatcher():
    """E2E test: partitioning complex SNN across paradigms."""
    equations = {
        "v": "v + I - decay",
        "noise": "rand() * sigma",
        "w_sum": "dot(weights, inputs)",
        "quantum_state": "entangle(v, ancilla)",
    }
    mapping = dispatch_omni_paradigm(equations)
    assert "v" in mapping.cmos_variables
    assert "noise" in mapping.thermodynamic_variables
    assert "w_sum" in mapping.optical_variables
    assert "quantum_state" in mapping.quantum_variables


def test_synthesize_reversible_logic():
    """E2E test: Reversible logic Toffoli/Fredkin gates."""
    equations = {
        "v": "v + I * R",
        "u": "u - v",
    }
    netlist = synthesize_reversible_logic(equations, bits=16)
    # v: 1 add (+), 1 mul (*) -> add=1, mul=1
    # u: 1 add (-) -> add=1, mul=0
    # Total ops_add = 2, ops_mul = 1
    # Toffoli expected = (2 * 3 * 16) + (1 * 16^2) = 96 + 256 = 352
    assert netlist.toffoli_gates == 352
    assert netlist.fredkin_gates == 256
    # Ancilla expected = (2 * 16) + (1 * 16^2) = 32 + 256 = 288
    assert netlist.ancilla_bits == 288


def test_map_wetware_mea():
    """E2E test: mapping to MEA electrodes."""
    mapping_high = map_wetware_mea(populations=1000, connectivity=0.8)
    assert mapping_high.electrode_count == 1000
    assert mapping_high.stimulation_freq_hz == 40.0
    assert mapping_high.spatial_density == "High"

    mapping_low = map_wetware_mea(populations=10, connectivity=0.1)
    assert mapping_low.electrode_count == 100
    assert mapping_low.stimulation_freq_hz == 8.0
    assert mapping_low.spatial_density == "Standard"


def test_synthesize_morphology():
    """E2E test: Morphological Auto-Synthesizer."""
    eq_mesh = {"v": "v+I"}  # Low interdependency
    morph_mesh = synthesize_morphology(eq_mesh, max_generations=5)
    assert morph_mesh.topology == "2D Mesh"

    eq_hyper = {"a": "a+b+c+d", "b": "a+b+c+d", "c": "a+b+c+d", "d": "a+b+c+d"}
    morph_hyper = synthesize_morphology(eq_hyper, max_generations=10)
    assert morph_hyper.topology == "Hypercube"


def test_enforce_cognitive_bounds():
    """E2E test: inserting kill switches."""
    eqs = {"v": "v_old + dV", "u": "u_old + dU"}
    bounds = {"v": (-65.0, 30.0)}
    safe = enforce_cognitive_bounds(eqs, bounds)
    assert safe.switches_inserted == 2
    assert "> 30.0" in safe.safe_equations["v"]
    assert "< -65.0" in safe.safe_equations["v"]
    assert safe.safe_equations["u"] == "u_old + dU"


def test_generate_adiabatic_clocks():
    """E2E test: generate multi-phase clocks."""
    clocks = generate_adiabatic_clocks(phases=4, freq_mhz=100.0)
    # 100 MHz = 10,000 ps period.
    # 4 phases -> Each segment is 2500 ps.
    assert len(clocks) == 4
    for clk in clocks:
        assert clk.rise_ps == 2500.0
        assert clk.hold_ps == 2500.0


def test_route_holographic_interconnects():
    """E2E test: optical holographic router."""
    router = route_holographic_interconnects(num_neurons=1000, connections=10_000_000)
    # Fanout = 10_000_000 // 1000 = 10_000
    assert router.optical_fanout_per_beam == 10000
    assert router.slm_grid_size[0] >= 4096  # Huge grid needed

