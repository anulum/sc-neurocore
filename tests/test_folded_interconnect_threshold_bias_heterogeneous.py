# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (threshold_bias_heterogeneous) from former test_folded_interconnect.py

from __future__ import annotations

from tests.folded_interconnect_support import *  # noqa: F403


def test_folded_external_source_threshold_matches_direct() -> None:
    import numpy as np

    # External-weighted input gated per-column by a NIR source Threshold: a column
    # contributes its (un-multiplied) weight only when its external input exceeds the
    # threshold. Currents straddle the thresholds so the gates toggle.
    n_dst, n_src = 5, 3
    weights = np.array(
        [[1.2, 0.8, 0.6], [1.0, 0.9, 0.7], [1.3, 0.5, 0.4], [0.9, 1.0, 0.6], [1.1, 0.7, 0.5]],
        dtype=np.float32,
    )
    src_thr = np.array([1.0, 1.2, 0.8], dtype=np.float32)
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights, source_threshold=src_thr)
    ng = NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )
    direct_raster, folded_raster = _parity_rasters(ng, [1.5, 1.1, 1.3], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded external source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "source-threshold workload should spike"


def test_folded_destination_threshold_matches_direct() -> None:
    import numpy as np

    # Mixed fan-in into one population: a destination-thresholded external connection
    # (the whole weighted sum, gated per neuron, emits one spike-magnitude) plus a plain
    # baseline external connection. The baseline alone stays sub-threshold; only neurons
    # whose thresholded connection also fires reach the LIF threshold, so the per-neuron
    # destination threshold toggles which neurons spike.
    n_dst = 4
    thr_w = np.full((n_dst, 3), 0.8, dtype=np.float32)  # raw ≈ (2.0+1.5+1.0)*0.8 = 3.6
    dst_thr = np.array([1.0, 3.0, 4.0, 5.0], dtype=np.float32)  # neurons 0,1 fire; 2,3 do not
    base_w = np.full((n_dst, 2), 0.3, dtype=np.float32)  # baseline ≈ 0.6, sub-threshold alone
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(src="stim", dst="pop0", weights=thr_w, destination_threshold=dst_thr),
            ConnectionSpec(src="base", dst="pop0", weights=base_w),
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    # External bus layout follows connection order: stim (3 lanes) then base (2 lanes).
    direct_raster, folded_raster = _parity_rasters(ng, [2.0, 1.5, 1.0, 1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded destination-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "destination-threshold workload should spike"


def test_folded_external_bias_matches_direct() -> None:
    import numpy as np

    # External-weighted input plus a per-destination-neuron bias constant. The bias is
    # added in ACC_WIDTH to the connection's term sum, so the steady-state LIF current
    # (≈ weighted input + bias) crosses the threshold for some neurons and not others —
    # the per-neuron bias ROM toggles which spike (positive, near-zero, and negative
    # biases all exercised).
    n_dst, n_src = 4, 2
    weights = np.full((n_dst, n_src), 0.2, dtype=np.float32)  # weighted ≈ 0.4 per neuron
    bias = np.array([1.0, 0.7, 0.3, -0.5], dtype=np.float32)  # I ≈ [1.4, 1.1, 0.7, -0.1]
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights, bias=bias)
    ng = NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )
    direct_raster, folded_raster = _parity_rasters(ng, [1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded bias raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "biased workload should spike"


def test_folded_bias_with_destination_threshold_matches_direct() -> None:
    import numpy as np

    # A destination-thresholded connection whose per-neuron bias participates in the
    # ``raw`` accumulator (raw = weighted sum + bias, compared against the per-neuron
    # destination threshold), mixed with a plain baseline connection. Equal weights
    # leave the bias as the discriminator: with raw ≈ 1.75 + bias and a threshold of
    # 2.0, neurons whose bias lifts raw above 2.0 emit a spike-magnitude that — together
    # with the sub-threshold baseline — pushes the LIF over, while the others stay quiet.
    n_dst = 4
    thr_w = np.full((n_dst, 2), 0.5, dtype=np.float32)  # weighted ≈ (2.0+1.5)*0.5 = 1.75
    bias = np.array([0.5, 0.0, -1.0, 0.3], dtype=np.float32)  # raw ≈ [2.25,1.75,0.75,2.05]
    dst_thr = np.full(n_dst, 2.0, dtype=np.float32)  # neurons 0,3 fire; 1,2 do not
    base_w = np.full((n_dst, 2), 0.3, dtype=np.float32)  # baseline ≈ 0.6, sub-threshold alone
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(
                src="stim",
                dst="pop0",
                weights=thr_w,
                bias=bias,
                destination_threshold=dst_thr,
            ),
            ConnectionSpec(src="base", dst="pop0", weights=base_w),
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    # External bus layout follows connection order: stim (2 lanes) then base (2 lanes).
    direct_raster, folded_raster = _parity_rasters(ng, [2.0, 1.5, 1.0, 1.0], n_dst)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded bias+destination-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "biased threshold workload should spike"


def test_folded_analogue_source_matches_direct() -> None:
    import numpy as np

    # An analogue li source population (membrane voltage, no spikes) feeds a lif
    # population: the lif input multiplies each source voltage by a weight (>> fraction),
    # reading the prior-tick committed voltage from the global v_bus — exactly the direct
    # path's registered v_out term. The li pop (pop 0) is driven by per-neuron external
    # current; its voltage rises toward I so the lif fan-in eventually crosses threshold.
    n_a, n_b = 3, 3
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=n_a, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    weights = np.full((n_b, n_a), 0.5, dtype=np.float32)
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=weights)],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    # External lanes drive the li pop (the connection-less first population).
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.5, 3.0], n_a + n_b)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded analogue-source raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "analogue-fed lif population should spike"


def test_folded_analogue_source_threshold_matches_direct() -> None:
    import numpy as np

    # An analogue li source gated by a per-column source Threshold: a column contributes
    # its (un-multiplied) sign-extended weight only while the source voltage exceeds the
    # column threshold. The thresholds straddle the per-neuron steady-state voltages so
    # the gates toggle as the li voltages rise (column 2's threshold is never reached).
    n_a, n_b = 3, 3
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=n_a, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=n_b, params={}, dt=1.0)
    weights = np.full((n_b, n_a), 0.8, dtype=np.float32)  # gated weight ≈ 0.8 per passing column
    src_thr = np.array([1.0, 2.5, 5.0], dtype=np.float32)  # col 0 early, col 1 later, col 2 never
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=weights, source_threshold=src_thr)],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.0, 2.0], n_a + n_b)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded analogue source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "gated analogue-fed lif should spike"


def test_folded_heterogeneous_v_threshold_matches_direct() -> None:
    import numpy as np

    # A population whose neurons carry different firing thresholds: the folded PE exposes
    # P_V_THRESHOLD on a port and streams each neuron's own value from a per-neuron ROM.
    # The thresholds straddle the reachable membrane range — neuron 0 (0.3) and 1 (0.5)
    # fire, neuron 2 (50.0) never does — so a fold that applied one shared threshold would
    # diverge. (Before the parameter ROM the fold baked only the first neuron's value.)
    pop = NeuronSpec(
        name="pop0",
        neuron_type="lif",
        n_neurons=3,
        params={"v_threshold": np.array([0.3, 0.5, 50.0])},
        dt=1.0,
    )
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(src="stim", dst="pop0", weights=np.full((3, 1), 1.0, np.float32))
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    q = Q88(data_width=_DW, fraction=_FR)
    _pe, folded_top = _build_top_folded(
        "sc_fold_test_folded", quantise_graph(ng, q), data_width=_DW, fraction=_FR
    )
    assert ".P_V_THRESHOLD(param_v_threshold_lif)" in folded_top  # streamed from the ROM

    direct_raster, folded_raster = _parity_rasters(ng, [1.5], 3)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded heterogeneous-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "low-threshold neurons should spike"
    # Neuron 2 (threshold 50.0, LSB-most in the MSB-first raster) must never fire.
    assert all(row[0] == "0" for row in direct_raster), "the high-threshold neuron must stay silent"


def test_folded_heterogeneous_tau_matches_direct() -> None:
    import numpy as np

    # Heterogeneous membrane time constants: a larger tau leaks more slowly and integrates
    # more input, so the neurons reach threshold at different times. The folded PE streams
    # each neuron's own tau from the parameter ROM; a shared tau would desynchronise them.
    pop = NeuronSpec(
        name="pop0",
        neuron_type="lif",
        n_neurons=3,
        params={"tau": np.array([4.0, 12.0, 40.0])},
        dt=1.0,
    )
    ng = NeuronGraph(
        populations=[pop],
        connections=[
            ConnectionSpec(src="stim", dst="pop0", weights=np.full((3, 1), 1.0, np.float32))
        ],
        input_pop="stim",
        output_pop="pop0",
        dt=1.0,
    )
    q = Q88(data_width=_DW, fraction=_FR)
    _pe, folded_top = _build_top_folded(
        "sc_fold_test_folded", quantise_graph(ng, q), data_width=_DW, fraction=_FR
    )
    assert ".P_TAU(param_tau_lif)" in folded_top

    direct_raster, folded_raster = _parity_rasters(ng, [1.5], 3)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    # The three time constants produce three distinct spike trains, so the ROM genuinely
    # feeds each neuron its own tau (a shared tau would collapse them).
    assert len({"".join(row[k] for row in direct_raster) for k in range(3)}) == 3
    assert folded_raster == direct_raster, (
        "folded heterogeneous-tau raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "the tau workload should spike"
