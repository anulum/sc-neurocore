# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (handoff_accept) from former test_scnir_handoff_audit.py

from __future__ import annotations

from tests.scnir_handoff_audit_support import *  # noqa: F403


def test_audit_scnir_hdl_handoff_accepts_complete_compile_output(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)

    report = audit_scnir_hdl_handoff(handoff)

    assert report.module_name == "mixed_audit_net"
    assert report.stream_count == 3
    assert report.source_module_count == 3
    assert report.hierarchy_instance_count == 1
    assert report.hierarchy_port_count == 3
    assert report.hierarchy_instances == {
        "top.mixed": {
            "module_name": "mixed_audit_net_core",
            "ports": [
                {
                    "port_name": "li_state_i",
                    "direction": "input",
                    "stream_id": "pop.li.state",
                    "signal_kind": "analogue_state",
                    "bit_width": 16,
                },
                {
                    "port_name": "lif_spike_o",
                    "direction": "output",
                    "stream_id": "pop.lif.spike",
                    "signal_kind": "spike",
                    "bit_width": 1,
                },
                {
                    "port_name": "weight_i",
                    "direction": "input",
                    "stream_id": "conn.li_to_lif.weight",
                    "signal_kind": "weight",
                    "bit_width": 16,
                },
            ],
        }
    }
    assert report.signal_routes["analogue_state"] == "direct_mac"
    assert report.external_input_count == 2
    assert report.external_inputs == (
        {"source": "sensor_a", "offset": 0, "width": 2},
        {"source": "sensor_b", "offset": 2, "width": 1},
    )
    assert "scnir_document.json" in report.artefacts
    assert report.as_dict()["external_inputs"] == [
        {"source": "sensor_a", "offset": 0, "width": 2},
        {"source": "sensor_b", "offset": 2, "width": 1},
    ]
    assert report.as_dict()["hierarchy_port_count"] == 3
    assert report.as_dict()["status"] == "valid"
    assert "mixed_audit_net_core.v" in report.artefacts


def test_audit_scnir_hdl_handoff_accepts_real_compile_nir_output(tmp_path: Path) -> None:
    nir = pytest.importorskip("nir")
    model_path = tmp_path / "model.nir"
    out_dir = tmp_path / "compiled"
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )
    nir.write(str(model_path), graph)

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "compile-nir",
            str(model_path),
            "--module-name",
            "real_handoff_net",
            "--T",
            "512",
            "--source-kind",
            "sobol",
            "--base-seed",
            "101",
            "--output",
            str(out_dir),
        ],
    ):
        rc = main()

    assert rc == 0
    report = audit_scnir_hdl_handoff(out_dir)
    assert report.module_name == "real_handoff_net"
    assert report.stream_count == 2
    assert report.source_module_count == 2
    assert report.hierarchy_instance_count == 0
    assert report.hierarchy_port_count == 0
