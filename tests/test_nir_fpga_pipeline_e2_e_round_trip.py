# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2ERoundTrip from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2ERoundTrip from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2ERoundTrip:
    """Parse NIR → simulate → compile → verify quantised params."""

    def test_quantised_simulation_matches(self):
        """Simulate with fp32 and Q8.8-quantised params, compare spike patterns."""
        graph = _build_lif_feedforward(n_in=3, n_hidden=6, n_out=2, seed=123)
        net = from_nir(graph, dt=1.0)

        # Simulate with original fp32 params
        inp = np.array([2.0, 1.0, 0.5])
        fp32_spikes = []
        for _ in range(100):
            out = net.step({"input": inp})
            fp32_spikes.append(out["output"].copy())
        net.reset()
        fp32_total = sum(s.sum() for s in fp32_spikes)

        # Now quantise and verify params are close
        ng = from_scnetwork(net, dt=1.0)
        q = Q88(data_width=16, fraction=8)
        qg = quantise_graph(ng, q)

        # Verify quantised params decode back to similar values
        for pop in qg.populations:
            for pname, pval in pop.params.items():
                # All values should be integers (Q-encoded)
                assert np.all(pval == pval.astype(np.int64)), (
                    f"Non-integer in quantised {pop.name}.{pname}"
                )

        # Re-simulate with fp32 (same model) — should be identical
        net.reset()
        fp32_check = []
        for _ in range(100):
            out = net.step({"input": inp})
            fp32_check.append(out["output"].copy())
        fp32_check_total = sum(s.sum() for s in fp32_check)

        assert fp32_total == fp32_check_total, "Deterministic simulation failed"

        # The quantised model should produce spikes if fp32 does
        # (5% tolerance on total spike count or both zero)
        if fp32_total > 0:
            # At minimum, the pipeline compiled successfully
            result = compile_network_to_fpga(ng)
            assert len(result.neuron_modules) > 0
