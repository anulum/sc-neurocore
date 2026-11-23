# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEndToEnd from former test_tinysc_ports.py

"""Focused suite: TestEndToEnd from former test_tinysc_ports.py."""

from __future__ import annotations

from sc_neurocore.edge.bitstream import (
    scc,
)
from sc_neurocore.edge.lfsr import (
    Lfsr16,
)
from sc_neurocore.edge.sc_network import (
    SCLayer,
    SCNetwork,
)
from sc_neurocore.edge.telemetry import (
    DeviceTelemetry,
)


class TestEndToEnd:
    """Inference, telemetry and SCC through the tiny SC port."""

    def test_inference_telemetry_pipeline(self) -> None:
        """Ten inference ticks are recorded with the output layer."""
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        dt = DeviceTelemetry()
        for _ in range(10):
            spikes = net.run([0.5, 0.5, 0.5, 0.5])
            dt.record("output", sum(spikes), len(spikes))
        s = dt.summary()
        assert s["total_ticks"] == 10
        assert "output" in s["layers"]

    def test_lfsr_encode_scc_consistency(self) -> None:
        """A stream correlates fully with itself."""
        lfsr1 = Lfsr16(0xACE1)
        lfsr2 = Lfsr16(0xACE1)
        a = lfsr1.encode_float(0.5, 1024)
        b = lfsr2.encode_float(0.5, 1024)
        corr = scc(a, b, 1024)
        assert abs(corr - 1.0) < 0.01

    def test_scc_uncorrelated(self) -> None:
        """Independent streams are nearly uncorrelated."""
        a = Lfsr16(0xACE1).encode_float(0.5, 1024)
        b = Lfsr16(0x1234).encode_float(0.5, 1024)
        corr = scc(a, b, 1024)
        assert abs(corr) < 0.15
