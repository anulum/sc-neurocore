# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.doctor (architecture diagnostics)

from __future__ import annotations

import numpy as np

from sc_neurocore.doctor import diagnose, Diagnosis, DiagnosticReport
from sc_neurocore.doctor.diagnose import Severity


class TestDiagnosticReport:
    def test_empty(self):
        r = DiagnosticReport(target="ice40")
        assert r.score == 100
        assert r.has_critical is False

    def test_score_with_findings(self):
        r = DiagnosticReport(
            target="ice40",
            findings=[
                Diagnosis("test", Severity.WARNING, "msg", "fix"),
                Diagnosis("test", Severity.CRITICAL, "msg", "fix"),
            ],
        )
        assert r.score < 100
        assert r.has_critical is True

    def test_summary(self):
        r = DiagnosticReport(
            target="artix7",
            findings=[Diagnosis("hw", Severity.WARNING, "high util", "prune")],
        )
        s = r.summary()
        assert "artix7" in s
        assert "high util" in s
        assert "prune" in s


class TestDiagnoseHardware:
    def test_small_network_fits(self):
        r = diagnose([(4, 2)], target="artix7")
        hw = [f for f in r.findings if f.category.startswith("hardware")]
        assert len(hw) >= 1

    def test_overprovisioned(self):
        r = diagnose([(4, 2)], target="artix7")
        over = [f for f in r.findings if f.category == "hardware_overprovisioned"]
        assert len(over) >= 1

    def test_large_network_exceeds(self):
        r = diagnose([(256, 128), (128, 64)], target="ice40", bitstream_length=512)
        hw = [f for f in r.findings if f.category == "hardware_fit"]
        assert any(f.severity == Severity.CRITICAL for f in hw)


class TestDiagnoseWeights:
    def test_sparse_weights(self):
        w = [np.zeros((10, 10))]
        r = diagnose([(10, 10)], weights=w, target="artix7")
        sparse = [f for f in r.findings if f.category == "weight_sparsity"]
        assert len(sparse) >= 1

    def test_outlier_weights(self):
        w = [np.ones((10, 10)) * 0.01]
        w[0][0, 0] = 100.0
        r = diagnose([(10, 10)], weights=w, target="artix7")
        outlier = [f for f in r.findings if f.category == "weight_outliers"]
        assert len(outlier) >= 1

    def test_sc_range_warning(self):
        w = [np.random.randn(5, 5) * 5]
        r = diagnose([(5, 5)], weights=w, target="artix7")
        sc = [f for f in r.findings if f.category == "weight_sc_range"]
        assert len(sc) >= 1

    def test_healthy_weights(self):
        w = [np.random.rand(10, 10) * 0.5]
        r = diagnose([(10, 10)], weights=w, target="artix7")
        problems = [
            f for f in r.findings if f.category.startswith("weight") and f.severity != Severity.OK
        ]
        assert len(problems) == 0


class TestDiagnoseSpikeRates:
    def test_dead_neurons(self):
        rates = [np.zeros(20)]
        r = diagnose([(10, 20)], spike_rates=rates, target="artix7")
        dead = [f for f in r.findings if f.category == "dead_neurons"]
        assert len(dead) >= 1
        assert dead[0].severity == Severity.CRITICAL

    def test_saturated_neurons(self):
        rates = [np.ones(10)]
        r = diagnose([(10, 10)], spike_rates=rates, target="artix7")
        sat = [f for f in r.findings if f.category == "saturated_neurons"]
        assert len(sat) >= 1

    def test_healthy_rates(self):
        rates = [np.full(20, 0.15)]
        r = diagnose([(10, 20)], spike_rates=rates, target="artix7")
        ok = [f for f in r.findings if f.category == "spike_efficiency"]
        assert len(ok) >= 1
        assert ok[0].severity == Severity.OK


class TestDiagnoseArchitecture:
    def test_bottleneck(self):
        r = diagnose([(256, 256), (256, 8)], target="artix7")
        bn = [f for f in r.findings if f.category == "architecture_bottleneck"]
        assert len(bn) >= 1

    def test_small_capacity(self):
        r = diagnose([(64, 4)], target="artix7")
        cap = [f for f in r.findings if f.category == "architecture_capacity"]
        assert len(cap) >= 1


class TestDiagnoseCodingEfficiency:
    def test_overprovisioned_coding(self):
        r = diagnose([(10, 8)], target="artix7", bitstream_length=512)
        cod = [f for f in r.findings if f.category == "coding_overprovisioned"]
        assert len(cod) >= 1

    def test_underprovisioned_coding(self):
        layers = [(64, 64), (64, 64), (64, 64), (64, 32)]
        r = diagnose(layers, target="artix7", bitstream_length=32)
        cod = [f for f in r.findings if f.category == "coding_underprovisioned"]
        assert len(cod) >= 1


class TestDiagnoseIntegration:
    def test_full_diagnosis(self):
        layers = [(64, 32), (32, 10)]
        weights = [np.random.randn(32, 64) * 0.3, np.random.randn(10, 32) * 0.3]
        rates = [np.random.uniform(0.05, 0.3, 32), np.random.uniform(0.05, 0.3, 10)]
        r = diagnose(layers, weights=weights, spike_rates=rates, target="artix7")
        assert isinstance(r, DiagnosticReport)
        assert r.score >= 0
        assert r.score <= 100
        s = r.summary()
        assert "Architecture Doctor" in s
