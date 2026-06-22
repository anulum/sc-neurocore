# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

import json

from sc_neurocore.compiler.intelligence import explore_pareto


class TestMultiTargetComparison:
    """Compile-once, compare-N-targets."""

    def test_compare_three_targets(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "ice40", "loihi2"],
        )
        assert len(results) == 3
        assert results[0].target == "artix7"
        assert results[1].target == "ice40"

    def test_dsp_targets_have_dsps(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b"},
            ["artix7", "bae_rad750"],
        )
        artix = results[0]
        rad = results[1]
        assert artix.estimated_dsps > 0
        assert rad.estimated_dsps == 0

    def test_format_report(self):
        from sc_neurocore.compiler.intelligence import (
            compare_targets,
            format_comparison_report,
        )

        results = compare_targets(
            {"v": "a * b + c"},
            ["artix7", "loihi2"],
        )
        report = format_comparison_report(results)
        assert "Multi-Target" in report
        assert "artix7" in report
        assert "loihi2" in report
        assert "Pipeline" in report

    def test_critical_path_consistent(self):
        from sc_neurocore.compiler.intelligence import compare_targets

        results = compare_targets(
            {"v": "a * b * c"},
            ["artix7", "ice40"],
        )
        # Same equations → same depth
        assert results[0].critical_path_depth == results[1].critical_path_depth


class TestCompilationSummary:
    """End-to-end compilation summary generation."""

    def test_summary_contains_sections(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a * b + c"},
            "artix7",
        )
        assert "## Module:" in s
        assert "### Equations" in s
        assert "### Target Platform" in s
        assert "### Fixed-Point Configuration" in s
        assert "### Resource Estimation" in s
        assert "### Pipeline Analysis" in s
        assert "### Applicable Features" in s

    def test_fpga_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
        )
        assert "TMR wrapper" in s
        assert "Bitstream encryption" in s

    def test_photonic_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "lightmatter_passage",
        )
        assert "MZI weight encoding" in s

    def test_neuromorphic_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "loihi2",
        )
        assert "On-chip learning" in s

    def test_in_memory_features(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "upmem_pim",
        )
        assert "PIM layout planner" in s

    def test_verilog_lines_shown(self):
        from sc_neurocore.compiler.intelligence import (
            generate_compilation_summary,
        )

        s = generate_compilation_summary(
            "sc_lif",
            {"v": "a + b"},
            "artix7",
            verilog_lines=150,
        )
        assert "150 lines" in s


class TestPareto(unittest.TestCase):
    def test_non_empty(self):
        pts = explore_pareto({"v": "-(v)/tau + I"})
        self.assertGreater(len(pts), 0)

    def test_non_dominated(self):
        pts = explore_pareto({"v": "a", "u": "b"})
        for i, p in enumerate(pts):
            for j, q in enumerate(pts):
                if i != j:
                    self.assertFalse(
                        q.power_mw <= p.power_mw
                        and q.area_luts <= p.area_luts
                        and q.latency_ns <= p.latency_ns
                        and (
                            q.power_mw < p.power_mw
                            or q.area_luts < p.area_luts
                            or q.latency_ns < p.latency_ns
                        ),
                        f"Point {i} dominated by {j}",
                    )

    def test_sorted_by_power(self):
        pts = explore_pareto({"v": "a"})
        powers = [p.power_mw for p in pts]
        self.assertEqual(powers, sorted(powers))


class TestProvenanceChain:
    """Cryptographic audit trail."""

    def test_chain_length(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain(
            "sc_lif",
            {"v": "a + b"},
        )
        assert len(chain) == 3
        assert chain[0].stage == "source_equations"
        assert chain[1].stage == "compilation_config"
        assert chain[2].stage == "verilog_generation"

    def test_hash_chain_linked(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain(
            "sc_lif",
            {"v": "a + b"},
        )
        assert chain[0].output_hash == chain[1].input_hash
        assert chain[1].output_hash == chain[2].input_hash

    def test_genesis(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        assert chain[0].input_hash == "genesis"

    def test_json_format(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
            format_provenance_json,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        j = format_provenance_json(chain)
        data = json.loads(j)
        assert "sc_neurocore_provenance" in data
        assert len(data["sc_neurocore_provenance"]["chain"]) == 3

    def test_deterministic_hashes(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        c1 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        c2 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert c1[0].output_hash == c2[0].output_hash


class TestModelComplexity:
    def test_compute_bound(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "a * b + c * d - e / f"})
        assert m.classification == "compute_bound"
        assert m.recommended_paradigm == "fpga"

    def test_simple_model(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "a + b"})
        assert m.compute_ops == 1

    def test_cross_refs(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "u + w", "u": "v", "w": "v + u"})
        assert m.comm_ratio > 0


class TestPortability:
    def test_simple_model(self):
        from sc_neurocore.compiler.intelligence import score_portability

        s = score_portability({"v": "a + b"})
        assert s.score > 50
        assert s.compatible_profiles > 0

    def test_complex_model(self):
        from sc_neurocore.compiler.intelligence import score_portability

        s = score_portability({"v": "a*b*c/d*e/f*g*h"})
        assert len(s.blockers) > 0


class TestCompilationReport:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_compilation_report

        md = generate_compilation_report("sc_lif", {"v": "a"}, "artix7")
        assert "# SC-NeuroCore Compilation Report" in md
        assert "artix7" in md
        assert "Carbon" in md

    def test_no_carbon(self):
        from sc_neurocore.compiler.intelligence import generate_compilation_report

        md = generate_compilation_report(
            "sc_lif",
            {"v": "a"},
            "artix7",
            include_carbon=False,
        )
        assert "Carbon" not in md


class TestPortabilityManyVariables:
    """A model with more than four state variables raises the register-file
    blocker that the single-variable portability cases never trigger."""

    def test_many_state_variables_flagged_as_blocker(self):
        from sc_neurocore.compiler.intelligence import score_portability

        eqs = {f"v{i}": "a + b" for i in range(5)}
        s = score_portability(eqs)
        assert any("state variables" in blocker for blocker in s.blockers)
