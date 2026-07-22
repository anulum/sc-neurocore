# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust performance test guard contracts

"""Static guards for host-dependent Rust wall-clock unit tests."""

from __future__ import annotations

from pathlib import Path


_INTERNEURON_PERFORMANCE_TESTS = (
    ("engine/src/neurons/interneurons/pv_fast_spiking.rs", "pv_performance_5k_steps"),
    ("engine/src/neurons/interneurons/sst_neuron.rs", "sst_performance_10k_steps"),
    ("engine/src/neurons/interneurons/vip_neuron.rs", "vip_performance_10k_steps"),
    (
        "engine/src/neurons/interneurons/chandelier_neuron.rs",
        "chandelier_performance_5k_steps",
    ),
    (
        "engine/src/neurons/interneurons/cerebellar_basket_neuron.rs",
        "basket_performance_5k_steps",
    ),
    (
        "engine/src/neurons/interneurons/martinotti_neuron.rs",
        "martinotti_performance_10k_steps",
    ),
)


def test_interneuron_wall_clock_performance_tests_are_ignored() -> None:
    """Interneuron timing smoke tests are opt-in, not default cargo-test gates."""
    ignore_marker = (
        '#[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]'
    )

    for source_path, test_name in _INTERNEURON_PERFORMANCE_TESTS:
        source = Path(source_path).read_text(encoding="utf-8")
        marker = f"fn {test_name}()"
        position = source.index(marker)
        prefix = source[max(0, position - 180) : position]
        assert ignore_marker in prefix, f"{test_name} must be opt-in"
