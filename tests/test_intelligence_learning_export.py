# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — On-chip learning export contracts

"""Contracts for on-chip learning parameter and configuration export."""

from __future__ import annotations

import json

import pytest


class TestOnChipLearning:
    """STDP / reward-modulated learning parameter export."""

    def test_default_stdp_params(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params()
        assert p.learning_rule == "stdp"
        assert p.tau_plus_ms == 20.0
        assert p.a_plus == 0.01
        assert p.target_platform == "akida2"

    def test_rstdp_params(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params(
            learning_rule="rstdp",
            reward_tau_ms=500.0,
        )
        assert p.learning_rule == "rstdp"
        assert p.reward_tau_ms == 500.0

    def test_json_export(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params()
        cfg = export_learning_config(p, output_format="json")
        data = json.loads(cfg)
        assert data["learning_rule"] == "stdp"
        assert "time_constants" in data
        assert "weight_bounds" in data

    def test_yaml_export(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        p = generate_learning_params(target="brainscales2")
        cfg = export_learning_config(p, output_format="yaml")
        assert "learning_rule: stdp" in cfg
        assert "brainscales2" in cfg
        assert "tau_plus_ms:" in cfg

    def test_rejects_unknown_export_format(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
            export_learning_config,
        )

        params = generate_learning_params()
        with pytest.raises(ValueError, match="Unsupported learning config format"):
            export_learning_config(params, output_format="toml")

    def test_custom_weight_bounds(self) -> None:
        from sc_neurocore.compiler.intelligence import (
            generate_learning_params,
        )

        p = generate_learning_params(w_max=2.0, w_min=-1.0)
        assert p.w_max == 2.0
        assert p.w_min == -1.0
