# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (emitter_config) from former test_kuramoto_rtl.py

from __future__ import annotations

from tests.test_hdl_gen.kuramoto_rtl_support import *  # noqa: F403

def test_kuramoto_emitter_has_expected_ports_and_helpers() -> None:
    emitter = KuramotoEmitter(
        module_name="kuramoto_top",
        n_oscillators=3,
        omegas=[0.9, 1.0, 1.1],
        initial_phases=[0.0, 0.2, 0.4],
    )
    code = emitter.generate()
    assert "module kuramoto_top" in code
    assert "input wire step_en" in code
    assert "output reg update_done" in code
    assert "function automatic signed [DATA_WIDTH-1:0] sin_lut;" in code
    assert "wire signed [DATA_WIDTH-1:0] phase_diff_0_1" in code
    assert "assign phase_bus[71:48] = phase_reg_2;" in code


def test_kuramoto_emitter_rejects_configuration_mismatch() -> None:
    try:
        KuramotoEmitter(
            n_oscillators=3,
            omegas=[1.0, 1.1],
            initial_phases=[0.0, 0.1, 0.2],
        )
    except ValueError as exc:
        assert "omegas length must equal n_oscillators" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for omega length mismatch")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_oscillators": 0}, "n_oscillators must be >= 1"),
        ({"data_width": 15}, "data_width must be >= 16"),
        ({"fraction": 0}, "fraction must satisfy 0 < fraction < data_width"),
        ({"fraction": 24}, "fraction must satisfy 0 < fraction < data_width"),
        ({"lut_size": 8}, "lut_size must be a power of two >= 16"),
        ({"lut_size": 48}, "lut_size must be a power of two >= 16"),
        (
            {"n_oscillators": 2, "initial_phases": [0.0]},
            "initial_phases length must equal n_oscillators",
        ),
    ],
)
def test_kuramoto_emitter_rejects_invalid_structural_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KuramotoEmitter(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"dt": float("nan")}, "dt must be finite and positive"),
        ({"coupling": float("inf")}, "coupling must be finite"),
        ({"omegas": [1.0, float("nan")]}, "omegas must contain only finite values"),
        (
            {"initial_phases": [0.0, float("-inf")]},
            "initial_phases must contain only finite values",
        ),
    ],
)
def test_kuramoto_emitter_rejects_invalid_numerical_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    base_kwargs: dict[str, Any] = {
        "n_oscillators": 2,
        "omegas": [1.0, 1.1],
        "initial_phases": [0.0, 0.2],
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        KuramotoEmitter(**base_kwargs)


def test_kuramoto_emitter_rejects_fixed_point_format_that_cannot_hold_phase_modulus() -> None:
    with pytest.raises(ValueError, match="fixed-point format cannot represent 2pi"):
        KuramotoEmitter(data_width=16, fraction=15)


def test_kuramoto_emitter_rejects_configuration_that_requires_multi_wrap_step() -> None:
    with pytest.raises(ValueError, match="single-step phase advance must stay below 2pi"):
        KuramotoEmitter(n_oscillators=1, omegas=[100.0], dt=0.1)


