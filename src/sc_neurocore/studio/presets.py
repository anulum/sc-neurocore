# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Curated preset experiments for Studio

from __future__ import annotations

from typing import Any

PRESETS: list[dict[str, Any]] = [
    {
        "id": "threshold",
        "title": "Threshold Behavior",
        "description": "Gradually increase current to find the spiking threshold. Below threshold: subthreshold oscillations. Above: tonic spiking. The transition is sharp in IF models, smooth in conductance models.",
        "mode": "ode",
        "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
        "threshold": "v > -50",
        "reset": "v = -65",
        "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        "init": {"v": -65.0},
        "dt": 0.1,
        "duration": 200.0,
        "current": 15.0,
        "protocol": "ramp",
        "suggested_view": "trace",
    },
    {
        "id": "type1_vs_type2",
        "title": "Type I vs Type II Excitability",
        "description": "Type I neurons (e.g., QIF/theta) can fire at arbitrarily low rates near threshold. Type II neurons (e.g., HH) jump to a minimum frequency. Compare using f-I curves.",
        "mode": "model",
        "model_name": "HodgkinHuxleyNeuron",
        "current": 8.0,
        "duration": 300.0,
        "protocol": "constant",
        "suggested_view": "fi-curve",
    },
    {
        "id": "adaptation",
        "title": "Spike-Frequency Adaptation",
        "description": "The AdEx model shows decreasing firing rate over time due to the adaptation current w. The first ISI is shorter than later ones. Visible in the ISI histogram as a rightward tail.",
        "mode": "model",
        "model_name": "AdExNeuron",
        "current": 500.0,
        "duration": 500.0,
        "protocol": "step",
        "suggested_view": "isi",
    },
    {
        "id": "bursting",
        "title": "Bursting Patterns",
        "description": "The Hindmarsh-Rose model produces characteristic burst-pause-burst patterns. The phase portrait shows the slow variable z driving transitions between quiescent and active states.",
        "mode": "model",
        "model_name": "HindmarshRoseNeuron",
        "current": 3.0,
        "duration": 500.0,
        "protocol": "constant",
        "suggested_view": "phase",
    },
    {
        "id": "refractory",
        "title": "Refractory Period",
        "description": "After a spike, the HH model enters a refractory period where no stimulus can trigger another spike. Use pulse current to probe recovery: short inter-pulse intervals fail, longer ones succeed.",
        "mode": "model",
        "model_name": "HodgkinHuxleyNeuron",
        "current": 20.0,
        "duration": 100.0,
        "protocol": "pulse",
        "suggested_view": "trace",
    },
    {
        "id": "resonance",
        "title": "Subthreshold Resonance",
        "description": "Some neurons preferentially respond to inputs at a specific frequency. The GIF model shows resonance near its intrinsic frequency. Use f-I curve to find the preferred input strength.",
        "mode": "model",
        "model_name": "GIFNeuron",
        "current": 5.0,
        "duration": 300.0,
        "protocol": "constant",
        "suggested_view": "trace",
    },
    {
        "id": "fitzhugh_oscillator",
        "title": "Relaxation Oscillation",
        "description": "The FitzHugh-Nagumo model is a 2D reduction of HH. The phase portrait reveals the limit cycle: fast voltage jumps along the v-nullcline, slow recovery along the w-nullcline.",
        "mode": "ode",
        "equations": ["dv/dt = v - v**3 / 3 - w + I", "dw/dt = epsilon * (v + a - b * w)"],
        "threshold": "",
        "reset": "",
        "params": {"epsilon": 0.08, "a": 0.7, "b": 0.8},
        "init": {"v": -1.0, "w": -0.5},
        "dt": 0.1,
        "duration": 400.0,
        "current": 0.5,
        "protocol": "constant",
        "suggested_view": "phase",
    },
    {
        "id": "fpga_precision",
        "title": "Float vs Q8.8 Precision",
        "description": "Compare float64 simulation with Q8.8 fixed-point (the format used in FPGA synthesis). Small quantisation errors accumulate over time. Use the Precision tab to see the error trace.",
        "mode": "ode",
        "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
        "threshold": "v > -50",
        "reset": "v = -65",
        "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        "init": {"v": -65.0},
        "dt": 0.1,
        "duration": 200.0,
        "current": 30.0,
        "protocol": "constant",
        "suggested_view": "precision",
        "studio_actions": [
            {
                "id": "auto_tune_adaptive_precision",
                "label": "Auto-Tune <0.1% Error",
                "method": "POST",
                "endpoint": "/api/adaptive-precision/auto-tune",
                "objective": "minimal_luts_under_error_target",
                "target_error_percent": 0.1,
                "payload_template": {
                    "layer_weights": [[[0.2, 0.4], [0.6, 0.8]], [0.3, 0.7]],
                    "layer_names": ["input", "readout"],
                    "target_error_percent": 0.1,
                    "min_bits": 4,
                    "max_bits": 16,
                    "min_length": 32,
                    "max_length": 4096,
                    "confidence": 0.95,
                },
            },
            {
                "id": "generate_adaptive_precision_formal_bundle",
                "label": "Generate Formal Bundle",
                "method": "POST",
                "endpoint": "/api/adaptive-precision/formal-bundle",
                "evidence_boundary": (
                    "bundle_generation_only_no_symbiyosys_execution_no_silicon_claim"
                ),
                "payload_template": {
                    "layer_weights": [[[0.2, 0.4], [0.6, 0.8]], [0.3, 0.7]],
                    "layer_names": ["input", "readout"],
                    "target_error_percent": 0.1,
                    "module_name": "adaptive_precision_plan",
                },
            },
        ],
    },
    {
        "id": "chaos",
        "title": "Chaotic Spiking",
        "description": "The Chialvo map neuron exhibits chaotic dynamics: sensitive dependence on initial conditions, irregular spike timing, broad ISI distribution. The phase portrait shows a strange attractor.",
        "mode": "model",
        "model_name": "ChialvoMapNeuron",
        "current": 0.04,
        "duration": 500.0,
        "protocol": "constant",
        "suggested_view": "phase",
    },
    {
        "id": "hardware_loihi",
        "title": "Neuromorphic Hardware: Loihi2",
        "description": "The Loihi2 neuron uses integer fixed-point arithmetic matching Intel's neuromorphic chip. Compare its discrete dynamics with the smooth HH model. Hardware neurons trade precision for power efficiency.",
        "mode": "model",
        "model_name": "Loihi2Neuron",
        "current": 50.0,
        "duration": 200.0,
        "protocol": "constant",
        "suggested_view": "trace",
    },
]


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "description": p["description"],
            "suggested_view": p.get("suggested_view", "trace"),
        }
        for p in PRESETS
    ]


def get_preset(preset_id: str) -> dict[str, Any] | None:
    return next((p for p in PRESETS if p["id"] == preset_id), None)


def get_preset_actions(preset_id: str) -> list[dict[str, Any]]:
    preset = get_preset(preset_id)
    if not preset:
        return []
    actions = preset.get("studio_actions", [])
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def get_preset_action(preset_id: str, action_id: str) -> dict[str, Any] | None:
    actions = get_preset_actions(preset_id)
    return next((action for action in actions if action.get("id") == action_id), None)


def list_preset_action_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset in PRESETS:
        preset_id = preset.get("id")
        if not isinstance(preset_id, str):
            continue
        actions = preset.get("studio_actions", [])
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_id = action.get("id")
            endpoint = action.get("endpoint")
            method = action.get("method")
            if not isinstance(action_id, str) or not isinstance(endpoint, str):
                continue
            rows.append(
                {
                    "preset_id": preset_id,
                    "action_id": action_id,
                    "endpoint": endpoint,
                    "method": method if isinstance(method, str) else None,
                }
            )
    return rows
