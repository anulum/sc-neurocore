# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python code generation from Studio configurations

from __future__ import annotations

from typing import Any


def generate_model_script(
    model_name: str,
    params: dict[str, float] | None = None,
    duration: float = 100.0,
    current: float = 10.0,
    dt: float = 0.1,
) -> str:
    """Generate a standalone Python script that reproduces the current simulation."""
    param_args = ""
    if params:
        non_default = {k: v for k, v in params.items()}
        if non_default:
            param_args = ", ".join(f"{k}={v}" for k, v in non_default.items())

    n_steps = int(duration / dt)

    return f"""import numpy as np
from sc_neurocore.neurons.models import {model_name}

neuron = {model_name}({param_args})
n_steps = {n_steps}
voltages = np.empty(n_steps)
spikes = []

for t in range(n_steps):
    spike = neuron.step(current={current})
    voltages[t] = neuron.v
    if spike:
        spikes.append(t)

time = np.arange(n_steps) * {dt}
print(f"{{len(spikes)}} spikes in {{n_steps}} steps ({{len(spikes) / ({duration} / 1000):.1f}} Hz)")

# Plot (uncomment if matplotlib available)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10, 3))
# ax.plot(time, voltages, linewidth=0.8)
# for s in spikes:
#     ax.axvline(s * {dt}, color='red', alpha=0.3, linewidth=0.5)
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Voltage (mV)')
# ax.set_title('{model_name}')
# plt.tight_layout()
# plt.show()
"""


def generate_ode_script(
    equations: list[str],
    threshold: str | None = None,
    reset: str | None = None,
    params: dict[str, float] | None = None,
    init: dict[str, float] | None = None,
    duration: float = 100.0,
    current: float = 10.0,
    dt: float = 0.1,
) -> str:
    """Generate a standalone Python script for a custom ODE simulation."""
    eq_lines = ",\n        ".join(f'"{eq}"' for eq in equations)
    param_str = repr(params) if params else "{}"
    init_str = repr(init) if init else "{}"
    n_steps = int(duration / dt)

    return f"""import numpy as np
from sc_neurocore.neurons.equation_builder import from_equations

neuron = from_equations(
        {eq_lines},
        threshold={repr(threshold)},
        reset={repr(reset)},
        params={param_str},
        init={init_str},
        dt={dt},
)

n_steps = {n_steps}
states = {{k: np.empty(n_steps) for k in neuron.state}}
spikes = []

for t in range(n_steps):
    spike = neuron.step(I={current})
    for k in neuron.state:
        states[k][t] = neuron.state[k]
    if spike:
        spikes.append(t)

time = np.arange(n_steps) * {dt}
print(f"{{len(spikes)}} spikes in {{n_steps}} steps")

# Plot
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10, 3))
# for k, v in states.items():
#     ax.plot(time, v, label=k, linewidth=0.8)
# ax.legend()
# ax.set_xlabel('Time (ms)')
# plt.tight_layout()
# plt.show()
"""


def generate_oneliner(
    model_name: str | None = None,
    params: dict[str, float] | None = None,
    current: float = 10.0,
) -> str:
    """Generate a copy-paste Python one-liner for notebook use."""
    if model_name:
        args = ", ".join(f"{k}={v}" for k, v in (params or {}).items())
        return f"from sc_neurocore.neurons.models import {model_name}; n = {model_name}({args}); [n.step(current={current}) for _ in range(1000)]"
    return ""


def classify_firing_pattern(
    spikes: list[int],
    n_steps: int,
    dt: float,
) -> dict[str, Any]:
    """Classify the firing pattern from spike indices."""
    if len(spikes) == 0:
        return {"pattern": "silent", "description": "No spikes detected"}

    duration_s = n_steps * dt / 1000.0
    rate = len(spikes) / duration_s if duration_s > 0 else 0

    if len(spikes) < 3:
        return {
            "pattern": "single_spike",
            "description": f"Only {len(spikes)} spike(s)",
            "rate_hz": round(rate, 1),
        }

    import numpy as np

    isis = np.diff(spikes).astype(float) * dt
    isi_mean = float(np.mean(isis))
    isi_cv = float(np.std(isis) / isi_mean) if isi_mean > 0 else 0

    # Detect bursting: look for bimodal ISI (short intra-burst + long inter-burst)
    if len(isis) >= 4:
        sorted_isis = np.sort(isis)
        median_isi = float(np.median(isis))
        short = isis[isis < median_isi * 0.5]
        long = isis[isis > median_isi * 1.5]
        if len(short) > 1 and len(long) > 0:
            ratio = float(np.mean(long)) / float(np.mean(short)) if np.mean(short) > 0 else 1
            if ratio > 3:
                return {
                    "pattern": "bursting",
                    "description": f"Burst-pause pattern (ISI ratio {ratio:.1f}x)",
                    "rate_hz": round(rate, 1),
                    "isi_cv": round(isi_cv, 3),
                    "burst_isi_ms": round(float(np.mean(short)), 2),
                    "inter_burst_ms": round(float(np.mean(long)), 2),
                }

    # Detect adaptation: ISIs increase over time
    if len(isis) >= 5:
        first_third = np.mean(isis[: len(isis) // 3])
        last_third = np.mean(isis[-len(isis) // 3 :])
        if last_third > first_third * 1.3:
            return {
                "pattern": "adapting",
                "description": f"Spike-frequency adaptation ({first_third:.1f}→{last_third:.1f} ms ISI)",
                "rate_hz": round(rate, 1),
                "isi_cv": round(isi_cv, 3),
            }

    if isi_cv < 0.15:
        return {
            "pattern": "tonic",
            "description": f"Regular tonic firing (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }

    if isi_cv < 0.5:
        return {
            "pattern": "irregular",
            "description": f"Irregular spiking (CV={isi_cv:.3f})",
            "rate_hz": round(rate, 1),
            "isi_cv": round(isi_cv, 3),
        }

    return {
        "pattern": "chaotic",
        "description": f"Highly irregular/chaotic (CV={isi_cv:.3f})",
        "rate_hz": round(rate, 1),
        "isi_cv": round(isi_cv, 3),
    }
