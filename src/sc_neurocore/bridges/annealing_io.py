# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing serialization and visualization

"""Deterministic model export and human-readable visualization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges.annealing_models import IsingModel, QUBOModel


def _install_temporary(source: Path, destination: Path) -> None:
    """Atomically install a completed temporary file."""
    os.replace(source, destination)


def _atomic_json_write(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON file with deterministic UTF-8 content."""
    destination = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _install_temporary(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def export_ising_json(model: IsingModel, path: str | Path) -> None:
    """Write a canonical JSON representation of an Ising model."""
    if not isinstance(model, IsingModel):
        raise ValueError("model must be an IsingModel")
    _atomic_json_write(
        path,
        {
            "type": "ising",
            "n_qubits": model.n_qubits,
            "source": model.source,
            "offset": model.offset,
            "h": {str(index): value for index, value in model.h.items()},
            "J": {f"{first},{second}": value for (first, second), value in model.J.items()},
            "qubit_labels": {str(index): label for index, label in model.qubit_labels.items()},
        },
    )


def export_qubo_json(model: QUBOModel, path: str | Path) -> None:
    """Write a canonical JSON representation of a QUBO model."""
    if not isinstance(model, QUBOModel):
        raise ValueError("model must be a QUBOModel")
    _atomic_json_write(
        path,
        {
            "type": "qubo",
            "n_qubits": model.n_qubits,
            "source": model.source,
            "offset": model.offset,
            "Q": {f"{first},{second}": value for (first, second), value in model.Q.items()},
            "qubit_labels": {str(index): label for index, label in model.qubit_labels.items()},
        },
    )


def export_bqm(model: IsingModel) -> Any | None:
    """Return a dimod spin BQM, or ``None`` when dimod is unavailable."""
    if not isinstance(model, IsingModel):
        raise ValueError("model must be an IsingModel")
    return backends.build_spin_bqm(model.h, model.J, model.offset)


def visualize_ising(model: IsingModel) -> str:
    """Return a stable Unicode text rendering of fields and couplings."""
    if not isinstance(model, IsingModel):
        raise ValueError("model must be an IsingModel")
    lines = [
        f"┌{'=' * 50}┐",
        f"│ Ising Model: {model.source:<34} │",
        f"│ Qubits: {model.n_qubits:<4}  Couplers: {len(model.J):<5}          │",
        f"│ Offset: {model.offset:<40.4f} │",
        f"└{'=' * 50}┘",
        "",
        "  Biases (h):",
    ]
    for index in sorted(model.h):
        label = model.qubit_labels.get(index, f"q{index}")
        bar = "█" * min(int(abs(model.h[index]) * 20), 20)
        sign = "+" if model.h[index] >= 0.0 else "-"
        lines.append(f"    {label:>8}: {sign}{bar:<20} ({model.h[index]:+.4f})")

    lines.extend(("", "  Couplings (J):"))
    for first, second in sorted(model.J):
        first_label = model.qubit_labels.get(first, f"q{first}")
        second_label = model.qubit_labels.get(second, f"q{second}")
        strength = model.J[(first, second)]
        kind = "ferro" if strength < 0.0 else "anti"
        lines.append(f"    {first_label:>8} ─── {second_label:<8}: {strength:+.4f} [{kind}]")
    return "\n".join(lines)
