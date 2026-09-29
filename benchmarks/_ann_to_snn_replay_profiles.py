# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public dense IF comparison profiles

"""Build deterministic, untrained workloads for full-response runtime comparisons."""

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_dispatch import ReplayBackend
from sc_neurocore.conversion.if_parameters import FloatArray, OutputMode, parameter_snapshot


@dataclass(frozen=True)
class ReplayProfile:
    """Declare one explicit-frame or encoded public runtime workload.

    Parameters
    ----------
    name : str
        Unique fixed workload and numerical-mode identifier.
    model : ConvertedSNN
        Owned untrained finite coefficient stack.
    values : ndarray
        Actual explicit frames or encoder input values.
    trace : bool
        Retain every state/event vector for explicit replay.
    encoding : {'constant', 'poisson'} or None
        Public run encoder, or explicit-frame replay when None.
    """

    name: str
    model: ConvertedSNN
    values: FloatArray
    trace: bool
    encoding: Literal["constant", "poisson"] | None = None

    def execute(self, backend: ReplayBackend) -> tuple[FloatArray, ...]:
        """Call the actual public runtime and retain every requested numerical vector.

        Parameters
        ----------
        backend : str
            Explicit provider; no automatic fallback participates in comparisons.

        Returns
        -------
        tuple of ndarray
            Owned output, final states and complete requested state/event traces.
        """
        if self.encoding is not None:
            return (
                self.model.run(self.values, input_mode=self.encoding, seed=71, backend=backend),
            )
        response = self.model.replay(
            self.values, trace=self.trace, binary_inputs=False, backend=backend
        )
        return (
            response.output,
            *response.final_state,
            *response.state_trace,
            *response.spike_trace,
        )

    def input_digest(self) -> str:
        """Bind all coupled parameters and actual drive to a reproducible workload digest.

        Returns
        -------
        str
            SHA-256 of declared metadata and shaped little-endian numerical inputs.
        """
        snapshot = parameter_snapshot(
            self.model.weights,
            self.model.biases,
            self.model.thresholds,
            self.model.initial_membrane_fraction,
            self.model.output_mode,
            layer_membrane_fractions=self.model.layer_membrane_fractions,
        )
        arrays = (*snapshot.weights, *(b for b in snapshot.biases if b is not None), self.values)
        metadata = (
            self.name,
            self.model.T,
            snapshot.thresholds,
            snapshot.layer_membrane_fractions,
            snapshot.output_mode,
            tuple(bias is None for bias in snapshot.biases),
            self.trace,
            self.encoding,
        )
        return hashlib.sha256(
            repr(metadata).encode() + response_digest(arrays).encode()
        ).hexdigest()


def response_digest(arrays: tuple[FloatArray, ...]) -> str:
    """Hash every shaped response in order using canonical little-endian float64 bytes.

    Parameters
    ----------
    arrays : tuple of ndarray
        Ordered complete output, state and event vectors.

    Returns
    -------
    str
        SHA-256 binding each shape and every numerical response bit.
    """
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(repr(array.shape).encode())
        digest.update(array.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def replay_profiles() -> list[ReplayProfile]:
    """Build twenty deterministic workloads across modes, traces, empty axes and encoders.

    Returns
    -------
    list of ReplayProfile
        Fixed seeded finite untrained workloads shared by every runtime.
    """
    rng = np.random.default_rng(71)
    modes: tuple[OutputMode, ...] = ("spikes", "linear")
    encoders: tuple[Literal["constant", "poisson"], ...] = ("constant", "poisson")
    profiles = []
    for label, steps, batch, inputs, widths in (
        ("single", 64, 1, 1, (1,)),
        ("dense", 64, 8, 16, (32, 8)),
        ("blocks", 129, 4, 16, (32, 8)),
        ("empty", 0, 2, 16, (32, 8)),
    ):
        weights = []
        biases: list[FloatArray | None] = []
        previous = inputs
        for index, width in enumerate(widths):
            weights.append(rng.normal(0.1, 0.2, (width, previous)))
            biases.append(None if index == 0 else rng.normal(0, 0.02, width))
            previous = width
        values = rng.random((steps, batch, inputs))
        for mode in modes:
            model = ConvertedSNN(
                weights,
                biases,
                [0.75] * len(widths),
                T=max(1, steps),
                output_mode=mode,
                layer_membrane_fractions=[0.5] * len(widths),
            )
            for trace in (False, True):
                profiles.append(
                    ReplayProfile(f"{label}:{mode}:trace={trace}", model, values, trace)
                )
            if label == "blocks":
                encoded = rng.random((batch, inputs))
                for encoding in encoders:
                    profiles.append(
                        ReplayProfile(f"{label}:{mode}:{encoding}", model, encoded, False, encoding)
                    )
    return profiles
