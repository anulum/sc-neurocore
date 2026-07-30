# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sampled Yoon asynchronous pulse sigma-delta encoder

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_LIMIT = 1.0e12


@dataclass
class SigmaDeltaNeuron:
    """Discrete-time specialization of Yoon's APSDM encoder.

    ``sigma`` is the output of the disclosed integrating prefilter and
    ``reconstruction`` is the local feedback signal. Each sample implements
    equations 20-27 and the exponentially decaying reconstruction of equation
    40 from WO2016022241A1: integrate the input, decay the reconstruction,
    compare their difference with the upper threshold ``delta / 2``, then add
    one reconstruction quantum ``delta`` for a unipolar event.

    This clocked specialization is source-bound but does not claim exact
    continuous-time event timing. The former bipolar accumulator is retained
    separately as :class:`SCSigmaDeltaAccumulatorNeuron`.

    Reference: Y. C. Yoon, IEEE TNNLS 28(5), 2017,
    doi:10.1109/TNNLS.2016.2526029; primary equations WO2016022241A1.
    """

    sigma: float = 0.0
    reconstruction: float = 0.0
    delta: float = 1.0
    tau_reconstruction: float = 10.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate the complete encoder state and configuration."""
        self._validate_state()

    @property
    def error(self) -> float:
        """Return the current prefilter-minus-reconstruction error."""
        return self.sigma - self.reconstruction

    def _validate_state(self) -> None:
        values = (
            self.sigma,
            self.reconstruction,
            self.delta,
            self.tau_reconstruction,
            self.dt,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SigmaDeltaNeuron state and parameters must be finite")
        if abs(self.sigma) > _STATE_LIMIT or abs(self.reconstruction) > _STATE_LIMIT:
            raise ValueError("SigmaDeltaNeuron state is outside the safety envelope")
        if self.delta <= 0.0 or self.tau_reconstruction <= 0.0 or self.dt <= 0.0:
            raise ValueError("SigmaDeltaNeuron delta, time constant, and dt must be positive")

    def step(self, current: float) -> int:
        """Advance one atomic sampled APSDM transition and return 0 or 1."""
        if not math.isfinite(current):
            raise ValueError("SigmaDeltaNeuron input current must be finite")
        self._validate_state()
        sigma = self.sigma + self.dt * current
        reconstruction = self.reconstruction * math.exp(-self.dt / self.tau_reconstruction)
        if not math.isfinite(sigma) or not math.isfinite(reconstruction):
            raise ValueError("SigmaDeltaNeuron candidate state must be finite")
        spike = sigma - reconstruction >= 0.5 * self.delta
        if spike:
            reconstruction += self.delta
        if (
            not math.isfinite(reconstruction)
            or abs(sigma) > _STATE_LIMIT
            or abs(reconstruction) > _STATE_LIMIT
        ):
            raise ValueError("SigmaDeltaNeuron candidate left the safety envelope")
        self.sigma = sigma
        self.reconstruction = reconstruction
        return int(spike)

    def reset(self) -> None:
        """Clear both dynamic encoder states while retaining configuration."""
        self.sigma = 0.0
        self.reconstruction = 0.0
