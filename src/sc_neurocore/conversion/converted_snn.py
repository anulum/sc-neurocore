# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Portable converted dense integrate-and-fire network

"""Run exported dense IF networks with owned parameters and explicit replay."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from .if_parameters import FloatArray, IFParameters, OutputMode, parameter_snapshot
from .if_encoding import simulate_encoded
from .if_resources import DEFAULT_WORKING_BYTES
from .if_replay import IFReplayResult
from .if_dispatch import ReplayBackend, replay_backend


@dataclass(init=False)
class ConvertedSNN:
    """A dense IF stack with deterministic input encoding and replayable state.

    Parameters
    ----------
    weights : sequence of array_like
        Output-by-input matrices. Constructor inputs are copied to float64.
    biases : sequence of array_like or None
        Constant per-step currents in each layer's normalized threshold units.
    thresholds : sequence of float
        Positive finite thresholds, one per layer.
    T : int
        Positive timestep budget, at most ``2**53`` for exact count arithmetic.
    initial_membrane_fraction : float
        Default IF membrane preload in threshold units; QCFS uses ``0.5``.
    output_scale : float
        Positive finite source activation units per unit of decoded rate.
    output_mode : {'spikes', 'linear'}
        IF spike-count output or integrated signed linear readout. A linear
        final layer has no threshold/reset events and starts at zero.
    max_working_bytes : int
        Numeric buffer budget for constructor coefficient snapshots. Runtime
        calls accept their own budget; each defaults to 256 MiB.

    layer_membrane_fractions : sequence of float, optional
        Per-layer preloads for mixed activation routes. None uses the global
        fraction; the final linear integrator always starts at zero.

    Notes
    -----
    Public coefficient arrays are owned by this object. Replays snapshot and
    validate them again, so caller edits cannot bypass shape/domain admission.
    The ``dense-if-f64-sequential-v1`` profile orders input-column reductions
    and separates multiplication/addition instead of using BLAS reductions.
    """

    weights: list[FloatArray]
    biases: list[FloatArray | None]
    thresholds: list[float]
    T: int
    initial_membrane_fraction: float
    output_scale: float
    output_mode: OutputMode
    layer_membrane_fractions: list[float] | None

    def __init__(
        self,
        weights: Sequence[npt.ArrayLike],
        biases: Sequence[npt.ArrayLike | None],
        thresholds: Sequence[float],
        T: int,
        initial_membrane_fraction: float = 0.0,
        output_scale: float = 1.0,
        output_mode: OutputMode = "spikes",
        *,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
        layer_membrane_fractions: Sequence[float] | None = None,
    ) -> None:
        """Copy and validate all coupled parameters before exposing the network."""
        if type(T) is not int or not 0 < T <= 2**53:
            raise ValueError("T must be a positive integer no greater than 2**53")
        output_scale = float(output_scale)
        if not np.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("output scale must be finite and positive")
        parameters = parameter_snapshot(
            weights,
            biases,
            thresholds,
            initial_membrane_fraction,
            output_mode,
            max_working_bytes=max_working_bytes,
            layer_membrane_fractions=layer_membrane_fractions,
        )
        self.weights = [weight.copy() for weight in parameters.weights]
        self.biases = [None if bias is None else bias.copy() for bias in parameters.biases]
        self.thresholds = list(parameters.thresholds)
        self.T = T
        self.initial_membrane_fraction = parameters.initial_membrane_fraction
        self.output_scale = float(output_scale)
        self.output_mode = output_mode
        self.layer_membrane_fractions = (
            None if layer_membrane_fractions is None else list(parameters.layer_membrane_fractions)
        )

    @property
    def n_layers(self) -> int:
        """Return the current number of connected weighted layers."""
        return len(self.weights)

    def _parameters(self, max_working_bytes: int) -> IFParameters:
        """Freeze the current public coefficients for one complete operation."""
        return parameter_snapshot(
            self.weights,
            self.biases,
            self.thresholds,
            self.initial_membrane_fraction,
            self.output_mode,
            max_working_bytes=max_working_bytes,
            layer_membrane_fractions=self.layer_membrane_fractions,
        )

    def replay(
        self,
        inputs: npt.ArrayLike,
        *,
        initial_state: Sequence[npt.ArrayLike] | None = None,
        trace: bool = False,
        binary_inputs: bool = True,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
        backend: ReplayBackend = "auto",
    ) -> IFReplayResult:
        """Replay explicit frames, optionally continuing independently owned states.

        Parameters
        ----------
        inputs : array_like
            ``(steps, batch, input_neurons)`` frames in ``[0, 1]``.
        initial_state : sequence of array_like, optional
            ``(batch, output_neurons)`` state per layer, copied before execution.
        trace : bool
            Capture every post-step state and every IF spike event.
        binary_inputs : bool
            Require exact zero/one events; False admits bounded current drive.

        max_working_bytes : int
            Positive numeric buffer budget; excludes caller storage and runtime overhead.
        backend : {"auto", "numpy", "rust", "go", "mojo", "julia"}
            Explicit replay runtime; auto orders configured Rust, Go, Mojo and Julia runtimes
            by a validated measured comparison, else statically, before the NumPy floor.

        Returns
        -------
        IFReplayResult
            Incremental spike counts or the cumulative signed linear readout,
            final states and requested traces, all in independently owned buffers.
        """
        return replay_backend(
            self._parameters(max_working_bytes),
            inputs,
            initial_state=initial_state,
            trace=trace,
            binary_inputs=binary_inputs,
            max_working_bytes=max_working_bytes,
            backend=backend,
        )

    def run(
        self,
        x: npt.ArrayLike,
        *,
        input_mode: Literal["poisson", "constant"] = "poisson",
        seed: int = 42,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
        backend: ReplayBackend = "auto",
    ) -> FloatArray:
        """Encode and simulate one input vector or a batch for the stored budget.

        Parameters
        ----------
        x : array_like
            ``(input_neurons,)`` or ``(batch, input_neurons)`` values in ``[0, 1]``.
        input_mode : {'poisson', 'constant'}
            NumPy MT19937 Bernoulli event encoding or direct constant current.
            The legacy default uses seed 42 and the same row-major uniform stream.
        seed : int
            Unsigned 32-bit MT19937 seed used by Poisson event encoding.

        max_working_bytes : int
            Positive numeric buffer budget; excludes caller storage and runtime overhead.
        backend : {"auto", "numpy", "rust", "go", "mojo", "julia"}
            Explicit replay runtime; auto orders configured Rust, Go, Mojo and Julia runtimes
            by a validated measured comparison, else statically, before the NumPy floor.

        Returns
        -------
        ndarray
            Accumulated final-layer response, retaining the input batch shape.
            Spiking output is a count; linear output is an integrated current.

        Raises
        ------
        ValueError
            If encoding, seed, input geometry or domains are invalid.
        """
        return simulate_encoded(
            self._parameters(max_working_bytes),
            x,
            self.T,
            input_mode,
            seed,
            max_working_bytes,
            backend,
        )

    def rates(
        self,
        x: npt.ArrayLike,
        *,
        input_mode: Literal["poisson", "constant"] = "poisson",
        seed: int = 42,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
        backend: ReplayBackend = "auto",
    ) -> FloatArray:
        """Decode accumulated responses into the source activation's units.

        Parameters
        ----------
        x : array_like
            Input vector or batch in ``[0, 1]``.
        input_mode : {'poisson', 'constant'}
            Declared input encoding passed to :meth:`run`.
        seed : int
            MT19937 seed for Poisson input encoding.

        max_working_bytes : int
            Positive numeric buffer budget; excludes caller storage and runtime overhead.
        backend : {"auto", "numpy", "rust", "go", "mojo", "julia"}
            Explicit replay runtime; auto orders configured Rust, Go, Mojo and Julia runtimes
            by a validated measured comparison, else statically, before the NumPy floor.

        Returns
        -------
        ndarray
            Mean output response rescaled to the source ANN's activation units.
        """
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("output scale must be finite and positive")
        return (
            self.run(
                x,
                input_mode=input_mode,
                seed=seed,
                max_working_bytes=max_working_bytes,
                backend=backend,
            )
            / self.T
            * self.output_scale
        )

    def classify(
        self,
        x: npt.ArrayLike,
        *,
        max_working_bytes: int = DEFAULT_WORKING_BYTES,
        backend: ReplayBackend = "auto",
    ) -> npt.NDArray[np.intp] | np.intp:
        """Return the first maximal response index for a vector or each batch row.

        Parameters
        ----------
        x : array_like
            Input vector or batch in ``[0, 1]``.

        max_working_bytes : int
            Positive numeric buffer budget; excludes caller storage and runtime overhead.
        backend : {"auto", "numpy", "rust", "go", "mojo", "julia"}
            Explicit replay runtime; auto orders configured Rust, Go, Mojo and Julia runtimes
            by a validated measured comparison, else statically, before the NumPy floor.

        Returns
        -------
        integer or ndarray
            Index of the greatest accumulated output; ties select the first.
        """
        predictions: npt.NDArray[np.intp] | np.intp = np.argmax(
            self.run(x, max_working_bytes=max_working_bytes, backend=backend), axis=-1
        )
        return predictions
