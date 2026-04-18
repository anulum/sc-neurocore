# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — exception hierarchy

"""SC-NeuroCore exception hierarchy.

All public exceptions inherit from ``SCNeuroError`` so callers can
catch broad or narrow::

    try:
        layer.forward(bad_input)
    except SCEncodingError:
        ...
    except SCNeuroError:
        ...
"""

from __future__ import annotations


class SCNeuroError(Exception):
    """Base exception for all SC-NeuroCore errors."""


# --- Domain exceptions (ValueError subclasses for backward compat) ---


class SCEncodingError(SCNeuroError, ValueError):
    """Probability or bitstream value outside valid range."""


class SCConfigError(SCNeuroError, ValueError):
    """Invalid configuration parameter (layer size, threshold, etc.).

    Reserved for future use — v3.14.0 callers raise plain
    ``ValueError`` for configuration errors. Migration tracked
    under task #36.
    """


class SCWeightError(SCNeuroError, ValueError):
    """Weight value or shape mismatch.

    Reserved for future use — v3.14.0 weight loaders raise plain
    ``ValueError`` or ``KeyError`` for shape/value errors.
    Migration tracked under task #36.
    """


class SCCompilerError(SCNeuroError, ValueError):
    """Compiler pipeline configuration or target error.

    Raised by ``sc_neurocore.compiler.pipeline`` for unknown FPGA
    target, invalid output names, and path-escape detection.
    """


# --- Runtime exceptions ---


class SCDependencyError(SCNeuroError, RuntimeError):
    """Optional dependency (JAX, Torch, PennyLane, Qiskit) not installed.

    Raised by JAX backend, quantum hardware bridge, learning
    callbacks (W&B, TensorBoard) and similar feature-gated paths.
    """


class SCHardwareError(SCNeuroError, RuntimeError):
    """FPGA/hardware driver or bitstream error.

    Raised by the PYNQ driver (missing IP block) and by Verilog
    export when the model is not in the LIF whitelist.
    """


# --- Existing specific exceptions ---


class BitstreamOverflowError(SCEncodingError):
    """Bitstream length exceeds the maximum supported width.

    Reserved for future use — v3.14.0 bitstream encoders saturate
    silently or raise plain ``ValueError`` for related issues
    (e.g. the recent Q8.8 dt-underflow guard). Migration tracked
    under task #36.
    """


class SeedCollisionError(SCNeuroError):
    """Two encoders received the same LFSR seed, breaking decorrelation.

    Reserved for future use — v3.14.0 the encoder API does not
    detect seed collisions. Implementing this would require a
    global seed registry across all encoders. Tracked under
    task #36.
    """


class BitwidthMismatchError(SCNeuroError):
    """Operands have incompatible fixed-point widths.

    Reserved for future use — v3.14.0 the compiler hard-codes
    Q8.8 throughout, so multi-width layouts are not supported.
    This exception will fire once the compiler accepts mixed
    widths. Tracked under task #36.
    """


class CoverageGateError(SCNeuroError):
    """Test coverage fell below the required threshold.

    Reserved for future use — v3.14.0 coverage gating lives in
    CI workflow YAML, not in Python code. The exception is here
    so coverage-aware tooling has a stable raise target.
    Tracked under task #36.
    """


class HardwareSimMismatchError(SCHardwareError):
    """Python golden model and Verilog RTL produced different results.

    Reserved for future use — v3.14.0 the cosim suite raises
    plain ``AssertionError`` from pytest assertions instead of
    this typed exception. Migration tracked under task #36.
    """


class IRCompilationError(SCCompilerError):
    """IR graph failed verification or code generation.

    Reserved for future use — v3.14.0 the compiler raises the
    parent ``SCCompilerError`` (line 90 of pipeline.py)
    rather than this leaf type. Migration tracked under task #36.
    """
