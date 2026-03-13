# SPDX-License-Identifier: AGPL-3.0-or-later
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
    """Invalid configuration parameter (layer size, threshold, etc.)."""


class SCWeightError(SCNeuroError, ValueError):
    """Weight value or shape mismatch."""


class SCCompilerError(SCNeuroError, ValueError):
    """Compiler pipeline configuration or target error."""


# --- Runtime exceptions ---


class SCDependencyError(SCNeuroError, RuntimeError):
    """Optional dependency (JAX, Torch, PennyLane, Qiskit) not installed."""


class SCHardwareError(SCNeuroError, RuntimeError):
    """FPGA/hardware driver or bitstream error."""


# --- Existing specific exceptions ---


class BitstreamOverflowError(SCEncodingError):
    """Bitstream length exceeds the maximum supported width."""


class SeedCollisionError(SCNeuroError):
    """Two encoders received the same LFSR seed, breaking decorrelation."""


class BitwidthMismatchError(SCNeuroError):
    """Operands have incompatible fixed-point widths."""


class CoverageGateError(SCNeuroError):
    """Test coverage fell below the required threshold."""


class HardwareSimMismatchError(SCHardwareError):
    """Python golden model and Verilog RTL produced different results."""


class IRCompilationError(SCCompilerError):
    """IR graph failed verification or code generation."""
