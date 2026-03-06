# SPDX-License-Identifier: AGPL-3.0-or-later
"""Custom exception hierarchy for SC-NeuroCore."""


class SCNeuroError(Exception):
    """Base exception for all SC-NeuroCore errors."""


class BitstreamOverflowError(SCNeuroError):
    """Bitstream length exceeds the maximum supported width."""


class SeedCollisionError(SCNeuroError):
    """Two encoders received the same LFSR seed, breaking decorrelation."""


class BitwidthMismatchError(SCNeuroError):
    """Operands have incompatible fixed-point widths."""


class CoverageGateError(SCNeuroError):
    """Test coverage fell below the required threshold."""


class HardwareSimMismatchError(SCNeuroError):
    """Python golden model and Verilog RTL produced different results."""


class IRCompilationError(SCNeuroError):
    """IR graph failed verification or code generation."""
