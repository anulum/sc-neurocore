# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed descriptions of existing codec contracts

"""Immutable codec descriptions, wire contracts and runtime probe results.

Descriptions record existing interfaces and formats. They do not certify
performance, decoder hardening or complete native implementations. Convert a
description with :func:`dataclasses.asdict` for JSON serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

NativeCodecScope: TypeAlias = Literal[
    "none", "predictor_kernels", "signal_kernels", "channel_coder", "packet_coder"
]
CodecFidelity: TypeAlias = Literal["lossless", "lossy", "configurable"]


@dataclass(frozen=True, slots=True)
class CodecWireFormat:
    """Describe a currently readable or writable wire representation.

    Parameters
    ----------
    name : str
        Format identifier used by a variant's ``formats`` field.
    magic_hex : str or None
        Leading marker in hexadecimal, or None when there is no fixed marker.
    byte_order : str
        Field byte order; individual bit layouts are described in the format guide.
    dimensions_in_header : bool
        Whether dimensions occur in the representation, independently of whether
        the current public decoder still requires explicit dimensions.
    required_decode_metadata : tuple of str
        Configuration, sample counts or preceding state required outside the packet.
    checksum : str or None
        Embedded integrity check, or None when the format has none.
    readable, writable : bool
        Whether current public decoders and encoders accept or emit this format.
    notes : tuple of str
        Compatibility restrictions and interpretation details.
    """

    name: str
    magic_hex: str | None
    byte_order: str
    dimensions_in_header: bool
    required_decode_metadata: tuple[str, ...]
    checksum: str | None
    readable: bool
    writable: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodecVariant:
    """Describe one existing mode, predictor or event-timing choice.

    Parameters
    ----------
    name : str
        Value of the descriptor's variant parameter, or its sole fixed variant.
    fidelity : {'lossless', 'lossy', 'configurable'}
        Preservation of the descriptor's input under the documented input contract.
        Configurable variants depend on parameters described in ``notes``.
    native_scope : str
        Existing Rust component scope; a kernel or packet coder does not imply
        a standalone native implementation of its enclosing Python container.
    native_module : str or None
        Importable module containing the required native callables.
    native_symbols : tuple of str
        Callables needed for both directions of the declared native component.
    optional_modules : tuple of str
        Additional Python modules required by this particular variant.
    formats : tuple of str
        Format identifiers from the enclosing descriptor.
    notes : tuple of str
        Fidelity conditions, state rules and native implementation boundaries.
    """

    name: str
    fidelity: CodecFidelity
    native_scope: NativeCodecScope
    native_module: str | None
    native_symbols: tuple[str, ...]
    optional_modules: tuple[str, ...]
    formats: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodecDescriptor:
    """Describe a codec family or a distinct existing stream API.

    Parameters
    ----------
    name, title : str
        Stable lookup identifier and human-readable name.
    category : {'family', 'stream_api'}
        Six batch families are distinguished from two additional stream APIs.
    family : str
        Related algorithm family; stream APIs do not increase the family count.
    input_kind : {'binary_raster', 'numeric_waveform', 'int16_waveform'}
        Input contract, with time rows and channel columns.
    output_kind : str
        Decoded result kind, rather than a promise of a common decoder signature.
    encoder_entrypoint, decoder_entrypoint : str
        Importable public class paths. Batch codecs use one class in both directions.
    encode_method, decode_method : str
        Public methods on those classes.
    encoder_required_parameters, decoder_required_parameters : tuple of str
        Required constructor arguments; defaults remain defined by the public classes.
    decode_required_parameters : tuple of str
        Required decode method arguments after the payload.
    configuration_parameters : tuple of str
        Existing encoder constructor parameter names.
    variant_parameter : str or None
        Constructor parameter selecting a variant, or None for a fixed interface.
    backend_parameter : bool
        Whether the encoder accepts ``backend='auto'|'python'|'rust'`` explicitly.
        Automatic predictive kernels are not such a selector.
    input_dtype : str
        Intended public input values; coercion is defined by the actual class.
    input_layout : str
        Time-major two-dimensional layout.
    time_unit : str
        Timestamp and window-count unit; sample rate is external unless configured.
    stateful : bool
        Whether model state persists between packets on the same instance.
    variants : tuple of CodecVariant
        Existing choices, with the constructor default first.
    wire_formats : tuple of CodecWireFormat
        Read/write representations and their external metadata requirements.
    uses_codecs : tuple of str
        Other catalogue entries used inside this implementation.
    notes : tuple of str
        Input assumptions, additional methods and integration boundaries.
    """

    name: str
    title: str
    category: Literal["family", "stream_api"]
    family: str
    input_kind: Literal["binary_raster", "numeric_waveform", "int16_waveform"]
    output_kind: Literal[
        "binary_raster", "waveform_result", "int16_waveform", "spike_events_and_power"
    ]
    encoder_entrypoint: str
    decoder_entrypoint: str
    encode_method: str
    decode_method: str
    encoder_required_parameters: tuple[str, ...]
    decoder_required_parameters: tuple[str, ...]
    decode_required_parameters: tuple[str, ...]
    configuration_parameters: tuple[str, ...]
    variant_parameter: str | None
    backend_parameter: bool
    stateful: bool
    variants: tuple[CodecVariant, ...]
    wire_formats: tuple[CodecWireFormat, ...]
    input_dtype: str
    input_layout: str = "time_channel"
    time_unit: str = "sample_index"
    uses_codecs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def variant(self, name: str | None = None) -> CodecVariant:
        """Return a named variant, or the constructor default.

        Parameters
        ----------
        name : str or None
            Exact case-sensitive variant name; None selects the first variant.

        Returns
        -------
        CodecVariant
            Immutable variant description.

        Raises
        ------
        ValueError
            If the descriptor has no variants or the requested name is unknown.
        """
        if name is None and self.variants:
            return self.variants[0]
        for variant in self.variants:
            if variant.name == name:
                return variant
        available = ", ".join(variant.name for variant in self.variants)
        raise ValueError(f"Unknown variant {name!r} for {self.name!r}. Available: {available}")


@dataclass(frozen=True, slots=True)
class CodecAvailability:
    """Report an explicit runtime probe without selecting an encoder backend.

    Parameters
    ----------
    name, variant : str
        Catalogue and variant identifiers that were probed.
    python_available : bool
        Public Python entry points and optional variant modules are importable.
    rust_components_available : bool
        Required native callables are importable now. This does not certify a
        complete Rust container or change existing instances' cached backend choices.
    native_scope : str
        Scope of the native callables, as declared by the variant.
    backend_choices : tuple of str
        Explicit backend selector values available to the public Python interface.
        Empty when that interface has no backend parameter or cannot be loaded.
    diagnostics : tuple of str
        Missing modules, missing symbols or actual loader errors observed by the probe.
    """

    name: str
    variant: str
    python_available: bool
    rust_components_available: bool
    native_scope: NativeCodecScope
    backend_choices: tuple[str, ...]
    diagnostics: tuple[str, ...]
