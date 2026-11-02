# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec discovery and explicit runtime availability

"""Describe existing codecs without constructing them or probing optional backends."""

from __future__ import annotations

from importlib import import_module

from ._catalogue_data import DESCRIPTORS
from .catalogue_types import CodecAvailability, CodecDescriptor


def list_codec_descriptors() -> tuple[CodecDescriptor, ...]:
    """Return immutable descriptions of six families and two stream APIs.

    Returns
    -------
    tuple of CodecDescriptor
        Descriptions sorted by name, including optional variants independently
        of installation. Use ``probe_codec`` for an explicit runtime check.
    """
    return DESCRIPTORS


def describe_codec(name: str) -> CodecDescriptor:
    """Look up an existing codec's interface and wire contracts.

    Parameters
    ----------
    name : str
        Exact case-sensitive catalogue identifier.

    Returns
    -------
    CodecDescriptor
        Static immutable description. Lookup performs no backend probing.

    Raises
    ------
    ValueError
        If the name is not in the catalogue.
    """
    for descriptor in DESCRIPTORS:
        if descriptor.name == name:
            return descriptor
    available = ", ".join(item.name for item in DESCRIPTORS)
    raise ValueError(f"Unknown codec {name!r}. Available: {available}")


def _check_module(module_name: str, symbols: tuple[str, ...], diagnostics: list[str]) -> str:
    """Check actual imports and callable exports, retaining loader diagnostics.

    Parameters
    ----------
    module_name : str
        Module owning the required public exports.
    symbols : tuple of str
        Required callable names, including lazily resolved module attributes.
    diagnostics : list of str
        Destination for actual import errors and missing callable names.

    Returns
    -------
    str
        ``available``, ``missing`` or ``broken`` for the observed module state.
    """
    try:
        module = import_module(module_name)
        missing = tuple(symbol for symbol in symbols if not callable(getattr(module, symbol, None)))
    except ImportError as exc:
        diagnostics.append(f"{module_name}: {type(exc).__name__}: {exc}")
        return "missing"
    except (OSError, RuntimeError) as exc:
        diagnostics.append(f"{module_name}: {type(exc).__name__}: {exc}")
        return "broken"
    if missing:
        diagnostics.append(f"{module_name}: missing callable(s): {', '.join(missing)}")
        return "missing"
    return "available"


def probe_codec(name: str, variant: str | None = None) -> CodecAvailability:
    """Probe public entry points and actual native component imports explicitly.

    Parameters
    ----------
    name : str
        Exact catalogue identifier.
    variant : str or None
        Mode, predictor or timing choice; None selects the constructor default.

    Returns
    -------
    CodecAvailability
        Import availability and diagnostics. This probe neither constructs a
        codec nor qualifies data, hardware, performance or full Rust containers.
        Predictive kernels are cached by their owning module; installing an engine
        later does not update existing instances. Start a fresh process in that case.
        Native loader OSError or RuntimeError excludes ``auto`` from advertised
        choices; ImportError permits the classes' documented Python fallback.

    Raises
    ------
    ValueError
        If the codec or variant name is unknown.
    """
    descriptor = describe_codec(name)
    selected = descriptor.variant(variant)
    diagnostics: list[str] = []
    python_available = True
    for entrypoint in dict.fromkeys((descriptor.encoder_entrypoint, descriptor.decoder_entrypoint)):
        module, _, symbol = entrypoint.rpartition(".")
        if _check_module(module, (symbol,), diagnostics) != "available":
            python_available = False
    for module in selected.optional_modules:
        if _check_module(module, (), diagnostics) != "available":
            python_available = False

    native_status = "missing"
    if selected.native_module is not None:
        native_status = _check_module(selected.native_module, selected.native_symbols, diagnostics)
    choices: tuple[str, ...] = ()
    if descriptor.backend_parameter and python_available:
        choices = ("python",)
        if native_status != "broken":
            choices = ("auto", "python")
        if native_status == "available":
            choices += ("rust",)
    return CodecAvailability(
        name=descriptor.name,
        variant=selected.name,
        python_available=python_available,
        rust_components_available=native_status == "available",
        native_scope=selected.native_scope,
        backend_choices=choices,
        diagnostics=tuple(diagnostics),
    )
