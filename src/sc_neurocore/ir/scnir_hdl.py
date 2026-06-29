# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR stochastic source metadata to HDL artefacts

"""Materialise SC-NIR source metadata into concrete HDL artefacts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping

from ..hdl_gen import Lfsr16Emitter, Sobol16Emitter
from ..hdl_gen._ident import sanitize_ident
from .scnir_schema import SCNIRDocument, SCNIRStream, scnir_to_dict

SCNIRHDLSourceKind = Literal["lfsr16", "sobol16"]

_STREAM_IDENT_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class SCNIRHDLSourceManifestEntry:
    """Serialisable manifest row for one emitted stochastic source module."""

    stream_id: str
    layer: str
    module_name: str
    source_kind: SCNIRHDLSourceKind
    seed: int
    bitstream_length: int
    encoding: str
    signal_kind: str
    delay_steps: int | tuple[int, ...]
    total_bits: int
    fractional_bits: int
    transforms: tuple[dict[str, object], ...] = ()
    online_learning: Mapping[str, Any] | None = None
    lfsr_polynomial: str | None = None
    tap_mask: int | None = None
    sobol_dimension: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready representation."""

        return {
            "stream_id": self.stream_id,
            "layer": self.layer,
            "module_name": self.module_name,
            "source_kind": self.source_kind,
            "seed": self.seed,
            "bitstream_length": self.bitstream_length,
            "encoding": self.encoding,
            "signal_kind": self.signal_kind,
            "delay_steps": (
                self.delay_steps if isinstance(self.delay_steps, int) else list(self.delay_steps)
            ),
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
            "transforms": [dict(transform) for transform in self.transforms],
            "online_learning": (
                dict(self.online_learning) if self.online_learning is not None else None
            ),
            "lfsr_polynomial": self.lfsr_polynomial,
            "tap_mask": self.tap_mask,
            "sobol_dimension": self.sobol_dimension,
        }


@dataclass(frozen=True, slots=True)
class SCNIRHDLSourceBundle:
    """Concrete HDL source modules plus the manifest that explains them."""

    modules: dict[str, str]
    manifest: tuple[SCNIRHDLSourceManifestEntry, ...]

    def manifest_dicts(self) -> tuple[dict[str, object], ...]:
        """Return deterministic JSON-ready manifest rows."""

        return tuple(entry.as_dict() for entry in self.manifest)


def build_scnir_source_bundle(document: SCNIRDocument) -> SCNIRHDLSourceBundle:
    """Emit deterministic HDL source modules for every SC-NIR stream.

    Only source kinds with the standard threshold-bit output contract are
    materialised here. Unsupported SC-NIR source kinds fail closed instead of
    being lowered to semantically incompatible RTL.
    """

    scnir_to_dict(document)
    modules: dict[str, str] = {}
    manifest: list[SCNIRHDLSourceManifestEntry] = []

    for index, stream in enumerate(document.streams):
        module_name = _module_name_for_stream(stream, index)
        entry, verilog = _emit_stream_source(stream, module_name=module_name)
        if module_name in modules:  # pragma: no cover - unreachable: the per-stream index makes every module name unique
            raise ValueError(f"duplicate SC-NIR source module name {module_name!r}")
        modules[module_name] = verilog
        manifest.append(entry)

    return SCNIRHDLSourceBundle(modules=modules, manifest=tuple(manifest))


def _emit_stream_source(
    stream: SCNIRStream,
    *,
    module_name: str,
) -> tuple[SCNIRHDLSourceManifestEntry, str]:
    source = stream.source
    seed = _require_seed(stream)

    if source.kind == "lfsr":
        source_kind: SCNIRHDLSourceKind = "lfsr16"
        entry = _manifest_entry(
            stream,
            module_name=module_name,
            source_kind=source_kind,
            seed=seed,
            lfsr_polynomial=source.lfsr_polynomial,
            tap_mask=source.tap_mask,
        )
        return entry, Lfsr16Emitter(module_name=module_name, seed=seed).generate()

    if source.kind == "sobol":
        source_kind = "sobol16"
        entry = _manifest_entry(
            stream,
            module_name=module_name,
            source_kind=source_kind,
            seed=seed,
            sobol_dimension=source.sobol_dimension,
        )
        return entry, Sobol16Emitter(module_name=module_name, seed=seed).generate()

    raise ValueError(
        f"source_kind {source.kind!r} for stream {stream.stream_id!r} cannot be "
        "materialised as a 16-bit threshold stochastic source module"
    )


def _manifest_entry(
    stream: SCNIRStream,
    *,
    module_name: str,
    source_kind: SCNIRHDLSourceKind,
    seed: int,
    lfsr_polynomial: str | None = None,
    tap_mask: int | None = None,
    sobol_dimension: int | None = None,
) -> SCNIRHDLSourceManifestEntry:
    return SCNIRHDLSourceManifestEntry(
        stream_id=stream.stream_id,
        layer=stream.layer,
        module_name=module_name,
        source_kind=source_kind,
        seed=seed & 0xFFFF,
        bitstream_length=stream.bitstream_length,
        encoding=stream.encoding,
        signal_kind=stream.signal_kind,
        delay_steps=(
            stream.delay_steps
            if isinstance(stream.delay_steps, int)
            else tuple(int(value) for value in stream.delay_steps)
        ),
        total_bits=stream.precision.total_bits,
        fractional_bits=stream.precision.fractional_bits,
        transforms=tuple(
            {
                "kind": transform.kind,
                "position": transform.position,
                "comparison": transform.comparison,
                "values": [float(value) for value in transform.values],
            }
            for transform in stream.transforms
        ),
        online_learning=stream.online_learning,
        lfsr_polynomial=lfsr_polynomial,
        tap_mask=tap_mask,
        sobol_dimension=sobol_dimension,
    )


def _require_seed(stream: SCNIRStream) -> int:
    seed = stream.source.seed
    if seed is None:
        raise ValueError(
            f"source_kind {stream.source.kind!r} stream {stream.stream_id!r} needs seed"
        )
    return seed & 0xFFFF


def _module_name_for_stream(stream: SCNIRStream, index: int) -> str:
    fragment = _STREAM_IDENT_FRAGMENT_RE.sub("_", stream.stream_id.strip()).strip("_")
    fragment = re.sub(r"_+", "_", fragment) or "stream"
    prefix = f"scnir_src_{index:03d}_"
    limit = 63 - len(prefix)
    return sanitize_ident(f"{prefix}{fragment[:limit]}", context="SC-NIR source module name")
