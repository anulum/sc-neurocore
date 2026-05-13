# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR conversion utilities

"""Conversion utilities that attach SC-NIR metadata to NIR-derived graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

from .scnir_schema import (
    SCNIRCorrelationConstraint,
    SCNIRDocument,
    SCNIREncoding,
    SCNIROverflow,
    SCNIRPrecision,
    SCNIRRounding,
    SCNIRSource,
    SCNIRSourceKind,
    SCNIRStream,
    SCNIRSignalKind,
    write_scnir,
)

_DEFAULT_LFSR_POLYNOMIAL = "x^16 + x^14 + x^13 + x^11 + 1"
_DEFAULT_LFSR_TAP_MASK = 0xB400
_MAX_SEED = (1 << 64) - 1
_STREAM_FRAGMENT_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(frozen=True, slots=True)
class SCNIRConversionConfig:
    """Configuration for deterministic SC-NIR metadata export."""

    bitstream_length: int
    data_width: int = 16
    fraction: int = 8
    accumulator_bits: int | None = None
    base_seed: int = 1
    source_kind: Literal["lfsr", "sobol", "halton"] = "lfsr"
    rounding: SCNIRRounding = "nearest_even"
    overflow: SCNIROverflow = "saturate"
    seed_domain: str = "scnir-default"
    max_abs_correlation: float = 0.0
    producer: str = "sc-neurocore"

    def __post_init__(self) -> None:
        if not isinstance(self.bitstream_length, int) or self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be a positive integer")
        if not isinstance(self.data_width, int) or self.data_width <= 0:
            raise ValueError("data_width must be a positive integer")
        if not isinstance(self.fraction, int) or self.fraction < 0:
            raise ValueError("fraction must be a non-negative integer")
        if self.fraction >= self.data_width:
            raise ValueError("fraction must be smaller than data_width")
        if self.accumulator_bits is not None and self.accumulator_bits < self.data_width:
            raise ValueError("accumulator_bits must be greater than or equal to data_width")
        if not isinstance(self.base_seed, int) or not 0 <= self.base_seed <= _MAX_SEED:
            raise ValueError("base_seed must fit in uint64")
        if not 0.0 <= self.max_abs_correlation <= 1.0:
            raise ValueError("max_abs_correlation must be in [0, 1]")
        if not self.seed_domain:
            raise ValueError("seed_domain must be non-empty")
        if not self.producer:
            raise ValueError("producer must be non-empty")

    @property
    def resolved_accumulator_bits(self) -> int:
        """Accumulator width used by exported precision metadata."""

        return self.accumulator_bits if self.accumulator_bits is not None else self.data_width * 2


def build_scnir_from_neuron_graph(
    neuron_graph: Any,
    *,
    config: SCNIRConversionConfig,
) -> SCNIRDocument:
    """Build an SC-NIR document from an existing NIR-derived NeuronGraph."""

    streams: list[SCNIRStream] = []
    pop_stream_ids: dict[str, str] = {}

    for pop in neuron_graph.populations:
        neuron_type = str(pop.neuron_type)
        signal_kind = _population_signal_kind(neuron_type)
        stream_id = _population_stream_id(str(pop.name), signal_kind=signal_kind)
        pop_stream_ids[str(pop.name)] = stream_id
        streams.append(
            SCNIRStream(
                stream_id=stream_id,
                layer=str(pop.name),
                bitstream_length=config.bitstream_length,
                encoding=_population_encoding(neuron_type),
                precision=_precision(config, signed=False),
                source=_source(config, len(streams)),
                signal_kind=signal_kind,
                delay_steps=0,
                correlation_constraints=(),
            )
        )

    for conn in neuron_graph.connections:
        dst = str(conn.dst)
        dst_stream_id = pop_stream_ids.get(dst)
        if dst_stream_id is None:
            raise ValueError(f"Connection destination {dst!r} has no population stream")

        stream_index = len(streams)
        streams.append(
            SCNIRStream(
                stream_id=_connection_stream_id(str(conn.src), dst),
                layer=dst,
                bitstream_length=config.bitstream_length,
                encoding="bipolar",
                precision=_precision(config, signed=True),
                source=_source(config, stream_index),
                signal_kind="weight",
                delay_steps=int(getattr(conn, "delay_steps", 0)),
                correlation_constraints=(
                    SCNIRCorrelationConstraint(
                        peer_stream_id=dst_stream_id,
                        policy="max_correlation",
                        max_abs_correlation=config.max_abs_correlation,
                        seed_domain=config.seed_domain,
                    ),
                ),
            )
        )

    return SCNIRDocument(producer=config.producer, streams=tuple(streams))


def export_scnir_from_nir(
    model_path: str | Path,
    *,
    output_path: str | Path,
    config: SCNIRConversionConfig,
    dt: float = 1.0,
) -> SCNIRDocument:
    """Read a NIR model, export SC-NIR metadata, and write it to JSON."""

    import nir as nir_lib

    from sc_neurocore.nir_bridge import from_nir, from_scnetwork

    graph = nir_lib.read(str(model_path))
    network = from_nir(graph, dt=dt)
    neuron_graph = from_scnetwork(network, dt=dt)
    document = build_scnir_from_neuron_graph(neuron_graph, config=config)
    write_scnir(output_path, document)
    return document


def _precision(config: SCNIRConversionConfig, *, signed: bool) -> SCNIRPrecision:
    return SCNIRPrecision(
        signed=signed,
        total_bits=config.data_width,
        fractional_bits=config.fraction,
        accumulator_bits=config.resolved_accumulator_bits,
        rounding=config.rounding,
        overflow=config.overflow,
    )


def _source(config: SCNIRConversionConfig, stream_index: int) -> SCNIRSource:
    seed = config.base_seed + stream_index
    if seed > _MAX_SEED:
        raise ValueError("source seed allocation exceeds uint64")

    kind: SCNIRSourceKind = config.source_kind
    if kind == "lfsr":
        return SCNIRSource(
            kind="lfsr",
            seed=seed,
            lfsr_polynomial=_DEFAULT_LFSR_POLYNOMIAL,
            tap_mask=_DEFAULT_LFSR_TAP_MASK,
        )
    if kind == "sobol":
        return SCNIRSource(kind="sobol", seed=seed, sobol_dimension=stream_index + 1)
    if kind == "halton":
        return SCNIRSource(kind="halton", seed=seed, halton_base=_nth_prime(stream_index + 1))
    raise ValueError(f"Unsupported source kind: {kind!r}")


def _population_encoding(neuron_type: str) -> SCNIREncoding:
    if neuron_type in {"lif", "if", "cuba_lif"}:
        return "unipolar"
    return "bipolar"


def _population_signal_kind(neuron_type: str) -> SCNIRSignalKind:
    if neuron_type in {"li", "cuba_li", "integrator"}:
        return "analogue_state"
    return "spike"


def _population_stream_id(name: str, *, signal_kind: SCNIRSignalKind) -> str:
    suffix = "state" if signal_kind == "analogue_state" else "spike"
    return f"pop.{_stream_fragment(name)}.{suffix}"


def _connection_stream_id(src: str, dst: str) -> str:
    return f"conn.{_stream_fragment(src)}_to_{_stream_fragment(dst)}.weight"


def _stream_fragment(value: str) -> str:
    cleaned = _STREAM_FRAGMENT_RE.sub("_", value.strip())
    cleaned = cleaned.strip("_.:-")
    if not cleaned:
        cleaned = "stream"
    if not cleaned[0].isalpha():
        cleaned = f"s_{cleaned}"
    return cleaned[:96]


def _nth_prime(index: int) -> int:
    if index < 1:
        raise ValueError("prime index must be positive")
    primes: list[int] = []
    candidate = 2
    while len(primes) < index:
        if all(candidate % prime != 0 for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes[-1]
