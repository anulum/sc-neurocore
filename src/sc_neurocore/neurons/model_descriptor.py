# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Declarative neuron model descriptor (schema v2)

"""Declarative model descriptor — the single source of truth for one model.

A descriptor carries everything the catalogue, documentation, benchmarks, and UX
need for a model: discovery taxonomy, provenance, parameters with units and
ranges, dynamics, the backend matrix, and reproducibility. It extends the v1
``model_schemas`` format (``metadata``/``state``/``parameters``/``dynamics``/…)
with the discovery and provenance columns, and is validated against controlled
vocabularies so the metadata cannot drift into free text.

The descriptor is intentionally tolerant of partially-curated content: every
curation field is optional, and :func:`descriptor_completeness_tier` reports how
complete a descriptor is on the science axis (0-3) so coverage can grow as the
library is tuned without blocking early authoring. No field is ever fabricated —
uncurated values are simply absent.

Two evidence facets — :class:`Validation` (the class-correct dynamics check) and
:class:`Silicon` (the realisation ladder from compile-clean RTL to signed PPA) —
carry the committed proof anchors for the deeper science and silicon tiers. They
are recorded outcomes, never derived: the dual-axis scoring that reads them
(``science_tier`` S0-S5, ``silicon_tier`` H0-H5) lives in
:mod:`sc_neurocore.neurons.descriptor_tiers`, so a tier can never be inflated
ahead of the evidence in the descriptor (master plan invariant I7).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

MODEL_DESCRIPTOR_SCHEMA_VERSION = 2

BIOPHYSICAL_DETAILS = frozenset({"point", "reduced", "multicompartment", "conductance"})
MATURITIES = frozenset({"reference", "experimental", "validated"})
BACKEND_NAMES = ("python", "rust", "julia", "go", "mojo")
BACKEND_STATUSES = frozenset({"implemented", "planned", "unsupported"})
BACKEND_PARITIES = frozenset({"exact", "ulp-bounded", "approximate", "n/a"})

# The metric by which a model is validated against its publication, chosen for the
# model's class (§4 of the catalogue-to-silicon master plan): spike-count parity
# for deterministic point neurons, distributional agreement for stochastic models,
# trajectory error for smooth ODEs, per-compartment agreement for multicompartment.
VALIDATION_METRICS = frozenset({"none", "parity", "statistical", "trajectory", "per_compartment"})
# Terminal silicon tier a model's deployability class is expected to reach; empty
# means the class has not been declared yet (so the model cannot be certified
# perfect). H0-H5 mirror the silicon axis in :mod:`descriptor_tiers`.
SILICON_TARGET_TIERS = frozenset({"", "H0", "H1", "H2", "H3", "H4", "H5"})

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A single model parameter with its physical semantics.

    Parameters
    ----------
    name:
        Parameter identifier matching the model's constructor field.
    default:
        Default numeric value.
    unit:
        Physical unit (for example ``mV``, ``ms``, ``pA``); empty when uncurated.
    value_range:
        Optional ``(min, max)`` admissible range.
    biological_range:
        Optional ``(min, max)`` biologically plausible range.
    meaning:
        Human-readable description; empty when uncurated.
    """

    name: str
    default: float
    unit: str = ""
    value_range: tuple[float, float] | None = None
    biological_range: tuple[float, float] | None = None
    meaning: str = ""

    @property
    def is_curated(self) -> bool:
        """True when the parameter has a unit, a range, and a meaning."""
        return bool(self.unit) and self.value_range is not None and bool(self.meaning)


@dataclass(frozen=True, slots=True)
class StateVariableSpec:
    """A model state variable with its initial value and semantics."""

    name: str
    init: float
    unit: str = ""
    meaning: str = ""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Citation and licensing provenance for a model."""

    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str = ""
    paper_title: str = ""
    url: str = ""
    license_note: str = ""
    doi_is_translation: bool = False
    """True when ``doi`` points to a later translation/reprint of the cited work.

    A pre-DOI historical work (e.g. Lapicque 1907) has no DOI of its own; the
    honest, verifiable reference is a modern translation with a DOI whose registry
    author and year are the translator's, not the original's. The verifier then
    confirms the DOI *resolves* (anti-fabrication) without demanding the original's
    author and year match the translation's.
    """

    @property
    def is_citeable(self) -> bool:
        """True when authors, a year, and a valid DOI are all present."""
        return bool(self.authors) and self.year is not None and bool(self.doi)


@dataclass(frozen=True, slots=True)
class BackendSupport:
    """Implementation status and numeric parity for one compute backend."""

    name: str
    status: str
    parity: str = "n/a"


@dataclass(frozen=True, slots=True)
class Reproducibility:
    """Reference-run reproducibility anchors for a model.

    ``golden_trace_sha256_variants`` is a finite allowlist for measured
    byte-level variants of the same numerically bounded trace, such as NumPy
    transcendental kernels selected by different x86 SIMD capabilities. The
    primary digest remains mandatory for a reproducible descriptor.
    """

    reference_config: str = ""
    seed: int | None = None
    golden_trace_sha256: str = ""
    golden_trace_sha256_variants: tuple[str, ...] = ()
    golden_citation: str = ""

    @property
    def is_reproducible(self) -> bool:
        """True when a reference config and a golden trace digest are present."""
        return bool(self.reference_config) and bool(self.golden_trace_sha256)

    @property
    def golden_trace_digests(self) -> tuple[str, ...]:
        """Return the primary digest followed by measured compatible variants."""
        if not self.golden_trace_sha256:
            return self.golden_trace_sha256_variants
        return (self.golden_trace_sha256, *self.golden_trace_sha256_variants)


@dataclass(frozen=True, slots=True)
class Validation:
    """Class-correct validation evidence for a model's dynamics.

    Records the outcome of two checks (§3-§4 of the catalogue-to-silicon master
    plan): whether the model's discretised dynamics were confirmed faithful to
    the publication (``dynamics_faithful`` — the three-way schema/class/paper
    agreement), and by which metric the model was validated for its class, with
    committed evidence. Every field is a recorded outcome, never derived, so the
    science tier can read them as ground truth.

    Parameters
    ----------
    dynamics_faithful:
        True when the schema-DSL equations, parameters, dt, threshold, and reset
        were confirmed to match the publication (and the hand class where one
        exists). Gates the faithful-dynamics tier S4.
    metric:
        The class-appropriate validation metric from :data:`VALIDATION_METRICS`
        (``"none"`` until validated).
    operating_point:
        Human-readable statement of the operating point the validation used
        (for example an input drive or a parameter regime).
    tolerance:
        The honest agreement tolerance achieved (for example ``"0 spikes"`` or a
        distributional distance), as a citeable string.
    evidence:
        A path, citation, or digest pointing at the committed validation evidence.
        Together with a non-``"none"`` metric this gates the validated tier S5.
    """

    dynamics_faithful: bool = False
    metric: str = "none"
    operating_point: str = ""
    tolerance: str = ""
    evidence: str = ""

    @property
    def is_class_validated(self) -> bool:
        """True when a non-trivial metric and committed evidence are both present."""
        return self.metric != "none" and bool(self.evidence)


@dataclass(frozen=True, slots=True)
class Silicon:
    """Ladder of committed silicon-realisation evidence for a model.

    Each rung of the silicon axis (H0-H5) is only credited when its evidence
    anchor is recorded, so a silicon tier can never be claimed ahead of proof
    (master plan invariant I7). ``compiles`` is the H0 anchor (iverilog-valid
    RTL); each higher boolean requires its companion report to count.

    Parameters
    ----------
    compiles:
        RTL lowers to iverilog-valid Verilog (compile-clean). The H0 anchor.
    cosim_validated:
        Python<->Verilog agreement by the class-correct metric was demonstrated;
        credited for H1 only alongside ``cosim_evidence``.
    synthesised:
        Passes a real synthesis flow (for example Yosys); credited for H2 only
        alongside ``synth_report``.
    timing_closed:
        Meets a stated clock on a target device with reported resources;
        credited for H3 only alongside ``timing_report`` and ``clock_mhz``.
    formally_equivalent:
        Machine-checked Python-semantics<->RTL equivalence in CI; credited for H4
        only alongside ``equivalence_proof``.
    ppa_signed:
        Tool-level RTL->GDSII signoff (open PDK) with clean DRC/LVS/STA; credited
        for H5 only alongside ``ppa_report``.
    cosim_evidence, synth_report, timing_report, equivalence_proof, ppa_report:
        Paths, citations, or digests pointing at the committed proof for each rung.
    target_device:
        The device the timing/resource numbers were characterised on.
    clock_mhz:
        The clock the design closes at (MHz); required for the H3 credit.
    target_tier:
        The terminal H-tier this model's deployability class is expected to reach,
        from :data:`SILICON_TARGET_TIERS` (``""`` until declared).
    terminal_reason:
        Why the model terminates at ``target_tier`` (for example a research
        multicompartment model that need not reach signed PPA).
    """

    compiles: bool = False
    cosim_validated: bool = False
    synthesised: bool = False
    timing_closed: bool = False
    formally_equivalent: bool = False
    ppa_signed: bool = False
    cosim_evidence: str = ""
    synth_report: str = ""
    timing_report: str = ""
    equivalence_proof: str = ""
    ppa_report: str = ""
    target_device: str = ""
    clock_mhz: float | None = None
    target_tier: str = ""
    terminal_reason: str = ""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """The full declarative descriptor for one neuron model."""

    name: str
    class_name: str
    module: str
    display_name: str
    summary: str
    family: str
    category: str
    biophysical_detail: str
    maturity: str
    intended_use: tuple[str, ...]
    hardware_fit: tuple[str, ...]
    behavior_tags: tuple[str, ...]
    provenance: Provenance
    state: tuple[StateVariableSpec, ...]
    parameters: tuple[ParameterSpec, ...]
    dt: float
    integration_method: str
    dynamics: Mapping[str, str]
    backends: tuple[BackendSupport, ...]
    reproducibility: Reproducibility
    documentation_slug: str = ""
    notes: str = ""
    validation: Validation = field(default_factory=Validation)
    silicon: Silicon = field(default_factory=Silicon)
    schema_version: int = MODEL_DESCRIPTOR_SCHEMA_VERSION


def descriptor_completeness_tier(descriptor: ModelDescriptor) -> int:
    """Return the science-axis completeness kernel (0-3) a descriptor satisfies.

    This is the S0-S3 base of the science axis — the discovery-and-curation
    tiers that need no execution evidence. The full science axis (adding the
    faithful-dynamics tier S4 and the class-validated tier S5) and the silicon
    axis (H0-H5) are derived from this kernel plus the descriptor's evidence
    facets by :mod:`sc_neurocore.neurons.descriptor_tiers`.

    Tier 0 — exists and identifies a real model (class, module, params, state).
    Tier 1 — discovery taxonomy declared (family and category). Behaviour tags
             are an optional measured facet, not a tier requirement, so a tier
             never depends on running a simulation.
    Tier 2 — scientifically curated: citeable provenance (authors + year + DOI)
             and every parameter curated (unit + range + meaning).
    Tier 3 — engineering-verified: at least two implemented backends and a
             reproducibility anchor (reference config + golden trace digest).
    """
    if not descriptor.parameters and not descriptor.state:
        return 0
    tier = 0
    has_taxonomy = bool(descriptor.family) and bool(descriptor.category)
    if has_taxonomy:
        tier = 1
    # A parameterless model (for example the theta phase model) is vacuously
    # parameter-curated; otherwise every parameter must carry unit, range, meaning.
    params_curated = all(p.is_curated for p in descriptor.parameters)
    if tier == 1 and descriptor.provenance.is_citeable and params_curated:
        tier = 2
    implemented = sum(1 for b in descriptor.backends if b.status == "implemented")
    if tier == 2 and implemented >= 2 and descriptor.reproducibility.is_reproducible:
        tier = 3
    return tier


class ModelDescriptorError(ValueError):
    """Raised when a descriptor payload violates the schema contract."""


def parse_model_descriptor(payload: Mapping[str, object]) -> ModelDescriptor:
    """Validate a descriptor payload and return a :class:`ModelDescriptor`.

    Parameters
    ----------
    payload:
        A loaded descriptor mapping (from TOML/JSON), using the v2 section
        layout: ``metadata``, ``provenance``, ``state``, ``parameters``,
        ``integration``, ``dynamics``, ``backends``, ``reproducibility``,
        ``documentation``.

    Returns
    -------
    ModelDescriptor
        The validated descriptor.

    Raises
    ------
    ModelDescriptorError
        If any required identifier is missing or a controlled-vocabulary field
        carries an unknown value.
    """
    metadata = _section(payload, "metadata")
    version = metadata.get("schema_version", 1)
    if version != MODEL_DESCRIPTOR_SCHEMA_VERSION:
        raise ModelDescriptorError(
            f"unsupported descriptor schema_version {version!r}; expected "
            f"{MODEL_DESCRIPTOR_SCHEMA_VERSION}"
        )
    name = _required_str(metadata, "name")
    class_name = _required_str(metadata, "class_name")
    module = _required_str(metadata, "module")
    biophysical = _vocab(
        metadata.get("biophysical_detail", "point"), BIOPHYSICAL_DETAILS, "biophysical_detail"
    )
    maturity = _vocab(metadata.get("maturity", "experimental"), MATURITIES, "maturity")
    descriptor = ModelDescriptor(
        name=name,
        class_name=class_name,
        module=module,
        display_name=_opt_str(metadata, "display_name") or name,
        summary=_opt_str(metadata, "summary"),
        family=_opt_str(metadata, "family"),
        category=_opt_slug(metadata, "category"),
        biophysical_detail=biophysical,
        maturity=maturity,
        intended_use=_str_tuple(metadata.get("intended_use")),
        hardware_fit=_str_tuple(metadata.get("hardware_fit")),
        behavior_tags=_str_tuple(metadata.get("behavior_tags")),
        provenance=_parse_provenance(_section(payload, "provenance", required=False)),
        state=_parse_state(_section(payload, "state", required=False)),
        parameters=_parse_parameters(_section(payload, "parameters", required=False)),
        dt=_float(_section(payload, "integration", required=False).get("dt", 0.1)),
        integration_method=_opt_str(_section(payload, "integration", required=False), "method")
        or "euler",
        dynamics=_parse_dynamics(_section(payload, "dynamics", required=False)),
        backends=_parse_backends(_section(payload, "backends", required=False)),
        reproducibility=_parse_reproducibility(
            _section(payload, "reproducibility", required=False)
        ),
        documentation_slug=_opt_str(_section(payload, "documentation", required=False), "slug"),
        notes=_opt_str(_section(payload, "documentation", required=False), "notes"),
        validation=_parse_validation(_section(payload, "validation", required=False)),
        silicon=_parse_silicon(_section(payload, "silicon", required=False)),
    )
    return descriptor


def _section(
    payload: Mapping[str, object], key: str, *, required: bool = True
) -> dict[str, object]:
    value = payload.get(key)
    if value is None:
        if required:
            raise ModelDescriptorError(f"descriptor is missing the [{key}] section")
        return {}
    if not isinstance(value, Mapping):
        raise ModelDescriptorError(f"descriptor [{key}] must be a table")
    return dict(value)


def _required_str(section: Mapping[str, object], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ModelDescriptorError(f"descriptor metadata requires a non-empty {key!r}")
    return value


def _opt_str(section: Mapping[str, object], key: str) -> str:
    value = section.get(key)
    return value if isinstance(value, str) else ""


def _opt_slug(section: Mapping[str, object], key: str) -> str:
    value = _opt_str(section, key)
    if value and not _SLUG.fullmatch(value):
        raise ModelDescriptorError(f"descriptor {key!r} must be a slug, got {value!r}")
    return value


def _opt_bool(section: Mapping[str, object], key: str) -> bool:
    value = section.get(key, False)
    if not isinstance(value, bool):
        raise ModelDescriptorError(f"descriptor {key!r} must be a boolean")
    return value


def _vocab(value: object, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ModelDescriptorError(
            f"descriptor {field_name!r} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ModelDescriptorError("descriptor tag fields must be lists of strings")
    return tuple(str(item) for item in value)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelDescriptorError(f"expected a number, got {value!r}")
    return float(value)


def _opt_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ModelDescriptorError("a range must be a [min, max] pair")
    return (_float(value[0]), _float(value[1]))


def _parse_provenance(section: Mapping[str, object]) -> Provenance:
    doi = _opt_str(section, "doi")
    if doi and not _DOI.fullmatch(doi):
        raise ModelDescriptorError(f"invalid DOI {doi!r}")
    year = section.get("year")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise ModelDescriptorError("provenance year must be an integer")
    authors = section.get("authors")
    author_tuple = _str_tuple(authors) if authors is not None else ()
    if not author_tuple and isinstance(section.get("author"), str):
        author_tuple = (str(section["author"]),)
    return Provenance(
        authors=author_tuple,
        year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        doi=doi,
        paper_title=_opt_str(section, "paper_title"),
        url=_opt_str(section, "url"),
        license_note=_opt_str(section, "license_note"),
        doi_is_translation=bool(section.get("doi_is_translation", False)),
    )


def _parse_state(section: Mapping[str, object]) -> tuple[StateVariableSpec, ...]:
    specs: list[StateVariableSpec] = []
    for name, value in section.items():
        if isinstance(value, Mapping):
            specs.append(
                StateVariableSpec(
                    name=name,
                    init=_float(value.get("init", 0.0)),
                    unit=_opt_str(value, "unit"),
                    meaning=_opt_str(value, "meaning"),
                )
            )
        else:
            specs.append(StateVariableSpec(name=name, init=_float(value)))
    return tuple(specs)


def _parse_parameters(section: Mapping[str, object]) -> tuple[ParameterSpec, ...]:
    specs: list[ParameterSpec] = []
    for name, value in section.items():
        if isinstance(value, Mapping):
            specs.append(
                ParameterSpec(
                    name=name,
                    default=_float(value.get("default", 0.0)),
                    unit=_opt_str(value, "unit"),
                    value_range=_opt_range(value.get("range")),
                    biological_range=_opt_range(value.get("biological_range")),
                    meaning=_opt_str(value, "meaning"),
                )
            )
        else:
            specs.append(ParameterSpec(name=name, default=_float(value)))
    return tuple(specs)


def _parse_dynamics(section: Mapping[str, object]) -> dict[str, str]:
    dynamics: dict[str, str] = {}
    for name, value in section.items():
        if isinstance(value, Mapping):
            expr = value.get("expr", "")
            dynamics[name] = str(expr) if isinstance(expr, str) else ""
        elif isinstance(value, str):
            dynamics[name] = value
    return dynamics


def _parse_backends(section: Mapping[str, object]) -> tuple[BackendSupport, ...]:
    backends: list[BackendSupport] = []
    for name, value in section.items():
        status = "implemented"
        parity = "n/a"
        if isinstance(value, Mapping):
            status = _vocab(value.get("status", "implemented"), BACKEND_STATUSES, "backend status")
            parity = _vocab(value.get("parity", "n/a"), BACKEND_PARITIES, "backend parity")
        elif isinstance(value, str):
            status = _vocab(value, BACKEND_STATUSES, "backend status")
        backends.append(BackendSupport(name=name, status=status, parity=parity))
    return tuple(backends)


def _parse_reproducibility(section: Mapping[str, object]) -> Reproducibility:
    digest = _opt_str(section, "golden_trace_sha256")
    if digest and not _SHA256_HEX.fullmatch(digest):
        raise ModelDescriptorError(f"invalid golden_trace_sha256 {digest!r}")
    raw_variants = section.get("golden_trace_sha256_variants", ())
    if not isinstance(raw_variants, Sequence) or isinstance(raw_variants, str):
        raise ModelDescriptorError("golden_trace_sha256_variants must be a list of SHA-256 strings")
    variants: list[str] = []
    for variant in raw_variants:
        if not isinstance(variant, str) or not _SHA256_HEX.fullmatch(variant):
            raise ModelDescriptorError(f"invalid golden_trace_sha256_variants entry {variant!r}")
        variants.append(variant)
    if variants and not digest:
        raise ModelDescriptorError("golden_trace_sha256_variants require a primary digest")
    if len(set((digest, *variants))) != 1 + len(variants):
        raise ModelDescriptorError("golden_trace_sha256_variants must be unique and non-primary")
    seed = section.get("seed")
    return Reproducibility(
        reference_config=_opt_str(section, "reference_config"),
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        golden_trace_sha256=digest,
        golden_trace_sha256_variants=tuple(variants),
        golden_citation=_opt_str(section, "golden_citation"),
    )


def _parse_validation(section: Mapping[str, object]) -> Validation:
    metric = _vocab(section.get("metric", "none"), VALIDATION_METRICS, "validation metric")
    return Validation(
        dynamics_faithful=_opt_bool(section, "dynamics_faithful"),
        metric=metric,
        operating_point=_opt_str(section, "operating_point"),
        tolerance=_opt_str(section, "tolerance"),
        evidence=_opt_str(section, "evidence"),
    )


def _parse_silicon(section: Mapping[str, object]) -> Silicon:
    target = section.get("target_tier", "")
    if not isinstance(target, str) or target not in SILICON_TARGET_TIERS:
        raise ModelDescriptorError(
            f"silicon target_tier must be one of {sorted(SILICON_TARGET_TIERS)}, got {target!r}"
        )
    clock = section.get("clock_mhz")
    if clock is not None and (isinstance(clock, bool) or not isinstance(clock, (int, float))):
        raise ModelDescriptorError("silicon clock_mhz must be a number")
    return Silicon(
        compiles=_opt_bool(section, "compiles"),
        cosim_validated=_opt_bool(section, "cosim_validated"),
        synthesised=_opt_bool(section, "synthesised"),
        timing_closed=_opt_bool(section, "timing_closed"),
        formally_equivalent=_opt_bool(section, "formally_equivalent"),
        ppa_signed=_opt_bool(section, "ppa_signed"),
        cosim_evidence=_opt_str(section, "cosim_evidence"),
        synth_report=_opt_str(section, "synth_report"),
        timing_report=_opt_str(section, "timing_report"),
        equivalence_proof=_opt_str(section, "equivalence_proof"),
        ppa_report=_opt_str(section, "ppa_report"),
        target_device=_opt_str(section, "target_device"),
        clock_mhz=float(clock) if clock is not None else None,
        target_tier=target,
        terminal_reason=_opt_str(section, "terminal_reason"),
    )


__all__ = [
    "BACKEND_NAMES",
    "BACKEND_PARITIES",
    "BACKEND_STATUSES",
    "BIOPHYSICAL_DETAILS",
    "MATURITIES",
    "MODEL_DESCRIPTOR_SCHEMA_VERSION",
    "SILICON_TARGET_TIERS",
    "VALIDATION_METRICS",
    "BackendSupport",
    "ModelDescriptor",
    "ModelDescriptorError",
    "ParameterSpec",
    "Provenance",
    "Reproducibility",
    "Silicon",
    "StateVariableSpec",
    "Validation",
    "descriptor_completeness_tier",
    "parse_model_descriptor",
]
