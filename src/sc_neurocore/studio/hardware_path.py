# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One-command hardware path of a catalogue model

"""Take one catalogue model from its schema to synthesis, refusing by name at each stage.

:func:`run_hardware_path` composes the Studio functions that already exist, in
this order, and writes every artefact into one output directory:

1. **Capabilities.** :func:`~sc_neurocore.studio.model_capabilities.model_capabilities`
   must enable compile, co-simulation and synthesis for the model, and the
   requested integrator and Q format must be a co-simulated combination of it.
   Place and route is optional: when the target has no place-and-route tool,
   or its tool is not installed, the path records that it was not run and why.
2. **Configuration and RTL.**
   :func:`~sc_neurocore.studio.model_compile_configuration.resolve_model_compile_configuration`
   resolves the schema, and its RTL is written with the traceability record
   the resolved configuration's ``to_traceability`` builds.
3. **Co-simulation.** :func:`~sc_neurocore.studio.model_cosim.run_model_cosim`
   compares the RTL with the generated bit-true C kernel. A report that is not
   bit-exact is written and the path stops: RTL that disagrees with its
   reference is never synthesised.
4. **Guarantee summary.** The classes of
   :data:`~sc_neurocore.compiler.guarantee_summary.GUARANTEE_CLASSES` are used:
   the co-simulation is ``CO-SIMULATED ONLY``; the model's catalogue formal job
   is ``FORMAL`` only when its RTL file is byte-identical to the emitted RTL,
   ``PROVEN AT OTHER PARAMETERS`` otherwise, and ``NONE`` when the model has no
   job. No formal job is run here. The readiness labels come from the sealed
   ``verified_readiness.json``. The classification is
   :mod:`sc_neurocore.studio.model_guarantee`'s; this stage records it.
5. **Synthesis.** :func:`~sc_neurocore.studio.synthesis.run_synthesis_terminal`
   (Yosys, then nextpnr) for a target whose place-and-route tool is installed;
   otherwise :func:`~sc_neurocore.studio.synthesis.run_synthesis` with place
   and route recorded as not run.
6. **Manifest.** ``hardware_path.json`` holds every stage record, the SHA-256
   of every artefact, the tool versions the composed functions report and the
   boundary statement; ``hardware_path.txt`` is the same for people.

Synthesis is mandatory because the capability gate requires it: a host without
Yosys is refused at the first stage rather than producing a partial path. The
same holds for the co-simulation tools.

:func:`supported_models` derives, for every catalogue model, whether the path
is supported and the exact reason it is refused, from the same capability
matrix; it is not a hand-kept list.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.studio.model_capabilities import FORMAL_INVENTORY, model_capabilities
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import (
    ResolvedModelCompileConfiguration,
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_cosim import ModelCosimExecution, run_model_cosim
from sc_neurocore.studio.model_guarantee import (
    MODEL_GUARANTEE_SCHEMA_VERSION as MODEL_GUARANTEE_SCHEMA_VERSION,
    formal_guarantee as formal_guarantee,
    model_guarantee_summary as model_guarantee_summary,
    readiness_labels as readiness_labels,
    render_model_guarantee_text as render_model_guarantee_text,
)
from sc_neurocore.studio.model_numeric_contracts import STUDIO_Q_FORMATS
from sc_neurocore.studio.readiness_seal import SEAL_PATH
from sc_neurocore.studio.synthesis import (
    check_tools,
    run_synthesis,
    run_synthesis_terminal,
    supported_targets,
)
from sc_neurocore.studio.synthesis_profiles import _TARGETS
from sc_neurocore.studio.synthesis_provenance import ToolStatusMap

HARDWARE_PATH_SCHEMA_VERSION = "sc-neurocore.studio.hardware-path.v1"
HARDWARE_PATH_SUPPORT_SCHEMA_VERSION = "sc-neurocore.studio.hardware-path-support.v1"

#: The Studio's own request defaults: ``ModelCosimRequest`` (current, steps)
#: and ``SynthesisTerminalRequest`` (target) in ``studio/api/schemas.py``.
DEFAULT_CURRENT = 10.0
DEFAULT_N_STEPS = 128
DEFAULT_TARGET = "ecp5"

#: Operations the capability gate requires; place and route is optional.
REQUIRED_OPERATIONS = ("compile", "cosimulate", "synthesise")

#: Targets whose Studio flow has a place-and-route tool.
PLACE_AND_ROUTE_TARGETS = tuple(
    target for target, profile in _TARGETS.items() if profile.get("pnr") is not None
)

MANIFEST_NAME = "hardware_path.json"
SUMMARY_NAME = "hardware_path.txt"


class HardwarePathRefusal(Exception):
    """A stage of the hardware path refused, by name and with a code.

    Parameters
    ----------
    stage : str
        The stage that refused: ``output``, ``capabilities``, ``configuration``,
        ``cosimulation``, ``synthesis`` or ``place_and_route``.
    code : str
        A stable refusal code, such as ``cosim-not-bit-exact``.
    message : str
        What was refused, naming the model, parameter, tool or operation.

    Attributes
    ----------
    manifest_path : Path or None
        The ``hardware_path.json`` written for the refused run, or ``None``
        when the refusal came before the output directory was accepted.
    """

    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(f"{stage} refused ({code}): {message}")
        self.stage = stage
        self.code = code
        self.message = message
        self.manifest_path: Path | None = None

    def to_public_dict(self) -> dict[str, str]:
        """Return the refusal as it is recorded in the manifest.

        Returns
        -------
        dict of str to str
            ``stage``, ``code`` and ``message``.
        """
        return {"stage": self.stage, "code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class HardwarePathResult:
    """A completed hardware path.

    Attributes
    ----------
    out_dir : Path
        The directory that holds every artefact.
    manifest_path : Path
        ``hardware_path.json``.
    summary_path : Path
        ``hardware_path.txt``.
    manifest : dict
        The manifest as written.
    """

    out_dir: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]


class _Recorder:
    """Write artefacts into one directory and keep their digests and the stage records."""

    def __init__(self, out_dir: Path, request: Mapping[str, object]) -> None:
        self.out_dir = out_dir
        self.request = dict(request)
        self.stages: list[dict[str, Any]] = []
        self.artifacts: dict[str, dict[str, object]] = {}
        self.tools: dict[str, object] = {}
        self.model: str = str(request["model"])

    def write(self, relative: str, payload: bytes) -> str:
        """Write one artefact and return its SHA-256."""
        path = self.out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.artifacts[relative] = {"sha256": digest, "bytes": len(payload)}
        return digest

    def write_json(self, relative: str, payload: object) -> str:
        """Write one JSON artefact and return its SHA-256."""
        return self.write(relative, _json_bytes(payload))

    def stage(self, name: str, **record: object) -> None:
        """Append one completed stage record."""
        self.stages.append({"stage": name, "status": "completed", **record})

    def finish(
        self,
        *,
        refusal: HardwarePathRefusal | None,
        boundary: Mapping[str, list[str]],
    ) -> tuple[Path, Path, dict[str, Any]]:
        """Write the manifest and its text summary."""
        from sc_neurocore._version import VERSION as __version__

        manifest: dict[str, Any] = {
            "schema_version": HARDWARE_PATH_SCHEMA_VERSION,
            "status": "refused" if refusal is not None else "completed",
            "model": self.model,
            "request": self.request,
            "stages": self.stages,
            "refusal": refusal.to_public_dict() if refusal is not None else None,
            "artifacts": dict(sorted(self.artifacts.items())),
            "tools": self.tools,
            "package_version": __version__,
            "boundary": dict(boundary),
        }
        manifest_path = self.out_dir / MANIFEST_NAME
        summary_path = self.out_dir / SUMMARY_NAME
        manifest_path.write_bytes(_json_bytes(manifest))
        summary_path.write_text(render_hardware_path_text(manifest), encoding="utf-8")
        return manifest_path, summary_path, manifest


def run_hardware_path(
    model_name: str,
    *,
    out_dir: Path,
    q_format: str | None = None,
    integrator: str | None = None,
    params: Mapping[str, float] | None = None,
    dt: float | None = None,
    current: float = DEFAULT_CURRENT,
    n_steps: int = DEFAULT_N_STEPS,
    target: str = DEFAULT_TARGET,
    module_name: str | None = None,
    formal_inventory: Path = FORMAL_INVENTORY,
    readiness_seal: Path = SEAL_PATH,
    cosimulate: Callable[..., ModelCosimExecution] = run_model_cosim,
) -> HardwarePathResult:
    """Run one catalogue model through capabilities, RTL, co-simulation and synthesis.

    Parameters
    ----------
    model_name : str
        A catalogue model, such as ``LapicqueNeuron``.
    out_dir : Path
        A new or empty directory; every artefact is written below it.
    q_format : str or None, optional
        Fixed-point format; ``None`` takes the catalogue's default format of
        the model (the smallest that holds it).
    integrator : str or None, optional
        Integrator; ``None`` takes the model's schema default.
    params : Mapping[str, float] or None, optional
        Schema parameter overrides.
    dt : float or None, optional
        Step override; ``None`` keeps the schema's step.
    current : float, optional
        Constant input of the requested co-simulation run (Studio default 10.0).
    n_steps : int, optional
        Cycles of the requested co-simulation run, 1 to 2048 (Studio default
        128); the fixed stress schedule runs after it.
    target : str, optional
        Synthesis target (Studio terminal default ``ecp5``).
    module_name : str or None, optional
        Verilog module name; ``None`` takes the Studio default
        ``sc_<schema>_neuron``.
    formal_inventory : Path, optional
        The catalogue formal inventory; its jobs' RTL files are read beside it.
    readiness_seal : Path, optional
        The sealed verified-readiness record.
    cosimulate : callable, optional
        The co-simulation, called as ``cosimulate(configuration, current=...,
        n_steps=...)``; :func:`~sc_neurocore.studio.model_cosim.run_model_cosim`.
        Whatever it returns passes the same bit-exact check before any later stage.

    Returns
    -------
    HardwarePathResult
        The completed path and its manifest.

    Raises
    ------
    HardwarePathRefusal
        Any stage refused. Unless the output directory itself was refused,
        ``hardware_path.json`` with status ``refused`` and every artefact
        written up to the refusal are in ``out_dir``, and ``manifest_path``
        names the manifest.
    """
    out_dir = Path(out_dir)
    _accept_output(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request: dict[str, object] = {
        "model": model_name,
        "q_format": q_format,
        "integrator": integrator,
        "params": dict(params or {}),
        "dt": dt,
        "current": current,
        "n_steps": n_steps,
        "target": target,
        "module_name": module_name,
    }
    recorder = _Recorder(out_dir, request)
    boundary: dict[str, list[str]] = {"establishes": [], "does_not_establish": []}
    try:
        _run_stages(
            recorder,
            boundary,
            q_format=q_format,
            integrator=integrator,
            params=dict(params or {}),
            dt=dt,
            current=current,
            n_steps=n_steps,
            target=target,
            module_name=module_name,
            formal_inventory=formal_inventory,
            readiness_seal=readiness_seal,
            cosimulate=cosimulate,
        )
    except HardwarePathRefusal as refusal:
        boundary["does_not_establish"].append(
            f"anything after the {refusal.stage} stage: the path was refused there ({refusal.code})"
        )
        manifest_path, _summary, _manifest = recorder.finish(refusal=refusal, boundary=boundary)
        refusal.manifest_path = manifest_path
        raise
    manifest_path, summary_path, manifest = recorder.finish(refusal=None, boundary=boundary)
    return HardwarePathResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
    )


def _accept_output(out_dir: Path) -> None:
    """Refuse a symbolic link, a file, or a directory that already holds files."""
    if out_dir.is_symlink():
        raise HardwarePathRefusal(
            "output",
            "output-is-symlink",
            f"{out_dir} is a symbolic link; the hardware path writes into a real directory",
        )
    if out_dir.exists() and not out_dir.is_dir():
        raise HardwarePathRefusal(
            "output", "output-not-directory", f"{out_dir} exists and is not a directory"
        )
    if out_dir.is_dir() and any(out_dir.iterdir()):
        raise HardwarePathRefusal(
            "output",
            "output-not-empty",
            f"{out_dir} is not empty; the hardware path writes into a new or empty directory",
        )


def _run_stages(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    *,
    q_format: str | None,
    integrator: str | None,
    params: dict[str, float],
    dt: float | None,
    current: float,
    n_steps: int,
    target: str,
    module_name: str | None,
    formal_inventory: Path,
    readiness_seal: Path,
    cosimulate: Callable[..., ModelCosimExecution],
) -> None:
    """Run the five stages of one hardware path in order, each recording itself.

    Every stage writes its artefacts, records its stage entry and adds what it establishes,
    and what it does not, to the boundary; a refusal in one stage stops the run before the
    next. The capabilities stage fixes the integrator, number format and place-and-route
    choice the later stages use.

    Parameters
    ----------
    recorder : _Recorder
        The run's recorder, which holds the model name and writes every artefact.
    boundary : dict of str to list of str
        What the run establishes and does not establish, filled in stage order.
    q_format, integrator : str or None
        The caller's number format and integrator, or None for the model's defaults.
    params : dict of str to float
        Parameter overrides for the compiled schema.
    dt : float or None
        The time step, or None for the schema's own.
    current : float
        The co-simulation's input current.
    n_steps : int
        The co-simulation's cycle count.
    target : str
        The synthesis target.
    module_name : str or None
        The RTL module name, or None for the default.
    formal_inventory : Path
        The catalogue formal job inventory.
    readiness_seal : Path
        The Studio readiness seal.
    cosimulate : callable
        Runs the co-simulation of one configuration.

    Raises
    ------
    HardwarePathRefusal
        A stage refuses; the refusal names the stage, its reason code and the detail.
    """
    tool_status = check_tools()
    recorder.tools["eda"] = {name: entry.get("version") for name, entry in tool_status.items()}
    choice = _capabilities_stage(
        recorder,
        tool_status=tool_status,
        q_format=q_format,
        integrator=integrator,
        target=target,
        formal_inventory=formal_inventory,
    )
    configuration, verilog, traceability = _configuration_stage(
        recorder, boundary, choice, params=params, dt=dt, module_name=module_name
    )
    report = _cosimulation_stage(
        recorder, boundary, configuration, cosimulate=cosimulate, current=current, n_steps=n_steps
    )
    _guarantee_stage(
        recorder,
        boundary,
        configuration,
        verilog,
        report,
        formal_inventory=formal_inventory,
        readiness_seal=readiness_seal,
    )

    # 5. Synthesis.
    if choice.pnr_runs:
        _terminal_stage(recorder, boundary, verilog, target, traceability, report)
    else:
        _synthesis_only_stage(
            recorder, boundary, verilog, target, tool_status, choice.pnr_reason, report
        )
    boundary["does_not_establish"].append(
        "measured hardware behaviour, timing, power or resources: no device was "
        "programmed or measured"
    )


@dataclass(frozen=True, slots=True)
class _StageChoice:
    """What the capabilities stage fixes for the stages after it.

    Attributes
    ----------
    detail : dict or None
        The model's catalogue detail, read once and handed to the configuration stage.
    integrator : str
        The integrator the run compiles with.
    q_format : str
        The number format the run compiles with.
    pnr_runs : bool
        Whether place and route runs for the target.
    pnr_reason : str or None
        Why place and route does not run, when it does not.
    """

    detail: dict[str, Any] | None
    integrator: str
    q_format: str
    pnr_runs: bool
    pnr_reason: str | None


def _capabilities_stage(
    recorder: _Recorder,
    *,
    tool_status: Mapping[str, Mapping[str, object]],
    q_format: str | None,
    integrator: str | None,
    target: str,
    formal_inventory: Path,
) -> _StageChoice:
    """Gate the run on the model's capabilities and fix its configuration choices.

    Parameters
    ----------
    recorder : _Recorder
        The run's recorder.
    tool_status : mapping
        The installed EDA tools, as ``check_tools`` reports them.
    q_format, integrator : str or None
        The caller's choices, or None for the model's defaults.
    target : str
        The synthesis target.
    formal_inventory : Path
        The catalogue formal job inventory.

    Returns
    -------
    _StageChoice
        The detail and the choices the later stages use.

    Raises
    ------
    HardwarePathRefusal
        The model is unknown, a required operation is disabled, the target is unknown, or
        the chosen integrator and number format are not a supported combination.
    """
    model = recorder.model
    # 1. Capabilities.
    capabilities = model_capabilities(
        model, tool_status=tool_status, formal_inventory=formal_inventory
    )
    if capabilities is None:
        raise HardwarePathRefusal(
            "capabilities", "unknown-model", f"{model!r} is not a catalogue model"
        )
    operations: dict[str, Any] = capabilities["operations"]
    refusal = gate_refusal(operations)
    if refusal is not None:
        operation, reason = refusal
        raise HardwarePathRefusal(
            "capabilities", f"{operation}-refused", f"{model}: {operation} refused: {reason}"
        )
    if target not in supported_targets():
        raise HardwarePathRefusal(
            "capabilities",
            "unknown-target",
            f"synthesis target {target!r} is not one of {', '.join(supported_targets())}",
        )
    detail = get_model_detail(model)
    configuration_choices: dict[str, Any] = (detail or {})["compile_configuration"]
    chosen_integrator = integrator or str(configuration_choices["default_integrator"])
    chosen_format = q_format or str(configuration_choices["default_q_format"])
    _check_combination(model, operations, configuration_choices, chosen_integrator, chosen_format)
    pnr_runs, pnr_reason = _place_and_route_choice(operations["place_and_route"], target)
    recorder.stage(
        "capabilities",
        integrator=chosen_integrator,
        q_format=chosen_format,
        operations={
            name: {"enabled": entry["enabled"], "reason": entry["reason"]}
            for name, entry in operations.items()
        },
        place_and_route={"runs": pnr_runs, "reason": pnr_reason},
    )
    return _StageChoice(detail, chosen_integrator, chosen_format, pnr_runs, pnr_reason)


def _configuration_stage(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    choice: _StageChoice,
    *,
    params: dict[str, float],
    dt: float | None,
    module_name: str | None,
) -> tuple[ResolvedModelCompileConfiguration, str, Mapping[str, object]]:
    """Resolve the compile configuration, emit its RTL and record both.

    Parameters
    ----------
    recorder : _Recorder
        The run's recorder.
    boundary : dict of str to list of str
        What the run establishes and does not establish.
    choice : _StageChoice
        The capabilities stage's detail and choices.
    params : dict of str to float
        Parameter overrides for the compiled schema.
    dt : float or None
        The time step, or None for the schema's own.
    module_name : str or None
        The RTL module name, or None for the default.

    Returns
    -------
    tuple
        The resolved configuration, its Verilog and its public traceability record.

    Raises
    ------
    HardwarePathRefusal
        The configuration is refused.
    """
    model = recorder.model
    detail = choice.detail
    # 2. Configuration and RTL.
    payload: dict[str, object] = {
        "model_name": model,
        "q_format": choice.q_format,
        "integrator": choice.integrator,
        "params": params,
        "dt": dt,
        "module_name": module_name,
    }
    try:
        configuration = resolve_model_compile_configuration(
            payload, detail_getter=lambda _name: detail
        )
    except ValueError as error:
        raise HardwarePathRefusal(
            "configuration", "configuration-refused", f"{model}: {error}"
        ) from None
    verilog = configuration.to_verilog()
    rtl_relative = f"rtl/{configuration.module_name}.v"
    rtl_sha256 = recorder.write(rtl_relative, verilog.encode("utf-8"))
    traceability = configuration.to_traceability(verilog).to_public_dict()
    recorder.write_json("rtl/compile_traceability.json", traceability)
    recorder.write_json("rtl/configuration.json", configuration.to_public_dict())
    recorder.stage(
        "configuration",
        module_name=configuration.module_name,
        schema_name=configuration.schema_name,
        schema_sha256=configuration.schema_sha256,
        rtl=rtl_relative,
        rtl_sha256=rtl_sha256,
    )
    boundary["establishes"].append(
        f"{rtl_relative} (SHA-256 {rtl_sha256}) is the compiler's RTL for {model} at "
        f"{configuration.q_format.q_label} with {configuration.integrator}, dt "
        f"{configuration.dt!r} and parameter overrides {configuration.params}"
    )
    return configuration, verilog, traceability


def _cosimulation_stage(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    configuration: ResolvedModelCompileConfiguration,
    *,
    cosimulate: Callable[..., ModelCosimExecution],
    current: float,
    n_steps: int,
) -> Mapping[str, Any]:
    """Co-simulate the RTL against the bit-true kernel and pass on only a bit-exact report.

    Parameters
    ----------
    recorder : _Recorder
        The run's recorder.
    boundary : dict of str to list of str
        What the run establishes and does not establish.
    configuration : ResolvedModelCompileConfiguration
        The configuration whose RTL is co-simulated.
    cosimulate : callable
        Runs the co-simulation.
    current : float
        The input current.
    n_steps : int
        The cycle count.

    Returns
    -------
    Mapping
        The bit-exact co-simulation report.

    Raises
    ------
    HardwarePathRefusal
        The co-simulation is refused, fails or is not bit-exact.
    """
    model = recorder.model
    # 3. Co-simulation.
    try:
        execution = cosimulate(configuration, current=current, n_steps=n_steps)
    except ValueError as error:
        raise HardwarePathRefusal("cosimulation", "cosim-refused", f"{model}: {error}") from None
    except RuntimeError as error:
        raise HardwarePathRefusal("cosimulation", "cosim-failed", f"{model}: {error}") from None
    recorder.tools["cosimulation"] = execution.report["tools"]
    _write_cosim(recorder, execution)
    # Only the report this check returns is passed on: RTL whose co-simulation
    # is not bit-exact never reaches the guarantee summary or synthesis.
    report = require_bit_exact(execution.report, model_name=model)
    recorder.stage(
        "cosimulation",
        bit_exact=True,
        report="cosim/report.json",
        sample_count=report["sample_count"],
        stress_sample_count=report["stress"]["sample_count"],
    )
    stimulus = report["stimulus"]
    boundary["establishes"].append(
        f"bit-exact equality of that RTL with the generated bit-true C kernel at "
        f"{configuration.q_format.q_label}, for the recorded stimuli only: "
        f"{stimulus['n_steps']} cycles at current {stimulus['current']!r} "
        f"(word {stimulus['current_q']}) and the {report['stress']['sample_count']}-cycle "
        "stress schedule in cosim/report.json"
    )
    boundary["does_not_establish"].append(
        "equality with the floating-point model: the reference is the bit-true C kernel "
        f"at {configuration.q_format.q_label}, not the float model"
    )
    boundary["does_not_establish"].extend(
        f"co-simulation does not cover {item}" for item in report["boundary"]["not_covered"]
    )
    return report


def _guarantee_stage(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    configuration: ResolvedModelCompileConfiguration,
    verilog: str,
    report: Mapping[str, Any],
    *,
    formal_inventory: Path,
    readiness_seal: Path,
) -> None:
    """Classify what the RTL is guaranteed by, write the summary and record its formal reach.

    Parameters
    ----------
    recorder : _Recorder
        The run's recorder.
    boundary : dict of str to list of str
        What the run establishes and does not establish.
    configuration : ResolvedModelCompileConfiguration
        The compiled configuration.
    verilog : str
        Its RTL.
    report : Mapping
        The bit-exact co-simulation report.
    formal_inventory : Path
        The catalogue formal job inventory.
    readiness_seal : Path
        The Studio readiness seal.
    """
    # 4. Guarantee summary.
    summary = model_guarantee_summary(
        configuration,
        verilog,
        report,
        formal_inventory=formal_inventory,
        readiness_seal=readiness_seal,
    )
    recorder.write_json("guarantees.json", summary)
    recorder.write("guarantees.txt", render_model_guarantee_text(summary).encode("utf-8"))
    formal = summary["guarantees"][1]
    recorder.stage(
        "guarantees",
        classes=summary["classes"],
        formal_class=formal["class"],
        readiness=summary["readiness"],
    )
    if formal["class"] == "FORMAL":
        job = formal["job"]
        boundary["establishes"].append(
            f"the RTL is byte-identical to the RTL of catalogue formal job {job['module']}, "
            f"whose {job['claim']} at depth {job['depth']} the inventory records; the job "
            "was not run here"
        )
    boundary["does_not_establish"].append(
        "any formal result beyond bounded safety on a job's own RTL: catalogue jobs "
        "establish no equivalence with the model or the bit-true kernel and nothing "
        "after their depth"
    )
    if formal["class"] != "FORMAL":
        boundary["does_not_establish"].append(
            f"any formal property of this RTL ({formal['class']}: {formal['reason']})"
        )


def gate_refusal(operations: Mapping[str, Mapping[str, Any]]) -> tuple[str, str] | None:
    """Return the first required operation the capability matrix disables, and why.

    Parameters
    ----------
    operations : Mapping
        ``operations`` of a :func:`model_capabilities` matrix.

    Returns
    -------
    tuple of (str, str) or None
        The operation and its reason, or ``None`` when compile, co-simulation
        and synthesis are all enabled.
    """
    for name in REQUIRED_OPERATIONS:
        entry = operations[name]
        if not entry["enabled"]:
            return name, str(entry["reason"])
    return None


def _check_combination(
    model: str,
    operations: Mapping[str, Any],
    choices: Mapping[str, Any],
    integrator: str,
    q_format: str,
) -> None:
    """Refuse an integrator or format the capability matrix does not co-simulate."""
    declared = list(operations["compile"]["integrators"])
    if integrator not in declared:
        raise HardwarePathRefusal(
            "capabilities",
            "integrator-not-declared",
            f"integrator {integrator!r} is not declared for {model}; declared: "
            f"{', '.join(declared)}",
        )
    mirrored = mirrored_combinations(operations["cosimulate"])
    cosimulated = sorted({entry["integrator"] for entry in mirrored})
    if integrator not in cosimulated:
        raise HardwarePathRefusal(
            "capabilities",
            "integrator-not-cosimulated",
            f"no bit-true C kernel mirrors {model} under integrator {integrator!r}, so its RTL "
            f"cannot be co-simulated; co-simulated integrators: {', '.join(cosimulated)}",
        )
    offered = list(operations["compile"]["q_formats"])
    if q_format not in offered:
        contract = choices.get("numeric_contracts", {}).get(q_format)
        why = (
            f"; {contract['refusal']}"
            if isinstance(contract, Mapping) and contract.get("refusal")
            else ""
            if q_format in STUDIO_Q_FORMATS
            else f"; Studio compiles at {', '.join(STUDIO_Q_FORMATS)}"
        )
        raise HardwarePathRefusal(
            "capabilities",
            "q-format-not-offered",
            f"Q format {q_format!r} does not hold {model}; offered: {', '.join(offered)}{why}",
        )
    if {"integrator": integrator, "q_format": q_format} not in mirrored:
        raise HardwarePathRefusal(
            "capabilities",
            "combination-not-mirrored",
            f"no bit-true C kernel mirrors {model} under {integrator} at {q_format}",
        )


def mirrored_combinations(cosimulate: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return the integrator and format pairs a bit-true C kernel mirrors.

    Parameters
    ----------
    cosimulate : Mapping
        The ``cosimulate`` operation of a capability matrix.

    Returns
    -------
    list of dict
        ``{"integrator": ..., "q_format": ...}`` per mirrored pair, in matrix
        order; empty when the matrix lists none.
    """
    return [
        {"integrator": str(entry["integrator"]), "q_format": str(entry["q_format"])}
        for entry in cosimulate.get("combinations", [])
        if entry["mirrored"]
    ]


def _place_and_route_choice(pnr: Mapping[str, Any], target: str) -> tuple[bool, str | None]:
    """Say whether place and route runs for ``target`` and, if not, why."""
    if target in pnr.get("targets", []):
        return True, None
    disabled = pnr.get("disabled_targets", {})
    return False, str(disabled.get(target) or pnr["reason"])


def _write_cosim(recorder: _Recorder, execution: ModelCosimExecution) -> None:
    """Write the co-simulation report and the sources it compiled."""
    recorder.write_json("cosim/report.json", execution.report)
    recorder.write("cosim/reference.c", execution.reference_source.encode("utf-8"))
    recorder.write("cosim/testbench.v", execution.rtl_testbench.encode("utf-8"))


def require_bit_exact(report: Mapping[str, Any], *, model_name: str) -> Mapping[str, Any]:
    """Stop the path unless a co-simulation report is completed and bit-exact.

    Parameters
    ----------
    report : Mapping
        A ``studio.cosim-parity.v1`` report.
    model_name : str
        The model, named in the refusal.

    Returns
    -------
    Mapping
        ``report`` itself, which the later stages consume.

    Raises
    ------
    HardwarePathRefusal
        ``cosim-not-bit-exact`` at stage ``cosimulation``, naming the first
        differing cycle and signals of the requested run or of the stress
        schedule.
    """
    if report.get("status") == "completed" and report.get("bit_exact") is True:
        return report
    mismatch = report.get("first_mismatch")
    where = "requested run"
    if mismatch is None:
        mismatch = (report.get("stress") or {}).get("first_mismatch")
        where = "stress schedule"
    detail = (
        f"first difference in the {where} at cycle {mismatch['cycle']} on "
        f"{', '.join(mismatch['signals'])}"
        if isinstance(mismatch, Mapping)
        else f"status {report.get('status')!r}"
    )
    raise HardwarePathRefusal(
        "cosimulation",
        "cosim-not-bit-exact",
        f"{model_name}: the RTL does not agree with the bit-true C kernel ({detail}); "
        "the report is in cosim/report.json and the RTL is not synthesised",
    )


def _terminal_stage(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    verilog: str,
    target: str,
    traceability: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    """Run digest-bound synthesis and place and route, and record both."""
    try:
        execution = run_synthesis_terminal(
            verilog, target, compile_traceability=traceability, cosim_parity=report
        )
    except ValueError as error:
        raise HardwarePathRefusal(
            "synthesis", "terminal-refused", f"{recorder.model}: {error}"
        ) from None
    terminal = execution.report
    report_sha256 = recorder.write_json("synthesis/report.json", terminal)
    digests: dict[str, str | None] = {"report_sha256": report_sha256}
    if execution.netlist_json is not None:
        digests["netlist_sha256"] = recorder.write("synthesis/netlist.json", execution.netlist_json)
    if execution.routed_design is not None:
        suffix = "config" if target == "ecp5" else "asc"
        digests["routed_design_sha256"] = recorder.write(
            f"synthesis/routed.{suffix}", execution.routed_design
        )
    synthesis = terminal["synthesis"]
    if not synthesis.get("success"):
        raise HardwarePathRefusal(
            "synthesis", "synthesis-failed", f"{recorder.model}: {synthesis.get('error')}"
        )
    recorder.stage("synthesis", target=target, **_synthesis_record(synthesis), **digests)
    pnr = terminal["place_and_route"]
    if not terminal["success"]:
        raise HardwarePathRefusal(
            "place_and_route",
            "place-and-route-failed",
            f"{recorder.model}: {pnr.get('error') or 'nextpnr did not complete'} for {target}",
        )
    recorder.stage(
        "place_and_route",
        target=target,
        max_freq_mhz=pnr.get("max_freq_mhz"),
        critical_path=pnr.get("critical_path"),
    )
    boundary["establishes"].append(
        f"Yosys synthesised and nextpnr placed and routed the co-simulated RTL bytes for "
        f"{target}; the resource counts and the {pnr.get('max_freq_mhz')} MHz maximum "
        "frequency are the tools' estimates"
    )


def _synthesis_only_stage(
    recorder: _Recorder,
    boundary: dict[str, list[str]],
    verilog: str,
    target: str,
    tool_status: ToolStatusMap,
    pnr_reason: str | None,
    report: Mapping[str, Any],
) -> None:
    """Run Yosys alone and record place and route as not run, with the reason.

    Unlike the place-and-route terminal, :func:`run_synthesis` takes no
    co-simulation evidence, so the bit-exact check is repeated here: this stage
    never synthesises RTL whose co-simulation did not agree.
    """
    require_bit_exact(report, model_name=recorder.model)
    result = run_synthesis(verilog, target, tool_status=tool_status)
    report_sha256 = recorder.write_json("synthesis/report.json", result)
    if not result.get("success"):
        raise HardwarePathRefusal(
            "synthesis", "synthesis-failed", f"{recorder.model}: {result.get('error')}"
        )
    recorder.stage(
        "synthesis", target=target, report_sha256=report_sha256, **_synthesis_record(result)
    )
    recorder.stages.append(
        {"stage": "place_and_route", "status": "not run", "target": target, "reason": pnr_reason}
    )
    boundary["establishes"].append(
        f"Yosys synthesised the co-simulated RTL for {target}; the resource counts are the "
        "tool's estimates"
    )
    boundary["does_not_establish"].append(
        f"place and route or timing for {target}: not run ({pnr_reason})"
    )


def _synthesis_record(synthesis: Mapping[str, Any]) -> dict[str, object]:
    """Return the resource figures of one synthesis result."""
    return {
        "resources": synthesis.get("resources"),
        "fits_device": synthesis.get("fits_device"),
        "capacity_device": synthesis.get("capacity_device"),
    }


def supported_models(
    names: Sequence[str] | None = None,
    *,
    tool_status: Mapping[str, Mapping[str, object]] | None = None,
    capabilities: Callable[..., dict[str, Any] | None] = model_capabilities,
) -> list[dict[str, Any]]:
    """Say, for every catalogue model, whether the hardware path is supported and why not.

    Two verdicts are given. ``model_supported`` depends only on the model: it
    compiles and a bit-true C kernel mirrors at least one integrator and
    format. ``supported`` is the verdict on this host: the capability gate of
    :func:`run_hardware_path` also needs the co-simulation tools and Yosys.

    Parameters
    ----------
    names : Sequence[str] or None, optional
        Models to judge; every registered catalogue model when omitted.
    tool_status : Mapping or None, optional
        An EDA tool snapshot as :func:`~sc_neurocore.studio.synthesis.check_tools`
        returns; taken once when omitted.
    capabilities : Callable, optional
        The capability matrix; :func:`model_capabilities`.

    Returns
    -------
    list of dict
        Per model: ``model``, ``model_supported``, ``model_refusal`` (the
        operation and reason, or ``None``), ``supported``, ``refusal`` (the
        first required operation this host disables, or ``None``) and the
        mirrored ``combinations``.
    """
    if names is None:
        from sc_neurocore.neurons.model_registry import MODEL_MODULES

        names = sorted(MODEL_MODULES)
    tools = check_tools() if tool_status is None else tool_status
    rows: list[dict[str, Any]] = []
    for name in names:
        matrix = capabilities(name, tool_status=tools)
        if matrix is None:
            refusal = {"operation": "catalogue", "reason": "the catalogue holds no descriptor"}
            rows.append(
                {
                    "model": name,
                    "model_supported": False,
                    "model_refusal": refusal,
                    "supported": False,
                    "refusal": refusal,
                    "combinations": [],
                }
            )
            continue
        operations = matrix["operations"]
        combinations = mirrored_combinations(operations["cosimulate"])
        model_refusal: dict[str, str] | None = None
        if not operations["compile"]["enabled"]:
            model_refusal = {"operation": "compile", "reason": operations["compile"]["reason"]}
        elif not combinations:
            model_refusal = {
                "operation": "cosimulate",
                "reason": operations["cosimulate"]["reason"],
            }
        gate = gate_refusal(operations)
        rows.append(
            {
                "model": name,
                "model_supported": model_refusal is None,
                "model_refusal": model_refusal,
                "supported": gate is None,
                "refusal": None if gate is None else {"operation": gate[0], "reason": gate[1]},
                "combinations": combinations,
            }
        )
    return rows


def render_hardware_path_text(manifest: Mapping[str, Any]) -> str:
    """Render a hardware path manifest as text for people.

    Parameters
    ----------
    manifest : Mapping
        A manifest as :func:`run_hardware_path` writes it.

    Returns
    -------
    str
        Status, each stage, the refusal if any, the artefacts with their
        SHA-256 and the boundary statement.
    """
    lines = [
        f"Hardware path of {manifest['model']}: {manifest['status'].upper()}",
        f"({manifest['schema_version']}, sc-neurocore {manifest['package_version']})",
        "",
        "Stages:",
    ]
    for stage in manifest["stages"]:
        reason = f": {stage['reason']}" if stage.get("reason") else ""
        lines.append(f"  {stage['stage']}: {stage['status']}{reason}")
    refusal = manifest["refusal"]
    if refusal is not None:
        lines.append(f"  {refusal['stage']}: REFUSED ({refusal['code']}) {refusal['message']}")
    lines.extend(["", "Artefacts:"])
    lines.extend(f"  {path}  {entry['sha256']}" for path, entry in manifest["artifacts"].items())
    lines.extend(["", "Establishes:"])
    lines.extend(f"  - {item}" for item in manifest["boundary"]["establishes"])
    lines.extend(["", "Does not establish:"])
    lines.extend(f"  - {item}" for item in manifest["boundary"]["does_not_establish"])
    return "\n".join(lines) + "\n"


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


__all__ = [
    "DEFAULT_CURRENT",
    "DEFAULT_N_STEPS",
    "DEFAULT_TARGET",
    "HARDWARE_PATH_SCHEMA_VERSION",
    "HARDWARE_PATH_SUPPORT_SCHEMA_VERSION",
    "MODEL_GUARANTEE_SCHEMA_VERSION",
    "PLACE_AND_ROUTE_TARGETS",
    "REQUIRED_OPERATIONS",
    "HardwarePathRefusal",
    "HardwarePathResult",
    "formal_guarantee",
    "gate_refusal",
    "mirrored_combinations",
    "model_guarantee_summary",
    "readiness_labels",
    "render_hardware_path_text",
    "render_model_guarantee_text",
    "require_bit_exact",
    "run_hardware_path",
    "supported_models",
]
