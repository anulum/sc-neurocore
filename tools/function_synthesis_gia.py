# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native GIA equivalence with explicit output aggregation

"""Compare every named model bit through retained, full-domain native proofs."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_equivalent_verdict,
    digest,
    read_aiger_port_order,
)
from tools.function_synthesis_gia_partition import PARTITION_DIRECTORY, check_partitioned_outputs
from tools.function_synthesis_gia_protocol import (
    _CONSTRUCTION,
    _GENERATED,
    _IVY,
    _IVY_SCRIPT,
    _PAIRED,
    _PLAIN,
    _check_commands,
    _check_ivy_unsat,
    _check_log,
    _needs_further_proof,
    _paired_script,
    _singleton_limit,
    _watchdog_elapsed,
)
from tools.function_synthesis_gia_protocol import (
    check_gia_equivalent_verdict as check_gia_equivalent_verdict,
)
from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_toolchain import SynthesisToolchain, SynthesisToolError

_RECORD = "gia-proof.json"


def _encoded(payload: object) -> str:
    """Retain one canonical complete protocol record."""
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _sources() -> dict[str, str]:
    """Bind the actual engine, complete-interface admission and native command implementation."""
    return {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "function_synthesis_gia.py",
            "function_synthesis_gia_protocol.py",
            "function_synthesis_gia_partition.py",
            "function_synthesis_aiger.py",
            "function_synthesis_checkpoint_timeout.py",
            "function_synthesis_netlist.py",
            "function_synthesis_partition.py",
            "function_synthesis_retention.py",
            "function_synthesis_toolchain.py",
        )
    }


def _paths(reference: Path, candidate: Path, directory: Path) -> None:
    """Require distinct direct local exports before constructing native command tokens."""
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or reference == candidate
        or any(
            path.parent.resolve() != directory.resolve()
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*[.]aig", path.name) is None
            or path.name in (*_GENERATED, *_PAIRED)
            for path in (reference, candidate)
        )
    ):
        raise ValueError("Complete GIA proofs require distinct original local AIG exports")


def _artifacts(
    reference: Path, candidate: Path, directory: Path, stages: tuple[str, ...]
) -> dict[str, str]:
    """Require every complete input, transformation and native proof artifact to remain regular."""
    names = (reference.name, candidate.name, *_GENERATED, *stages)
    if any((directory / name).is_symlink() or not (directory / name).is_file() for name in names):
        raise ValueError("Complete GIA proof artifacts must be regular retained files")
    return {name: digest(directory / name) for name in names}


def _completed(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
    timestamp: str,
    native_seconds: object,
    singleton_seconds: object,
) -> dict[str, object]:
    """Derive complete current bindings from the actual native protocol artifacts."""
    _paths(reference, candidate, directory)
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo != UTC or parsed.isoformat() != timestamp:
        raise ValueError("A complete GIA proof requires its original canonical UTC timestamp")
    if (
        type(native_seconds) is not float
        or not math.isfinite(native_seconds)
        or not 0 < native_seconds <= 90
    ):
        raise ValueError("A complete GIA proof requires its original finite native deadline")
    paired = _needs_further_proof(directory / "equivalence.log", tools)
    ivy = paired and _needs_further_proof(directory / "paired-equivalence.log", tools)
    stages = (*_PAIRED, *_IVY) if ivy else _PAIRED if paired else ()
    artifacts = _artifacts(reference, candidate, directory, stages)
    singleton_seconds = _singleton_limit(native_seconds, singleton_seconds)
    _check_commands(
        reference, candidate, directory, tools, native_seconds, paired, singleton_seconds, ivy=ivy
    )
    for path in (reference, candidate, *(directory / name for name in _PLAIN)):
        check_aiger_header(path, interface)
    a = read_aiger_port_order(directory / "reference-io.log", interface)
    b = read_aiger_port_order(directory / "candidate-io.log", interface)
    if a != b:
        raise ValueError("GIA comparisons require identical complete named input and output order")
    if "[default = by name]" not in (directory / "cec-help.log").read_text(encoding="utf-8"):
        raise ValueError("The native normalization checker did not confirm its matching policy")
    for label in ("reference", "candidate"):
        check_equivalent_verdict(directory / (label + "-roundtrip.log"))
    check_aiger_header(directory / "all-differences.aig", interface)
    check_aiger_header(
        directory / "singleton.aig",
        AigerInterface(interface.inputs, frozenset({"all_original_and_primitive_differences_or"})),
    )
    partition = directory / PARTITION_DIRECTORY
    partitioned = ivy and _watchdog_elapsed(directory / "ivy-equivalence.log", tools)
    if ivy and not partitioned:
        _check_ivy_unsat(directory / "ivy-equivalence.log")
    if not partitioned and (partition.exists() or partition.is_symlink()):
        raise ValueError("Unexpected native output comparisons after a complete proof")
    record: dict[str, object] = {
        "schema_version": "sc-neurocore.complete-gia-equivalence.v2"
        if paired
        else "sc-neurocore.complete-gia-equivalence.v1",
        "engine": "complete-paired-output-cec-after-undecided-gia"
        if paired
        else "gia-complete-all-output-or",
        "status": "equivalent",
        "recorded_at": timestamp,
        "native_seconds": native_seconds,
        "reference": reference.name,
        "candidate": candidate.name,
        "input_order": list(a[0]),
        "output_order": list(a[1]),
        "input_bits": len(interface.inputs),
        "all_compared_output_bits": len(interface.outputs),
        "singleton_output_bits": 1,
        "latches": 0,
        "construction": _CONSTRUCTION,
        "producer_sha256": _sources(),
        "toolchain_sha256": tools.identity,
        "artifact_sha256": artifacts,
    }
    if singleton_seconds != _singleton_limit(native_seconds, None):
        record["schema_version"] = "sc-neurocore.complete-gia-equivalence.v3"
        record["singleton_seconds"] = singleton_seconds
    if ivy:
        record["schema_version"] = "sc-neurocore.complete-gia-equivalence.v4"
        record["engine"] = "ivy-complete-all-output-or-after-undecided-gia"
    if partitioned:
        record["schema_version"] = "sc-neurocore.complete-gia-equivalence.v5"
        record["engine"] = "complete-native-output-comparisons"
        record["output_partition"] = check_partitioned_outputs(
            directory / _PLAIN[0],
            directory / _PLAIN[1],
            *a,
            tools,
            partition,
            native_seconds,
            acquire=False,
        )
    return record


def read_gia_equivalence(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
) -> str:
    """Validate retained complete GIA evidence without executing or restamping a proof.

    Parameters
    ----------
    reference, candidate : pathlib.Path
        Original distinct complete AIG models directly in directory.
    interface : AigerInterface
        Every actual original input and compared output name.
    tools : SynthesisToolchain
        Currently inspected native executables and technology libraries.
    directory : pathlib.Path
        Complete source-bound normalization, all-output miter and native proof.

    Returns
    -------
    str
        Selected native engine from the completely validated original record.

    Raises
    ------
    OSError, ValueError
        For altered, linked, narrowed, stale or unqualified retained evidence.

    """
    tools.verify()
    _paths(reference, candidate, directory)
    path = directory / _RECORD
    if path.is_symlink() or not path.is_file():
        raise ValueError("A complete GIA proof requires a regular original protocol record")
    text = path.read_text(encoding="utf-8")
    record = object_mapping(json.loads(text))
    timestamp = record.get("recorded_at")
    native_seconds = record.get("native_seconds")
    engine = record.get("engine")
    if not isinstance(timestamp, str) or not isinstance(engine, str):
        raise ValueError("A complete GIA proof lacks its original qualification timestamp")
    expected = _completed(
        reference,
        candidate,
        interface,
        tools,
        directory,
        timestamp,
        native_seconds,
        record.get("singleton_seconds"),
    )
    if text != _encoded(expected):
        raise ValueError("Complete GIA evidence differs from its original qualified binding")
    return engine


def prove_gia_equivalence(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
    singleton_seconds: int | None = None,
) -> str:
    """Prove complete ordered models using an explicit OR of every output difference.

    Parameters
    ----------
    reference, candidate : pathlib.Path
        Existing actual complete AIG exports directly in directory.
    interface : AigerInterface
        Full original independent inputs and every compared output.
    tools : SynthesisToolchain
        Existing inspected native executable and library closure.
    directory : pathlib.Path
        Existing private export directory without earlier GIA acquisitions.
    timeout : float
        Finite positive per-command ceiling, at most ninety seconds. Native
        proof commands additionally request at most sixty internal seconds.
    singleton_seconds : int or None, optional
        Smaller internal allowance for the singleton checker, between one and
        the default native ceiling. None retains that default. Normalization,
        paired checking and the enclosing proof budget keep their original
        ceilings. A nondefault allowance is retained in a v3 protocol record.

    Returns
    -------
    str
        Actual native engine after complete original-record validation.

    Notes
    -----
    Native by-order normalization is admitted only after exact named-order
    comparison. Both complete round trips require explicit equality. The final
    GIA checker receives one scalar OR of all differences. If that checker
    explicitly exhausts its native deadline without deciding, a second CEC
    command compares the complete normalized output vectors. Both strategies
    retain every input and output. If the paired GIA checker also gives a
    recognized bounded undecided result, Ivy must prove the complete scalar
    difference miter UNSAT within sixty seconds. An elapsed Ivy command can
    select exact groups of original output comparisons under the same budget.
    Negative or malformed verdicts cannot select another strategy.
    Failed native attempts remain intact and receive no successful record.

    Raises
    ------
    OSError, ValueError
        For invalid budgets, mismatched orders or incomplete/unproved evidence.
    SynthesisToolError
        If an actual native command fails or reaches its unchanged deadline.

    """
    if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 90:
        raise ValueError("A complete GIA proof requires an admitted finite native deadline")
    deadline = time.monotonic() + 3300
    if tools.cohort_deadline is not None:
        deadline = min(deadline, tools.cohort_deadline)
    tools = replace(tools, cohort_deadline=deadline)
    timeout = float(timeout)
    singleton_seconds = _singleton_limit(timeout, singleton_seconds)
    _paths(reference, candidate, directory)
    if any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in (*_GENERATED, *_PAIRED, *_IVY, _RECORD, PARTITION_DIRECTORY)
    ):
        raise ValueError("Complete GIA proofs must not overwrite earlier native artifacts")
    before = {path.name: digest(path) for path in (reference, candidate)}
    code = _sources()
    seconds = max(1, min(60, int(timeout)))

    def run(label: str, script: str, ceiling: float = timeout) -> None:
        """Execute one actual bounded native transformation with original command custody."""
        tools.run(
            [str(tools.abc), "-c", script],
            directory,
            directory / (label + ".log"),
            timeout=ceiling,
        )
        _check_log(directory / (label + ".log"))

    def undecided(label: str, script: str) -> bool:
        """Retain an elapsed native check and require another complete proof."""
        log = directory / (label + ".log")
        try:
            run(label, script)
        except SynthesisToolError as error:
            if not error.timed_out or not _needs_further_proof(log, tools):
                raise
            return True
        return _needs_further_proof(log, tools)

    run("cec-help", "cec -h", min(timeout, 15.0))
    if "[default = by name]" not in (directory / "cec-help.log").read_text(encoding="utf-8"):
        raise ValueError("The native normalization checker did not confirm its matching policy")
    for path, label in ((reference, "reference"), (candidate, "candidate")):
        check_aiger_header(path, interface)
        run(label + "-io", f"read_aiger {path.name}; print_io", min(timeout, 15.0))
    order = read_aiger_port_order(directory / "reference-io.log", interface)
    if order != read_aiger_port_order(
        directory / "candidate-io.log",
        interface,
    ):
        raise ValueError("GIA comparisons require identical complete named input and output order")
    for path, label, plain in zip(
        (reference, candidate), ("reference", "candidate"), _PLAIN, strict=True
    ):
        run(label + "-normalize", f"read_aiger {path.name}; write_aiger {plain}")
        check_aiger_header(directory / plain, interface)
        run(label + "-roundtrip", f"cec -T {seconds} -n -v {path.name} {plain}")
        check_equivalent_verdict(directory / (label + "-roundtrip.log"))
    run("gia-construction", _CONSTRUCTION)
    check_aiger_header(directory / "all-differences.aig", interface)
    check_aiger_header(
        directory / "singleton.aig",
        AigerInterface(interface.inputs, frozenset({"all_original_and_primitive_differences_or"})),
    )
    if undecided("equivalence", f"&read singleton.aig; &cec -m -T {singleton_seconds} -v"):
        if undecided("paired-equivalence", _paired_script(timeout, tools)):
            ivy_log = directory / "ivy-equivalence.log"
            try:
                run("ivy-equivalence", _IVY_SCRIPT, min(timeout, 60.0))
            except SynthesisToolError as error:
                if not error.timed_out or not _watchdog_elapsed(ivy_log, tools):
                    raise
                check_partitioned_outputs(
                    directory / _PLAIN[0],
                    directory / _PLAIN[1],
                    *order,
                    tools,
                    directory / PARTITION_DIRECTORY,
                    timeout,
                    acquire=True,
                )
            else:
                _check_ivy_unsat(ivy_log)
    if before != {path.name: digest(path) for path in (reference, candidate)} or code != _sources():
        raise ValueError("Original complete models or the GIA proof implementation changed")
    tools.verify()
    timestamp = datetime.now(UTC).isoformat()
    record = _completed(
        reference, candidate, interface, tools, directory, timestamp, timeout, singleton_seconds
    )
    if time.monotonic() > deadline:
        raise ValueError("The complete native comparison exhausted its shared deadline")
    retained = directory / _RECORD
    with retained.open("x", encoding="utf-8") as stream:
        stream.write(_encoded(record))
    engine = read_gia_equivalence(reference, candidate, interface, tools, directory)
    if time.monotonic() > deadline:
        retained.unlink()
        raise ValueError("The complete native comparison readback exhausted its shared deadline")
    return engine
