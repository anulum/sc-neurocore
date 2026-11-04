# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native GIA equivalence with explicit output aggregation

"""Compare every named model bit through a retained, full-domain singleton miter."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_aiger_header,
    check_equivalent_verdict,
    digest,
    read_aiger_port_order,
)
from tools.function_synthesis_netlist import object_mapping
from tools.function_synthesis_toolchain import SynthesisToolchain

_RECORD = "gia-proof.json"
_CONSTRUCTION = (
    "&read reference-plain.aig; &miter candidate-plain.aig; "
    "&write all-differences.aig; &put; orpos; &get; &write singleton.aig"
)
_PLAIN = ("reference-plain.aig", "candidate-plain.aig")
_LOGS = (
    "cec-help",
    "reference-io",
    "candidate-io",
    "reference-normalize",
    "candidate-normalize",
    "reference-roundtrip",
    "candidate-roundtrip",
    "gia-construction",
    "equivalence",
)
_GENERATED = (*_PLAIN, "all-differences.aig", "singleton.aig") + tuple(
    name + suffix for name in _LOGS for suffix in (".log", ".command.json")
)


def _encoded(payload: object) -> str:
    """Retain one canonical complete protocol record."""
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _sources() -> dict[str, str]:
    """Bind the actual engine, complete-interface admission and native command implementation."""
    return {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "function_synthesis_gia.py",
            "function_synthesis_aiger.py",
            "function_synthesis_netlist.py",
            "function_synthesis_toolchain.py",
        )
    }


def _check_log(log: Path) -> None:
    """Refuse a native construction or normalization error even with a zero process status."""
    if re.search(
        r"^\s*(?:ERROR|Error|Aborting|Aborted|\*\* cmd error)\b",
        log.read_text(encoding="utf-8"),
        re.M,
    ):
        raise ValueError("A complete GIA native transformation did not succeed")


def _commands(reference: Path, candidate: Path, timeout: float) -> dict[str, tuple[str, float]]:
    """Derive the full original-domain protocol and each unchanged native ceiling."""
    seconds = max(1, min(60, int(timeout)))
    scripts = {"cec-help": ("cec -h", min(timeout, 15.0))}
    for path, label, plain in zip(
        (reference, candidate), ("reference", "candidate"), _PLAIN, strict=True
    ):
        scripts[label + "-io"] = (f"read_aiger {path.name}; print_io", min(timeout, 15.0))
        scripts[label + "-normalize"] = (f"read_aiger {path.name}; write_aiger {plain}", timeout)
        scripts[label + "-roundtrip"] = (f"cec -T {seconds} -n -v {path.name} {plain}", timeout)
    scripts["gia-construction"] = (_CONSTRUCTION, timeout)
    scripts["equivalence"] = (f"&read singleton.aig; &cec -m -T {seconds} -v", timeout)
    return scripts


def _check_commands(
    reference: Path, candidate: Path, directory: Path, tools: SynthesisToolchain, timeout: float
) -> None:
    """Bind every retained invocation to its inspected executables, script and finite deadline."""
    for label, (script, ceiling) in _commands(reference, candidate, timeout).items():
        expected = [
            str(tools.limits),
            "--core=0",
            "--",
            str(tools.deadline),
            "--signal=KILL",
            "--verbose",
            str(ceiling),
            str(tools.abc),
            "-s",
            "-c",
            script,
        ]
        actual: object = json.loads((directory / (label + ".command.json")).read_bytes())
        if actual != expected:
            raise ValueError("A complete GIA native command differs from its original protocol")
        _check_log(directory / (label + ".log"))


def _paths(reference: Path, candidate: Path, directory: Path) -> None:
    """Require distinct direct local exports before constructing native command tokens."""
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or reference == candidate
        or any(
            path.parent.resolve() != directory.resolve()
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*[.]aig", path.name) is None
            or path.name in _GENERATED
            for path in (reference, candidate)
        )
    ):
        raise ValueError("Complete GIA proofs require distinct original local AIG exports")


def _artifacts(reference: Path, candidate: Path, directory: Path) -> dict[str, str]:
    """Require every complete input, transformation and native proof artifact to remain regular."""
    names = (reference.name, candidate.name, *_GENERATED)
    if any((directory / name).is_symlink() or not (directory / name).is_file() for name in names):
        raise ValueError("Complete GIA proof artifacts must be regular retained files")
    return {name: digest(directory / name) for name in names}


def check_gia_equivalent_verdict(log: Path) -> None:
    """Require decisive complete equality from the known native GIA protocol.

    Parameters
    ----------
    log : pathlib.Path
        Actual native checker output for the already validated singleton miter.

    Notes
    -----
    This ABC version can reduce the entire miter to a constant, report its new
    engine undecided, then prove the reduced miter with its old checker. That
    transcript is admitted only with explicit empty-reduction and zero-node
    final checks. The existing single-verdict CEC admission remains unchanged.

    Raises
    ------
    OSError, ValueError
        For errors, negative, incomplete, contradictory or unrecognized transcripts.

    """
    try:
        check_equivalent_verdict(log)
        return
    except ValueError:
        text = log.read_text(encoding="utf-8")
    verdicts = [line for line in text.splitlines() if line.startswith("Networks are ")]
    if (
        len(verdicts) != 2
        or re.fullmatch(
            r"Networks are UNDECIDED after the new CEC engine\.[ \t]+Time =.*", verdicts[0]
        )
        is None
        or re.fullmatch(r"Networks are equivalent\.[ \t]+Time =.*", verdicts[1]) is None
        or re.search(r"^\s*(?:ERROR|Error)\b", text, re.M)
        or not (
            "Network after reduction is empty.\n" in text
            or re.search(r"^NBeg = 0\. NEnd = 0\.", text, re.M)
        )
    ):
        raise ValueError("Complete native GIA equivalence was not proved")
    markers = (
        re.escape(verdicts[0]),
        r"Calling the old CEC engine\.",
        r"Original miter:[ \t]+Nodes =[ \t]+0\.",
        r"Initial SAT:[ \t]+Nodes =[ \t]+0\.[ \t]+Time =[^\n]*",
        re.escape(verdicts[1]),
    )
    if re.search(r"\n".join(markers), text) is None:
        raise ValueError("The GIA fallback did not prove the complete reduced singleton miter")


def _completed(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
    timestamp: str,
    native_seconds: float,
) -> dict[str, object]:
    """Derive complete current bindings from the actual native protocol artifacts."""
    _paths(reference, candidate, directory)
    artifacts = _artifacts(reference, candidate, directory)
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo != UTC or parsed.isoformat() != timestamp:
        raise ValueError("A complete GIA proof requires its original canonical UTC timestamp")
    if (
        type(native_seconds) is not float
        or not math.isfinite(native_seconds)
        or not 0 < native_seconds <= 90
    ):
        raise ValueError("A complete GIA proof requires its original finite native deadline")
    _check_commands(reference, candidate, directory, tools, native_seconds)
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
    check_gia_equivalent_verdict(directory / "equivalence.log")
    return {
        "schema_version": "sc-neurocore.complete-gia-equivalence.v1",
        "engine": "gia-complete-all-output-or",
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


def read_gia_equivalence(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
) -> None:
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
    if not isinstance(timestamp, str) or type(native_seconds) is not float:
        raise ValueError("A complete GIA proof lacks its original qualification timestamp")
    expected = _completed(
        reference, candidate, interface, tools, directory, timestamp, native_seconds
    )
    if text != _encoded(expected):
        raise ValueError("Complete GIA evidence differs from its original qualified binding")


def prove_gia_equivalence(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
) -> None:
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

    Notes
    -----
    Native by-order normalization is admitted only after exact named-order
    comparison. Both complete round trips require explicit equality. The final
    checker receives one scalar OR of all differences, never a vector miter.
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
    timeout = float(timeout)
    _paths(reference, candidate, directory)
    if any(
        (directory / name).exists() or (directory / name).is_symlink()
        for name in (*_GENERATED, _RECORD)
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

    run("cec-help", "cec -h", min(timeout, 15.0))
    if "[default = by name]" not in (directory / "cec-help.log").read_text(encoding="utf-8"):
        raise ValueError("The native normalization checker did not confirm its matching policy")
    for path, label in ((reference, "reference"), (candidate, "candidate")):
        check_aiger_header(path, interface)
        run(label + "-io", f"read_aiger {path.name}; print_io", min(timeout, 15.0))
    if read_aiger_port_order(directory / "reference-io.log", interface) != read_aiger_port_order(
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
    run("equivalence", f"&read singleton.aig; &cec -m -T {seconds} -v")
    check_gia_equivalent_verdict(directory / "equivalence.log")
    if before != {path.name: digest(path) for path in (reference, candidate)} or code != _sources():
        raise ValueError("Original complete models or the GIA proof implementation changed")
    tools.verify()
    timestamp = datetime.now(UTC).isoformat()
    record = _completed(reference, candidate, interface, tools, directory, timestamp, timeout)
    with (directory / _RECORD).open("x", encoding="utf-8") as stream:
        stream.write(_encoded(record))
    read_gia_equivalence(reference, candidate, interface, tools, directory)
