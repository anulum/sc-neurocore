# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete native GIA command and verdict protocol

"""Bind complete GIA scripts, command ceilings and decisive native solver transcripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tools.function_synthesis_aiger import check_equivalent_verdict
from tools.function_synthesis_toolchain import SynthesisToolchain


_CONSTRUCTION = (
    "&read reference-plain.aig; &miter candidate-plain.aig; "
    "&write all-differences.aig; &put; orpos; &get; &write singleton.aig"
)


_PLAIN = ("reference-plain.aig", "candidate-plain.aig")


_PAIRED = ("paired-equivalence.log", "paired-equivalence.command.json")

_IVY = ("ivy-equivalence.log", "ivy-equivalence.command.json")
_IVY_SCRIPT = "read_aiger singleton.aig; iprove -N 5 -v"


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


def _check_log(log: Path) -> None:
    """Refuse a native construction or normalization error even with a zero process status."""
    if re.search(
        r"^\s*(?:ERROR|Error|Aborting|Aborted|\*\* cmd error)\b",
        log.read_text(encoding="utf-8"),
        re.M,
    ):
        raise ValueError("A complete GIA native transformation did not succeed")


def _singleton_limit(timeout: float, requested: object) -> int:
    """Admit a smaller internal singleton allowance without changing any enclosing deadline."""
    ceiling = max(1, min(60, int(timeout)))
    if requested is None:
        return ceiling
    if type(requested) is not int or not 1 <= requested <= ceiling:
        raise ValueError("A singleton allowance must be an integer within its native ceiling")
    return requested


def _commands(
    reference: Path, candidate: Path, timeout: float, singleton_seconds: int
) -> dict[str, tuple[str, float]]:
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
    scripts["equivalence"] = (f"&read singleton.aig; &cec -m -T {singleton_seconds} -v", timeout)
    return scripts


def _paired_script(timeout: float, tools: SynthesisToolchain) -> str:
    """Compare both complete normalised output vectors under the original native ceiling."""
    if dict(tools.versions)["yosys"].startswith("Yosys 0.33 "):
        return (
            "&read reference-plain.aig; "
            f"&cec -T {max(1, min(60, int(timeout)))} -v candidate-plain.aig"
        )
    # -n matches by order: identical named orders and both normalization round trips are proved.
    return f"cec -T {max(1, min(60, int(timeout)))} -n -v reference-plain.aig candidate-plain.aig"


def _watchdog_elapsed(log: Path, tools: SynthesisToolchain) -> bool:
    """Require an exact elapsed native command without a contradictory verdict."""
    if log.is_symlink() or not log.is_file():
        raise ValueError("GIA verdict must come from a retained regular log")
    text = log.read_text(encoding="utf-8")
    _check_log(log)
    watchdog = re.findall(
        r"^(?:[^\n]*[/ ])?timeout: sending signal KILL to command [^\n]*$",
        text,
        re.M,
    )
    if watchdog:
        expected = f"sending signal KILL to command '{tools.abc}'"
        if (
            len(watchdog) != 1
            or watchdog[0]
            not in (f"{tools.deadline}: {expected}", f"{tools.deadline.name}: {expected}")
            or watchdog[0] != text.rstrip().splitlines()[-1]
            or re.search(
                r"^(?:Networks are |UNSATISFIABLE\b|SATISFIABLE\b|UNDECIDED\b)", text, re.M
            )
        ):
            raise ValueError("The complete GIA watchdog conflicts with the native transcript")
        return True
    return False


def _needs_further_proof(log: Path, tools: SynthesisToolchain) -> bool:
    """Recognize a bounded GIA result that requires a separate complete proof."""
    if _watchdog_elapsed(log, tools):
        return True
    text = log.read_text(encoding="utf-8")
    verdicts = [line for line in text.splitlines() if line.startswith("Networks are ")]
    try:
        check_gia_equivalent_verdict(log)
        return False
    except ValueError:
        if (
            len(verdicts) != 1
            or re.fullmatch(
                r"Networks are UNDECIDED after the new CEC engine\.[ \t]+Time =.*",
                verdicts[0],
            )
            is None
            or not (
                len(re.findall(r"^Timed out after [0-9]+ seconds\.$", text, re.M)) == 1
                or (
                    not re.search(r"^Timed out after ", text, re.M)
                    and text.count("Network after reduction is empty.\n") == 1
                    and len(re.findall(r"^NBeg = [0-9]+\. NEnd = 0\.", text, re.M)) == 1
                )
            )
        ):
            raise
        return True


def _check_ivy_unsat(log: Path) -> None:
    """Require one explicit UNSAT verdict for the complete scalar difference miter."""
    _check_log(log)
    text = log.read_text(encoding="utf-8")
    verdicts = re.findall(r"^(?:UNSATISFIABLE|SATISFIABLE|UNDECIDED)\b[^\n]*", text, re.M)
    if (
        len(verdicts) != 1
        or re.fullmatch(r"UNSATISFIABLE[ \t]+Time[ \t]*=[ \t]*[0-9]+\.[0-9]+ sec", verdicts[0])
        is None
        or re.search(r"^\s*(?:Timed out|.*sending signal KILL to command)\b", text, re.M)
    ):
        raise ValueError("Ivy did not prove the complete difference miter unsatisfiable")


def _check_commands(
    reference: Path,
    candidate: Path,
    directory: Path,
    tools: SynthesisToolchain,
    timeout: float,
    paired: bool,
    singleton_seconds: int,
    *,
    ivy: bool,
) -> None:
    """Bind every retained invocation to its inspected executables, script and finite deadline."""
    commands = _commands(reference, candidate, timeout, singleton_seconds)
    if paired:
        commands["paired-equivalence"] = (_paired_script(timeout, tools), timeout)
    if ivy:
        commands["ivy-equivalence"] = (_IVY_SCRIPT, min(timeout, 60.0))
    for label, (script, ceiling) in commands.items():
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
