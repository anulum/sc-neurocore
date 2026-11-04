# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete named AIG equivalence

"""Require explicit native equivalence of every declared primary input and output."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from tools.function_synthesis_toolchain import SynthesisToolchain


def digest(path: Path) -> str:
    """Return the SHA-256 of a regular evidence file's complete current bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class AigerInterface:
    """Describe a complete named combinational proof domain.

    Parameters
    ----------
    inputs, outputs : frozenset of str
        Every unique actual primary bit name. All original function bits remain
        present; a primitive comparison may add universally independent values.

    """

    inputs: frozenset[str]
    outputs: frozenset[str]

    def __post_init__(self) -> None:
        """Refuse empty domains or labels that cannot identify one native port bit."""
        if (
            not self.inputs
            or not self.outputs
            or any(
                not name or any(char.isspace() or char == "=" for char in name)
                for name in self.inputs | self.outputs
            )
        ):
            raise ValueError("A native proof requires complete nonempty named bit domains")


def check_aiger_header(path: Path, interface: AigerInterface) -> None:
    """Require a binary, stateless export of the complete expected bit geometry.

    Parameters
    ----------
    path : pathlib.Path
        Actual binary AIGER export from the retained native synthesis command.
    interface : AigerInterface
        Complete independent inputs and compared outputs.

    Raises
    ------
    OSError, ValueError
        For missing, linked, malformed, stateful or narrowed exports.

    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("An AIGER proof export must be a regular artifact")
    with path.open("rb") as stream:
        line = stream.readline(256)
    fields = line.split()
    if (
        not line.endswith(b"\n")
        or len(fields) != 6
        or fields[0] != b"aig"
        or not all(field.isdigit() for field in fields[1:])
    ):
        raise ValueError("A native proof requires a complete binary combinational AIGER header")
    maximum, inputs, latches, outputs, gates = map(int, fields[1:])
    if (inputs, latches, outputs) != (
        len(interface.inputs),
        0,
        len(interface.outputs),
    ) or maximum != inputs + gates:
        raise ValueError("A native AIGER export changed the complete proof geometry")


def read_aiger_port_order(
    log: Path, interface: AigerInterface
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read complete validated original input and output names in native index order.

    Parameters
    ----------
    log : pathlib.Path
        Actual ABC print_io output, not a synthesized expected response.
    interface : AigerInterface
        Complete expected input and output names.

    Returns
    -------
    tuple of (tuple of str, tuple of str)
        Exact input and output index-to-name correspondence after full validation.

    Raises
    ------
    OSError, ValueError
        If a name is absent, repeated, substituted or associated with state.

    """
    text = log.read_text(encoding="utf-8")
    orders: list[tuple[str, ...]] = []
    for label, expected in [
        ("Primary inputs", interface.inputs),
        ("Primary outputs", interface.outputs),
        ("Latches", frozenset()),
    ]:
        lines = re.findall(r"^" + label + r" \(([0-9]+)\):[ \t]*(.*)$", text, re.M)
        if len(lines) != 1 or int(lines[0][0]) != len(expected):
            raise ValueError("ABC did not confirm the complete named proof interface")
        pairs = [token.split("=", 1) for token in lines[0][1].split()]
        if any(len(pair) != 2 or not pair[0].isdigit() for pair in pairs) or (
            [int(pair[0]) for pair in pairs] != list(range(len(expected)))
            or len(pairs) != len(expected)
            or {pair[1] for pair in pairs} != expected
        ):
            raise ValueError(
                "ABC reported repeated, reordered indices or different proof bit names"
            )
        if label != "Latches":
            orders.append(tuple(pair[1] for pair in pairs))
    return orders[0], orders[1]


def check_aiger_names(log: Path, interface: AigerInterface) -> None:
    """Require every actual original name exactly once in a complete native interface.

    Parameters
    ----------
    log : pathlib.Path
        Actual retained native ABC input, output and latch listing.
    interface : AigerInterface
        Every expected independent input and compared output bit name.

    Raises
    ------
    OSError, ValueError
        For missing, repeated, substituted or stateful native interface identities.

    """
    read_aiger_port_order(log, interface)


def check_equivalent_verdict(log: Path) -> None:
    """Admit exactly one explicit complete positive native CEC verdict.

    Parameters
    ----------
    log : pathlib.Path
        Actual retained native ABC CEC output.

    Raises
    ------
    OSError, ValueError
        For a negative, absent, inconclusive, conflicting or errored verdict,
        including native commands which returned zero without proving equality.

    """
    text = log.read_text(encoding="utf-8")
    verdicts = [line for line in text.splitlines() if line.startswith("Networks are ")]
    if (
        len(verdicts) != 1
        or re.fullmatch(
            r"Networks are equivalent(?: after structural hashing)?\.[ \t]*Time =.*",
            verdicts[0],
        )
        is None
        or re.search(r"^\s*(?:ERROR|Error)\b", text, re.M)
    ):
        raise ValueError("Complete native mapped equivalence was not proved")


def prove_aiger_equivalence(
    reference: Path,
    candidate: Path,
    interface: AigerInterface,
    tools: SynthesisToolchain,
    directory: Path,
    *,
    timeout: float,
) -> None:
    """Execute one full named-domain native comparison with retained evidence.

    Parameters
    ----------
    reference, candidate : pathlib.Path
        Actual binary exports located directly in the comparison directory.
    interface : AigerInterface
        Every independent input and compared output bit.
    tools : SynthesisToolchain
        Inspected existing native executables and installed libraries.
    directory : pathlib.Path
        Existing private directory containing both exports and no proof logs.
    timeout : float
        Finite per-command native ceiling, at most ninety seconds.
        The CEC internal bound never exceeds sixty seconds.

    Raises
    ------
    OSError, ValueError
        For altered exports, unsupported matching behavior, narrowed interfaces
        or any result without a decisive positive native proof.
    SynthesisToolError
        If a real native command fails or reaches its deadline.

    """
    if (
        isinstance(timeout, bool)
        or not 0 < timeout <= 90
        or any(
            path.parent.resolve() != directory.resolve()
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*[.]aig", path.name) is None
            for path in [reference, candidate]
        )
        or reference == candidate
    ):
        raise ValueError("A native proof requires distinct local exports and an admitted deadline")
    before = {path.name: digest(path) for path in [reference, candidate]}
    tools.run(
        [str(tools.abc), "-c", "cec -h"],
        directory,
        directory / "cec-help.log",
        timeout=min(timeout, 15),
    )
    if "[default = by name]" not in (directory / "cec-help.log").read_text(encoding="utf-8"):
        raise ValueError("The native checker did not confirm matching complete inputs by name")
    for path, label in [(reference, "reference"), (candidate, "candidate")]:
        check_aiger_header(path, interface)
        log = directory / (label + "-io.log")
        tools.run(
            [str(tools.abc), "-c", f"read_aiger {path.name}; print_io"],
            directory,
            log,
            timeout=min(timeout, 15),
        )
        check_aiger_names(log, interface)
    log = directory / "equivalence.log"
    tools.run(
        [
            str(tools.abc),
            "-c",
            f"cec -T {max(1, min(60, int(timeout)))} -v {reference.name} {candidate.name}",
        ],
        directory,
        log,
        timeout=timeout,
    )
    check_equivalent_verdict(log)
    if before != {path.name: digest(path) for path in [reference, candidate]}:
        raise ValueError("Complete native AIGER exports changed during their proof")
