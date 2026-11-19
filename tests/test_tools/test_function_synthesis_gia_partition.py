# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native output group proof qualification

"""Exercise original native output groups, sign cofactors and immutable readback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.function_synthesis_aiger import (
    AigerInterface,
    check_equivalent_verdict,
    read_aiger_port_order,
)
from tools.function_synthesis_gia_partition import check_partitioned_outputs
from tools.function_synthesis_toolchain import SynthesisToolchain


def _native_pair(
    root: Path, result: str
) -> tuple[Path, SynthesisToolchain, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Export two equivalent native implementations with different majority gates."""
    tools = SynthesisToolchain.inspect(root / "inspection")
    interface = AigerInterface(
        frozenset(f"argument0[{index}]" for index in range(3)),
        frozenset(f"result[{index}]" for index in range(4)),
    )
    orders = []
    for label, expression in (
        ("reference", "(a & b) | (a & c) | (b & c)"),
        ("candidate", "(a & b) | ((a | b) & c)"),
    ):
        (root / f"{label}.v").write_text(
            "module function_unit(input [2:0] argument0, output [3:0] result);\n"
            "wire a = argument0[0], b = argument0[1], c = argument0[2];\n"
            f"wire majority = {expression};\n"
            f"assign result = {result};\nendmodule\n"
        )
        tools.run(
            [
                str(tools.yosys),
                "-Q",
                "-T",
                "-p",
                f"read_verilog {label}.v; hierarchy -top function_unit; "
                f"proc; techmap; aigmap; opt_clean; write_aiger -symbols {label}.aig",
            ],
            root,
            root / f"{label}-export.log",
            timeout=90,
        )
        tools.run(
            [
                str(tools.abc),
                "-c",
                f"read_aiger {label}.aig; print_io; write_aiger {label}-plain.aig",
            ],
            root,
            root / f"{label}-io.log",
            timeout=90,
        )
        orders.append(read_aiger_port_order(root / f"{label}-io.log", interface))
    assert orders[0] == orders[1]
    return root, tools, orders[0]


@pytest.fixture(scope="module")
def real_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, SynthesisToolchain, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Retain the ordinary fixture with three repeated original outputs."""
    return _native_pair(
        tmp_path_factory.mktemp("actual-native-output-groups"),
        "{(a ^ b ^ c), majority, majority, majority}",
    )


@pytest.mark.parametrize(
    ("interrupted_stages", "comparisons"),
    [
        ((), 1),
        (("group-000",), 3),
        (("group-000", "group-000/sign-1", "group-000/sign-1/bit-0-0"), 7),
    ],
    ids=["direct", "sign-cases", "refined-cases"],
)
def test_actual_repeated_outputs_are_completely_proved_and_tampering_is_refused(
    real_pair: tuple[Path, SynthesisToolchain, tuple[tuple[str, ...], tuple[str, ...]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_stages: tuple[str, ...],
    comparisons: int,
) -> None:
    """Prove all repeated native outputs, including real deadline recovery and readback."""
    root, tools, order = real_pair
    source = (root / "reference-plain.aig", root / "candidate-plain.aig")
    comparison = tmp_path / "comparison"
    native_run = SynthesisToolchain.run
    interrupted = {comparison / stage / "equivalence.log" for stage in interrupted_stages}

    def shorter_comparisons(
        toolchain: SynthesisToolchain,
        command: list[str],
        directory: Path,
        log: Path,
        *,
        timeout: float,
    ) -> None:
        """Require real elapsed native commands at recorded branch points."""
        seconds = 0.0001 if log in interrupted else timeout
        native_run(toolchain, command, directory, log, timeout=seconds)

    monkeypatch.setattr(SynthesisToolchain, "run", shorter_comparisons)
    args = (*source, *order, tools, comparison, 90.0)
    result = check_partitioned_outputs(*args, acquire=True)
    assert result["representative_original_output_indices"] == [[0]]
    assert result["input_bits"] == 3 and result["output_bits"] == 4
    assert result["native_comparisons"] == comparisons
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in comparison.rglob("*")
        if path.is_file()
    }
    assert check_partitioned_outputs(*args, acquire=False) == result
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    path = comparison / "group-000/reference.aig"
    if comparisons == 7:
        group = comparison / "group-000"
        leaves = ("sign-0", "sign-1/bit-0-0/bit-1-0", "sign-1/bit-0-0/bit-1-1", "sign-1/bit-0-1")
        for leaf in leaves:
            check_equivalent_verdict(group / leaf / "equivalence.log")
        deepest = group / leaves[2] / "equivalence.log"
        saved = deepest.read_bytes()
        deepest.unlink()
        with pytest.raises(ValueError, match="bounded regular artifact"):
            check_partitioned_outputs(*args, acquire=False)
        deepest.write_bytes(saved)
        binding = deepest.with_name("reference-cofactor.command.json")
        original_command = binding.read_bytes()
        changed_command = json.loads(original_command)
        pivot = changed_command[-1].rindex("cof ")
        prefix, _, remainder = changed_command[-1][pivot:].partition(";")
        assert prefix.endswith(" 1")
        changed_command[-1] = changed_command[-1][:pivot] + prefix[:-1] + "0;" + remainder
        binding.write_text(json.dumps(changed_command))
        with pytest.raises(ValueError, match="retained checkpoint command differs"):
            check_partitioned_outputs(*args, acquire=False)
        binding.write_bytes(original_command)
        assert check_partitioned_outputs(*args, acquire=False) == result
    if not interrupted:
        first, *literals, body = source[0].read_bytes().split(b"\n", 5)
        original = first.split()
        compact = path.read_bytes().split(b"\n", 1)[0].split()
        assert compact[2:4] == original[2:4]
        assert int(compact[5]) < int(original[5])
        original[4] = b"1"
        unsliced = tmp_path / "original-with-unreachable-gates.aig"
        unsliced.write_bytes(b"\n".join((b" ".join(original), literals[0], body)))
        (tmp_path / "group-compact.aig").write_bytes(path.read_bytes())
        comparison_log = tmp_path / "retained-cone-equivalence.log"
        tools.run(
            [
                str(tools.abc),
                "-c",
                "cec -T 60 -n -v original-with-unreachable-gates.aig group-compact.aig",
            ],
            tmp_path,
            comparison_log,
            timeout=90,
        )
        check_equivalent_verdict(comparison_log)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed its original AIGER graph"):
        check_partitioned_outputs(*args, acquire=False)


def test_two_output_subgroups_are_proved_one_original_output_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require all four native leaves after the complete multi-output comparison expires."""
    root, tools, order = _native_pair(
        tmp_path, "{(majority ^ a), ~(majority ^ a), majority, ~majority}"
    )
    comparison = root / "comparison"
    group = comparison / "group-000"
    native_run = SynthesisToolchain.run

    def expire_original_group(
        toolchain: SynthesisToolchain,
        command: list[str],
        directory: Path,
        log: Path,
        *,
        timeout: float,
    ) -> None:
        """Exercise the operating-system deadline on the actual four-output CEC."""
        native_run(
            toolchain,
            command,
            directory,
            log,
            timeout=0.0001 if log == group / "equivalence.log" else timeout,
        )

    monkeypatch.setattr(SynthesisToolchain, "run", expire_original_group)
    args = (
        root / "reference-plain.aig",
        root / "candidate-plain.aig",
        *order,
        tools,
        comparison,
        90.0,
    )
    result = check_partitioned_outputs(*args, acquire=True)
    assert result["representative_original_output_indices"] == [[0, 1, 2, 3]]
    assert result["native_comparisons"] == 5
    for half in ("first", "second"):
        assert not (group / half / "equivalence.log").exists()
        for leaf in ("first", "second"):
            check_equivalent_verdict(group / half / leaf / "equivalence.log")
    assert check_partitioned_outputs(*args, acquire=False) == result
