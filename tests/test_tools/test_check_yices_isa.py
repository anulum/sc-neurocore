# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native object ISA admission controls

"""Exercise the actual SDK scanner with assembled supported and forbidden code."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

CHECKER = Path(__file__).resolve().parents[2] / "tools/check_yices_isa.sh"


def _scan(root: Path, instruction: str) -> subprocess.CompletedProcess[str]:
    """Assemble real machine code and invoke the public scanner CLI."""
    objects = root / "object directory"
    objects.mkdir()
    source = root / "instruction.s"
    source.write_text(
        ".intel_syntax noprefix\n.text\n.global isa_control\nisa_control:\n"
        + instruction
        + "\nret\n"
    )
    subprocess.run(
        ["gcc", "-c", str(source), "-o", str(objects / "control.o")],
        check=True,
        capture_output=True,
        timeout=15,
    )
    return subprocess.run(
        ["bash", str(CHECKER), str(objects), str(root / "scan")],
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize("instruction", ["mov eax, edi\nadd eax, 1", "mov edi, 1\ntzcnt eax, edi"])
def test_baseline_object_is_admitted(tmp_path: Path, instruction: str) -> None:
    """Admit actual baseline x86-64 machine code with spaces in its path."""
    result = _scan(tmp_path, instruction)
    assert result.returncode == 0, result.stderr
    assert "PASS: 1 objects" in result.stdout
    assert "deliberately excluded" in result.stdout
    assert (tmp_path / "scan/disassembly.txt").stat().st_size > 0
    assert (tmp_path / "scan/unsupported.txt").read_text() == ""


@pytest.mark.parametrize(
    "instruction",
    [
        "vpxor ymm0, ymm0, ymm0",
        "vpxor xmm0, xmm0, xmm0",
        "vpxord zmm0, zmm0, zmm0",
        "vzeroupper",
        "kmovw k1, eax",
        "andn eax, ebx, ecx",
        "bextr eax, ebx, ecx",
        "blsi eax, ebx",
        "blsmsk eax, ebx",
        "blsr eax, ebx",
        "bzhi eax, ebx, ecx",
        "mulx eax, ebx, ecx",
        "pdep eax, ebx, ecx",
        "pext eax, ebx, ecx",
        "rorx eax, ebx, 1",
        "sarx eax, ebx, ecx",
        "shlx eax, ebx, ecx",
        "shrx eax, ebx, ecx",
        "lzcnt eax, ebx",
        "adcx eax, ebx",
        "adox eax, ebx",
        "movbe eax, [rbx]",
        "rdseed eax",
        "rdrand eax",
        "sha256rnds2 xmm0, xmm1, xmm0",
        "sha1msg1 xmm0, xmm1",
        "gf2p8mulb xmm0, xmm1",
        "gf2p8affineqb xmm0, xmm1, 1",
        "clflushopt [rax]",
        "clwb [rax]",
        "vaesenc ymm0, ymm1, ymm2",
        "vpclmulqdq ymm0, ymm1, ymm2, 0",
    ],
)
def test_unsupported_machine_code_is_refused(tmp_path: Path, instruction: str) -> None:
    """Refuse real AVX/BMI/opmask code without executing unsupported instructions."""
    result = _scan(tmp_path, instruction)
    assert result.returncode == 1
    opcode = instruction.split()[0]
    # GNU objdump names the immediate-selected PCLMUL operation explicitly.
    assert ("vpclmul" if opcode == "vpclmulqdq" else opcode) in result.stderr
    assert "PASS:" not in result.stdout
    assert (tmp_path / "scan/unsupported.txt").read_text()


def test_missing_empty_and_invalid_objects_are_refused(tmp_path: Path) -> None:
    """A missing inventory or failed disassembler can never produce an ISA PASS."""
    objects = tmp_path / "objects"
    for case in ("missing", "empty", "invalid"):
        if case == "empty":
            objects.mkdir()
        if case == "invalid":
            (objects / "corrupt.o").write_text("not an object\n")
        result = subprocess.run(
            ["bash", str(CHECKER), str(objects), str(tmp_path / case)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0
        assert "PASS:" not in result.stdout
