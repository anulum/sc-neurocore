# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Execution of generated bit-true kernels under a per-step drive

"""Run the maintained bit-true C kernel of an equation neuron from Studio.

The only fixed-point arithmetic Studio reports as *fixed-point* is the
arithmetic of :func:`~sc_neurocore.compiler.intelligence.bit_true_kernel.generate_bittrue_kernel_from_neuron`,
the kernel that is proven bit-identical to the generated RTL. This module
compiles that kernel with the system C compiler, drives it with one encoded
input word per step and returns the integer state words and spike of every
step. Nothing here rounds in floating point on the kernel's behalf: a value
that the word cannot hold is the caller's rejection, not a clamp.

The same native-tool helpers serve the selected-model RTL co-simulation
(:mod:`sc_neurocore.studio.model_cosim`).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    c_word_type,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.equation_builder import EquationNeuron

NATIVE_TOOL_NAMES: tuple[str, ...] = ("gcc", "iverilog", "vvp")
BIT_TRUE_TOOL_NAMES: tuple[str, ...] = ("gcc",)
MAX_BIT_TRUE_STEPS = 2_000_000


class NativeToolUnavailable(RuntimeError):
    """Raised when a required native tool is not installed on the host.

    Parameters
    ----------
    tools:
        Names of the missing tools.
    purpose:
        What the tools were needed for, without repository paths.
    """

    def __init__(self, tools: Sequence[str], purpose: str) -> None:
        self.tools = tuple(tools)
        self.purpose = purpose
        super().__init__(
            f"{purpose} needs native tools unavailable on this host: {', '.join(self.tools)}."
        )

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "error": "native_tool_unavailable",
            "tools": list(self.tools),
            "reason": str(self),
        }


class NativeExecutionError(RuntimeError):
    """Raised when a native compile or run fails or exceeds its time budget."""


def resolve_native_tool(name: str) -> str | None:
    """Return the absolute path of a supported native tool, or ``None``."""
    if name not in NATIVE_TOOL_NAMES:
        raise ValueError(f"Unsupported native tool {name!r}.")
    return shutil.which(name)


def require_native_tools(names: Sequence[str], *, purpose: str) -> dict[str, str]:
    """Resolve every tool in ``names`` or raise :class:`NativeToolUnavailable`."""
    resolved = {name: resolve_native_tool(name) for name in names}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        raise NativeToolUnavailable(missing, purpose)
    return {name: str(path) for name, path in resolved.items()}


def run_native_command(
    command: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` without a shell and raise a bounded error on failure."""
    try:
        completed = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            check=False,
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise NativeExecutionError(f"Native command failed: {Path(command[0]).name}.") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")[:500]
        raise NativeExecutionError(
            f"Native command {Path(command[0]).name!r} exited {completed.returncode}: {detail}"
        )
    return completed


def native_tool_version(path: str, name: str) -> str:
    """Return the first version line a tool prints, or a stable placeholder."""
    argument = "-V" if name in {"iverilog", "vvp"} else "--version"
    try:
        completed = run_native_command([path, argument], timeout_seconds=5)
    except NativeExecutionError:
        return "available-version-unreported"
    lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    return lines[0][:200] if lines else "available-version-unreported"


@dataclass(frozen=True, slots=True)
class BitTrueTrace:
    """Integer state words and spikes of one bit-true kernel run.

    Parameters
    ----------
    module_name:
        Name the kernel was generated under.
    variables:
        Equation variable names in declaration order (the word columns).
    words:
        ``(n_steps, len(variables))`` post-step state words.
    spikes:
        ``(n_steps,)`` spike flag of every step.
    drive_words:
        ``(n_steps,)`` input words the kernel consumed.
    fraction:
        Fractional bits of the word format (for :meth:`decoded`).
    kernel_sha256, harness_sha256:
        Digests of the generated kernel source and of the harness ``main``.
    compiler:
        First version line of the C compiler used.
    """

    module_name: str
    variables: tuple[str, ...]
    words: np.ndarray[Any, Any]
    spikes: np.ndarray[Any, Any]
    drive_words: np.ndarray[Any, Any]
    fraction: int
    kernel_sha256: str
    harness_sha256: str
    compiler: str

    @property
    def n_steps(self) -> int:
        """Number of executed steps."""
        return int(self.words.shape[0])

    def decoded(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return every state trace decoded to float64 (``word / 2**fraction``)."""
        scale = float(1 << self.fraction)
        return {
            name: self.words[:, index].astype(np.float64) / scale
            for index, name in enumerate(self.variables)
        }

    def spike_steps(self) -> list[int]:
        """Return the raw step indices at which the kernel spiked."""
        return [int(step) for step in np.flatnonzero(self.spikes)]


def harness_main(neuron: EquationNeuron, module_name: str, data_width: int) -> str:
    """Return the C ``main`` that streams input words in and state rows out.

    The harness reads little-endian ``int64`` input words from the file named
    by its first argument, calls ``<module>_step`` once per word and appends
    one binary row ``[spike, <state words…>]`` of ``int64`` to the file named
    by its second argument. Binary rows avoid any text formatting of the
    words on either side.
    """
    variables = [sanitize_ident(name, context="state variable") for name in neuron.equations]
    word_type = c_word_type(data_width)
    row_assignments = "".join(
        f"        row[{index + 1}] = (int64_t)st.{name}_out;\n"
        for index, name in enumerate(variables)
    )
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc != 3) { return 2; }\n"
        '    FILE *in = fopen(argv[1], "rb");\n'
        "    if (in == NULL) { return 3; }\n"
        '    FILE *out = fopen(argv[2], "wb");\n'
        "    if (out == NULL) { fclose(in); return 4; }\n"
        f"    {module_name}_state_t st;\n"
        f"    {module_name}_reset(&st);\n"
        "    int64_t word;\n"
        f"    int64_t row[{len(variables) + 1}];\n"
        "    while (fread(&word, sizeof word, 1, in) == 1) {\n"
        f"        int spike = {module_name}_step(&st, ({word_type})word);\n"
        "        row[0] = (int64_t)spike;\n"
        f"{row_assignments}"
        "        if (fwrite(row, sizeof row, 1, out) != 1) { fclose(in); fclose(out); return 5; }\n"
        "    }\n"
        "    fclose(in);\n"
        "    if (fclose(out) != 0) { return 6; }\n"
        "    return 0;\n"
        "}\n"
    )


def run_bittrue_kernel(
    neuron: EquationNeuron,
    *,
    data_width: int,
    fraction: int,
    overflow: str,
    rounding: str,
    drive_words: Sequence[int] | np.ndarray[Any, Any],
    module_name: str = "sc_studio_bittrue",
    timeout_seconds: float = 120.0,
) -> BitTrueTrace:
    """Compile the neuron's bit-true kernel and run it under ``drive_words``.

    Parameters
    ----------
    neuron:
        Equation neuron whose equations, parameters, threshold and reset
        rules are lowered (``method`` must be ``euler`` or ``map``).
    data_width, fraction:
        Fixed-point word geometry.
    overflow, rounding:
        Accumulate overflow and product rounding policies of the kernel.
    drive_words:
        One already-encoded signed input word per step. The caller is
        responsible for representability; a word outside the ``data_width``
        range is rejected here rather than wrapped.
    module_name:
        Identifier for the generated kernel.
    timeout_seconds:
        Budget for each of the compile and the run.

    Returns
    -------
    BitTrueTrace
        Words and spikes of every step.

    Raises
    ------
    NativeToolUnavailable
        When no C compiler is installed.
    NativeExecutionError
        When compilation or execution fails or times out.
    ValueError
        For an empty or oversized drive, a word outside the format, or a
        kernel configuration the generator rejects.
    """
    words = np.asarray(drive_words, dtype=np.int64)
    if words.ndim != 1 or words.size < 1:
        raise ValueError("bit-true drive must be a non-empty one-dimensional sequence of words")
    if words.size > MAX_BIT_TRUE_STEPS:
        raise ValueError(f"bit-true drive exceeds {MAX_BIT_TRUE_STEPS} steps")
    max_word = (1 << (data_width - 1)) - 1
    min_word = -(1 << (data_width - 1))
    if int(words.max()) > max_word or int(words.min()) < min_word:
        raise ValueError(
            f"bit-true drive word outside the {data_width}-bit range [{min_word}, {max_word}]"
        )
    tools = require_native_tools(BIT_TRUE_TOOL_NAMES, purpose="Bit-true fixed-point execution")
    kernel = generate_bittrue_kernel_from_neuron(
        neuron,
        module_name,
        data_width=data_width,
        fraction=fraction,
        overflow=overflow,
        rounding=rounding,
    )
    harness = harness_main(neuron, module_name, data_width)
    variables = tuple(neuron.equations)
    columns = len(variables) + 1

    with tempfile.TemporaryDirectory(prefix="sc_studio_bittrue_") as temp_dir:
        root = Path(temp_dir)
        source_path = root / "kernel.c"
        binary_path = root / "kernel"
        drive_path = root / "drive.bin"
        trace_path = root / "trace.bin"
        source_path.write_text(kernel + "\n" + harness, encoding="utf-8")
        words.astype("<i8").tofile(drive_path)
        run_native_command(
            [tools["gcc"], "-O2", "-std=c11", "-o", str(binary_path), str(source_path)],
            timeout_seconds=timeout_seconds,
        )
        run_native_command(
            [str(binary_path), str(drive_path), str(trace_path)],
            timeout_seconds=timeout_seconds,
        )
        raw = np.fromfile(trace_path, dtype="<i8")

    expected = words.size * columns
    if raw.size != expected:
        raise NativeExecutionError(f"bit-true kernel emitted {raw.size} words, expected {expected}")
    rows = raw.reshape(words.size, columns)
    return BitTrueTrace(
        module_name=module_name,
        variables=variables,
        words=rows[:, 1:].copy(),
        spikes=rows[:, 0].astype(np.int8),
        drive_words=words,
        fraction=fraction,
        kernel_sha256=hashlib.sha256(kernel.encode("utf-8")).hexdigest(),
        harness_sha256=hashlib.sha256(harness.encode("utf-8")).hexdigest(),
        compiler=native_tool_version(tools["gcc"], "gcc"),
    )


__all__ = [
    "BIT_TRUE_TOOL_NAMES",
    "MAX_BIT_TRUE_STEPS",
    "NATIVE_TOOL_NAMES",
    "BitTrueTrace",
    "NativeExecutionError",
    "NativeToolUnavailable",
    "harness_main",
    "native_tool_version",
    "require_native_tools",
    "resolve_native_tool",
    "run_bittrue_kernel",
    "run_native_command",
]
