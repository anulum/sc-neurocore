#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cosimulation of sc_shd_event_p against the integer reference
"""Cosimulate ``hdl/shd_event/sc_shd_event_p.v`` against ``tools/shd_int_reference.py``.

Writes the five hex weight/delay tables and a hex stimulus stream, builds and runs
``hdl/shd_event/tb_sc_shd_event_p.v`` over the selected target filelist (``generic`` or
``xilinx_7series``) with Icarus Verilog or Verilator, and compares all 20 class accumulators
of every accepted sample against the integer reference. Refused input lengths (0 or above
1024) must return ``REFUSED 1``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import numpy.typing as npt

from tools import shd_int_reference as ref

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = REPO_ROOT / "hdl" / "shd_event"
BENCH = EVENT_DIR / "tb_sc_shd_event_p.v"
SIMULATION_TARGETS = ("generic", "xilinx_7series")
VALID_LANE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)
EVENT_COUNTER_NAMES = (
    "cycles",
    "l1_rows",
    "l2_rows",
    "l3_rows",
    "mem_rw",
    "leak_it",
    "w1_reads",
    "w1_writes",
    "w2_reads",
    "w2_writes",
    "q1_reads",
    "q2_reads",
    "q3_reads",
    "pushes",
    "pops",
    "overflow",
)

SampleInput = npt.NDArray[np.int64] | int


@dataclass(frozen=True)
class EventSampleResult:
    """Pair one sample's RTL accumulators and architectural counters with the reference.

    Attributes
    ----------
    rtl : list of int or None
        Twenty signed 32-bit class accumulators emitted by ``sc_shd_event_p``, or ``None``
        when the sample length was refused.
    reference : list of int or None
        Twenty reference accumulators computed by ``tools.shd_int_reference.run``, or
        ``None`` when the sample length was refused.
    counters : dict of str to int or None
        Sixteen simulation counters emitted after ``|`` by ``tb_sc_shd_event_p``, or
        ``None`` when the sample length was refused.
    refused : bool
        Whether the core raised ``error`` and refused the sample length.
    """

    rtl: list[int] | None
    reference: list[int] | None
    counters: dict[str, int] | None
    refused: bool

    @property
    def matches(self) -> bool:
        """Return whether the RTL accumulators equal the reference accumulators."""
        return self.rtl == self.reference

    @property
    def overflow(self) -> bool:
        """Return whether the core raised sticky bucket ``overflow`` on this sample."""
        return bool(self.counters is not None and self.counters["overflow"] != 0)


def _hex_bytes(values: npt.NDArray[np.int64]) -> str:
    """Format int8 values with lane 0 in the lowest byte, printed most significant first."""
    return "".join(f"{int(v) & 0xFF:02x}" for v in values[::-1])


def write_network(net: ref.IntNetwork, out_dir: Path) -> None:
    """Write the five hex weight and delay tables loaded by ``tb_sc_shd_event_p``.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network whose weights and delays are serialised.
    out_dir : Path
        Destination directory receiving ``q1.hex``, ``q2.hex``, ``q3.hex``, ``d1.hex``
        and ``d2.hex``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    q1 = np.asarray(net.q1, dtype=np.int64)
    q2 = np.asarray(net.q2, dtype=np.int64)
    q3 = np.asarray(net.q3, dtype=np.int64)
    (out_dir / "q1.hex").write_text(
        "".join(_hex_bytes(q1[:, i]) + "\n" for i in range(ref.N_INPUT)),
        encoding="utf-8",
    )
    (out_dir / "q2.hex").write_text(
        "".join(_hex_bytes(q2[:, j]) + "\n" for j in range(ref.N_HIDDEN)),
        encoding="utf-8",
    )
    (out_dir / "q3.hex").write_text(
        "".join(_hex_bytes(q3[:, j]) + "\n" for j in range(ref.N_HIDDEN)),
        encoding="utf-8",
    )
    (out_dir / "d1.hex").write_text(
        "".join(f"{int(x):02x}\n" for x in net.d1.tolist()),
        encoding="utf-8",
    )
    (out_dir / "d2.hex").write_text(
        "".join(f"{int(x):02x}\n" for x in net.d2.tolist()),
        encoding="utf-8",
    )


def write_stimulus(samples: Sequence[SampleInput], path: Path) -> int:
    """Write ``stimulus.hex`` for ``tb_sc_shd_event_p`` and return the word count.

    Parameters
    ----------
    samples : sequence of (ndarray or int)
        Accepted ``(T, 140)`` integer count matrices in ``0..15``, or bare integer lengths
        representing refused ``t_orig`` requests with no frame payload.
    path : Path
        Output file path for ``stimulus.hex``.

    Returns
    -------
    int
        Total number of 560-bit hex words written, sizing ``MAX_WORDS`` in the testbench.

    Raises
    ------
    ValueError
        If a frame array does not have shape ``(T, 140)`` or has counts outside ``0..15``.
    """
    lines = [f"{len(samples):x}"]
    for sample in samples:
        if isinstance(sample, int):
            lines.append(f"{sample & 0xFFFF:x}")
            continue
        counts = np.asarray(sample, dtype=np.int64)
        if (
            counts.ndim != 2
            or counts.shape[1] != ref.N_INPUT
            or int(counts.min(initial=0)) < 0
            or int(counts.max(initial=0)) > ref.COUNT_MAX
        ):
            raise ValueError("stimulus frames must be (T, 140) counts in 0..15")
        lines.append(f"{counts.shape[0]:x}")
        lines.extend("".join(f"{int(v):x}" for v in row[::-1]) for row in counts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def parameters(
    net: ref.IntNetwork,
    *,
    lanes: int = 8,
    cap1: int = 62,
    cap2: int = 39,
    wshape: int = 6,
    qshape: int = 12,
) -> dict[str, int]:
    """Build the Verilog parameter dictionary for ``tb_sc_shd_event_p``.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network supplying fraction bits, thresholds and widths.
    lanes : int, default 8
        Parallel LIF lanes ``P`` dividing 128 hidden neurons.
    cap1 : int, default 62
        Layer-1 time-wheel bucket capacity ``CAP1``.
    cap2 : int, default 39
        Layer-2 time-wheel bucket capacity ``CAP2``.
    wshape : int, default 6
        1P SRAM macro shape index ``WSHAPE`` in ``0..19`` for ``q1``, ``q2`` and ``q3``.
    qshape : int, default 12
        1P SRAM macro shape index ``QSHAPE`` in ``0..19`` for ``w1`` and ``w2``.

    Returns
    -------
    dict of str to int
        Parameter assignments passed to Icarus Verilog or Verilator.

    Raises
    ------
    ValueError
        If ``lanes`` does not divide 128, capacities are non-positive, or shape indices lie
        outside ``0..19``.
    """
    if lanes not in VALID_LANE_COUNTS:
        raise ValueError(f"lanes must be one of {VALID_LANE_COUNTS}, got {lanes}")
    if cap1 < 1 or cap2 < 1:
        raise ValueError("cap1 and cap2 must be positive")
    if not (0 <= wshape <= 19 and 0 <= qshape <= 19):
        raise ValueError("wshape and qshape must lie in 0..19")
    return {
        "FRAC_BITS": net.frac_bits,
        "THETA1": net.theta1,
        "THETA2": net.theta2,
        "V1_WIDTH": net.membrane_width(1),
        "V2_WIDTH": net.membrane_width(2),
        "WSHAPE": wshape,
        "QSHAPE": qshape,
        "CAP1": cap1,
        "CAP2": cap2,
        "P": lanes,
    }


def _filelist_path(target: str) -> Path:
    if target not in SIMULATION_TARGETS:
        raise ValueError(f"target must be one of {SIMULATION_TARGETS}, got {target!r}")
    return EVENT_DIR / f"{target}.f"


def _build(
    work: Path,
    params: dict[str, int],
    *,
    target: str,
    verilator: bool,
    timeout_seconds: float,
) -> list[str]:
    filelist = _filelist_path(target)
    if verilator:
        verilator_bin = shutil.which("verilator")
        if verilator_bin is None:
            raise RuntimeError("Verilator cosimulation requires verilator on PATH")
        flags = [f"-G{k}={v}" for k, v in params.items()]
        subprocess.run(
            [
                verilator_bin,
                "--binary",
                "--timing",
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                "--top-module",
                "tb_sc_shd_event_p",
                "-Mdir",
                str(work / "obj"),
                *flags,
                "-f",
                str(filelist.relative_to(REPO_ROOT)),
                str(BENCH.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return [str(work / "obj" / "Vtb_sc_shd_event_p")]
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise RuntimeError(
            "the SHD event-core cosimulation requires Icarus Verilog (iverilog and vvp)"
        )
    flags = [f"-Ptb_sc_shd_event_p.{k}={v}" for k, v in params.items()]
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-c",
            str(filelist.relative_to(REPO_ROOT)),
            "-o",
            str(work / "sim.vvp"),
            *flags,
            str(BENCH.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return [vvp, "-n", str(work / "sim.vvp")]


def _run_in_workdir(
    net: ref.IntNetwork,
    samples: Sequence[SampleInput],
    work: Path,
    *,
    target: str,
    lanes: int,
    cap1: int,
    cap2: int,
    wshape: int,
    qshape: int,
    verilator: bool,
    timeout_seconds: float,
) -> list[str]:
    work.mkdir(parents=True, exist_ok=True)
    write_network(net, work)
    words = write_stimulus(samples, work / "stimulus.hex")
    params = dict(
        parameters(net, lanes=lanes, cap1=cap1, cap2=cap2, wshape=wshape, qshape=qshape),
        MAX_WORDS=words + 1,
    )
    command = _build(
        work,
        params,
        target=target,
        verilator=verilator,
        timeout_seconds=timeout_seconds,
    )
    subprocess.run(
        command,
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return (work / "outputs.txt").read_text(encoding="utf-8").splitlines()


def simulate(
    net: ref.IntNetwork,
    samples: Sequence[SampleInput],
    *,
    timeout_seconds: float,
    target: str = "generic",
    lanes: int = 8,
    cap1: int = 62,
    cap2: int = 39,
    wshape: int = 6,
    qshape: int = 12,
    verilator: bool = False,
    work: Path | None = None,
) -> list[EventSampleResult]:
    """Cosimulate ``sc_shd_event_p`` on every sample and pair outputs with the reference.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network loaded through the idle ``load_*`` port.
    samples : sequence of (ndarray or int)
        Accepted ``(T, 140)`` count matrices or bare integer refused lengths.
    timeout_seconds : float
        Positive deadline in seconds applied separately to compilation and simulation.
    target : {"generic", "xilinx_7series"}, default "generic"
        Target filelist under ``hdl/shd_event/`` compiled into the simulation.
    lanes : int, default 8
        Parallel LIF lanes ``P``.
    cap1 : int, default 62
        Layer-1 time-wheel bucket capacity ``CAP1``.
    cap2 : int, default 39
        Layer-2 time-wheel bucket capacity ``CAP2``.
    wshape : int, default 6
        Weight-table macro shape index ``WSHAPE``.
    qshape : int, default 12
        Wheel-memory macro shape index ``QSHAPE``.
    verilator : bool, default False
        Use Verilator ``--binary --timing`` instead of Icarus Verilog when ``True``.
    work : Path, optional
        Explicit working directory; a temporary directory is created when omitted.

    Returns
    -------
    list of EventSampleResult
        One result per entry of ``samples``.

    Raises
    ------
    ValueError
        If ``timeout_seconds`` is not positive or any core parameter is out of range.
    RuntimeError
        If the required simulator is absent or the output line count disagrees with
        ``samples``.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if work is None:
        with tempfile.TemporaryDirectory(prefix="shd-event-cosim-") as tmp:
            lines = _run_in_workdir(
                net,
                samples,
                Path(tmp),
                target=target,
                lanes=lanes,
                cap1=cap1,
                cap2=cap2,
                wshape=wshape,
                qshape=qshape,
                verilator=verilator,
                timeout_seconds=timeout_seconds,
            )
    else:
        lines = _run_in_workdir(
            net,
            samples,
            work,
            target=target,
            lanes=lanes,
            cap1=cap1,
            cap2=cap2,
            wshape=wshape,
            qshape=qshape,
            verilator=verilator,
            timeout_seconds=timeout_seconds,
        )
    if len(lines) != len(samples):
        raise RuntimeError(f"testbench wrote {len(lines)} results for {len(samples)} samples")
    results: list[EventSampleResult] = []
    for sample, line in zip(samples, lines, strict=True):
        if isinstance(sample, int):
            results.append(EventSampleResult(None, None, None, refused=line.strip() == "REFUSED 1"))
            continue
        values_str, counters_str = line.split("|")
        counter_values = [int(x) for x in counters_str.split()]
        counters = dict(zip(EVENT_COUNTER_NAMES, counter_values, strict=True))
        ref_out = [int(x) for x in ref.run(net, sample).tolist()]
        results.append(
            EventSampleResult(
                [int(x) for x in values_str.split()],
                ref_out,
                counters,
                refused=False,
            )
        )
    return results


def _load_frames(path: Path) -> tuple[list[npt.NDArray[np.int64]], npt.NDArray[np.int64]]:
    with np.load(path) as data:
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        raw_frames = np.asarray(data["frames"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    frames = [raw_frames[offsets[i] : offsets[i + 1]] for i in range(len(lengths))]
    return frames, labels


def main(argv: Sequence[str] | None = None) -> int:
    """Cosimulate every sample of an ``.npz`` bundle on ``sc_shd_event_p``.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` is used when omitted.

    Returns
    -------
    int
        0 when every sample matches the integer reference with zero bucket overflows, 1
        otherwise.
    """
    description = (__doc__ or "").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("network", type=Path, help="integer network JSON (IntNetwork.to_json)")
    parser.add_argument("frames", type=Path, help="npz with frames, lengths and labels")
    parser.add_argument("--target", choices=SIMULATION_TARGETS, default="generic")
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--cap1", type=int, default=62)
    parser.add_argument("--cap2", type=int, default=39)
    parser.add_argument("--wshape", type=int, default=6)
    parser.add_argument("--qshape", type=int, default=12)
    parser.add_argument("--verilator", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        required=True,
        help="deadline in seconds for compilation and for simulation",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    net = ref.IntNetwork.from_json(args.network.read_text(encoding="utf-8"))
    frames, labels = _load_frames(args.frames)
    results = simulate(
        net,
        frames,
        timeout_seconds=args.timeout_seconds,
        target=args.target,
        lanes=args.lanes,
        cap1=args.cap1,
        cap2=args.cap2,
        wshape=args.wshape,
        qshape=args.qshape,
        verilator=args.verilator,
    )
    mismatches = [i for i, r in enumerate(results) if not r.matches]
    overflows = [i for i, r in enumerate(results) if r.overflow]
    correct = sum(
        int(np.argmax(r.rtl)) == int(y)
        for r, y in zip(results, labels.tolist(), strict=True)
        if r.rtl is not None
    )
    cycles_total = sum(r.counters["cycles"] for r in results if r.counters is not None)
    print(
        json.dumps(
            {
                "samples": len(results),
                "target": args.target,
                "lanes": args.lanes,
                "accumulator_mismatches": len(mismatches),
                "overflow_samples": len(overflows),
                "first_mismatches": mismatches[:5],
                "rtl_correct": correct,
                "cycles_total": cycles_total,
            }
        )
    )
    return 1 if (mismatches or overflows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
