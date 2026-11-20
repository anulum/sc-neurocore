#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI4-Lite cosimulation of sc_shd_event_axi against the integer reference
"""Cosimulate ``hdl/shd_event/sc_shd_event_axi.v`` against ``tools/shd_int_reference.py``.

Serialises the five hex weight/delay tables, computes the exact reflected IEEE CRC-32
(``zlib.crc32``) over the little-endian 64-bit group-word load stream, drives
``hdl/shd_event/tb_sc_shd_event_axi.v`` across the ``generic`` or ``xilinx_7series`` target
filelist in Icarus Verilog or Verilator, and verifies bit-exact class accumulators as well
as the negative AXI4-Lite / loader protocol suite.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib

import numpy as np

from tools import cosim_shd_event_core as core_cosim
from tools import shd_int_reference as ref

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = REPO_ROOT / "hdl" / "shd_event"
AXI_RTL = EVENT_DIR / "sc_shd_event_axi.v"
AXI_BENCH = EVENT_DIR / "tb_sc_shd_event_axi.v"
SIMULATION_TARGETS = core_cosim.SIMULATION_TARGETS

SampleInput = core_cosim.SampleInput


@dataclass(frozen=True)
class AxiSampleResult:
    """Pair one AXI-streamed sample's RTL accumulators with the integer reference.

    Attributes
    ----------
    rtl : list of int or None
        Twenty signed 32-bit class accumulators read from ``OUT_V_0..19``, or ``None``
        when ``CTRL.start`` refused the sample length with ``SLVERR``.
    reference : list of int or None
        Twenty reference accumulators computed by ``tools.shd_int_reference.run``, or
        ``None`` when the sample length was refused.
    refused : bool
        Whether ``CTRL.start`` refused the sample length with ``SLVERR``.
    """

    rtl: list[int] | None
    reference: list[int] | None
    refused: bool

    @property
    def matches(self) -> bool:
        """Return whether the AXI RTL accumulators equal the integer reference."""
        return self.rtl == self.reference


def pack_network_words(
    net: ref.IntNetwork,
    *,
    lanes: int = 8,
) -> list[tuple[int, int, int, int]]:
    """Pack ``net`` into ``(load_sel, load_addr, load_grp, word64)`` tuples.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network to serialise for the 64-bit AXI loader.
    lanes : int, default 8
        Parallel lane count ``P`` in ``1..8`` dividing 128 (up to 64 bits per word).

    Returns
    -------
    list of tuple of (int, int, int, int)
        Ordered ``(sel, addr, grp, word64)`` entries matching ``tb_sc_shd_event_p`` and
        ``sc_shd_event_axi``'s ``auto_increment`` geometry.

    Raises
    ------
    ValueError
        If ``lanes`` is not one of ``(1, 2, 4, 8)``.
    """
    if lanes not in (1, 2, 4, 8):
        raise ValueError(f"AXI 64-bit loader requires lanes in (1, 2, 4, 8), got {lanes}")
    q1 = np.asarray(net.q1, dtype=np.int64)
    q2 = np.asarray(net.q2, dtype=np.int64)
    q3 = np.asarray(net.q3, dtype=np.int64)
    d1 = np.asarray(net.d1, dtype=np.int64)
    d2 = np.asarray(net.d2, dtype=np.int64)

    words: list[tuple[int, int, int, int]] = []
    g_groups = ref.N_HIDDEN // lanes
    pc_passes = (ref.N_OUTPUT + lanes - 1) // lanes

    for r in range(ref.N_INPUT):
        for gq in range(g_groups):
            w64 = 0
            for k in range(lanes):
                w64 |= (int(q1[gq * lanes + k, r]) & 0xFF) << (8 * k)
            words.append((0, r, gq, w64))

    for r in range(ref.N_HIDDEN):
        for gq in range(g_groups):
            w64 = 0
            for k in range(lanes):
                w64 |= (int(q2[gq * lanes + k, r]) & 0xFF) << (8 * k)
            words.append((1, r, gq, w64))

    for r in range(ref.N_HIDDEN):
        for gq in range(pc_passes):
            w64 = 0
            for k in range(lanes):
                cls_idx = gq * lanes + k
                if cls_idx < ref.N_OUTPUT:
                    w64 |= (int(q3[cls_idx, r]) & 0xFF) << (8 * k)
            words.append((2, r, gq, w64))

    for r in range(ref.N_INPUT):
        words.append((3, r, 0, int(d1[r]) & 0x1F))

    for r in range(ref.N_HIDDEN):
        words.append((4, r, 0, int(d2[r]) & 0x1F))

    return words


def compute_words_crc32(words_u64: Sequence[int]) -> int:
    """Compute reflected IEEE CRC-32 over 64-bit words in little-endian byte order.

    Parameters
    ----------
    words_u64 : sequence of int
        Unsigned 64-bit group words in commit order.

    Returns
    -------
    int
        Unsigned 32-bit IEEE CRC-32 matching ``LOAD_CRC32`` in ``sc_shd_event_axi``.
    """
    payload = b"".join(struct.pack("<Q", w & 0xFFFF_FFFF_FFFF_FFFF) for w in words_u64)
    return zlib.crc32(payload) & 0xFFFF_FFFF


def network_crc32(net: ref.IntNetwork, *, lanes: int = 8) -> int:
    """Return the 32-bit IEEE CRC-32 of ``net``'s packed 64-bit load stream."""
    return compute_words_crc32([w for _sel, _addr, _grp, w in pack_network_words(net, lanes=lanes)])


def _build_axi(
    work: Path,
    params: dict[str, int],
    *,
    target: str,
    verilator: bool,
    timeout_seconds: float,
) -> list[str]:
    if target not in SIMULATION_TARGETS:
        raise ValueError(f"target must be one of {SIMULATION_TARGETS}, got {target!r}")
    filelist = EVENT_DIR / f"{target}.f"
    if verilator:
        verilator_bin = shutil.which("verilator")
        if verilator_bin is None:
            raise RuntimeError("Verilator AXI cosimulation requires verilator on PATH")
        flags = [f"-G{k}={v}" for k, v in params.items()]
        subprocess.run(
            [
                verilator_bin,
                "--binary",
                "--timing",
                "-j",
                "4",
                "-Wno-fatal",
                "-Wno-lint",
                "-Wno-style",
                "--top-module",
                "tb_sc_shd_event_axi",
                "-Mdir",
                str(work / "obj"),
                *flags,
                "-f",
                str(filelist.relative_to(REPO_ROOT)),
                str(AXI_RTL.relative_to(REPO_ROOT)),
                str(AXI_BENCH.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return [str(work / "obj" / "Vtb_sc_shd_event_axi")]
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise RuntimeError(
            "the SHD event AXI cosimulation requires Icarus Verilog (iverilog and vvp)"
        )
    flags = [f"-Ptb_sc_shd_event_axi.{k}={v}" for k, v in params.items()]
    subprocess.run(
        [
            iverilog,
            "-g2012",
            "-c",
            str(filelist.relative_to(REPO_ROOT)),
            "-o",
            str(work / "sim_axi.vvp"),
            *flags,
            str(AXI_RTL.relative_to(REPO_ROOT)),
            str(AXI_BENCH.relative_to(REPO_ROOT)),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return [vvp, "-n", str(work / "sim_axi.vvp")]


def _execute_bench(
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
    binary_spike_mode: bool,
    run_negative_tests: bool,
    poke_reserved_scales: bool = False,
    verilator: bool,
    timeout_seconds: float,
) -> list[str]:
    work.mkdir(parents=True, exist_ok=True)
    core_cosim.write_network(net, work)
    crc = network_crc32(net, lanes=lanes)
    (work / "expected_crc32.hex").write_text(f"{crc:08x}\n", encoding="utf-8")
    words = core_cosim.write_stimulus(samples, work / "stimulus.hex") if samples else 2
    params = dict(
        core_cosim.parameters(net, lanes=lanes, cap1=cap1, cap2=cap2, wshape=wshape, qshape=qshape),
        MAX_WORDS=max(words + 1, 4),
        BINARY_SPIKE_MODE=1 if binary_spike_mode else 0,
        RUN_NEGATIVE_TESTS=1 if run_negative_tests else 0,
        POKE_RESERVED_SCALES=1 if poke_reserved_scales else 0,
    )
    command = _build_axi(
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


def simulate_axi(
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
    binary_spike_mode: bool = False,
    poke_reserved_scales: bool = False,
    verilator: bool = False,
    work: Path | None = None,
) -> list[AxiSampleResult]:
    """Cosimulate ``sc_shd_event_axi`` over AXI4-Lite and compare with the reference.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network loaded through the AXI4-Lite loader registers.
    samples : sequence of (ndarray or int)
        Accepted ``(T, 140)`` count matrices or bare integer refused lengths.
    timeout_seconds : float
        Positive deadline in seconds applied separately to compilation and simulation.
    target : {"generic", "xilinx_7series"}, default "generic"
        Target filelist under ``hdl/shd_event/``.
    lanes : int, default 8
        Parallel lane count ``P`` in ``(1, 2, 4, 8)``.
    cap1 : int, default 62
        Layer-1 time-wheel bucket capacity ``CAP1``.
    cap2 : int, default 39
        Layer-2 time-wheel bucket capacity ``CAP2``.
    wshape : int, default 6
        Weight-table macro shape index ``WSHAPE``.
    qshape : int, default 12
        Wheel-memory macro shape index ``QSHAPE``.
    binary_spike_mode : bool, default False
        When ``True``, drive binary ``0/1`` frames via ``SPIKE_IN_0..4`` and load with
        ``auto_increment=0``; otherwise drive counts via ``FRAME_COUNT_0..17`` with
        ``auto_increment=1``.
    poke_reserved_scales : bool, default False
        When ``True``, attempt writes to the reserved scale offsets ``0x08..0x10``
        before running the sample set and verify ``SLVERR`` + readback ``0``.
    verilator : bool, default False
        Use Verilator ``--binary --timing`` instead of Icarus Verilog when ``True``.
    work : Path, optional
        Explicit working directory; a temporary directory is created when omitted.

    Returns
    -------
    list of AxiSampleResult
        One result per entry of ``samples``.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if binary_spike_mode:
        for sample in samples:
            if not isinstance(sample, int):
                arr = np.asarray(sample, dtype=np.int64)
                if int(arr.max(initial=0)) > 1 or int(arr.min(initial=0)) < 0:
                    raise ValueError("binary_spike_mode requires binary 0/1 frames")
    if work is None:
        with tempfile.TemporaryDirectory(prefix="shd-event-axi-cosim-") as tmp:
            lines = _execute_bench(
                net,
                samples,
                Path(tmp),
                target=target,
                lanes=lanes,
                cap1=cap1,
                cap2=cap2,
                wshape=wshape,
                qshape=qshape,
                binary_spike_mode=binary_spike_mode,
                run_negative_tests=False,
                poke_reserved_scales=poke_reserved_scales,
                verilator=verilator,
                timeout_seconds=timeout_seconds,
            )
    else:
        lines = _execute_bench(
            net,
            samples,
            work,
            target=target,
            lanes=lanes,
            cap1=cap1,
            cap2=cap2,
            wshape=wshape,
            qshape=qshape,
            binary_spike_mode=binary_spike_mode,
            run_negative_tests=False,
            poke_reserved_scales=poke_reserved_scales,
            verilator=verilator,
            timeout_seconds=timeout_seconds,
        )
    if len(lines) != len(samples):
        raise RuntimeError(f"AXI testbench wrote {len(lines)} results for {len(samples)} samples")
    results: list[AxiSampleResult] = []
    for sample, line in zip(samples, lines, strict=True):
        if isinstance(sample, int):
            results.append(AxiSampleResult(None, None, refused=line.strip() == "REFUSED 1"))
            continue
        ref_out = [int(x) for x in ref.run(net, sample).tolist()]
        results.append(AxiSampleResult([int(x) for x in line.split()], ref_out, refused=False))
    return results


def run_negative_protocol_suite(
    net: ref.IntNetwork,
    *,
    timeout_seconds: float,
    target: str = "generic",
    verilator: bool = False,
    work: Path | None = None,
) -> bool:
    """Execute ``tb_sc_shd_event_axi``'s self-checking negative protocol suite.

    Parameters
    ----------
    net : ref.IntNetwork
        Validated integer SHD network used for post-refusal valid load checks.
    timeout_seconds : float
        Positive deadline in seconds applied separately to compilation and simulation.
    target : {"generic", "xilinx_7series"}, default "generic"
        Target filelist under ``hdl/shd_event/``.
    verilator : bool, default False
        Use Verilator ``--binary --timing`` instead of Icarus Verilog when ``True``.
    work : Path, optional
        Explicit working directory; a temporary directory is created when omitted.

    Returns
    -------
    bool
        ``True`` when the RTL negative protocol suite completes and emits
        ``PASS_NEGATIVE_SUITE``.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if work is None:
        with tempfile.TemporaryDirectory(prefix="shd-event-axi-neg-") as tmp:
            lines = _execute_bench(
                net,
                [],
                Path(tmp),
                target=target,
                lanes=8,
                cap1=62,
                cap2=39,
                wshape=6,
                qshape=12,
                binary_spike_mode=False,
                run_negative_tests=True,
                verilator=verilator,
                timeout_seconds=timeout_seconds,
            )
    else:
        lines = _execute_bench(
            net,
            [],
            work,
            target=target,
            lanes=8,
            cap1=62,
            cap2=39,
            wshape=6,
            qshape=12,
            binary_spike_mode=False,
            run_negative_tests=True,
            verilator=verilator,
            timeout_seconds=timeout_seconds,
        )
    return lines == ["PASS_NEGATIVE_SUITE"]


def main(argv: Sequence[str] | None = None) -> int:
    """Cosimulate every sample of an ``.npz`` bundle through ``sc_shd_event_axi``.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` is used when omitted.

    Returns
    -------
    int
        0 when every sample matches the integer reference, 1 otherwise.
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
    parser.add_argument("--binary-spikes", action="store_true")
    parser.add_argument("--verilator", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        required=True,
        help="deadline in seconds for compilation and for simulation",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    net = ref.IntNetwork.from_json(args.network.read_text(encoding="utf-8"))
    frames, labels = core_cosim._load_frames(args.frames)
    results = simulate_axi(
        net,
        frames,
        timeout_seconds=args.timeout_seconds,
        target=args.target,
        lanes=args.lanes,
        cap1=args.cap1,
        cap2=args.cap2,
        wshape=args.wshape,
        qshape=args.qshape,
        binary_spike_mode=args.binary_spikes,
        verilator=args.verilator,
    )
    mismatches = [i for i, r in enumerate(results) if not r.matches]
    correct = sum(
        int(np.argmax(r.rtl)) == int(y)
        for r, y in zip(results, labels.tolist(), strict=True)
        if r.rtl is not None
    )
    print(
        json.dumps(
            {
                "samples": len(results),
                "target": args.target,
                "lanes": args.lanes,
                "accumulator_mismatches": len(mismatches),
                "first_mismatches": mismatches[:5],
                "rtl_correct": correct,
            }
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
