# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verification of Golden Reference #0 (sc_shd_event_p) and its target leaves

"""Verify the ``hdl/shd_event/`` RTL tree, synthesis report, and cosimulation driver.

Checks the 15-file inventory, pairwise-unique digests, absence of target preprocessor
conditionals, pinned comment-stripped digests of Golden Reference #0, kept Yosys
``synth_xilinx`` block-RAM inference report, and bit-exact cosimulation across the
``generic`` and ``xilinx_7series`` filelists in both Icarus Verilog and Verilator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys

import numpy as np
import numpy.typing as npt
import pytest

from tools import cosim_shd_event_core as cosim
from tools import shd_int_reference as ref
from tools.benchmark_evidence_gate import tracked_files

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = REPO_ROOT / "hdl" / "shd_event"
#: The license tag every kept file carries, joined at run time so that REUSE does not read this
#: test's own source as declaring a license expression.
SPDX_TAG = "SPDX-License-" + "Identifier: AGPL-3.0-or-later"
FIXTURE_NETWORK = REPO_ROOT / "hdl" / "fixtures" / "shd_int_network.json"
SYNTH_XILINX_REPORT = REPO_ROOT / "hdl" / "reports" / "sc_shd_event_p_xc7z020_synth_xilinx.txt"
SIM_TIMEOUT_S = 900.0

EXPECTED_EVENT_FILES = frozenset(
    {
        "generic.f",
        "ihp_sg13g2.f",
        "sc_cg.v",
        "sc_lanes.v",
        "sc_shapes.vh",
        "sc_shd_event_axi.v",
        "sc_shd_event_p.v",
        "tb_sc_shd_event_axi.v",
        "tb_sc_shd_event_p.v",
        "tech/generic/cells_clkgate.v",
        "tech/generic/sc_mem.v",
        "tech/generic/sc_sram_macro.v",
        "tech/ihp_sg13g2/cells_clkgate.v",
        "tech/ihp_sg13g2/macro_stubs.v",
        "tech/ihp_sg13g2/sc_sram_macro.v",
        "tech/xilinx_7series/sc_mem.v",
        "xilinx_7series.f",
    }
)

PINNED_COMMENT_STRIPPED_SHA256 = {
    "sc_shd_event_p.v": "f18e123d7d94602e46181c209fc0617169766f967ba7269944795ee2388df8c3",
    "sc_lanes.v": "6c9097934fb1b1fea9f374d0e94b3bfefa371efbe5c82a07ef748f4956007934",
    "sc_cg.v": "585bd9e647b87ce91bb55c97bb0d85b58c9ea7f229b44c4c084f125164bf4eb0",
    "sc_shapes.vh": "bd9c24fc0839d602fc3441e342ed0b919bdf3f8c7bb3ffa2c7c3c2c9147c08d2",
    "tech/generic/sc_mem.v": "c29fe5aa259fddf5fdf176d033817336b3243120c0a91fb862ffe3af306ed70c",
    "tech/ihp_sg13g2/sc_sram_macro.v": (
        "c25bf724519249aa159506f287f6a32eb5cf5ff077dfbba48a29354b060a888e"
    ),
    "tech/ihp_sg13g2/macro_stubs.v": (
        "c77d147968bc887f09cd27a425e65a4f4aa9bf929e631f34b8ff6054b4c00026"
    ),
}


def _trained_net() -> ref.IntNetwork:
    return ref.IntNetwork.from_json(FIXTURE_NETWORK.read_text(encoding="utf-8"))


def _random_net(
    seed: int,
    *,
    frac_bits: int | None = None,
    d1: npt.NDArray[np.int64] | None = None,
    d2: npt.NDArray[np.int64] | None = None,
) -> ref.IntNetwork:
    rng = np.random.default_rng(seed)
    return ref.IntNetwork(
        frac_bits=int(rng.integers(0, 4)) if frac_bits is None else frac_bits,
        d1=rng.integers(0, 31, ref.N_INPUT, dtype=np.int64) if d1 is None else d1,
        d2=rng.integers(0, 31, ref.N_HIDDEN, dtype=np.int64) if d2 is None else d2,
        q1=rng.integers(-127, 128, (128, 140), dtype=np.int64),
        q2=rng.integers(-127, 128, (128, 128), dtype=np.int64),
        q3=rng.integers(-127, 128, (20, 128), dtype=np.int64),
        theta1=500,
        theta2=250,
    )


def _sparse_frames(seed: int, steps: int) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 10, (steps, ref.N_INPUT)) == 0
    counts = rng.integers(1, 8, (steps, ref.N_INPUT), dtype=np.int64)
    return np.where(mask, counts, 0).astype(np.int64)


def test_event_tree_inventory_headers_and_unique_digests() -> None:
    """Every expected file exists, carries the 7-line SPDX header, and has a unique digest."""
    tracked = tracked_files(REPO_ROOT, "hdl/shd_event")
    if tracked is None:  # an unpacked source distribution has no Git work tree
        files = [p for p in EVENT_DIR.rglob("*") if p.is_file()]
    else:  # only Git's files: ignored simulation or synthesis output is not part of the tree
        files = [REPO_ROOT / name for name in tracked]
    actual = {p.relative_to(EVENT_DIR).as_posix() for p in files}
    assert actual == EXPECTED_EVENT_FILES

    digests: dict[str, str] = {}
    for rel in sorted(actual):
        path = EVENT_DIR / rel
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:7])
        assert SPDX_TAG in head, rel
        assert "ORCID: 0009-0009-3560-0851" in head, rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in digests, f"{rel} duplicates {digests[digest]}"
        digests[digest] = rel


def test_zero_target_preprocessor_conditionals_and_zero_branch_testbench() -> None:
    """Core and leaves carry no target ifdef; the derived testbench has zero branches."""
    directive_re = re.compile(r"^\s*`(ifdef|ifndef|elsif|else|endif)\b(.*)$")
    for path in sorted(EVENT_DIR.rglob("*.v*")):
        rel = path.relative_to(EVENT_DIR).as_posix()
        directives: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = directive_re.match(line)
            if match is not None:
                directives.append((match.group(1), match.group(2).strip()))
        if rel in {"tb_sc_shd_event_p.v", "tb_sc_shd_event_axi.v"}:
            assert directives == []
            continue
        for kind, rest in directives:
            assert kind in {"ifndef", "endif"}, f"{rel}: forbidden `{kind} {rest}"
            if kind == "ifndef":
                assert rest.split()[0] == "SYNTHESIS", f"{rel}: forbidden `ifndef {rest}"


def test_pinned_golden_reference_comment_stripped_digests() -> None:
    """Imported Golden Reference #0 RTL bodies match their pinned comment-stripped digests."""
    ev_lines = (EVENT_DIR / "sc_shd_event_p.v").read_text(encoding="utf-8").splitlines()
    assert len(ev_lines) == 345
    assert ev_lines[23] == "`timescale 1ns / 1ps"

    for rel, expected_sha in PINNED_COMMENT_STRIPPED_SHA256.items():
        lines = (EVENT_DIR / rel).read_text(encoding="utf-8").splitlines()
        for line in lines:
            if '"' in line:
                assert "//" not in line.split('"')[1], rel
        first_code = next(
            idx
            for idx, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("//")
        )
        stripped = (
            "\n".join(re.sub(r"//.*", "", line).rstrip() for line in lines[first_code:]) + "\n"
        ).encode("utf-8")
        assert hashlib.sha256(stripped).hexdigest() == expected_sha, rel


def test_target_filelists_resolve_and_lint_in_both_synthesis_modes(tmp_path: Path) -> None:
    """All three target filelists resolve existing paths and pass Verilator lint twice."""
    parsed_entries: dict[str, list[str]] = {}
    for name in ("generic.f", "ihp_sg13g2.f", "xilinx_7series.f"):
        entries = [
            line.strip()
            for line in (EVENT_DIR / name).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        parsed_entries[name] = entries
        for entry in entries:
            if entry.startswith("+incdir+"):
                assert (REPO_ROOT / entry.removeprefix("+incdir+")).is_dir()
            else:
                assert (REPO_ROOT / entry).is_file()

    assert "hdl/shd_event/tech/generic/sc_mem.v" in parsed_entries["ihp_sg13g2.f"]
    assert "hdl/shd_event/tech/generic/cells_clkgate.v" in parsed_entries["xilinx_7series.f"]

    ihp_stub = tmp_path / "ihp_lgcp_stub.v"
    ihp_stub.write_text(
        "(* blackbox *)\nmodule sg13g2_lgcp_1 (output GCLK, input GATE, input CLK);\nendmodule\n",
        encoding="utf-8",
    )
    verilator_bin = shutil.which("verilator")
    assert verilator_bin is not None
    for name in ("generic.f", "ihp_sg13g2.f", "xilinx_7series.f"):
        for synth_flag in ((), ("-DSYNTHESIS",)):
            extra = (str(ihp_stub),) if name == "ihp_sg13g2.f" else ()
            completed = subprocess.run(
                [
                    verilator_bin,
                    "--lint-only",
                    "-Wno-TIMESCALEMOD",
                    "-Wno-WIDTHEXPAND",
                    "-Wno-WIDTHTRUNC",
                    "-Wno-INITIALDLY",
                    "-Wno-BLKSEQ",
                    *synth_flag,
                    "-f",
                    f"hdl/shd_event/{name}",
                    *extra,
                    "--top-module",
                    "sc_shd_event_p",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=120.0,
            )
            assert completed.returncode == 0, completed.stderr


def _yosys_cell_count(section: str, cell_regex: str) -> int:
    """Sum Yosys ``stat`` cell counts across either ``count CELL`` or ``CELL count`` rows."""
    pattern = rf"(?:(\d+)\s+(?:{cell_regex})\b|\b(?:{cell_regex})\s+(\d+))"
    return sum(int(a or b) for a, b in re.findall(pattern, section))


def test_kept_synth_xilinx_report_infers_block_ram_for_every_sc_mem() -> None:
    """The kept xc7z020 synth_xilinx report infers RAMB36E1/RAMB18E1 for every sc_mem."""
    assert SYNTH_XILINX_REPORT.is_file()
    report_text = SYNTH_XILINX_REPORT.read_text(encoding="utf-8")
    assert SPDX_TAG in "\n".join(report_text.splitlines()[:7])
    sc_mem_blocks = re.findall(
        r"(===\s+\$paramod[^=]*\\sc_mem[^=]*===.*?)(?=\n===|\Z)",
        report_text,
        flags=re.DOTALL,
    )
    assert len(sc_mem_blocks) >= 5
    for block in sc_mem_blocks:
        ramb36 = _yosys_cell_count(block, "RAMB36E1")
        ramb18 = _yosys_cell_count(block, "RAMB18E1")
        flip_flops = _yosys_cell_count(block, "FD[A-Z]+")
        assert ramb36 + ramb18 > 0, block
        assert flip_flops < 1024, block

    flat_section = report_text[report_text.rfind("=== sc_shd_event_p ===") :]
    flat_ramb36 = _yosys_cell_count(flat_section, "RAMB36E1")
    flat_ramb18 = _yosys_cell_count(flat_section, "RAMB18E1")
    flat_bufg = _yosys_cell_count(flat_section, "BUFG")
    assert flat_ramb36 + flat_ramb18 > 0
    assert 1 <= flat_bufg <= 32


@pytest.mark.parametrize("target", ["generic", "xilinx_7series"])
@pytest.mark.parametrize("use_verilator", [False, True])
def test_trained_network_cosimulates_bit_exactly_across_targets_and_simulators(
    target: str,
    use_verilator: bool,
) -> None:
    """Both target filelists match the integer reference in Icarus Verilog and Verilator."""
    net = _trained_net()
    samples: list[cosim.SampleInput] = [
        _sparse_frames(2, 24),
        0,
        1025,
        _sparse_frames(3, 7),
    ]
    results = cosim.simulate(
        net,
        samples,
        timeout_seconds=SIM_TIMEOUT_S,
        target=target,
        verilator=use_verilator,
    )
    assert [r.refused for r in results] == [False, True, True, False]
    for idx in (0, 3):
        assert results[idx].matches
        assert not results[idx].overflow
        counters = results[idx].counters
        assert counters is not None
        assert counters["cycles"] > 0


def test_generic_and_xilinx_produce_identical_accumulators_and_counters(tmp_path: Path) -> None:
    """Generic and Xilinx 7-Series filelists agree on accumulators and all 16 counters."""
    net = _random_net(
        19,
        frac_bits=2,
        d1=np.arange(ref.N_INPUT, dtype=np.int64) % 31,
        d2=np.arange(ref.N_HIDDEN, dtype=np.int64) % 31,
    )
    probe_q1 = np.zeros((128, 140), dtype=np.int64)
    probe_q2 = np.zeros((128, 128), dtype=np.int64)
    probe_q3 = np.zeros((20, 128), dtype=np.int64)
    probe_q1[:, 0], probe_q1[:, 1] = -5, 10
    probe_q2[np.arange(128), np.arange(128)] = 127
    probe_q3[0, :] = 1
    leak_net = ref.IntNetwork(
        frac_bits=0,
        d1=np.zeros(ref.N_INPUT, dtype=np.int64),
        d2=np.zeros(ref.N_HIDDEN, dtype=np.int64),
        q1=probe_q1,
        q2=probe_q2,
        q3=probe_q3,
        theta1=7,
        theta2=1,
    )
    leak_frames = np.zeros((2, 140), dtype=np.int64)
    leak_frames[0, 0] = leak_frames[1, 1] = 1
    assert int(ref.run(leak_net, leak_frames)[0]) == 128

    gen_leak = cosim.simulate(
        leak_net,
        [leak_frames],
        timeout_seconds=SIM_TIMEOUT_S,
        target="generic",
        lanes=16,
        cap1=140,
        cap2=128,
        wshape=0,
        qshape=0,
        work=tmp_path / "gen_leak",
    )
    xil_leak = cosim.simulate(
        leak_net,
        [leak_frames],
        timeout_seconds=SIM_TIMEOUT_S,
        target="xilinx_7series",
        lanes=16,
        cap1=140,
        cap2=128,
        wshape=0,
        qshape=0,
        work=tmp_path / "xil_leak",
    )
    assert gen_leak[0].matches and xil_leak[0].matches
    assert gen_leak[0].counters == xil_leak[0].counters

    samples: list[cosim.SampleInput] = [_sparse_frames(21, 18), np.zeros((3, 140), dtype=np.int64)]
    gen_res = cosim.simulate(
        net, samples, timeout_seconds=SIM_TIMEOUT_S, target="generic", cap1=140, cap2=128
    )
    xil_res = cosim.simulate(
        net, samples, timeout_seconds=SIM_TIMEOUT_S, target="xilinx_7series", cap1=140, cap2=128
    )
    for gr, xr in zip(gen_res, xil_res, strict=True):
        assert gr.matches and xr.matches
        assert gr.rtl == xr.rtl
        assert gr.counters == xr.counters


def test_cli_main_and_bucket_overflow_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI returns 0 on clean bit-exact runs and 1 when a tiny bucket overflows."""
    net = _trained_net()
    net_path = tmp_path / "net.json"
    net_path.write_text(net.to_json() + "\n", encoding="utf-8")

    f0 = np.zeros((3, 140), dtype=np.int64)
    f0[0, :6] = 5
    npz_path = tmp_path / "bundle.npz"
    np.savez(
        npz_path,
        frames=f0,
        lengths=np.array([3], dtype=np.int64),
        labels=np.array([int(np.argmax(ref.run(net, f0)))], dtype=np.int64),
    )

    rc_ok = cosim.main(
        [str(net_path), str(npz_path), "--timeout-seconds", "300", "--target", "xilinx_7series"]
    )
    out_ok = json.loads(capsys.readouterr().out)
    assert rc_ok == 0
    assert out_ok["accumulator_mismatches"] == 0
    assert out_ok["overflow_samples"] == 0
    assert out_ok["rtl_correct"] == 1

    zero_delay_net = _random_net(
        9,
        d1=np.zeros(ref.N_INPUT, dtype=np.int64),
        d2=np.zeros(ref.N_HIDDEN, dtype=np.int64),
    )
    zd_path = tmp_path / "zd_net.json"
    zd_path.write_text(zero_delay_net.to_json() + "\n", encoding="utf-8")
    rc_ovf = cosim.main([str(zd_path), str(npz_path), "--cap1", "1", "--timeout-seconds", "300"])
    out_ovf = json.loads(capsys.readouterr().out)
    assert rc_ovf == 1
    assert out_ovf["overflow_samples"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["cosim_shd_event_core.py", str(net_path), str(npz_path), "--timeout-seconds", "300"],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(REPO_ROOT / "tools" / "cosim_shd_event_core.py"), run_name="__main__")
    assert exc_info.value.code == 0
    capsys.readouterr()


def test_driver_input_and_environment_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid parameters, stimulus shapes, missing simulators and truncated outputs fail closed."""
    net = _trained_net()
    valid_frame = np.zeros((1, 140), dtype=np.int64)

    with pytest.raises(ValueError, match="stimulus frames"):
        cosim.write_stimulus([np.full((2, 140), 16, dtype=np.int64)], tmp_path / "bad.hex")
    with pytest.raises(ValueError, match="lanes"):
        cosim.parameters(net, lanes=3)
    with pytest.raises(ValueError, match="positive"):
        cosim.parameters(net, cap1=0)
    with pytest.raises(ValueError, match="0..19"):
        cosim.parameters(net, wshape=20)
    with pytest.raises(ValueError, match="timeout_seconds"):
        cosim.simulate(net, [valid_frame], timeout_seconds=0.0)
    with pytest.raises(ValueError, match="target"):
        cosim.simulate(net, [valid_frame], timeout_seconds=10.0, target="unknown")

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="Icarus Verilog"):
        cosim.simulate(net, [valid_frame], timeout_seconds=10.0, verilator=False)
    with pytest.raises(RuntimeError, match="verilator"):
        cosim.simulate(net, [valid_frame], timeout_seconds=10.0, verilator=True)

    def _fake_build(
        work: Path,
        params: dict[str, int],
        *,
        target: str,
        verilator: bool,
        timeout_seconds: float,
    ) -> list[str]:
        del params, target, verilator, timeout_seconds
        (work / "outputs.txt").write_text("", encoding="utf-8")
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(cosim, "_build", _fake_build)
    with pytest.raises(RuntimeError, match="results for"):
        cosim.simulate(net, [valid_frame], timeout_seconds=10.0, work=tmp_path / "trunc")
