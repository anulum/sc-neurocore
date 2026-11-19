# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike quantiser contract and RTL parity tests

"""Verify ADC-to-spike reference semantics, RTL bit-true parity, lint, and formal proof."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

from tools.adc_to_spike_reference import ADCSpikeConfig, ADCToSpikeReference

REPO_ROOT = Path(__file__).resolve().parents[1]

HDL = REPO_ROOT / "hdl" / "sensors" / "adc_to_spike_quantiser.v"
FORMAL = REPO_ROOT / "hdl" / "formal" / "sensors" / "adc_to_spike_quantiser.sby"


def _simulate_rtl_steps(
    tmp_path: Path,
    cfg: ADCSpikeConfig,
    samples: list[int],
) -> list[tuple[int, int, int, int, int, int]]:
    """Run ``adc_to_spike_quantiser.v`` in Icarus Verilog and return per-step outputs."""
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    assert iverilog is not None and vvp is not None

    mask = (1 << cfg.adc_width) - 1
    stim_lines = "\n".join(f"        drive_sample({cfg.adc_width}'d{s & mask});" for s in samples)
    tb_source = textwrap.dedent(f"""\
        `default_nettype none
        `timescale 1ns/1ps
        module tb_adc_to_spike;
            reg clk = 1'b0;
            always #5 clk = ~clk;
            reg rst_n = 1'b0;
            reg signed [{cfg.adc_width - 1}:0] adc_sample = {cfg.adc_width}'sd0;
            reg adc_valid = 1'b0;
            wire adc_ready;
            wire [15:0] aer_address;
            wire aer_valid;
            reg aer_ready = 1'b1;
            wire aer_polarity;
            wire dropped_sample;
            wire threshold_error;
            wire [31:0] sample_count;
            wire [31:0] spike_count;
            wire signed [{cfg.q_total - 1}:0] last_window_q;
            wire window_complete;
            wire signed [{cfg.q_total - 1}:0] window_average_q;
            wire [31:0] window_spike_budget;
            wire [31:0] pending_spikes;

            adc_to_spike_quantiser #(
                .ADC_WIDTH({cfg.adc_width}),
                .Q_INT({cfg.q_int}),
                .Q_FRAC({cfg.q_frac}),
                .AER_ADDR_WIDTH(16),
                .DECIMATION({cfg.decimation}),
                .SIGNED_INPUT({1 if cfg.signed_input else 0}),
                .BASE_ADDRESS(16'd{cfg.base_address}),
                .NEGATIVE_OFFSET(16'd{cfg.negative_offset})
            ) dut (
                .clk(clk),
                .rst_n(rst_n),
                .adc_sample(adc_sample),
                .adc_valid(adc_valid),
                .adc_ready(adc_ready),
                .threshold_q({cfg.q_total}'d{cfg.threshold_q}),
                .aer_address(aer_address),
                .aer_valid(aer_valid),
                .aer_ready(aer_ready),
                .aer_polarity(aer_polarity),
                .dropped_sample(dropped_sample),
                .threshold_error(threshold_error),
                .sample_count(sample_count),
                .spike_count(spike_count),
                .last_window_q(last_window_q),
                .window_complete(window_complete),
                .window_average_q(window_average_q),
                .window_spike_budget(window_spike_budget),
                .pending_spikes(pending_spikes)
            );

            task record_and_tick(input valid_in, input [{cfg.adc_width - 1}:0] sample_in);
                reg step_valid;
                reg [15:0] step_addr;
                reg step_pol;
                begin
                    adc_valid = valid_in;
                    adc_sample = sample_in;
                    #1;
                    step_valid = aer_valid;
                    step_addr = aer_address;
                    step_pol = aer_polarity;
                    @(posedge clk);
                    #1;
                    $display("STEP %0d %0d %0d %0d %0d %0d",
                        step_valid, step_addr, step_pol,
                        last_window_q, pending_spikes, spike_count);
                end
            endtask

            task drive_sample(input [{cfg.adc_width - 1}:0] s);
                begin
                    while (!adc_ready) begin
                        record_and_tick(1'b0, {cfg.adc_width}'d0);
                    end
                    record_and_tick(1'b1, s);
                end
            endtask

            initial begin
                @(posedge clk);
                #1 rst_n = 1'b1;
        {stim_lines}
                while (aer_valid) begin
                    record_and_tick(1'b0, {cfg.adc_width}'d0);
                end
                $finish;
            end
        endmodule
        `default_nettype wire
    """)
    tb_path = tmp_path / "tb_adc_to_spike.sv"
    sim_bin = tmp_path / "tb_adc_to_spike.vvp"
    tb_path.write_text(tb_source, encoding="utf-8")
    subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_bin), str(HDL), str(tb_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    proc = subprocess.run([vvp, str(sim_bin)], check=True, capture_output=True, text=True)
    rows: list[tuple[int, int, int, int, int, int]] = []
    for line in proc.stdout.splitlines():
        if line.startswith("STEP "):
            parts = [int(tok) for tok in line.split()[1:]]
            rows.append((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]))
    return rows


def test_quantise_adc_signed_scaling_and_saturation() -> None:
    """Reference quantiser scales 12-bit signed ADC samples into Q8.8 bounds."""
    ref = ADCToSpikeReference(ADCSpikeConfig(adc_width=12, q_int=8, q_frac=8, threshold_q=256))

    assert ref.quantise_adc(1) == 16
    assert ref.quantise_adc(0x7FF) == 32752
    assert ref.quantise_adc(0x800) == -32768


def test_reference_decimates_and_emits_rate_coded_spikes() -> None:
    """Positive decimated windows emit positive-polarity rate-coded AER spikes."""
    ref = ADCToSpikeReference(ADCSpikeConfig(decimation=2, threshold_q=128, base_address=10))
    steps = ref.run([256, 256])

    emitted = [step for step in steps if step.aer_valid]

    assert steps[1].window_q == 256
    assert len(emitted) == 2
    assert [step.aer_address for step in emitted] == [10, 10]
    assert ref.spike_count == 2


def test_negative_windows_emit_negative_aer_address() -> None:
    """Negative decimated windows emit negative-polarity offset AER addresses."""
    ref = ADCToSpikeReference(
        ADCSpikeConfig(decimation=2, threshold_q=128, base_address=10, negative_offset=3)
    )
    steps = ref.run([-256 & 0xFFFF, -256 & 0xFFFF])

    emitted = [step for step in steps if step.aer_valid]

    assert steps[1].window_q == -256
    assert len(emitted) == 2
    assert [step.aer_address for step in emitted] == [13, 13]
    assert all(step.aer_polarity == 1 for step in emitted)


@pytest.mark.parametrize(
    ("cfg", "samples"),
    [
        (
            ADCSpikeConfig(
                adc_width=8,
                q_int=4,
                q_frac=4,
                decimation=2,
                signed_input=True,
                threshold_q=16,
                base_address=10,
                negative_offset=3,
            ),
            [-32, -32, -128, -128, 48, 48, -1, 0],
        ),
        (
            ADCSpikeConfig(
                adc_width=12,
                q_int=4,
                q_frac=4,
                decimation=2,
                signed_input=True,
                threshold_q=16,
                base_address=20,
                negative_offset=5,
            ),
            [-512, -512, -2048, -2048, -25, -25, 512, 512],
        ),
        (
            ADCSpikeConfig(
                adc_width=8,
                q_int=6,
                q_frac=6,
                decimation=2,
                signed_input=False,
                threshold_q=256,
                base_address=4,
                negative_offset=1,
            ),
            [0, 0, 64, 64, 192, 192, 255, 255],
        ),
    ],
)
def test_rtl_matches_reference_across_signed_and_scaled_regimes(
    tmp_path: Path,
    cfg: ADCSpikeConfig,
    samples: list[int],
) -> None:
    """Verilog RTL matches Python reference bit-for-bit across negative and scaled inputs."""
    ref = ADCToSpikeReference(cfg)
    ref_steps = ref.run(samples)
    rtl_rows = _simulate_rtl_steps(tmp_path, cfg, samples)

    assert len(rtl_rows) == len(ref_steps)
    for idx, (rtl, expected) in enumerate(zip(rtl_rows, ref_steps, strict=True)):
        r_valid, r_addr, r_pol, r_win, r_pend, r_cnt = rtl
        assert bool(r_valid) is expected.aer_valid, f"step {idx} valid"
        if expected.aer_valid:
            assert r_addr == expected.aer_address, f"step {idx} address"
            assert r_pol == expected.aer_polarity, f"step {idx} polarity"
        assert r_win == expected.window_q, f"step {idx} window_q"
        assert r_pend == expected.pending_spikes, f"step {idx} pending_spikes"
        assert r_cnt == expected.spike_count, f"step {idx} spike_count"


def test_backpressure_latches_drop_when_source_ignores_ready() -> None:
    """Reference latches dropped_sample when adc_valid fires while busy."""
    ref = ADCToSpikeReference(ADCSpikeConfig(decimation=1, threshold_q=1))

    first = ref.step(4, adc_valid=True, aer_ready=False)
    second = ref.step(4, adc_valid=True, aer_ready=False)

    assert first.accepted_sample is True
    assert first.pending_spikes == 4
    assert second.accepted_sample is False
    assert second.dropped_sample is True


def test_invalid_threshold_is_rejected_by_reference() -> None:
    """Zero threshold_q is rejected during configuration validation."""
    with pytest.raises(ValueError, match="threshold_q"):
        ADCSpikeConfig(threshold_q=0).validate()


def test_verilator_lints_adc_to_spike_quantiser() -> None:
    """Verilator lints adc_to_spike_quantiser.v with zero warnings."""
    assert shutil.which("verilator") is not None
    subprocess.run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            "--timing",
            "--Wno-DECLFILENAME",
            str(HDL),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_symbiyosys_proves_adc_to_spike_contract(tmp_path: Path) -> None:
    """SymbiYosys proves bounded safety and non-vacuity cover properties."""
    assert shutil.which("sby") is not None
    assert shutil.which("cvc5") is not None
    subprocess.run(
        ["sby", "-f", "--prefix", str(tmp_path / "adc_to_spike_sby"), str(FORMAL)],
        cwd=REPO_ROOT,
        check=True,
    )
