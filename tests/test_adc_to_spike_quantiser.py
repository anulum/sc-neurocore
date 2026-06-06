# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADC-to-spike quantiser module contract tests.

These tests cover the NEU-C.5 sensor-ingress contract: ADC samples are converted
bit-true into Q-format windows, decimated into deterministic rate-coded AER
spikes, and protected by explicit backpressure/drop telemetry.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from adc_to_spike_reference import ADCSpikeConfig, ADCToSpikeReference

HDL = REPO_ROOT / "hdl" / "sensors" / "adc_to_spike_quantiser.v"
FORMAL = REPO_ROOT / "hdl" / "formal" / "sensors" / "adc_to_spike_quantiser.sby"


def test_quantise_adc_signed_scaling_and_saturation() -> None:
    ref = ADCToSpikeReference(ADCSpikeConfig(adc_width=12, q_int=8, q_frac=8, threshold_q=256))

    assert ref.quantise_adc(1) == 16
    assert ref.quantise_adc(0x7FF) == 32752
    assert ref.quantise_adc(0x800) == -32768


def test_reference_decimates_and_emits_rate_coded_spikes() -> None:
    ref = ADCToSpikeReference(ADCSpikeConfig(decimation=2, threshold_q=128, base_address=10))
    steps = ref.run([256, 256])

    emitted = [step for step in steps if step.aer_valid]

    assert steps[1].window_q == 256
    assert len(emitted) == 2
    assert [step.aer_address for step in emitted] == [10, 10]
    assert ref.spike_count == 2


def test_negative_windows_emit_negative_aer_address() -> None:
    ref = ADCToSpikeReference(
        ADCSpikeConfig(decimation=2, threshold_q=128, base_address=10, negative_offset=3)
    )
    steps = ref.run([-256 & 0xFFFF, -256 & 0xFFFF])

    emitted = [step for step in steps if step.aer_valid]

    assert steps[1].window_q == -256
    assert len(emitted) == 2
    assert [step.aer_address for step in emitted] == [13, 13]
    assert all(step.aer_polarity == 1 for step in emitted)


def test_backpressure_latches_drop_when_source_ignores_ready() -> None:
    ref = ADCToSpikeReference(ADCSpikeConfig(decimation=1, threshold_q=1))

    first = ref.step(4, adc_valid=True, aer_ready=False)
    second = ref.step(4, adc_valid=True, aer_ready=False)

    assert first.accepted_sample is True
    assert first.pending_spikes == 4
    assert second.accepted_sample is False
    assert second.dropped_sample is True


def test_invalid_threshold_is_rejected_by_reference() -> None:
    with pytest.raises(ValueError, match="threshold_q"):
        ADCSpikeConfig(threshold_q=0).validate()


def test_verilator_lints_adc_to_spike_quantiser() -> None:
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
    assert shutil.which("sby") is not None
    assert shutil.which("cvc5") is not None
    subprocess.run(
        ["sby", "-f", "-d", str(tmp_path / "adc_to_spike_sby"), str(FORMAL)],
        cwd=REPO_ROOT,
        check=True,
    )
