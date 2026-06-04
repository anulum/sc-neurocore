"""Verilator/Python parity entry point for NEU-C.5 ADC-to-spike ingress.

The repository-local pytest surface exercises the Python reference and HDL lint.
This file remains the cosimulation contract entry point for runners that attach
a Verilator signal harness to `hdl/sensors/adc_to_spike_quantiser.v`.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from adc_to_spike_reference import ADCSpikeConfig, ADCToSpikeReference


def test_adc_to_spike_reference_cosim_trace_contract() -> None:
    ref = ADCToSpikeReference(
        ADCSpikeConfig(decimation=4, threshold_q=64, base_address=2, negative_offset=1)
    )
    steps = ref.run([64, 128, 192, 256, -64 & 0xFFFF, -128 & 0xFFFF, -192 & 0xFFFF, -256 & 0xFFFF])
    emitted = [(step.aer_address, step.aer_polarity) for step in steps if step.aer_valid]

    assert emitted[:2] == [(2, 0), (2, 0)]
    assert emitted[-2:] == [(3, 1), (3, 1)]
