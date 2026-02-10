"""Strict blueprint semantics tests for FixedPointLIFNeuron."""

from dataclasses import dataclass

from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif


def _mask(value: int, width: int) -> int:
    m = (1 << width) - 1
    v = value & m
    if v >= (1 << (width - 1)):
        v -= 1 << width
    return v


@dataclass
class _BlueprintLif:
    data_width: int = 16
    fraction: int = 8
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 256
    refractory_period: int = 2

    def __post_init__(self):
        self.v = self.v_rest
        self.refractory_counter = 0

    def step(self, leak_k: int, gain_k: int, i_t: int, noise_in: int = 0):
        diff = _mask(self.v_rest - self.v, 2 * self.data_width)
        leak_mul = diff * leak_k
        dv_leak = _mask(leak_mul >> self.fraction, self.data_width)

        in_mul = i_t * gain_k
        dv_in = _mask(in_mul >> self.fraction, self.data_width)

        v_next = _mask(
            self.v + dv_leak + dv_in + noise_in,
            self.data_width,
        )

        if v_next >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
            self.refractory_counter = self.refractory_period
        else:
            spike = 0
            self.v = v_next

        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_rest
            spike = 0

        return spike, _mask(self.v, self.data_width)


class TestLIFBlueprintSemantics:
    def test_100_steps_constant_input(self):
        ref = _BlueprintLif()
        v3 = V3Lif()
        for t in range(100):
            exp_spike, exp_v = ref.step(leak_k=20, gain_k=256, i_t=128, noise_in=0)
            got_spike, got_v = v3.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
            assert got_spike == exp_spike, f"Spike mismatch at step {t}"
            assert got_v == exp_v, f"Voltage mismatch at step {t}: expected={exp_v}, got={got_v}"

    def test_refractory_override_order(self):
        ref = _BlueprintLif(refractory_period=5)
        v3 = V3Lif(refractory_period=5)
        for t in range(200):
            exp_spike, exp_v = ref.step(20, 256, 200, 0)
            got_spike, got_v = v3.step(20, 256, 200, 0)
            assert got_spike == exp_spike, f"Spike mismatch at step {t}"
            assert got_v == exp_v, f"Voltage mismatch at step {t}"
