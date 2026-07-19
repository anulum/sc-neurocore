# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direction-selective RGC model contracts

"""Module-specific behavioural contracts for ``DirectionSelectiveRGC``."""

from __future__ import annotations

import pytest


class TestDirectionSelectiveRGC:
    @pytest.fixture()
    def on_cell(self):
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        return DirectionSelectiveRGC.new_on()

    @pytest.fixture()
    def off_cell(self):
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        return DirectionSelectiveRGC.new_off()

    def test_on_centre_flag(self, on_cell, off_cell):
        assert on_cell.is_on_centre is True
        assert off_cell.is_on_centre is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"theta": 0.0},
            {"is_on_centre": 1},
            {"w_centre": -0.01},
            {"w_surround": -0.01},
            {"direction_pref": float("nan")},
            {"dt": 0.0},
            {"v": float("inf")},
            {"_prev_intensity": -0.01},
            {"_surround": -0.01},
        ],
    )
    def test_rejects_non_physical_direction_selective_parameters(self, kwargs):
        """Retinal direction-selective state and tuning parameters must be physical."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        with pytest.raises(ValueError):
            DirectionSelectiveRGC(**kwargs)

    @pytest.mark.parametrize(
        ("intensity", "surround_mean"),
        [(float("nan"), 0.0), (0.0, float("inf")), (-0.01, 0.0), (0.0, -0.01)],
    )
    def test_rejects_non_physical_receptive_field_drive(self, intensity, surround_mean):
        """Optical centre and surround drives must be finite non-negative intensities."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        with pytest.raises(ValueError):
            DirectionSelectiveRGC.new_on().step_rf(intensity, surround_mean)

    def test_on_responds_to_light_increase(self, on_cell):
        """On-centre must respond to light onset (positive dI/dt)."""
        for _ in range(10):
            on_cell.step_rf(0.0, 0.0)
        spikes = sum(on_cell.step_rf(6.0, 0.0) for _ in range(30))
        assert spikes > 0

    def test_off_responds_to_light_decrease(self, off_cell):
        """Off-centre must respond to light offset (negative dI/dt)."""
        off_cell.theta = 0.1
        spikes = 0
        for i in range(400):
            intensity = 5.0 if (i // 10) % 2 == 0 else 0.0
            spikes += off_cell.step_rf(intensity, 0.0)
        assert spikes > 0

    def test_surround_inhibition_reduces_firing(self):
        """Surround illumination should reduce centre response."""
        from sc_neurocore.neurons.models import DirectionSelectiveRGC

        no_surr = DirectionSelectiveRGC.new_on()
        with_surr = DirectionSelectiveRGC.new_on()
        s_no = 0
        s_surr = 0
        for i in range(300):
            intensity = 3.0 if i % 10 == 0 else 0.0
            s_no += no_surr.step_rf(intensity, 0.0)
            s_surr += with_surr.step_rf(intensity, 2.0)
        assert s_surr <= s_no, "Surround should suppress firing"

    def test_temporal_derivative(self, on_cell):
        """Constant light should produce no spikes (zero dI/dt)."""
        # Warm up with constant light.
        for _ in range(100):
            on_cell.step_rf(3.0, 0.0)
        # After adaptation, constant light has no temporal derivative.
        late_spikes = sum(on_cell.step_rf(3.0, 0.0) for _ in range(100))
        assert late_spikes == 0, "Constant light should not drive On-centre"

    def test_reset(self, on_cell):
        for _ in range(50):
            on_cell.step_rf(5.0, 1.0)
        on_cell.reset()
        assert on_cell.v == 0.0
        assert on_cell._prev_intensity == 0.0
