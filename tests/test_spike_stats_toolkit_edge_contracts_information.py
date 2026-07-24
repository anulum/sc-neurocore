# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (information) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_te_short():
    assert transfer_entropy(np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8), lag=5) == 0.0


def test_entropy_short():
    r = spike_train_entropy(np.zeros(2, dtype=np.int8), word_length=5)
    assert np.isnan(r)


def test_noise_entropy_short():
    r = noise_entropy(np.zeros(2, dtype=np.int8), n_trials=1, word_length=5)
    assert np.isnan(r)


def test_ssi_empty():
    assert stimulus_specific_information(np.array([]), np.array([], dtype=int)) == 0.0


def test_ssi_zero_mean():
    assert stimulus_specific_information(np.zeros(10), np.zeros(10, dtype=int)) == 0.0


def test_kl_mi_short():
    assert kozachenko_leonenko_mi(np.zeros(2), np.zeros(2)) == 0.0


def test_time_rescaling_few():
    p, sig = time_rescaling_ks_test(np.array([0.1, 0.5]), rate_func=lambda t: 10.0)
    assert p == 1.0


def test_information_python_fallback_estimators(monkeypatch):
    monkeypatch.setattr(information_module, "_HAS_RUST", False)
    monkeypatch.setattr(information_module, "_ssc", None)

    alternating = np.array([1, 0, 1, 0] * 16, dtype=np.int8)
    copied = alternating.copy()
    inverted = 1 - alternating

    assert mutual_information(alternating, copied, bin_size=1) > 0.9
    assert transfer_entropy(alternating, np.roll(alternating, 1), bin_size=1, lag=1) >= 0.0

    entropy = spike_train_entropy(alternating, bin_size=1, word_length=2)
    assert entropy > 0.9

    repeated_trials = np.tile(np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8), 12)
    noise = noise_entropy(repeated_trials, n_trials=4, bin_size=1, word_length=3)
    assert np.isfinite(noise)
    assert noise >= 0.0
    monkeypatch.setattr(information_module, "spike_train_entropy", lambda *_args: float("nan"))
    assert np.isnan(noise_entropy(repeated_trials, n_trials=4, bin_size=1, word_length=3))

    counts = np.array([8, 9, 7, 1, 1, 2], dtype=np.float64)
    stimuli = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assert stimulus_specific_information(counts, stimuli) > 0.0
    assert stimulus_specific_information(np.array([1.0]), np.array([np.nan])) == 0.0

    x = np.linspace(0.0, 1.0, 24)
    y = x + np.linspace(0.0, 0.01, 24)
    assert kozachenko_leonenko_mi(x, y, k=3) >= 0.0
    assert kozachenko_leonenko_mi(x, inverted[: x.size], k=3) >= 0.0

    ks, passes = time_rescaling_ks_test(
        np.array([0.05, 0.18, 0.31, 0.45, 0.62, 0.8]),
        rate_func=lambda _t: 7.0,
        t_start=0.0,
        t_end=1.0,
    )
    assert 0.0 <= ks <= 1.0
    assert isinstance(passes, bool)


def test_information_rust_acceleration_delegation(monkeypatch):
    class RustCore:
        def py_spike_train_entropy(self, binned, word_length):
            assert binned.flags.c_contiguous
            assert word_length == 2
            return 1.5

        def py_kozachenko_leonenko_mi(self, x, y, k):
            assert x.flags.c_contiguous
            assert y.flags.c_contiguous
            assert k == 2
            return 0.25

    monkeypatch.setattr(information_module, "_HAS_RUST", True)
    monkeypatch.setattr(information_module, "_ssc", RustCore())

    assert (
        spike_train_entropy(np.array([1, 0, 1, 0], dtype=np.int8), bin_size=1, word_length=2) == 1.5
    )
    assert kozachenko_leonenko_mi(np.arange(8.0), np.arange(8.0), k=2) == 0.25
