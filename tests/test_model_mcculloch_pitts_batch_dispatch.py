# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts batch and dispatch contracts

"""Python batch, request validation, and backend-dispatch contracts."""

from .model_mcculloch_pitts_support import *


def test_python_batch_matches_scalar_truth_rows() -> None:
    """Varying excitation and inhibition return an exact contiguous binary trace."""
    counts = np.array([0, 1, 2, 3, _INT32_MAX], dtype=np.int64)
    flags = np.array([False, False, False, True, True], dtype=np.bool_)
    neuron = McCullochPittsNeuron(theta=2)
    events, event_count = neuron.simulate(counts, flags, backend="python")
    assert events.tolist() == [0, 0, 1, 0, 0]
    assert events.dtype == np.uint8
    assert events.flags.c_contiguous
    assert event_count == 1 == int(events.sum())
    assert [neuron.step(count, flag) for count, flag in zip(counts, flags, strict=True)] == (
        events.tolist()
    )


def test_batch_defaults_to_no_inhibition_and_accepts_empty_input() -> None:
    """Absent flags mean no veto; an empty stateless batch has zero events."""
    neuron = McCullochPittsNeuron(theta=2)
    events, count = neuron.simulate([0.0, 1.0, 2.0], backend="python")
    assert events.tolist() == [0, 0, 1]
    assert count == 1
    empty, empty_count = neuron.simulate([], [], backend="python")
    assert empty.shape == (0,)
    assert empty_count == 0


@pytest.mark.parametrize(
    ("counts", "flags", "message"),
    (
        (np.array(1), None, "one-dimensional"),
        (np.zeros((1, 1)), None, "one-dimensional"),
        ([0, 1], [False], "match"),
        ([0, 1], np.zeros((1, 2), dtype=np.bool_), "one-dimensional"),
        ([0, 1], [False, 1], "inhibitory_flags"),
        ([0, -1], None, r"excitatory_counts\[1\]"),
    ),
)
def test_batch_validation_fails_before_dispatch(
    counts: object,
    flags: object,
    message: str,
) -> None:
    """Malformed shapes and values cannot reach a native pointer boundary."""
    with pytest.raises(ValueError, match=message):
        McCullochPittsNeuron().simulate(
            cast(list[object], counts),
            cast(list[object] | None, flags),
            backend="python",
        )


def test_unknown_and_unavailable_backends_do_not_fall_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit backend selection is fail-closed."""
    neuron = McCullochPittsNeuron()
    with pytest.raises(ValueError, match="backend"):
        neuron.simulate([1], backend="cuda")

    from sc_neurocore.accel import mcculloch_pitts as backends

    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Go"):
        neuron.simulate([1], backend="go")
