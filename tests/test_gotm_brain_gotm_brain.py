# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGOTMBrain from former test_gotm_brain.py

"""Focused suite: TestGOTMBrain from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403

class TestGOTMBrain:
    def test_init(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        assert brain.n_neurons == 8
        assert len(brain.neurons) == 8
        assert brain._total_steps == 0

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="n_neurons"):
            GOTMBrain(n_neurons=0)

    def test_get_llm_guidance_returns_valid(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        directive = brain.get_llm_guidance("test context")
        # Must always return a valid directive regardless of LLM availability
        assert directive in ("FOCUS", "EXPLORE", "STABILIZE")

    def test_process_content(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        vector = np.random.rand(8)
        spikes = brain.process_content(vector, "FOCUS")
        assert isinstance(spikes, list)
        for s in spikes:
            assert 0 <= s < 8

    def test_process_content_returns_indices_for_spiking_neurons(self) -> None:
        """Spiking neuron steps are returned as their stable indices."""
        brain = GOTMBrain(n_neurons=3, bridge_backend="emulated")
        brain.neurons = cast(
            list[HybridFisherPosnerLIF],
            [
                _FixedSpikeNeuron(False),
                _FixedSpikeNeuron(True),
                _FixedSpikeNeuron(True),
            ],
        )

        assert brain.process_content(np.ones(3), "STABILIZE") == [1, 2]

    def test_import_marks_missing_local_llm_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Importing without a local llm module leaves the fallback path enabled."""
        import sc_neurocore.quantum_cognition.gotm_brain as canonical_gotm

        monkeypatch.delitem(sys.modules, "llm", raising=False)
        sys.meta_path.insert(0, _BlockingLLMFinder())
        sys.modules.pop(_GOTM_MODULE, None)
        try:
            module = importlib.import_module(_GOTM_MODULE)
        finally:
            sys.meta_path = [
                finder for finder in sys.meta_path if not isinstance(finder, _BlockingLLMFinder)
            ]
            sys.modules[_GOTM_MODULE] = canonical_gotm

        gotm = cast(_GotmBrainModule, module)
        assert gotm.HAS_LLM is False
        assert gotm._LLMEndpoint is None

    def test_process_content_padding(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        short_vector = np.ones(4)
        spikes = brain.process_content(short_vector, "EXPLORE")
        assert isinstance(spikes, list)

    def test_process_content_truncation(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        long_vector = np.ones(16)
        spikes = brain.process_content(long_vector, "STABILIZE")
        assert isinstance(spikes, list)

    def test_learn_step(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "test content", "code", 1.0)
        vector = np.random.rand(8)
        step = brain.learn_step(chunk, vector)
        assert isinstance(step, LearningStep)
        assert step.step_index == 0
        assert step.directive in ("FOCUS", "EXPLORE", "STABILIZE")
        assert brain._total_steps == 1

    def test_learn_from_repo(self, tmp_repo: Path) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        steps = brain.learn_from_repo(str(tmp_repo), max_chunks=5)
        assert len(steps) <= 5
        assert brain._total_steps == len(steps)

    def test_get_learning_state(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "state test", "code", 1.0)
        brain.learn_step(chunk, np.random.rand(4))
        state = brain.get_learning_state()
        assert state["n_neurons"] == 4
        assert state["total_steps"] == 1
        assert "pool_state" in state

    def test_get_history(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "history test", "code", 1.0)
        brain.learn_step(chunk, np.random.rand(4))
        history = brain.get_history()
        assert len(history) == 1
        assert history[0]["directive"] in ("FOCUS", "EXPLORE", "STABILIZE")

    def test_reset(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        brain.learn_step(
            ContentChunk("R", "a.py", 0, "reset test", "code", 1.0),
            np.random.rand(4),
        )
        brain.reset()
        assert brain._total_steps == 0
        assert len(brain._history) == 0

    def test_repr(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        r = repr(brain)
        assert "GOTMBrain" in r
        assert "n_neurons=4" in r

    def test_entanglement_evolves(self, tmp_repo: Path) -> None:
        """After learning, entanglement should have structure (non-uniform)."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        initial_ent = brain.pool.entanglement_map.copy()
        brain.learn_from_repo(str(tmp_repo), max_chunks=10)
        if brain._total_steps > 0 and sum(n._total_spikes for n in brain.neurons) > 0:
            assert not np.allclose(brain.pool.entanglement_map, initial_ent)
