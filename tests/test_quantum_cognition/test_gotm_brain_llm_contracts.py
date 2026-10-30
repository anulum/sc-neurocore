# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


"""Local-LLM contract tests for the GOTM brain learning directive path."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from typing import Generator, Protocol, cast

import pytest

import sc_neurocore.quantum_cognition.gotm_brain as canonical_gotm

_GOTM_MODULE = "sc_neurocore.quantum_cognition.gotm_brain"


class _GotmBrainModule(Protocol):
    """Typed view of the dynamically reloaded GOTM brain module."""

    HAS_LLM: bool
    GOTMBrain: type[canonical_gotm.GOTMBrain]
    _LLMEndpoint: type[object] | None


class _MutableLLMModule(Protocol):
    """Typed view of the fake local llm module used for re-import tests."""

    Endpoint: type[object]
    chat: Callable[..., str]


def _fake_llm_module(chat: Callable[..., str]) -> types.ModuleType:
    """Build a local llm module matching the production import contract."""
    module = types.ModuleType("llm")

    class Endpoint:
        """Minimal endpoint object accepted by GOTMBrain."""

    mutable_module = cast(_MutableLLMModule, module)
    mutable_module.Endpoint = Endpoint
    mutable_module.chat = chat
    return module


def _reload_gotm_with_llm(
    monkeypatch: pytest.MonkeyPatch,
    llm_module: types.ModuleType,
) -> _GotmBrainModule:
    """Reload gotm_brain against a supplied local llm module."""
    monkeypatch.setitem(sys.modules, "llm", llm_module)
    sys.modules.pop(_GOTM_MODULE, None)
    module = importlib.import_module(_GOTM_MODULE)
    return cast(_GotmBrainModule, module)


@pytest.fixture(autouse=True)
def _restore_gotm_module() -> Generator[None, None, None]:
    """Restore the canonical module after dynamic import contract tests."""
    try:
        yield
    finally:
        sys.modules[_GOTM_MODULE] = canonical_gotm


def test_gotm_brain_import_detects_local_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing with a local llm module enables the LLM directive path."""

    def chat(_prompt: str, **_kwargs: object) -> str:
        """Return a valid directive with punctuation."""
        return "focus."

    gotm = _reload_gotm_with_llm(monkeypatch, _fake_llm_module(chat))

    assert gotm.HAS_LLM is True
    assert gotm._LLMEndpoint is not None
    brain = gotm.GOTMBrain(n_neurons=4, bridge_backend="emulated", llm_endpoint=gotm._LLMEndpoint())
    assert brain.get_llm_guidance("structured algebra context") == "FOCUS"


def test_gotm_brain_invalid_llm_directive_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown local LLM replies fall back to the stable directive."""

    def chat(_prompt: str, **_kwargs: object) -> str:
        """Return a syntactically valid but unsupported directive."""
        return "ponder"

    gotm = _reload_gotm_with_llm(monkeypatch, _fake_llm_module(chat))
    assert gotm._LLMEndpoint is not None

    brain = gotm.GOTMBrain(n_neurons=4, bridge_backend="emulated", llm_endpoint=gotm._LLMEndpoint())
    assert brain.get_llm_guidance("topology context") == "STABILIZE"


def test_gotm_brain_llm_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local LLM runtime errors fall back to the stable directive."""

    def chat(_prompt: str, **_kwargs: object) -> str:
        """Raise the same kind of local endpoint error production catches."""
        raise RuntimeError("local llm offline")

    gotm = _reload_gotm_with_llm(monkeypatch, _fake_llm_module(chat))
    assert gotm._LLMEndpoint is not None

    brain = gotm.GOTMBrain(n_neurons=4, bridge_backend="emulated", llm_endpoint=gotm._LLMEndpoint())
    assert brain.get_llm_guidance("offline context") == "STABILIZE"
