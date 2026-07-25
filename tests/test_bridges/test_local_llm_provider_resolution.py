# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM provider resolution tests

"""Validate automatic provider selection and endpoint resolution."""

from __future__ import annotations

import pytest

from sc_neurocore.bridges.local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMProvider,
)


def test_auto_provider_prefers_ollama_for_default_port() -> None:
    config = LocalLLMConfig()
    assert config.resolved_provider() is LocalLLMProvider.OLLAMA


def test_auto_provider_falls_back_to_chat_completions_for_generic_host() -> None:
    """A non-Ollama base URL resolves AUTO to the chat-completions provider."""
    config = LocalLLMConfig(base_url="http://localhost:8000", provider=LocalLLMProvider.AUTO)

    assert config.resolved_provider() is LocalLLMProvider.CHAT_COMPLETIONS


def test_endpoint_rejects_unresolved_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint built from an unresolved AUTO provider raises a clear error."""
    monkeypatch.setattr(LocalLLMConfig, "resolved_provider", lambda self: LocalLLMProvider.AUTO)
    bridge = LocalLLMBridge(LocalLLMConfig())

    with pytest.raises(LocalLLMError, match="Unsupported provider"):
        bridge._endpoint()
