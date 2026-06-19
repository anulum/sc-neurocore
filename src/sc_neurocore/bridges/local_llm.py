# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM bridge for spike-derived prompts

"""Local LLM bridge for spike-derived prompts and analysis.

This module is intentionally local-only and opt-in. It does not talk to any
hosted service. Supported endpoint styles:

- Ollama-style chat API at ``/api/chat``
- generic chat-completions API at ``/v1/chat/completions``

The bridge is useful when SC-NeuroCore data should be explained or summarised
through a locally hosted language model without introducing a cloud dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


class LocalLLMError(RuntimeError):
    """Raised when the local LLM endpoint is unavailable or malformed."""


class LocalLLMProvider(Enum):
    """Local LLM endpoint protocol."""

    AUTO = "auto"
    OLLAMA = "ollama"
    CHAT_COMPLETIONS = "chat_completions"


@dataclass(frozen=True)
class LocalLLMConfig:
    """Connection and generation settings for a local LLM endpoint."""

    base_url: str = "http://127.0.0.1:11434"
    provider: LocalLLMProvider = LocalLLMProvider.AUTO
    model: str = "qwen2.5:7b-instruct"
    timeout_sec: float = 30.0
    temperature: float = 0.2
    max_tokens: int | None = 256
    system_prompt: str = (
        "You are a local analysis model summarising stochastic neural activity. "
        "Be precise, concise, and report uncertainty when the signal is weak."
    )
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_provider(self) -> LocalLLMProvider:
        """Resolve AUTO to a concrete provider using the configured URL."""
        if self.provider is not LocalLLMProvider.AUTO:
            return self.provider
        if ":11434" in self.base_url or self.base_url.rstrip("/").endswith("/api"):
            return LocalLLMProvider.OLLAMA
        return LocalLLMProvider.CHAT_COMPLETIONS


@dataclass(frozen=True)
class LocalLLMResponse:
    """Structured response from a local LLM endpoint."""

    text: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict[str, Any]


class SpikePromptAdapter:
    """Convert spike activity into compact text suitable for local LLM prompts."""

    @staticmethod
    def summarise_rates(
        rates_hz: np.ndarray[Any, Any],
        *,
        neuron_labels: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        """Summarise per-neuron firing rates as compact ranked text."""
        flat = np.asarray(rates_hz, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return "No neurons were provided."
        labels = neuron_labels or [f"n{i}" for i in range(flat.size)]
        order = np.argsort(flat)[::-1][: max(1, min(top_k, flat.size))]
        lines = [
            f"mean_rate_hz={float(np.mean(flat)):.4f}",
            f"std_rate_hz={float(np.std(flat)):.4f}",
            f"max_rate_hz={float(np.max(flat)):.4f}",
            "top_neurons:",
        ]
        for idx in order:
            lines.append(f"- {labels[int(idx)]}: {float(flat[int(idx)]):.4f} Hz")
        return "\n".join(lines)

    @staticmethod
    def raster_summary(
        raster: np.ndarray[Any, Any],
        *,
        dt_ms: float = 1.0,
        neuron_labels: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        """Summarise a boolean spike raster into rates and density statistics."""
        arr = np.asarray(raster)
        if arr.ndim != 2:
            raise ValueError("raster must have shape (time, neurons)")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return "Empty spike raster."
        duration_s = arr.shape[0] * dt_ms * 1e-3
        if duration_s <= 0.0:
            raise ValueError("dt_ms must be positive")
        counts = arr.astype(np.float64).sum(axis=0)
        rates = counts / duration_s
        density = float(arr.mean())
        prefix = [
            f"timesteps={arr.shape[0]}",
            f"neurons={arr.shape[1]}",
            f"dt_ms={dt_ms:.6f}",
            f"density={density:.6f}",
        ]
        return "\n".join(
            prefix
            + [SpikePromptAdapter.summarise_rates(rates, neuron_labels=neuron_labels, top_k=top_k)]
        )


class LocalLLMBridge:
    """Thin client for local LLM chat endpoints."""

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        provider = self.config.resolved_provider()
        if provider is LocalLLMProvider.OLLAMA:
            return f"{base}/api/chat"
        if provider is LocalLLMProvider.CHAT_COMPLETIONS:
            return f"{base}/v1/chat/completions"
        raise LocalLLMError(f"Unsupported provider: {provider.value}")

    @staticmethod
    def _validate_endpoint(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise LocalLLMError("Local LLM endpoint must use http or https")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise LocalLLMError("Local LLM endpoint must be loopback-local")
        return url

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        req = Request(
            self._validate_endpoint(self._endpoint()),
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.config.timeout_sec) as resp:  # nosec B310
                body = resp.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LocalLLMError(f"Local LLM HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LocalLLMError(f"Local LLM endpoint unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LocalLLMError("Local LLM request timed out") from exc
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LocalLLMError("Local LLM returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise LocalLLMError("Local LLM returned non-object JSON")
        return parsed

    def chat(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LocalLLMResponse:
        """Send a prompt to the configured local LLM chat endpoint."""
        provider = self.config.resolved_provider()
        sys_prompt = system_prompt or self.config.system_prompt
        model_name = model or self.config.model
        temp = self.config.temperature if temperature is None else temperature
        token_limit = self.config.max_tokens if max_tokens is None else max_tokens
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if provider is LocalLLMProvider.OLLAMA:
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temp},
            }
            if token_limit is not None:
                payload["options"]["num_predict"] = token_limit
            raw = self._post_json(payload)
            message = raw.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise LocalLLMError("Ollama response missing message.content")
            prompt_tokens = None
            completion_tokens = None
            if isinstance(raw.get("prompt_eval_count"), int):
                prompt_tokens = int(raw["prompt_eval_count"])
            if isinstance(raw.get("eval_count"), int):
                completion_tokens = int(raw["eval_count"])
            finish_reason = raw.get("done_reason")
            if finish_reason is not None:
                finish_reason = str(finish_reason)
            return LocalLLMResponse(
                text=message["content"],
                model=str(raw.get("model", model_name)),
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw=raw,
            )

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
        }
        if token_limit is not None:
            payload["max_tokens"] = token_limit
        raw = self._post_json(payload)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalLLMError("chat-completions response missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LocalLLMError("chat-completions choice is malformed")
        message = first.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalLLMError("chat-completions response missing choices[0].message.content")
        usage_obj = raw.get("usage")
        usage: dict[str, Any] = usage_obj if isinstance(usage_obj, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return LocalLLMResponse(
            text=message["content"],
            model=str(raw.get("model", model_name)),
            finish_reason=str(first["finish_reason"])
            if first.get("finish_reason") is not None
            else None,
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            completion_tokens=int(completion_tokens)
            if isinstance(completion_tokens, int)
            else None,
            raw=raw,
        )

    def analyse_spike_raster(
        self,
        raster: np.ndarray[Any, Any],
        *,
        question: str = "Summarise the neural activity and identify the most active neurons.",
        dt_ms: float = 1.0,
        neuron_labels: list[str] | None = None,
        top_k: int = 8,
        model: str | None = None,
    ) -> LocalLLMResponse:
        """Summarise a spike raster through the local LLM."""
        summary = SpikePromptAdapter.raster_summary(
            raster,
            dt_ms=dt_ms,
            neuron_labels=neuron_labels,
            top_k=top_k,
        )
        prompt = f"{question}\n\nSpike summary:\n{summary}"
        return self.chat(prompt, model=model)
