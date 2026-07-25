# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Local LLM HTTP response test support

"""HTTP response helpers shared by local-LLM transport tests."""

from __future__ import annotations

import pytest


class _FakeHTTPResponse:
    """Provide the response context-manager contract used by ``urlopen``."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        """Return the configured response body."""
        return self._body


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    """Replace the local-LLM module's HTTP transport for one test."""
    monkeypatch.setattr("sc_neurocore.bridges.local_llm.urlopen", handler)
