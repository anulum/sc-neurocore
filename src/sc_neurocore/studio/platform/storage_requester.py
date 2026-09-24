# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — authenticated requester of the current HTTP request

"""Carry the gateway's decision for one HTTP request to its storage operations.

The security middleware authenticates the principal and authorises the route;
only then does it open a delegation for the rest of that request. Storage
clients read it to name the requester and the route whose policy the storage
authority re-applies. Code outside a request has no delegation, and the
isolated facade refuses to act for nobody.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester


@dataclass(frozen=True, slots=True)
class Delegation:
    """The allowed request: who asked, through which route, under which trace."""

    requester: StorageRequester | None
    method: str
    route: str
    request_id: str


_CURRENT: ContextVar[Delegation | None] = ContextVar("studio_storage_delegation", default=None)


@contextmanager
def delegated(
    principal: Principal | None, *, method: str, route: str, request_id: str
) -> Iterator[Delegation]:
    """Open the delegation of one authorised request for its duration.

    Parameters
    ----------
    principal : Principal or None
        The gateway-authenticated principal; ``None`` for a public route.
    method, route : str
        The HTTP method and route template the gateway authorised.
    request_id : str
        The request's trace identifier.
    """
    requester = (
        None
        if principal is None
        else StorageRequester(
            principal_id=principal.principal_id, roles=tuple(sorted(principal.roles))
        )
    )
    delegation = Delegation(requester=requester, method=method, route=route, request_id=request_id)
    token = _CURRENT.set(delegation)
    try:
        yield delegation
    finally:
        _CURRENT.reset(token)


def current_delegation() -> Delegation | None:
    """Return the delegation of the request being served, if any."""
    return _CURRENT.get()


__all__ = ["Delegation", "current_delegation", "delegated"]
