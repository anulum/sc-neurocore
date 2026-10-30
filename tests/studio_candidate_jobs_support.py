# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual candidate job HTTP workflow observations

"""Wait for actual candidate task state through its requester-owned HTTP route."""

from __future__ import annotations

import time
from typing import Any

from starlette.testclient import TestClient


def finish_candidate_job(
    client: TestClient, status_route: str, *, timeout: float = 45.0
) -> dict[str, Any]:
    """Observe a real submitted process to completion, rejecting failed recovery states.

    Parameters
    ----------
    client : TestClient
        Actual Studio application transport, including its current actor headers.
    status_route : str
        Route returned by the actual submission receipt.
    timeout : float
        Finite observation deadline in seconds; never a process-restart trigger.

    Returns
    -------
    dict[str, Any]
        The completed durable public job record.

    Raises
    ------
    AssertionError
        A route refused, the worker failed/interrupted, or observation expired.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(status_route)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] == "completed":
            return dict(record)
        assert record["status"] in {"pending", "running", "cancelling"}, record
        time.sleep(0.05)
    raise AssertionError("candidate job did not complete before its observation deadline")
