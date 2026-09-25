# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Concurrent first requests build the model catalogue once

"""A page asks for the model list, its facets and a query at the same moment.

On a cold start each of those requests needs the whole catalogue. Built once
per request, the builds compete for the interpreter and every one of them is
slow; built once and shared, the later requests wait for the first.
"""

from __future__ import annotations

import threading
from typing import Any

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio import model_catalogue as catalogue


def test_concurrent_first_calls_share_one_build() -> None:
    catalogue._models_cache = None
    start = threading.Barrier(6)
    results: list[list[dict[str, Any]]] = []
    lock = threading.Lock()

    def request() -> None:
        start.wait()
        models = catalogue.list_models()
        with lock:
            results.append(models)

    threads = [threading.Thread(target=request) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    assert len(results) == 6
    assert all(models is results[0] for models in results)
    assert {row["name"] for row in results[0]} == set(_CLASS_TO_MODULE)
