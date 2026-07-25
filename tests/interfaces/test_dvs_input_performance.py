# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input performance contracts

"""Focused DVS input performance contracts."""

from tests.interfaces.dvs_input_support import *


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dvs_perf_small() -> None:
    """Benchmark processing a small event batch."""
    layer = DVSInputLayer(height=32, width=32)
    events = [(i % 32, i % 32, float(i), 1) for i in range(100)]
    start = time.perf_counter()
    _ = layer.process_events(events)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
