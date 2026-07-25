# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream current-source opt-in performance gate

"""Opt-in stepping performance gate for BitstreamCurrentSource."""

from tests.sources.bitstream_current_source_support import *


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_source_perf_small():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    """Benchmark a short stepping loop."""
    source = _make_source(length=256)
    start = time.perf_counter()
    for _ in range(256):
        _ = source.step()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
