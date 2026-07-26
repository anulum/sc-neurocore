# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo Mojo backend tests

"""ULP-bounded Mojo FitzHugh–Nagumo parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fitzhugh_nagumo_backends_support import _CURRENTS, _MOJO_ATOL, _mojo, _run


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Nagumo backend unavailable")
@pytest.mark.parametrize("current", _CURRENTS)
def test_mojo_trace_ulp_bounded_and_exact_spikes(current: float) -> None:
    ref, ref_spikes, _rv, _rw = _run("python", current=current)
    got, spikes, _vf, _wf = _run("mojo", current=current)
    np.testing.assert_allclose(got, ref, atol=_MOJO_ATOL, rtol=0.0)
    assert spikes == ref_spikes


@pytest.mark.skipif(not _mojo(), reason="Mojo FitzHugh-Nagumo backend unavailable")
def test_mojo_band_does_not_amplify() -> None:
    ref, _rs, _rv, _rw = _run("python", current=0.5, n=50000)
    got, _gs, _vf, _wf = _run("mojo", current=0.5, n=50000)
    assert float(np.max(np.abs(got - ref))) < 1e-9
