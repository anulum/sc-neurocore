# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli maintenance tests

"""Exercise cli maintenance behaviour through the public CLI."""

from __future__ import annotations

from unittest import mock

from tests.cli_test_support import run_cli


def test_benchmark_delegates_to_subprocess() -> None:
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = run_cli("benchmark")
    assert rc == 0
    m.assert_called_once()


def test_preflight_delegates_to_subprocess() -> None:
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = run_cli("preflight")
    assert rc == 0
    m.assert_called_once()
