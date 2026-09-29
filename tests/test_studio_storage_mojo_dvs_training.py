# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo DVS isolated event training acceptance

"""Exercise operator-selected Mojo DVS training through actual isolated worker generations."""

from pathlib import Path

from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.test_accel_mojo_dvs import mojo_dvs_executable as mojo_dvs_executable
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_native_event_training import _compare_isolated_training


def test_mojo_dvs_isolated_checkpoint_matches_numpy(
    base: Path, tmp_path: Path, authority: Authority, mojo_dvs_executable: Path
) -> None:
    """Real worker training preserves full checkpoint state and refuses a corrupted runtime."""
    _compare_isolated_training(
        base, tmp_path, authority, ("mojo", mojo_dvs_executable), dataset="dvs_cifar10"
    )
