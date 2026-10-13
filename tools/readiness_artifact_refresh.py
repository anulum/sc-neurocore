# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Refresh receipt-derived catalogue artefacts

"""Render and replace the three catalogue views derived from facet receipts."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path

from sc_neurocore.studio.model_catalogue import readiness_seal_payload
from sc_neurocore.studio.readiness_seal import SEAL_PATH, render_seal
from tools.model_profile_ledger import DEFAULT_OUTPUT as PROFILE_OUTPUT
from tools.model_profile_ledger import render_ledger as render_profiles
from tools.readiness_evidence_ledger import DEFAULT_OUTPUT as READINESS_OUTPUT
from tools.readiness_evidence_ledger import render_ledger as render_readiness

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReadinessArtifactError(RuntimeError):
    """A receipt's derived catalogue artefacts could not be published."""


def write_receipt_license(receipt: Path) -> Path:
    """Create the ownership sidecar of a newly recorded receipt.

    Parameters
    ----------
    receipt : Path
        New immutable JSON receipt produced by the existing recorder.

    Returns
    -------
    Path
        Adjacent ownership sidecar, created without overwriting another file.

    Raises
    ------
    ReadinessArtifactError
        If the sidecar cannot be created. The receipt remains intact.
    """
    sidecar = receipt.with_name(receipt.name + ".license")
    text = (
        "SPDX-License-Identifier: AGPL-3.0-or-later\n"
        "Commercial license available\n"
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.\n"
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.\n"
        "ORCID: 0009-0009-3560-0851\n"
        "Contact: www.anulum.li | protoscience@anulum.li\n"
        f"SC-NeuroCore — {receipt.name}\n"
        "SPDX-FileCopyrightText: 1996-2026 Miroslav Šotek <protoscience@anulum.li> (concepts)\n"
        "SPDX-FileCopyrightText: 2020-2026 Miroslav Šotek <protoscience@anulum.li> (code)\n"
    )
    try:
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ReadinessArtifactError("could not create the receipt ownership sidecar") from error
    return sidecar


def refresh_readiness_artifacts() -> tuple[Path, Path, Path]:
    """Refresh the profile ledger, evidence ledger and installed Studio seal.

    All payloads are rendered and staged before any existing output is replaced.
    Each replacement is atomic; this is not a transaction across three files.

    Returns
    -------
    tuple of Path
        Profile ledger, readiness evidence ledger and Studio seal paths.

    Raises
    ------
    ReadinessArtifactError
        If rendering, staging or replacement fails. A published receipt is
        retained, and the caller must report failure rather than success.
    """
    targets = (REPO_ROOT / PROFILE_OUTPUT, REPO_ROOT / READINESS_OUTPUT, SEAL_PATH)
    try:
        for target in targets:
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ReadinessArtifactError("a readiness artefact target is not a regular file")
        payloads = (render_profiles(), render_readiness(), render_seal(readiness_seal_payload()))
        with ExitStack() as cleanup:
            pending: list[Path] = []
            for target, payload in zip(targets, payloads):
                mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=target.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    cleanup.callback(temporary.unlink, missing_ok=True)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(mode)
                pending.append(temporary)
            for temporary, target in zip(pending, targets):
                os.replace(temporary, target)
    except (OSError, ValueError) as error:
        raise ReadinessArtifactError(
            "could not refresh the receipt-derived readiness artefacts"
        ) from error
    return targets
