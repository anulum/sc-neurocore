# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sealed artefact chunk reception

"""Receive actual socket frames under independent client content and frame limits."""

import hashlib
import os
import socket
import time

import pytest

from sc_neurocore.studio.platform.storage_artifact_chunks import receive_artifact_chunks
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactUnavailable
from sc_neurocore.studio.platform.storage_finish_protocol import FinishArtifact
from sc_neurocore.studio.platform.storage_peer import write_verified_frame


@pytest.mark.parametrize("size", [0, 64, 129])
def test_chunked_reads_preserve_empty_full_and_partial_content(size: int) -> None:
    """Exact chunks and a final remainder reproduce the declared bytes."""
    content = bytes(range(size))
    artifact = FinishArtifact(
        relative_path="weights.bin", size_bytes=size, sha256=hashlib.sha256(content).hexdigest()
    )
    sender, receiver = socket.socketpair()
    deadline = time.monotonic() + 5
    with sender, receiver:
        for offset in range(0, size, 64):
            write_verified_frame(
                sender,
                content[offset : offset + 64],
                expected_uid=os.getuid(),
                max_bytes=64,
                deadline=deadline,
            )
        assert (
            receive_artifact_chunks(
                receiver,
                artifact,
                expected_service_uid=os.getuid(),
                frame_max_bytes=64,
                max_artifact_bytes=1024,
                deadline=deadline,
            )
            == content
        )


@pytest.mark.parametrize("fault", ["frame", "budget", "oversize", "short", "digest"])
def test_bad_budgets_and_untrusted_content_never_reach_the_caller(fault: str) -> None:
    """Client bounds precede content reads; chunk shape and complete digest are checked."""
    artifact = FinishArtifact(relative_path="weights.bin", size_bytes=7, sha256="0" * 64)
    sender, receiver = socket.socketpair()
    deadline = time.monotonic() + 5
    with sender, receiver:
        if fault in {"short", "digest"}:
            write_verified_frame(
                sender,
                b"x" if fault == "short" else b"altered",
                expected_uid=os.getuid(),
                max_bytes=64,
                deadline=deadline,
            )
        with pytest.raises(
            ValueError if fault in {"frame", "budget"} else StudioJobArtifactUnavailable
        ):
            receive_artifact_chunks(
                receiver,
                artifact,
                expected_service_uid=os.getuid(),
                frame_max_bytes=0 if fault == "frame" else 64,
                max_artifact_bytes=0 if fault == "budget" else (1 if fault == "oversize" else 1024),
                deadline=deadline,
            )
