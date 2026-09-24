# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — versioned storage operation refusal tests

"""Keep operation routing unambiguous before authority handlers run."""

import pytest

from sc_neurocore.studio.platform.storage_operation import classify_storage_operation


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (b'{"schema_version":"studio.storage.record.v2","operation":"record"}', "record"),
        (
            b'{"schema_version":"studio.storage.admission.v1","operation":"admit_named"}',
            "admit_named",
        ),
        (b'{"schema_version":"studio.storage.supervision.v1","operation":"start"}', "supervision"),
        (
            b'{"schema_version":"studio.storage.supervision.v1","operation":"heartbeat"}',
            "supervision",
        ),
        (b'{"schema_version":"studio.storage.finish.v1","operation":"finish"}', "finish"),
        (b'{"schema_version":"studio.storage.query.v1","operation":"query"}', "query"),
        (b'{"schema_version":"studio.storage.cancel.v1","operation":"cancel"}', "cancel"),
        (b'{"schema_version":"studio.storage.artifact.v1","operation":"artifact"}', "artifact"),
        (b'{"schema_version":"studio.storage.purge.v1","operation":"purge"}', "purge"),
    ],
)
def test_exact_versioned_operations_dispatch(metadata: bytes, expected: str) -> None:
    """Only the reviewed operation and version pairs select a handler."""
    assert classify_storage_operation(metadata) == expected


@pytest.mark.parametrize(
    "metadata,reason",
    [
        (b"\xff", "invalid storage operation JSON"),
        (b"{", "invalid storage operation JSON"),
        (b'[{"operation":"record"}]', "storage operation must be an object"),
        (
            b'{"schema_version":"studio.storage.record.v2","operation":"admit_named"}',
            "unsupported storage operation",
        ),
        (
            b'{"schema_version":"studio.storage.record.v1","operation":"record"}',
            "unsupported storage operation",
        ),
        (
            b'{"schema_version":"studio.storage.supervision.v1","operation":"complete"}',
            "unsupported storage operation",
        ),
        (
            b'{"schema_version":"studio.storage.finish.v1","operation":"start"}',
            "unsupported storage operation",
        ),
        (
            b'{"schema_version":"studio.storage.supervision.v1","operation":"record"}',
            "unsupported storage operation",
        ),
        (
            b'{"schema_version":"studio.storage.record.v2","operation":"record","operation":"admit_named"}',
            "duplicate storage operation field",
        ),
        (
            b'{"schema_version":"studio.storage.record.v2","operation":"record","payload":{"a":1,"a":2}}',
            "duplicate storage operation field",
        ),
        (
            b'{"schema_version":"studio.storage.record.v2","operation":"record","payload":NaN}',
            "nonfinite storage operation constant",
        ),
    ],
)
def test_ambiguous_or_unreviewed_operations_fail_before_routing(
    metadata: bytes, reason: str
) -> None:
    """UTF-8, JSON, key and version ambiguity never select authority work."""
    with pytest.raises(ValueError, match=reason):
        classify_storage_operation(metadata)
