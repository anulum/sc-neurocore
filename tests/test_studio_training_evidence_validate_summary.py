# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (validate_summary) from former test_studio_training_evidence.py

from __future__ import annotations

from tests.studio_training_evidence_support import *  # noqa: F403

def test_validate_training_evidence_summary_accepts_verified_summary() -> None:
    """Evidence summary validator accepts verified Training Monitor summaries."""

    summary = build_training_evidence_summary(_training_record(), _training_payload_reader())
    assert isinstance(summary, dict)
    summary["duration_seconds"] = 1.5

    validated = validate_training_evidence_summary(summary)

    assert validated == summary


def test_validate_training_evidence_summary_rejects_unavailable_summary() -> None:
    """Evidence summary validator rejects bounded unavailable summaries."""

    with pytest.raises(ValueError, match="action_kind"):
        validate_training_evidence_summary(_unavailable_training_summary())


def test_validate_training_evidence_summary_rejects_forged_artifact_path() -> None:
    """Evidence summary validator rejects non-confined artifact metadata."""

    summary = build_training_evidence_summary(_training_record(), _training_payload_reader())
    assert isinstance(summary, dict)
    evidence_artifact = summary["evidence_artifact"]
    assert isinstance(evidence_artifact, dict)
    evidence_artifact["relative_path"] = "../training/evidence.json"

    with pytest.raises(ValueError, match="path"):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        (lambda payload: payload.__setitem__("schema_version", "studio.old.v1"), "schema"),
        (lambda payload: payload.__setitem__("action_kind", "studio.compile"), "action"),
        (
            lambda payload: payload.__setitem__("evidence_classification", "compile"),
            "classification",
        ),
        (lambda payload: payload.__setitem__("payload_sha256", "bad"), "payload digest"),
        (lambda payload: payload.__setitem__("result_artifacts", None), "result artifacts"),
        (lambda payload: payload.__setitem__("replay_route", ""), "replay_route"),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_fields(
    mutator: Callable[[dict[str, object]], None],
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed top-level fields."""

    summary = _valid_training_summary()
    mutator(summary)

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("evidence_artifact", "error_match"),
    [
        (None, "evidence_artifact"),
        ({"relative_path": "training/evidence.txt", "sha256": "0" * 64, "size_bytes": 128}, "path"),
        (
            {"relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH, "sha256": "bad", "size_bytes": 128},
            "digest",
        ),
        (
            {
                "relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH,
                "sha256": "0" * 64,
                "size_bytes": -1,
            },
            "size",
        ),
        (
            {
                "relative_path": TRAINING_EVIDENCE_ARTIFACT_PATH,
                "sha256": "0" * 64,
                "size_bytes": 128,
                1: "bad",
            },
            "must be JSON",
        ),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_evidence_artifact(
    evidence_artifact: object,
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed evidence artifact metadata."""

    summary = _valid_training_summary()
    summary["evidence_artifact"] = evidence_artifact

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("result_artifact", "error_match"),
    [
        ("bad", "result_artifacts"),
        ({"relative_path": "/training/status.json", "sha256": "2" * 64, "size_bytes": 256}, "path"),
        ({"relative_path": "training/status.json", "sha256": "bad", "size_bytes": 256}, "digest"),
        ({"relative_path": "training/status.json", "sha256": "2" * 64, "size_bytes": -1}, "size"),
    ],
)
def test_validate_training_evidence_summary_rejects_invalid_result_artifact(
    result_artifact: object,
    error_match: str,
) -> None:
    """Evidence summary validator rejects malformed result artifact metadata."""

    summary = _valid_training_summary()
    summary["result_artifacts"] = [result_artifact]

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(summary)


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        ({"schema_version": float("nan")}, "must be JSON"),
        ({1: "bad"}, "must be JSON"),
        (
            {"schema_version": TRAINING_EVIDENCE_SUMMARY_SCHEMA_VERSION, "bad": object()},
            "must be JSON",
        ),
    ],
)
def test_validate_training_evidence_summary_rejects_non_portable_json(
    payload: Mapping[object, object],
    error_match: str,
) -> None:
    """Evidence summary validator rejects non-portable JSON payloads."""

    with pytest.raises(ValueError, match=error_match):
        validate_training_evidence_summary(cast(Mapping[str, object], payload))


