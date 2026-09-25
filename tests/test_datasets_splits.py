# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Group splits keep every speaker or recording on one side

"""No group straddles two splits, the plan is reproducible, and a tampered plan is caught."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.datasets.manifest import EventDatasetManifest, build_manifest
from sc_neurocore.datasets.splits import (
    SPLIT_SCHEMA,
    group_overlap,
    group_split,
    leaked_groups,
    split_plan_from_dict,
)
from tests.event_dataset_support import write_nmnist, write_shd

# Ten speakers with uneven sample counts, as real recordings have.
TRAIN_SPEAKERS = [speaker for speaker in range(10) for _ in range(3 + speaker % 4)]


@pytest.fixture
def shd(tmp_path: Path) -> EventDatasetManifest:
    write_shd(tmp_path, {"train": TRAIN_SPEAKERS, "test": [0, 10, 11]})
    return build_manifest("shd", tmp_path, version="1.0")


def test_a_split_by_speaker_puts_every_speaker_on_one_side(shd: EventDatasetManifest) -> None:
    plan = group_split(shd, fractions={"train": 0.8, "validation": 0.2}, seed=3)

    train, validation = set(plan.groups["train"]), set(plan.groups["validation"])
    assert train.isdisjoint(validation)
    assert train | validation == {f"speaker:{speaker}" for speaker in range(10)}
    for name, positions in plan.assignment.items():
        assert {shd.samples[p].group for p in positions} == set(plan.groups[name])
        assert all(shd.samples[p].split == "train" for p in positions)
    # Every source sample is placed exactly once; the published test split is untouched.
    placed = sorted(p for positions in plan.assignment.values() for p in positions)
    assert placed == list(range(len(TRAIN_SPEAKERS)))
    assert leaked_groups(shd, plan) == ()
    # Whole speakers cannot hit 20 % exactly; the plan is as close as one speaker allows.
    share = len(plan.assignment["validation"]) / len(TRAIN_SPEAKERS)
    assert abs(share - 0.2) <= 6 / len(TRAIN_SPEAKERS)


def test_the_same_seed_gives_the_same_plan_and_another_seed_may_not(
    shd: EventDatasetManifest,
) -> None:
    fractions = {"train": 0.7, "validation": 0.3}
    first = group_split(shd, fractions=fractions, seed=1)
    assert group_split(shd, fractions=fractions, seed=1) == first
    others = {group_split(shd, fractions=fractions, seed=seed).digest for seed in range(2, 8)}
    assert others - {first.digest}


def test_every_part_gets_a_group_even_when_its_share_is_small(tmp_path: Path) -> None:
    write_nmnist(tmp_path, {"train": {0: 3}})
    manifest = build_manifest("nmnist", tmp_path, version="1")
    plan = group_split(manifest, fractions={"a": 0.98, "b": 0.01, "c": 0.01}, seed=0)
    assert all(len(groups) == 1 for groups in plan.groups.values())


def test_the_publishers_shared_groups_are_reported(shd: EventDatasetManifest) -> None:
    assert group_overlap(shd) == {"speaker:0": ("train", "test")}


@pytest.mark.parametrize(
    ("fractions", "source", "message"),
    [
        ({"train": 1.0}, "train", "at least two parts"),
        ({"train": 0.9, "validation": 0.2}, "train", "sum to"),
        ({"train": 1.2, "validation": -0.2}, "train", "positive and finite"),
        ({"train": float("nan"), "validation": 0.5}, "train", "positive and finite"),
        ({"train": 0.5, "validation": 0.5}, "valid", "has no 'valid' samples"),
    ],
)
def test_an_impossible_request_is_refused(
    shd: EventDatasetManifest, fractions: dict[str, float], source: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        group_split(shd, fractions=fractions, source_split=source)


def test_more_parts_than_groups_is_refused(shd: EventDatasetManifest) -> None:
    with pytest.raises(ValueError, match="3 groups cannot fill 4 splits"):
        group_split(
            shd, fractions={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, source_split="test"
        )


def test_a_plan_read_back_is_checked_against_its_manifest(
    shd: EventDatasetManifest, tmp_path: Path
) -> None:
    plan = group_split(shd, fractions={"train": 0.8, "validation": 0.2}, seed=0)
    data: dict[str, Any] = json.loads(json.dumps(plan.to_dict()))
    assert split_plan_from_dict(data) == plan

    # Moving one sample of a training speaker into validation leaks that speaker.
    moved = data["assignment"]["train"].pop(0)
    data["assignment"]["validation"].append(moved)
    tampered = split_plan_from_dict(data)
    assert leaked_groups(shd, tampered) == (shd.samples[moved].group,)

    write_shd(tmp_path / "other", {"train": [1, 2]})
    other = build_manifest("shd", tmp_path / "other", version="1.0")
    with pytest.raises(ValueError, match="another manifest"):
        leaked_groups(other, plan)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"schema": "other.v0"}, "is not 'sc-neurocore.event-dataset-split.v1'"),
        ({"extra": 1}, "exactly the fields"),
    ],
)
def test_a_plan_that_is_not_exactly_this_schema_is_refused(
    shd: EventDatasetManifest, edit: dict[str, object], message: str
) -> None:
    data = group_split(shd, fractions={"train": 0.5, "validation": 0.5}).to_dict()
    assert data["schema"] == SPLIT_SCHEMA
    data.update(edit)
    with pytest.raises(ValueError, match=message):
        split_plan_from_dict(data)
