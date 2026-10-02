# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Splits that keep every group on one side

"""Divide a published split of an event dataset without leaking groups.

A validation score means something only when no group — a speaker, a
recording — contributes samples to both the data a model learned from and the
data it is judged on. :func:`group_split` divides one published split by
whole groups, deterministically from a seed, and records the plan with the
manifest it was drawn from. :func:`group_overlap` reports groups the
publisher's own splits share, which a user should know before comparing a
score with published ones.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sc_neurocore.datasets.refusals import DatasetRefusal

from .manifest import EventDatasetManifest

SPLIT_SCHEMA = "sc-neurocore.event-dataset-split.v1"


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Which samples of a manifest go to which split, by whole groups.

    Attributes
    ----------
    manifest_digest:
        Digest of the manifest the positions refer to.
    source_split:
        The published split that was divided.
    seed:
        Seed of the group order.
    fractions:
        Requested share of samples per split, in declaration order.
    assignment:
        Split name to the positions of its samples in ``manifest.samples``.
    groups:
        Split name to the groups it holds.
    """

    manifest_digest: str
    source_split: str
    seed: int
    fractions: tuple[tuple[str, float], ...]
    assignment: Mapping[str, tuple[int, ...]]
    groups: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form, with the schema identifier."""
        return {
            "schema": SPLIT_SCHEMA,
            "manifest_digest": self.manifest_digest,
            "source_split": self.source_split,
            "seed": self.seed,
            "fractions": [[name, share] for name, share in self.fractions],
            "assignment": {name: list(positions) for name, positions in self.assignment.items()},
            "groups": {name: list(groups) for name, groups in self.groups.items()},
        }

    @property
    def digest(self) -> str:
        """``sha256:`` over the canonical JSON form."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _group_order(groups: list[str], seed: int) -> list[str]:
    def key(group: str) -> str:
        return hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest()

    return sorted(groups, key=lambda group: (key(group), group))


def group_split(
    manifest: EventDatasetManifest,
    *,
    fractions: Mapping[str, float],
    source_split: str = "train",
    seed: int = 0,
) -> SplitPlan:
    """Divide one published split into new splits made of whole groups.

    Groups are taken in an order fixed by ``seed``; each goes to the split
    whose sample count is furthest below its requested share. The shares are
    therefore met as closely as whole groups allow, never by cutting a group.

    Parameters
    ----------
    manifest:
        The dataset manifest.
    fractions:
        Share of the source split's samples per new split; positive, summing
        to one.
    source_split:
        The published split to divide; other published splits are untouched.
    seed:
        Seed of the group order.

    Returns
    -------
    SplitPlan
        The plan, tied to the manifest's digest.

    Raises
    ------
    ValueError
        On shares that are not positive or do not sum to one, an unknown
        source split, or fewer groups than requested splits.
    """
    names = list(fractions)
    shares = [float(fractions[name]) for name in names]
    if len(names) < 2:
        raise DatasetRefusal("a split needs at least two parts")
    if not all(math.isfinite(share) and share > 0 for share in shares):
        raise DatasetRefusal(f"every share must be positive and finite; got {dict(fractions)}")
    if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetRefusal(f"the shares sum to {sum(shares)}, not 1")
    members: dict[str, list[int]] = {}
    for position, sample in enumerate(manifest.samples):
        if sample.split == source_split:
            members.setdefault(sample.group, []).append(position)
    if not members:
        raise DatasetRefusal(
            f"the manifest has no {source_split!r} samples; its splits are {manifest.splits()}"
        )
    if len(members) < len(names):
        raise DatasetRefusal(
            f"{len(members)} groups cannot fill {len(names)} splits without cutting a group"
        )
    total = sum(len(positions) for positions in members.values())
    counts = dict.fromkeys(names, 0)
    placed: dict[str, list[str]] = {name: [] for name in names}
    order = _group_order(list(members), seed)
    for index, group in enumerate(order):
        remaining = len(order) - index
        empty = [name for name in names if not placed[name]]
        if len(empty) == remaining:
            # Every split still without a group takes one of the last groups.
            target = empty[0]
        else:
            target = max(
                names,
                key=lambda name: (fractions[name] * total - counts[name], -names.index(name)),
            )
        placed[target].append(group)
        counts[target] += len(members[group])
    assignment = {
        name: tuple(sorted(position for group in placed[name] for position in members[group]))
        for name in names
    }
    return SplitPlan(
        manifest_digest=manifest.digest,
        source_split=source_split,
        seed=seed,
        fractions=tuple((name, float(fractions[name])) for name in names),
        assignment=assignment,
        groups={name: tuple(sorted(placed[name])) for name in names},
    )


def leaked_groups(manifest: EventDatasetManifest, plan: SplitPlan) -> tuple[str, ...]:
    """Return the groups that have samples in more than one split of a plan.

    Parameters
    ----------
    manifest:
        The manifest the plan was drawn from.
    plan:
        The plan to check.

    Returns
    -------
    tuple of str
        Leaked groups, sorted; empty for a sound plan.

    Raises
    ------
    ValueError
        When the plan was drawn from another manifest.
    """
    if plan.manifest_digest != manifest.digest:
        raise DatasetRefusal("the plan was drawn from another manifest")
    seen: dict[str, set[str]] = {}
    for name, positions in plan.assignment.items():
        for position in positions:
            seen.setdefault(manifest.samples[position].group, set()).add(name)
    return tuple(sorted(group for group, splits in seen.items() if len(splits) > 1))


def group_overlap(manifest: EventDatasetManifest) -> dict[str, tuple[str, ...]]:
    """Report groups that the publisher's own splits share.

    Parameters
    ----------
    manifest:
        The dataset manifest.

    Returns
    -------
    dict
        Each shared group to the published splits it appears in; empty when
        the published splits keep every group apart.
    """
    seen: dict[str, dict[str, None]] = {}
    for sample in manifest.samples:
        seen.setdefault(sample.group, {})[sample.split] = None
    return {group: tuple(splits) for group, splits in sorted(seen.items()) if len(splits) > 1}


def split_plan_from_dict(data: Mapping[str, Any]) -> SplitPlan:
    """Read a plan's JSON form, refusing anything it does not define.

    A plan read back is not trusted to be sound: check it against its
    manifest with :func:`validate_split_plan` before training on it.

    Parameters
    ----------
    data:
        The parsed JSON.

    Returns
    -------
    SplitPlan
        The plan.

    Raises
    ------
    ValueError
        On another schema, missing or unknown fields, invalid types, duplicate
        positions, or inconsistent split names. Values are never coerced.
    """
    keys = {
        "schema",
        "manifest_digest",
        "source_split",
        "seed",
        "fractions",
        "assignment",
        "groups",
    }
    if not isinstance(data, Mapping) or set(data) != keys:
        raise DatasetRefusal(f"a split plan has exactly the fields {sorted(keys)}")
    if data["schema"] != SPLIT_SCHEMA:
        raise DatasetRefusal(f"split schema {data['schema']!r} is not {SPLIT_SCHEMA!r}")
    digest = _split_text(data["manifest_digest"], "manifest_digest")
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise DatasetRefusal("manifest_digest must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in digest[7:]):
        raise DatasetRefusal("manifest_digest must be a sha256 digest")
    source = _split_text(data["source_split"], "source_split")
    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise DatasetRefusal("seed must be an integer in [0, 2**32)")
    fractions = _split_fractions(data["fractions"])
    names = {name for name, _ in fractions}
    assignments = _split_mapping(data["assignment"], names, "assignment")
    declared_groups = _split_mapping(data["groups"], names, "groups")
    assignment: dict[str, tuple[int, ...]] = {}
    groups: dict[str, tuple[str, ...]] = {}
    seen: set[int] = set()
    for name, _ in fractions:
        positions: list[int] = []
        for position in assignments[name]:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise DatasetRefusal("sample positions must be non-negative integers")
            if position in seen:
                raise DatasetRefusal("each sample position must appear exactly once")
            seen.add(position)
            positions.append(position)
        assignment[name] = tuple(positions)
        labels = tuple(_split_text(group, "group") for group in declared_groups[name])
        if len(set(labels)) != len(labels):
            raise DatasetRefusal("group declarations must not contain duplicates")
        groups[name] = labels
    return SplitPlan(digest, source, seed, fractions, assignment, groups)


def validate_split_plan(manifest: EventDatasetManifest, plan: SplitPlan) -> None:
    """Check a complete split's sample and group custody before training.

    Parameters
    ----------
    manifest:
        Manifest whose source split is being divided.
    plan:
        Imported or generated plan. Every source sample must occur once,
        every part must be non-empty, and declared groups must match samples.

    Raises
    ------
    ValueError
        If the schema, manifest digest, source split, sample membership,
        coverage, group declarations or separation is invalid.

    Notes
    -----
    Requested fractions are allocation goals, not an exact sample-count
    constraint: indivisible groups can prevent exact fraction matching.
    """
    checked = split_plan_from_dict(plan.to_dict())
    if checked.manifest_digest != manifest.digest:
        raise DatasetRefusal("the plan was drawn from another manifest")
    source = {
        position
        for position, sample in enumerate(manifest.samples)
        if sample.split == checked.source_split
    }
    if not source:
        raise DatasetRefusal("the manifest has no samples in the plan's source split")
    selected = {position for positions in checked.assignment.values() for position in positions}
    if selected != source:
        raise DatasetRefusal("the plan must contain every source sample once and no other samples")
    for name, positions in checked.assignment.items():
        if not positions:
            raise DatasetRefusal("every split must contain samples")
        actual = {manifest.samples[position].group for position in positions}
        if actual != set(checked.groups[name]):
            raise DatasetRefusal("declared groups must match the assigned samples")
    if leaked_groups(manifest, checked):
        raise DatasetRefusal("a sample group occurs in more than one split")


def _split_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise DatasetRefusal(f"{field} must be a non-empty string")
    return value


def _split_fractions(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise DatasetRefusal("fractions must declare at least two splits")
    result: list[tuple[str, float]] = []
    names: set[str] = set()
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise DatasetRefusal("each fraction must contain a split name and share")
        name = _split_text(pair[0], "split name")
        share = pair[1]
        if name in names:
            raise DatasetRefusal("split names must be unique")
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise DatasetRefusal("every share must be a positive finite number")
        try:
            number = float(share)
        except OverflowError as exc:
            raise DatasetRefusal("every share must be a positive finite number") from exc
        if not math.isfinite(number) or number <= 0:
            raise DatasetRefusal("every share must be a positive finite number")
        names.add(name)
        result.append((name, number))
    if not math.isclose(sum(share for _, share in result), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetRefusal("the shares must sum to one")
    return tuple(result)


def _split_mapping(value: object, names: set[str], field: str) -> dict[str, list[object]]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise DatasetRefusal(f"{field} must have exactly the declared split names")
    result: dict[str, list[object]] = {}
    for name in names:
        items = value[name]
        if not isinstance(items, list):
            raise DatasetRefusal(f"{field} entries must be lists")
        result[name] = items
    return result


__all__ = [
    "SPLIT_SCHEMA",
    "SplitPlan",
    "group_overlap",
    "group_split",
    "leaked_groups",
    "split_plan_from_dict",
    "validate_split_plan",
]
