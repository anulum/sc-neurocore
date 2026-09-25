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
        raise ValueError("a split needs at least two parts")
    if not all(math.isfinite(share) and share > 0 for share in shares):
        raise ValueError(f"every share must be positive and finite; got {dict(fractions)}")
    if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"the shares sum to {sum(shares)}, not 1")
    members: dict[str, list[int]] = {}
    for position, sample in enumerate(manifest.samples):
        if sample.split == source_split:
            members.setdefault(sample.group, []).append(position)
    if not members:
        raise ValueError(
            f"the manifest has no {source_split!r} samples; its splits are {manifest.splits()}"
        )
    if len(members) < len(names):
        raise ValueError(
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
        raise ValueError("the plan was drawn from another manifest")
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
    manifest with :func:`leaked_groups` before training on it.

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
        On another schema or a missing or unknown field.
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
        raise ValueError(f"a split plan has exactly the fields {sorted(keys)}")
    if data["schema"] != SPLIT_SCHEMA:
        raise ValueError(f"split schema {data['schema']!r} is not {SPLIT_SCHEMA!r}")
    return SplitPlan(
        manifest_digest=str(data["manifest_digest"]),
        source_split=str(data["source_split"]),
        seed=int(data["seed"]),
        fractions=tuple((str(name), float(share)) for name, share in data["fractions"]),
        assignment={
            str(name): tuple(int(position) for position in positions)
            for name, positions in data["assignment"].items()
        },
        groups={
            str(name): tuple(str(group) for group in groups)
            for name, groups in data["groups"].items()
        },
    )


__all__ = [
    "SPLIT_SCHEMA",
    "SplitPlan",
    "group_overlap",
    "group_split",
    "leaked_groups",
    "split_plan_from_dict",
]
