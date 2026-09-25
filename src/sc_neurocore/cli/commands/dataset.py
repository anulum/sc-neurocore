# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Record, verify and split event datasets from the command line

"""Record which event-dataset files a study used, check them, and split them by group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_MANIFEST_DATASETS = ("nmnist", "shd", "dvs_cifar10")


def add_dataset_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``dataset manifest``, ``dataset verify`` and ``dataset split``.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "dataset",
        help="Record, verify and split event datasets",
        description=(
            "Record the files and samples of an event dataset you hold, check a directory "
            "against that record, and divide a published split by whole groups. Nothing is "
            "downloaded."
        ),
    )
    actions = parser.add_subparsers(dest="dataset_action", metavar="ACTION", required=True)

    manifest = actions.add_parser(
        "manifest",
        help="Write a manifest of the dataset files under a directory",
        description="Hash every file and record every sample with its split, label and group.",
    )
    manifest.add_argument("name", choices=_MANIFEST_DATASETS, help="Dataset")
    manifest.add_argument("root", help="Directory in the layout the dataset's loader reads")
    manifest.add_argument(
        "--version",
        required=True,
        dest="data_version",
        help="The release of the data you hold, as its publisher names it",
    )
    manifest.add_argument("--output", "-o", required=True, help="Manifest JSON to write")
    manifest.set_defaults(handler=run_dataset_manifest)

    verify = actions.add_parser(
        "verify",
        help="Check a directory against a manifest",
        description="List missing, changed and unlisted files. Exit status 1 when any exist.",
    )
    verify.add_argument("manifest", help="Manifest JSON")
    verify.add_argument("root", help="Directory holding the dataset")
    verify.set_defaults(handler=run_dataset_verify)

    split = actions.add_parser(
        "split",
        help="Divide a published split by whole groups",
        description=(
            "Assign whole groups (speakers, recordings) to new splits so that no group is on "
            "both sides. Example: --part train=0.8 --part validation=0.2"
        ),
    )
    split.add_argument("manifest", help="Manifest JSON")
    split.add_argument(
        "--part",
        action="append",
        required=True,
        metavar="NAME=SHARE",
        help="A new split and its share of samples; give two or more",
    )
    split.add_argument("--source", default="train", help="Published split to divide")
    split.add_argument("--seed", type=int, default=0, help="Seed of the group order")
    split.add_argument("--output", "-o", required=True, help="Split plan JSON to write")
    split.set_defaults(handler=run_dataset_split)


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run_dataset_manifest(args: argparse.Namespace) -> int:
    """Write the manifest of a dataset directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``dataset manifest`` arguments.

    Returns
    -------
    int
        Zero on success, two when the directory or the version is refused.
    """
    from sc_neurocore.datasets.manifest import build_manifest
    from sc_neurocore.datasets.splits import group_overlap

    try:
        manifest = build_manifest(args.name, args.root, version=args.data_version)
        _write_json(args.output, manifest.to_dict())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    print(f"{manifest.dataset.title} {manifest.version}: {len(manifest.files)} files")
    for split in manifest.splits():
        samples = [sample for sample in manifest.samples if sample.split == split]
        groups = {sample.group for sample in samples}
        print(f"  {split}: {len(samples)} samples in {len(groups)} groups")
    print(f"  licence: {manifest.dataset.licence} ({manifest.dataset.licence_url})")
    print(f"  cite: {manifest.dataset.citation}, doi:{manifest.dataset.doi}")
    shared = group_overlap(manifest)
    if shared:
        print(f"  published splits share {len(shared)} groups: {', '.join(shared)}")
    print(f"  manifest: {args.output} ({manifest.digest})")
    return 0


def run_dataset_verify(args: argparse.Namespace) -> int:
    """Compare a directory with a manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``dataset verify`` arguments.

    Returns
    -------
    int
        Zero when the directory holds exactly the manifest's bytes, one when it
        differs, two when the manifest cannot be read.
    """
    from sc_neurocore.datasets.manifest import manifest_from_dict, verify_manifest

    try:
        manifest = manifest_from_dict(_read_json(args.manifest))
        report = verify_manifest(manifest, args.root)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    for label, paths in (
        ("missing", report.missing),
        ("changed", report.changed),
        ("not in the manifest", report.unexpected),
    ):
        for path in paths:
            print(f"{label}: {path}")
    if report.ok:
        print(f"{len(manifest.files)} files match {manifest.digest}")
        return 0
    return 1


def _parts(values: list[str]) -> dict[str, float]:
    parts: dict[str, float] = {}
    for value in values:
        name, separator, share = value.partition("=")
        if not separator or not name or name in parts:
            raise ValueError(f"--part must be a new NAME=SHARE; got {value!r}")
        parts[name] = float(share)
    return parts


def run_dataset_split(args: argparse.Namespace) -> int:
    """Divide a published split by whole groups and write the plan.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``dataset split`` arguments.

    Returns
    -------
    int
        Zero on success, two when the manifest or the request is refused.
    """
    from sc_neurocore.datasets.manifest import manifest_from_dict
    from sc_neurocore.datasets.splits import group_split

    try:
        manifest = manifest_from_dict(_read_json(args.manifest))
        plan = group_split(
            manifest, fractions=_parts(args.part), source_split=args.source, seed=args.seed
        )
        _write_json(args.output, plan.to_dict())
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    total = sum(len(positions) for positions in plan.assignment.values())
    for name, positions in plan.assignment.items():
        print(
            f"{name}: {len(positions)} samples ({len(positions) / total:.1%}) "
            f"in {len(plan.groups[name])} groups"
        )
    print(f"plan: {args.output} ({plan.digest})")
    return 0
