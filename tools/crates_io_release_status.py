# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — crates.io release status of the engine crate

"""Report whether the engine crate version in a Cargo manifest is already on crates.io.

The publish workflow uploads the crate only when this reports ``unpublished``,
and a dispatch without publishing runs the same query, so the registry path is
proven before a tag needs it. crates.io refuses clients that do not identify
themselves, so every query names this project and its contact URL.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import tomllib
import urllib.error
import urllib.request

CRATES_API = "https://crates.io/api/v1/crates"
USER_AGENT = "sc-neurocore-release (+https://github.com/anulum/sc-neurocore)"
_BODY_LIMIT = 4096


def crate_identity(manifest: Path) -> tuple[str, str]:
    """Read the literal package name and version from a Cargo manifest.

    Parameters
    ----------
    manifest : Path
        Cargo.toml of the crate to publish.

    Returns
    -------
    tuple of str
        Package name and version.

    Raises
    ------
    ValueError
        The manifest has no package table or no literal name and version, for
        example a version inherited from a workspace.
    """
    package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package")
    name = package.get("name") if isinstance(package, dict) else None
    version = package.get("version") if isinstance(package, dict) else None
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError(f"{manifest} names no literal package name and version.")
    return name, version


def release_status(name: str, version: str, api: str = CRATES_API, timeout: float = 30.0) -> str:
    """Ask the registry whether one crate version exists.

    Parameters
    ----------
    name : str
        Crate name.
    version : str
        Exact crate version.
    api : str, optional
        Base address of the crates API.
    timeout : float, optional
        Seconds to wait for the connection and each read before failing.

    Returns
    -------
    str
        ``published`` for HTTP 200 and ``unpublished`` for HTTP 404.

    Raises
    ------
    RuntimeError
        The registry answered with any other status; the message keeps the
        status and the start of the response body.
    OSError
        The registry could not be reached.
    """
    request = urllib.request.Request(f"{api}/{name}/{version}", headers={"User-Agent": USER_AGENT})
    try:
        # The address is the fixed crates API or an explicit test or mirror base.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            status = response.status
    except urllib.error.HTTPError as error:
        with error:
            if error.code == 404:
                return "unpublished"
            body = error.read(_BODY_LIMIT).decode("utf-8", "replace")
        raise RuntimeError(f"crates.io answered HTTP {error.code} for {name} {version}: {body}")
    if status != 200:
        raise RuntimeError(f"crates.io answered HTTP {status} for {name} {version}.")
    return "published"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the release status of the crate a manifest describes.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments by default.

    Returns
    -------
    int
        Zero after printing ``published`` or ``unpublished``; one after a
        manifest or registry failure, which is reported on standard error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api", default=CRATES_API)
    args = parser.parse_args(argv)
    try:
        name, version = crate_identity(args.manifest)
        status = release_status(name, version, args.api)
    except (OSError, RuntimeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"crates.io release status unavailable: {error}", file=sys.stderr)
        return 1
    print(status)
    print(f"{name} {version} is {status} on crates.io.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
