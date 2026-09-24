#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Accept an installed Studio through its own launcher

"""Launch an installed Studio the way a user does and accept what it serves.

A wheel can pass every packaging check and still hand a user an empty page: the
interface missing from the package, mounted where its assets do not resolve, or
served from a checkout that happened to sit beside the installation. This tool
takes the interpreter of an installation, checks that its ``sc_neurocore``
lives outside the source checkout, starts ``sc-neurocore studio`` through that
interpreter with no desktop browser, and then requires over HTTP:

* the root redirects to the interface's published path;
* the interface page loads, and every asset it names under that path is served;
* the model catalogue lists exactly the models the installed registry maps.

The server is stopped afterwards however the checks end. The exit status is 0
only when every check held; the report says which one failed and why.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

UI_PATH = "/studios/sc-neurocore/"
_ASSET = re.compile(r'(?:src|href)="(/studios/sc-neurocore/[^"]+)"')
_LOCATE = (
    "import json, sc_neurocore\n"
    "from sc_neurocore.neurons.models import _CLASS_TO_MODULE\n"
    "print(json.dumps({'package': sc_neurocore.__file__, 'catalogue': len(_CLASS_TO_MODULE)}))\n"
)


def installation_environment() -> dict[str, str]:
    """Return the environment an installation runs in, as a user starts it.

    ``PYTHONPATH`` and ``PYTHONHOME`` of the caller are dropped: a caller that
    points them at a source tree would otherwise have the installation import
    that tree, and the acceptance would examine something no user runs.
    ``BROWSER`` names the command :mod:`webbrowser` runs; ``true`` opens nothing.
    """
    kept = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    return {**kept, "BROWSER": "true"}


class AcceptanceFailure(RuntimeError):
    """An installed Studio did not serve what a user needs."""


@dataclass(frozen=True)
class Response:
    """One HTTP exchange, without following redirects."""

    status: int
    headers: dict[str, str]
    body: bytes


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def fetch(url: str, *, timeout: float = 10.0) -> Response:
    """GET ``url`` without following redirects.

    Raises
    ------
    urllib.error.URLError
        When nothing answers at ``url``.
    """
    try:
        with _OPENER.open(url, timeout=timeout) as reply:
            return Response(
                reply.status, {k.lower(): v for k, v in reply.headers.items()}, reply.read()
            )
    except urllib.error.HTTPError as reply:
        return Response(reply.code, {k.lower(): v for k, v in reply.headers.items()}, reply.read())


def asset_paths(index_html: str) -> list[str]:
    """Return every asset the interface page loads from its published path, in order."""
    return list(dict.fromkeys(_ASSET.findall(index_html)))


def locate_installation(python: str, *, checkout: Path) -> dict[str, object]:
    """Return where ``python`` imports ``sc_neurocore`` from and its catalogue size.

    Raises
    ------
    AcceptanceFailure
        When the package does not import, or imports from ``checkout``.
    """
    with tempfile.TemporaryDirectory() as neutral:
        located = subprocess.run(
            [python, "-c", _LOCATE],
            capture_output=True,
            cwd=neutral,
            env=installation_environment(),
            text=True,
            timeout=120,
            check=False,
        )
    if located.returncode != 0:
        raise AcceptanceFailure(f"sc_neurocore does not import: {located.stderr.strip()}")
    found: dict[str, object] = json.loads(located.stdout)
    package = Path(str(found["package"])).resolve()
    if package.is_relative_to(checkout.resolve()):
        raise AcceptanceFailure(
            f"sc_neurocore imports from the checkout ({package}), not an installation"
        )
    return found


def check_served(origin: str, *, catalogue: int) -> dict[str, object]:
    """Require the root redirect, the interface, its assets and the whole catalogue.

    Raises
    ------
    AcceptanceFailure
        Naming the first check that did not hold.
    """
    root = fetch(f"{origin}/")
    if root.status not in (302, 307) or root.headers.get("location") != UI_PATH:
        raise AcceptanceFailure(
            f"the root answered {root.status} to {root.headers.get('location')!r}, not a redirect to {UI_PATH}"
        )
    page = fetch(f"{origin}{UI_PATH}")
    assets = asset_paths(page.body.decode("utf-8", "replace")) if page.status == 200 else []
    if not assets:
        raise AcceptanceFailure(
            f"the interface page answered {page.status} and names no asset under {UI_PATH}"
        )
    for asset in assets:
        served = fetch(f"{origin}{asset}")
        if served.status != 200 or not served.body:
            raise AcceptanceFailure(f"the asset {asset} answered {served.status}")
    models = fetch(f"{origin}/api/models")
    listed = len(json.loads(models.body)) if models.status == 200 else None
    if listed != catalogue:
        raise AcceptanceFailure(
            f"the catalogue listed {listed} models; the installed registry maps {catalogue}"
        )
    return {"assets": len(assets), "models": listed}


def wait_until_listening(origin: str, server: subprocess.Popen[str], *, timeout: float) -> None:
    """Return once the server answers its health route.

    Raises
    ------
    AcceptanceFailure
        When the server exits first or does not answer within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise AcceptanceFailure(
                f"the Studio exited with status {server.returncode} before listening"
            )
        try:
            if fetch(f"{origin}/api/health", timeout=2).status == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.5)
    raise AcceptanceFailure(f"the Studio did not answer within {timeout:g} s")


def accept(python: str, *, port: int, checkout: Path, timeout: float = 120.0) -> dict[str, object]:
    """Launch the installed Studio through its CLI and run every check.

    Returns
    -------
    dict
        The installation's package path and what was served.

    Raises
    ------
    AcceptanceFailure
        When a check does not hold.
    """
    found = locate_installation(python, checkout=checkout)
    origin = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as neutral:
        server = subprocess.Popen(
            [python, "-m", "sc_neurocore.cli", "studio", "--port", str(port)],
            cwd=neutral,
            env=installation_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            wait_until_listening(origin, server, timeout=timeout)
            served = check_served(origin, catalogue=int(str(found["catalogue"])))
        finally:
            server.terminate()
            server.wait(timeout=30)
    return {"package": found["package"], **served}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the acceptance and print its report as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True, help="interpreter of the installation to accept")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source checkout the installation must not import from",
    )
    args = parser.parse_args(argv)
    try:
        report = accept(args.python, port=args.port, checkout=args.checkout)
    except AcceptanceFailure as failure:
        print(json.dumps({"accepted": False, "reason": str(failure)}))
        return 1
    print(json.dumps({"accepted": True, **report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
