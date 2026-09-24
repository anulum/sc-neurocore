# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The installed-Studio acceptance refuses what a user could not use

"""Each refusal of the release acceptance, against real processes and servers.

The accepted path runs in ``tests/test_studio_distribution.py`` on a built wheel.
Here every way an installation can fail a user is reproduced for real: an
environment that cannot import the package, one that imports the checkout, a
server that dies or never listens, and Studio frontends mounted by the Studio's
own code with a piece missing. Those servers carry a health route and nothing
else of the API, which is the missing piece each case needs.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import sysconfig
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("fastapi")

_REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    path = _REPO / "tools" / "studio_installed_acceptance.py"
    spec = importlib.util.spec_from_file_location("studio_installed_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _environment(root: Path, *paths: Path) -> Path:
    """A virtual environment whose ``sys.path`` gains exactly ``paths``."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=120
    )
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    (Path(purelib) / "acceptance.pth").write_text(
        "".join(f"{path}\n" for path in paths), encoding="utf-8"
    )
    return python


_SERVER = textwrap.dedent(
    """
    import sys
    import uvicorn
    from fastapi import FastAPI
    from sc_neurocore.studio.api.frontend import mount_studio_frontend

    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    mount_studio_frontend(app, app_module_file=sys.argv[1])
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
    """
)


@pytest.fixture
def frontend(tmp_path: Path) -> Iterator[tuple[Path, Callable[[], str]]]:
    """Serve whatever frontend a case writes, mounted by the Studio's own code.

    Yields the build directory, empty, and a function that starts the server
    and returns its origin: the mount is decided when the server starts, so a
    case writes its build first.
    """
    dist = tmp_path / "studio" / "frontend" / "dist"
    dist.mkdir(parents=True)
    anchor = tmp_path / "one" / "two" / "three" / "four" / "app.py"
    anchor.parent.mkdir(parents=True)
    port = _free_port()
    started: list[subprocess.Popen[str]] = []

    def start() -> str:
        server = subprocess.Popen(
            [sys.executable, "-c", _SERVER, str(anchor), str(port)], text=True
        )
        started.append(server)
        origin = f"http://127.0.0.1:{port}"
        tool.wait_until_listening(origin, server, timeout=60)
        return origin

    yield dist, start
    for server in started:
        server.terminate()
        server.wait(timeout=30)


Frontend = tuple[Path, Callable[[], str]]


def test_asset_paths_keeps_the_published_path_once_each_in_order() -> None:
    page = (
        '<script src="/studios/sc-neurocore/assets/b.js"></script>'
        '<link href="/studios/sc-neurocore/assets/a.css">'
        '<script src="/studios/sc-neurocore/assets/b.js"></script>'
        '<link href="data:image/svg+xml,x"><script src="/elsewhere/c.js"></script>'
    )
    assert tool.asset_paths(page) == [
        "/studios/sc-neurocore/assets/b.js",
        "/studios/sc-neurocore/assets/a.css",
    ]


def test_an_environment_without_the_package_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller's PYTHONPATH on the checkout does not lend the installation a package."""
    monkeypatch.setenv("PYTHONPATH", str(_REPO / "src"))
    python = _environment(tmp_path / "empty")
    with pytest.raises(tool.AcceptanceFailure, match="sc_neurocore does not import"):
        tool.locate_installation(str(python), checkout=_REPO)
    assert tool.main(["--python", str(python)]) == 1


def test_an_environment_importing_the_checkout_is_refused(tmp_path: Path) -> None:
    python = _environment(tmp_path / "checkout", _REPO / "src", Path(sysconfig.get_path("purelib")))
    with pytest.raises(tool.AcceptanceFailure, match="imports from the checkout"):
        tool.locate_installation(str(python), checkout=_REPO)


def test_a_studio_without_an_interface_does_not_redirect(frontend: Frontend) -> None:
    _dist, start = frontend
    with pytest.raises(tool.AcceptanceFailure, match="the root answered 404 to None"):
        tool.check_served(start(), catalogue=1)


def test_an_interface_page_naming_no_asset_is_refused(frontend: Frontend) -> None:
    dist, start = frontend
    (dist / "index.html").write_text("<html>no assets</html>", encoding="utf-8")
    with pytest.raises(tool.AcceptanceFailure, match="answered 200 and names no asset"):
        tool.check_served(start(), catalogue=1)


def test_an_asset_the_page_names_but_the_server_lacks_is_refused(frontend: Frontend) -> None:
    dist, start = frontend
    (dist / "index.html").write_text(
        '<script src="/studios/sc-neurocore/assets/missing.js"></script>', encoding="utf-8"
    )
    with pytest.raises(
        tool.AcceptanceFailure, match="asset /studios/sc-neurocore/assets/missing.js answered 404"
    ):
        tool.check_served(start(), catalogue=1)


def test_a_catalogue_the_server_does_not_list_is_refused(frontend: Frontend) -> None:
    dist, start = frontend
    (dist / "assets").mkdir()
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "index.html").write_text(
        '<script src="/studios/sc-neurocore/assets/app.js"></script>', encoding="utf-8"
    )
    with pytest.raises(
        tool.AcceptanceFailure, match="listed None models; the installed registry maps 185"
    ):
        tool.check_served(start(), catalogue=185)


def test_a_server_that_exits_before_listening_is_refused() -> None:
    server = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], text=True)
    with pytest.raises(tool.AcceptanceFailure, match="exited with status 3 before listening"):
        tool.wait_until_listening(f"http://127.0.0.1:{_free_port()}", server, timeout=60)


def test_a_server_that_never_answers_is_refused() -> None:
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], text=True)
    try:
        with pytest.raises(tool.AcceptanceFailure, match="did not answer within 1 s"):
            tool.wait_until_listening(f"http://127.0.0.1:{_free_port()}", server, timeout=1)
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_a_server_whose_health_route_fails_is_refused(tmp_path: Path) -> None:
    """Something else listening on the port answers, but not as a Studio."""
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        origin = f"http://127.0.0.1:{port}"
        with pytest.raises(tool.AcceptanceFailure, match="did not answer within 3 s"):
            tool.wait_until_listening(origin, server, timeout=3)
        assert tool.fetch(f"{origin}/api/health").status == 404
    finally:
        server.terminate()
        server.wait(timeout=30)
