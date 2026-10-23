# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop renderer network confinement integration

"""Exercise the shipped interface through actual WebEngine request and page policies."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, Qt, QTimer, QUrl
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.desktop.webview import DesktopPage, DesktopRequestPolicy

pytestmark = pytest.mark.usefixtures("application")


@pytest.fixture
def url_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Register an isolated real XDG URL handler that fetches the selected local URL.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable owner of the MIME registry and received HTTP response.
    monkeypatch : pytest.MonkeyPatch
        Scope XDG selection to this test process and its actual OS children.

    Returns
    -------
    pathlib.Path
        Response file written by the real curl process launched through xdg-open.
    """
    curl = shutil.which("curl")
    assert curl is not None, "The Linux desktop integration lane requires curl."
    config = tmp_path / "url-config"
    applications = tmp_path / "url-data" / "applications"
    config.mkdir()
    applications.mkdir(parents=True)
    response = tmp_path / "external-response.json"
    command = (
        f"{curl} --disable --noproxy 127.0.0.1 --max-time 5 --connect-timeout 2 "
        f"--silent --show-error --fail --output {response} %u"
    )
    (applications / "studio-test-url.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Studio URL integration receiver\n"
        "MimeType=x-scheme-handler/http;\nTerminal=false\nExec=" + command + "\n"
    )
    (config / "mimeapps.list").write_text(
        "[Default Applications]\nx-scheme-handler/http=studio-test-url.desktop\n"
    )
    for key, value in {
        "XDG_CONFIG_HOME": config,
        "XDG_DATA_HOME": applications.parent,
        "XDG_CONFIG_DIRS": tmp_path / "system-config",
        "XDG_DATA_DIRS": tmp_path / "system-data",
    }.items():
        monkeypatch.setenv(key, str(value))
    selected = subprocess.run(
        ["xdg-mime", "query", "default", "x-scheme-handler/http"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert selected.stdout.strip() == "studio-test-url.desktop"
    return response


@pytest.fixture
def receiver() -> Iterator[tuple[str, list[str]]]:
    """Observe actual HTTP arrivals at a different, CORS-permissive loopback origin.

    Yields
    ------
    tuple[str, list[str]]
        Live network origin and the paths received by its HTTP server.
    """
    received: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Record a network arrival and permit the browser to read its response."""
            received.append(self.path)
            body = b'{"receiver":true}'
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            """Keep the dedicated received-path list as the network observation.

            Parameters
            ----------
            _format : str
                Standard-library access-log format.
            *args : object
                Access-log values, already represented by the received request.
            """

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", received
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def _script(page: QWebEnginePage, source: str) -> object:
    loop = QEventLoop()
    timeout = QTimer(loop)
    timeout.setSingleShot(True)
    poll = QTimer(loop)
    poll.setSingleShot(True)
    result: list[object] = []

    def receive(value: object) -> None:
        decoded: object = json.loads(value) if isinstance(value, str) and value else None
        if decoded is not None:
            result.append(decoded)
            loop.quit()
        else:
            poll.start(25)

    poll.timeout.connect(lambda: page.runJavaScript("JSON.stringify((" + source + "))", receive))
    timeout.timeout.connect(loop.quit)
    poll.start(0)
    timeout.start(90_000)
    loop.exec()
    poll.stop()
    timeout.stop()
    assert result, "The real renderer did not finish the requested observation."
    return result[0]


def _arrival(received: list[str], target: str, response: Path) -> None:
    loop = QEventLoop()
    poll = QTimer(loop)
    deadline = QTimer(loop)
    deadline.setSingleShot(True)

    def complete() -> bool:
        return (
            target in received
            and response.is_file()
            and response.read_bytes() == b'{"receiver":true}'
        )

    poll.timeout.connect(lambda: loop.quit() if complete() else None)
    deadline.timeout.connect(loop.quit)
    poll.start(25)
    deadline.start(5_000)
    if not complete():
        loop.exec()
    poll.stop()
    deadline.stop()
    assert complete(), "The selected URL did not reach the real XDG handler."


def test_renderer_confines_traffic_and_routes_user_selected_links(
    desktop_backend: DesktopBackend, receiver: tuple[str, list[str]], url_handler: Path
) -> None:
    """Confine actual renderer traffic and dispatch user-selected links through XDG.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh owner of the actual installed HTTP backend and project profile.
    receiver : tuple[str, list[str]]
        Separately listening HTTP server that permits cross-origin reads.
    url_handler : pathlib.Path
        Response destination of the test user's actual XDG URL application.
    """
    external, received = receiver
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(external + "/control", timeout=5) as response:
        assert json.load(response) == {"receiver": True}
    assert received == ["/control"]
    control = subprocess.run(
        ["xdg-open", external + "/os-control"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert control.returncode == 0, control.stdout + control.stderr
    _arrival(received, "/os-control", url_handler)
    controls = ["/control", "/os-control"]
    assert received == controls
    instance = desktop_backend.start()
    profile = QWebEngineProfile()
    policy = DesktopRequestPolicy(instance.origin, profile)
    profile.setUrlRequestInterceptor(policy)
    view = QWebEngineView()
    page = DesktopPage(profile, instance.origin, view)
    view.setPage(page)
    view.resize(1440, 960)
    view.show()
    try:
        view.load(QUrl(instance.origin + "/studios/sc-neurocore/"))
        assert (
            _script(
                page,
                "document.readyState === 'complete' && "
                "document.querySelector('input[aria-label=\"Search models\"]') ? true : null",
            )
            is True
        )
        assert page.webChannel() is None
        assert (
            _script(
                page,
                "(() => { window.desktopPolicyResult = null; (async () => {"
                "const socket = url => new Promise(resolve => {"
                " const ws = new WebSocket(url);"
                " ws.onopen = () => { ws.close(); resolve('open'); };"
                " ws.onerror = () => resolve('refused'); });"
                "const health = await (await fetch('/api/health')).json();"
                "const foreign = await fetch("
                + json.dumps(external + "/blocked-http")
                + ").then(() => 'arrived', () => 'refused');"
                "const ownSocket = await socket(location.origin.replace('http:', 'ws:') + '/ws/progress');"
                "const foreignSocket = await socket("
                + json.dumps(external.replace("http:", "ws:") + "/blocked-ws")
                + "); window.desktopPolicyResult = {health, foreign, ownSocket, foreignSocket};"
                "})().catch(error => { window.desktopPolicyResult = {error: String(error)}; });"
                "return true; })()",
            )
            is True
        )
        assert _script(page, "window.desktopPolicyResult") == {
            "health": {"status": "ok"},
            "foreign": "refused",
            "ownSocket": "open",
            "foreignSocket": "refused",
        }
        assert received == controls
        assert page.url().toString() == instance.origin + "/studios/sc-neurocore/"
        navigation = QSignalSpy(view.loadFinished)
        view.load(QUrl(external + "/automatic-navigation"))
        assert navigation.count() or navigation.wait(5_000)
        assert navigation.at(0) == [False]
        assert page.url().toString() == instance.origin + "/studios/sc-neurocore/"
        assert received == controls
        popup_urls: list[str] = []
        page.newWindowRequested.connect(
            lambda request: popup_urls.append(request.requestedUrl().toString())
        )
        assert _script(page, "(() => { window.open(location.href); return true; })()") is True
        assert popup_urls == [page.url().toString()]
        assert page.url().toString() == instance.origin + "/studios/sc-neurocore/"
        assert received == controls
        for target, path in [("_self", "/user-link"), ("_blank", "/user-popup")]:
            url_handler.unlink()
            position = _script(
                page,
                "(() => { document.getElementById('desktop-policy-link')?.remove();"
                "const a = document.createElement('a'); a.id = 'desktop-policy-link';"
                "a.style.cssText = 'position:fixed;left:10px;top:10px;width:200px;height:40px;z-index:2147483647';"
                "a.textContent = 'External link'; a.href = "
                + json.dumps(external + path)
                + "; a.target = "
                + json.dumps(target)
                + "; document.body.append(a);"
                "a.addEventListener('click', event => { window.desktopPolicyClick = event.isTrusted; });"
                "window.desktopPolicyClick = null; const r = a.getBoundingClientRect();"
                "return {x: r.x + r.width/2, y: r.y + r.height/2}; })()",
            )
            assert isinstance(position, dict)
            x, y = position["x"], position["y"]
            assert isinstance(x, (int, float)) and isinstance(y, (int, float))
            QTest.mouseClick(
                view.focusProxy() or view, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(y))
            )
            assert _script(page, "window.desktopPolicyClick") is True
            _arrival(received, path, url_handler)
            controls.append(path)
            assert received == controls
            assert page.url().toString() == instance.origin + "/studios/sc-neurocore/"
    finally:
        view.close()
        view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
