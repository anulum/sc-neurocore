# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop page navigation and request policy

"""Keep the desktop renderer on its owned backend without a native web bridge."""

from __future__ import annotations

from PySide6.QtCore import QObject, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import (
    QWebEngineNewWindowRequest,
    QWebEnginePage,
    QWebEngineProfile,
    QWebEngineUrlRequestInfo,
    QWebEngineUrlRequestInterceptor,
)


def _owned_url(url: QUrl, origin: QUrl, *, websocket: bool = False) -> bool:
    """Match an origin without accepting URL credentials or a different port."""
    scheme = "ws" if websocket else origin.scheme()
    return (
        url.scheme() == scheme
        and url.host() == origin.host()
        and url.port() == origin.port()
        and not url.userName()
        and not url.password()
    )


class DesktopRequestPolicy(QWebEngineUrlRequestInterceptor):
    """Admit only owned HTTP/WebSocket traffic and local renderer resources."""

    def __init__(self, origin: str, parent: QObject) -> None:
        """Bind a renderer to its admitted backend origin.

        Parameters
        ----------
        origin : str
            Exact origin returned by desktop backend admission.
        parent : PySide6.QtCore.QObject
            Qt owner retaining the interceptor for the profile lifetime.
        """
        super().__init__(parent)
        self._origin = QUrl(origin)
        self._blob_prefix = "blob:" + origin + "/"

    def interceptRequest(self, info: QWebEngineUrlRequestInfo) -> None:
        """Block remote/file requests while permitting owned API and UI resources.

        Parameters
        ----------
        info : PySide6.QtWebEngineCore.QWebEngineUrlRequestInfo
            Actual renderer request, modified in place when refused.
        """
        url = info.requestUrl()
        admitted = (
            _owned_url(url, self._origin)
            or _owned_url(url, self._origin, websocket=True)
            or url.scheme() in {"data", "qrc", "about"}
            or url.toString().startswith(self._blob_prefix)
        )
        info.block(not admitted)


class DesktopPage(QWebEnginePage):
    """Keep application navigation local; user-selected external links use the OS."""

    def __init__(self, profile: QWebEngineProfile, origin: str, parent: QObject) -> None:
        """Use one persistent profile without exposing any native web channel.

        Parameters
        ----------
        profile : PySide6.QtWebEngineCore.QWebEngineProfile
            User-specific persistent browser profile.
        origin : str
            Exact admitted application origin.
        parent : PySide6.QtCore.QObject
            Qt owner of the page.
        """
        super().__init__(profile, parent)
        self._origin = QUrl(origin)
        self.newWindowRequested.connect(self._new_window)

    def acceptNavigationRequest(
        self, url: QUrl | str, navigation_type: QWebEnginePage.NavigationType, is_main_frame: bool
    ) -> bool:
        """Admit owned navigation and open only user-selected HTTP links externally.

        Parameters
        ----------
        url : PySide6.QtCore.QUrl or str
            Requested navigation target.
        navigation_type : PySide6.QtWebEngineCore.QWebEnginePage.NavigationType
            Qt's actual navigation cause, including user link activation.
        is_main_frame : bool
            Whether the request would replace the application document.

        Returns
        -------
        bool
            Whether this page may navigate to the target.
        """
        url = QUrl(url)
        if _owned_url(url, self._origin):
            return True
        if (
            is_main_frame
            and navigation_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked
            and url.scheme() in {"http", "https"}
            and not url.userName()
            and not url.password()
        ):
            QDesktopServices.openUrl(url)
        return False

    def _new_window(self, request: QWebEngineNewWindowRequest) -> None:
        """Allow an external user link without granting automatic popup windows."""
        url = request.requestedUrl()
        if (
            request.isUserInitiated()
            and url.scheme() in {"http", "https"}
            and not _owned_url(url, self._origin)
            and not url.userName()
            and not url.password()
        ):
            QDesktopServices.openUrl(url)
