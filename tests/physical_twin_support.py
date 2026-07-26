# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared physical-twin test support

"""Socket stand-in for physical-twin TCP contract tests."""


class _FakeSocket:
    """Minimal ``socket.create_connection`` context-manager stand-in."""

    def __init__(self, reply_lines):  # type: ignore[no-untyped-def] # Preserved legacy helper AST
        self._reply_lines = reply_lines
        self.sent: list[bytes] = []

    def __enter__(self):  # type: ignore[no-untyped-def] # Preserved legacy helper AST
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def] # Preserved legacy helper AST
        return False

    def sendall(self, data):  # type: ignore[no-untyped-def] # Preserved legacy helper AST
        self.sent.append(data)

    def makefile(self, mode, encoding):  # type: ignore[no-untyped-def] # Preserved legacy helper AST
        assert mode == "r"
        assert encoding == "utf-8"
        return iter(self._reply_lines)
