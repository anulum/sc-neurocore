# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent native save-dialog test control

"""Send real X events only to the specified test process's export dialog."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def owned_dialogs(owner_pid: int) -> list[str]:
    """List the visible export dialogs that belong to the test application.

    Parameters
    ----------
    owner_pid : int
        Exact test application PID on its private X display.

    Returns
    -------
    list of str
        Identifiers of the owned visible ``Save Studio export`` windows.

    Raises
    ------
    AssertionError
        The X window search itself fails.
    """
    found = subprocess.run(
        ["xdotool", "search", "--onlyvisible", "--name", "^Save Studio export$"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert found.returncode in (0, 1), found.stderr
    owned = []
    for identifier in found.stdout.split():
        owner = subprocess.run(
            ["xdotool", "getwindowpid", identifier],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if owner.stdout.strip() == str(owner_pid):
            owned.append(identifier)
    return owned


def operate_dialog(owner_pid: int, action: str, destination: Path, attempts: int = 3) -> str:
    """Save or cancel through the owned native dialog without sharing Qt's thread.

    Window activation can complete before keyboard focus reaches the dialog,
    so input is confirmed by the dialog closing and repeated when it remains
    open. A dialog that never accepts the input is cancelled, because an open
    modal dialog would otherwise block the test application indefinitely.

    Parameters
    ----------
    owner_pid : int
        Exact test application PID on its private X display.
    action : str
        Save to the supplied path or cancel with Escape.
    destination : pathlib.Path
        Actual filename selected through the real dialog.
    attempts : int
        Complete input sequences sent before the dialog is cancelled.

    Returns
    -------
    str
        Identifier of the owned native window that received the input.

    Raises
    ------
    AssertionError
        The native window is absent, an X utility fails or the dialog stays open.
    subprocess.SubprocessError
        A bounded native control operation fails.
    """
    deadline = time.monotonic() + 40
    while not (dialogs := owned_dialogs(owner_pid)):
        assert time.monotonic() < deadline, "The real save dialog did not appear."
        time.sleep(0.05)
    identifier = dialogs[0]
    try:
        for _attempt in range(attempts):
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", identifier], check=True, timeout=5
            )
            if action == "save":
                for keys in ["ctrl+l", "ctrl+a"]:
                    subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", keys], check=True, timeout=5
                    )
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--delay", "1", str(destination)],
                    check=True,
                    timeout=5,
                )
            subprocess.run(
                [
                    "xdotool",
                    "key",
                    "--clearmodifiers",
                    "Return" if action == "save" else "Escape",
                ],
                check=True,
                timeout=5,
            )
            closed = time.monotonic() + 10
            while identifier in owned_dialogs(owner_pid):
                if time.monotonic() >= closed:
                    break
                time.sleep(0.05)
            else:
                return identifier
    except subprocess.SubprocessError:
        subprocess.run(
            ["xdotool", "key", "--window", identifier, "Escape"],
            capture_output=True,
            timeout=3,
        )
        raise
    for command in (
        ["xdotool", "windowactivate", "--sync", identifier],
        ["xdotool", "key", "--clearmodifiers", "Escape"],
    ):
        subprocess.run(command, capture_output=True, timeout=5)
    raise AssertionError("The real save dialog stayed open after every native input.")


def main() -> int:
    """Run one bounded native-dialog action and persist its actual owner/window receipt.

    Returns
    -------
    int
        Zero after completed native input; failure otherwise propagates normally.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner_pid", type=int)
    parser.add_argument("action", choices=("save", "cancel"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    identifier = operate_dialog(args.owner_pid, args.action, args.destination)
    args.receipt.write_text(
        json.dumps({"window": identifier, "owner_pid": args.owner_pid, "action": args.action})
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
