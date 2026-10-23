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


def operate_dialog(owner_pid: int, action: str, destination: Path) -> str:
    """Save or cancel through the owned native dialog without sharing Qt's thread.

    Parameters
    ----------
    owner_pid : int
        Exact test application PID on its private X display.
    action : str
        Save to the supplied path or cancel with Escape.
    destination : pathlib.Path
        Actual filename selected through the real dialog.

    Returns
    -------
    str
        Identifier of the owned native window that received the input.

    Raises
    ------
    AssertionError
        The native window is absent or an X utility fails.
    subprocess.SubprocessError
        A bounded native control operation fails.
    """
    deadline = time.monotonic() + 40
    while True:
        found = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--name", "^Save Studio export$"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        assert found.returncode in (0, 1), found.stderr
        for identifier in found.stdout.split():
            owner = subprocess.run(
                ["xdotool", "getwindowpid", identifier],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if owner.stdout.strip() != str(owner_pid):
                continue
            try:
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
                return identifier
            except subprocess.SubprocessError:
                subprocess.run(
                    ["xdotool", "key", "--window", identifier, "Escape"],
                    capture_output=True,
                    timeout=3,
                )
                raise
        assert time.monotonic() < deadline, "The real save dialog did not appear."
        time.sleep(0.05)


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
