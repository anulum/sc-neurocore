# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual embedded interpreter record probe

"""Run inside the selected embedded interpreter and verify its real installed files."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import re
import sys
import sysconfig
from pathlib import Path


def probe_runtime() -> dict[str, object]:
    """Read actual standalone runtime identity and check every installed RECORD.

    Returns
    -------
    dict
        Actual prefix, interpreter, platform, installation scheme and
        the complete installed distribution/RECORD inventory. No package is
        imported to obtain its version and no runtime file is modified.

    Raises
    ------
    ValueError
        The prefix is external, an installed declaration is invalid, or any
        declared file, byte length or content hash differs.
    OSError
        An actual installed record or payload cannot be read.
    """
    root = Path(sys.prefix).resolve()
    if Path(sys.base_prefix).resolve() != root:
        raise ValueError("The desktop runtime depends on an external Python prefix.")
    scheme = {
        name: sysconfig.get_path(source)
        for name, source in {
            "purelib": "purelib",
            "platlib": "platlib",
            "scripts": "scripts",
            "headers": "include",
            "data": "data",
        }.items()
    }
    if any(not Path(path).resolve().is_relative_to(root) for path in scheme.values()):
        raise ValueError("An installation scheme escapes the embedded runtime.")
    packages: dict[str, str] = {}
    records: dict[str, str] = {}
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for distribution in metadata.distributions():
        # https://packaging.python.org/en/latest/specifications/name-normalization/
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        files = distribution.files
        if name in packages or files is None:
            raise ValueError("An installed distribution is duplicate or lacks a RECORD.")
        candidates = [p for p in files if str(p).endswith(".dist-info/RECORD")]
        if len(candidates) != 1:
            raise ValueError("An installed distribution must have exactly one RECORD.")
        record = Path(str(distribution.locate_file(candidates[0]))).resolve(strict=True)
        if not record.is_relative_to(root):
            raise ValueError("An installed RECORD escapes the embedded runtime.")
        raw = record.read_bytes()
        checked = 0
        paths: set[str] = set()
        for filename, encoded, size in csv.reader(io.StringIO(raw.decode("utf-8"))):
            if filename in paths:
                raise ValueError("An installed RECORD repeats a path.")
            paths.add(filename)
            path = Path(str(distribution.locate_file(filename))).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("An installed record path escapes or is not a file.")
            if encoded:
                algorithm, expected = encoded.split("=", 1)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, algorithm).digest()
                actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                if actual != expected or path.stat().st_size != int(size):
                    raise ValueError("Installed RECORD content differs: " + filename)
            checked += 1
        packages[name] = distribution.version
        records[name], counts[name] = str(record), checked
        digests[name] = hashlib.sha256(raw).hexdigest()
    return {
        "prefix": str(root),
        "python_version": sys.version.split()[0],
        "platform": sysconfig.get_platform(),
        "scheme": scheme,
        "packages": packages,
        "records": records,
        "record_entries": counts,
        "record_sha256": digests,
    }


def main() -> int:
    """Print the actual embedded runtime inventory for its supervising producer.

    Returns
    -------
    int
        Zero after complete real inventory verification. Invalid runtime/files
        raise with the original diagnostic and a nonzero process exit.
    """
    print(json.dumps(probe_runtime(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
