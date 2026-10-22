# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec package works without the packages the base distribution leaves out

"""The waveform codecs import and round-trip with every package excluded from the wheel blocked."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = """
import importlib.abc
import sys

BLOCKED = {blocked!r}


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ModuleNotFoundError(f"No module named {{name!r}} (not in the base distribution)")
        return None


sys.meta_path.insert(0, Block())
import numpy as np

from sc_neurocore.spike_codec import LosslessStreamDecoder, LosslessStreamEncoder, WaveformCodec
from sc_neurocore.spike_codec.spike_events import SpikeEventDecoder, SpikeEventEncoder

x = (np.arange(400).reshape(200, 2) % 50 * 64).astype(np.int16)
for backend in ("python", "auto"):
    packet = LosslessStreamEncoder(2, 16, 1, backend).encode(x)
    assert (LosslessStreamDecoder(2, 16, 1, backend).decode(packet, 200) == x).all()
    stream, _ = WaveformCodec(mode="lossless", backend=backend).compress(x)
    assert (WaveformCodec.decompress(stream, backend=backend).waveform == x).all()
    events = SpikeEventEncoder(2, 19531.0, backend=backend).encode(x)
    assert SpikeEventDecoder(2, 20, backend=backend).decode(events, 200).events.shape == (10, 2)
print("ok")
"""


def test_codecs_work_without_packages_excluded_from_the_base_distribution() -> None:
    """Importing the codec package must not need a package the wheel excludes (e.g. world_model)."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["packages"]["find"]["exclude"]
    blocked = sorted({pattern.removesuffix(".*") for pattern in patterns})
    assert "sc_neurocore.world_model" in blocked
    script = textwrap.dedent(_SCRIPT.format(blocked=blocked))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]
