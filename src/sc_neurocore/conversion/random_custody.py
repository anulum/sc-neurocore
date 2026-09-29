# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Caller random-generator custody while running source code

"""Run user source code without leaving any trace in the caller's random generators.

Conversion executes user ``forward`` code while tracing and calibrating. That code
may draw from Python's, NumPy's or PyTorch's global generators, on the CPU or on
an accelerator. The generators are process-global, so saving and restoring them
is only sound if no other conversion interleaves: every custody section in the
process is serialised by one reentrant lock.
"""

import random
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

_CUSTODY = threading.RLock()


@contextmanager
def preserved_random_state() -> Iterator[None]:
    """Restore every global generator the body may advance, even when it raises.

    Covers Python ``random``, NumPy's legacy global generator, the PyTorch CPU
    generator and the generator of every device of the current accelerator type
    (``torch.accelerator``), through ``torch.random.fork_rng``. Sections are
    serialised across threads; other code that draws concurrently outside a
    section is not isolated.

    Yields
    ------
    None
        Control while the caller's states are held.
    """
    with _CUSTODY:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        device_type = getattr(torch.accelerator.current_accelerator(), "type", "cuda")
        devices = range(torch.get_device_module(device_type).device_count())
        try:
            with torch.random.fork_rng(devices=devices, device_type=device_type):
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


__all__ = ["preserved_random_state"]
