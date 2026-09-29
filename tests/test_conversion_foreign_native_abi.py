# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Foreign native IF library refusal acceptance

"""Refuse real foreign libraries that break the ABI-one ownership contract.

The maintained Rust, Go and Mojo libraries satisfy the contract by construction,
so they cannot exercise the guards that protect against a library that does
not. Each case here compiles a genuine C shared library with the system
compiler that breaks exactly one clause -- the version it reports, the status
or handle it publishes, or the result view it lends -- and configures it as
the owner would. The public model call must refuse it with the named error;
no Python function, status or pointer is substituted.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_LIBRARY = r"""
#include <stddef.h>
#include <stdint.h>

typedef struct { void *data; size_t length; } view_t;
static double storage[64];

uint32_t sc_if_abi_version(void) { return SC_ABI; }

int32_t sc_if_replay(const void *request, void **handle) {
    (void)request;
    *handle = SC_NULL_HANDLE ? NULL : (void *)storage;
    return SC_STATUS;
}

int32_t sc_if_buffer(void *handle, uint32_t kind, size_t index, view_t *view) {
    (void)handle; (void)kind; (void)index;
    view->data = SC_NULL_DATA ? NULL : (void *)storage;
    view->length = SC_LENGTH;
    return SC_BUFFER_STATUS;
}

void sc_if_free(void *handle) { (void)handle; }
"""

_DEFAULTS = {
    "SC_ABI": "1",
    "SC_STATUS": "0",
    "SC_NULL_HANDLE": "0",
    "SC_BUFFER_STATUS": "0",
    "SC_NULL_DATA": "0",
    "SC_LENGTH": "1",
}

_PROBE = r"""
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False
    )
    measurement.start()
from sc_neurocore.conversion import ConvertedSNN
try:
    ConvertedSNN([[[1.0]]], [None], [1.0], T=7).run([1.0], input_mode="constant", backend="rust")
except RuntimeError as error:
    print("refused:", error, flush=True)
else:
    raise AssertionError("foreign native library accepted")
finally:
    if measurement is not None:
        measurement.stop()
        measurement.save()
"""


@pytest.mark.parametrize(
    "variant,breach,message",
    [
        ("abi-two", {"SC_ABI": "2"}, "incompatible ownership ABI"),
        ("unknown-status", {"SC_STATUS": "7"}, "failed to publish an owned result"),
        ("null-handle", {"SC_NULL_HANDLE": "1"}, "failed to publish an owned result"),
        ("buffer-refused", {"SC_BUFFER_STATUS": "1"}, "result buffer unavailable"),
        ("wrong-length", {"SC_LENGTH": "999"}, "result dimensions are invalid"),
        ("null-data", {"SC_NULL_DATA": "1"}, "result dimensions are invalid"),
    ],
)
def test_public_run_refuses_a_foreign_library_breaking_one_clause(
    tmp_path: Path, variant: str, breach: dict[str, str], message: str
) -> None:
    """Configure a real nonconforming library and require the exact public refusal."""
    source = tmp_path / "foreign.c"
    source.write_text(_LIBRARY)
    library = tmp_path / f"{variant}.so"
    defines = [f"-D{name}={value}" for name, value in {**_DEFAULTS, **breach}.items()]
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Werror",
            *defines,
            "-o",
            str(library),
            str(source),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    parent_data = os.environ.get("COVERAGE_FILE", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            f"{parent_data}-foreign-{variant}" if parent_data else "",
            str(ROOT / "src/sc_neurocore/conversion"),
        ],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            SC_NEUROCORE_IF_RUST_LIB=str(library),
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert message in result.stdout, result.stdout
