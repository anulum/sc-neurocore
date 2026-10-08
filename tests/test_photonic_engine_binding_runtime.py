# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed photonic worker refusal and retained formulas

"""Exercise photonic public admission, serial boundaries and recovery formulas."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.engine_requirement import installed_engine_origins, require_engine

require_engine()


@pytest.mark.parametrize("operation", ["route", "channels", "power", "pairs"])
def test_photonic_worker_refusal_and_formula_recovery(operation: str) -> None:
    """Preserve input columns and recover the real photonic operation after refusal.

    Parameters
    ----------
    operation
        Public routing, WDM, path power or geometric pair calculation.
    """
    program = """
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import cast
import numpy as np
import sc_neurocore_engine as engine
import sc_neurocore_engine.sc_neurocore_engine as native
origins = cast(dict[str, dict[str, str]], json.loads(sys.argv[1]))
for module in (engine, native):
    filename = module.__file__
    assert isinstance(filename, str)
    path = Path(filename).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    assert str(path) == origins[module.__name__]['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == origins[module.__name__]['sha256']
engine.set_num_threads(0)
pressure = sys.platform == 'linux'
operation = sys.argv[2]
import sc_neurocore_engine.photonics as photonics
filename = photonics.__file__
assert isinstance(filename, str)
facade_path = Path(filename).resolve()
assert facade_path.is_relative_to(Path(sys.prefix).resolve())
facade_sha = hashlib.sha256(facade_path.read_bytes()).hexdigest()
assert engine.py_ph_route_waveguides is native.py_ph_route_waveguides
assert engine.py_ph_analyze_crosstalk is native.py_ph_analyze_crosstalk
assert engine.py_ph_analyze_power_budget is native.py_ph_analyze_power_budget
assert engine.py_ph_analyze_crosstalk_pairs is native.py_ph_analyze_crosstalk_pairs
for n in (0, 1):
    assert native.py_ph_route_waveguides([], n)['n_segments'] == 0
assert native.py_ph_analyze_crosstalk([0], [], [1.], [0.])['channel_ids'] == []
assert native.py_ph_analyze_power_budget([0], [], [1.])['n_paths'] == 0
assert native.py_ph_analyze_crosstalk_pairs([], [], [], [])['num_pairs'] == 0
np.testing.assert_array_equal(native.py_ph_mzi_transfer_matrix(0.), [1., 0., 0., 0., 0., 0., 1., 0.])
np.testing.assert_array_equal(native.py_ph_cascade_mzi([]), [1., 0., 0., 0., 0., 0., 1., 0.])
assert native.py_ph_analyze_crosstalk_bank(1, 200., 10.)['num_pairs'] == 0
adjacency = [0., 1., 0., 0., 0., 2., 1., 0.]
ids = [0, 1, 2, 99]
wavelengths = [1550., 1551., 1560.]
bandwidths = [1., 1., 1., 0.]
powers = [0., 1., 2., 99.]
sources, targets, losses = [0, 1, 99], [1, 2], [2., 21., 99.]
pairs_a, pairs_b, gaps, lengths = [0, 1, 2], [1, 2, 3], [200., 300., 400.], [10., 20., 0.]
before = json.dumps([adjacency, ids, wavelengths, bandwidths, powers, sources, targets, losses,
                     pairs_a, pairs_b, gaps, lengths])
if pressure:
    import resource
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    virtual = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmSize:'))) * 1024
    previous = resource.getrlimit(resource.RLIMIT_AS)
    limit = virtual + 1024**2
    assert previous[1] == resource.RLIM_INFINITY or limit <= previous[1]
    resource.setrlimit(resource.RLIMIT_AS, (limit, previous[1]))
error: MemoryError | None = None
try:
    for n, short in ((3, adjacency[:7]), (sys.maxsize, [])):
        try:
            native.py_ph_route_waveguides(short, n)
        except ValueError as caught:
            assert str(caught) == 'adjacency storage must cover every off-diagonal entry'
        else:
            raise AssertionError('truncated or unrepresentable adjacency was accepted')
    try:
        native.py_ph_analyze_crosstalk_pairs([0], [], [200.], [10.])
    except ValueError as caught:
        assert str(caught) == 'pairs_a, pairs_b, gaps_nm, lengths_um must be equal length'
    else:
        raise AssertionError('unequal pair columns were accepted')
    if pressure:
        if operation == 'route':
            native.py_ph_route_waveguides(adjacency, 3)
        elif operation == 'channels':
            native.py_ph_analyze_crosstalk(ids, wavelengths, bandwidths, powers)
        elif operation == 'power':
            native.py_ph_analyze_power_budget(sources, targets, losses)
        else:
            native.py_ph_analyze_crosstalk_pairs(pairs_a, pairs_b, gaps, lengths)
except MemoryError as caught:
    error = caught
finally:
    if pressure:
        resource.setrlimit(resource.RLIMIT_AS, previous)
if pressure:
    assert error is not None
    assert str(error) == 'parallel execution resources cannot be allocated'
if operation == 'route':
    result = native.py_ph_route_waveguides(adjacency, 3)
    assert result['sources'] == [0, 0, 1] and result['targets'] == [1, 2, 2]
    assert result['crossings'] == [0, 0, 1] and result['n_segments'] == 3
    np.testing.assert_allclose(result['lengths_um'], [250., 250., 500.], rtol=0., atol=0.)
    np.testing.assert_allclose(result['losses_db'], [0.05, 0.05, 0.18], rtol=1e-14, atol=0.)
elif operation == 'channels':
    result = photonics.get_crosstalk_analyzer()(ids, wavelengths, bandwidths, powers)
    expected_adj = [sum(other != own and abs(wavelengths[i] - wavelengths[j]) < bandwidths[i] * 3.
                        for j, other in enumerate(ids[:3])) for i, own in enumerate(ids[:3])]
    expected_xt = -25. + 10. * np.log10(np.maximum(expected_adj, 1))
    assert result['channel_ids'] == ids[:3] and result['n_adjacent'] == expected_adj
    np.testing.assert_allclose(result['crosstalk_db'], expected_xt, rtol=1e-14)
    np.testing.assert_allclose(result['osnr_db'], np.array(powers[:3]) - expected_xt, rtol=1e-14)
elif operation == 'power':
    result = native.py_ph_analyze_power_budget(sources, targets, losses)
    assert result['n_paths'] == 2 and result['passed'] == [True, False]
    np.testing.assert_array_equal(result['total_losses_db'], losses[:2])
    np.testing.assert_array_equal(result['margins_db'], [18., -1.])
else:
    result = photonics.get_crosstalk_pair_analyzer()(pairs_a, pairs_b, gaps, lengths)
    assert result['pair_a'] == pairs_a and result['pair_b'] == pairs_b and result['num_pairs'] == 3
    decay = 1550. / (2. * np.pi * np.sqrt(3.48**2 - 1.45**2))
    kappa = np.pi * 0.1 * np.exp(-np.array(gaps) / decay) / 1.55
    ratios = np.sin(kappa * np.array(lengths)) ** 2
    isolation = np.full(3, 300.)
    active = ratios >= 1e-15
    isolation[active] = -10. * np.log10(np.maximum(ratios[active], 1e-30))
    np.testing.assert_allclose(result['coupling_coefficient_per_um'], kappa, rtol=1e-14)
    np.testing.assert_allclose(result['coupling_ratio'], ratios, rtol=1e-14)
    np.testing.assert_allclose(result['isolation_db'], isolation, rtol=1e-14)
assert result['backend'] == 'rust'
assert json.dumps([adjacency, ids, wavelengths, bandwidths, powers, sources, targets, losses,
                   pairs_a, pairs_b, gaps, lengths]) == before
assert Path(str(photonics.__file__)).resolve() == facade_path
assert hashlib.sha256(facade_path.read_bytes()).hexdigest() == facade_sha
try:
    engine.set_num_threads(2)
except ValueError as caught:
    assert str(caught) == 'Cannot set thread pool: The global thread pool has already been initialized.'
else:
    raise AssertionError('healthy parallel execution did not fix the shared pool')
print(json.dumps({'operation': operation, 'pressure_exercised': pressure, 'recovery': True}))
"""
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    environment.pop("RAYON_NUM_THREADS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            program,
            json.dumps(installed_engine_origins()),
            operation,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "operation": operation,
        "pressure_exercised": sys.platform == "linux",
        "recovery": True,
    }
