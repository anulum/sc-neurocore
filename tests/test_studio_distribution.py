# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Built distribution model resource contracts

"""Verify actual wheels preserve every declared package resource and reference page."""

from dataclasses import asdict
from fnmatch import fnmatchcase
from pathlib import Path
from types import ModuleType
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import tarfile
import zipfile

import pytest

from sc_neurocore.neurons.model_identity import catalogue_counts
from sc_neurocore.studio.codegen import generate_experiment_script, generate_oneliner
from sc_neurocore.studio.model_catalogue import corpus_revision, list_models
from sc_neurocore.studio.experiment_spec import resolve_experiment
from sc_neurocore.studio.replay_pack import build_replay_pack
from sc_neurocore.studio.candidate_run import review_packet, simulate_candidate
from tests.studio_candidate_support import adex_candidate

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


CONVERSION_BENCHMARK = (
    "benchmarks/bench_ann_to_snn_replay.py",
    "benchmarks/_ann_to_snn_replay_profiles.py",
    "benchmarks/_ann_to_snn_replay_measurement.py",
)
"""Comparison scripts the source distribution carries for the installed benchmark command."""


@pytest.fixture(scope="module")
def distribution_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy build inputs without editable metadata or modifying the working tree."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path_factory.mktemp("studio-distribution") / "source"
    source.mkdir()
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "docs/api/models",
            *CONVERSION_BENCHMARK,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    inputs = set(tracked.stdout.rstrip("\0").split("\0"))
    inputs.update(("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "src/build_support.py"))
    for name in sorted(inputs):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    return source


def build_distribution(
    source: Path, command: str, output: Path
) -> subprocess.CompletedProcess[str]:
    """Invoke the configured setuptools backend in a bounded separate process.

    Parameters
    ----------
    source:
        Disposable source tree containing real project build inputs.
    command:
        Backend operation, either build_wheel or build_sdist.
    output:
        Dedicated distribution output directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured build status and diagnostics for acceptance or refusal checks.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os, sys
tracer = None
coverage_file = os.environ.get("STUDIO_BUILD_COVERAGE_FILE")
if coverage_file:
    import coverage
    tracer = coverage.Coverage(
        data_file=coverage_file, data_suffix=True, config_file=False, branch=True,
        include=["*/src/build_support.py"],
    )
    tracer.start()
try:
    from setuptools import build_meta
    getattr(build_meta, sys.argv[1])(sys.argv[2])
finally:
    if tracer is not None:
        tracer.stop()
        tracer.save()
""",
            command,
            str(output),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def declared_package_resources(source: Path) -> tuple[str, ...]:
    """Return the resource patterns the distribution declares for the package.

    Parameters
    ----------
    source:
        Source tree whose ``pyproject.toml`` is the resource contract.

    Returns
    -------
    tuple of str
        The ``sc_neurocore`` package-data patterns, relative to the package.
    """
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    return tuple(project["tool"]["setuptools"]["package-data"]["sc_neurocore"])


def matches_resource(relative: str, pattern: str) -> bool:
    """Match a package-relative path against one declared pattern, segment by segment.

    Parameters
    ----------
    relative:
        Path inside the package, with ``/`` separators.
    pattern:
        Declared package-data pattern; ``*`` never crosses a directory.

    Returns
    -------
    bool
        Whether the path is one the pattern declares.
    """
    parts, pattern_parts = relative.split("/"), pattern.split("/")
    return len(parts) == len(pattern_parts) and all(
        fnmatchcase(part, expected) for part, expected in zip(parts, pattern_parts, strict=True)
    )


@pytest.mark.parametrize("through_sdist", [False, True], ids=["wheel", "sdist-wheel"])
def test_wheel_contains_exact_model_resources(
    distribution_source: Path, tmp_path: Path, through_sdist: bool
) -> None:
    """Both build paths ship every declared resource and canonical page byte-for-byte.

    Every package-data pattern must match at least one tracked source file, and
    the wheel must hold exactly those files under that pattern -- so a resource
    class the catalogue needs cannot silently drop out of the distribution.
    """
    source = distribution_source
    if through_sdist:
        result = build_distribution(source, "build_sdist", tmp_path / "sdist")
        assert result.returncode == 0, result.stdout + result.stderr
        (archive_path,) = (tmp_path / "sdist").glob("*.tar.gz")
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp_path / "unpacked", filter="data")
        (source,) = (tmp_path / "unpacked").iterdir()
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (tmp_path / "wheel").glob("*.whl")
    package = source / "src" / "sc_neurocore"
    with zipfile.ZipFile(wheel_path) as wheel:
        members = [name for name in wheel.namelist() if name.startswith("sc_neurocore/")]
        for pattern in declared_package_resources(source):
            declared_files = sorted(
                path.relative_to(package).as_posix()
                for path in package.rglob("*")
                if path.is_file()
                and matches_resource(path.relative_to(package).as_posix(), pattern)
            )
            assert declared_files, f"declared resource pattern {pattern!r} matches no source file"
            shipped = sorted(
                name.removeprefix("sc_neurocore/")
                for name in members
                if matches_resource(name.removeprefix("sc_neurocore/"), pattern)
            )
            assert shipped == declared_files, pattern
            for relative in declared_files:
                assert wheel.read(f"sc_neurocore/{relative}") == (package / relative).read_bytes()
        for directory, pattern, destination in (("docs/api/models", "*.md", "studio/model_docs"),):
            originals = sorted((distribution_source / directory).glob(pattern))
            assert originals, directory
            expected = {f"sc_neurocore/{destination}/{path.name}" for path in originals}
            actual = {
                name for name in wheel.namelist() if name.startswith(f"sc_neurocore/{destination}/")
            }
            assert actual == expected
            for path in originals:
                assert wheel.read(f"sc_neurocore/{destination}/{path.name}") == path.read_bytes()
        installed = tmp_path / "installed"
        wheel.extractall(installed)
    verify_installed_replays(installed, tmp_path)
    verify_installed_candidate_jobs(installed, tmp_path)


def verify_installed_candidate_jobs(installed: Path, workspace: Path) -> None:
    """Compare installed process jobs with the same ordered source calculations.

    Parameters
    ----------
    installed : Path
        Actual extracted wheel, searched before existing dependency packages.
    workspace : Path
        Consumer directory outside the checkout, holding original requests.
    """
    cases = []
    for numerical in ["euler", "gauss_seidel", "diffusion"]:
        document = adex_candidate()
        document["model"]["state"] = {"w": 7.0, "v": -65.0}
        document["reference_tests"] = document["reference_tests"][:1]
        document["reference_tests"][0]["steps"] = 20
        if numerical == "gauss_seidel":
            document["model"]["integration"]["method"] = "gauss_seidel"
            dynamics = document["model"]["dynamics"]
            document["model"]["dynamics"] = {"w": dynamics["w"], "v": dynamics["v"]}
        elif numerical == "diffusion":
            for name, magnitude, unit in [
                ("sigma_v", 0.5, "mV / ms**0.5"),
                ("sigma_w", 0.25, "pA / ms**0.5"),
            ]:
                document["model"]["parameters"][name] = magnitude
                document["units"]["parameters"][name] = unit
                document["model"]["profile"]["parameters"][name] = "source"
                document["model"]["profile"]["units"][name] = unit
            document["model"]["dynamics"] = {"w": "sigma_w * xi", "v": "sigma_v * xi"}
            document["model"]["threshold"]["rng_seed"] = 17
        for operation in ["simulate", "review-packet"]:
            cases.append(
                {
                    "operation": operation,
                    "body": {
                        "candidate": document,
                        **({"current": 800.0, "steps": 20} if operation == "simulate" else {}),
                    },
                    "expected": simulate_candidate(document, current=800.0, steps=20)
                    if operation == "simulate"
                    else review_packet(document),
                }
            )
    (workspace / "candidate-cases.json").write_text(json.dumps(cases), encoding="utf-8")
    probe = """
import hashlib, json, sys, time
from pathlib import Path
assert sys.flags.isolated and sys.flags.no_site
installed = Path(sys.argv[1])
sys.path[:0] = [str(installed), sys.argv[2]]
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(installed)
from starlette.testclient import TestClient
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
app = create_app(StudioRuntimeSettings(job_root_path='candidate-jobs', enforce_route_policies=True))
headers = {'sec-fetch-site': 'none', 'x-studio-principal': 'alice'}
with TestClient(app, base_url='http://127.0.0.1', headers=headers) as client:
    before = client.get('/api/candidates/jobs').json()
    candidate = json.loads(Path('candidate-cases.json').read_text())[0]['body']['candidate']
    for current in [True, '800']:
        response = client.post('/api/candidates/simulate', json={'candidate': candidate, 'current': current, 'steps': 2})
        assert response.status_code == 422, response.text
    assert client.get('/api/candidates/jobs').json() == before
    for case in json.loads(Path('candidate-cases.json').read_text()):
        response = client.post('/api/candidates/' + case['operation'], json=case['body'])
        assert response.status_code == 202, response.text
        receipt = response.json()
        deadline = time.monotonic() + 45
        while True:
            record = client.get(receipt['status_route']).json()
            if record['status'] == 'completed':
                break
            assert record['status'] in {'pending', 'running'}, record
            assert time.monotonic() < deadline, record
            time.sleep(0.02)
        assert record['owner'] == 'alice' and record['execution_model'] == 'process'
        assert record['result'] == case['expected']
        assert record['admission']['candidate_sha256'] == case['expected']['candidate_sha256']
        for artifact in record['artifacts']:
            response = client.get(receipt['status_route'] + '/artifacts/' + artifact['relative_path'])
            assert response.status_code == 200, response.text
            assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
            if artifact['relative_path'] == 'request.json':
                assert response.json() == case['body']
                assert hashlib.sha256(response.content).hexdigest() == receipt['request_sha256']
                assert list(response.json()['candidate']['model']['state']) == ['w', 'v']
    parent = installed / 'sc_neurocore/neurons/model_schemas/adex.toml'
    original_parent = parent.read_bytes()
    corrupt_parent = b'{"metadata":"private-corrupt-parent-marker",'
    parent.write_bytes(corrupt_parent)
    try:
        case = json.loads(Path('candidate-cases.json').read_text())[0]
        response = client.post('/api/candidates/review-packet', json={'candidate': case['body']['candidate']})
        assert response.status_code == 202, response.text
        route = response.json()['status_route']
        deadline = time.monotonic() + 45
        while True:
            record = client.get(route).json()
            if record['status'] not in {'pending', 'running'}:
                break
            assert time.monotonic() < deadline, record
            time.sleep(0.02)
        assert record['status'] == 'failed', record
        assert record['result'] is None
        assert 'private-corrupt-parent-marker' not in json.dumps(record)
        assert parent.read_bytes() == corrupt_parent
    finally:
        parent.write_bytes(original_parent)
    task = installed / 'sc_neurocore/studio/candidate_job_tasks.py'
    retained = task.with_suffix('.retained')
    task.rename(retained)
    try:
        case = json.loads(Path('candidate-cases.json').read_text())[0]
        response = client.post('/api/candidates/simulate', json=case['body'])
        assert response.status_code == 202, response.text
        route = response.json()['status_route']
        deadline = time.monotonic() + 45
        while True:
            record = client.get(route).json()
            if record['status'] not in {'pending', 'running'}:
                break
            assert time.monotonic() < deadline, record
            time.sleep(0.02)
        assert record['status'] == 'failed', record
        assert record['result'] is None
    finally:
        retained.rename(task)
    assert app.state.studio_job_manager.unreaped_workers == ()
assert all(Path(module.__file__).is_relative_to(installed)
    for name, module in tuple(sys.modules.items())
    if name.startswith('sc_neurocore.') and getattr(module, '__file__', None))
print('six installed Euler/Gauss-Seidel/diffusion HTTP jobs match source; corrupt parent and missing installed worker refuse')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(installed), sysconfig.get_path("purelib")],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def verify_installed_replays(installed: Path, workspace: Path) -> None:
    """Execute replay packs, scripts and one-liners with editable hooks disabled.

    Compare every scalar/vector sample, spike, drive digest and state boundary
    for nondefault-timestep, nonconstant-drive, rate and seeded stochastic cases.
    The installed catalogue must also report exactly the counts the source tree
    does, so a receipt-bound identity is receipt-bound only because its receipt
    was shipped and resolved inside the installed package. An offered execution
    lane must run a kernel and agree with the Python lane; a lane the wheel
    does not provide must refuse an explicit request. Dependencies reuse the test interpreter's site-packages; this is package
    isolation, not a fresh dependency-resolution or platform receipt.

    Parameters
    ----------
    installed:
        Extracted wheel root, used before dependency search paths.
    workspace:
        Disposable working directory containing no source checkout.
    """
    requests = [
        {
            "name": "AdExNeuron",
            "dt": 0.05,
            "duration": 5.0,
            "current": 1000.0,
            "protocol": "step",
        },
        {"name": "PoissonNeuron", "params": {"seed": 77, "rate_hz": 150}, "duration": 5.0},
        {"name": "WilsonCowanUnit", "duration": 5.0},
        {"name": "AmariNeuralField", "duration": 5.0},
    ]
    packs = [build_replay_pack(request) for request in requests]
    (workspace / "packs.json").write_text(json.dumps(packs), encoding="utf-8")
    for index, pack in enumerate(packs):
        spec = resolve_experiment(pack["request"])
        for form, generator in (
            ("script", generate_experiment_script),
            ("oneliner", generate_oneliner),
        ):
            (workspace / f"{index}-{form}.py").write_text(
                generator(spec, pack["request"]), encoding="utf-8"
            )
    probe = """
import contextlib, importlib.util, io, json, runpy, site, sys
from pathlib import Path
assert sys.flags.isolated and sys.flags.no_site
# PythonCall initializes builtins.help even when -S suppresses site startup.
isolated_paths = tuple(sys.path)
site.sethelper()
assert tuple(sys.path) == isolated_paths
installed = Path(sys.argv[1])
sys.path[:0] = [str(installed), sys.argv[2]]
packs = json.load(sys.stdin)
import sc_neurocore
assert Path(sc_neurocore.__file__).is_relative_to(installed)
from sc_neurocore.studio.model_catalogue import model_documentation
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.studio.replay_pack import replay_pack, replay_expectation
from dataclasses import asdict
from sc_neurocore.neurons.model_identity import catalogue_counts, iter_source_catalogue
from sc_neurocore.neurons.model_receipts import RECEIPT_DIRECTORY
assert RECEIPT_DIRECTORY.is_relative_to(installed)
counts = asdict(catalogue_counts())
assert counts == json.loads(sys.argv[3]), counts
# The Studio browses the same catalogue, in the same health, with the same
# proven readiness: receipts whose subjects a wheel lacks would read as stale.
from sc_neurocore.studio.model_catalogue import corpus_revision, list_models
listed = list_models()
studio_catalogue = {
    "revision": corpus_revision(listed),
    "readiness": sorted(
        [m["name"], m["verified_science_label"], m["verified_silicon_label"], m["is_perfect_verified"]]
        for m in listed
    ),
}
checkout_catalogue = json.loads(sys.argv[4])
differing = sorted(
    {tuple(row) for row in studio_catalogue["readiness"]}
    ^ {tuple(row) for row in checkout_catalogue["readiness"]}
)
assert studio_catalogue == checkout_catalogue, ("installed Studio catalogue differs", differing[:20])
bound = [record for record in iter_source_catalogue() if record.revalidation == "receipt-bound"]
assert len(bound) == counts["receipt_bound_complete"] > 0
from sc_neurocore.runtime_lanes import ACCEL_ROOT, lane_statuses
assert ACCEL_ROOT.is_relative_to(installed)
lanes = {status.lane: status for status in lane_statuses()}
assert lanes["python"].resources_present
# The wheel ships one Julia model source, but no built Go or Mojo libraries.
assert (ACCEL_ROOT / "julia/neurons/adaptive_threshold_if.jl").is_file()
assert not (ACCEL_ROOT / "julia/neurons/alpha.jl").exists()
assert lanes["julia"].distribution == "bundled-source"
assert lanes["julia"].resources_present == (importlib.util.find_spec("juliacall") is not None)
assert not any(lanes[lane].resources_present for lane in ("go", "mojo")), lanes
# Each lane the installation offers runs; a lane it lacks is refused, never
# silently replaced by Python.
import numpy as np
from sc_neurocore.accel.adaptive_threshold_if import simulate_adaptive_threshold_if
from sc_neurocore.accel.alpha import simulate_alpha
adaptive_drive = [0.0] * 40 + [80.0] * 40
adaptive_reference = simulate_adaptive_threshold_if(current=adaptive_drive, backend="python")
assert adaptive_reference["spike_count"] > 0
if lanes["julia"].resources_present:
    adaptive_native = simulate_adaptive_threshold_if(current=adaptive_drive, backend="julia")
    assert adaptive_native.keys() == adaptive_reference.keys()
    for key in ("v", "theta", "v_final", "theta_final"):
        np.testing.assert_allclose(adaptive_native[key], adaptive_reference[key], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(adaptive_native["spikes"], adaptive_reference["spikes"])
    assert adaptive_native["spike_count"] == adaptive_reference["spike_count"]
else:
    try:
        simulate_adaptive_threshold_if(current=adaptive_drive, backend="julia")
    except RuntimeError as refusal:
        assert "unavailable" in str(refusal)
    else:
        raise AssertionError("the Julia kernel ran without its runtime")
drive = [0.3] * 40 + [1.4] * 40
reference = simulate_alpha(exc_current=drive, backend="python")
assert int(np.sum(reference["spikes"])) > 0
if lanes["rust"].resources_present:
    native = simulate_alpha(exc_current=drive, backend="rust")
    np.testing.assert_allclose(native["v"], reference["v"], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(native["spikes"], reference["spikes"])
    refused = ("julia", "go", "mojo")
else:
    refused = ("rust", "julia", "go", "mojo")
for lane in refused:
    try:
        simulate_alpha(exc_current=drive, backend=lane)
    except RuntimeError as refusal:
        assert "unavailable" in str(refusal)
    else:
        raise AssertionError(f"the {lane} lane ran without its resources")
for index, pack in enumerate(packs):
    name = pack["request"]["name"]
    assert load_descriptor(name) is not None, name
    assert model_documentation(name) is not None, name
    result = replay_pack(pack)
    assert result["verdict"] == "match", result["differences"]
    assert result["worst_state_deviation"] == 0.0
    assert result["runtime_differences"] == []
    for form, result_name in (("script", "result"), ("oneliner", "r")):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            namespace = runpy.run_path(f"{index}-{form}.py", run_name="__main__")
        assert "spikes in" in output.getvalue(), output.getvalue()
        assert replay_expectation(namespace[result_name]) == pack["expectation"], (name, form)
        if form == "script":
            assert namespace["REQUEST"] == pack["request"]
            assert namespace["EXPERIMENT_SHA256"] == pack["experiment_sha256"]
    pack["expectation"]["spike_count"] += 1
    assert replay_pack(pack)["verdict"] == "mismatch"
print("four model families replay exactly; both script forms reproduce full traces; tampering rejected")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(installed),
            sysconfig.get_path("purelib"),
            json.dumps(asdict(catalogue_counts())),
            json.dumps(_studio_catalogue()),
        ],
        cwd=workspace,
        input=json.dumps(packs),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "four model families replay exactly" in result.stdout


def _studio_catalogue() -> dict[str, object]:
    """Return the checkout's Studio catalogue revision and proven readiness per model."""
    listed = list_models()
    return {
        "revision": corpus_revision(listed),
        "readiness": sorted(
            [
                model["name"],
                model["verified_science_label"],
                model["verified_silicon_label"],
                model["is_perfect_verified"],
            ]
            for model in listed
        ),
    }


def test_build_refuses_missing_model_pages(distribution_source: Path, tmp_path: Path) -> None:
    """A source tree without reference pages must fail rather than ship an undocumented Studio."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    pages = source / "docs/api/models"
    pages.rename(source / "withheld-model-pages")
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode != 0
    assert "No model reference pages" in result.stdout + result.stderr


def test_build_refuses_a_missing_conversion_benchmark_source(
    distribution_source: Path, tmp_path: Path
) -> None:
    """A source tree without one comparison script fails instead of shipping a broken command.

    Parameters
    ----------
    distribution_source : Path
        Disposable copy of the real build inputs.
    tmp_path : Path
        Workspace for the altered copy and the refused build.
    """
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    (source / CONVERSION_BENCHMARK[1]).unlink()
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode != 0
    assert "Missing conversion benchmark source" in result.stdout + result.stderr
    assert not list((tmp_path / "wheel").glob("*.whl"))


def test_rebuild_replaces_modified_and_removed_pages(
    distribution_source: Path, tmp_path: Path
) -> None:
    """Reused build directories must not preserve deleted pages or older-timestamp content."""
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    first = build_distribution(source, "build_wheel", tmp_path / "first")
    assert first.returncode == 0, first.stdout + first.stderr
    pages = sorted((source / "docs/api/models").glob("*.md"))
    removed, changed = pages[:2]
    removed.rename(source / removed.name)
    changed.write_text("# Revised reference page\n", encoding="utf-8")
    os.utime(changed, (1, 1))
    second = build_distribution(source, "build_wheel", tmp_path / "second")
    assert second.returncode == 0, second.stdout + second.stderr
    (wheel_path,) = (tmp_path / "second").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        assert f"sc_neurocore/studio/model_docs/{removed.name}" not in wheel.namelist()
        assert wheel.read(f"sc_neurocore/studio/model_docs/{changed.name}") == changed.read_bytes()


def studio_installed_acceptance() -> ModuleType:
    """Load the release acceptance tool, which lives outside the package."""
    path = Path(__file__).resolve().parents[1] / "tools" / "studio_installed_acceptance.py"
    spec = importlib.util.spec_from_file_location("studio_installed_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def installation_environment(installed: Path, root: Path) -> Path:
    """Create a virtual environment whose ``sc_neurocore`` is the extracted wheel.

    The test interpreter's packages supply the dependencies; path files inside
    them (an editable install of the checkout) are not processed, so the only
    ``sc_neurocore`` the environment can import is ``installed``.

    Returns
    -------
    pathlib.Path
        The environment's interpreter.
    """
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=120
    )
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    (Path(purelib) / "installed_wheel.pth").write_text(
        f"{installed}\n{sysconfig.get_path('purelib')}\n", encoding="utf-8"
    )
    return python


def free_port() -> int:
    """Return a port nothing listens on at the moment of asking."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _with_built_ui(source: Path) -> Path:
    """Give a disposable source tree a small real UI build, as ``npm run build`` would."""
    ui = source / "studio/frontend/dist"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text(
        '<html><script src="/studios/sc-neurocore/assets/app.js"></script></html>',
        encoding="utf-8",
    )
    (ui / "assets/app.js").write_text("console.log('studio')", encoding="utf-8")
    return ui


def test_the_documented_workflows_run_from_an_installed_wheel(
    distribution_source: Path, tmp_path: Path
) -> None:
    """Each block of the supported-workflows page prints its documented output from a wheel.

    The wheel is built from the tracked sources, extracted, and imported by a
    separate environment whose working directory is outside the repository, so
    the blocks cannot reach a module that the wheel does not ship.

    Parameters
    ----------
    distribution_source : Path
        Disposable copy of the real build inputs.
    tmp_path : Path
        Workspace for the wheel, the extracted package and the environment.
    """
    from tests.test_supported_workflows import documented_workflows, run_workflow

    result = build_distribution(distribution_source, "build_wheel", tmp_path / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (tmp_path / "wheel").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel.extractall(installed)
    python = installation_environment(installed, tmp_path / "environment")
    outside = tmp_path / "outside"
    outside.mkdir()
    located = run_workflow(
        str(python), "import sc_neurocore; print(sc_neurocore.__file__)", outside
    )
    assert Path(located.stdout.strip()).is_relative_to(installed), located.stderr
    workflows = documented_workflows()
    assert len(workflows) == 4
    for name, (code, expected) in workflows.items():
        ran = run_workflow(str(python), code, outside)
        assert ran.returncode == 0, (name, ran.stderr)
        assert ran.stdout == expected, name


def test_a_built_ui_ships_in_the_wheel_and_is_served_installed(
    distribution_source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`pip install` then `sc-neurocore studio` must open a user interface, not an empty page.

    The installed copy is launched by the release acceptance tool through its own
    CLI, in an environment whose only ``sc_neurocore`` is the wheel's.
    """
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    ui = _with_built_ui(source)
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (tmp_path / "wheel").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel_path) as wheel:
        for relative in ("index.html", "assets/app.js"):
            assert (
                wheel.read(f"sc_neurocore/studio/frontend_dist/{relative}")
                == (ui / relative).read_bytes()
            )
        wheel.extractall(installed)
    environment = installation_environment(installed, tmp_path / "environment")
    acceptance = studio_installed_acceptance()
    assert acceptance.main(["--python", str(environment), "--port", str(free_port())]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] is True
    assert Path(report["package"]).is_relative_to(installed)
    assert report["assets"] >= 1
    assert report["models"] == len(list_models())


def test_a_release_build_without_the_ui_is_refused(
    distribution_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release workflows set the requirement, so a published wheel cannot lack its UI."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_UI", "required")
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode != 0
    assert "No built Studio UI" in result.stdout + result.stderr


def test_a_source_build_without_the_ui_carries_none(
    distribution_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source install without Node.js still builds; it simply has no UI to serve.

    The tree is first built with a UI, so the second build reuses a build
    directory that still holds one: a UI removed from the source must not
    survive in the next wheel.
    """
    monkeypatch.delenv("SC_NEUROCORE_STUDIO_UI", raising=False)
    source = tmp_path / "source"
    shutil.copytree(
        distribution_source, source, ignore=shutil.ignore_patterns("build", "*.egg-info")
    )
    ui = _with_built_ui(source)
    first = build_distribution(source, "build_wheel", tmp_path / "with-ui")
    assert first.returncode == 0, first.stdout + first.stderr
    shutil.rmtree(ui)
    result = build_distribution(source, "build_wheel", tmp_path / "wheel")
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel_path,) = (tmp_path / "wheel").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        assert not [name for name in wheel.namelist() if "/frontend_dist/" in name]
