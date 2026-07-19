# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PyPI metrics implementation and workflow contract tests

from __future__ import annotations

import importlib.util
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "pypi_downloads.py"
SPEC = importlib.util.spec_from_file_location("pypi_downloads", TOOL)
assert SPEC is not None and SPEC.loader is not None
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)

SAMPLE = {
    "data": [
        {"category": "without_mirrors", "date": "2026-07-17", "downloads": 12},
        {"category": "with_mirrors", "date": "2026-07-17", "downloads": 31},
        {"category": "without_mirrors", "date": "2026-07-18", "downloads": 14},
        {"category": "with_mirrors", "date": "2026-07-18", "downloads": 35},
    ],
    "package": "sc-neurocore",
}


def test_detect_package_reads_project_name() -> None:
    assert dl.detect_package(ROOT / "pyproject.toml") == "sc-neurocore"


class _FakeResponse:
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(SAMPLE).encode()


def test_http_get_accepts_only_the_quoted_fixed_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_urlopen(url: str, *, timeout: int) -> _FakeResponse:
        calls.append((url, timeout))
        return _FakeResponse()

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    assert dl.fetch_overall("sc/neurocore", dl._http_get) == SAMPLE
    assert calls == [("https://pypistats.org/api/packages/sc%2Fneurocore/overall", 30)]


@pytest.mark.parametrize(
    "url",
    (
        "http://pypistats.org/api/packages/sc-neurocore/overall",
        "https://example.invalid/api/packages/sc-neurocore/overall",
        "https://user@pypistats.org/api/packages/sc-neurocore/overall",
        "https://pypistats.org:443/api/packages/sc-neurocore/overall",
        "https://pypistats.org/api/packages/sc/neurocore/overall",
        "https://pypistats.org/api/packages/sc-neurocore/overall?mirrored=true",
        "https://pypistats.org/api/packages/sc-neurocore/overall#fragment",
        "https://pypistats.org/not-the-api",
    ),
)
def test_http_get_rejects_untrusted_url_before_open(
    url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_urlopen(*_args: object, **_kwargs: object) -> None:
        pytest.fail("urlopen must not run for an untrusted URL")

    monkeypatch.setattr(dl.urllib.request, "urlopen", unexpected_urlopen)
    with pytest.raises(ValueError, match="fixed HTTPS API"):
        dl._http_get(url)


def test_daily_counts_ignores_malformed_rows() -> None:
    payload: dict[str, Any] = {
        "data": [
            *SAMPLE["data"],
            {"category": "unknown", "date": "2026-07-18", "downloads": 99},
            {"category": "with_mirrors", "date": "", "downloads": 99},
            {"category": "with_mirrors", "date": "2026-07-19", "downloads": "bad"},
        ]
    }
    assert dl.daily_counts(payload) == {
        "2026-07-17": {"without_mirrors": 12, "with_mirrors": 31},
        "2026-07-18": {"without_mirrors": 14, "with_mirrors": 35},
    }


def test_csv_roundtrip_and_upsert_preserve_old_history(tmp_path: Path) -> None:
    path = tmp_path / "downloads" / "sc-neurocore.csv"
    old = {"2026-01-01": {"without_mirrors": 1, "with_mirrors": 2}}
    fresh = dl.daily_counts(SAMPLE)
    dl.write_csv(path, dl.merge(old, fresh))
    assert dl.read_csv(path) == {**old, **fresh}
    assert path.read_text(encoding="utf-8").splitlines()[0] == ("date,without_mirrors,with_mirrors")


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://pypistats.org", code, "error", headers, None)


def test_retry_recovers_and_honours_bounded_retry_after() -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def flaky(_url: str) -> bytes:
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(429, "7")
        return json.dumps(SAMPLE).encode()

    assert dl.fetch_overall_with_retry("sc-neurocore", flaky, waits.append) == SAMPLE
    assert waits == [7.0]
    assert dl._retry_delay(_http_error(429, "99999"), 30.0) == dl.MAX_RETRY_AFTER


def test_retry_exhaustion_is_a_clean_skip() -> None:
    waits: list[float] = []

    def unavailable(_url: str) -> bytes:
        raise _http_error(503)

    assert dl.fetch_overall_with_retry("sc-neurocore", unavailable, waits.append) is None
    assert waits == list(dl.RETRY_SCHEDULE)


def test_main_writes_history_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "sc-neurocore.csv"
    rc = dl.main(
        ["--package", "sc-neurocore", "--csv", str(path)],
        fetch=lambda _url: json.dumps(SAMPLE).encode(),
    )
    assert rc == 0
    assert dl.read_csv(path)["2026-07-18"]["without_mirrors"] == 14
    assert "sc-neurocore: 2 days recorded" in capsys.readouterr().out


def test_metrics_workflows_share_one_serialized_branch_writer() -> None:
    benchmark_text = (ROOT / ".github/workflows/benchmark.yml").read_text(encoding="utf-8")
    download_text = (ROOT / ".github/workflows/pypi-downloads.yml").read_text(encoding="utf-8")
    assert benchmark_text.count("group: metrics-branch-writer") == 1
    assert download_text.count("group: metrics-branch-writer") == 1
    assert "gh-pages-branch: metrics" in benchmark_text
    assert "benchmark-data-dir-path: benchmarks/criterion" in benchmark_text
    assert "git push origin metrics" in download_text


def test_download_workflow_uses_pinned_actions_and_write_is_job_scoped() -> None:
    path = ROOT / ".github/workflows/pypi-downloads.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    uses = [step["uses"] for step in workflow["jobs"]["snapshot"]["steps"] if "uses" in step]
    assert all("@" in action and len(action.rsplit("@", 1)[1].split()[0]) == 40 for action in uses)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["snapshot"]["permissions"] == {"contents": "write"}
