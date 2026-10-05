# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dependency advisory response custody

"""Reject incomplete advisory reports, including ignored and unchecked packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

OSV_ENDPOINT = "https://api.osv.dev/v1/querybatch"
OSV_RESPONSE_LIMIT = 16 * 1024 * 1024
OSV_PAGE_LIMIT = 100


def osv_page_outcome(
    number: int,
    raw: bytes,
    pending: list[tuple[int, dict[str, Any]]],
    seen: set[tuple[int, str]],
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    """Validate one raw OSV response page against the queries it must answer.

    Parameters
    ----------
    number : int
        One-based number of this page within the bounded audit.
    raw : bytes
        Response body exactly as received, read with one detection byte.
    pending : list of tuple
        Query positions and query objects submitted for this page, in order.
    seen : set of tuple
        Query positions and continuation tokens already followed.

    Returns
    -------
    tuple of list
        Findings as query position and advisory record, and the continuation
        queries that the next page must answer. Nothing is filtered.

    Raises
    ------
    ValueError
        The body exceeds the evidence limit, omits or refuses a query, carries
        an invalid advisory or continuation, or continues past the page budget.
    """
    if len(raw) > OSV_RESPONSE_LIMIT:
        raise ValueError("OSV response exceeds the audit evidence limit.")
    payload = json.loads(raw)
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(pending):
        raise ValueError("OSV response does not cover every requested identity.")
    findings: list[tuple[int, dict[str, Any]]] = []
    following: list[tuple[int, dict[str, Any]]] = []
    for (index, query), row in zip(pending, rows):
        if not isinstance(row, dict) or "error" in row:
            raise ValueError("OSV returned an unanswered package query.")
        records = row.get("vulns", [])
        if not isinstance(records, list):
            raise ValueError("OSV vulnerability records are invalid.")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError("OSV vulnerability identity is invalid.")
            if not record["id"]:
                raise ValueError("OSV vulnerability identity is empty.")
            findings.append((index, record))
        token = row.get("next_page_token")
        if token is not None:
            if not isinstance(token, str) or not token or (index, token) in seen:
                raise ValueError("OSV pagination is invalid or repeated.")
            following.append((index, {**query, "page_token": token}))
    if following and number >= OSV_PAGE_LIMIT:
        raise ValueError("OSV audit exceeded its bounded page budget.")
    return findings, following


def audit_osv_queries(queries: tuple[dict[str, Any], ...], output_dir: Path) -> dict[str, Any]:
    """Query every exact identity and retain all OSV response pages.

    Parameters
    ----------
    queries : tuple of dict
        OSV package/version identities extracted from a committed manifest.
    output_dir : Path
        Fresh directory for requests, raw response bytes and transport receipts.

    Returns
    -------
    dict
        Complete per-query results and a blocking pass/fail decision.

    Raises
    ------
    OSError, ValueError
        HTTP, JSON, response cardinality or pagination custody is incomplete.
        Findings of every severity block; no reachability filter is applied.

    Notes
    -----
    HTTP refusals retain their status and bounded response body. Transport
    failures retain exception types without exception text or response headers.
    """
    if not queries:
        raise ValueError("An OSV audit requires exact package identities.")
    output_dir.mkdir(parents=True, exist_ok=False)
    pending = [(index, dict(query)) for index, query in enumerate(queries)]
    results: list[list[dict[str, Any]]] = [[] for _ in queries]
    seen: set[tuple[int, str]] = set()
    pages: list[dict[str, Any]] = []
    while pending:
        number = len(pages) + 1
        request_bytes = json.dumps({"queries": [query for _, query in pending]}).encode("utf-8")
        (output_dir / f"request-{number}.json").write_bytes(request_bytes)
        request = urllib.request.Request(
            OSV_ENDPOINT,
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        receipt: dict[str, str | int | None] = {
            "endpoint": OSV_ENDPOINT,
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "http_status": None,
        }
        receipt_path = output_dir / f"response-{number}-receipt.json"
        try:
            try:
                response = urllib.request.urlopen(  # nosec B310 - fixed https OSV_ENDPOINT
                    request, timeout=30
                )
            except urllib.error.HTTPError as error:
                response = error
            with response:
                receipt["http_status"] = response.status
                raw = response.read(OSV_RESPONSE_LIMIT + 1)
        except OSError as error:
            receipt["error_type"] = type(error).__name__
            receipt["reason_type"] = type(getattr(error, "reason", error)).__name__
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise
        (output_dir / f"response-{number}.json").write_bytes(raw)
        receipt["response_bytes"] = len(raw)
        receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if receipt["http_status"] != 200:
            raise ValueError("OSV query did not return HTTP 200.")
        findings, pending = osv_page_outcome(number, raw, pending, seen)
        for index, finding in findings:
            results[index].append(finding)
        seen.update((index, query["page_token"]) for index, query in pending)
        pages.append({"number": number, "sha256": hashlib.sha256(raw).hexdigest()})
    return {
        "endpoint": OSV_ENDPOINT,
        "queries": queries,
        "results": results,
        "pages": pages,
        "coverage_complete": True,
        "passed": not any(results),
    }


def pixi_report_passed(payload: object, package_count: int) -> bool:
    """Require complete Pixi audit coverage with no finding or suppression.

    Parameters
    ----------
    payload : object
        JSON from the pinned pixi-audit invocation, with no severity filter.
    package_count : int
        Number of package records in the committed Pixi lock.

    Returns
    -------
    bool
        True only for complete, unsuppressed, vulnerability-free coverage.

    Raises
    ------
    ValueError
        Required response lists, counters or package coverage are invalid.
        Unsupported channels are unchecked and cannot qualify as audited.
    """
    if not isinstance(payload, dict) or package_count <= 0:
        raise ValueError("Pixi audit report is invalid.")
    for key in ("vulnerabilities", "ignored", "unchecked", "unmatched_ignores"):
        if not isinstance(payload.get(key), list):
            raise ValueError("Pixi audit coverage lists are unavailable.")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Pixi audit coverage counters are unavailable.")
    for key in ("audited", "vulnerable", "ignored", "unchecked"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Pixi audit coverage counter is invalid.")
    if summary["audited"] + summary["unchecked"] != package_count:
        raise ValueError("Pixi audit does not cover the complete lock.")
    counters = (
        ("vulnerable", "vulnerabilities"),
        ("ignored", "ignored"),
        ("unchecked", "unchecked"),
    )
    if any(summary[counter] != len(payload[section]) for counter, section in counters):
        raise ValueError("Pixi audit coverage lists disagree with their counters.")
    return not any(
        payload[key] for key in ("vulnerabilities", "ignored", "unchecked", "unmatched_ignores")
    )
