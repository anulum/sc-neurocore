# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dependency advisory response custody

"""Reject incomplete advisory reports, including ignored and unchecked packages."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import ssl
import time
from typing import Any, Protocol
import urllib.error
import urllib.request

OSV_ENDPOINT = "https://api.osv.dev/v1/querybatch"
OSV_RESPONSE_LIMIT = 16 * 1024 * 1024
OSV_PAGE_LIMIT = 100
OSV_TIMEOUT_SECONDS = 30
OSV_ATTEMPTS = 3
OSV_RETRY_DELAYS = (2.0, 6.0)
OSV_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class OsvOpener(Protocol):
    """Send one prepared OSV request; `urllib.request.urlopen` is the default."""

    def __call__(self, request: urllib.request.Request, /, *, timeout: float) -> Any:
        """Return the response, or raise `HTTPError` or another `OSError`."""


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


def send_osv_page(
    request_bytes: bytes,
    number: int,
    output_dir: Path,
    opener: OsvOpener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int | None, bytes]:
    """Send one OSV page, retrying a transient failure a bounded number of times.

    Parameters
    ----------
    request_bytes : bytes
        The page's JSON request body, already written as `request-<number>.json`.
    number : int
        One-based number of this page within the audit.
    output_dir : Path
        Directory that receives the page's response, receipt and failed attempts.
    opener : OsvOpener
        Sends the request; tests pass a proxied or recording opener.
    sleep : callable
        Waits before the next attempt.

    Returns
    -------
    tuple
        The final HTTP status and the response body read with one detection byte.

    Raises
    ------
    OSError
        Every one of the `OSV_ATTEMPTS` attempts failed in transport, or the
        server's certificate failed verification; the receipt keeps each
        failure's exception types, never its text or headers.

    Notes
    -----
    A transport failure, or HTTP 429, 500, 502, 503 or 504, is retried after
    `OSV_RETRY_DELAYS`. A failed certificate check and any other status, such
    as a 400 refusal, are final at once. Each retried HTTP body is kept as
    `response-<number>-attempt-<k>.json`, and the receipt lists every retried
    attempt in order.
    """
    receipt: dict[str, Any] = {
        "endpoint": OSV_ENDPOINT,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "http_status": None,
        "retried_attempts": [],
    }
    receipt_path = output_dir / f"response-{number}-receipt.json"
    attempt = 0
    while True:
        attempt += 1
        request = urllib.request.Request(
            OSV_ENDPOINT,
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status: int | None = None
        try:
            try:
                response = opener(request, timeout=OSV_TIMEOUT_SECONDS)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                status = response.status
                raw = response.read(OSV_RESPONSE_LIMIT + 1)
        except OSError as error:
            # Only URLError wraps the underlying failure; an ssl.SSLError's own
            # `reason` attribute is a text code, not an exception.
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            failure = {
                "http_status": status,
                "error_type": type(error).__name__,
                "reason_type": type(reason).__name__,
            }
            # A failed certificate check is not transient: it is final at once.
            if attempt < OSV_ATTEMPTS and not isinstance(reason, ssl.SSLCertVerificationError):
                receipt["retried_attempts"].append(failure)
                sleep(OSV_RETRY_DELAYS[attempt - 1])
                continue
            receipt.update(failure)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise
        if status in OSV_RETRY_STATUSES and attempt < OSV_ATTEMPTS:
            (output_dir / f"response-{number}-attempt-{attempt}.json").write_bytes(raw)
            receipt["retried_attempts"].append(
                {
                    "http_status": status,
                    "response_bytes": len(raw),
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            sleep(OSV_RETRY_DELAYS[attempt - 1])
            continue
        break
    receipt["http_status"] = status
    (output_dir / f"response-{number}.json").write_bytes(raw)
    receipt["response_bytes"] = len(raw)
    receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return status, raw


def audit_osv_queries(
    queries: tuple[dict[str, Any], ...],
    output_dir: Path,
    opener: OsvOpener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Query every exact identity and retain all OSV response pages.

    Parameters
    ----------
    queries : tuple of dict
        OSV package/version identities extracted from a committed manifest.
    output_dir : Path
        Fresh directory for requests, raw response bytes and transport receipts.
    opener : OsvOpener
        Sends each request; see `send_osv_page`.
    sleep : callable
        Waits between the attempts `send_osv_page` retries.

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
    A transient failure is retried within `send_osv_page`'s bound, and every
    retried attempt stays in the page's receipt.
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
        status, raw = send_osv_page(request_bytes, number, output_dir, opener, sleep)
        if status != 200:
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
