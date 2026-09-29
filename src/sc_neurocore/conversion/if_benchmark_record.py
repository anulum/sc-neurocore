# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Full-response measured timing record admission

"""Validate full response agreement and recompute the latency used for native ordering."""

import math
import statistics
from typing import Any

NATIVE_BACKENDS = ("rust", "go", "mojo", "julia")


def positive_integer(value: object) -> bool:
    """Check an actual positive integer, excluding JSON booleans.

    Parameters
    ----------
    value : object
        Untrusted numeric record field.

    Returns
    -------
    bool
        True only for a positive integral timing or repetition count.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validated_timing_order(record: dict[str, Any]) -> tuple[str, ...]:
    """Require all five bit-matched corpora and derive ordering from actual raw warm samples.

    Parameters
    ----------
    record : dict
        Parsed comparison after schema, runtime, source and artifact admission.

    Returns
    -------
    tuple of str
        Native runtimes sorted by equal-weight geometric mean warm call latency.

    Raises
    ------
    ValueError, KeyError, TypeError
        Incomplete corpus, invalid samples or inconsistent recorded aggregates.
    """
    meta = record["meta"]
    if not positive_integer(meta["samples"]) or meta["samples"] < 3:
        raise ValueError("at least three actual warm samples required")
    if not positive_integer(meta["warmup"]):
        raise ValueError("actual warmup required")
    providers = record["backends"]
    if set(providers) != {"numpy", *NATIVE_BACKENDS}:
        raise ValueError("complete five-runtime comparison required")
    corpus = [
        (case["name"], case["input_sha256"], case["response_sha256"])
        for case in providers["numpy"]["cases"]
    ]
    if len(corpus) != 20 or len({entry[0] for entry in corpus}) != 20:
        raise ValueError("twenty distinct verified workloads required")
    for name, input_sha, response_sha in corpus:
        if not isinstance(name, str) or not name:
            raise ValueError("named workloads required")
        for digest in (input_sha, response_sha):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("full input and response SHA-256 required")
    timings: list[tuple[float, str]] = []
    for name, entry in providers.items():
        if (
            entry["available"] is not True
            or entry["used"] is not True
            or entry["full_bit_parity"] is not True
        ):
            raise ValueError("all actual providers must have verified complete responses")
        actual = [
            (case["name"], case["input_sha256"], case["response_sha256"]) for case in entry["cases"]
        ]
        if actual != corpus:
            raise ValueError("complete provider response corpus disagrees")
        medians = []
        for case in entry["cases"]:
            samples = case["samples_ns"]
            if (
                not isinstance(samples, list)
                or len(samples) != meta["samples"]
                or not all(positive_integer(sample) for sample in samples)
                or not positive_integer(case["first_call_ns"])
            ):
                raise ValueError("complete positive raw call times required")
            median = statistics.median(samples) / 1e6
            if case["median_call_ms"] != median:
                raise ValueError("recorded workload median disagrees with raw times")
            medians.append(median)
        latency = statistics.geometric_mean(medians)
        if not math.isfinite(latency) or entry["median_call_ms"] != latency:
            raise ValueError("recorded aggregate disagrees with measured warm medians")
        if name in NATIVE_BACKENDS:
            timings.append((latency, name))
    return tuple(name for _, name in sorted(timings))
