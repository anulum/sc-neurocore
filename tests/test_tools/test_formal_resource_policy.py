# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed resource-budget and attempt-order contracts

"""Reject inferred budgets, solver changes and invalid formal retry sequences."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from tools.formal_resource_policy import (
    AMARI_JOB,
    AMARI_RECEIPT,
    CHIALVO_JOB,
    CHIALVO_REPAIR,
    HISTORICAL_AMARI,
    MEMORY_RETRY,
    PRIMARY,
    YICES_RETRY,
    next_attempt,
    select_contract,
)


@pytest.mark.parametrize(
    "name,solver,memory",
    [
        (PRIMARY, "z3", 40),
        (MEMORY_RETRY, "z3", 64),
        (YICES_RETRY, "yices-smt2", 64),
        (HISTORICAL_AMARI, "z3", 64),
    ],
)
def test_index_and_review_select_fixed_budget(name: str, solver: str, memory: int) -> None:
    """An explicit reviewed choice selects the budget, never the observed peak."""
    entry = {"resource_contract": name, "measurement": {"memory_limit_bytes": 1}}
    review = {"verdict": "CLEAR", "receipt_sha256": AMARI_RECEIPT, "resource_contract": name}
    contract = select_contract(entry, review, AMARI_RECEIPT, AMARI_JOB)
    assert contract.solver == solver
    policy = contract.policy()
    assert policy.memory_bytes == memory * 1024**3
    assert policy.timeout_seconds == 7200
    assert policy.hostname == "god-of-the-math"
    assert policy.audit_memory_bytes == 8 * 1024**3
    assert policy.audit_timeout_seconds == 1800


@pytest.mark.parametrize("name", [None, 40, {}, "", "z3-primary-64", "unreviewed"])
def test_missing_unknown_and_inferred_contracts_refuse(name: object) -> None:
    """A plausible observed 64 GiB receipt cannot supply an absent contract."""
    with pytest.raises(ValueError, match="resource contract"):
        select_contract(
            {"resource_contract": name, "memory_limit_bytes": 64 * 1024**3},
            {},
            AMARI_RECEIPT,
            AMARI_JOB,
        )


@pytest.mark.parametrize(
    "review",
    [
        {},
        {"verdict": "PENDING", "receipt_sha256": AMARI_RECEIPT, "resource_contract": PRIMARY},
        {"verdict": "CLEAR", "receipt_sha256": "0" * 64, "resource_contract": PRIMARY},
        {"verdict": "CLEAR", "receipt_sha256": AMARI_RECEIPT, "resource_contract": MEMORY_RETRY},
    ],
)
def test_review_binds_same_receipt_and_choice(review: Mapping[str, object]) -> None:
    """A changed review binding invalidates an otherwise registered budget."""
    with pytest.raises(ValueError, match="Independent review"):
        select_contract({"resource_contract": PRIMARY}, review, AMARI_RECEIPT, AMARI_JOB)


@pytest.mark.parametrize(
    "sha,job", [("0" * 64, AMARI_JOB), (AMARI_RECEIPT, "hdl/formal/other.sby")]
)
def test_historical_exception_cannot_expand(sha: str, job: str) -> None:
    """The historical 64 GiB case is one exact receipt and one exact job."""
    review = {"verdict": "CLEAR", "receipt_sha256": sha, "resource_contract": HISTORICAL_AMARI}
    with pytest.raises(ValueError, match="exact Amari"):
        select_contract({"resource_contract": HISTORICAL_AMARI}, review, sha, job)


@pytest.mark.parametrize(
    "attempts,expected",
    [
        ([], PRIMARY),
        ([(PRIMARY, "PASS")], None),
        ([(PRIMARY, "TIMEOUT")], YICES_RETRY),
        ([(PRIMARY, "MEMORY")], MEMORY_RETRY),
        ([(PRIMARY, "MEMORY"), (MEMORY_RETRY, "PASS")], None),
        ([(PRIMARY, "MEMORY"), (MEMORY_RETRY, "TIMEOUT")], YICES_RETRY),
        ([(PRIMARY, "MEMORY"), (MEMORY_RETRY, "MEMORY")], YICES_RETRY),
        ([(PRIMARY, "TIMEOUT"), (YICES_RETRY, "PASS")], None),
        ([(PRIMARY, "TIMEOUT"), (YICES_RETRY, "TIMEOUT")], None),
        ([(PRIMARY, "TIMEOUT"), (YICES_RETRY, "MEMORY")], None),
        ([(PRIMARY, "MEMORY"), (MEMORY_RETRY, "MEMORY"), (YICES_RETRY, "TIMEOUT")], None),
        ([(PRIMARY, "MEMORY"), (MEMORY_RETRY, "TIMEOUT"), (YICES_RETRY, "PASS")], None),
    ],
)
def test_exact_owner_attempt_sequence(
    attempts: list[tuple[str, str]], expected: str | None
) -> None:
    """TIMEOUT takes Yices; MEMORY40 requires its single Z3 retry first."""
    assert next_attempt(attempts) == expected


@pytest.mark.parametrize("outcome", ["FAIL", "ERROR", "UNKNOWN", "SIGILL", "", "UNPROVEN"])
@pytest.mark.parametrize(
    "prefix,contract",
    [([], PRIMARY), ([(PRIMARY, "MEMORY")], MEMORY_RETRY), ([(PRIMARY, "TIMEOUT")], YICES_RETRY)],
)
def test_bad_outcome_stays_red_at_every_stage(
    prefix: list[tuple[str, str]], contract: str, outcome: str
) -> None:
    """A failure or tool error cannot become permission for a retry."""
    with pytest.raises(ValueError, match="remains RED"):
        next_attempt([*prefix, (contract, outcome)])


@pytest.mark.parametrize(
    "attempts",
    [
        [(YICES_RETRY, "TIMEOUT")],
        [(MEMORY_RETRY, "MEMORY")],
        [(HISTORICAL_AMARI, "PASS")],
        [(PRIMARY, "MEMORY"), (YICES_RETRY, "TIMEOUT")],
        [(PRIMARY, "TIMEOUT"), (MEMORY_RETRY, "PASS")],
        [(PRIMARY, "TIMEOUT"), (PRIMARY, "TIMEOUT")],
        [(PRIMARY, "PASS"), (YICES_RETRY, "PASS")],
        [(PRIMARY, "TIMEOUT"), (YICES_RETRY, "MEMORY"), (YICES_RETRY, "PASS")],
    ],
)
def test_invalid_attempt_order_refuses(attempts: list[tuple[str, str]]) -> None:
    """No repeated Yices or sequence after PASS is implicitly admitted."""
    with pytest.raises(ValueError, match="attempt order"):
        next_attempt(attempts)


@pytest.mark.parametrize("job", [CHIALVO_JOB, AMARI_JOB])
def test_model_specific_solver_repair_budget(job: str) -> None:
    """Keep the crash repair separate from general resource attempt order."""
    entry = {"resource_contract": CHIALVO_REPAIR}
    review = {
        "verdict": "CLEAR",
        "receipt_sha256": "selected-receipt",
        "resource_contract": CHIALVO_REPAIR,
    }
    if job == CHIALVO_JOB:
        contract = select_contract(entry, review, "selected-receipt", job)
        assert contract.solver == "yices-smt2"
        assert contract.policy().memory_bytes == 64 * 1024**3
    else:
        with pytest.raises(ValueError, match="only to Chialvo"):
            select_contract(entry, review, "selected-receipt", job)
    with pytest.raises(ValueError, match="attempt order"):
        next_attempt([(CHIALVO_REPAIR, "PASS")])
