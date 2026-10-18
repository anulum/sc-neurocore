# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser-login throttle tests

"""Verify finite login windows, cooldown recovery and secret-free throttle status."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sc_neurocore.studio.platform.auth_throttle import (
    THROTTLED_BROWSER_LOGIN_REASON,
    StudioBrowserLoginThrottle,
)

UTC = timezone.utc


def test_studio_browser_login_throttle_locks_after_configured_failures() -> None:
    """Share the failure budget across normalized spellings of the same login."""
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=300.0,
        cooldown_seconds=60.0,
        clock=lambda: current,
    )

    assert throttle.check("Operator").allowed
    assert throttle.record_failure(" operator ").allowed
    locked = throttle.record_failure("OPERATOR")
    denied = throttle.check("operator")

    assert not locked.allowed
    assert locked.reason == THROTTLED_BROWSER_LOGIN_REASON
    assert locked.retry_after_seconds == 60
    assert not denied.allowed
    assert denied.reason == THROTTLED_BROWSER_LOGIN_REASON


def test_studio_browser_login_throttle_snapshot_reports_secret_free_counts() -> None:
    """Prune expired failures and report aggregate cooldown state without login names."""
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=300.0,
        cooldown_seconds=60.0,
        clock=lambda: current,
    )

    throttle.record_failure("operator")
    throttle.record_failure("operator")
    throttle.record_failure("other")

    assert throttle.snapshot().to_public_dict() == {
        "active_bucket_count": 2,
        "locked_bucket_count": 1,
        "max_retry_after_seconds": 60,
    }
    current = current + timedelta(seconds=61)
    assert throttle.snapshot().to_public_dict() == {
        "active_bucket_count": 1,
        "locked_bucket_count": 0,
        "max_retry_after_seconds": 0,
    }
    current = current + timedelta(seconds=300)
    assert throttle.snapshot().to_public_dict() == {
        "active_bucket_count": 0,
        "locked_bucket_count": 0,
        "max_retry_after_seconds": 0,
    }


def test_studio_browser_login_throttle_expires_failure_window() -> None:
    """Discard failures outside the configured window before admitting another attempt."""
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=10.0,
        cooldown_seconds=60.0,
        clock=lambda: current,
    )

    throttle.record_failure("operator")
    current = current + timedelta(seconds=11)
    decision = throttle.record_failure("operator")

    assert decision.allowed
    assert throttle.check("operator").allowed


def test_studio_browser_login_throttle_reset_after_success() -> None:
    """Clear previous failures when a browser login succeeds."""
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=300.0,
        cooldown_seconds=60.0,
        clock=lambda: current,
    )

    throttle.record_failure("operator")
    throttle.record_success("operator")
    decision = throttle.record_failure("operator")

    assert decision.allowed


def test_studio_browser_login_throttle_unlocks_after_cooldown() -> None:
    """Admit another attempt once the configured cooldown has elapsed."""
    current = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=1,
        failure_window_seconds=300.0,
        cooldown_seconds=60.0,
        clock=lambda: current,
    )

    throttle.record_failure("operator")
    current = current + timedelta(seconds=61)

    assert throttle.check("operator").allowed


def test_studio_browser_login_throttle_rejects_invalid_max_failures() -> None:
    """Refuse a throttle configuration with no positive failure budget."""
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=0,
            failure_window_seconds=300.0,
            cooldown_seconds=60.0,
        )


def test_studio_browser_login_throttle_rejects_invalid_failure_window() -> None:
    """Refuse a throttle configuration with no positive failure window."""
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=1,
            failure_window_seconds=0.0,
            cooldown_seconds=60.0,
        )


def test_studio_browser_login_throttle_rejects_invalid_cooldown() -> None:
    """Refuse a throttle configuration with no positive cooldown."""
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=1,
            failure_window_seconds=300.0,
            cooldown_seconds=0.0,
        )


def test_pending_login_reservations_are_never_evicted_for_another_user() -> None:
    """Refuse a new key when every finite registry slot has an active verifier."""
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=1, failure_window_seconds=300, cooldown_seconds=60, max_buckets=2
    )
    with (
        throttle.attempt("first", client_id="local") as first,
        throttle.attempt("second", client_id="local") as second,
    ):
        assert first.allowed and second.allowed
        with throttle.attempt("third", client_id="local") as third:
            assert not third.allowed
            assert third.reason == THROTTLED_BROWSER_LOGIN_REASON
        assert not throttle.record_failure("third", client_id="local").allowed
        assert not throttle.record_failure("first", client_id="local").allowed
        assert not throttle.record_failure("first", client_id="local").allowed
        assert throttle.snapshot().active_bucket_count == 2
        assert throttle.snapshot().locked_bucket_count == 1
    assert throttle.snapshot().active_bucket_count == 1
    throttle.record_success("absent", client_id="local")
    assert throttle.snapshot().locked_bucket_count == 1
    throttle.record_success("first", client_id="local")
    assert throttle.snapshot().active_bucket_count == 0


def test_login_registry_requires_a_positive_slot_budget() -> None:
    """Refuse a registry which cannot retain even one login key."""
    with pytest.raises(ValueError, match="bucket cap must be positive"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=1,
            failure_window_seconds=300,
            cooldown_seconds=60,
            max_buckets=0,
        )


def test_login_key_churn_preserves_live_failures_and_lockouts_at_default_capacity() -> None:
    """Refuse new keys after saturation without resetting existing failure budgets."""
    current = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=300,
        cooldown_seconds=60,
        clock=lambda: current,
    )
    throttle.record_failure("locked")
    assert not throttle.record_failure("locked").allowed
    throttle.record_failure("failing")
    for index in range(1022):
        assert throttle.record_failure(f"retained-{index}").allowed
    for index in range(1025):
        with throttle.attempt(f"new-{index}", client_id="local") as decision:
            assert not decision.allowed
            assert decision.reason == THROTTLED_BROWSER_LOGIN_REASON
        assert not throttle.record_failure(f"new-{index}", client_id="local").allowed
    assert throttle.snapshot().active_bucket_count == 1024
    assert throttle.snapshot().locked_bucket_count == 1
    assert not throttle.check("locked").allowed
    assert not throttle.record_failure("failing").allowed
    assert throttle.snapshot().locked_bucket_count == 2


@pytest.mark.parametrize("reset", ["success", "admin", "cooldown"])
def test_login_registry_reclaims_cleared_or_expired_lockout_slots(reset: str) -> None:
    """Reuse a slot after successful login, password rotation or cooldown expiry."""
    current = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=1,
        failure_window_seconds=10,
        cooldown_seconds=60,
        max_buckets=1,
        clock=lambda: current,
    )
    assert not throttle.record_failure("locked").allowed
    current += timedelta(seconds=11)
    with throttle.attempt("new", client_id="local") as denied:
        assert not denied.allowed
    if reset == "success":
        throttle.record_success("locked")
    elif reset == "admin":
        throttle.clear_user("locked")
    else:
        current += timedelta(seconds=50)
    with throttle.attempt("new", client_id="local") as admitted:
        assert admitted.allowed
    assert throttle.snapshot().active_bucket_count == 0


def test_login_registry_reclaims_an_expired_failure_window_without_a_snapshot() -> None:
    """Admit a new key when the full registry's sole failure has naturally expired."""
    current = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=2,
        failure_window_seconds=10,
        cooldown_seconds=60,
        max_buckets=1,
        clock=lambda: current,
    )
    assert throttle.record_failure("failing").allowed
    current += timedelta(seconds=11)
    with throttle.attempt("new", client_id="local") as admitted:
        assert admitted.allowed
        assert throttle.record_failure("new", client_id="local").allowed
    assert throttle.snapshot().active_bucket_count == 1
