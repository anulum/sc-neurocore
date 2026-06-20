# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser-login throttle tests

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sc_neurocore.studio.platform.auth_throttle import (
    THROTTLED_BROWSER_LOGIN_REASON,
    StudioBrowserLoginThrottle,
)

UTC = timezone.utc


def test_studio_browser_login_throttle_locks_after_configured_failures() -> None:
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


def test_studio_browser_login_throttle_expires_failure_window() -> None:
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
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=0,
            failure_window_seconds=300.0,
            cooldown_seconds=60.0,
        )


def test_studio_browser_login_throttle_rejects_invalid_failure_window() -> None:
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=1,
            failure_window_seconds=0.0,
            cooldown_seconds=60.0,
        )


def test_studio_browser_login_throttle_rejects_invalid_cooldown() -> None:
    with pytest.raises(ValueError, match="Studio login throttle"):
        StudioBrowserLoginThrottle(
            max_failed_attempts=1,
            failure_window_seconds=300.0,
            cooldown_seconds=0.0,
        )
