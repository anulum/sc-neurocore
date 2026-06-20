# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser-login throttling

"""Browser-login throttling contracts for SC-NeuroCore Studio."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
THROTTLED_BROWSER_LOGIN_REASON = "browser_login_throttled"


@dataclass(frozen=True, slots=True)
class StudioLoginThrottleDecision:
    """Decision returned before a browser login attempt is evaluated.

    Parameters
    ----------
    allowed:
        Whether the login attempt may proceed to password verification.
    reason:
        Stable denial reason for audit rows and API responses.
    retry_after_seconds:
        Whole-second cooldown remaining when the attempt is denied.
    """

    allowed: bool
    reason: str | None = None
    retry_after_seconds: int | None = None


@dataclass(slots=True)
class _ThrottleBucket:
    """Mutable failure bucket for one normalized browser-login key."""

    failure_times: list[datetime]
    locked_until: datetime | None = None


class StudioBrowserLoginThrottle:
    """Windowed lockout guard for Studio browser-login attempts.

    The guard stores only normalized login keys and failure timestamps in memory.
    It does not persist or inspect password material. A successful login clears
    the bucket for the normalized key.
    """

    def __init__(
        self,
        *,
        max_failed_attempts: int,
        failure_window_seconds: float,
        cooldown_seconds: float,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_failed_attempts <= 0:
            raise ValueError("Studio login throttle max failed attempts must be positive.")
        if failure_window_seconds <= 0:
            raise ValueError("Studio login throttle failure window must be positive.")
        if cooldown_seconds <= 0:
            raise ValueError("Studio login throttle cooldown must be positive.")
        self._max_failed_attempts = max_failed_attempts
        self._failure_window = timedelta(seconds=failure_window_seconds)
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._clock = clock or self._utc_now
        self._buckets: dict[str, _ThrottleBucket] = {}

    def check(self, username: str) -> StudioLoginThrottleDecision:
        """Return whether a browser-login attempt may proceed."""

        key = self._key(username)
        now = self._now()
        bucket = self._buckets.get(key)
        if bucket is None:
            return StudioLoginThrottleDecision(allowed=True)
        self._prune(bucket, now)
        if bucket.locked_until is not None and now < bucket.locked_until:
            remaining = int((bucket.locked_until - now).total_seconds())
            return StudioLoginThrottleDecision(
                allowed=False,
                reason=THROTTLED_BROWSER_LOGIN_REASON,
                retry_after_seconds=max(1, remaining),
            )
        if not bucket.failure_times:
            self._buckets.pop(key, None)
        return StudioLoginThrottleDecision(allowed=True)

    def record_failure(self, username: str) -> StudioLoginThrottleDecision:
        """Record a failed browser-login attempt and return the new state."""

        key = self._key(username)
        now = self._now()
        bucket = self._buckets.setdefault(key, _ThrottleBucket(failure_times=[]))
        self._prune(bucket, now)
        bucket.failure_times.append(now)
        if len(bucket.failure_times) >= self._max_failed_attempts:
            bucket.locked_until = now + self._cooldown
            return StudioLoginThrottleDecision(
                allowed=False,
                reason=THROTTLED_BROWSER_LOGIN_REASON,
                retry_after_seconds=max(1, int(self._cooldown.total_seconds())),
            )
        return StudioLoginThrottleDecision(allowed=True)

    def record_success(self, username: str) -> None:
        """Clear the failure bucket after a successful browser login."""

        self._buckets.pop(self._key(username), None)

    def _prune(self, bucket: _ThrottleBucket, now: datetime) -> None:
        cutoff = now - self._failure_window
        bucket.failure_times = [value for value in bucket.failure_times if value >= cutoff]
        if bucket.locked_until is not None and now >= bucket.locked_until:
            bucket.locked_until = None

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    @staticmethod
    def _key(username: str) -> str:
        cleaned = username.strip().casefold()
        return cleaned if cleaned else "<blank>"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)
