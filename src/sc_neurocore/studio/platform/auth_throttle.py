# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio browser-login throttling

"""Browser-login throttling contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


@dataclass(frozen=True, slots=True)
class StudioLoginThrottleSnapshot:
    """Secret-free aggregate state for browser-login throttling.

    Parameters
    ----------
    active_bucket_count:
        Number of normalized login keys with live failure or lockout state.
    locked_bucket_count:
        Number of live buckets currently locked out.
    max_retry_after_seconds:
        Largest remaining retry interval across locked buckets, or ``0`` when
        no bucket is locked.
    """

    active_bucket_count: int
    locked_bucket_count: int
    max_retry_after_seconds: int

    def to_public_dict(self) -> dict[str, int]:
        """Return a secret-free throttle aggregate for operator APIs."""
        return {
            "active_bucket_count": self.active_bucket_count,
            "locked_bucket_count": self.locked_bucket_count,
            "max_retry_after_seconds": self.max_retry_after_seconds,
        }


@dataclass(slots=True)
class _ThrottleBucket:
    """Mutable failure bucket for one normalized browser-login key."""

    failure_times: list[datetime]
    locked_until: datetime | None = None
    pending: int = 0


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
        max_buckets: int = 1024,
    ) -> None:
        if max_failed_attempts <= 0:
            raise ValueError("Studio login throttle max failed attempts must be positive.")
        if failure_window_seconds <= 0:
            raise ValueError("Studio login throttle failure window must be positive.")
        if cooldown_seconds <= 0:
            raise ValueError("Studio login throttle cooldown must be positive.")
        if max_buckets <= 0:
            raise ValueError("Studio login throttle bucket cap must be positive.")
        self._max_failed_attempts = max_failed_attempts
        self._failure_window = timedelta(seconds=failure_window_seconds)
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._clock = clock or self._utc_now
        self._max_buckets = max_buckets
        self._guard = threading.RLock()
        self._buckets: OrderedDict[tuple[str, str], _ThrottleBucket] = OrderedDict()

    @contextmanager
    def attempt(self, username: str, *, client_id: str) -> Iterator[StudioLoginThrottleDecision]:
        """Reserve an attempt before verification, bounded per username and direct client.

        Concurrent verifiers share the failure budget. A full registry reclaims
        only expired or cleared keys; live failures, lockouts and in-flight
        keys remain protected. Proxy headers do not define client identity.
        """
        bucket: _ThrottleBucket | None = None
        with self._guard:
            decision = self.check(username, client_id=client_id)
            if decision.allowed:
                bucket = self._bucket(self._key(username, client_id))
                if (
                    bucket is None
                    or len(bucket.failure_times) + bucket.pending >= self._max_failed_attempts
                ):
                    decision = StudioLoginThrottleDecision(False, THROTTLED_BROWSER_LOGIN_REASON, 1)
                    bucket = None
                else:
                    bucket.pending += 1
        try:
            yield decision
        finally:
            if bucket is not None:
                with self._guard:
                    bucket.pending -= 1

    def _bucket(self, key: tuple[str, str]) -> _ThrottleBucket | None:
        """Return an LRU bucket without discarding live failure or reservation state."""
        if key in self._buckets:
            self._buckets.move_to_end(key)
            return self._buckets[key]
        if len(self._buckets) >= self._max_buckets:
            idle = None
            now = self._now()
            for candidate, value in self._buckets.items():
                self._prune(value, now)
                if not value.pending and not value.failure_times and value.locked_until is None:
                    idle = candidate
                    break
            if idle is None:
                return None
            del self._buckets[idle]
        bucket = _ThrottleBucket(failure_times=[])
        self._buckets[key] = bucket
        return bucket

    def check(self, username: str, *, client_id: str = "") -> StudioLoginThrottleDecision:
        """Return whether a browser-login attempt may proceed."""
        with self._guard:
            key = self._key(username, client_id)
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
            if not bucket.failure_times and not bucket.pending:
                self._buckets.pop(key, None)
            return StudioLoginThrottleDecision(allowed=True)

    def record_failure(self, username: str, *, client_id: str = "") -> StudioLoginThrottleDecision:
        """Record a failed browser-login attempt and return the new state."""
        with self._guard:
            key = self._key(username, client_id)
            now = self._now()
            bucket = self._bucket(key)
            if bucket is None:
                return StudioLoginThrottleDecision(False, THROTTLED_BROWSER_LOGIN_REASON, 1)
            self._prune(bucket, now)
            if len(bucket.failure_times) < self._max_failed_attempts:
                bucket.failure_times.append(now)
            if len(bucket.failure_times) >= self._max_failed_attempts:
                bucket.locked_until = now + self._cooldown
                return StudioLoginThrottleDecision(
                    allowed=False,
                    reason=THROTTLED_BROWSER_LOGIN_REASON,
                    retry_after_seconds=max(1, int(self._cooldown.total_seconds())),
                )
            return StudioLoginThrottleDecision(allowed=True)

    def record_success(self, username: str, *, client_id: str = "") -> None:
        """Clear the failure bucket after a successful browser login."""
        with self._guard:
            bucket = self._buckets.get(self._key(username, client_id))
            if bucket is not None:
                bucket.failure_times.clear()
                bucket.locked_until = None

    def snapshot(self) -> StudioLoginThrottleSnapshot:
        """Return aggregate lockout state without exposing login keys."""
        with self._guard:
            now = self._now()
            active_bucket_count = 0
            locked_bucket_count = 0
            max_retry_after_seconds = 0
            stale_keys: list[tuple[str, str]] = []
            for key, bucket in self._buckets.items():
                self._prune(bucket, now)
                if not bucket.failure_times and bucket.locked_until is None and not bucket.pending:
                    stale_keys.append(key)
                    continue
                active_bucket_count += 1
                if bucket.locked_until is not None and now < bucket.locked_until:
                    locked_bucket_count += 1
                    retry_after_seconds = max(
                        1,
                        int((bucket.locked_until - now).total_seconds()),
                    )
                    max_retry_after_seconds = max(
                        max_retry_after_seconds,
                        retry_after_seconds,
                    )
            for key in stale_keys:
                self._buckets.pop(key, None)
            return StudioLoginThrottleSnapshot(
                active_bucket_count=active_bucket_count,
                locked_bucket_count=locked_bucket_count,
                max_retry_after_seconds=max_retry_after_seconds,
            )

    def clear_user(self, username: str) -> None:
        """Clear failure state for every client after an administrator rotates a password."""
        normalized, _ = self._key(username, "")
        with self._guard:
            for key, bucket in self._buckets.items():
                if key[0] == normalized:
                    bucket.failure_times.clear()
                    bucket.locked_until = None

    def _prune(self, bucket: _ThrottleBucket, now: datetime) -> None:
        cutoff = now - self._failure_window
        bucket.failure_times = [value for value in bucket.failure_times if value >= cutoff]
        if bucket.locked_until is not None and now >= bucket.locked_until:
            bucket.locked_until = None
            bucket.failure_times.clear()

    def _now(self) -> datetime:
        return self._clock().astimezone(UTC)

    @staticmethod
    def _key(username: str, client_id: str) -> tuple[str, str]:
        cleaned = username.strip().casefold()
        return cleaned if cleaned else "<blank>", client_id

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)
