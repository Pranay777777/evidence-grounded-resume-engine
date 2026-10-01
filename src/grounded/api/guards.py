"""What stands between an HTTP request and a paid model call (ADR-016).

- **API keys.** `API_KEYS` holds `name=sha256:daily_token_budget` entries;
  only hashes are configured, never keys. `python -m grounded.api keygen`
  makes one and prints the key once.
- **Per-key budgets.** Tokens a key spends are counted per UTC day; a key
  over its budget gets 429 before any model is called. A cache hit costs 0.
- **Circuit breaker.** After consecutive provider failures the breaker opens:
  requests fail fast with 503 and `Retry-After` instead of piling onto a
  provider that is down or rate limiting, then one trial call closes it again.

The ledger and breaker are per process and in memory - enough for one
instance; step 60 (auth + multi-tenancy) moves budgets into the database.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol


class Budgeted(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def daily_tokens(self) -> int: ...


@dataclass(frozen=True)
class ApiKey:
    name: str
    digest: str
    daily_tokens: int
    tenant: str = "default"


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def parse_keys(config: str) -> list[ApiKey]:
    """`name=sha256hex:budget[:tenant],...` - whitespace ignored."""
    keys = []
    for entry in filter(None, (e.strip() for e in config.split(","))):
        try:
            name, rest = entry.split("=", 1)
            parts = rest.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(entry)
            tenant = parts[2].strip() if len(parts) == 3 else "default"
            key = ApiKey(name.strip(), parts[0].strip().lower(), int(parts[1]), tenant)
        except ValueError as exc:
            raise ValueError(
                f"API_KEYS entry '{entry}' is not name=sha256:budget[:tenant]"
            ) from exc
        if len(key.digest) != 64 or not all(c in "0123456789abcdef" for c in key.digest):
            raise ValueError(f"API_KEYS entry '{key.name}' needs a 64-character sha256 hex digest")
        keys.append(key)
    return keys


def authenticate(presented: str | None, keys: list[ApiKey]) -> ApiKey | None:
    if not presented:
        return None
    candidate = digest(presented)
    match = None
    for key in keys:  # compare against every key: no early exit to time
        if hmac.compare_digest(candidate, key.digest):
            match = key
    return match


def new_key() -> str:
    return "gr_" + secrets.token_urlsafe(32)


@dataclass
class Ledger:
    """Tokens spent per key per UTC day."""

    today: Callable[[], date] = lambda: datetime.now(UTC).date()
    spent: dict[tuple[str, date], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def used(self, key: Budgeted) -> int:
        return self.spent.get((key.name, self.today()), 0)

    def remaining(self, key: Budgeted) -> int:
        return max(key.daily_tokens - self.used(key), 0)

    def charge(self, key: Budgeted, tokens: int) -> None:
        with self._lock:
            slot = (key.name, self.today())
            self.spent[slot] = self.spent.get(slot, 0) + tokens


class CircuitOpenError(RuntimeError):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"provider circuit open; retry in {retry_after:.0f}s")
        self.retry_after = retry_after


@dataclass
class CircuitBreaker:
    failures_to_open: int = 3
    cooldown_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self.opened_at >= self.cooldown_s else "open"

    def before_call(self) -> None:
        """Raise if open; in half-open, let one trial through."""
        with self._lock:
            if self.opened_at is not None:
                waited = self.clock() - self.opened_at
                if waited < self.cooldown_s:
                    raise CircuitOpenError(self.cooldown_s - waited)

    def success(self) -> None:
        with self._lock:
            self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failures_to_open or self.opened_at is not None:
                self.opened_at = self.clock()
