"""Replaceable in-memory rate limiting for AI-powered endpoints."""

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from ipaddress import ip_address

from fastapi import HTTPException, Request

from backend.config import AI_RATE_LIMIT_PER_HOUR, AI_RATE_LIMIT_PER_MINUTE


class InMemoryRateLimiter:
    """Thread-safe sliding-window limiter keyed by client identity."""

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._clock = clock
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = self._clock()
        with self._lock:
            requests = self._requests[client_id]
            while requests and requests[0] <= now - 3600:
                requests.popleft()
            minute_count = sum(timestamp > now - 60 for timestamp in requests)
            if minute_count >= self.per_minute or len(requests) >= self.per_hour:
                return False
            requests.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()


def get_client_id(request: Request) -> str:
    """Use proxy headers only when the direct peer is explicitly trusted."""
    peer_ip = request.client.host if request.client else "unknown"
    trusted_proxies = {
        value.strip()
        for value in os.getenv("TRUSTED_PROXY_IPS", "").split(",")
        if value.strip()
    }
    if peer_ip in trusted_proxies:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        forwarded_ip = forwarded_for.split(",", 1)[0].strip()
        if forwarded_ip:
            try:
                return str(ip_address(forwarded_ip))
            except ValueError:
                pass
    return peer_ip


ai_rate_limiter = InMemoryRateLimiter(
    per_minute=AI_RATE_LIMIT_PER_MINUTE,
    per_hour=AI_RATE_LIMIT_PER_HOUR,
)


def enforce_ai_rate_limit(request: Request) -> None:
    if not ai_rate_limiter.allow(get_client_id(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many AI requests. Please wait a little and try again.",
        )
