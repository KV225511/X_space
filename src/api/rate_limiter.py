"""AstroFoundation — Turn-Based & Sliding Window Rate Limiter.

Enforces a strict limit of requests per turn (default: 50 requests per turn or
60-second client window), dynamically configured from the environment (.env
via `RATE_LIMIT_PER_TURN` and `RATE_LIMIT_WINDOW_SECONDS`).

Protects model inference workers, GPU memory, and external catalog API quotas.
Returns HTTP 429 Too Many Requests when the limit is exceeded.
"""

import logging
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Automatically load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("astrofoundation.rate_limiter")


class TurnRateLimiter:
    """
    Thread-safe in-memory rate limiter tracking request counts per turn/client.
    Dynamically reads RATE_LIMIT_PER_TURN (default 50) and RATE_LIMIT_WINDOW_SECONDS
    (default 60.0) from the environment.
    """

    def __init__(
        self,
        default_max_requests: int = 50,
        default_window_seconds: float = 60.0,
    ):
        self._default_max_requests = default_max_requests
        self._default_window_seconds = default_window_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    @property
    def max_requests(self) -> int:
        """Dynamically resolve rate limit from environment (default 50)."""
        val = os.environ.get("RATE_LIMIT_PER_TURN")
        if val is not None:
            try:
                return int(val)
            except ValueError:
                logger.warning("Invalid RATE_LIMIT_PER_TURN '%s', using default %d", val, self._default_max_requests)
        return self._default_max_requests

    @max_requests.setter
    def max_requests(self, val: int):
        self._default_max_requests = val

    @property
    def window_seconds(self) -> float:
        """Dynamically resolve sliding window duration from environment (default 60.0s)."""
        val = os.environ.get("RATE_LIMIT_WINDOW_SECONDS")
        if val is not None:
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid RATE_LIMIT_WINDOW_SECONDS '%s', using default %.1f",
                    val,
                    self._default_window_seconds,
                )
        return self._default_window_seconds

    @window_seconds.setter
    def window_seconds(self, val: float):
        self._default_window_seconds = val

    def is_allowed(self, client_key: str) -> Tuple[bool, int, float]:
        """
        Check if the request from client_key is allowed under the current turn quota.

        Args:
            client_key: Unique identifier for the client turn or session.

        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        now = time.time()
        window_duration = self.window_seconds
        limit = self.max_requests
        window_start = now - window_duration

        with self._lock:
            # Evict timestamps outside active sliding window
            timestamps = [t for t in self._history[client_key] if t > window_start]
            self._history[client_key] = timestamps

            if len(timestamps) >= limit:
                # Turn quota exhausted
                oldest = timestamps[0]
                retry_after = max(0.1, round(window_duration - (now - oldest), 1))
                return False, 0, retry_after

            # Allow request and append timestamp
            timestamps.append(now)
            self._history[client_key] = timestamps
            remaining = limit - len(timestamps)
            return True, remaining, 0.0

    def reset(self, client_key: Optional[str] = None):
        """Reset history for a specific client key or all clients."""
        with self._lock:
            if client_key is not None:
                self._history.pop(client_key, None)
            else:
                self._history.clear()

    def reload_env(self):
        """Re-parse .env file into os.environ."""
        load_dotenv(override=True)


# Global rate limiter instance configured with env-backed defaults (50 req/turn)
rate_limiter = TurnRateLimiter(default_max_requests=50, default_window_seconds=60.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware enforcing the turn-based rate limit quota across API endpoints.
    Identifies clients by X-Turn-ID header, X-Session-ID, X-Forwarded-For, or client host IP.
    """

    EXCLUDED_PATHS = {"/docs", "/redoc", "/openapi.json", "/health"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Identify client key: prefers explicit turn header, then session, then IP
        client_key = (
            request.headers.get("X-Turn-ID")
            or request.headers.get("X-Session-ID")
            or request.headers.get("X-Forwarded-For")
            or (request.client.host if request.client else "unknown_client")
        )

        allowed, remaining, retry_after = rate_limiter.is_allowed(client_key)
        limit = rate_limiter.max_requests

        if not allowed:
            logger.warning(
                "Rate limit exceeded for %s: %d requests per turn limit reached",
                client_key,
                limit,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": (
                        f"Maximum {limit} requests per turn allowed. "
                        f"Quota reset in {retry_after} seconds."
                    ),
                    "retry_after_seconds": retry_after,
                    "max_requests_per_turn": limit,
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


def check_turn_rate_limit(client_key: str = "default_turn") -> Tuple[bool, int, float]:
    """Utility function to check turn rate limit programmatically for any API call."""
    return rate_limiter.is_allowed(client_key)
