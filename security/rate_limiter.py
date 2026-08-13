"""
Rate Limiter & Download Throttling for Ardhanarishvara.
Enforces MANDATORY SECURITY REQUIREMENT #1:
- Ensures all dataset API calls and external downloads observe rate limits and backoff pauses.
"""

import time
from functools import wraps
from security.sanitized_logging import RateLimitExceededException, log_info

_LAST_CALL_TIMESTAMP = 0.0
MIN_CALL_INTERVAL_SEC = 0.5  # minimum 500ms delay between consecutive download requests
MAX_CALLS_PER_MINUTE = 60
_CALL_TIMESTAMPS = []


def rate_limit_downloads(max_per_min: int = MAX_CALLS_PER_MINUTE, min_interval_sec: float = MIN_CALL_INTERVAL_SEC):
    """Decorator to rate-limit external API calls / dataset fetching functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _LAST_CALL_TIMESTAMP, _CALL_TIMESTAMPS

            now = time.time()
            
            # Remove timestamps older than 60 seconds
            _CALL_TIMESTAMPS = [t for t in _CALL_TIMESTAMPS if now - t < 60.0]

            if len(_CALL_TIMESTAMPS) >= max_per_min:
                raise RateLimitExceededException(
                    f"Rate limit exceeded: maximum {max_per_min} requests per minute allowed."
                )

            # Enforce minimum spacing interval
            elapsed = now - _LAST_CALL_TIMESTAMP
            if elapsed < min_interval_sec:
                sleep_time = min_interval_sec - elapsed
                time.sleep(sleep_time)
                now = time.time()

            _LAST_CALL_TIMESTAMP = now
            _CALL_TIMESTAMPS.append(now)

            log_info(f"[RateLimiter] Call permitted to {func.__name__} (requests in last minute: {len(_CALL_TIMESTAMPS)})")
            return func(*args, **kwargs)
        return wrapper
    return decorator
