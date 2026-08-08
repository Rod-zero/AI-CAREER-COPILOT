import pytest

from backend.rate_limiter import ai_rate_limiter


@pytest.fixture(autouse=True)
def reset_ai_rate_limiter() -> None:
    ai_rate_limiter.reset()

