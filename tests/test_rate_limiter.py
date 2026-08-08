from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from starlette.requests import Request

from backend.main import app
from backend.rate_limiter import InMemoryRateLimiter, get_client_id
from backend.services.llm_service import StructuredJobDescription


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_requests_within_minute_limit_are_allowed() -> None:
    limiter = InMemoryRateLimiter(5, 20, clock=FakeClock())

    assert [limiter.allow("client-a") for _ in range(5)] == [True] * 5


def test_minute_limit_is_enforced_without_sleeping() -> None:
    limiter = InMemoryRateLimiter(5, 20, clock=FakeClock())

    for _ in range(5):
        assert limiter.allow("client-a")

    assert not limiter.allow("client-a")


def test_hour_limit_is_enforced_without_sleeping() -> None:
    clock = FakeClock()
    limiter = InMemoryRateLimiter(100, 2, clock=clock)
    assert limiter.allow("client-a")
    clock.now = 61
    assert limiter.allow("client-a")
    clock.now = 122

    assert not limiter.allow("client-a")


def test_different_clients_have_separate_limits() -> None:
    limiter = InMemoryRateLimiter(1, 1, clock=FakeClock())

    assert limiter.allow("client-a")
    assert not limiter.allow("client-a")
    assert limiter.allow("client-b")


def test_forwarded_ip_is_used_only_for_trusted_proxy(monkeypatch) -> None:
    scope = {
        "type": "http",
        "client": ("10.0.0.1", 1234),
        "headers": [(b"x-forwarded-for", b"203.0.113.5, 10.0.0.1")],
    }
    request = Request(scope)
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)
    assert get_client_id(request) == "10.0.0.1"

    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1")
    assert get_client_id(request) == "203.0.113.5"


def test_all_gemini_endpoints_are_rate_limited() -> None:
    protected_paths = {
        "/extract-jd",
        "/analyze-resume/llm",
        "/analyze-profile/llm",
        "/tailor-resume",
        "/tailor-resume/upload",
    }
    protected_route_paths = {
        route.path
        for route in app.routes
        if route.path in protected_paths
        and any(
            dependency.call is not None
            and dependency.call.__name__ == "enforce_ai_rate_limit"
            for dependency in route.dependant.dependencies
        )
    }

    assert protected_route_paths == protected_paths


@patch("backend.main.extract_jd_with_gemini")
def test_protected_endpoint_returns_429(mock_extract: Mock, monkeypatch) -> None:
    extraction = StructuredJobDescription(
        job_title=None,
        seniority_level=None,
        responsibilities=[],
        required_skills=[],
        preferred_skills=[],
        required_experience=[],
        preferred_experience=[],
        education_requirements=[],
        tools_and_technologies=[],
        domain_knowledge=[],
        soft_skills=[],
    )
    mock_extract.return_value = extraction
    limiter = InMemoryRateLimiter(1, 20, clock=FakeClock())
    monkeypatch.setattr("backend.rate_limiter.ai_rate_limiter", limiter)
    client = TestClient(app)

    assert client.post("/extract-jd", json={"job_description": "Join us."}).status_code == 200
    response = client.post("/extract-jd", json={"job_description": "Join us."})

    assert response.status_code == 429
    assert response.json() == {
        "detail": "Too many AI requests. Please wait a little and try again."
    }
    mock_extract.assert_called_once()
