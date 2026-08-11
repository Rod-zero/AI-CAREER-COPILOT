from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.agent.career_agent import CareerAgentResult
from backend.main import app
from backend.rate_limiter import InMemoryRateLimiter
from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
)

client = TestClient(app)
REQUEST = {
    "user_request": "Assess my fit for this role.",
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services using Python.",
    "skills": ["Python", "APIs"],
    "project_experience": "Built a production API.",
}


@patch("backend.main.run_career_agent")
def test_career_agent_endpoint_returns_result_and_delegates(mock_run: Mock) -> None:
    mock_run.return_value = CareerAgentResult(
        final_answer="Your Python API experience is relevant.",
        tools_used=["analyze_candidate_fit"],
        tool_call_count=1,
    )

    response = client.post("/career-agent", json=REQUEST)

    assert response.status_code == 200
    assert response.json() == {
        "final_answer": "Your Python API experience is relevant.",
        "tools_used": ["analyze_candidate_fit"],
        "tool_call_count": 1,
    }
    mock_run.assert_called_once_with(**REQUEST)


@pytest.mark.parametrize(
    ("field", "expected_detail"),
    [
        ("user_request", "User request must not be empty."),
        ("job_description", "Job description must not be empty."),
    ],
)
@patch("backend.main.run_career_agent")
def test_career_agent_endpoint_rejects_whitespace_only_required_text(
    mock_run: Mock,
    field: str,
    expected_detail: str,
) -> None:
    request = {**REQUEST, field: "   "}

    response = client.post("/career-agent", json=request)

    assert response.status_code == 400
    assert response.json() == {"detail": expected_detail}
    mock_run.assert_not_called()


@pytest.mark.parametrize(
    ("agent_error", "expected_status", "expected_detail"),
    [
        (
            MissingAPIKeyError("secret internal detail"),
            500,
            "LLM Career Agent is not configured.",
        ),
        (
            InvalidModelOutputError("raw model output"),
            502,
            "The LLM service returned an invalid response.",
        ),
        (
            GeminiRequestError("SDK failure detail"),
            502,
            "The LLM service is currently unavailable.",
        ),
    ],
)
@patch("backend.main.run_career_agent")
def test_career_agent_endpoint_maps_agent_errors(
    mock_run: Mock,
    agent_error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    mock_run.side_effect = agent_error

    response = client.post("/career-agent", json=REQUEST)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_career_agent_endpoint_uses_ai_rate_limiter() -> None:
    route = next(route for route in app.routes if route.path == "/career-agent")

    assert any(
        dependency.call is not None
        and dependency.call.__name__ == "enforce_ai_rate_limit"
        for dependency in route.dependant.dependencies
    )


@patch("backend.main.run_career_agent")
def test_career_agent_endpoint_returns_429_when_rate_limited(
    mock_run: Mock,
    monkeypatch,
) -> None:
    mock_run.return_value = CareerAgentResult(
        final_answer="Relevant recommendation.",
        tools_used=[],
        tool_call_count=0,
    )
    limiter = InMemoryRateLimiter(1, 20, clock=lambda: 0.0)
    monkeypatch.setattr("backend.rate_limiter.ai_rate_limiter", limiter)

    assert client.post("/career-agent", json=REQUEST).status_code == 200
    response = client.post("/career-agent", json=REQUEST)

    assert response.status_code == 429
    assert response.json() == {
        "detail": "Too many AI requests. Please wait a little and try again."
    }
    mock_run.assert_called_once()
