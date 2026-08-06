from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "frontend" / "app.py"
PROFILE_VALUES = {
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "skills_text": "Python, APIs",
    "project_experience": "Built a production API.",
}
ANALYSIS = {
    "match_score": 82,
    "strengths": ["Python"],
    "skill_gaps": ["LLM evaluation"],
    "next_steps": ["Build an LLM project"],
}


def _filled_app(mode: str | None = None) -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run()
    if mode is not None:
        app.radio[0].set_value(mode)
    app.text_area[0].set_value(PROFILE_VALUES["current_background"])
    app.text_input[0].set_value(PROFILE_VALUES["target_role"])
    app.text_input[1].set_value(PROFILE_VALUES["skills_text"])
    app.text_area[1].set_value(PROFILE_VALUES["project_experience"])
    return app


@pytest.mark.parametrize(
    ("mode", "expected_path", "expected_label"),
    [
        (None, "/analyze-profile/llm", "Gemini AI analysis"),
        ("Rule-based analysis", "/analyze-profile", "Rule-based analysis"),
    ],
)
@patch("requests.post")
def test_frontend_submits_selected_analysis_mode(
    mock_post: Mock,
    mode: str | None,
    expected_path: str,
    expected_label: str,
) -> None:
    mock_post.return_value.json.return_value = ANALYSIS

    app = _filled_app(mode)
    app.button[0].click().run()

    assert not app.exception
    assert app.radio[0].value == (mode or "AI analysis")
    mock_post.assert_called_once_with(
        f"http://localhost:8000{expected_path}",
        json={
            "current_background": PROFILE_VALUES["current_background"],
            "target_role": PROFILE_VALUES["target_role"],
            "skills": ["Python", "APIs"],
            "project_experience": PROFILE_VALUES["project_experience"],
        },
        timeout=10,
    )
    assert app.info[0].value == f"Result source: {expected_label}"
    assert app.metric[0].value == "82%"


@patch("requests.post")
def test_frontend_shows_safe_error_and_preserves_form_data(mock_post: Mock) -> None:
    mock_post.side_effect = requests.HTTPError("secret backend detail")

    app = _filled_app()
    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == (
        "The profile analysis could not be completed. Please try again."
    )
    assert "secret backend detail" not in app.error[0].value
    assert app.text_area[0].value == PROFILE_VALUES["current_background"]
    assert app.text_input[0].value == PROFILE_VALUES["target_role"]
    assert app.text_input[1].value == PROFILE_VALUES["skills_text"]
    assert app.text_area[1].value == PROFILE_VALUES["project_experience"]
