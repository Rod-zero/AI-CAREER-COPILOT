"""Gemini-backed profile analysis service."""

import json
import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()


class LLMServiceError(RuntimeError):
    """Base exception for profile-analysis failures."""


class MissingAPIKeyError(LLMServiceError):
    """Raised when Gemini credentials are not configured."""


class InvalidModelOutputError(LLMServiceError):
    """Raised when Gemini returns malformed or unexpected data."""


class GeminiRequestError(LLMServiceError):
    """Raised when a request to Gemini fails."""


class ProfileAnalysis(BaseModel):
    """Validated structure returned by Gemini."""

    model_config = ConfigDict(extra="forbid", strict=True)

    match_score: int = Field(ge=0, le=100)
    strengths: list[str]
    skill_gaps: list[str]
    next_steps: list[str]


def analyze_profile_with_gemini(
    current_background: str,
    target_role: str,
    skills: list[str],
    project_experience: str,
) -> ProfileAnalysis:
    """Analyze a candidate profile with Gemini and return validated output."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    profile = {
        "current_background": current_background,
        "target_role": target_role,
        "skills": skills,
        "project_experience": project_experience,
    }
    prompt = (
        "Analyze this candidate profile for the target role. Return only JSON with "
        "exactly these fields: match_score (an integer from 0 to 100), strengths "
        "(an array of strings), skill_gaps (an array of strings), and next_steps "
        "(an array of strings).\n\nProfile:\n"
        f"{json.dumps(profile, ensure_ascii=False)}"
    )

    try:
        response = genai.Client(api_key=api_key).models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    except Exception as exc:
        raise GeminiRequestError("Gemini profile analysis request failed.") from exc

    try:
        return ProfileAnalysis.model_validate_json(response.text)
    except (TypeError, ValueError, ValidationError) as exc:
        raise InvalidModelOutputError(
            "Gemini returned invalid profile analysis output."
        ) from exc
