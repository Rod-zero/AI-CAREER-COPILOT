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
    job_description: str,
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
        "job_description": job_description,
        "skills": skills,
        "project_experience": project_experience,
    }
    prompt = (
        "Analyze the candidate primarily against the supplied job description. "
        "Identify its important required and preferred qualifications, then compare "
        "them only against evidence explicitly supplied in the candidate profile. "
        "Treat only explicitly stated candidate information as confirmed evidence. "
        "Do not infer a specific platform, tool, cloud environment, or depth of "
        "experience unless it is stated. Do not invent requirements unsupported by "
        "the job description. Use 'Confirmed skill gap' only when the candidate "
        "explicitly states that they lack the skill or experience. When candidate "
        "evidence for a qualification is absent, label it 'Not evidenced in the "
        "supplied profile.' Absence of evidence is not proof that the candidate "
        "lacks the qualification. Calculate a calibrated match_score from 0 to 100, "
        "weighting required job qualifications more heavily than preferred ones; a "
        "missing preferred qualification should reduce the score only modestly. "
        "Prioritize the most important requirements from the job description. Return "
        "at most 5 strengths, 5 skill_gaps, and 5 next_steps, with no overlapping or "
        "repetitive bullets. Order next_steps by highest expected impact on the "
        "candidate's fit. Write every next_steps item directly to the candidate with "
        "an actionable verb such as "
        "build, add, learn, demonstrate, quantify, tailor, or document. Never write "
        "next_steps as questions or actions for a recruiter or interviewer. Prefer "
        "concrete resume, portfolio, project, and learning actions. Keep all results "
        "concise and actionable. Do not include or recommend hidden chain-of-thought "
        "or private reasoning. Return only JSON with exactly these fields: "
        "match_score (an integer from 0 to 100), strengths (an array of strings), "
        "skill_gaps (an array of strings), and next_steps (an array of strings)."
        "\n\nCandidate profile and job description:\n"
        f"{json.dumps(profile, ensure_ascii=False)}"
    )

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
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
