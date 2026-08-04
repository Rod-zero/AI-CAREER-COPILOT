from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Career Copilot API")


class ProfileAnalysisRequest(BaseModel):
    current_background: str
    target_role: str
    skills: list[str]
    project_experience: str


class ProfileAnalysisResponse(BaseModel):
    match_score: int
    strengths: list[str]
    skill_gaps: list[str]
    next_steps: list[str]


ROLE_SKILLS = {
    "agentic_ai_engineer": {
        "python",
        "machine learning",
        "llm",
        "rag",
        "langchain",
        "apis",
        "git",
        "testing",
        "agent orchestration",
        "llm evaluation",
    },
    "ai_engineer": {"python", "machine learning", "llm", "rag", "apis", "git", "testing"},
    "machine_learning_engineer": {
        "python",
        "machine learning",
        "sql",
        "git",
        "testing",
        "apis",
        "deployment",
    },
    "data_scientist": {"python", "sql", "statistics", "machine learning", "data analysis"},
    "data": {"python", "sql", "statistics", "machine learning"},
    "software": {"python", "git", "testing", "apis"},
    "frontend": {"javascript", "typescript", "react", "css"},
    "product": {"product strategy", "analytics", "user research", "communication"},
}
ROLE_MATCHERS = (
    (("agentic",), "agentic_ai_engineer"),
    (("ai engineer",), "ai_engineer"),
    (("machine learning", "ml engineer"), "machine_learning_engineer"),
    (("data scientist",), "data_scientist"),
    (("data",), "data"),
    (("software",), "software"),
    (("frontend",), "frontend"),
    (("product",), "product"),
)
DEFAULT_SKILLS = {"communication", "problem solving", "project management"}


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}


@app.post("/analyze-profile", response_model=ProfileAnalysisResponse)
def analyze_profile(profile: ProfileAnalysisRequest) -> ProfileAnalysisResponse:
    """Return a deterministic first-pass assessment of a candidate profile."""
    target_role = profile.target_role.lower()
    expected_skills = next(
        (
            ROLE_SKILLS[role]
            for keywords, role in ROLE_MATCHERS
            if any(keyword in target_role for keyword in keywords)
        ),
        DEFAULT_SKILLS,
    )
    candidate_skills = {skill.strip().lower() for skill in profile.skills if skill.strip()}
    matched_skills = sorted(expected_skills & candidate_skills)
    missing_skills = sorted(expected_skills - candidate_skills)

    skill_score = len(matched_skills) / len(expected_skills) * 80
    score = skill_score
    if profile.current_background.strip():
        score += 10
    if len(profile.project_experience.strip()) >= 50:
        score += 10

    strengths = [f"Demonstrated skill: {skill.title()}" for skill in matched_skills]
    if profile.project_experience.strip():
        strengths.append("Hands-on project experience")
    if not strengths:
        strengths.append("Clear interest in the target role")

    skill_gaps = [skill.title() for skill in missing_skills]
    next_steps = [f"Build a small project using {skill.title()}." for skill in missing_skills[:2]]
    next_steps.append(f"Tailor your resume to highlight experience relevant to {profile.target_role}.")

    return ProfileAnalysisResponse(
        match_score=min(round(score), 100),
        strengths=strengths,
        skill_gaps=skill_gaps,
        next_steps=next_steps,
    )
