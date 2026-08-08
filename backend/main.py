from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
    ScoreBreakdown,
    analyze_profile_with_gemini,
    tailor_resume_with_gemini,
)
from backend.services.resume_parser import extract_text_from_pdf

app = FastAPI(title="AI Career Copilot API")


class ProfileAnalysisRequest(BaseModel):
    current_background: str
    target_role: str
    job_description: str = Field(min_length=1)
    skills: list[str]
    project_experience: str


class ProfileAnalysisResponse(BaseModel):
    match_score: int
    score_breakdown: ScoreBreakdown | None = None
    strengths: list[str]
    skill_gaps: list[str]
    next_steps: list[str]


class ResumeAnalysisResponse(ProfileAnalysisResponse):
    resume_text: str


class ResumeTailoringResponse(BaseModel):
    top_changes: list[str]
    skills_to_emphasize: list[str]
    experiences_to_emphasize: list[str]
    missing_keywords: list[str]
    bullet_rewrite_suggestions: list[str]
    overall_advice: list[str]


class UploadedResumeTailoringResponse(ResumeTailoringResponse):
    resume_text: str


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


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)) -> dict[str, str]:
    """Extract text from an uploaded PDF resume."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    text = extract_text_from_pdf(await file.read())
    if not text:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no extractable text.",
        )

    return {"filename": file.filename or "", "text": text}


@app.post("/analyze-resume/llm", response_model=ResumeAnalysisResponse)
async def analyze_resume_llm(
    file: UploadFile = File(...),
    job_description: str = Form(...),
) -> ResumeAnalysisResponse:
    """Analyze an uploaded PDF resume against a job description with Gemini."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description must not be empty.")

    resume_text = extract_text_from_pdf(await file.read())
    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no extractable text.",
        )

    try:
        analysis = analyze_profile_with_gemini(
            current_background=resume_text,
            target_role="",
            job_description=job_description,
            skills=[],
            project_experience="",
        )
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=500,
            detail="LLM profile analysis is not configured.",
        ) from None
    except InvalidModelOutputError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service returned an invalid response.",
        ) from None
    except GeminiRequestError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service is currently unavailable.",
        ) from None

    return ResumeAnalysisResponse(**analysis.model_dump(), resume_text=resume_text)


@app.post(
    "/analyze-profile",
    response_model=ProfileAnalysisResponse,
    response_model_exclude_none=True,
)
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


@app.post("/analyze-profile/llm", response_model=ProfileAnalysisResponse)
def analyze_profile_llm(profile: ProfileAnalysisRequest) -> ProfileAnalysisResponse:
    """Return a Gemini-generated assessment of a candidate profile."""
    try:
        analysis = analyze_profile_with_gemini(
            current_background=profile.current_background,
            target_role=profile.target_role,
            job_description=profile.job_description,
            skills=profile.skills,
            project_experience=profile.project_experience,
        )
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=500,
            detail="LLM profile analysis is not configured.",
        ) from None
    except InvalidModelOutputError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service returned an invalid response.",
        ) from None
    except GeminiRequestError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service is currently unavailable.",
        ) from None

    return ProfileAnalysisResponse(**analysis.model_dump())


@app.post("/tailor-resume", response_model=ResumeTailoringResponse)
def tailor_resume(profile: ProfileAnalysisRequest) -> ResumeTailoringResponse:
    """Return Gemini-generated recommendations for tailoring a resume to a job."""
    try:
        recommendations = tailor_resume_with_gemini(
            current_background=profile.current_background,
            target_role=profile.target_role,
            job_description=profile.job_description,
            skills=profile.skills,
            project_experience=profile.project_experience,
        )
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=500,
            detail="LLM resume tailoring is not configured.",
        ) from None
    except InvalidModelOutputError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service returned an invalid response.",
        ) from None
    except GeminiRequestError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service is currently unavailable.",
        ) from None

    return ResumeTailoringResponse(**recommendations.model_dump())


@app.post("/tailor-resume/upload", response_model=UploadedResumeTailoringResponse)
async def tailor_uploaded_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...),
) -> UploadedResumeTailoringResponse:
    """Tailor an uploaded PDF resume and return its extracted text for reuse."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description must not be empty.")

    resume_text = extract_text_from_pdf(await file.read())
    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="The PDF contains no extractable text.",
        )

    try:
        recommendations = tailor_resume_with_gemini(
            current_background=resume_text,
            target_role="",
            job_description=job_description,
            skills=[],
            project_experience="",
        )
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=500,
            detail="LLM resume tailoring is not configured.",
        ) from None
    except InvalidModelOutputError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service returned an invalid response.",
        ) from None
    except GeminiRequestError:
        raise HTTPException(
            status_code=502,
            detail="The LLM service is currently unavailable.",
        ) from None

    return UploadedResumeTailoringResponse(
        **recommendations.model_dump(), resume_text=resume_text
    )
