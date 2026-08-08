import hashlib
import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Career Copilot", page_icon="🧭")

def positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
MAX_RESUME_SIZE_MB = positive_int_setting("MAX_RESUME_SIZE_MB", 5)
MAX_RESUME_SIZE_BYTES = MAX_RESUME_SIZE_MB * 1024 * 1024
MAX_JD_CHARS = positive_int_setting("MAX_JD_CHARS", 20_000)
ANALYSIS_MODES = {
    "AI analysis": ("/analyze-profile/llm", "Gemini AI analysis"),
    "Rule-based analysis": ("/analyze-profile", "Rule-based analysis"),
}
SCORE_LABELS = {
    "technical_skills": "Technical skills",
    "domain_experience": "Domain experience",
    "seniority": "Seniority",
    "role_specific_requirements": "Role-specific requirements",
    "education": "Education",
    "communication_leadership": "Communication / leadership",
}


def display_http_error(error: requests.HTTPError, default_message: str) -> None:
    status_code = getattr(error.response, "status_code", None)
    if status_code == 429:
        st.error("Too many AI requests. Please wait a little and try again.")
    elif status_code == 413:
        st.error(f"The resume PDF is too large. Maximum size is {MAX_RESUME_SIZE_MB} MB.")
    elif status_code == 422:
        st.error("Some submitted text is too long. Please shorten it and try again.")
    else:
        st.error(default_message)


def display_score_breakdown(analysis: dict) -> None:
    score_breakdown = analysis.get("score_breakdown")
    if not score_breakdown:
        return

    st.subheader("Score breakdown")
    for field, label in SCORE_LABELS.items():
        st.metric(label, f"{score_breakdown[field]}%")


def display_tailoring_recommendations(recommendations: dict) -> None:
    st.subheader("Resume Tailoring Recommendations")
    sections = (
        ("Top changes before applying", "top_changes"),
        ("Skills to emphasize", "skills_to_emphasize"),
        ("Experiences to emphasize", "experiences_to_emphasize"),
        ("Missing keywords", "missing_keywords"),
        ("Overall advice", "overall_advice"),
    )
    for heading, field in sections:
        st.markdown(f"### {heading}")
        if recommendations[field]:
            for item in recommendations[field]:
                st.markdown(f"- {item}")
        else:
            st.write("No recommendations in this category.")

    st.markdown("### Bullet rewrite suggestions")
    if recommendations["bullet_rewrite_suggestions"]:
        st.caption("Use the copy button on each suggestion to copy it.")
        for item in recommendations["bullet_rewrite_suggestions"]:
            st.code(item, language=None, wrap_lines=True)
    else:
        st.write("No recommendations in this category.")


def display_jd_extraction(extraction: dict) -> None:
    st.subheader("Structured JD Requirements")
    title = extraction.get("job_title") or "Not specified"
    seniority = extraction.get("seniority_level") or "Not specified"
    st.write(f"**Job title:** {title}")
    st.write(f"**Seniority:** {seniority}")

    sections = (
        ("Required skills", "required_skills"),
        ("Preferred skills", "preferred_skills"),
        ("Required experience", "required_experience"),
        ("Preferred experience", "preferred_experience"),
        ("Education", "education_requirements"),
        ("Tools & technologies", "tools_and_technologies"),
        ("Domain knowledge", "domain_knowledge"),
        ("Soft skills", "soft_skills"),
        ("Responsibilities", "responsibilities"),
    )
    copy_lines = [f"Job title: {title}", f"Seniority: {seniority}"]
    for heading, field in sections:
        st.markdown(f"### {heading}")
        items = extraction[field]
        if items:
            for item in items:
                st.markdown(f"- {item}")
        else:
            st.write("Not specified.")
        copy_lines.extend(["", f"{heading}:", *[f"- {item}" for item in items]])

    with st.expander("Copy all extracted requirements"):
        st.code("\n".join(copy_lines), language=None, wrap_lines=True)


st.title("AI Career Copilot")
st.write("See how your experience lines up with your next role.")
st.subheader("Analyze your resume")

with st.form("resume-analysis-form"):
    resume_file = st.file_uploader("Upload your resume", type=["pdf"])
    resume_job_description = st.text_area(
        "Target job description",
        placeholder="Paste the job's responsibilities and qualifications.",
    )
    analyze_column, tailor_column, extract_column = st.columns(3)
    with analyze_column:
        resume_submitted = st.form_submit_button("Analyze Resume", type="primary")
    with tailor_column:
        resume_tailoring_submitted = st.form_submit_button("Tailor Resume")
    with extract_column:
        jd_extraction_submitted = st.form_submit_button("Extract JD Requirements")

if jd_extraction_submitted and not resume_job_description.strip():
    st.error("Please enter a job description before extracting requirements.")
elif jd_extraction_submitted and len(resume_job_description) > MAX_JD_CHARS:
    st.error(
        f"The job description is too long. Maximum length is {MAX_JD_CHARS:,} characters."
    )
elif jd_extraction_submitted:
    try:
        with st.spinner("Extracting structured JD requirements..."):
            response = requests.post(
                f"{API_BASE_URL}/extract-jd",
                json={"job_description": resume_job_description},
                timeout=60,
            )
            response.raise_for_status()
            st.session_state["jd_extraction"] = response.json()
    except requests.Timeout:
        st.error("JD extraction took too long. Please try again.")
    except requests.ConnectionError:
        st.error("Could not connect to the backend. Make sure the FastAPI server is running.")
    except requests.HTTPError as exc:
        display_http_error(exc, "JD extraction could not be completed. Please try again.")
    except requests.RequestException:
        st.error("JD extraction could not be completed. Please try again.")
    except ValueError:
        st.error("The backend returned an invalid response.")

resume_action_submitted = resume_submitted or resume_tailoring_submitted
if resume_action_submitted and resume_file is None:
    if resume_submitted:
        st.error("Please upload a PDF resume before starting the analysis.")
    else:
        st.error("Please upload a PDF resume before tailoring your resume.")
elif resume_action_submitted and not resume_job_description.strip():
    if resume_submitted:
        st.error("Please enter a job description before analyzing your resume.")
    else:
        st.error("Please enter a job description before tailoring your resume.")
elif resume_action_submitted and len(resume_job_description) > MAX_JD_CHARS:
    st.error(
        f"The job description is too long. Maximum length is {MAX_JD_CHARS:,} characters."
    )
elif resume_action_submitted and len(resume_file.getvalue()) > MAX_RESUME_SIZE_BYTES:
    st.error(f"The resume PDF is too large. Maximum size is {MAX_RESUME_SIZE_MB} MB.")
elif resume_action_submitted:
    resume_bytes = resume_file.getvalue()
    context_key = hashlib.sha256(
        resume_bytes + resume_job_description.encode("utf-8")
    ).hexdigest()
    cached_text = (
        st.session_state.get("resume_text")
        if st.session_state.get("resume_context_key") == context_key
        else None
    )
    try:
        action_label = "Analyzing your resume with AI" if resume_submitted else "Tailoring your resume"
        with st.spinner(f"{action_label}..."):
            if cached_text:
                endpoint = "/analyze-profile/llm" if resume_submitted else "/tailor-resume"
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json={
                        "current_background": cached_text,
                        "target_role": "",
                        "job_description": resume_job_description,
                        "skills": [],
                        "project_experience": "",
                    },
                    timeout=60,
                )
            else:
                endpoint = (
                    "/analyze-resume/llm"
                    if resume_submitted
                    else "/tailor-resume/upload"
                )
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    files={
                        "file": (
                            resume_file.name,
                            resume_bytes,
                            resume_file.type,
                        )
                    },
                    data={"job_description": resume_job_description},
                    timeout=60,
                )
            response.raise_for_status()
            result = response.json()
            extracted_text = cached_text or result.get("resume_text")
            if not isinstance(extracted_text, str) or not extracted_text.strip():
                raise ValueError
    except requests.Timeout:
        if resume_submitted:
            st.error("The resume analysis took too long. Please try again.")
        else:
            st.error("Resume tailoring took too long. Please try again.")
    except requests.ConnectionError:
        st.error("Could not connect to the backend. Make sure the FastAPI server is running.")
    except requests.HTTPError as exc:
        default_message = (
            "The resume analysis could not be completed. Please try again."
            if resume_submitted
            else "Resume tailoring could not be completed. Please try again."
        )
        display_http_error(exc, default_message)
    except requests.RequestException:
        if resume_submitted:
            st.error("The resume analysis could not be completed. Please try again.")
        else:
            st.error("Resume tailoring could not be completed. Please try again.")
    except ValueError:
        st.error("The backend returned an invalid response.")
    else:
        if st.session_state.get("resume_context_key") != context_key:
            st.session_state.pop("resume_analysis", None)
            st.session_state.pop("resume_tailoring", None)
        st.session_state["resume_context_key"] = context_key
        st.session_state["resume_text"] = extracted_text
        if resume_submitted:
            result["resume_text"] = extracted_text
            st.session_state["resume_analysis"] = result
        else:
            st.session_state["resume_tailoring"] = result

if "jd_extraction" in st.session_state:
    display_jd_extraction(st.session_state["jd_extraction"])

if "resume_analysis" in st.session_state:
    resume_analysis = st.session_state["resume_analysis"]
    st.metric("Match score", f"{resume_analysis['match_score']}%")
    display_score_breakdown(resume_analysis)

    st.subheader("Strengths")
    for strength in resume_analysis["strengths"]:
        st.markdown(f"- {strength}")

    st.subheader("Skill gaps")
    if resume_analysis["skill_gaps"]:
        for gap in resume_analysis["skill_gaps"]:
            st.markdown(f"- {gap}")
    else:
        st.write("No gaps identified in this analysis.")

    st.subheader("Next steps")
    for step in resume_analysis["next_steps"]:
        st.markdown(f"- {step}")

if "resume_tailoring" in st.session_state:
    display_tailoring_recommendations(st.session_state["resume_tailoring"])

st.divider()
with st.expander("No resume? Enter profile manually"):
    with st.form("profile-analysis-form"):
        analysis_mode = st.radio(
            "Analysis mode",
            options=list(ANALYSIS_MODES),
            index=0,
            horizontal=True,
        )
        current_background = st.text_area(
            "Current background",
            placeholder="For example: 3 years in operations and customer support",
        )
        target_role = st.text_input("Target role", placeholder="For example: Data Analyst")
        job_description = st.text_area(
            "Job description",
            placeholder="Paste the job's responsibilities and qualifications.",
        )
        skills_text = st.text_input(
            "Skills (comma-separated)",
            placeholder="Python, SQL, communication",
        )
        project_experience = st.text_area(
            "Project experience",
            placeholder="Describe a relevant project and what you contributed.",
        )
        submitted = st.form_submit_button("Analyze Profile")
        tailoring_submitted = st.form_submit_button("Get Resume Tailoring Recommendations")

    if (submitted or tailoring_submitted) and not job_description.strip():
        st.error("Please enter a job description before analyzing your profile.")
    elif submitted or tailoring_submitted:
        payload = {
            "current_background": current_background,
            "target_role": target_role,
            "job_description": job_description,
            "skills": [skill.strip() for skill in skills_text.split(",") if skill.strip()],
            "project_experience": project_experience,
        }
        if tailoring_submitted:
            endpoint, result_label = "/tailor-resume", "Resume tailoring recommendations"
        else:
            endpoint, result_label = ANALYSIS_MODES[analysis_mode]

        try:
            with st.spinner(f"Running {result_label.lower()}..."):
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}", json=payload, timeout=60
                )
                response.raise_for_status()
                analysis = response.json()
        except requests.Timeout:
            st.error("The AI analysis took too long. Please try again.")
        except requests.ConnectionError:
            st.error(
                "Could not connect to the backend. Make sure the FastAPI server is running."
            )
        except requests.HTTPError as exc:
            display_http_error(
                exc,
                "The profile analysis could not be completed. Please try again.",
            )
        except requests.RequestException:
            st.error("The profile analysis could not be completed. Please try again.")
        except ValueError:
            st.error("The backend returned an invalid response.")
        else:
            st.info(f"Result source: {result_label}")
            if tailoring_submitted:
                display_tailoring_recommendations(analysis)
            else:
                st.metric("Match score", f"{analysis['match_score']}%")
                display_score_breakdown(analysis)

                st.subheader("Strengths")
                for strength in analysis["strengths"]:
                    st.markdown(f"- {strength}")

                st.subheader("Skill gaps")
                if analysis["skill_gaps"]:
                    for gap in analysis["skill_gaps"]:
                        st.markdown(f"- {gap}")
                else:
                    st.write("No gaps identified in this first-pass analysis.")

                st.subheader("Next steps")
                for step in analysis["next_steps"]:
                    st.markdown(f"- {step}")
