import hashlib
import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Career Copilot", page_icon="🤖")


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
INPUT_METHODS = ("Upload resume PDF", "Enter profile manually")
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


def display_analysis(analysis: dict) -> None:
    st.subheader("Candidate Fit Analysis")
    st.metric("Match score", f"{analysis['match_score']}%")
    display_score_breakdown(analysis)

    st.markdown("### Strengths")
    for strength in analysis["strengths"]:
        st.markdown(f"- {strength}")

    st.markdown("### Skill gaps")
    if analysis["skill_gaps"]:
        for gap in analysis["skill_gaps"]:
            st.markdown(f"- {gap}")
    else:
        st.write("No gaps identified in this analysis.")

    st.markdown("### Next steps")
    for step in analysis["next_steps"]:
        st.markdown(f"- {step}")


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


def display_career_agent_result(result: dict) -> None:
    st.subheader("Career Agent Recommendation")
    st.write(result["final_answer"])
    st.metric("Tool calls", result["tool_call_count"])
    st.markdown("### Capabilities selected by the Agent")
    if result["tools_used"]:
        for tool_name in result["tools_used"]:
            st.markdown(f"- {tool_name.replace('_', ' ').title()}")
    else:
        st.write("The Agent answered without calling a tool.")


def manual_profile_payload(
    current_background: str,
    target_role: str,
    job_description: str,
    skills_text: str,
    project_experience: str,
) -> dict:
    return {
        "current_background": current_background,
        "target_role": target_role,
        "job_description": job_description,
        "skills": [skill.strip() for skill in skills_text.split(",") if skill.strip()],
        "project_experience": project_experience,
    }


def request_error_message(action: str) -> str:
    return {
        "ai_analysis": "The profile analysis could not be completed. Please try again.",
        "rule_analysis": "The profile analysis could not be completed. Please try again.",
        "career_agent": "The Career Agent could not complete this request. Please try again.",
        "tailor_resume": "Resume tailoring could not be completed. Please try again.",
        "extract_jd": "JD extraction could not be completed. Please try again.",
    }[action]


st.title("AI Career Copilot")
st.write("See how your experience lines up with your next role.")

st.header("1. Candidate Input")
input_method = st.radio(
    "How would you like to provide your experience?",
    options=INPUT_METHODS,
    horizontal=True,
)

with st.form("career-copilot-form"):
    if input_method == "Upload resume PDF":
        resume_file = st.file_uploader("Upload your resume", type=["pdf"])
        current_background = ""
        skills_text = ""
        project_experience = ""
    else:
        resume_file = None
        current_background = st.text_area(
            "Current background",
            placeholder="For example: 3 years in operations and customer support",
        )
        skills_text = st.text_input(
            "Skills (comma-separated)",
            placeholder="Python, SQL, communication",
        )
        project_experience = st.text_area(
            "Project experience",
            placeholder="Describe a relevant project and what you contributed.",
        )

    st.header("2. Target Job")
    target_role = st.text_input("Target role", placeholder="For example: Data Analyst")
    job_description = st.text_area(
        "Job description",
        placeholder="Paste the job's responsibilities and qualifications.",
    )

    st.header("3. Analysis / Action")
    career_agent_goal = st.text_area(
        "What would you like the Career Agent to do?",
        placeholder="Evaluate my fit for this role and tell me how to improve my resume.",
        help=(
            "For example: Evaluate my fit for this role and tell me how to improve "
            "my resume."
        ),
    )

    action_columns = st.columns(5 if input_method == "Enter profile manually" else 4)
    with action_columns[0]:
        ai_submitted = st.form_submit_button("AI Analysis", type="primary")
    if input_method == "Enter profile manually":
        with action_columns[1]:
            rule_submitted = st.form_submit_button("Rule-based Analysis")
        next_column = 2
    else:
        rule_submitted = False
        next_column = 1
    with action_columns[next_column]:
        career_agent_submitted = st.form_submit_button("Run Career Agent")
    with action_columns[next_column + 1]:
        tailoring_submitted = st.form_submit_button("Tailor Resume")
    with action_columns[next_column + 2]:
        jd_extraction_submitted = st.form_submit_button("Extract JD Requirements")

    if input_method == "Upload resume PDF":
        st.caption(
            "Rule-based analysis is available with manual profile input because the "
            "current rule matcher requires a structured skill list."
        )

selected_action = next(
    (
        action
        for submitted, action in (
            (ai_submitted, "ai_analysis"),
            (rule_submitted, "rule_analysis"),
            (career_agent_submitted, "career_agent"),
            (tailoring_submitted, "tailor_resume"),
            (jd_extraction_submitted, "extract_jd"),
        )
        if submitted
    ),
    None,
)

if selected_action and not job_description.strip():
    st.error("Please enter a job description before continuing.")
elif selected_action and len(job_description) > MAX_JD_CHARS:
    st.error(
        f"The job description is too long. Maximum length is {MAX_JD_CHARS:,} characters."
    )
elif career_agent_submitted and not career_agent_goal.strip():
    st.error("Please tell the Career Agent what you would like it to do.")
elif career_agent_submitted and len(career_agent_goal) > MAX_JD_CHARS:
    st.error(
        f"The Career Agent request is too long. Maximum length is "
        f"{MAX_JD_CHARS:,} characters."
    )
elif (
    selected_action in {"ai_analysis", "career_agent", "tailor_resume"}
    and input_method == "Upload resume PDF"
    and resume_file is None
):
    st.error("Please upload a PDF resume before continuing.")
elif (
    selected_action in {"ai_analysis", "career_agent", "tailor_resume"}
    and input_method == "Upload resume PDF"
    and len(resume_file.getvalue()) > MAX_RESUME_SIZE_BYTES
):
    st.error(f"The resume PDF is too large. Maximum size is {MAX_RESUME_SIZE_MB} MB.")
elif selected_action:
    profile_payload = manual_profile_payload(
        current_background=current_background,
        target_role=target_role,
        job_description=job_description,
        skills_text=skills_text,
        project_experience=project_experience,
    )
    try:
        with st.spinner("Running your selected action..."):
            if selected_action == "extract_jd":
                response = requests.post(
                    f"{API_BASE_URL}/extract-jd",
                    json={"job_description": job_description},
                    timeout=60,
                )
                response.raise_for_status()
                st.session_state["jd_extraction"] = response.json()
            elif selected_action == "career_agent":
                if input_method == "Upload resume PDF":
                    resume_bytes = resume_file.getvalue()
                    resume_file_key = hashlib.sha256(resume_bytes).hexdigest()
                    resume_text = (
                        st.session_state.get("resume_text")
                        if st.session_state.get("resume_file_key") == resume_file_key
                        else None
                    )
                    if not resume_text:
                        parse_response = requests.post(
                            f"{API_BASE_URL}/parse-resume",
                            files={
                                "file": (
                                    resume_file.name,
                                    resume_bytes,
                                    resume_file.type,
                                )
                            },
                            timeout=60,
                        )
                        parse_response.raise_for_status()
                        resume_text = parse_response.json()["text"]
                        if not isinstance(resume_text, str) or not resume_text.strip():
                            raise ValueError
                        st.session_state["resume_file_key"] = resume_file_key
                        st.session_state["resume_text"] = resume_text
                    agent_profile = {
                        "current_background": resume_text,
                        "target_role": target_role,
                        "job_description": job_description,
                        "skills": [],
                        "project_experience": "",
                    }
                else:
                    agent_profile = profile_payload
                response = requests.post(
                    f"{API_BASE_URL}/career-agent",
                    json={"user_request": career_agent_goal, **agent_profile},
                    timeout=60,
                )
                response.raise_for_status()
                st.session_state["career_agent_result"] = response.json()
            elif input_method == "Upload resume PDF":
                resume_bytes = resume_file.getvalue()
                endpoint = (
                    "/analyze-resume/llm"
                    if selected_action == "ai_analysis"
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
                    data={"job_description": job_description},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                resume_text = result.get("resume_text")
                if not isinstance(resume_text, str) or not resume_text.strip():
                    raise ValueError
                st.session_state["resume_file_key"] = hashlib.sha256(
                    resume_bytes
                ).hexdigest()
                st.session_state["resume_text"] = resume_text
                result_key = (
                    "profile_analysis"
                    if selected_action == "ai_analysis"
                    else "resume_tailoring"
                )
                st.session_state[result_key] = result
            else:
                endpoint = {
                    "ai_analysis": "/analyze-profile/llm",
                    "rule_analysis": "/analyze-profile",
                    "tailor_resume": "/tailor-resume",
                }[selected_action]
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json=profile_payload,
                    timeout=60,
                )
                response.raise_for_status()
                result_key = (
                    "resume_tailoring"
                    if selected_action == "tailor_resume"
                    else "profile_analysis"
                )
                st.session_state[result_key] = response.json()
    except requests.Timeout:
        st.error("The request took too long. Please try again.")
    except requests.ConnectionError:
        st.error("Could not connect to the backend. Make sure the FastAPI server is running.")
    except requests.HTTPError as exc:
        display_http_error(exc, request_error_message(selected_action))
    except requests.RequestException:
        st.error(request_error_message(selected_action))
    except (KeyError, TypeError, ValueError):
        st.error("The backend returned an invalid response.")

st.header("4. Results")
if "profile_analysis" in st.session_state:
    display_analysis(st.session_state["profile_analysis"])
if "career_agent_result" in st.session_state:
    display_career_agent_result(st.session_state["career_agent_result"])
if "resume_tailoring" in st.session_state:
    display_tailoring_recommendations(st.session_state["resume_tailoring"])
if "jd_extraction" in st.session_state:
    display_jd_extraction(st.session_state["jd_extraction"])

if not any(
    key in st.session_state
    for key in (
        "profile_analysis",
        "career_agent_result",
        "resume_tailoring",
        "jd_extraction",
    )
):
    st.info("Choose an action above to see results here.")
