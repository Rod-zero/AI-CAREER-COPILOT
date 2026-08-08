"""Environment-backed safety limits with local-development defaults."""

import os


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


AI_RATE_LIMIT_PER_MINUTE = _positive_int("AI_RATE_LIMIT_PER_MINUTE", 5)
AI_RATE_LIMIT_PER_HOUR = _positive_int("AI_RATE_LIMIT_PER_HOUR", 20)
MAX_RESUME_SIZE_MB = _positive_int("MAX_RESUME_SIZE_MB", 5)
MAX_RESUME_SIZE_BYTES = MAX_RESUME_SIZE_MB * 1024 * 1024
MAX_JD_CHARS = _positive_int("MAX_JD_CHARS", 20_000)
MAX_BACKGROUND_CHARS = _positive_int("MAX_BACKGROUND_CHARS", 30_000)
MAX_PROJECT_CHARS = _positive_int("MAX_PROJECT_CHARS", 20_000)
MAX_TARGET_ROLE_CHARS = _positive_int("MAX_TARGET_ROLE_CHARS", 500)
MAX_SKILLS = _positive_int("MAX_SKILLS", 200)
MAX_SKILL_CHARS = _positive_int("MAX_SKILL_CHARS", 200)

