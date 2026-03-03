# core/grading.py
from dotenv import load_dotenv
load_dotenv()

from grading_agent import grade_submission

def grade_one(
    rubric: str,
    student_answer: str,
    strictness: str = "lenient",
    provider_preference: str = "groq",
) -> dict:
    graded = grade_submission(
        rubric,
        student_answer,
        strictness=strictness,
        provider_preference=provider_preference,
    ) or {}

    return {
        "score_0_10": graded.get("score", 0),
        "feedback": graded.get("feedback", ""),
        "provider": graded.get("provider", provider_preference),
    }