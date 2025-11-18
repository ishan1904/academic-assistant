from groq import Groq
import os, json, re

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import json, re

def extract_json(text: str):
    """
    Ultra-forgiving JSON extractor for Groq outputs.
    Tries to repair malformed JSON by inserting missing commas, braces, or brackets.
    Falls back to regex-based Q/A parsing if needed.
    """
    if not text or not isinstance(text, str):
        return {}

    # Step 1: Find the most likely JSON-like block
    match = re.search(r'(\[.*\]|\{.*\})', text, flags=re.DOTALL)
    candidate = match.group(1) if match else text.strip()
    candidate = candidate.strip()

    # Step 2: Basic cleanups
    cleaned = candidate
    cleaned = re.sub(r"(?<![}\]])\s*([}\]])", r"\1", cleaned)   # balance braces
    cleaned = cleaned.replace("\n", " ").replace("\r", "")
    cleaned = cleaned.replace("True", "true").replace("False", "false").replace("None", "null")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)            # remove trailing commas
    cleaned = re.sub(r"([}\]])\s*([{\[])", r"\1,\2", cleaned)   # add commas between objects if missing

    # Step 3: Try to close open brackets if cut off
    if cleaned.count("[") > cleaned.count("]"):
        cleaned += "]"
    if cleaned.count("{") > cleaned.count("}"):
        cleaned += "}"

    # Step 4: Try parsing
    try:
        data = json.loads(cleaned)
        print("[extract_json] Parsed valid JSON after repair.")
    except Exception as e:
        print(f"[extract_json] Still failed ({e}). Attempting truncated recovery.")
        # Try to salvage partial list
        partial = re.sub(r'[^}\]]+$', '', cleaned)  # trim trailing junk
        if not partial.strip().endswith(']'):
            partial += "]"
        try:
            data = json.loads(partial)
            print("[extract_json] Partial JSON recovered successfully.")
        except Exception as e2:
            print(f"[extract_json] Final fallback failed ({e2}). Using heuristic extraction.")
            data = []
            qa_blocks = re.findall(
                r"(?:Question|Q|Exercise|Problem)\s*[:\s\-]*(.*?)\s*(?:Answer|A)[:\s\-]+(.*?)(?=(?:Question|Q|Exercise|Problem)\s*[:\s\-]|\Z)",
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
            for q, a in qa_blocks:
                data.append({"question": q.strip(), "answer": a.strip()})


    # Step 5: Normalize
    if isinstance(data, dict):
        data = [data.get("qas", data)]
    if not isinstance(data, list):
        return {}

    result = {
        i + 1: {"question": str(item.get("question", "")).strip(),
                "answer": str(item.get("answer", "")).strip()}
        for i, item in enumerate(data)
        if isinstance(item, dict)
    }
    print(f"[extract_json] Extracted {len(result)} Q/A pairs successfully.")
    return result


# def extract_json(text: str):
#     """
#     Ultra-robust JSON extractor for LLM outputs.
#     Handles markdown fences, stray text before/after, unmatched braces,
#     single quotes, and trailing commas.  Always returns a dict or None.
#     """
#     import json, re

#     if not text or not isinstance(text, str):
#         return None

#     # --- 1️⃣ Normalize obvious noise ---
#     cleaned = (
#         text.replace("```json", "")
#         .replace("```", "")
#         .replace("**", "")
#         .replace("\r", "")
#         .strip()
#     )

#     # --- 2️⃣ Try to isolate the first JSON object ---
#     # capture the longest balanced-looking {...} block
#     match = re.search(r"\{[\s\S]*?\}", cleaned)
#     if match:
#         candidate = match.group(0)
#     else:
#         # if no braces at all, maybe the model returned key:value lines
#         candidate = "{" + cleaned + "}"

#     # --- 3️⃣ Basic repairs ---
#     candidate = candidate.replace("\n", " ").replace("\ufeff", "")
#     candidate = re.sub(r"(?<!\\)'", '"', candidate)      # single → double quotes
#     candidate = re.sub(r",\s*([}\]])", r"\1", candidate) # remove trailing commas

#     # If braces are unbalanced, patch them
#     open_b, close_b = candidate.count("{"), candidate.count("}")
#     if open_b > close_b:
#         candidate += "}" * (open_b - close_b)
#     elif close_b > open_b:
#         candidate = "{" * (close_b - open_b) + candidate

#     # --- 4️⃣ Parse or fall back safely ---
#     try:
#         return json.loads(candidate)
#     except Exception as e:
#         print(f"[extract_json] JSON parse failed: {e}\nRaw candidate: {candidate[:200]}")
#         return None
def extract_json_ai(text: str):
    """
    Uses the Groq LLaMA 3.1 model to extract structured Q/A pairs
    when regex fails (used by robust_split_into_qas fallback).
    """
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
    You are a strict JSON generator. Extract all distinct question–answer pairs
    from the following exam text. Each object must contain:
      - "question": the question text
      - "answer": the full corresponding answer text
    Output strictly as a JSON array, no commentary.

    Exam text:
    {text}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract structured Q/A pairs in JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        result = completion.choices[0].message.content.strip()

        # Try parsing once before returning
        try:
            return json.loads(result)
        except Exception:
            return result
    except Exception as e:
        print(f"[ERROR] extract_json_ai Groq call failed: {e}")
        return []


def normalize_score(score):
    """
    Normalize the model's score to a clean 0–10 scale.
    Handles:
    - 0–1 → multiply by 10
    - 0–5 → multiply by 2
    - 6–10 → leave as is
    - Above 10 → clamp to 10
    - None or invalid → return 0
    """
    try:
        score = float(score)

        if 0 <= score <= 1:
            return round(score * 10, 2)
        elif 0 < score <= 5:
            return round(score * 2, 2)
        elif 5 < score <= 10:
            return round(score, 2)
        else:
            return 10  # cap everything above 10
    except Exception:
        return 0




def grade_submission(rubric, answer, strictness="normal"):
    """
    Hybrid grader:
    - Detects MCQs, short, and long answers automatically.
    - Auto-scores MCQs.
    - Uses Groq for short and long answers with adjusted strictness.
    """

    import re, json

    rubric = (rubric or "").strip()
    answer = (answer or "").strip()

    # 🟩 1. Detect MCQ (A/B/C/D) and auto-grade
    if (
        len(rubric) == 1
        and rubric.upper() in ["A", "B", "C", "D"]
        and len(answer) == 1
        and answer.upper() in ["A", "B", "C", "D"]
    ):
        score = 10 if rubric.upper() == answer.upper() else 0
        feedback = "Correct choice ✅" if score == 10 else f"Incorrect. Correct answer was {rubric.upper()}."
        return {"score": score, "feedback": feedback, "provider": "auto"}

    # 🟨 2. Detect short answers (less than ~6 words)
    if len(answer.split()) < 6:
        chosen_mode = "lenient"
    
        # 🟦 Detect multiple-choice (checkbox) style: multiple letters separated by commas or spaces
    if re.fullmatch(r"([A-Da-d]\s*[,\s]*)+", rubric) and re.fullmatch(r"([A-Da-d]\s*[,\s]*)+", answer):
        rubric_set = {x.strip().upper() for x in re.split(r"[,\s]+", rubric) if x.strip()}
        answer_set = {x.strip().upper() for x in re.split(r"[,\s]+", answer) if x.strip()}
        intersection = rubric_set & answer_set
        if rubric_set:
            ratio = len(intersection) / len(rubric_set)
            # award partial credit for partially correct selections
            score = round(10 * ratio, 2)
        else:
            score = 0
        # small penalty if they selected too many extra options
        extras = len(answer_set - rubric_set)
        if extras:
            score = max(score - 2 * extras, 0)

        feedback = (
            f"Selected: {', '.join(sorted(answer_set))}. "
            f"Correct options: {', '.join(sorted(rubric_set))}. "
            f"Score based on overlap ({len(intersection)}/{len(rubric_set)} correct)."
        )
        return {"score": score, "feedback": feedback, "provider": "auto"}


    # 🟥 3. Detect long or conceptual answers
    else:
        chosen_mode = "strict" if strictness == "strict" else "normal"

    # --- Strictness instructions ---
    if chosen_mode == "lenient":
        grading_instructions = (
            "Grade generously but responsibly. Focus on conceptual correctness. "
            "Do not penalize for minor wording, formatting, or rounding differences. "
            "Encourage the student's effort while still identifying key missing ideas."
        )
    elif chosen_mode == "strict":
        grading_instructions = (
            "Grade conservatively and with high academic standards. "
            "Full credit requires precise terminology and complete reasoning. "
            "Reduce points for vague or incomplete logic or missing definitions."
        )
    else:
        grading_instructions = (
            "Grade fairly. Reward accurate understanding and correct reasoning, "
            "but deduct for factual or conceptual errors."
        )

    # --- Build LLM prompt ---
    msg = (
        "You are an AI teaching assistant grading student answers.\n"
        f"{grading_instructions}\n"
        "Provide feedback in 2–4 sentences explaining correctness, missing ideas, and improvements.\n"
        "Return ONLY JSON (no markdown, no commentary) in one of these formats:\n"
        "{'score': <0–10>, 'feedback': '<text>'}\n"
        f"Rubric (Correct Answer): {rubric}\n\n"
        f"Student Answer: {answer}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            top_p=1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only assistant. "
                        "Always output exactly one JSON object with 'score' and 'feedback' or 'evaluation'."
                    ),
                },
                {"role": "user", "content": msg},
            ],
        )

        raw_text = (resp.choices[0].message.content or "").strip()
        parsed = None

        # --- Try parsing JSON flexibly ---
        try:
            parsed = json.loads(raw_text)
        except Exception:
            cleaned = raw_text.replace("```json", "").replace("```", "")
            cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            m = re.search(r"\{[\s\S]*?\}", cleaned)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        # --- Fallback handling ---
        if not parsed or not isinstance(parsed, dict):
            # Try to pull numeric score if present
            m = re.search(r"(\d+(\.\d+)?)", raw_text)
            score = float(m.group(1)) if m else 0
            feedback = re.sub(r"[\{\}\[\]]", "", raw_text)
            return {"score": score, "feedback": feedback.strip(), "provider": "groq"}

        # --- Normalize ---
        parsed["provider"] = "groq"
        parsed["score"] = normalize_score(parsed.get("score", 0))
        fb = parsed.get("feedback") or parsed.get("evaluation") or ""
        if not isinstance(fb, str):
            fb = str(fb)
        parsed["feedback"] = fb.strip()
        return parsed

    except Exception as e:
        return {"score": 0, "feedback": f"Grader exception: {e}", "provider": "groq"}
