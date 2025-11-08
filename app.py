
# -*- coding: utf-8 -*-
"""
UTA Research • Grading Agent (Single-File Flask App)
----------------------------------------------------
Cleaned & humanized version with debug-friendly logs.

What’s inside:
  • Login / Signup / Dashboard
  • Sync + Async grading routes
  • Robust PDF text extraction:
      - PyPDF2 for typed PDFs
      - Gemini OCR fallback for scanned/handwritten
  • Resilient Q/A splitter with numeric ordering
  • Weighted scoring (auto-detect "(10 points)", etc.)
  • Upload helpers (PDF/DOCX/TXT/PPTX) for non-grading features
  • Clear section headers + consistent DEBUG/INFO logs

Default grading strictness: "normal"  (per user preference)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import io
import glob
import json
import time
import uuid
import traceback
from datetime import datetime, timedelta
from functools import wraps
import threading

from flask import (
    Flask, render_template, request, redirect, url_for, send_file,
    jsonify, session, flash, make_response
)
from werkzeug.utils import secure_filename

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

from PyPDF2 import PdfReader
from pptx import Presentation
import docx
from docx import Document

# OCR / Vision
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai

# OpenAI / Anthropic (for upload features like question generation)
import openai
import anthropic
from openai import OpenAI

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# App & Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)

# Core env
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "fallback_secret")
app.config["SESSION_COOKIE_SECURE"] = False  # True behind HTTPS in prod
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Paths
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads/")
app.config["OUTPUT_FOLDER"] = os.getenv("OUTPUT_FOLDER", "output/")
app.config["TEMP_FOLDER"]   = os.getenv("TEMP_FOLDER", "temp/")

for d in [app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"], app.config["TEMP_FOLDER"]]:
    os.makedirs(d, exist_ok=True)

if os.name != "nt":
    # Only set POSIX perms when not on Windows
    os.chmod(app.config["UPLOAD_FOLDER"], 0o755)
    os.chmod(app.config["OUTPUT_FOLDER"], 0o755)
    os.chmod(app.config["TEMP_FOLDER"], 0o755)

# APIs / keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
openai.api_key = OPENAI_API_KEY
claude = anthropic.Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Tesseract (optional: set Windows path if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# DB
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

with app.app_context():
    db.create_all()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: auth, filters, permissions
# ─────────────────────────────────────────────────────────────────────────────
@app.template_filter("is_iterable")
def is_iterable(value):
    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes))

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf", "docx", "txt", "ppt", "pptx"}

# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction & OCR Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_ocr_layout(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "").replace("\u00a0", " ")
    text = re.sub(r"\s{2,}", " ", text)
    # Line breaks before numbered headers like 1. / 2) / Exercise 3:
    text = re.sub(r"(?<!^)(?=\s*(Exercise|Question|Q|#)?\s*\d+\s*[:.)])", "\n", text)
    # Ensure key labels start new lines
    text = re.sub(r"(?i)(?<=\b)(Correct\s*Answer|Student\s*Answer|Evaluation|Score)\s*[:\-]?", r"\n\1:", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _merge_math_fragments(text: str) -> str:
    """Reduce false-new-question triggers from equations like 0.2088, 14), log2(...)."""
    if not text:
        return ""
    lines, out, buf = text.split("\n"), [], ""
    for line in lines:
        if re.match(r"^\s*(\d{1,3}\)|\d+\.\d+|log\d|√|−|\+|=)", line):
            buf += " " + line.strip()
        else:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append(line.strip())
    if buf:
        out.append(buf.strip())
    return "\n".join(out)

def extract_text_from_pdf_auto(pdf_path: str) -> str:
    print(f"[INFO] extract_text_from_pdf_auto → {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        plain = "".join([(p.extract_text() or "") for p in reader.pages]).strip()
        # ✅ Accept any non-empty text from PyPDF2
        if len(plain) > 0:
            print(f"[INFO] PyPDF2 extracted {len(plain)} chars — skipping OCR.")
            return plain
        else:
            print("[INFO] No text found by PyPDF2. Switching to Gemini OCR.")
    except Exception as e:
        print(f"[WARN] PyPDF2 failed ({e}). Switching to Gemini OCR.")

    # Keep Gemini fallback, but it will run only if PyPDF2 returns NOTHING
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        response = model.generate_content(
            ["Extract visible text from this exam paper without summarizing.",
             {"mime_type": "application/pdf", "data": pdf_bytes}]
        )
        ocr_text = getattr(response, "text", None) or str(response)
        cleaned = re.sub(r"\s{2,}", " ", ocr_text).strip()
        print(f"[INFO] Gemini OCR extracted {len(cleaned)} chars.")
        return cleaned
    except Exception as e:
        print(f"[ERROR] Gemini OCR failed: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning & Q/A Splitting
# ─────────────────────────────────────────────────────────────────────────────
def _sanitize_equations(text: str) -> str:
    """Protect decimals like 0.2088 from looking like 'question starts'."""
    if not text:
        return ""
    return re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)

def _preprocess_exam_text(raw: str) -> str:
    """Flatten dividers & normalize headers before splitting."""
    if not raw:
        return ""
    t = raw
    t = re.sub(r"-{3,}", "\n", t)  # divider lines → newline
    t = re.sub(r"(?i)End of (Answer Key|simulated student answers).*", "", t)
    t = re.sub(r"(?i)(?=\bExercise\s*\d+\b)", "\n", t)
    t = re.sub(r"(?i)(?=\bQuestion\s*\d+\b)", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_into_qas(text: str) -> dict:
    """
    Universal Q–A / Exercise splitter:
    • Detects MCQ blocks with 'Answer:'
    • Detects numbered Q&A without labels (like 1. ... [answer] 2. ...)
    • Falls back to analytical/blank-line splitting
    """

    if not text or not isinstance(text, str):
        return {}

    # Normalize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    qas = {}

    # ---- 1️⃣ MCQ-style detection (explicit 'Answer:' at least twice) ----
    if text.count("Answer:") >= 2:
        pattern = (
            r"(?is)"  # ignore case, dot matches newlines
            r"(?:^|\n)"  # start of line or text
            r"((?:Question|Exercise)?\s*\d+[.)\s:-]+.*?"  # capture header + body
            r"Answer\s*[:\-]\s*[A-Da-d0-9])"  # include Answer line
            r"(?=(?:\n\s*(?:Question|Exercise)?\s*\d+[.)]|$))"  # stop at next question or end
        )

        matches = re.findall(pattern, text)
        for i, block in enumerate(matches, 1):
            lines = block.strip().splitlines()
            title = lines[0].strip()
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            qas[i] = {"question": title, "answer": content}

    # ---- 2️⃣ Detect unlabeled numeric questions (no 'Answer:' keyword) ----
    if not qas:
        pattern = (
            r"(?is)"
            r"(?:^|\n)"                              # Start of text or line
            r"(?:Q(?:uestion)?\s*)?(\d+)[.)]?\s*(.*?)"  # Capture question number + text
            r"(?:\n(?!\s*\d+[.)]))(.+?)(?=(?:\n\s*\d+[.)]|$))"  # Capture answer until next number
        )
        matches = re.findall(pattern, text)
        for i, m in enumerate(matches, 1):
            q_num, question, answer = m
            qas[int(q_num)] = {
                "question": question.strip(),
                "answer": answer.strip(),
            }

    # ---- 3️⃣ Otherwise use analytical fallback ----
    if not qas:
        pattern = (
            r"(?is)"
            r"(?:^|\n)"
            r"((?:Exercise|Question|Problem|Task|Q|\()\s*\d+[A-Za-z]?[.:)\-–]?\s*.*?)"
            r"(?=(?:\n\s*(?:Exercise|Question|Problem|Task|Q|\()\s*\d+|\Z))"
        )
        matches = re.findall(pattern, text)
        for i, block in enumerate(matches, 1):
            lines = block.strip().splitlines()
            title = lines[0].strip()
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            qas[i] = {"question": title, "answer": content}

    # ---- 4️⃣ Fallback: blank-line chunks ----
    if not qas:
        chunks = re.split(r"\n\s*\n", text)
        for i, chunk in enumerate(chunks, 1):
            chunk = chunk.strip()
            if len(chunk.split()) > 5:
                qas[i] = {"question": chunk, "answer": ""}

    # ---- 5️⃣ Post-processing cleanup for MCQ answers ----
    for q_num, qa in qas.items():
        ans_text = qa.get("answer", "")

        # Extract "Answer: X" or "Ans: X" patterns (case-insensitive)
        m = re.search(r"(?i)\bAns(?:wer)?\s*[:\-]?\s*([A-Da-d0-9]+)\b", ans_text)
        if m:
            qas[q_num]["answer"] = m.group(1).strip().upper()
        else:
            # Handle text/numeric answers like "Answer: Paris"
            m2 = re.search(r"(?i)\bAns(?:wer)?\s*[:\-]?\s*([A-Za-z0-9 ]{1,30})", ans_text)
            if m2:
                qas[q_num]["answer"] = m2.group(1).strip()

        # Remove leftover "Answer:" text fragments
        qas[q_num]["answer"] = re.sub(r"(?i)\bAnswer[:\-]?\s*[A-Da-d0-9]+\b", "", qas[q_num]["answer"]).strip()

    print(f"[split_into_qas] Parsed {len(qas)} QAs")
    return {int(k): v for k, v in qas.items() if v.get("question") or v.get("answer")}

##worked for grade_run
# def split_into_qas(text: str) -> dict:
#     """
#     Universal Q–A / Exercise splitter:
#     • Detects MCQ blocks with 'Answer:'
#     • Detects numbered Q&A without labels (like 1. ... [answer] 2. ...)
#     • Falls back to analytical/blank-line splitting
#     """

#     if not text or not isinstance(text, str):
#         return {}

#     # Normalize newlines and spaces
#     text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
#     text = re.sub(r"[ \t]+", " ", text).strip()

#     qas = {}

#     # ---- 1️ MCQ-style detection (has 'Answer:' at least twice) ----
#     if text.count("Answer:") >= 2:
#         pattern = (
#             r"(?is)"  # ignore case, dot matches newlines
#             r"(?:^|\n)"  # start of line or text
#             r"(?:(?:^|\n)(?:Question|Exercise)?\s*(\d+)[\.\):])"  # capture header line and text
#             r"(?:\n\s*[A-D]\)[^\n]*){1,5}"  # 1–5 options
#             r"\s*Answer\s*[:\-]\s*[A-Da-d]"  # explicit Answer line
#             r"(?=(?:\n\s*\d+\s*[.)]|Q\s*\d+|Question\s*\d+|\Z))"  # stop at next question or end
#         )

#         matches = re.findall(pattern, text)
#         for i, block in enumerate(matches, 1):
#             lines = block.strip().splitlines()
#             title = lines[0].strip()
#             content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
#             qas[i] = {"question": title, "answer": content}

#     # ---- 2️ NEW: Detect unlabeled numeric questions (no 'Answer:' keyword) ----
#     if not qas:
#         pattern = (
#             r"(?is)"
#             r"(?:^|\n)"                              # Start of text or line
#             r"(?:Q(?:uestion)?\s*)?(\d+)[.)]?\s*(.*?)"  # Capture question number + text
#             r"(?:\n(?!\s*\d+[.)]))(.+?)(?=(?:\n\s*\d+[.)]|$))"  # Capture answer until next number
#         )
#         matches = re.findall(pattern, text)
#         for i, m in enumerate(matches, 1):
#             q_num, question, answer = m
#             qas[int(q_num)] = {
#                 "question": question.strip(),
#                 "answer": answer.strip(),
#             }

#     # ---- 3️ Otherwise use analytical fallback ----
#     if not qas:
#         pattern = (
#             r"(?is)"
#             r"(?:^|\n)"
#             r"((?:Exercise|Question|Problem|Task|Q|\()\s*\d+[A-Za-z]?[.:)\-–]?\s*.*?)"
#             r"(?=(?:\n\s*(?:Exercise|Question|Problem|Task|Q|\()\s*\d+|\Z))"
#         )
#         matches = re.findall(pattern, text)
#         for i, block in enumerate(matches, 1):
#             lines = block.strip().splitlines()
#             title = lines[0].strip()
#             content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
#             qas[i] = {"question": title, "answer": content}

#     # ---- 4️  Fallback: blank-line chunks ----
#     if not qas:
#         chunks = re.split(r"\n\s*\n", text)
#         for i, chunk in enumerate(chunks, 1):
#             chunk = chunk.strip()
#             if len(chunk.split()) > 5:
#                 qas[i] = {"question": chunk, "answer": ""}

#     print(f"[split_into_qas] Parsed {len(qas)} QAs")
#     return {int(k): v for k, v in qas.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Grading Pipeline (sync/async shared)
# ─────────────────────────────────────────────────────────────────────────────
def process_grading(key_source, ans_source, from_text=False, strictness="normal"):
    """
    Unified grading pipeline used by both sync and async flows.
    Returns: (results_dict, avg_score, key_qas)
    """
    print(f"[INFO] process_grading(strictness={strictness}, from_text={from_text})")
    from grading_agent import grade_submission

    # 1) Load text
    if from_text:
        key_text, ans_text = key_source, ans_source
    else:
        key_text = extract_text_from_pdf_auto(key_source)
        ans_text = extract_text_from_pdf_auto(ans_source)

    print(f"[DEBUG] key_text_len={len(key_text)} | ans_text_len={len(ans_text)}")

    # 2) Clean/preprocess
    def _norm_quotes(s: str) -> str:
        if not isinstance(s, str):
            return ""
        return (
            s.replace("‘","'").replace("’","'")
             .replace("“", '"').replace("”", '"')
             .replace("\r", "").replace("\u00a0", " ")
        )
    key_text = _norm_quotes(_preprocess_exam_text(_sanitize_equations(key_text)))
    ans_text = _norm_quotes(_preprocess_exam_text(_sanitize_equations(ans_text)))

    # 3) Split
    try:
        key_qas = split_into_qas(key_text)
        ans_qas = split_into_qas(ans_text)
    except Exception as e:
        print(f"[ERROR] split_into_qas failed: {e}")
        key_qas, ans_qas = {}, {}

    if not key_qas:
        print("[ERROR] No questions parsed from key text.")
        return {}, 0, {}

    # 4) Align (simple title similarity)
    import difflib
    aligned = {}
    for q in key_qas:
        kq = key_qas[q]["question"]
        best_id, best_ratio = None, 0.0
        for aq_id, aq in ans_qas.items():
            r = difflib.SequenceMatcher(None, kq, aq["question"]).ratio()
            if r > best_ratio:
                best_ratio, best_id = r, aq_id
        if best_id is not None:
            aligned[q] = ans_qas[best_id]
    ans_qas = aligned
    print(f"[DEBUG] alignment → matched {len(ans_qas)} answers.")

    # 5) Grade
    results = {}
    for q_num in sorted(key_qas.keys()):
        rubric = key_qas[q_num].get("answer", "")
        student = ans_qas.get(q_num, {}).get("answer", "")
        try:
            graded = grade_submission(rubric, student, strictness)
        except Exception as e:
            graded = {"score": None, "feedback": f"Agent error: {e}"}

        if isinstance(graded, str):
            graded = {"score": None, "feedback": graded}
        if isinstance(graded.get("feedback"), dict):
            graded["feedback"] = json.dumps(graded["feedback"], indent=2)

        results[q_num] = {
            "question": key_qas[q_num].get("question", ""),
            "correct_answer": rubric,
            "student_answer": student,
            "evaluation": graded.get("feedback", ""),
            "score": graded.get("score", 0) or 0,
        }

    # 6) Average (0–10)
    scores = [v["score"] for v in results.values() if isinstance(v["score"], (int, float))]
    avg = round(sum(scores) / len(scores), 2) if scores else 0
    print(f"[INFO] process_grading → avg={avg}")
    return results, avg, key_qas

# ─────────────────────────────────────────────────────────────────────────────
# Weighting
# ─────────────────────────────────────────────────────────────────────────────
QUESTION_WEIGHTS = {
    1: 25,
    2: 20,
    3: 25,
    4: 30,
}
DEFAULT_WEIGHT = 10
POINT_PATTERNS = [
    r"\((\d+)\s*points?\)",
    r"\b(\d+)\s*points?\b",
    r"\b(\d+)\s*pts?\b",
    r"-\s*(\d+)\s*points?\b",
]

def extract_weight_from_text(title_or_body: str, fallback: int) -> int:
    if not title_or_body:
        return fallback
    for pat in POINT_PATTERNS:
        m = re.search(pat, title_or_body, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return fallback

def apply_dynamic_weights(results: dict, key_qas: dict):
    total_awarded = 0.0
    total_possible = 0.0
    for q_num, item in results.items():
        key_text = f"{key_qas.get(q_num, {}).get('question', '')} {key_qas.get(q_num, {}).get('answer', '')}"
        weight = extract_weight_from_text(key_text, QUESTION_WEIGHTS.get(q_num, DEFAULT_WEIGHT))
        score = float(item.get("score", 0) or 0)
        awarded = round((score / 10.0) * weight, 2)

        item["weight"] = weight
        item["awarded_points"] = awarded
        total_awarded += awarded
        total_possible += weight

    weighted_pct = round((total_awarded / total_possible) * 100, 2) if total_possible else 0.0
    return results, total_awarded, total_possible, weighted_pct

# ─────────────────────────────────────────────────────────────────────────────
# Upload Helpers (non-grading features)
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[ERROR] read DOCX: {e}")
        return ""

def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] read TXT: {e}")
        return ""

def extract_text_from_ppt_with_slides(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PPTX not found: {filepath}")
        presentation = Presentation(filepath)
        slide_data = []
        extracted = []
        print(f"[INFO] PPTX slides: {len(presentation.slides)}")
        for i, slide in enumerate(presentation.slides, start=1):
            buf = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    buf.append(shape.text.strip())
            if buf:
                slide_text = "\n".join(buf)
                extracted.append(f"Slide {i}:\n{slide_text}\n")
                slide_data.append({"slide_number": i, "text": slide_text})
        out = "\n".join(extracted).strip()
        print(f"[INFO] PPTX extracted chars: {len(out)}")
        return out, slide_data
    except Exception as e:
        print(f"[ERROR] PPTX extract: {e}\n{traceback.format_exc()}")
        return "", []

# ─────────────────────────────────────────────────────────────────────────────
# AI query helpers (OpenAI / Claude) for upload features
# ─────────────────────────────────────────────────────────────────────────────
def query_openai_api(prompt: str):
    if not OPENAI_API_KEY or client is None:
        return {"error": "OpenAI API key not configured."}
    try:
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced instructor."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
    except Exception as e:
        print(f"[ERROR] OpenAI query: {e}\n{traceback.format_exc()}")
        return {"error": f"OpenAI error: {e}"}

def query_claude_api(prompt: str):
    if claude is None:
        return {"error": "Anthropic API key not configured."}
    try:
        resp = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        # New SDK returns content as list of blocks
        if hasattr(resp, "content") and isinstance(resp.content, list):
            combined = "".join(getattr(b, "text", "") for b in resp.content).strip()
        else:
            combined = getattr(resp, "content", "") or ""
        try:
            return json.loads(combined)
        except json.JSONDecodeError:
            return combined
    except Exception as e:
        print(f"[ERROR] Claude query: {e}\n{traceback.format_exc()}")
        return {"error": f"Claude error: {e}"}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder for upload features
# ─────────────────────────────────────────────────────────────────────────────
def generate_prompt(content, task, total_questions=20, mcq_count=5, tf_count=5, others_count=10,
                    recommender_name=None, position=None, email=None,
                    num_students=None, questions_per_student=None):
    if task == "questions":
        return f"""
        Generate {total_questions} questions from the provided content:
        - MCQs: {mcq_count}
        - True/False: {tf_count}
        - Other types: {others_count}
        For each question include:
        - Question text
        - Correct answer
        - Difficulty (easy/medium/hard)
        - Reference page/slide
        Ensure diversity; do not repeat.
        Content:
        {content}
        """
    elif task == "concepts":
        return f"""
        Identify core concepts from the content. Output 10–15 concise bullet points, each:
        "Concept: short explanation". Avoid verbosity.
        Content:
        {content}
        """
    elif task == "recommendation":
        today = datetime.today().strftime("%B %d, %Y")
        return f"""
        I am {recommender_name}, a {position} at {email}. Write a formal "To Whom It May Concern" Letter of Recommendation
        (date: {today}) in 3–4 paragraphs (6–7 sentences each), friendly-professional tone, emphasizing academic background,
        technical skills, leadership, and contributions. Include my name, title, and email at top and in signature.
        Base it on this content:
        {content}
        """
    elif task == "exam_papers":
        return f"""
        Generate {num_students} distinct sets of {questions_per_student} MCQs each, sourced from the slides below.
        For each question include 4 options (A–D), one correct answer, and the slide number. Provide two sets:
        1) Student version (questions + options)
        2) Answer key (question + slide + correct answer)
        Slides:
        {content}
        """
    elif task == "structured":
        # Keep minimal for this cleaned version
        return f"""
        Analyze the syllabus text and produce a JSON with sections:
        - AI-Completable Tasks
        - AI-Enhanced Processes
        - New Learning Frontiers
        - Skill Development
        - Core Competencies
        - AI-Supported Learning
        - Course Objectives and AI Influence
        - Risk Assessment and Overreliance
        - Advanced Learning Opportunities
        - Critical Thinking
        Content:
        {content}
        """

# ─────────────────────────────────────────────────────────────────────────────
# Routes: Core
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")

# ── Auth
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")
        existing = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing:
            flash("Username or email already exists.", "error")
            return render_template("signup.html")
        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(username=username, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash("Signup successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("login.html")
        user = User.query.filter_by(email=email).first()
        if not user or not bcrypt.check_password_hash(user.password, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")
        session["user_id"] = user.id
        session["username"] = user.username
        session.permanent = True
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return redirect(url_for("home"))

# ─────────────────────────────────────────────────────────────────────────────
# Routes: Grading (Sync)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/grade", methods=["GET"])
@login_required
def grade_form():
    return render_template("grade_upload.html")

@app.route("/grade", methods=["POST"])
@login_required
def grade_run():
    from grading_agent import grade_submission  # ensure import for runtime
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def ajax_error(msg, status=400):
        return make_response(f'<div class="error">{msg}</div>', status)

    key_pdf = request.files.get("key_pdf")
    ans_pdf = request.files.get("answer_pdf")
    if not key_pdf or not ans_pdf or key_pdf.filename.strip() == "" or ans_pdf.filename.strip() == "":
        msg = "Please upload both the key PDF and the answer sheet PDF."
        if is_ajax:
            return ajax_error(msg)
        flash(msg, "error")
        return redirect(url_for("grade_form"))

    # Save temp
    key_name = f"{uuid.uuid4().hex}_{secure_filename(key_pdf.filename)}"
    ans_name = f"{uuid.uuid4().hex}_{secure_filename(ans_pdf.filename)}"
    key_path = os.path.join(app.config["TEMP_FOLDER"], key_name)
    ans_path = os.path.join(app.config["TEMP_FOLDER"], ans_name)
    key_pdf.save(key_path); ans_pdf.save(ans_path)

    try:
        # Extract → Process → Grade
        key_text = extract_text_from_pdf_auto(key_path)
        ans_text = extract_text_from_pdf_auto(ans_path)

        results, avg_score, key_qas = process_grading(
            key_text, ans_text, from_text=True, strictness="normal"
        )

        # Save JSON
        out_name = f"grading_results_{uuid.uuid4().hex}.json"
        out_path = os.path.join(app.config["OUTPUT_FOLDER"], out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Apply weights
        try:
            results, total_awarded, total_possible, weighted_pct = apply_dynamic_weights(results, key_qas)
            average_display = f"Total: {total_awarded:.2f} / {total_possible} points ({weighted_pct}%)"
        except Exception as e:
            print(f"[WARN] Weight calc failed: {e}")
            average_display = None

        # Ensure numeric order in template
        results = dict(sorted(results.items(), key=lambda kv: int(kv[0])))

        template = "grade_result_fragment.html" if is_ajax else "grade_result.html"
        return render_template(
            template,
            results=results,
            avg_score=avg_score,
            average_display=average_display,
            download_filename=out_name
        )
    finally:
        for p in [key_path, ans_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception as e: print(f"[WARN] Could not remove temp {p}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Routes: Grading (Async)
# ─────────────────────────────────────────────────────────────────────────────
TASKS = {}

def _async_worker(task_id, key_path, ans_path, mode="normal"):
    print(f"[INFO] async worker started: {task_id}")
    try:
        TASKS[task_id]["status"] = "Extracting text..."
        key_text = extract_text_from_pdf_auto(key_path)
        ans_text = extract_text_from_pdf_auto(ans_path)

        TASKS[task_id]["status"] = "Grading..."
        results, avg_score, key_qas = process_grading(key_text, ans_text, from_text=True, strictness=mode)

        try:
            results, total_awarded, total_possible, weighted_pct = apply_dynamic_weights(results, key_qas)
            TASKS[task_id]["average_display"] = f"Total: {total_awarded:.2f} / {total_possible} points ({weighted_pct}%)"
        except Exception as e:
            print(f"[WARN] Async weight calc failed: {e}")
            TASKS[task_id]["average_display"] = None

        TASKS[task_id]["status"] = f"Completed (Avg Score {avg_score}/10)"
        TASKS[task_id]["result"] = dict(sorted(results.items(), key=lambda kv: int(kv[0])))
    except Exception as e:
        TASKS[task_id]["status"] = f"Failed: {e}"
        TASKS[task_id]["result"] = None
        print(f"[ERROR] async worker: {e}\n{traceback.format_exc()}")
    finally:
        for p in [key_path, ans_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception as e: print(f"[WARN] Could not remove temp {p}: {e}")

@app.route("/grade_async", methods=["POST"])
@login_required
def grade_async():
    key_file = request.files.get("key_pdf")
    ans_file = request.files.get("ans_pdf")
    if not key_file or not ans_file:
        flash("Please upload both files.", "error")
        return redirect(url_for("dashboard"))

    task_id = str(uuid.uuid4())
    key_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_key.pdf")
    ans_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_ans.pdf")
    key_file.save(key_path); ans_file.save(ans_path)

    TASKS[task_id] = {"status": "Starting...", "result": None}
    threading.Thread(target=_async_worker, args=(task_id, key_path, ans_path, "normal"), daemon=True).start()

    return redirect(url_for("task_status", task_id=task_id))

@app.route("/status/<task_id>")
@login_required
def task_status(task_id):
    task = TASKS.get(task_id)
    if not task:
        return "Task not found", 404
    if not str(task["status"]).startswith("Completed"):
        return f"""
        <h3>Status: {task['status']}</h3>
        <meta http-equiv='refresh' content='5'>
        <p>Please wait... grading may take a few minutes.</p>
        """
    return f"""
    <h3>Status: {task['status']}</h3>
    <p>Redirecting to results...</p>
    <script>
    setTimeout(() => {{ window.location.href = '/result/{task_id}'; }}, 1000);
    </script>
    """

@app.route("/result/<task_id>")
@login_required
def task_result(task_id):
    task = TASKS.get(task_id)
    if not task or not task.get("result"):
        return "Result not found or grading not completed.", 404
    avg_match = re.search(r"Avg Score ([0-9.]+)/10", str(task["status"]))
    avg_score = avg_match.group(1) if avg_match else "N/A"
    return render_template("grade_result_fragment.html",
                           results=task["result"],
                           avg_score=avg_score,
                           average_display=task.get("average_display"))

# ─────────────────────────────────────────────────────────────────────────────
# Routes: Upload Features (questions / concepts / rec letters / exam papers)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    try:
        file = (request.files.get("file")
                or request.files.get("exam_files")
                or request.files.get("chapter_file")
                or request.files.get("resume_file"))
        if not file or file.filename == "":
            flash("No file selected. Please upload a valid file.", "error")
            return redirect(url_for("dashboard"))

        if not allowed_file(file.filename):
            flash("Unsupported file type.", "error")
            return redirect(url_for("dashboard"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"[INFO] Upload saved @ {filepath}")

        ext = filename.rsplit(".", 1)[1].lower()
        if ext == "pdf":
            content = extract_text_from_pdf_auto(filepath)
        elif ext == "docx":
            content = extract_text_from_docx(filepath)
        elif ext == "txt":
            content = extract_text_from_txt(filepath)
        elif ext in {"ppt", "pptx"}:
            content, _ = extract_text_from_ppt_with_slides(filepath)
        else:
            content = ""

        if not isinstance(content, str) or not content.strip():
            flash("The uploaded file appears to be empty or unsupported.", "error")
            return redirect(url_for("dashboard"))

        action = request.form.get("action")
        ai_model = request.form.get("ai_model", "openai").strip().lower()
        print(f"[DEBUG] upload action={action} model={ai_model} content_len={len(content)}")

        # Questions
        if action == "questions":
            total_questions = int(request.form.get("total_questions", 20))
            mcq_count = int(request.form.get("mcq_count", 5))
            tf_count = int(request.form.get("tf_count", 5))
            others_count = int(request.form.get("others_count", total_questions - mcq_count - tf_count))
            if total_questions != (mcq_count + tf_count + others_count):
                flash("Total must equal MCQs + T/F + Other.", "error")
                return redirect(url_for("dashboard"))
            prompt = generate_prompt(content, "questions", total_questions, mcq_count, tf_count, others_count)
            result = query_claude_api(prompt) if ai_model == "claude" else query_openai_api(prompt)
            if not result:
                flash("No questions generated.", "error")
                return redirect(url_for("dashboard"))
            out_txt = json.dumps(result) if isinstance(result, dict) else str(result)
            out_name = f"questions_{uuid.uuid4().hex}.txt"
            with open(os.path.join(app.config["OUTPUT_FOLDER"], out_name), "w", encoding="utf-8") as f:
                f.write(out_txt)
            return render_template("questions.html", questions=result, question_filename=out_name)

        # Exam papers
        if action == "exam_papers":
            try:
                num_students = int(request.form.get("num_students"))
                questions_per_student = int(request.form.get("questions_per_student"))
                if num_students <= 0 or questions_per_student <= 0:
                    raise ValueError
            except Exception:
                flash("Provide valid numbers for students and questions.", "error")
                return redirect(url_for("dashboard"))
            prompt = generate_prompt(content, "exam_papers", num_students=num_students, questions_per_student=questions_per_student)
            result = query_claude_api(prompt) if ai_model == "claude" else query_openai_api(prompt)
            if not result or (isinstance(result, str) and not result.strip()):
                flash("No exam papers generated.", "error")
                return redirect(url_for("dashboard"))
            out_txt = json.dumps(result) if isinstance(result, dict) else str(result)
            out_name = f"exam_papers_{uuid.uuid4().hex}.txt"
            with open(os.path.join(app.config["OUTPUT_FOLDER"], out_name), "w", encoding="utf-8") as f:
                f.write(out_txt)
            return render_template("exam_papers.html", exam_papers=result, exam_filename=out_name)

        # Concepts
        if action == "concepts":
            prompt = generate_prompt(content, "concepts")
            result = query_claude_api(prompt) if ai_model == "claude" else query_openai_api(prompt)
            if not result:
                flash("No core concepts generated.", "error")
                return redirect(url_for("dashboard"))
            sections = []
            if isinstance(result, dict) and result.get("sections"):
                sections = result["sections"]
            else:
                # Parse simple bullets "1. Concept: description"
                if isinstance(result, str):
                    for line in result.split("\n"):
                        s = line.strip()
                        if not s: continue
                        if s[0].isdigit() and ". " in s and ": " in s:
                            concept = s.split(". ", 1)[1].split(": ", 1)[0].strip()
                            desc = s.split(": ", 1)[1].strip()
                            sections.append({"section": concept, "description": desc, "recommendations": []})
            if not sections:
                flash("No core concepts available.", "error")
                return redirect(url_for("dashboard"))
            out_name = f"concepts_{uuid.uuid4().hex}.txt"
            with open(os.path.join(app.config["OUTPUT_FOLDER"], out_name), "w", encoding="utf-8") as f:
                for i, s in enumerate(sections, 1):
                    f.write(f"{i}. {s['section']}: {s['description']}\n")
            return render_template("concepts.html", core_concepts=sections, concepts_filename=out_name, error=None)

        # Recommendation letter
        if action == "recommendation":
            recommender_name = request.form.get("recommender_name", "").strip()
            position = request.form.get("position", "").strip()
            email = request.form.get("email", "").strip()
            if not (recommender_name and position and email):
                flash("Missing recommender details.", "error")
                return redirect(url_for("dashboard"))
            prompt = generate_prompt(content, "recommendation",
                                     recommender_name=recommender_name, position=position, email=email)
            result = query_claude_api(prompt) if ai_model == "claude" else query_openai_api(prompt)
            text = result if isinstance(result, str) else json.dumps(result) if isinstance(result, dict) else ""
            out_name = f"recommendation_{uuid.uuid4().hex}.txt"
            with open(os.path.join(app.config["OUTPUT_FOLDER"], out_name), "w", encoding="utf-8") as f:
                f.write(text)
            return render_template("recommendation.html", recommendation=text, lor_filename=out_name)

        flash("Invalid action specified.", "error")
        return redirect(url_for("dashboard"))
    except Exception as e:
        print(f"[ERROR] /upload failed: {e}\n{traceback.format_exc()}")
        flash("An unexpected error occurred.", "error")
        return redirect(url_for("dashboard"))

# ─────────────────────────────────────────────────────────────────────────────
# Routes: Downloads / Utilities
# ─────────────────────────────────────────────────────────────────────────────
def create_filtered_doc_file(professor_name, course_name, filtered_sections):
    filename = f"selected_sections_{int(time.time())}.docx"
    doc_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    doc = Document()
    doc.add_heading("Course Details", level=1)
    doc.add_paragraph(f"Professor Name: {professor_name}")
    doc.add_paragraph(f"Course Name: {course_name}")
    doc.add_heading("Selected Sections", level=2)
    for section in filtered_sections:
        doc.add_heading(section.get("section", "No Title"), level=3)
        desc = section.get("description", "")
        if isinstance(desc, str):
            doc.add_paragraph(desc.replace("\n", " "))
        elif isinstance(desc, list):
            for point in desc:
                if isinstance(point, dict):
                    doc.add_paragraph(f"• {point.get('objective', '')}", style="List Bullet")
                    if point.get("justification"):
                        doc.add_paragraph(f"  Justification: {point['justification']}", style="List Bullet 2")
                else:
                    doc.add_paragraph(f"• {point}", style="List Bullet")
        examples = section.get("examples") or section.get("EXAMPLES") or section.get("recommendations", [])
        if examples:
            doc.add_paragraph("Examples:", style="List Bullet")
            for ex in examples:
                doc.add_paragraph(f"  - {ex}", style="List Bullet 2")
    doc.save(doc_path)
    return doc_path

@app.route("/download/questions/<filename>", methods=["GET"])
@login_required
def download_questions(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("No questions file available for download.", "error")
    return redirect(url_for("dashboard"))

@app.route("/download_recommendation/<filename>", methods=["GET"])
@login_required
def download_recommendation(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("No recommendation file available for download.", "error")
    return redirect(url_for("dashboard"))

@app.route("/download/exam_papers/<exam_filename>", methods=["GET"])
@login_required
def download_exam_papers(exam_filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], exam_filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("No exam papers file available for download.", "error")
    return redirect(url_for("dashboard"))

@app.route("/download/recommendations/<recommendation_filename>", methods=["GET"])
@login_required
def download_recommendations(recommendation_filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], recommendation_filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("No recommendations file available for download.", "error")
    return redirect(url_for("dashboard"))

@app.route("/debug_session")
@login_required
def debug_session():
    return jsonify({
        "user_id": session.get("user_id"),
        "username": session.get("username"),
        "session_keys": list(session.keys())
    })

# ─────────────────────────────────────────────────────────────────────────────
# Housekeeping
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_old_files(folder, max_age_hours=24):
    now = time.time()
    for path in glob.glob(os.path.join(folder, "*")):
        if os.path.isfile(path):
            age = now - os.path.getmtime(path)
            if age > max_age_hours * 3600:
                try:
                    os.remove(path)
                    print(f"[INFO] Removed old file: {path}")
                except Exception as e:
                    print(f"[WARN] Could not remove {path}: {e}")

with app.app_context():
    cleanup_old_files(app.config["UPLOAD_FOLDER"])
    cleanup_old_files(app.config["OUTPUT_FOLDER"])

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Debug=True to preserve the detailed logs you requested
    app.run(debug=True)
