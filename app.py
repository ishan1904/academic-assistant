# -*- coding: utf-8 -*-
"""
AI Academic Assistant (Flask)

This application provides a set of AI-assisted academic utilities through a
Flask web interface. It is designed to work with real-world, imperfect inputs
such as PDFs and slide decks, and focuses on reliability and clarity rather
than purely model-driven output.

Core features include:

- User authentication (signup, login, logout) backed by a simple SQLite database
- AI-powered grading:
  - Synchronous and asynchronous grading workflows
  - Comparison of a key (answer key PDF) against student submissions (PDF)
  - Robust text extraction using PyPDF2 with OCR fallback for scanned documents
- Chapter-level academic assistance:
  - Core concept extraction from lecture PDFs or PPT/PPTX files
  - Question generation based on uploaded chapter material
- Exam paper generation:
  - Supports PDF and PPT/PPTX inputs
  - Generates per-student question sets with structured output
- Recommendation letter generation:
  - Produces professional, role-agnostic recommendation letters from uploaded resumes

Design notes:
- All AI features are accessed through a unified upload endpoint with explicit actions.
- Most routes require authentication (see @login_required).
- The system prioritizes predictable output formats and defensive handling of
  large or noisy documents.

Terminology:
- “Key” refers to the correct answer sheet used for grading.
- “Answer” refers to the student’s submitted response.
"""

from __future__ import annotations
import os
import re
import json
import uuid
import traceback
import threading
import socket
import time
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Tuple, Optional
from translations import get_translation
from core.parsing import split_into_qas_by_qid
from core.docstore import init_db, put_doc, get_role, list_qids, get_qa
from core.cache import (
    init_cache,
    get_cached,
    set_cached,
    get_cached_ai_response,
    set_cached_ai_response,
)
from core.grading import grade_one

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader

# OCR helper (Gemini)
from ocr_gemini import extract_text_with_gemini

# Optional “upload feature” providers
import anthropic
from openai import OpenAI

import google.generativeai as genai


init_db()
init_cache()

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
load_dotenv(override=True)
socket.setdefaulttimeout(180)  # global network timeout (seconds)

app = Flask(__name__)
WSGIRequestHandler.protocol_version = "HTTP/1.1"


# ===============================
# i18n (English / Spanish)
# ===============================

SUPPORTED_LANGS = {"en", "es"}


def get_lang():
    return session.get("lang", "en")

@app.context_processor
def inject_i18n():
    def t(key):
        return get_translation(get_lang(), key)
    return {
        "t": t,
        "get_lang": get_lang
    }


# Flask session + basic security settings
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "fallback_secret")
app.config["SESSION_COOKIE_SECURE"] = False  # set True behind HTTPS in production
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=1)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

# Paths
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config["OUTPUT_FOLDER"] = os.getenv("OUTPUT_FOLDER", "output")
app.config["TEMP_FOLDER"] = os.getenv("TEMP_FOLDER", "temp")

for folder in (app.config["UPLOAD_FOLDER"], app.config["OUTPUT_FOLDER"], app.config["TEMP_FOLDER"]):
    os.makedirs(folder, exist_ok=True)

# -----------------------------------------------------------------------------
# API keys (optional depending on which features you use)
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model names are configurable via env so deployments don't break when defaults change.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
claude_client = anthropic.Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# -------------------------------------------------------------------
# Database configuration
# -------------------------------------------------------------------

# --- Database config: PostgreSQL on Railway, SQLite locally ---
DATABASE_URL = os.getenv("DATABASE_URL")

# Railway/Heroku sometimes use "postgres://", SQLAlchemy wants "postgresql://"
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


with app.app_context():
    db.create_all()

# -----------------------------------------------------------------------------
# Template filters
# -----------------------------------------------------------------------------
@app.template_filter("nl2br")
def nl2br(value: str) -> str:
    """Convert newlines to <br> for clean HTML display."""
    return (value or "").replace("\n", "<br>")


# -----------------------------------------------------------------------------
# Auth helpers
# -----------------------------------------------------------------------------
def login_required(fn):
    """Redirect to /login if a route requires authentication."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)

    return wrapper


def allowed_file(filename: str) -> bool:
    """Allow only known safe extensions."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in {"pdf", "docx", "txt", "ppt", "pptx"}


# -----------------------------------------------------------------------------
# PDF extraction
# -----------------------------------------------------------------------------
def extract_text_from_pdf_auto(pdf_path: str, force_ocr: bool = False) -> str:
    """
    Extract text from a PDF.

    Strategy:
    1) Use PyPDF2 first (fast, works well for typed PDFs)
    2) If PyPDF2 returns no text (or force_ocr=True), use Gemini OCR
    """
    print(f"[INFO] extract_text_from_pdf_auto → {pdf_path} (force_ocr={force_ocr})")

    if force_ocr:
        return extract_text_with_gemini(pdf_path)

    # Fast path: typed PDF extraction
    try:
        reader = PdfReader(pdf_path)
        plain = "".join((page.extract_text() or "") for page in reader.pages).strip()
        if plain:
            print(f"[INFO] PyPDF2 extracted {len(plain)} chars — skipping OCR.")
            return plain
    except Exception as e:
        print(f"[WARN] PyPDF2 failed ({e}). Will try OCR.")

    # Fallback: Gemini OCR on the whole PDF
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            [
                "Extract visible text from this exam paper. Do not summarize.",
                {"mime_type": "application/pdf", "data": pdf_bytes},
            ]
        )
        ocr_text = getattr(response, "text", None) or str(response)
        cleaned = re.sub(r"\s{2,}", " ", ocr_text).strip()
        print(f"[INFO] Gemini OCR extracted {len(cleaned)} chars.")
        return cleaned
    except Exception as e:
        print(f"[ERROR] Gemini OCR failed: {e}")
        return ""


# -----------------------------------------------------------------------------
# Text preprocessing and splitting helpers
# -----------------------------------------------------------------------------
def _sanitize_equations(text: str) -> str:
    """Protect decimals like 0.2088 from being treated as a numbered question."""
    return re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text or "")


def _preprocess_exam_text(raw: str) -> str:
    """Normalize separators and add predictable breaks before headings."""
    if not raw:
        return ""

    # Normalize line endings
    t = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Replace long dash separators
    t = re.sub(r"-{3,}", "\n", t)

    # Remove common footer noise
    t = re.sub(r"(?i)End of (Answer Key|simulated student answers).*", "", t)

    # ---- CRITICAL CLEANUP (prevents phantom questions) ----

    # Remove Gemini / OCR preamble text
    t = re.sub(r"(?is)^Here is the extracted visible text.*?\n", "", t).strip()

    # Remove standalone page markers like "**Page 6:**" or "Page 6:"
    t = re.sub(r"(?im)^\s*\*{0,2}\s*Page\s*\d+\s*:\s*\*{0,2}\s*$", "", t)

    # Remove inline page markers that appear mid-line
    t = re.sub(r"\*{0,2}\s*Page\s*\d+\s*:\s*\*{0,2}", " ", t)

    # Also remove bare "Page 6" lines (OCR variant)
    t = re.sub(r"(?im)^\s*\*{0,2}\s*Page\s*\d+\s*\*{0,2}\s*$", "", t)

    # ---- FORCE BREAKS BEFORE QUESTION MARKERS (PDF merge fix) ----
    # Example: "Paris 2. Define photosynthesis" -> "Paris\n2. Define photosynthesis"
    # Safe because decimals are protected earlier by _sanitize_equations()
    t = re.sub(r"(?<!\n)(?<=\S)\s+(\d{1,3})\s*[\.\)]\s+", r"\n\1. ", t)

    # Break before "Exercise N", "Question N", "Q N"
    t = re.sub(r"(?i)(?=\bExercise\s*\d+\b)", "\n", t)
    t = re.sub(r"(?i)(?=\bQuestion\s*\d+\b)", "\n", t)
    t = re.sub(r"(?i)(?=\bQ\s*\d+\b)", "\n", t)

    # Normalize blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def _clean_symbols(text: str) -> str:
    """Undo placeholder tokens and normalize a few common OCR symbols."""
    return (
        (text or "")
        .replace("<DOT>", ".")
        .replace("≈", "=")
        .replace("=>", "=")
        .replace("≤", "<=")
        .replace("≥", ">=")
    )

def _split_by_named_headers(t: str) -> List[Dict[str, str]]:
    """
    Split by Exercise/Question/Q headers only.
    Prevents splitting inside solution steps (1., 2., tables, etc.)
    """
    header = r"(?:Exercise|Question|Problem|Task|Q)\s*\(?\s*\d+[A-Za-z]?\s*\)?\s*[:\.\)\-–]?"
    pat = re.compile(
        rf"(?:(?<=\n)|^)\s*((?:{header})\s*.*?)(?=(?:\n\s*(?:{header})\s|\Z))",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = pat.findall(t)
    out: List[Dict[str, str]] = []
    for block in matches:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        title = lines[0] if lines else ""
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        out.append({"question": title, "answer": body})
    return out

def split_by_exercise(full_text: str) -> Dict[str, str]:
    """Split on headings like 'Exercise 1', returning {'exercise 1': '...'} blocks."""
    if not full_text:
        return {}

    t = full_text.replace("\r\n", "\n").replace("\r", "\n")
    heading_pat = re.compile(r"(?im)^\s*(exercise)\s*[:\-]?\s*(\d+)\b.*$")

    hits = [(m.start(), m.group(2)) for m in heading_pat.finditer(t)]
    if not hits:
        return {}

    blocks: Dict[str, str] = {}
    for i, (start, num) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        blocks[f"exercise {num}"] = t[start:end].strip()
    return blocks


def split_numbered_qa_pairs(text: str) -> Dict[str, str]:
    """
    Split answers that look like:
      1) A
      2. Paris
      Q3: option C
    Returns {'q1': 'A', 'q2': 'Paris', ...}
    """
    if not text:
        return {}

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    pat = re.compile(
        r"""
        (?ms)
        ^\s*
        (?:Q\s*)?(\d{1,3})
        \s*[\).\:]?\s*
        (?:Ans(?:wer)?\s*[:\-]?\s*)?
        (.*?)
        (?=^\s*(?:Q\s*)?\d{1,3}[\).\:]|\Z)
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    blocks: Dict[str, str] = {}
    for m in pat.finditer(t):
        num = m.group(1)
        ans = (m.group(2) or "").strip()
        if not ans:
            continue
        blocks[f"q{num}"] = ans.upper() if ans.upper() in {"A", "B", "C", "D", "E"} and len(ans) <= 3 else ans
    return blocks


def build_key_anchors(question_text: str) -> List[str]:
    """Generate anchor strings ('exercise 2', 'q3', ...) from a key question title."""
    if not question_text:
        return []

    t = question_text.lower()
    anchors: List[str] = []

    m = re.search(r"(?i)\bexercise\s*(\d+)\b", t)
    if m:
        n = m.group(1)
        anchors += [f"exercise {n}", f"ex {n}", f"ex{n}"]

    m = re.search(r"(?i)\bq(?:uestion)?\s*(\d+)\b", t)
    if m:
        n = m.group(1)
        anchors += [f"q{n}", f"q {n}", f"question {n}"]

    anchors = [re.sub(r"\s+", " ", a.strip().lower()) for a in anchors]
    # preserve order while removing duplicates
    return list(dict.fromkeys(anchors))


def match_student_text(
    *,
    key_question: Dict[str, str],
    q_index: int,
    student_blocks_ex: Dict[str, str],
    student_blocks_qa: Dict[int, str],
    full_student_text: str,
) -> Tuple[str, str]:
    """
    Match student text to a key question.

    Returns:
        (student_text_for_grading, match_mode)

    match_mode ∈ {
        "exercise-anchor",
        "numbered-qa-num",
        "numbered-qa-index",
        "full-fallback",
        "no-match"
    }
    """

    question_title = (key_question.get("question") or "").strip()

    # ✅ detect "real question"
    is_real_question = bool(
        re.search(r"(?i)\b(exercise|question|problem|task|q)\b", question_title)
        or "?" in question_title
    )

    # ------------------------------------------------------------
    # 1) Exercise-based matching (Exercise 1, Exercise 2, etc.)
    # ------------------------------------------------------------
    m = re.search(r"(?i)\bexercise\s*(\d+)\b", question_title)
    if m:
        ex_key = f"exercise {m.group(1)}".lower()
        ex_key = re.sub(r"\s+", " ", ex_key)
        if ex_key in student_blocks_ex:
            return student_blocks_ex[ex_key], "exercise-anchor"

    # ------------------------------------------------------------
    # 2) Numbered matching by actual number in the KEY title (best for MCQs)
    #    e.g. "12. Which..." or "12) Which..."
    # ------------------------------------------------------------
    mnum = re.match(r"^\s*(\d{1,3})\s*[\.\)]\s+", question_title)
    if mnum:
        qnum = int(mnum.group(1))
        if qnum in student_blocks_qa:
            return student_blocks_qa[qnum], "numbered-qa-num"

    # ------------------------------------------------------------
    # 3) Numbered matching by enumerate index (fallback)
    # ------------------------------------------------------------
    if q_index in student_blocks_qa:
        return student_blocks_qa[q_index], "numbered-qa-index"

    # ------------------------------------------------------------
    # 4) Full fallback (ONLY when it makes sense)
    #    - For MCQ / multi-question numbered tests, full-fallback is harmful
    #      because it repeats the whole paper for every question.
    # ------------------------------------------------------------
    multi_question_numbered_test = (len(student_blocks_qa) >= 2)

    if is_real_question and full_student_text.strip() and not multi_question_numbered_test:
        return full_student_text, "full-fallback"

    return "", "no-match"


def _fallback_split_numbered_blocks(t: str) -> List[Dict[str, str]]:
    """
    Stricter split for text like:
      1. What is ...?
         Paris
      2) Define ...
         ...
    This is meant to rescue small tests/MCQs that otherwise collapse into 1 block.
    """
    if not t:
        return []

    pat = re.compile(r"(?ms)^\s*(\d{1,3})\s*[\.\)]\s+(.*?)(?=^\s*\d{1,3}\s*[\.\)]\s+|\Z)")
    out: List[Dict[str, str]] = []

    for m in pat.finditer(t):
        qnum = m.group(1)
        block = (m.group(2) or "").strip()
        if not block:
            continue

        # First line is usually question text; remaining is "answer-ish" (or options)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        title = f"{qnum}. {lines[0]}" if lines else f"{qnum}."
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        # ✅ ADD THIS BLOCK RIGHT HERE (before out.append)
        block_text = (title + " " + body).strip()
        word_count = len(re.findall(r"\w+", block_text))

        looks_like_question = ("?" in title) or ("?" in block_text) or re.search(
            r"(?i)\b(answer|ans|option|choose|select|which)\b", block_text
        )
        looks_like_mcq = bool(re.search(r"(?im)^\s*[A-D]\)\s+", body)) or bool(re.search(r"(?im)^\s*[A-D]\.\s+", body))

        # Reject very tiny junk fragments unless they're MCQ-ish
        if word_count < 6 and not looks_like_mcq:
            continue

        # Reject numbered table rows / calculation steps (common in big original test)
        if not looks_like_question and not looks_like_mcq:
            continue

        out.append({"question": title, "answer": body})

    return out


# -----------------------------------------------------------------------------
# Grading pipeline
# -----------------------------------------------------------------------------
def robust_split_into_qas(text: str) -> List[Dict[str, str]]:
    """
    Split exam text into Q/A blocks.

    Priority:
    1) If Exercise/Question/Q headers exist (>=2), split ONLY by those.
       (Prevents step-by-step numbered lines inside answers becoming fake questions.)
    2) Otherwise, allow numbered question splitting (1., 2), etc.) for small tests/MCQs.
    """
    if not text:
        return []

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # ✅ FIRST: try splitting by named headers only
    named_blocks = _split_by_named_headers(t)
    if len(named_blocks) >= 2:
        return named_blocks

    # ---- SECOND: allow numbered questions when named headers are not present ----
    header = r"(?:Exercise|Question|Problem|Task|Q)\s*\d+[A-Za-z]?[\.:)\-–]?"
    numbered = r"\d{1,3}\s*[\.\)]"

    pattern = re.compile(
        rf"(?:(?<=\n)|^)\s*((?:{header}|{numbered})\s*.*?)(?=(?:\n\s*(?:{header}|{numbered})\s|\Z))",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(t)

    if matches:
        out: List[Dict[str, str]] = []
        for block in matches:
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            title = lines[0] if lines else ""
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            out.append({"question": title, "answer": body})

        # Rescue case for pure numbered small tests (3Q / MCQ)
        if len(out) == 1:
            n_hits = len(re.findall(r"(?m)^\s*\d{1,3}\s*[\.\)]\s+", t))
            if n_hits >= 2:
                strict = _fallback_split_numbered_blocks(t)
                if len(strict) >= 2:
                    return strict

        return out

    # Last resort fallback: paragraph chunks
    chunks = [c.strip() for c in re.split(r"\n\s*\n", t) if len(c.strip().split()) > 5]
    return [{"question": c, "answer": ""} for c in chunks]

def build_student_numbered_index(full_student_text: str) -> Dict[int, str]:
    """
    Build a robust {question_number: block_text} index from the student's text.

    Uses split_numbered_qa_pairs first (your existing method), but if that fails
    (common with PDFs/MCQs), falls back to _fallback_split_numbered_blocks which
    is stricter and works well for MCQs.
    """
    if not full_student_text:
        return {}

    # Primary (your existing splitter)
    try:
        primary = split_numbered_qa_pairs(full_student_text) or {}
    except Exception:
        primary = {}

    # If primary seems OK, return it
    if isinstance(primary, dict) and len(primary) >= 2:
        # ensure keys are ints
        fixed: Dict[int, str] = {}
        for k, v in primary.items():
            try:
                fixed[int(k)] = v
            except Exception:
                pass
        if len(fixed) >= 2:
            return fixed

    # Fallback: strict numbered block splitter
    strict_blocks = _fallback_split_numbered_blocks(full_student_text)
    out: Dict[int, str] = {}

    for b in strict_blocks:
        qtitle = (b.get("question") or "").strip()
        body = (b.get("answer") or "").strip()

        m = re.match(r"^\s*(\d{1,3})\s*[\.\)]", qtitle)
        if not m:
            continue
        qnum = int(m.group(1))

        # Keep full block for grading (question + options + answer line if present)
        combined = qtitle
        if body:
            combined = combined + "\n" + body

        out[qnum] = combined.strip()

    return out

def _extract_mcq_letter(text: str) -> str:
    """
    Extract answer choice letter from text.
    Supports: 'Answer: B', 'Ans - B', 'Correct: C', '(B)', 'B)' etc.
    """
    if not text:
        return ""

    t = text.strip()

    # Strong patterns first
    m = re.search(r"(?i)\b(?:answer|ans|correct)\s*[:=\-]\s*([A-D])\b", t)
    if m:
        return m.group(1).upper()

    # Standalone "Answer B" (no colon)
    m = re.search(r"(?i)\b(?:answer|ans|correct)\s+([A-D])\b", t)
    if m:
        return m.group(1).upper()

    # Last resort: find a single letter on its own line
    m = re.search(r"(?m)^\s*([A-D])\s*$", t)
    if m:
        return m.group(1).upper()

    return ""


def _is_mcq_block(rubric: str) -> bool:
    """
    Heuristic: if the key contains options A) B) C) and an answer letter.
    """
    if not rubric:
        return False
    has_options = bool(re.search(r"(?m)^\s*A\)\s+.+\n\s*B\)\s+.+\n\s*C\)\s+.+", rubric))
    has_answer = bool(re.search(r"(?i)\b(?:answer|ans|correct)\s*[:=\-]\s*[A-D]\b", rubric))
    return has_options and has_answer

def build_student_sequential_blocks(full_student_text: str) -> Dict[int, str]:
    """
    Fallback splitter when the student didn't number answers (common in small tests).
    Splits into paragraph-like blocks and assigns 1..N.
    """
    if not full_student_text:
        return {}

    t = full_student_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return {}

    # Split on blank lines first
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", t) if c.strip()]

    # If still only one chunk, try splitting on lines that look like standalone short answers
    if len(chunks) <= 1:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        # group lines into blocks; new block when line looks like a new Q marker OR long question sentence
        chunks = []
        buf = []
        for ln in lines:
            if re.match(r"^\d{1,3}[\.\)]\s+", ln):
                if buf:
                    chunks.append("\n".join(buf).strip())
                    buf = []
                buf.append(ln)
            else:
                buf.append(ln)
        if buf:
            chunks.append("\n".join(buf).strip())

    out: Dict[int, str] = {}
    for i, c in enumerate(chunks, start=1):
        out[i] = c
    return out


def process_grading(
    key_source: str,
    ans_source: str,
    *,
    from_text: bool = False,
    strictness: str = "normal",
    force_ocr: bool = False,
    grading_provider: str = "groq",
    key_doc_id: str | None = None, 
    student_doc_id: str | None = None,
) -> Tuple[Dict[int, Dict[str, Any]], float, Dict[int, Dict[str, str]]]:
    """
    Main grading pipeline.

    Returns:
      results: {1: {...}, 2: {...}}
      avg_score (0-10 scale)
      key_qas_dict: {1: {"question":..., "answer":...}, ...} used for weighting
    """


    # ---------------------------------------------------------
    # 1) Load and preprocess text
    # ---------------------------------------------------------
    if from_text:
        key_text, ans_text = key_source, ans_source
    else:
        key_text = extract_text_from_pdf_auto(key_source)
        ans_text = extract_text_from_pdf_auto(ans_source, force_ocr=force_ocr)

    key_text = _clean_symbols(_preprocess_exam_text(_sanitize_equations(key_text)))
    ans_text = _clean_symbols(_preprocess_exam_text(_sanitize_equations(ans_text)))

    if not key_text.strip():
        return {}, 0.0, {}

    # ---------------------------------------------------------
    # 2) Parse Q/A blocks from KEY
    # ---------------------------------------------------------
    key_qas_list = robust_split_into_qas(key_text)

    # ✅ If the key clearly uses Exercise/Question headers, lock to those only.
    # Prevents numbered "steps" inside solutions from becoming fake questions.
    header_hits = len(
        re.findall(r"(?i)\b(?:exercise|question|problem|task|q)\s*\d+\b", key_text)
    )
    if header_hits >= 2:
        key_qas_list = _split_by_named_headers(key_text)

    if not key_qas_list:
        return {}, 0.0, {}

    full_student_text = (ans_text or "").strip()

    # ---------------------------------------------------------
    # 3) Build student block indexes once
    # ---------------------------------------------------------
    student_blocks_ex = {
        re.sub(r"\s+", " ", k.strip().lower()): v
        for k, v in split_by_exercise(full_student_text).items()
    }

    # IMPORTANT: use your robust builder (recommended)
    # If you didn't add build_student_numbered_index(), revert to split_numbered_qa_pairs()
    try:
        student_blocks_qa = build_student_numbered_index(full_student_text)
    except Exception:
        student_blocks_qa = {}

    # ✅ NEW: If numbered parsing failed (or only gave Q1), use sequential block fallback
    # This is common in small tests where students don't label 2/3/4...
    if len(student_blocks_qa) < 2 and full_student_text.strip():
        seq_blocks = build_student_sequential_blocks(full_student_text)
        if len(seq_blocks) >= 2:
            student_blocks_qa = seq_blocks


    # ---------------------------------------------------------
    # 4) Grade each question
    # ---------------------------------------------------------
    results: Dict[int, Dict[str, Any]] = {}
    total_score = 0.0
    scored = 0

    MAX_DISPLAY_CHARS = 2500

    for idx, kqa in enumerate(key_qas_list, start=1):
        rubric = (kqa.get("answer") or "").strip()

        student_for_grading, match_mode = match_student_text(
            key_question=kqa,
            q_index=idx,
            student_blocks_ex=student_blocks_ex,
            student_blocks_qa=student_blocks_qa,
            full_student_text=full_student_text,
        )

        student_for_display = (student_for_grading or "").strip()
        if len(student_for_display) > MAX_DISPLAY_CHARS:
            student_for_display = student_for_display[:MAX_DISPLAY_CHARS] + "…"

        # -----------------------------------------------------
        # ✅ NO-MATCH: don't call LLM
        # -----------------------------------------------------
        if match_mode == "no-match" or not student_for_grading.strip():
            graded = {
                "score": 0,
                "feedback": "Answer not found or could not be matched to this question."
            }
            score = 0.0

        else:
            # -------------------------------------------------
            # ✅ MCQ AUTO-GRADING (no LLM)
            # -------------------------------------------------
            if _is_mcq_block(rubric):
                correct_letter = _extract_mcq_letter(rubric)
                student_letter = _extract_mcq_letter(student_for_grading)

                if correct_letter and student_letter:
                    if student_letter == correct_letter:
                        score = 10.0
                        graded = {"score": 10, "feedback": f"Correct. Selected {student_letter}."}
                    else:
                        score = 0.0
                        graded = {"score": 0, "feedback": f"Incorrect. Selected {student_letter}; correct is {correct_letter}."}
                else:
                    score = 0.0
                    graded = {"score": 0, "feedback": "Could not extract answer choice (A-D) from the response."}

            else:
                # -------------------------------------------------
                # ✅ Non-MCQ: LLM grading as before
                # -------------------------------------------------
                # ---- MCP-style grading: cache + per-question ----

                # ---- MCP-style grading: cache + per-question ----
                cached = (
                    get_cached(key_doc_id, student_doc_id, idx, strictness,  grading_provider)
                    if key_doc_id and student_doc_id
                    else None
                )

                if cached:
                    print(f"[CACHE HIT] grading q={idx} provider={grading_provider}")
                    out = cached
                    out["cached"] = True
                else:
                    print(f"[CACHE MISS] grading q={idx} provider={grading_provider}")
                    out = grade_one(
                            rubric,
                            student_for_grading,
                            strictness=strictness,
                            provider_preference=grading_provider,
                        )
                    if key_doc_id and student_doc_id:
                        set_cached(key_doc_id, student_doc_id, idx, strictness, grading_provider, out)
                    out["cached"] = False

                score = float(out.get("score_0_10", 0) or 0)
                graded = {"score": score, "feedback": out.get("feedback", "")}

        total_score += score
        scored += 1

        results[idx] = {
            "question": kqa.get("question", "") or f"Q{idx}",
            "correct_answer": (rubric or "").strip(),
            "student_answer": (student_for_display or "[No matched student block detected]").strip(),
            "evaluation": (graded.get("feedback", "") or "").strip(),
            "score": score,
            "match_mode": match_mode,
            "cached": out.get("cached", False) if match_mode != "no-match" and not _is_mcq_block(rubric) else False,
        }

        print(
            f"[DEBUG] Q{idx} match={match_mode} "
            f"score={score} cached={results[idx]['cached']} "
            f"preview={student_for_display[:120]!r}"
        )

    avg = round(total_score / scored, 2) if scored else 0.0
    key_qas_dict = {i: qa for i, qa in enumerate(key_qas_list, start=1)}

    return results, avg, key_qas_dict


# -----------------------------------------------------------------------------
# Weighting
# -----------------------------------------------------------------------------
QUESTION_WEIGHTS = {1: 25, 2: 20, 3: 25, 4: 30}
DEFAULT_WEIGHT = 10
POINT_PATTERNS = [
    r"\((\d+)\s*points?\)",
    r"\b(\d+)\s*points?\b",
    r"\b(\d+)\s*pts?\b",
    r"-\s*(\d+)\s*points?\b",
]


def extract_weight_from_text(title_or_body: str, fallback: int) -> int:
    for pat in POINT_PATTERNS:
        m = re.search(pat, title_or_body or "", re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return fallback


def apply_dynamic_weights(results: Dict[int, Dict[str, Any]], key_qas: Dict[int, Dict[str, str]]):
    total_awarded = 0.0
    total_possible = 0.0

    for q_num, item in results.items():
        k_data = key_qas.get(int(q_num), {})
        key_text = f"{k_data.get('question', '')} {k_data.get('answer', '')}"
        weight = extract_weight_from_text(key_text, QUESTION_WEIGHTS.get(int(q_num), DEFAULT_WEIGHT))

        score = float(item.get("score", 0) or 0)
        awarded = round((score / 10.0) * weight, 2)

        item["weight"] = weight
        item["awarded_points"] = awarded

        total_awarded += awarded
        total_possible += weight

    weighted_pct = round((total_awarded / total_possible) * 100, 2) if total_possible else 0.0
    return results, total_awarded, total_possible, weighted_pct


# -----------------------------------------------------------------------------
# Upload utilities (optional features)
# -----------------------------------------------------------------------------
def extract_text_from_txt(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] read TXT: {e}")
        return ""


def extract_text_from_docx(filepath: str) -> str:
    try:
        import docx  # lazy import so it is only required if you use DOCX uploads

        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[ERROR] read DOCX: {e}")
        return ""


def extract_text_from_ppt_with_slides(filepath: str):
    try:
        from pptx import Presentation  # lazy import for optional feature

        pres = Presentation(filepath)
        extracted: List[str] = []
        slide_data: List[Dict[str, Any]] = []

        for i, slide in enumerate(pres.slides, start=1):
            buf: List[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and (shape.text or "").strip():
                    buf.append(shape.text.strip())
            if buf:
                slide_text = "\n".join(buf)
                extracted.append(f"Slide {i}:\n{slide_text}\n")
                slide_data.append({"slide_number": i, "text": slide_text})

        return "\n".join(extracted).strip(), slide_data
    except Exception as e:
        print(f"[ERROR] PPTX extract: {e}\n{traceback.format_exc()}")
        return "", []


def query_openai_api(prompt: str):
    """Call OpenAI for the upload-utility features (questions/concepts/syllabus/letters/exams)."""
    if not openai_client:
        return {"error": "OpenAI API key not configured."}

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an experienced instructor."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()

        # Strip markdown code fences if the model wraps JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        # Try JSON parse, else return plain text
        try:
            return json.loads(content)
        except Exception:
            return content

    except Exception as e:
        print(f"[ERROR] OpenAI call failed: {e}\n{traceback.format_exc()}")
        return {"error": f"OpenAI call failed: {str(e)}"}


def query_claude_api(prompt: str):
    """Call Anthropic Claude for the upload-utility features."""
    if not claude_client:
        return {"error": "Anthropic API key not configured."}

    try:
        model = (os.getenv("CLAUDE_MODEL") or CLAUDE_MODEL or "").strip()
        print("CLAUDE_MODEL repr:", repr(model)) 

        resp = claude_client.messages.create(
            model=model,            
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        combined = "".join(getattr(b, "text", "") for b in getattr(resp, "content", [])).strip()

        if combined.startswith("```json"):
            combined = combined.replace("```json", "").replace("```", "").strip()
        elif combined.startswith("```"):
            combined = combined.replace("```", "").strip()

        try:
            return json.loads(combined)
        except Exception:
            return combined

    except Exception as e:
        print(f"[ERROR] Claude call failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Claude call failed: {str(e)}"}
    


def run_ai_with_cache(task: str, provider: str, prompt: str):
    """
    Shared cached AI runner for non-grading features.
    Cache key is based on task + provider + full prompt.
    """
    provider = (provider or "openai").strip().lower()
    task = (task or "general").strip().lower()

    cached = get_cached_ai_response(task, provider, prompt)
    if cached is not None:
        print(f"[CACHE HIT] task={task} provider={provider}")
        return {
            "data": cached,
            "cached": True
        }

    print(f"[CACHE MISS] task={task} provider={provider}")

    if provider == "claude":
        result = query_claude_api(prompt)
    else:
        result = query_openai_api(prompt)

    set_cached_ai_response(task, provider, prompt, result)

    return {
        "data": result,
        "cached": False
    }

def generate_prompt(content: str, task: str, **kwargs) -> str:
    """Build prompts for upload features."""
    task = (task or "").strip().lower()

    if task == "concepts":
        return (
            "Identify the core concepts from the content.\n"
            "Return JSON ONLY in this format:\n"
            "{ 'concepts': [ { 'title': '...', 'description': '...' } ] }\n\n"
            "Rules:\n"
            "- Extract the main concepts only\n"
            "- Keep descriptions concise and clear\n"
            "- Do not return markdown\n"
            "- Do not return bullets\n\n"
            "Content:\n"
            + content
        )

    if task == "questions":
        total = int(kwargs.get("total_questions", 20))
        mcq = int(kwargs.get("mcq_count", 5))
        tf = int(kwargs.get("tf_count", 5))
        other = int(kwargs.get("others_count", max(0, total - mcq - tf)))
        return (
            f"Generate {total} questions from the provided content.\n"
            f"- MCQs: {mcq}\n- True/False: {tf}\n- Other: {other}\n"
            "Return JSON ONLY with this schema:\n"
            "{ 'questions': [ { 'type': 'MCQ|TF|Short|Long', 'question': '...', 'options': ['A..','B..','C..','D..'] (MCQ only), 'answer': '...', 'difficulty': 'easy|medium|hard', 'reference': 'page/slide ...' } ] }\n\n"
            "Content:\n"
            + content
        )

    if task == "structured":
        # syllabus enhancement
        return (
            "Analyze the following course syllabus and provide structured recommendations.\n"
            "Return JSON ONLY with this schema:\n"
            "{\n"
            "  'course_overview': '...',\n"
            "  'learning_objectives': ['...'],\n"
            "  'pacing_suggestions': ['...'],\n"
            "  'assessment_alignment': ['...'],\n"
            "  'clarity_improvements': ['...'],\n"
            "  'red_flags_or_gaps': ['...']\n"
            "}\n\n"
            "Syllabus Content:\n"
            + content
        )

    if task == "recommendation":
        today = datetime.today().strftime("%B %d, %Y")
        recommender_name = (kwargs.get("recommender_name") or "").strip()
        position = (kwargs.get("position") or "").strip()
        email = (kwargs.get("email") or "").strip()

        header = "To Whom It May Concern,"
        if recommender_name and position and email:
            header = (
                f"{recommender_name}\n{position}\n{email}\n\nTo Whom It May Concern,"
            )

        return (
            "Write a formal, professional recommendation letter for the candidate described in the resume below.\n"
            f"Date: {today}\n\n"
            f"Use this recommender header exactly:\n{header}\n\n"
            "Include: relationship/context, 3–5 strengths with examples, and a strong closing endorsement.\n"
            "Formatting requirements (must follow exactly):\n"
            """
                - Use plain text only.
                - Preserve line breaks.
                - The top block must be exactly:

                <Recommender Name>
                <Recommender Role/Title or leave blank>
                <Recommender Email or leave blank>
                <Month DD, YYYY>

                (blank line)
                To Whom It May Concern,

                (blank line)
                <Body paragraphs separated by a blank line>

                (blank line)
                Sincerely,

                <Recommender Name only>

                Rules:
                - Do not put name/title/email on the same line.
                - "Sincerely," must be on its own line.
                - The final line must contain only the recommender name (no title, no email).

            """
            "Resume Content:\n"
            + content
        )

    if task == "exam_papers":
        num_students = int(kwargs.get("num_students", 1))
        qps = int(kwargs.get("questions_per_student", 5))
        return (
            f"Generate {num_students} unique exam papers.\n"
            f"Each exam must contain exactly {qps} questions.\n"
            "The exams should be equivalent difficulty and cover the same key topics.\n"
            "Return JSON ONLY with this schema:\n"
            "{ 'exams': [ { 'student': 1, 'questions': [ { 'question': '...', 'answer': '...', 'difficulty': 'easy|medium|hard', 'reference': 'slide/page ...' } ] } ] }\n\n"
            "Lecture Content:\n"
            + content
        )

    if task == "research":
        audience = (kwargs.get("audience") or "General academic audience").strip()
        depth = (kwargs.get("depth") or "Detailed").strip()
        scope = (kwargs.get("scope") or "General overview").strip()
        today = datetime.today().strftime("%B %d, %Y")

        return (
            "You are a senior research analyst. Produce a structured research document, not a casual chat response.\n\n"
            f"Topic:\n{content}\n\n"
            f"Audience:\n{audience}\n\n"
            f"Depth:\n{depth}\n\n"
            f"Scope:\n{scope}\n\n"
            "STRICTLY follow this structure and use the headings exactly as written:"
            f"# {content} — Research Brief\n\n"
            f"**Prepared:** {today}\n"
            f"**Scope:** {scope}\n\n"
            "## Executive Summary\n"
            "[3-5 sentences with the most important findings]\n\n"
            "## Background\n"
            "[Definitions, context, and why this matters]\n\n"
            "## Key Findings\n\n"
            "### Finding 1\n"
            "[Explain clearly with specifics]\n"
            "- Source: [Link or reference if available]\n\n"
            "### Finding 2\n"
            "[Explain clearly with specifics]\n"
            "- Source: [Link or reference if available]\n\n"
            "### Finding 3\n"
            "[Explain clearly with specifics]\n"
            "- Source: [Link or reference if available]\n\n"
            "## Data & Statistics\n"
            "- [Stat 1] — [Source]\n"
            "- [Stat 2] — [Source]\n"
            "- [Stat 3] — [Source]\n\n"
            "## Considerations\n"
            "[Risks, limitations, conflicting viewpoints, uncertainties]\n\n"
            "## Recommendations\n"
            "[Actionable next steps based on the research]\n\n"
            "## Sources\n"
            "1. [Full citation or reference]\n"
            "2. [Full citation or reference]\n"
            "3. [Full citation or reference]\n\n"
            "Rules:\n"
            "Do not repeat the raw input text.\n"
            "Do not paste the uploaded content verbatim.\n"
            "Use the content only as source material to create the research brief.\n"
            "Start directly with the title and structured sections.\n"
            "- Write in plain, professional language.\n"
            "- Make the output self-contained.\n"
            "- Lead with the most important information.\n"
            "- Include actual numbers or examples when possible.\n"
            "- If something cannot be verified, say so clearly.\n"
            "- Keep it structured and document-style.\n"
            "If exact sources are uncertain, clearly label them as illustrative examples.\n"
        )

    return "Task not supported."


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()

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
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()

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


@app.route("/debug_session")
def debug_session():
    """Small debug endpoint to confirm session values."""
    return jsonify({k: session.get(k) for k in ("user_id", "username")})

@app.route("/set_lang/<lang>")
def set_lang(lang):
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    session["lang"] = lang
    return redirect(request.referrer or url_for("dashboard"))


# -----------------------------------------------------------------------------
# Routes: Grading (sync)
# -----------------------------------------------------------------------------
@app.route("/grade", methods=["GET"])
@login_required
def grade_form():
    return redirect(url_for("dashboard"))


@app.route("/grade", methods=["POST"])
@login_required
def grade_run():
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def ajax_error(msg: str, status: int = 400):
        return make_response(f'<div class="error">{msg}</div>', status)

    key_pdf = request.files.get("key_pdf")
    ans_pdf = request.files.get("answer_pdf") or request.files.get("ans_pdf")

    if not key_pdf or not ans_pdf:
        return ajax_error("Please upload both files.") if is_ajax else redirect(url_for("dashboard"))

    task_id = uuid.uuid4().hex
    key_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_key.pdf")
    ans_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_ans.pdf")

    try:
        key_pdf.save(key_path)
        ans_pdf.save(ans_path)

        force_ocr = "force_ocr" in request.form
        grading_provider = (request.form.get("grading_provider") or "groq").strip().lower()


        # Extract once, then grade from text (lets us reuse extraction in async too)
        key_text = extract_text_from_pdf_auto(key_path)
        ans_text = extract_text_from_pdf_auto(ans_path, force_ocr=force_ocr)

        key_doc_id = stable_doc_id(key_text, "key")
        stu_doc_id = stable_doc_id(ans_text, "student")

        key_qas = split_into_qas_by_qid(key_text)
        stu_qas = split_into_qas_by_qid(ans_text)

        put_doc(key_doc_id, role="key", qas=key_qas)
        put_doc(stu_doc_id, role="student", qas=stu_qas)

        results, avg_score, key_qas = process_grading(
            key_text, ans_text,
            from_text=True,
            strictness="normal",
            force_ocr=force_ocr,
            grading_provider=grading_provider,
            key_doc_id=key_doc_id,
            student_doc_id=stu_doc_id,
        )

        # Weighting (optional)
        average_display = None
        try:
            results, total_awarded, total_possible, weighted_pct = apply_dynamic_weights(results, key_qas)
            average_display = f"Total: {total_awarded:.2f} / {total_possible} points ({weighted_pct}%)"
        except Exception as e:
            print(f"[WARN] Weight calc failed: {e}")

        # Save JSON output for download
        out_name = f"grading_results_{uuid.uuid4().hex}.json"
        out_path = os.path.join(app.config["OUTPUT_FOLDER"], out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        template = "grade_result_fragment.html" if is_ajax else "grade_result.html"
        html_output = render_template(
            template,
            results=results,
            avg_score=avg_score,
            average_display=average_display,
            download_filename=out_name,
        )

        resp = make_response(html_output)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    except Exception as e:
        print(f"[ERROR] grade_run failed: {e}\n{traceback.format_exc()}")
        return ajax_error("Grading failed. Check server logs.", 500) if is_ajax else redirect(url_for("dashboard"))
    finally:
        for p in (key_path, ans_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# Routes: Grading (async)
# -----------------------------------------------------------------------------
TASKS: Dict[str, Dict[str, Any]] = {}


def _async_worker(task_id: str, key_path: str, ans_path: str, force_ocr: bool = False, grading_provider: str = "groq"):
    print(f"[INFO] async worker started: {task_id} (force_ocr={force_ocr})")

    # --- NEW: timestamps for timeout + better status reporting ---
    TASKS[task_id]["started_at"] = time.time()
    TASKS[task_id]["phase"] = "start"
    TASKS[task_id]["error"] = None

    try:
        # -------------------- EXTRACT --------------------
        TASKS[task_id]["phase"] = "extract"
        TASKS[task_id]["status"] = "Extracting text (key)"
        key_text = extract_text_from_pdf_auto(key_path, force_ocr=force_ocr)  # keep consistent

        TASKS[task_id]["status"] = "Extracting text (student)"
        ans_text = extract_text_from_pdf_auto(ans_path, force_ocr=force_ocr)

        # -------------------- PRE-CHECK --------------------
        # If extraction produced almost nothing, fail fast (avoids wasting calls)
        if not key_text or len(key_text.strip()) < 100:
            raise RuntimeError("Key text extraction too short/empty.")
        if not ans_text or len(ans_text.strip()) < 100:
            raise RuntimeError("Student text extraction too short/empty.")

        # -------------------- GRADE --------------------
        TASKS[task_id]["phase"] = "grade"
        TASKS[task_id]["status"] = "Grading (parsing questions)"

        key_doc_id = stable_doc_id(key_text, "key")
        stu_doc_id = stable_doc_id(ans_text, "student")

        key_qas_map = split_into_qas_by_qid(key_text)
        stu_qas_map = split_into_qas_by_qid(ans_text)

        put_doc(key_doc_id, role="key", qas=key_qas_map)
        put_doc(stu_doc_id, role="student", qas=stu_qas_map)

        # process_grading already uses your robust splitters, preprocess, etc.
        results, avg_score, key_qas = process_grading(
            key_text, ans_text,
            from_text=True,
            strictness="normal",
            force_ocr=force_ocr,
            grading_provider=grading_provider,
            key_doc_id=key_doc_id,
            student_doc_id=stu_doc_id,
        )

        # --- NEW: cap runaway parse explosions ---
        # If something created 50+ fake questions, stop.
        # (Tune the limit if you have truly huge exams.)
        if results and len(results) > 30:
            raise RuntimeError(
                f"Parsed too many question blocks ({len(results)}). "
                "Likely OCR/page-marker noise caused split explosion."
            )

        # -------------------- WEIGHTS --------------------
        TASKS[task_id]["phase"] = "weight"
        TASKS[task_id]["status"] = "Computing totals"
        try:
            results, total_awarded, total_possible, weighted_pct = apply_dynamic_weights(results, key_qas)
            TASKS[task_id]["average_display"] = f"Total: {total_awarded:.2f} / {total_possible} points ({weighted_pct}%)"
        except Exception:
            TASKS[task_id]["average_display"] = None

        # -------------------- DONE --------------------
        TASKS[task_id]["phase"] = "done"
        TASKS[task_id]["status"] = f"Completed (Avg Score {avg_score}/10)"
        TASKS[task_id]["result"] = results

    except Exception as e:
        TASKS[task_id]["phase"] = "error"
        TASKS[task_id]["status"] = f"Failed: {e}"
        TASKS[task_id]["error"] = f"{type(e).__name__}: {e}"
        TASKS[task_id]["result"] = None
        print(f"[ERROR] async worker: {e}\n{traceback.format_exc()}")

    finally:
        TASKS[task_id]["finished_at"] = time.time()

        for p in (key_path, ans_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.route("/grade_async", methods=["POST"])
@login_required
def grade_async():
    key_file = request.files.get("key_pdf")
    ans_file = request.files.get("ans_pdf") or request.files.get("answer_pdf")
    if not key_file or not ans_file:
        return jsonify({"error": t("upload_both")}), 400
    

    task_id = str(uuid.uuid4())
    key_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_key.pdf")
    ans_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_ans.pdf")

    key_file.save(key_path)
    ans_file.save(ans_path)

    force_ocr = "force_ocr" in request.form

    # ✅ NEW: add timestamps + error fields so status_json can stop polling safely
    TASKS[task_id] = {
        "status": "Starting",
        "result": None,
        "average_display": None,
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "phase": "start",
    }

    grading_provider = (request.form.get("grading_provider") or "groq").strip().lower()


    threading.Thread(
        target=_async_worker,
        args=(task_id, key_path, ans_path, force_ocr, grading_provider),
        daemon=True
    ).start()

    return jsonify({"task_id": task_id})


@app.route("/status_json/<task_id>")
@login_required
def status_json(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"error": "Task not found"}, 404

    status_text = str(task.get("status", "") or "")
    phase = task.get("phase")
    started_at = task.get("started_at")

    # ✅ NEW: hard timeout (prevents infinite polling)
    # Tune this as needed (10 min is a safe starting point)
    TIMEOUT_SECONDS = 600
    if started_at and not status_text.startswith("Completed") and not status_text.startswith("Failed"):
        elapsed = time.time() - float(started_at)
        if elapsed > TIMEOUT_SECONDS:
            task["status"] = "Failed: Timeout (grading took too long)"
            task["error"] = "Timeout: grading exceeded time limit; likely parse explosion or API delay."
            task["phase"] = "error"
            task["finished_at"] = time.time()
            status_text = task["status"]
            phase = task["phase"]

    # ✅ If failed, stop polling and return the error
    if status_text.startswith("Failed"):
        return {
            "status": status_text,
            "done": True,
            "error": task.get("error") or status_text
        }

    # ✅ If not completed yet, keep polling
    if not status_text.startswith("Completed"):
        return {
            "status": status_text,
            "done": False,
            "phase": phase
        }

    # ✅ Completed → return result HTML fragment
    avg_match = re.search(r"Avg Score ([0-9.]+)/10", status_text)
    avg_score = avg_match.group(1) if avg_match else None

    html = render_template(
        "grade_result_fragment.html",
        results=task.get("result", {}) or {},
        avg_score=avg_score,
        average_display=task.get("average_display"),
    )
    return {"status": "done", "done": True, "html": html}


@app.route("/status/<task_id>")
@login_required
def task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return "Task not found", 404

    if not str(task.get("status", "")).startswith("Completed"):
        return f"""
        <h3>Status: {task.get('status','')}</h3>
        <meta http-equiv='refresh' content='5'>
        <p>Please wait. Grading may take a few minutes.</p>
        """

    return f"""
    <h3>Status: {task.get('status','')}</h3>
    <p>Redirecting to results.</p>
    <script>
      setTimeout(() => {{ window.location.href = '/result/{task_id}'; }}, 800);
    </script>
    """


@app.route("/result/<task_id>")
@login_required
def task_result(task_id: str):
    task = TASKS.get(task_id)
    if not task or not task.get("result"):
        return "Result not found or grading not completed.", 404

    avg_match = re.search(r"Avg Score ([0-9.]+)/10", str(task.get("status", "")))
    avg_score = avg_match.group(1) if avg_match else "N/A"

    return render_template(
        "grade_result_fragment.html",
        results=task["result"],
        avg_score=avg_score,
        average_display=task.get("average_display"),
    )


# -----------------------------------------------------------------------------
# Routes: Upload features
# -----------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    """Handle non-grading upload utilities (syllabus/chapter/resume/exam)."""
    action = (request.form.get("action") or "").strip().lower()
    ai_model = (request.form.get("ai_model") or "openai").strip().lower()
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]

    def run_ai(prompt: str):
        return run_ai_with_cache(action, ai_model, prompt)
    # ----------------------------
    # Exam papers: multi PPT/PPTX
    # ----------------------------
    if action == "exam_papers":
        files = request.files.getlist("exam_files")
        if not files or all((not f or not f.filename) for f in files):
            msg = "No files selected. Please upload at least one PPT/PPTX or PDF."
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("dashboard"))

        extracted_texts: List[str] = []
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                continue

            fname = secure_filename(f.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{fname}")
            try:
                f.save(path)
                ext = fname.rsplit(".", 1)[1].lower()
                if ext in {"ppt", "pptx"}:
                    txt, _ = extract_text_from_ppt_with_slides(path)
                elif ext == "pdf":
                    txt = extract_text_from_pdf_auto(path)
                elif ext == "docx":
                    txt = extract_text_from_docx(path)
                elif ext == "txt":
                    txt = extract_text_from_txt(path)
                else:
                    txt = ""
                if txt and txt.strip():
                    extracted_texts.append(txt)
            except Exception as e:
                print(f"[ERROR] exam upload processing failed for {fname}: {e}\n{traceback.format_exc()}")
                continue

        if not extracted_texts:
            msg = "Could not extract text from the uploaded files."
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("dashboard"))

        combined = "\n\n".join(extracted_texts)

        try:
            num_students = int(request.form.get("num_students", 1))
            questions_per_student = int(request.form.get("questions_per_student", 5))
        except Exception:
            msg = "Please enter valid numbers for students and questions per student."
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("dashboard"))

        prompt = generate_prompt(
            combined,
            "exam_papers",
            num_students=num_students,
            questions_per_student=questions_per_student,
        )

        ai_response = run_ai(prompt)

        payload = {
            "action": action,
            "result": ai_response.get("data"),
        }
        if is_ajax:
            return jsonify(payload)

        # Non-AJAX fallback: render the dashboard with a visible result block
        return render_template("index.html", success=json.dumps(payload, indent=2))

    # ----------------------------
    # Single-file tasks
    # ----------------------------
    topic = request.form.get("topic", "").strip()

    file = (
        request.files.get("file")
        or request.files.get("chapter_file")
        or request.files.get("resume_file")
    )

    content = None
    filepath = None

    # Research supports topic-only input
    if action == "research" and topic:
        content = topic

    else:
        if not file or not file.filename:
            msg = "No file selected. Please upload a valid file."
            if action == "research":
                msg = "Please upload a file or enter a research topic."
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("dashboard"))

        if not allowed_file(file.filename):
            msg = "Unsupported file type."
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash(msg, "error")
            return redirect(url_for("dashboard"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
        file.save(filepath)

        try:
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

            if not content or not str(content).strip():
                msg = "The uploaded file appears to be empty or unsupported."
                if is_ajax:
                    return jsonify({"action": action, "error": msg}), 400
                flash(msg, "error")
                return redirect(url_for("dashboard"))

        except Exception as e:
            msg = f"Error extracting content from the uploaded file: {e}"
            if is_ajax:
                return jsonify({"action": action, "error": msg}), 400
            flash("Error extracting content from the uploaded file.", "error")
            return redirect(url_for("dashboard"))

    # Build prompt per action
    if action == "questions":
        prompt = generate_prompt(
            content,
            "questions",
            total_questions=int(request.form.get("total_questions", 20)),
            mcq_count=int(request.form.get("mcq_count", 5)),
            tf_count=int(request.form.get("tf_count", 5)),
            others_count=int(request.form.get("others_count", 10)),
        )
    elif action == "concepts":
        prompt = generate_prompt(content, "concepts")
    elif action == "structured":
        prompt = generate_prompt(content, "structured")
    elif action == "recommendation":
        prompt = generate_prompt(
            content,
            "recommendation",
            recommender_name=request.form.get("recommender_name", ""),
            position=request.form.get("position", ""),
            email=request.form.get("email", ""),
        )
    elif action == "research":
        prompt = generate_prompt(
            content,
            "research",
            audience=request.form.get("audience", "").strip(),
            depth=request.form.get("depth", "").strip(),
            scope=request.form.get("scope", "").strip(),
        )
    else:
        msg = f"Invalid action: {action}"
        if is_ajax:
            return jsonify({"action": action, "error": msg}), 400
        flash(msg, "error")
        return redirect(url_for("dashboard"))

    ai_response = run_ai(prompt)
    payload = {
        "action": action,
        "result": ai_response.get("data")
    }
    if is_ajax:
        return jsonify(payload)

    return render_template("index.html", success=json.dumps(payload, indent=2))

def stable_doc_id(text: str, role: str) -> str:
    """
    Build a stable cache identity from extracted text instead of random UUIDs.
    This allows repeated grading runs of the same content to hit cache.
    """
    version = "grading-v1"
    normalized = (text or "").strip()
    raw = f"{version}||{role}||{normalized}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

# -----------------------------------------------------------------------------
# Download routes for generated files (kept compatible with your templates)
# -----------------------------------------------------------------------------
@app.route("/download/questions/<filename>")
@login_required
def download_questions(filename: str):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/download/recommendations/<recommendation_filename>")
@login_required
def download_recommendations(recommendation_filename: str):
    path = os.path.join(app.config["OUTPUT_FOLDER"], recommendation_filename)
    return send_file(path, as_attachment=True, download_name=recommendation_filename)


@app.route("/download/exam_papers/<exam_filename>")
@login_required
def download_exam_papers(exam_filename: str):
    path = os.path.join(app.config["OUTPUT_FOLDER"], exam_filename)
    return send_file(path, as_attachment=True, download_name=exam_filename)


# -----------------------------------------------------------------------------
# Local dev entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # In production, prefer: gunicorn -w 2 -b 0.0.0.0:5000 app:app
    app.run(debug=True)
