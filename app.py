# -*- coding: utf-8 -*-
"""
Grading Agent (Flask)

This app supports:
- User auth (signup/login/logout) with a simple SQLite database
- Grading (sync + async) by comparing an answer key PDF to a student's PDF
- PDF extraction with a fast typed-text path (PyPDF2) and an OCR fallback (Gemini)
- Basic “upload utilities” (question generation, concepts, recommendation letters, exam papers)

Notes for beginners:
- The “key” in this code is the correct answer sheet.
- The “answer” is the student's submission.
- Most endpoints require login (see @login_required).
"""

from __future__ import annotations

import os
import re
import json
import uuid
import traceback
import threading
import socket
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Tuple, Optional

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


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
load_dotenv(override=True)
socket.setdefaulttimeout(180)  # global network timeout (seconds)

app = Flask(__name__)
WSGIRequestHandler.protocol_version = "HTTP/1.1"

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

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
claude_client = anthropic.Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------------------------------------------------------
# Database (SQLite)
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL

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

        model = genai.GenerativeModel("gemini-2.5-flash")
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
    t = raw
    t = re.sub(r"-{3,}", "\n", t)
    t = re.sub(r"(?i)End of (Answer Key|simulated student answers).*", "", t)
    t = re.sub(r"(?i)(?=\bExercise\s*\d+\b)", "\n", t)
    t = re.sub(r"(?i)(?=\bQuestion\s*\d+\b)", "\n", t)
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
    key_question: Dict[str, Any],
    q_index: int,
    student_blocks_ex: Dict[str, str],
    student_blocks_qa: Dict[str, str],
    full_student_text: str,
) -> Tuple[str, str]:
    """
    Choose the best student snippet to grade for this question.

    Priority:
    1) Match by explicit “Exercise N” anchors in the key
    2) If key mentions Exercise N, try qN (NOT q_index)
    3) Fallback to q_index only if we have no better clue
    4) Full fallback
    """
    qtext = (key_question.get("question") or "")
    qtext_l = qtext.lower()

    # 1) Exercise anchor match (best)
    anchors = build_key_anchors(qtext)
    for a in anchors:
        if a in student_blocks_ex:
            return student_blocks_ex[a], "exercise-anchor"

    # 2) If the key explicitly says "Exercise N", use that N to match qN
    m = re.search(r"(?i)\bexercise\s*[:\-]?\s*(\d+)\b", qtext_l)
    if m:
        ex_num = m.group(1)
        qid = f"q{ex_num}"
        if qid in student_blocks_qa:
            return student_blocks_qa[qid], "numbered-qa-ex"

    # 3) Only now fall back to index mapping
    qid = f"q{q_index}"
    if qid in student_blocks_qa:
        return student_blocks_qa[qid], "numbered-qa-index"

    return full_student_text, "full-fallback"




# -----------------------------------------------------------------------------
# Grading pipeline
# -----------------------------------------------------------------------------
def robust_split_into_qas(text: str) -> List[Dict[str, str]]:
    """
    Primary splitter used for both key and student sheets.

    It tries to find blocks like:
      Exercise 1: ...
      Question 2 - ...

    If it can't find anything useful, it falls back to a coarse chunk split.
    """
    if not text:
        return []

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    pattern = re.compile(
        r"(?:(?<=\n)|^)\s*((?:Exercise|Question|Problem|Task|Q)\s*\d+[A-Za-z]?[\.:)\-–]?\s*.*?)"
        r"(?=(?:\n\s*(?:Exercise|Question|Problem|Task|Q)\s*\d+|\Z))",
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
        return out

    # last-resort fallback: paragraph chunks
    chunks = [c.strip() for c in re.split(r"\n\s*\n", t) if len(c.strip().split()) > 5]
    return [{"question": c, "answer": ""} for c in chunks]


def process_grading(
    key_source: str,
    ans_source: str,
    *,
    from_text: bool = False,
    strictness: str = "normal",
    force_ocr: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], float, Dict[int, Dict[str, str]]]:
    """
    Main grading pipeline.
    Returns:
      results: {1: {...}, 2: {...}}
      avg_score (0-10 scale)
      key_qas_dict: {1: {"question":..., "answer":...}, ...} used for weighting
    """
    from grading_agent import grade_submission  # your Groq-based grader

    # 1) Load and preprocess text
    if from_text:
        key_text, ans_text = key_source, ans_source
    else:
        key_text = extract_text_from_pdf_auto(key_source)
        ans_text = extract_text_from_pdf_auto(ans_source, force_ocr=force_ocr)

    key_text = _clean_symbols(_preprocess_exam_text(_sanitize_equations(key_text)))
    ans_text = _clean_symbols(_preprocess_exam_text(_sanitize_equations(ans_text)))

    if not key_text.strip():
        return {}, 0.0, {}

    # 2) Parse Q/A blocks
    key_qas_list = robust_split_into_qas(key_text)
    if not key_qas_list:
        return {}, 0.0, {}

    full_student_text = (ans_text or "").strip()

    # 3) Build student block indexes once (avoid re-splitting in the loop)
    student_blocks_ex = {
        re.sub(r"\s+", " ", k.strip().lower()): v
        for k, v in split_by_exercise(full_student_text).items()
    }
    student_blocks_qa = split_numbered_qa_pairs(full_student_text)

    # 4) Grade each question
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

        # Add a small amount of extra context (helps when student answers are messy)
        student_context = full_student_text[:1200]
        student_input = f"[ANSWER BLOCK]\n{student_for_grading}\n\n[EXTRA CONTEXT]\n{student_context}"

        graded = grade_submission(rubric, student_input, strictness)
        score = float(graded.get("score", 0) or 0)

        total_score += score
        scored += 1

        results[idx] = {
            "question": kqa.get("question", "") or f"Q{idx}",
            "correct_answer": rubric,
            "student_answer": student_for_display or "[No matched student block detected]",
            "evaluation": graded.get("feedback", ""),
            "score": score,
            "match_mode": match_mode,
        }

        print(f"[DEBUG] Q{idx} match={match_mode} score={score} preview={student_for_display[:120]!r}")

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
    if not openai_client:
        return {"error": "OpenAI API key not configured."}
    resp = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an experienced instructor."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    # If the model returns JSON-as-text, try to parse it.
    try:
        return json.loads(content)
    except Exception:
        return content


def query_claude_api(prompt: str):
    if not claude_client:
        return {"error": "Anthropic API key not configured."}
    resp = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    combined = "".join(getattr(b, "text", "") for b in getattr(resp, "content", [])).strip()
    try:
        return json.loads(combined)
    except Exception:
        return combined


def generate_prompt(content: str, task: str, **kwargs) -> str:
    """Build prompts for upload features."""
    if task == "concepts":
        return (
            "Identify core concepts from the content. Output 10–15 concise bullet points, each "
            "'Concept: short explanation'.\n\nContent:\n"
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
            "For each question include: question text, correct answer, difficulty, and a reference page/slide.\n\n"
            "Content:\n"
            + content
        )

    if task == "recommendation":
        today = datetime.today().strftime("%B %d, %Y")
        return (
            "Write a formal 'To Whom It May Concern' recommendation letter.\n"
            f"Date: {today}\n\n"
            + content
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

        # Extract once, then grade from text (lets us reuse extraction in async too)
        key_text = extract_text_from_pdf_auto(key_path)
        ans_text = extract_text_from_pdf_auto(ans_path, force_ocr=force_ocr)

        results, avg_score, key_qas = process_grading(
            key_text, ans_text, from_text=True, strictness="normal", force_ocr=force_ocr
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


def _async_worker(task_id: str, key_path: str, ans_path: str, force_ocr: bool = False):
    print(f"[INFO] async worker started: {task_id} (force_ocr={force_ocr})")
    try:
        TASKS[task_id]["status"] = "Extracting text"
        key_text = extract_text_from_pdf_auto(key_path)
        ans_text = extract_text_from_pdf_auto(ans_path, force_ocr=force_ocr)

        TASKS[task_id]["status"] = "Grading"
        results, avg_score, key_qas = process_grading(key_text, ans_text, from_text=True, strictness="normal")

        try:
            results, total_awarded, total_possible, weighted_pct = apply_dynamic_weights(results, key_qas)
            TASKS[task_id]["average_display"] = f"Total: {total_awarded:.2f} / {total_possible} points ({weighted_pct}%)"
        except Exception:
            TASKS[task_id]["average_display"] = None

        TASKS[task_id]["status"] = f"Completed (Avg Score {avg_score}/10)"
        TASKS[task_id]["result"] = results
    except Exception as e:
        TASKS[task_id]["status"] = f"Failed: {e}"
        TASKS[task_id]["result"] = None
        print(f"[ERROR] async worker: {e}\n{traceback.format_exc()}")
    finally:
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
        return jsonify({"error": "Please upload both files."}), 400

    task_id = str(uuid.uuid4())
    key_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_key.pdf")
    ans_path = os.path.join(app.config["TEMP_FOLDER"], f"{task_id}_ans.pdf")

    key_file.save(key_path)
    ans_file.save(ans_path)

    force_ocr = "force_ocr" in request.form
    TASKS[task_id] = {"status": "Starting", "result": None, "average_display": None}

    threading.Thread(target=_async_worker, args=(task_id, key_path, ans_path, force_ocr), daemon=True).start()
    return jsonify({"task_id": task_id})


@app.route("/status_json/<task_id>")
@login_required
def status_json(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"error": "Task not found"}, 404

    if not str(task.get("status", "")).startswith("Completed"):
        return {"status": task.get("status", ""), "done": False}

    status_text = str(task.get("status", ""))
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
    file = (
        request.files.get("file")
        or request.files.get("exam_files")
        or request.files.get("chapter_file")
        or request.files.get("resume_file")
    )

    if not file or not file.filename:
        flash("No file selected. Please upload a valid file.", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Unsupported file type.", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
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

        if not content.strip():
            flash("The uploaded file appears to be empty or unsupported.", "error")
            return redirect(url_for("dashboard"))

        action = (request.form.get("action") or "").strip().lower()
        ai_model = (request.form.get("ai_model") or "openai").strip().lower()

        prompt = generate_prompt(
            content,
            action,
            total_questions=request.form.get("total_questions", 20),
            mcq_count=request.form.get("mcq_count", 5),
            tf_count=request.form.get("tf_count", 5),
            others_count=request.form.get("others_count", 10),
        )

        if ai_model == "claude":
            result = query_claude_api(prompt)
        else:
            result = query_openai_api(prompt)

        # Render on dashboard or return JSON (depending on your templates)
        return jsonify({"action": action, "result": result})

    finally:
        # keep uploads if you want a history; otherwise delete here
        pass


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
