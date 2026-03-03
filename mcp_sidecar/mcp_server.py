# mcp_sidecar/mcp_server.py
import os
import sys
import uuid
import pdfplumber
from mcp.server.fastmcp import FastMCP

# Ensure project root is on path BEFORE importing core/*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.parsing import split_into_qas_by_qid
from core.grading import grade_one
from core.docstore import init_db, put_doc, get_role, list_qids, get_qa
from core.cache import init_cache, get_cached, set_cached

init_db()
init_cache()

mcp = FastMCP("academic-assistant-mcp")


def extract_text_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        parts = []
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


@mcp.tool()
def ingest_pdf(path: str, role: str) -> dict:
    """
    role: 'key' or 'student'
    """
    text = extract_text_pdf(path)
    qas = split_into_qas_by_qid(text)
    doc_id = uuid.uuid4().hex
    put_doc(doc_id, role=role, qas=qas)
    qids = sorted(list(qas.keys()))
    return {"doc_id": doc_id, "role": role, "qids": qids, "count": len(qids)}


@mcp.tool()
def list_qids_tool(doc_id: str) -> dict:
    qids = list_qids(doc_id)
    return {"doc_id": doc_id, "qids": qids, "count": len(qids)}


@mcp.tool()
def grade_question(key_doc_id: str, student_doc_id: str, qid: int, strictness: str = "lenient") -> dict:
    qid = int(qid)

    if get_role(key_doc_id) != "key":
        return {"qid": qid, "error": "Invalid key_doc_id"}
    if get_role(student_doc_id) != "student":
        return {"qid": qid, "error": "Invalid student_doc_id"}

    # Cache check
    cached = get_cached(key_doc_id, student_doc_id, qid, strictness)
    if cached:
        return {"qid": qid, **cached, "cached": True}

    key_qa = get_qa(key_doc_id, qid)
    stu_qa = get_qa(student_doc_id, qid)

    if not key_qa:
        return {"qid": qid, "error": "Question not found in key"}

    rubric = key_qa["body"] or ""
    student_answer = (stu_qa["body"] if stu_qa else "") or ""

    print(f"[CTX] Q{qid}: rubric_chars={len(rubric)} student_chars={len(student_answer)}")

    out = grade_one(rubric, student_answer, strictness=strictness)
    set_cached(key_doc_id, student_doc_id, qid, strictness, out)

    return {"qid": qid, **out, "cached": False}


@mcp.tool()
def grade_all_common(key_doc_id: str, student_doc_id: str, strictness: str = "lenient") -> dict:
    # Only grade questions that exist in BOTH key and student
    if get_role(key_doc_id) != "key":
        return {"error": "Invalid key_doc_id"}
    if get_role(student_doc_id) != "student":
        return {"error": "Invalid student_doc_id"}

    key_qids = set(list_qids(key_doc_id))
    stu_qids = set(list_qids(student_doc_id))
    common = sorted(list(key_qids & stu_qids))

    results = {}
    total = 0.0

    for qid in common:
        r = grade_question(key_doc_id, student_doc_id, int(qid), strictness=strictness)
        # r includes qid + maybe cached flag; keep only essentials
        results[int(qid)] = {
            "score_0_10": r.get("score_0_10", 0),
            "feedback": r.get("feedback", ""),
            "provider": r.get("provider", ""),
            "cached": r.get("cached", False),
        }
        total += float(results[int(qid)]["score_0_10"] or 0)

    avg = round(total / len(common), 2) if common else 0.0
    return {"count": len(common), "qids": common, "avg_score_0_10": avg, "results": results}

@mcp.tool()
def grade_all_key_qids(key_doc_id: str, student_doc_id: str, strictness: str = "lenient") -> dict:
    if get_role(key_doc_id) != "key":
        return {"error": "Invalid key_doc_id"}
    if get_role(student_doc_id) != "student":
        return {"error": "Invalid student_doc_id"}

    key_qids = list_qids(key_doc_id)
    results = {}
    total = 0.0

    for qid in key_qids:
        r = grade_question(key_doc_id, student_doc_id, int(qid), strictness=strictness)
        # if student missing, grade_question already uses blank answer
        results[int(qid)] = {
            "score_0_10": r.get("score_0_10", 0),
            "feedback": r.get("feedback", ""),
            "provider": r.get("provider", ""),
            "cached": r.get("cached", False),
        }
        total += float(results[int(qid)]["score_0_10"] or 0)

    avg = round(total / len(key_qids), 2) if key_qids else 0.0
    return {"count": len(key_qids), "qids": key_qids, "avg_score_0_10": avg, "results": results}


if __name__ == "__main__":
    print("✅ MCP server starting: academic-assistant-mcp")
    mcp.run()