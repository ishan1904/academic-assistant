# core/docstore_sqlite.py
import os, json, time
from sqlalchemy import create_engine, text

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mcp_docstore.sqlite")

_ENGINE = None

def _db_url() -> str:
    # Prefer MCP_DATABASE_URL if you want to isolate MCP tables from your user DB,
    # otherwise it will just use Railway's DATABASE_URL.
    url = os.getenv("MCP_DATABASE_URL") or os.getenv("DATABASE_URL")

    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Local fallback: sqlite file
    if not url:
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        url = f"sqlite:///{os.path.abspath(SQLITE_PATH)}"

    return url

def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(_db_url(), pool_pre_ping=True, future=True)
    return _ENGINE

def _dialect_name() -> str:
    return _engine().dialect.name  # "sqlite" or "postgresql" etc.

def init_db():
    ddl_docs = """
    CREATE TABLE IF NOT EXISTS docs(
        doc_id TEXT PRIMARY KEY,
        role TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )
    """
    ddl_qas = """
    CREATE TABLE IF NOT EXISTS qas(
        doc_id TEXT NOT NULL,
        qid INTEGER NOT NULL,
        title TEXT,
        body TEXT,
        PRIMARY KEY(doc_id, qid)
    )
    """
    with _engine().begin() as con:
        con.execute(text(ddl_docs))
        con.execute(text(ddl_qas))

def put_doc(doc_id: str, role: str, qas: dict):
    d = _dialect_name()

    if d == "postgresql":
        upsert_docs = text("""
            INSERT INTO docs(doc_id, role, created_at)
            VALUES(:doc_id, :role, :created_at)
            ON CONFLICT (doc_id) DO UPDATE
            SET role = EXCLUDED.role,
                created_at = EXCLUDED.created_at
        """)
        upsert_qas = text("""
            INSERT INTO qas(doc_id, qid, title, body)
            VALUES(:doc_id, :qid, :title, :body)
            ON CONFLICT (doc_id, qid) DO UPDATE
            SET title = EXCLUDED.title,
                body  = EXCLUDED.body
        """)
    else:
        # sqlite
        upsert_docs = text("""
            INSERT OR REPLACE INTO docs(doc_id, role, created_at)
            VALUES(:doc_id, :role, :created_at)
        """)
        upsert_qas = text("""
            INSERT OR REPLACE INTO qas(doc_id, qid, title, body)
            VALUES(:doc_id, :qid, :title, :body)
        """)

    now = time.time()

    with _engine().begin() as con:
        con.execute(upsert_docs, {"doc_id": doc_id, "role": role, "created_at": now})
        for qid, qa in (qas or {}).items():
            con.execute(
                upsert_qas,
                {
                    "doc_id": doc_id,
                    "qid": int(qid),
                    "title": (qa or {}).get("title", ""),
                    "body": (qa or {}).get("body", ""),
                },
            )

def get_role(doc_id: str):
    with _engine().connect() as con:
        row = con.execute(text("SELECT role FROM docs WHERE doc_id=:doc_id"), {"doc_id": doc_id}).fetchone()
        return row[0] if row else None

def list_qids(doc_id: str):
    with _engine().connect() as con:
        rows = con.execute(
            text("SELECT qid FROM qas WHERE doc_id=:doc_id ORDER BY qid"),
            {"doc_id": doc_id},
        ).fetchall()
        return [r[0] for r in rows]

def get_qa(doc_id: str, qid: int):
    with _engine().connect() as con:
        row = con.execute(
            text("SELECT title, body FROM qas WHERE doc_id=:doc_id AND qid=:qid"),
            {"doc_id": doc_id, "qid": int(qid)},
        ).fetchone()
        if not row:
            return None
        return {"title": row[0], "body": row[1]}