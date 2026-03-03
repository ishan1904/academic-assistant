# core/cache.py
import os, json, time, hashlib
from sqlalchemy import create_engine, text

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mcp_docstore.sqlite")

_ENGINE = None

def _db_url() -> str:
    url = os.getenv("MCP_DATABASE_URL") or os.getenv("DATABASE_URL")

    if url and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

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
    return _engine().dialect.name

def init_cache():
    ddl = """
    CREATE TABLE IF NOT EXISTS grade_cache(
        cache_key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )
    """
    with _engine().begin() as con:
        con.execute(text(ddl))

def _key(*parts) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_cached(key_doc_id, student_doc_id, qid, strictness, provider):
    ck = _key(key_doc_id, student_doc_id, qid, strictness, provider)
    with _engine().connect() as con:
        row = con.execute(
            text("SELECT value_json FROM grade_cache WHERE cache_key=:ck"),
            {"ck": ck},
        ).fetchone()
        return json.loads(row[0]) if row else None

def set_cached(key_doc_id, student_doc_id, qid, strictness, provider, value: dict):
    ck = _key(key_doc_id, student_doc_id, qid, strictness, provider)
    d = _dialect_name()

    if d == "postgresql":
        stmt = text("""
            INSERT INTO grade_cache(cache_key, value_json, created_at)
            VALUES(:ck, :vj, :ts)
            ON CONFLICT (cache_key) DO UPDATE
            SET value_json = EXCLUDED.value_json,
                created_at = EXCLUDED.created_at
        """)
    else:
        stmt = text("""
            INSERT OR REPLACE INTO grade_cache(cache_key, value_json, created_at)
            VALUES(:ck, :vj, :ts)
        """)

    with _engine().begin() as con:
        con.execute(stmt, {"ck": ck, "vj": json.dumps(value), "ts": time.time()})