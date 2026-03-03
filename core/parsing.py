# core/parsing.py
import re

QID_HEADER_RE = re.compile(
    r"(?im)^(?:\s*(Exercise|Question|Q)\s*(\d+)[A-Za-z]?\s*[:.\-–)]|\s*\((\d+)\)|\s*(\d+)\s*[\).:\-–])"
)

def split_into_qas_by_qid(text: str) -> dict:
    if not text:
        return {}
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    blocks, cur = [], []

    def is_header(line: str) -> bool:
        return bool(QID_HEADER_RE.match(line.strip()))

    for line in lines:
        if is_header(line):
            if cur:
                blocks.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    qas = {}
    for blk in blocks:
        header = (blk[0] if blk else "").strip()
        m = QID_HEADER_RE.match(header)
        if not m:
            continue

        qid = None
        for gi in (2, 3, 4):
            g = m.group(gi) if m and gi <= m.lastindex else None
            if g and g.isdigit():
                qid = int(g)
                break
        if qid is None:
            continue

        # Include header line content after the QID marker if answer is inline
        header = (blk[0] if blk else "").strip()
        body_lines = blk[1:]

        # If no body lines, extract remainder of header after the matched prefix
        if not body_lines:
            # remove the matched QID prefix
            m = QID_HEADER_RE.match(header)
            if m:
                body_text = header[m.end():].strip()
            else:
                body_text = ""
        else:
            body_text = "\n".join(body_lines).strip()

        qas[qid] = {"title": header, "body": body_text}
                
    return qas