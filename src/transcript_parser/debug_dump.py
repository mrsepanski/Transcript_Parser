from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Pattern, Sequence, Set

try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore


@dataclass
class Token:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass
class Row:
    y: float
    toks: List[Token]


def _to_str(v: object) -> str:
    s = "" if v is None else str(v)
    return s.strip()


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def parse_pages_arg(p: Optional[str]) -> Optional[Set[int]]:
    if not p:
        return None
    parts: List[int] = []
    for chunk in p.split(","):
        chunk_s = chunk.strip()
        if not chunk_s:
            continue
        if "-" in chunk_s:
            a, b = chunk_s.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
            except ValueError:
                continue
            start, end = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            parts.extend(range(start, end + 1))
        else:
            try:
                parts.append(int(chunk_s))
            except ValueError:
                continue
    return set(parts)


def group_words_into_rows(words: Sequence[dict], y_tol: float = 3.0) -> List[Row]:  # type: ignore[type-arg]
    rows: List[Row] = []
    cur: Optional[Row] = None
    for w in words:
        t = _to_str(w.get("text"))
        if not t:
            continue
        top = _to_float(w.get("top"))
        x0 = _to_float(w.get("x0"))
        x1 = _to_float(w.get("x1"), x0)
        bottom = _to_float(w.get("bottom"), top + 8.0)
        tok = Token(t, x0, x1, top, bottom)
        if cur is None or abs(top - cur.y) > y_tol:
            if cur is not None:
                cur.toks.sort(key=lambda tt: tt.x0)
                rows.append(cur)
            cur = Row(y=top, toks=[tok])
        else:
            cur.toks.append(tok)
    if cur is not None:
        cur.toks.sort(key=lambda tt: tt.x0)
        rows.append(cur)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="debug-dump", description="Dump pdfplumber token rows for debugging"
    )
    ap.add_argument("pdf", help="Path to PDF")
    ap.add_argument("--pages", help="Pages to include, e.g. 1-2,4", default=None)
    ap.add_argument("--grep", help="Regex to filter rows", default=None)
    args = ap.parse_args()

    if pdfplumber is None:
        print("pdfplumber unavailable in this environment.")
        return

    path = Path(args.pdf)
    if not path.exists():
        print("File not found:", path)
        return

    page_set = parse_pages_arg(args.pages)
    rx: Optional[Pattern[str]] = re.compile(args.grep, re.I) if args.grep else None

    with pdfplumber.open(path) as pdf:  # type: ignore[misc]
        for pidx, page in enumerate(pdf.pages, start=1):  # type: ignore[attr-defined]
            if page_set and pidx not in page_set:
                continue
            words = page.extract_words() or []  # type: ignore[call-arg, assignment]
            words.sort(key=lambda w: (_to_float(w.get("top")), _to_float(w.get("x0"))))  # type: ignore[index]
            rows = group_words_into_rows(words, y_tol=3.0)
            for r in rows:
                joined = " ".join(tok.text for tok in r.toks)
                if rx and not rx.search(joined):
                    continue
                print(f"[page {pidx} y={r.y:.1f}] {joined}")
                for tok in r.toks:
                    print(f"   - {tok.text!r} @ x0={tok.x0:.1f}..{tok.x1:.1f} y={tok.y0:.1f}")
                print("-" * 60)


if __name__ == "__main__":
    main()
