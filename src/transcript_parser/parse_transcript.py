from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

SUBJECT_ALIASES: dict[str, list[str]] = {
    "math": ["MATH", "MAT", "MTH", "MA", "MATG"],
    "stat": ["STAT", "STA"],
    "cs": ["CS", "CSC", "CSCI", "CSE", "COSC"],
    "physics": ["PHYS", "PHY"],
    "chem": ["CHEM", "CHM"],
    "bio": ["BIOL", "BIO"],
    "econ": ["ECON", "ECN"],
    "engr": ["ENGR", "EGR"],
}

COURSE_PAT_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _normalize_text(s: str) -> str:
    # Normalize common PDF oddities
    s = s.replace("\u00a0", " ")  # NBSP to space
    s = s.replace("\xa0", " ")  # alternate NBSP escape
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _course_pattern(prefixes: Iterable[str]) -> re.Pattern[str]:
    # Accept PREFIX 123, PREFIX-123, PREFIX:123, PREFIX\n123, and PREFIX123
    key = tuple(sorted({p.upper() for p in prefixes}))
    if key in COURSE_PAT_CACHE:
        return COURSE_PAT_CACHE[key]
    pfx_group = "(?:" + "|".join(re.escape(p) for p in key) + ")"
    sep = r"(?:[\s\u00A0:\-])*"  # spaces/NBSP/colon/dash, any count
    # Real word-boundary \b (important for tests expecting 'MATH 101')
    pat = re.compile(rf"(?<!\w){pfx_group}{sep}(\d{{3}}[A-Z]?)\b", re.IGNORECASE)
    COURSE_PAT_CACHE[key] = pat
    return pat


def _expand_subjects(subjects: Iterable[str]) -> list[str]:
    expanded: set[str] = set()
    for s in subjects:
        s_key = s.lower()
        if s_key in SUBJECT_ALIASES:
            expanded.update(SUBJECT_ALIASES[s_key])
        else:
            expanded.add(s.upper())
    return sorted(expanded)


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    if pdfplumber is None:
        return ""
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for p in pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            parts.append(t)
    return _normalize_text("\n".join(parts))


def find_courses_in_text(text: str, prefixes: Iterable[str]) -> list[tuple[str, str]]:
    pat = _course_pattern(prefixes)
    found: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\xa0", " ").replace("\u00a0", " ")
        for m in pat.finditer(line):
            # Isolate the alphabetic prefix
            prefix_match = m.group(0)[: m.group(0).find(m.group(1))]
            prefix_clean = re.sub(r"[^A-Za-z]", "", prefix_match).upper()
            code = f"{prefix_clean} {m.group(1)}"
            # Snippet: text after match, cut before next course token on same line
            after = line[m.end() :]
            after = re.split(r"\b[A-Z]{2,}\s*[-:\s]?\s*\d{3}[A-Z]?\b", after, maxsplit=1)[0]
            snippet = after.strip()
            found.append((code, snippet))
    return found


def run_file(path: Path, subjects: list[str]) -> tuple[list[tuple[str, str]], str]:
    prefixes = _expand_subjects(subjects)
    text = extract_pdf_text(path)
    matches = find_courses_in_text(text, prefixes)
    text_source = "pdf"
    # (Optional OCR fallback can be plugged here; current flow is pdf-first only.)
    return matches, text_source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="transcript-parser")
    parser.add_argument("inputs", nargs="+", help="PDF file(s)")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject labels, e.g. math stat cs")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    for inp in args.inputs:
        p = Path(inp)
        base = p.name
        print(f"Results for {base}")

        matches, text_source = run_file(p, args.subjects)

        if not matches:
            print(" [no course codes detected]")
        else:
            for code, snippet in matches:
                print(f"  {code} — {snippet}" if snippet else f"  {code}")

        print(f"Parsed {base} (subjects: {', '.join(args.subjects)}; text_source={text_source})")


if __name__ == "__main__":
    main()
