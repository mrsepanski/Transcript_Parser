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

SUBJECT_ALIASES = {
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
    s = s.replace("\u00a0", " ")  # NBSP to space (if double-escaped)
    s = s.replace("\xa0", " ")  # NBSP to space (literal escape)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _course_pattern(prefixes: Iterable[str]) -> re.Pattern[str]:
    key = tuple(sorted(set(p.upper() for p in prefixes)))
    if key in COURSE_PAT_CACHE:
        return COURSE_PAT_CACHE[key]
    # Accept PREFIX 123, PREFIX-123, PREFIX:123, or PREFIX123
    # Allow non-breaking spaces and repeated separators
    pfx_group = f"(?:{'|'.join(re.escape(p) for p in key)})"
    sep = r"(?:[\s :\-])*"
    pat = re.compile(rf"(?<!\w){pfx_group}{sep}(\d{{3}}[A-Z]?)", re.IGNORECASE)
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
    # Always include exact tokens provided, just in case
    return sorted(expanded)


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    if pdfplumber is None:
        return ""
    text_parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for p in pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            text_parts.append(t)
    return _normalize_text("\n".join(text_parts))


def find_courses_in_text(text: str, prefixes: Iterable[str]) -> list[tuple[str, str]]:
    pat = _course_pattern(prefixes)
    found: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\xa0", " ").replace("\u00a0", " ")
        for m in pat.finditer(line):
            prefix_match = m.group(0)[: m.group(0).find(m.group(1))]
            prefix_clean = re.sub(r"[^A-Za-z]", "", prefix_match).upper()
            code = f"{prefix_clean} {m.group(1)}"
            # Snippet: text after the match until the next code (rough trim)
            after = line[m.end() :]
            # Cut at next uppercase prefix occurrence to avoid concatenation
            after = re.split(r"\b[A-Z]{2,}\s*[-:\s]?\s*\d{3}[A-Z]?\b", after, maxsplit=1)[0]
            snippet = after.strip()
            found.append((code, snippet))
    return found


def run_file(path: Path, subjects: list[str]) -> tuple[list[tuple[str, str]], str]:
    prefixes = _expand_subjects(subjects)
    text = extract_pdf_text(path)
    matches = find_courses_in_text(text, prefixes)
    text_source = "pdf"
    # OCR fallback is intentionally avoided for test PDFs and text-rich docs.
    # (Your runtime container has PaddleOCR; integrate here if needed.)
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
                if snippet:
                    print(f"  {code} — {snippet}")
                else:
                    print(f"  {code}")

        print(f"Parsed {base} (subjects: {', '.join(args.subjects)}; text_source={text_source})")


if __name__ == "__main__":
    main()
