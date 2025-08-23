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


def _normalize_text(s: str) -> str:
    # Convert non-breaking spaces to regular spaces and normalize dashes.
    s = s.replace("\xa0", " ")  # actual NBSP char
    s = s.replace("\u00a0", " ")  # literal backslash-u sequence if present
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _expand_subjects(subjects: Iterable[str]) -> list[str]:
    expanded: set[str] = set()
    for s in subjects:
        key = s.lower()
        if key in SUBJECT_ALIASES:
            expanded.update(SUBJECT_ALIASES[key])
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


# Generic course code pattern (prefix then number), case-insensitive.
# We'll filter prefixes against the subject expansion.
GENERIC_CODE_PAT = re.compile(r"(?i)\b([A-Z]{2,})\s*[-:\s\xa0]?\s*(\d{3}[A-Z]?)\b")


def find_courses_in_text(text: str, allowed_prefixes: Iterable[str]) -> list[tuple[str, str]]:
    allowed = {p.upper() for p in allowed_prefixes}
    results: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = _normalize_text(raw_line)
        for m in GENERIC_CODE_PAT.finditer(line):
            prefix = m.group(1).upper()
            if prefix not in allowed:
                continue
            number = m.group(2).upper()
            code = f"{prefix} {number}"
            # Snippet is text after the match, until the next obvious code on same line
            after = line[m.end() :]
            after = re.split(r"\b[A-Z]{2,}\s*[-:\s\xa0]?\s*\d{3}[A-Z]?\b", after, maxsplit=1)[0]
            snippet = after.strip()
            results.append((code, snippet))
    return results


def run_file(path: Path, subjects: list[str]) -> tuple[list[tuple[str, str]], str]:
    prefixes = _expand_subjects(subjects)
    text = extract_pdf_text(path)
    matches = find_courses_in_text(text, prefixes)
    text_source = "pdf"
    # OCR fallback can be added later if needed; not required for the smoke test.
    return matches, text_source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="transcript-parser")
    parser.add_argument("inputs", nargs="+", help="PDF file(s)")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject labels, e.g. math stat cs")
    parser.add_argument("--out", default=None, help="Optional JSON output path (unused in smoke test)")
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
