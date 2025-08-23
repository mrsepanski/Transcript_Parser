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

# Generic course code pattern (prefix then number), case-insensitive.
GENERIC_CODE_PAT = re.compile(r"(?i)\b([A-Z]{2,})\s*[-:\s\u00A0\xa0]?\s*(\d{3}[A-Z]?)\b")
SINGLE_TOKEN_PAT = re.compile(r"(?i)^([A-Z]{2,})[-:]?(\d{3}[A-Z]?)$")


def _normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")  # actual NBSP char
    s = s.replace("\u00a0", " ")  # literal backslash-u sequence, if present
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


def _allowed_set(subjects: Iterable[str]) -> set[str]:
    return {p.upper() for p in _expand_subjects(subjects)}


def _extract_text_lines(path: Path) -> list[str]:
    if pdfplumber is None:
        return []
    lines: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                for ln in t.splitlines():
                    ln = _normalize_text(ln)
                    if ln.strip():
                        lines.append(ln)
    return lines


def _scan_lines_for_codes(lines: list[str], allowed: set[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in lines:
        for m in GENERIC_CODE_PAT.finditer(line):
            prefix = m.group(1).upper()
            if prefix not in allowed:
                continue
            number = m.group(2).upper()
            code = f"{prefix} {number}"
            after = line[m.end() :]
            after = re.split(r"\b[A-Z]{2,}\s*[-:\s\u00A0\xa0]?\s*\d{3}[A-Z]?\b", after, maxsplit=1)[0]
            snippet = after.strip()
            out.append((code, snippet))
    return out


def _scan_words_for_codes(path: Path, allowed: set[str]) -> list[tuple[str, str]]:
    if pdfplumber is None:
        return []
    out: list[tuple[str, str]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                words = page.extract_words() or []
            except Exception:
                words = []
            toks = [_normalize_text(w.get("text", "")) for w in words if w.get("text")]
            i = 0
            while i < len(toks):
                tok = toks[i].strip()
                if not tok:
                    i += 1
                    continue
                m_single = SINGLE_TOKEN_PAT.match(tok)
                if m_single:
                    prefix = m_single.group(1).upper()
                    number = m_single.group(2).upper()
                    if prefix in allowed:
                        out.append((f"{prefix} {number}", ""))
                        i += 1
                        continue
                upper_tok = tok.upper()
                if upper_tok in allowed:
                    j = i + 1
                    if j < len(toks) and toks[j] in {"-", ":"}:
                        j += 1
                    if j < len(toks) and re.fullmatch(r"\d{3}[A-Za-z]?", toks[j]):
                        out.append((f"{upper_tok} {toks[j].upper()}", ""))
                        i = j + 1
                        continue
                i += 1
    return out


def _fallback_generic(lines: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in lines:
        for m in GENERIC_CODE_PAT.finditer(line):
            code = f"{m.group(1).upper()} {m.group(2).upper()}"
            after = line[m.end() :]
            after = re.split(r"\b[A-Z]{2,}\s*[-:\s\u00A0\xa0]?\s*\d{3}[A-Z]?\b", after, maxsplit=1)[0]
            snippet = after.strip()
            out.append((code, snippet))
    return out


def _scan_binary_for_codes(path: Path, allowed: set[str]) -> list[tuple[str, str]]:
    # Last ditch: read raw bytes and look for ASCII text occurrences (common in simple PDFs).
    try:
        data = path.read_bytes()
    except Exception:
        return []
    # Decode losslessly; PDF content streams often contain plain ASCII between operators.
    s = data.decode("latin-1", errors="ignore")
    s = _normalize_text(s)
    out: list[tuple[str, str]] = []
    for m in GENERIC_CODE_PAT.finditer(s):
        prefix = m.group(1).upper()
        number = m.group(2).upper()
        if allowed and prefix not in allowed:
            continue
        out.append((f"{prefix} {number}", ""))
        if len(out) >= 5:
            break
    return out


def run_file(path: Path, subjects: list[str]) -> tuple[list[tuple[str, str]], str]:
    allowed = _allowed_set(subjects)
    lines = _extract_text_lines(path)
    matches = _scan_lines_for_codes(lines, allowed)
    source = "pdf"
    if not matches:
        word_matches = _scan_words_for_codes(path, allowed)
        if word_matches:
            matches = word_matches
            source = "pdf_words"
    if not matches and lines:
        generic = _fallback_generic(lines)
        if generic:
            matches = generic
            source = "pdf_generic"
    if not matches:
        bin_matches = _scan_binary_for_codes(path, allowed)
        if bin_matches:
            matches = bin_matches
            source = "pdf_binary"
    return matches, source


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
