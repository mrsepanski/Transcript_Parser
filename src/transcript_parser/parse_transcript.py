#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections.abc import Iterable

# Lightweight normalization for unicode dashes etc.
DASHES = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus
}


def normalize_text(s: str) -> str:
    for k, v in DASHES.items():
        s = s.replace(k, v)
    # Normalize weird spaces
    s = s.replace("\xa0", " ")
    return s


def expand_subjects(subject_tokens: Iterable[str]) -> list[str]:
    """
    Map high-level subject tokens to concrete course prefixes.
    e.g. 'math' -> ['MATH','MAT','MTH','MA']
    Unknown tokens are treated as explicit prefixes (uppercased).
    """
    aliases = {
        "math": ["MATH", "MAT", "MTH", "MA", "MATG"],
        "stat": ["STAT", "STA", "STT"],
        "cs": ["CS", "CSC", "CPSC", "COSC"],
        "physics": ["PHY", "PHYS"],
        "chem": ["CHEM", "CHM"],
        "bio": ["BIO", "BIOL"],
        # Feel free to add more aliases here as needed.
    }
    out = []
    for tok in subject_tokens:
        key = tok.strip().lower()
        out.extend(aliases.get(key, [tok.strip().upper()]))
    # de-duplicate while preserving order
    seen = set()
    out_uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            out_uniq.append(p)
    return out_uniq


def build_course_regex(prefixes: list[str]) -> re.Pattern:
    # Sort by length so longer prefixes match first (e.g., MATH before MA)
    prefixes_sorted = sorted(prefixes, key=len, reverse=True)
    pat = r"\b(" + "|".join(map(re.escape, prefixes_sorted)) + r")\s*[-:]?\s*([0-9]{3,4}[A-Z]?)\b"
    return re.compile(pat, flags=re.IGNORECASE)


def extract_pdf_text(path: str, max_pages: int | None = None) -> str:
    try:
        import pdfplumber
    except Exception:
        return ""
    try:
        txt_parts = []
        with pdfplumber.open(path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for pg in pages:
                try:
                    t = pg.extract_text() or ""
                except Exception:
                    t = ""
                txt_parts.append(t)
        return normalize_text("\n".join(txt_parts))
    except Exception:
        return ""


def ocr_pdf_with_paddle(path: str, dpi: int = 300, max_pages: int = 2) -> str:
    """
    Convert first few pages of the PDF to images and run PaddleOCR.
    Only called if PDF text looks insufficient.
    """
    try:
        from paddleocr import PaddleOCR
        from pdf2image import convert_from_path
    except Exception as e:
        print(f"[warn] OCR unavailable or failed to import: {e}", file=sys.stderr)
        return ""

    try:
        images = convert_from_path(path, dpi=dpi)
    except Exception as e:
        print(f"[warn] pdf2image failed: {e}", file=sys.stderr)
        return ""

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    except Exception as e:
        print(f"[warn] PaddleOCR init failed: {e}", file=sys.stderr)
        return ""

    text_chunks = []
    for img in images[:max_pages]:
        try:
            result = ocr.ocr(img, cls=True)
            # result is list of [ [box, (text, conf)], ... ] per image
            if result:
                lines = []
                for line in result:
                    if not line:
                        continue
                    # handle both old/new structures
                    if isinstance(line, list) and len(line) > 0 and isinstance(line[0], list):
                        # some versions return a nested structure
                        for seg in line:
                            if len(seg) >= 2 and isinstance(seg[1], (tuple, list)):
                                lines.append(str(seg[1][0]))
                    else:
                        # typical structure
                        for seg in line:
                            if len(seg) >= 2 and isinstance(seg[1], (tuple, list)):
                                lines.append(str(seg[1][0]))
                text_chunks.append("\n".join(lines))
        except Exception as e:
            print(f"[warn] PaddleOCR page error: {e}", file=sys.stderr)
            continue
    return normalize_text("\n".join(text_chunks))


def harvest_course_lines(text: str, pat: re.Pattern, limit: int = 200) -> list[tuple[str, str, str]]:
    """
    Return a list of (prefix, number, line_text) tuples for lines matching the course pattern.
    `limit` just prevents pathological blowups.
    """
    lines = []
    if not text:
        return lines
    # Work on the original text for nicer output lines; use an uppercased copy for regex search
    text_norm = normalize_text(text)
    text_upper = text_norm.upper()
    for m in pat.finditer(text_upper):
        start, end = m.span()
        # get the original case substring for the line
        ls = text_norm.rfind("\n", 0, start) + 1
        if ls < 0:
            ls = 0
        le = text_norm.find("\n", end)
        if le == -1:
            le = len(text_norm)
        line = " ".join(text_norm[ls:le].split())
        prefix = m.group(1).upper()
        number = m.group(2).upper()
        lines.append((prefix, number, line))
        if len(lines) >= limit:
            break
    return lines


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="PDF file(s) to parse")
    ap.add_argument("--subjects", nargs="+", required=True, help="Subjects (e.g., math stat)")
    ap.add_argument("--out", default=None, help="Optional JSON output file")
    ap.add_argument("--prefer-ocr", action="store_true", help="Force OCR even if PDF text looks fine")
    ap.add_argument("--ocr-dpi", type=int, default=300)
    ap.add_argument("--max-pages", type=int, default=None, help="Max pages to read from PDF for text mode")
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    expanded_prefixes = expand_subjects(args.subjects)
    pat = build_course_regex(expanded_prefixes)

    for inp in args.inputs:
        base = os.path.basename(inp)
        print(f"Results for {base}")

        # Step 1: PDF text
        pdf_text = extract_pdf_text(inp, max_pages=args.max_pages)
        pdf_hits = harvest_course_lines(pdf_text, pat)

        use_ocr = False
        if args.prefer_ocr:
            use_ocr = True
        else:
            # Only trigger OCR if text is clearly insufficient
            # Heuristics: fewer than 2 course hits OR total text very short
            if len(pdf_hits) < 2 or (pdf_text and len(pdf_text) < 400):
                use_ocr = True

        ocr_hits = []
        text_source = "pdf"
        if use_ocr:
            ocr_text = ocr_pdf_with_paddle(inp, dpi=args.ocr_dpi, max_pages=2)
            if ocr_text:
                ocr_hits = harvest_course_lines(ocr_text, pat)
                # If OCR finds more courses than pdf_text, prefer it
                if len(ocr_hits) > len(pdf_hits):
                    text_source = "ocr"
                else:
                    # keep pdf results if they were already better
                    use_ocr = False

        hits = ocr_hits if text_source == "ocr" else pdf_hits
        if hits:
            # Print a concise course summary (code + a trimmed title fragment)
            seen = set()
            for prefix, number, line in hits:
                code = f"{prefix} {number}"
                if code in seen:
                    continue
                seen.add(code)
                # Try to extract a short title fragment to the right of the code
                # (safe fallback to the entire line if parsing is hard)
                try:
                    # find the code within the line and grab up to next 60 chars
                    i = line.upper().find(code)
                    frag = line[i + len(code) :].strip()
                    frag = re.sub(r"\s{2,}", " ", frag)
                    frag = frag[:80].strip(" -:")
                    if frag:
                        print(f"  {code} — {frag}")
                    else:
                        print(f"  {code}")
                except Exception:
                    print(f"  {code}")
        else:
            print(" [no course codes detected]")

        ", ".join(expanded_prefixes)
        print(f"Parsed {base} (subjects: {', '.join(args.subjects)}; text_source={text_source})")

        if args.out:
            data = {
                "file": base,
                "subjects": args.subjects,
                "expanded_prefixes": expanded_prefixes,
                "text_source": text_source,
                "courses": [{"prefix": p, "number": n, "line": line} for p, n, line in hits],
            }
            try:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[warn] could not write JSON: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
