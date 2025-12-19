
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 1: Rule/Regex-based "readable" redaction.

Input:  JSONL where each line has at least: {"raw_text": "...", "pii": {...}, ...}
Output: JSONL with an added field "sanitized_text" (and preserved original fields).

Usage:
  python run_baseline1_regex_friendly.py --in feedback_en.jsonl --out pred_baseline1.jsonl
"""

import argparse, json, re
from typing import Dict

DASH = r"[-\u2013\u2014]"  # hyphen, en dash, em dash

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{4}\b")
SOCIAL_RE = re.compile(r"@\w{2,}")
STUDENT_ID_RE = re.compile(r"\b\d{8}\b")
SECTION_RE = re.compile(r"\bSection\s+[A-Za-z0-9]+\b")
TIME_RE = re.compile(rf"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{{1,2}}:\d{{2}}(?:\s*{DASH}\s*\d{{1,2}}:\d{{2}})?\b")
CLASSROOM_RE = re.compile(r"\b(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+)?(?:Hall|Auditorium|Annex|Building|Center|Lab|Room|Lecture|Classroom|Pod|Seminar)\s+[A-Za-z0-9-]*\d[A-Za-z0-9-]*\b")
DORM_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+(?:Dorm|Hall|House|Tower|Court))\b")
DORM_ROOM_RE = re.compile(r"\b(?:dorm\s+room|room)\s+[0-9]{1,2}[A-Za-z]\b", re.IGNORECASE)

SCORE_PATTERNS = [
    re.compile(r"\b(?:got|scored|score(?:d)?|grade(?:d)?|final(?:\s+score)?|midterm(?:\s+score)?|exam(?:\s+score)?|quiz(?:\s+score)?)\s*(?:was|:|=)?\s*(\d{1,3})\b", re.IGNORECASE),
]

def score_bucket(n: int) -> str:
    if n >= 100:
        return "a perfect score"
    if n >= 95:
        return "a score in the high 90s"
    if n >= 90:
        return "a score in the 90s"
    if n >= 80:
        return "a score in the 80s"
    if n >= 70:
        return "a score in the 70s"
    if n >= 60:
        return "a score in the 60s"
    return "a score below 60"

def redact_friendly(text: str) -> str:
    out = text

    # Structured identifiers
    out = EMAIL_RE.sub("an email address", out)
    out = PHONE_RE.sub("a phone number", out)
    out = SOCIAL_RE.sub("a social handle", out)

    # Student ID (preserve label if present)
    out = re.sub(r"\bStudent ID:\s*\d{8}\b", "Student ID: (redacted)", out, flags=re.IGNORECASE)
    out = STUDENT_ID_RE.sub("a student ID", out)

    # Course logistics
    out = SECTION_RE.sub("a course section", out)
    out = TIME_RE.sub("a class time", out)
    out = CLASSROOM_RE.sub("a classroom location", out)

    # Dorm
    out = DORM_RE.sub("a dorm", out)
    out = DORM_ROOM_RE.sub("a dorm room", out)

    # Accommodation sentence-level (simple heuristic)
    out = re.sub(
        r"(?i)\b(?:accommodation request|accommodation note|requesting accommodation|i need an accommodation)\b[^.]*\.",
        "Accommodation request (details omitted).",
        out,
    )

    # Scores (contextual numbers only)
    for pat in SCORE_PATTERNS:
        def _repl(m: re.Match) -> str:
            num = int(m.group(1))
            if num > 100:
                return m.group(0)
            return m.group(0).replace(m.group(1), score_bucket(num))
        out = pat.sub(_repl, out)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL")
    args = ap.parse_args()

    n = 0
    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj.get("raw_text", "")
            obj["sanitized_text"] = redact_friendly(raw)
            obj["baseline"] = "baseline1_regex"
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(f"[OK] Wrote {n} lines to {args.out}")

if __name__ == "__main__":
    main()
