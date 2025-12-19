#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation for redaction / privacy-preserving transcription on your synthetic dataset.

We compute:
A) Exact Leak Rate (privacy): whether each gold PII value still appears verbatim in sanitized_text.
B) Utility vs Oracle-Friendly (content retention): token-level F1 between sanitized_text and an oracle-friendly reference
   built by replacing gold PII values in raw_text with human-readable generalizations.
C) Over-redaction / distortion proxies: length ratio vs raw, and (1 - sequence similarity) vs raw.

Input:
  --gold: JSONL with post_id, raw_text, pii (dict)
  --pred: JSONL with post_id, sanitized_text
Output:
  --out_csv: per-type summary CSV
  --out_overall_json (optional): overall metrics JSON

Usage:
  python evaluate_redaction.py --gold feedback_en.jsonl --pred pred_baseline1.jsonl --out_csv summary.csv --out_overall_json overall.json
"""

import argparse, json, re, difflib
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def tokens(s: str):
    return TOKEN_RE.findall(s.lower())

def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum(min(ca[t], cb[t]) for t in (ca.keys() & cb.keys()))
    p = overlap / max(1, len(ta))
    r = overlap / max(1, len(tb))
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def score_bucket(n: str) -> str:
    try:
        x = float(n)
    except Exception:
        return "a score"
    if x < 0 or x > 100:
        return "a score"
    tens = int(x // 10) * 10
    if tens >= 100:
        return "a perfect score"
    return f"in the {tens}s"

def friendly_for_key(key: str, value: str) -> str:
    if key == "email": return "my school email"
    if key == "phone": return "my phone number"
    if key == "social_handle": return "my social handle"
    if key == "student_id": return "my student ID"
    if key == "section": return "the section"
    if key == "class_time": return "the class time"
    if key == "classroom": return "the classroom location"
    if key == "dorm": return "my dorm"
    if key == "dorm_room": return "my dorm room"
    if key == "other_person_name": return "a classmate"
    if key == "self_name": return "me"
    if key in ("exact_score", "other_person_score"): return score_bucket(value)
    if key == "unique_detail": return "a specific personal detail"
    if key == "accommodation_note": return "an accommodation request"
    if key == "instructor_mentioned": return "the instructor"
    return "redacted"

def build_oracle_friendly(raw_text: str, pii: dict) -> str:
    oracle = raw_text
    for k, v in pii.items():
        if v is None:
            continue
        v = str(v)
        if v.strip() == "":
            continue
        oracle = oracle.replace(v, friendly_for_key(k, v))
    return oracle

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_overall_json", default=None)
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred = load_jsonl(args.pred)

    pred_map = {str(x.get("post_id")): x.get("sanitized_text", "") for x in pred}

    by_type = defaultdict(lambda: {"total": 0, "leaked": 0})
    overall_total = 0
    overall_leaked = 0
    util_f1 = []
    len_ratio_raw = []
    change_ratio_raw = []

    for g in gold:
        pid = str(g.get("post_id"))
        raw = g.get("raw_text", "")
        pii = g.get("pii", {}) or {}
        out = pred_map.get(pid, "")

        # B/C
        oracle = build_oracle_friendly(raw, pii)
        util_f1.append(token_f1(out, oracle))
        len_ratio_raw.append(len(out) / max(1, len(raw)))
        change_ratio_raw.append(1.0 - seq_ratio(out, raw))

        # A
        for k, v in pii.items():
            if v is None:
                continue
            v = str(v)
            if v.strip() == "":
                continue
            overall_total += 1
            by_type[k]["total"] += 1
            if v in out:
                overall_leaked += 1
                by_type[k]["leaked"] += 1

    # write per-type + overall rows
    rows = []
    rows.append({
        "pii_type": "OVERALL",
        "total": overall_total,
        "leaked": overall_leaked,
        "leak_rate": (overall_leaked / overall_total) if overall_total else 0.0,
        "avg_token_f1_vs_oracle_friendly": sum(util_f1) / max(1, len(util_f1)),
        "avg_len_ratio_vs_raw": sum(len_ratio_raw) / max(1, len(len_ratio_raw)),
        "avg_change_ratio_vs_raw": sum(change_ratio_raw) / max(1, len(change_ratio_raw)),
    })
    for k, s in sorted(by_type.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        tot, leak = s["total"], s["leaked"]
        rows.append({"pii_type": k, "total": tot, "leaked": leak, "leak_rate": (leak / tot) if tot else 0.0})

    # CSV
    import csv
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.out_overall_json:
        overall = rows[0]
        with open(args.out_overall_json, "w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
