# generate_course_kb_csv_dmx.py
# pip install -U openai pandas

import os, json, time, argparse, logging, csv
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI

FIELDS: List[str] = [
    "term",
    "course_code",
    "course_name",
    "instructor",
    "location",
    "class_size", 
    "meetings_per_week",
    "assessment_structure",
    "has_midterm",
    "has_final",
    "avg_score",
    "q1", "median", "q3",
    "min_score", "max_score",
    "grade_A_pct", "grade_B_pct", "grade_C_pct", "grade_D_pct", "grade_F_pct",
    "notes",
]

EXAMPLE_ROW: Dict[str, str] = {
    "term": "2025Fall",
    "course_code": "UG114",
    "course_name": "Deep Learning",
    "instructor": "Caikun Xu",
    "location": "Building A, Room 203",
    "class_size": "120",
    "meetings_per_week": "2",
    "assessment_structure": "Project 40%, Midterm 20%, Final 40%",
    "has_midterm": "yes",
    "has_final": "yes",
    "avg_score": "68",
    "q1": "60",
    "median": "70",
    "q3": "78",
    "min_score": "20",
    "max_score": "98",
    "grade_A_pct": "0.18",
    "grade_B_pct": "0.30",
    "grade_C_pct": "0.32",
    "grade_D_pct": "0.12",
    "grade_F_pct": "0.08",
    "notes": "Synthetic entry for a fictional university. Stats are fictional and internally consistent.",
}
# ============================================================

BASE_URL = "https://www.dmxapi.cn/v1"  


# --------------------- logging ---------------------
def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("course_kb_gen")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# --------------------- prompt ---------------------
def build_prompt(fields: List[str], example: Dict[str, str], n_rows: int) -> str:
    example_json = json.dumps(example, ensure_ascii=False, indent=2)
    fields_list = ", ".join(fields)


    k = max(2, n_rows // 10)

    return f"""
You are generating a synthetic course prior-knowledge table for a university course-feedback-wall project.

Return EXACTLY {n_rows} rows as a JSON object with this shape:
{{
  "rows": [{{ ... }}, ...]
}}

Each row MUST contain ALL fields (as strings) in this exact order:
{fields_list}

Hard rules:
- Everything must be fictional. Do NOT use real people, real course codes, or real school names.
- Keep numbers plausible and internally consistent:
  - min <= q1 <= median <= q3 <= max
  - grade_*_pct are decimals in [0,1] and should roughly sum to 1 (±0.03 ok)
- Use natural-looking campus locations like "Building X, Room YYY" (fictional).
- "notes" should be 1-2 neutral sentences.

Diversity requirements (IMPORTANT):
Within these {n_rows} rows, include AT LEAST:
- {k} "very hard" courses: avg_score <= 55 AND grade_A_pct <= 0.08
- {k} "very easy" courses: avg_score >= 85 AND grade_A_pct >= 0.50
- {k} "very small" classes: class_size <= 15
- {k} "very large" classes: class_size >= 250
These categories can overlap, but make sure the set covers all of them.

Example row (follow style, but create new fictional courses):
{example_json}

ONLY output valid JSON. No markdown. No extra text.
""".strip()


# --------------------- validation ---------------------
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def validate_rows(rows: List[Dict[str, Any]], logger: logging.Logger) -> None:
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("rows must be a non-empty list")

    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"row {i} is not an object")

        missing = [f for f in FIELDS if f not in r]
        extra = [k for k in r.keys() if k not in set(FIELDS)]
        if missing:
            raise ValueError(f"row {i} missing fields: {missing}")
        if extra:
            raise ValueError(f"row {i} has extra fields: {extra}")

        for f in FIELDS:
            r[f] = "" if r[f] is None else str(r[f])

        mn = _to_float(r["min_score"]); q1 = _to_float(r["q1"]); md = _to_float(r["median"])
        q3 = _to_float(r["q3"]); mx = _to_float(r["max_score"])
        if None not in (mn, q1, md, q3, mx):
            if not (mn <= q1 <= md <= q3 <= mx):
                logger.warning(f"Row {i} quartiles not monotonic: min/q1/med/q3/max={mn,q1,md,q3,mx}")

        a = _to_float(r["grade_A_pct"]); b = _to_float(r["grade_B_pct"]); c = _to_float(r["grade_C_pct"])
        d = _to_float(r["grade_D_pct"]); f = _to_float(r["grade_F_pct"])
        if None not in (a,b,c,d,f):
            s = a+b+c+d+f
            if not (0.97 <= s <= 1.03):
                logger.warning(f"Row {i} grade pct sum={s:.3f} (expected ~1.0)")


# --------------------- LLM call ---------------------
def chat_json(client: OpenAI, model: str, prompt: str, logger: logging.Logger, max_retries: int = 4) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.25,
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.25,
                )

            text = resp.choices[0].message.content or ""
            data = json.loads(text)
            if "rows" not in data:
                raise ValueError("Missing top-level key 'rows'")
            validate_rows(data["rows"], logger)
            return data

        except Exception as e:
            last_err = e
            logger.warning(f"[Retry {attempt}/{max_retries}] {type(e).__name__}: {e}")
            prompt = prompt + "\n\nReminder: Output ONLY a JSON object with top-level key 'rows'."
            time.sleep(0.5)

    raise RuntimeError(f"Failed to get valid JSON after {max_retries} tries: {last_err}")


# --------------------- streaming CSV write ---------------------
def append_rows_to_csv(out_csv: str, rows: List[Dict[str, str]], write_header: bool) -> None:
    with open(out_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})


def generate_csv(
    out_csv: str,
    total_rows: int,
    batch_size: int,
    model: str,
    log_level: str,
    sleep_s: float,
):
    logger = setup_logger(log_level)

    api_key = os.getenv("DMX_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DMX_API_KEY (or OPENAI_API_KEY) in your environment.")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    logger.info(f"BASE_URL={BASE_URL}")
    logger.info(f"MODEL={model}")
    logger.info(f"OUT={out_csv}")
    logger.info(f"TOTAL_ROWS={total_rows}, BATCH_SIZE={batch_size}")

    if os.path.exists(out_csv):
        logger.warning(f"{out_csv} exists -> overwriting")
        os.remove(out_csv)

    remaining = total_rows
    wrote_header = False
    produced = 0
    batch_id = 0

    while remaining > 0:
        batch_id += 1
        n = min(batch_size, remaining)
        logger.info(f"[Batch {batch_id}] requesting {n} rows... (remaining={remaining})")

        prompt = build_prompt(FIELDS, EXAMPLE_ROW, n)
        data = chat_json(client, model=model, prompt=prompt, logger=logger)

        batch_rows = data["rows"]
        # 边跑边写出
        append_rows_to_csv(out_csv, batch_rows, write_header=not wrote_header)
        wrote_header = True

        produced += len(batch_rows)
        remaining -= n
        logger.info(f"[Batch {batch_id}] wrote {len(batch_rows)} rows -> produced={produced}, remaining={remaining}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    logger.info(f"[DONE] Wrote {produced} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic course KB CSV via DMX OpenAI-compatible API.")
    parser.add_argument("--out", default="course_kb.csv", help="output csv path")
    parser.add_argument("--total", type=int, default=60, help="total rows to generate")
    parser.add_argument("--batch", type=int, default=10, help="rows per request")
    parser.add_argument("--model", default=os.getenv("DMX_MODEL", "gpt-4o-mini"), help="model name")
    parser.add_argument("--log", default="INFO", help="log level: DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between batches")
    args = parser.parse_args()

    generate_csv(
        out_csv=args.out,
        total_rows=args.total,
        batch_size=args.batch,
        model=args.model,
        log_level=args.log,
        sleep_s=args.sleep,
    )


if __name__ == "__main__":
    main()
