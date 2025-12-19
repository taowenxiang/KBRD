#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 3: Prompted LLM "readable redaction" (no prior-knowledge / no retrieval)

Upgrades:
- Stream writing: finish one -> write one JSONL line immediately (flush).
- write_mode: overwrite / append / resume
  - overwrite: truncate and re-generate from scratch
  - append: append to existing file (may duplicate)
  - resume: read existing output, skip post_id already done, append remaining
- Progress logs: show processed / total / ok / fail / skip and speed

Example:
  python baseline3.py --in ../data/feedback_en.jsonl --out ../evaluation/pred_baseline3.jsonl \
    --model gpt-5.2 --base_url https://www.dmxapi.cn/v1 --max_workers 8 --write_mode resume

Env:
  DMX_API_KEY, DMX_BASE_URL (preferred)
  OPENAI_API_KEY, OPENAI_BASE_URL (fallback)
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, Tuple, Optional, Iterable

# ---------------- Prompts ----------------
SYSTEM_PROMPT = (
    "You are a privacy-preserving rewriting engine for course feedback. "
    "Rewrite text to remove personal data and identifying details, while keeping the meaning and usefulness. "
    "Use natural language generalizations (readable). Do NOT use bracket placeholders like [EMAIL]. "
    "Minimize edits outside sensitive spans."
)

USER_TEMPLATE = """Redact and generalize sensitive information in the text below.

Rules (examples, not exhaustive):
- Email addresses -> "my school email" or "an email address"
- Phone numbers -> "my phone number"
- Student IDs -> "my student ID"
- Social handles -> "my social handle"
- People names -> "a classmate" / "a teammate" (keep role if present)
- Exact scores/grades -> coarse bucket like "in the 90s", "in the 80s", "below 60" (do NOT keep exact numbers)
- Class time -> "the class time" (no exact day/time)
- Classroom location -> "the classroom location" (no exact room/building)
- Dorm / dorm room -> "my dorm" / "my dorm room"
- Unique identifying details -> generalize to non-identifying description
- Accommodation requests -> "an accommodation request (details omitted)"

Hard constraints:
- Do NOT output bracket placeholders like [EMAIL], [PHONE], etc.
- Do NOT invent new facts.
- Return ONLY a JSON object with key "sanitized_text".

Text:
\"\"\"{raw_text}\"\"\"
"""

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# ---------------- Helpers ----------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, log_file: Optional[str] = None):
    line = f"{now_ts()} | {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def safe_post_id(obj: Dict[str, Any], fallback: int) -> str:
    pid = obj.get("post_id", fallback)
    return str(pid)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_done_post_ids(out_path: str, log_file: Optional[str] = None) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    bad = 0
    total = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                pid = obj.get("post_id", None)
                if pid is not None:
                    done.add(str(pid))
            except Exception:
                bad += 1
    if total > 0:
        log(f"[resume] scanned existing out_jsonl: lines={total}, parsed_post_ids={len(done)}, bad_lines={bad}", log_file)
    return done

def _score_bucketize_in_text(text: str) -> str:
    patterns = [
        re.compile(r"\b(got|scored|ended with|ended at|finished with|grade(?:d)?)\s+(\d{1,3})\b", re.IGNORECASE),
    ]
    def bucket(n: int) -> str:
        if n >= 100: return "a perfect score"
        if n >= 90:  return "in the 90s"
        if n >= 80:  return "in the 80s"
        if n >= 70:  return "in the 70s"
        if n >= 60:  return "in the 60s"
        return "below 60"

    out = text
    for pat in patterns:
        def repl(m):
            try:
                n = int(m.group(2))
            except Exception:
                return m.group(0)
            return f"{m.group(1)} {bucket(n)}"
        out = pat.sub(repl, out)
    return out

def call_llm(raw_text: str, client: Any, model: str, temperature: float, max_tokens: int) -> str:
    user_prompt = USER_TEMPLATE.format(raw_text=raw_text)

    # Prefer responses API if available, else fallback to chat.completions
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content

    m = JSON_RE.search(text.strip())
    if not m:
        raise ValueError(f"Model did not return a JSON object. Got: {text[:200]}...")
    obj = json.loads(m.group(0))
    out = obj.get("sanitized_text", "")
    if not isinstance(out, str) or not out.strip():
        raise ValueError(f"JSON missing sanitized_text. Got keys: {list(obj.keys())}")
    return out.strip()

def looks_like_quota_exhausted(err: Exception) -> bool:
    s = str(err).lower()
    # 覆盖你这次的: "token amount has been exhausted", "remainquota"
    return ("remainquota" in s) or ("token amount has been exhausted" in s) or ("quota" in s and "exhaust" in s) or ("error code: 401" in s)

def process_one(idx: int, obj: Dict[str, Any], args, client: Any) -> Tuple[int, Dict[str, Any], bool]:
    """
    Returns: (idx, out_obj, fatal_stop)
    fatal_stop=True means quota/auth exhausted: best to stop submitting further tasks.
    """
    post_id = safe_post_id(obj, idx)
    raw_text = obj.get("raw_text", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    if args.dry_run:
        sanitized = raw_text
        sanitized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "my school email", sanitized)
        sanitized = re.sub(r"\+?\d{1,3}-\d{3}-\d{4}\b", "my phone number", sanitized)
        sanitized = re.sub(r"@\w{2,}\b", "my social handle", sanitized)
        sanitized = re.sub(r"\b\d{8}\b", "my student ID", sanitized)
        sanitized = _score_bucketize_in_text(sanitized)
        return idx, {
            "post_id": post_id,
            "sanitized_text": sanitized,
            "method": "baseline3_llm_friendly_dryrun",
        }, False

    last_err = None
    for attempt in range(args.retries):
        try:
            sanitized = call_llm(
                raw_text=raw_text,
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if args.post_bucketize_scores:
                sanitized = _score_bucketize_in_text(sanitized)

            return idx, {
                "post_id": post_id,
                "sanitized_text": sanitized,
                "method": "baseline3_llm_friendly",
            }, False

        except Exception as e:
            last_err = e
            if looks_like_quota_exhausted(e):
                # fatal: stop quickly and keep partial outputs
                return idx, {
                    "post_id": post_id,
                    "sanitized_text": raw_text if args.fail_open_keep_raw else "",
                    "method": "baseline3_llm_friendly",
                    "error": f"FATAL_QUOTA_EXHAUSTED: {str(e)}",
                }, True
            time.sleep(min(2 ** attempt, 8))

    # non-fatal failure: return an error record instead of crashing whole run
    return idx, {
        "post_id": post_id,
        "sanitized_text": raw_text if args.fail_open_keep_raw else "",
        "method": "baseline3_llm_friendly",
        "error": f"FAILED_AFTER_RETRIES: {str(last_err)}",
    }, False

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL")
    ap.add_argument("--model", default="gpt-5.2", help="LLM model name")

    ap.add_argument("--base_url", default=os.getenv("DMX_BASE_URL", os.getenv("OPENAI_BASE_URL", "")),
                    help="OpenAI-compatible base URL (env: DMX_BASE_URL / OPENAI_BASE_URL)")
    ap.add_argument("--api_key", default=os.getenv("DMX_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                    help="API key (env: DMX_API_KEY / OPENAI_API_KEY)")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512, help="Max output tokens")
    ap.add_argument("--max_workers", type=int, default=8, help="Concurrency")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true", help="No API calls; use heuristic stub output")
    ap.add_argument("--post_bucketize_scores", action="store_true", help="Optional post-pass to coarsen obvious score patterns")

    ap.add_argument("--write_mode", choices=["overwrite", "append", "resume"], default="overwrite",
                    help="overwrite: truncate file; append: always append; resume: skip post_ids already in out then append")
    ap.add_argument("--log_every", type=int, default=20, help="Log progress every N completed items")
    ap.add_argument("--log_file", default="", help="Optional log file path")
    ap.add_argument("--max_items", type=int, default=0, help="For quick test: only process first N items (0=all)")

    ap.add_argument("--fail_open_keep_raw", action="store_true",
                    help="If API fails, keep raw_text as sanitized_text (so you can still inspect; but leak rate will be higher). "
                         "If not set, failed items output sanitized_text=''")

    args = ap.parse_args()
    log_file = args.log_file.strip() or None

    ensure_parent_dir(args.out)

    # load input
    data = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if args.max_items and len(data) >= args.max_items:
                break

    total_in = len(data)

    # resume: build done set
    done_ids = set()
    if args.write_mode == "resume":
        done_ids = load_done_post_ids(args.out, log_file)

    # build todo list
    todo = []
    skipped = 0
    for idx, obj in enumerate(data):
        pid = safe_post_id(obj, idx)
        if args.write_mode == "resume" and pid in done_ids:
            skipped += 1
            continue
        todo.append((idx, obj))

    total_todo = len(todo)

    # open output file
    if args.write_mode == "overwrite":
        out_mode = "w"
    else:
        out_mode = "a"

    # init client
    client = None
    if not args.dry_run:
        if not args.api_key:
            raise SystemExit("Missing API key. Set DMX_API_KEY/OPENAI_API_KEY or pass --api_key.")
        try:
            from openai import OpenAI
        except Exception as e:
            raise SystemExit("Missing dependency 'openai'. Install with: pip install -U openai") from e

        client_kwargs = {"api_key": args.api_key}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        client = OpenAI(**client_kwargs)

    log(f"[start] in={args.inp}  out={args.out}  mode={args.write_mode}  out_open={out_mode}", log_file)
    log(f"[start] total_in={total_in}  skipped={skipped}  todo={total_todo}  workers={args.max_workers}  model={args.model}  dry_run={args.dry_run}", log_file)

    t0 = time.time()
    completed = 0
    ok = 0
    fail = 0
    fatal_seen = False

    # streaming write
    with open(args.out, out_mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            # backpressure: keep at most max_workers futures inflight
            it = iter(todo)
            inflight = {}

            def submit_next() -> bool:
                try:
                    idx, obj = next(it)
                except StopIteration:
                    return False
                fut = ex.submit(process_one, idx, obj, args, client)
                inflight[fut] = idx
                return True

            # fill initial
            for _ in range(max(1, args.max_workers)):
                if not submit_next():
                    break

            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    idx = inflight.pop(fut)
                    try:
                        _, out_obj, fatal_stop = fut.result()
                    except Exception as e:
                        # should be rare: capture unexpected exceptions
                        pid = safe_post_id(data[idx], idx) if idx < len(data) else str(idx)
                        out_obj = {
                            "post_id": pid,
                            "sanitized_text": data[idx].get("raw_text", "") if (args.fail_open_keep_raw and idx < len(data)) else "",
                            "method": "baseline3_llm_friendly",
                            "error": f"UNEXPECTED_EXCEPTION: {repr(e)}",
                        }
                        fatal_stop = False

                    # write immediately
                    f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    f_out.flush()

                    completed += 1
                    if "error" in out_obj:
                        fail += 1
                    else:
                        ok += 1

                    if fatal_stop and not fatal_seen:
                        fatal_seen = True
                        log(f"[fatal] quota/auth exhausted detected. Will stop submitting new tasks, keep partial outputs.", log_file)

                    # progress log
                    if (completed % args.log_every) == 0 or completed == total_todo:
                        dt = max(1e-6, time.time() - t0)
                        speed = completed / dt
                        log(f"[prog] {completed}/{total_todo} done | ok={ok} fail={fail} skip={skipped} | {speed:.2f} items/s", log_file)

                    # submit more unless fatal
                    if not fatal_seen:
                        submit_next()

                if fatal_seen:
                    # best effort: cancel pending futures (not yet running)
                    for fut in list(inflight.keys()):
                        fut.cancel()
                    inflight.clear()
                    break

    dt = max(1e-6, time.time() - t0)
    log(f"[done] wrote_output={args.out} | processed={completed}/{total_todo} ok={ok} fail={fail} skip={skipped} | elapsed={dt:.1f}s", log_file)


if __name__ == "__main__":
    main()
