# -*- coding: utf-8 -*-
"""
ours_pipeline.py (consolidated)

Features:
- Logs to BOTH console and a log file (default: same as --out but .log)
- Streaming JSONL output (flush_every)
- Resume mode (skip post_ids already in output)
- Select-ID mode: --id_txt <file> where each line is a post_id; only process those ids
  * In select-ID mode, results are written to a patch file then merged back into --out
- Robust per-sample error handling (one bad sample won't crash the run)
- JSON serialization helper for numpy/pandas scalars (int64, float64, etc.)

Run:
  python -m src.ours_pipeline ...
or
  python src/ours_pipeline.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# --- Import shim
try:
    from .kb import CourseKB
    from .detectors import detect_candidates
    from .prompts import build_planner_messages, build_writer_messages
    from .llm import LLMClient, LLMConfig
    from .verify import verify_output
except Exception:  # pragma: no cover
    from kb import CourseKB  # type: ignore
    from detectors import detect_candidates  # type: ignore
    from prompts import build_planner_messages, build_writer_messages  # type: ignore
    from llm import LLMClient, LLMConfig  # type: ignore
    from verify import verify_output  # type: ignore


# -----------------------------
# JSON helpers
# -----------------------------
def _json_default(o: Any) -> Any:
    """Make numpy/pandas scalars JSON-serializable."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    return str(o)


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_file: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("ours")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Avoid duplicate handlers if re-imported
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)

    # Always file log
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(fh)

    # keep external libs quieter unless verbose
    if not verbose:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


# -----------------------------
# IO
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_post_ids_from_jsonl(path: str) -> Set[str]:
    done: Set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("post_id", None)
            if pid is None:
                continue
            done.add(str(pid))
    return done


def read_id_txt(path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.add(s)
    return ids


def _safe_get(rec: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


# -----------------------------
# Prompt fallback (single-stage) for planner failure
# -----------------------------
def build_single_stage_messages(
    raw_text: str,
    course_kb: Dict[str, Any],
    cand_dicts: List[Dict[str, Any]],
    policy_strictness: int,
) -> List[Dict[str, str]]:
    """
    Single-stage instruction: directly produce sanitized_text using KB + candidates.
    NOTE: JSON dumps uses default=_json_default to avoid int64 serialization errors.
    """
    sys_prompt = (
        "You are a privacy-preserving rewriting assistant.\n"
        "Rewrite the student's feedback so that sensitive details are generalized, "
        "while preserving meaning and being human-readable.\n"
        "Use the provided course knowledge to replace exact grades/times/locations with meaningful ranges or descriptions.\n"
        "Return STRICT JSON object only.\n"
        "Schema:\n"
        "{\n"
        '  "sanitized_text": string,\n'
        '  "checks": {"leaks": [string], "notes": [string]}\n'
        "}\n"
        "No markdown, no code fences.\n"
    )

    user = (
        f"policy_strictness={policy_strictness}\n"
        f"raw_text={raw_text}\n\n"
        f"course_kb={json.dumps(course_kb, ensure_ascii=False, default=_json_default)}\n\n"
        f"candidates={json.dumps(cand_dicts, ensure_ascii=False, default=_json_default)}\n"
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


# -----------------------------
# Merge patch back into out
# -----------------------------
def merge_patch_into_out(out_path: Path, patch_path: Path, logger: logging.Logger) -> None:
    """
    Replace lines in out_path whose post_id appears in patch, writing a new out file atomically.
    - If out doesn't exist: rename patch -> out
    - If out exists: stream through out, replacing first occurrence per id, skipping duplicates
    """
    if not patch_path.exists():
        logger.warning(f"Patch file not found: {patch_path}")
        return

    if not out_path.exists():
        patch_path.replace(out_path)
        logger.info(f"Merged patch: out did not exist, renamed {patch_path} -> {out_path}")
        return

    # Load patch lines into memory (small: only failed ids)
    patch_map: Dict[str, str] = {}
    with open(patch_path, "r", encoding="utf-8") as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("post_id", None)
            if pid is None:
                continue
            patch_map[str(pid)] = json.dumps(obj, ensure_ascii=False, default=_json_default)

    if not patch_map:
        logger.warning("Patch file had no usable JSON lines; merge skipped.")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    replaced: Set[str] = set()

    with open(out_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                pid = obj.get("post_id", None)
            except Exception:
                # keep malformed lines as-is
                fout.write(line if line.endswith("\n") else line + "\n")
                continue

            pid_s = str(pid) if pid is not None else None
            if pid_s and pid_s in patch_map:
                if pid_s in replaced:
                    # drop duplicate old lines for same id
                    continue
                fout.write(patch_map[pid_s] + "\n")
                replaced.add(pid_s)
            else:
                fout.write(line if line.endswith("\n") else line + "\n")

        # Append patch ids that were not present in out
        for pid_s, json_line in patch_map.items():
            if pid_s not in replaced:
                fout.write(json_line + "\n")

    tmp_path.replace(out_path)
    patch_path.unlink(missing_ok=True)
    logger.info(f"Merged patch into {out_path}. Replaced {len(replaced)} ids; appended {len(patch_map)-len(replaced)} new ids.")


# -----------------------------
# Per-sample processing
# -----------------------------
def process_one(
    rec: Dict[str, Any],
    kb: CourseKB,
    planner: Optional[LLMClient],
    writer: Optional[LLMClient],
    base_policy_strictness: int,
    max_retries: int,
    save_plan: bool,
    save_debug: bool,
    stage_log: bool,
) -> Dict[str, Any]:
    logger = logging.getLogger("ours")
    t0 = time.monotonic()

    post_id = rec.get("post_id", None)
    raw_text = _safe_get(rec, "raw_text", "text", default="") or ""
    term = _safe_get(rec, "term", default=None)
    course_code = _safe_get(rec, "course_code", default=None)
    course_name = _safe_get(rec, "course_name", default=None)

    try:
        kb_row = kb.retrieve(term=term, course_code=course_code, course_name=course_name)
        course_kb = kb_row.to_prompt_dict() if kb_row else {
            "term": term, "course_code": course_code, "course_name": course_name,
            "score_stats": {"avg": None, "q1": None, "median": None, "q3": None, "min": None, "max": None},
            "assessment_structure": "",
            "extra": {},
        }

        candidates = detect_candidates(raw_text)
        cand_dicts = [c.__dict__ for c in candidates]
        gold_pii = rec.get("pii", None) if isinstance(rec.get("pii", None), dict) else None

        attempts: List[Dict[str, Any]] = []
        sanitized_text = ""

        if stage_log:
            logger.info(f"[{post_id}] start strict={base_policy_strictness} len={len(raw_text)}")

        if planner is None or writer is None:
            raise RuntimeError("planner/writer must be provided")

        strictness = base_policy_strictness
        plan: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries + 1):
            # Stage 1: planner (may fail -> fallback single-stage)
            try:
                plan_msgs = build_planner_messages(raw_text, course_kb, cand_dicts, policy_strictness=strictness)
                if stage_log:
                    logger.info(f"[{post_id}] planner -> sending (strict={strictness})")
                tp0 = time.monotonic()
                plan = planner.chat_json(plan_msgs)
                tp = time.monotonic() - tp0
                if stage_log:
                    logger.info(f"[{post_id}] planner <- ok in {tp:.2f}s")
            except Exception as e:
                logger.warning(f"[{post_id}] planner failed (strict={strictness}): {e} -> fallback single-stage writer")
                # single-stage writer
                write_msgs = build_single_stage_messages(raw_text, course_kb, cand_dicts, strictness)
                tw0 = time.monotonic()
                out_json = writer.chat_json(write_msgs)
                tw = time.monotonic() - tw0
                sanitized_text = (out_json.get("sanitized_text", "") or "").strip()
                ver = verify_output(sanitized_text, cand_dicts, gold_pii=gold_pii)
                attempts.append({
                    "policy_strictness": strictness,
                    "plan": None,
                    "fallback": "single_stage",
                    "verifier_passed": ver.passed,
                    "verifier_findings": [f.__dict__ for f in ver.findings],
                    "timing_s": {"planner": None, "writer": round(tw, 3), "verify": 0.0},
                })
                if ver.passed or attempt == max_retries:
                    break
                strictness = min(2, strictness + 1)
                continue

            # Stage 2: writer
            write_msgs = build_writer_messages(raw_text, course_kb, plan)
            if stage_log:
                logger.info(f"[{post_id}] writer -> sending")
            tw0 = time.monotonic()
            out_json = writer.chat_json(write_msgs)
            tw = time.monotonic() - tw0
            sanitized_text = (out_json.get("sanitized_text", "") or "").strip()
            if stage_log:
                logger.info(f"[{post_id}] writer <- ok in {tw:.2f}s out_len={len(sanitized_text)}")

            ver = verify_output(sanitized_text, cand_dicts, gold_pii=gold_pii)
            attempts.append({
                "policy_strictness": strictness,
                "plan": plan if save_plan else None,
                "writer_checks": out_json.get("checks", None),
                "verifier_passed": ver.passed,
                "verifier_findings": [f.__dict__ for f in ver.findings],
                "timing_s": {"planner": None, "writer": round(tw, 3), "verify": 0.0},
            })

            if stage_log:
                logger.info(f"[{post_id}] verify passed={ver.passed} findings={len(ver.findings)}")

            if ver.passed:
                break
            if attempt < max_retries:
                logger.warning(
                    f"[{post_id}] retry {attempt+1}/{max_retries} "
                    f"(strict {strictness}->{min(2, strictness+1)}) findings={[f.kind for f in ver.findings][:3]}"
                )
            strictness = min(2, strictness + 1)

        final: Dict[str, Any] = {
            "post_id": post_id,
            "sanitized_text": sanitized_text,
            "method": "ours_two_stage_prior",
            "verifier_passed": attempts[-1]["verifier_passed"] if attempts else False,
            "verifier_findings": attempts[-1]["verifier_findings"] if attempts else [],
            "attempts": attempts,
            "meta": {"total_s": round(time.monotonic() - t0, 3)},
        }
        if save_plan and plan is not None:
            final["final_plan"] = plan
        if save_debug:
            final["course_kb"] = course_kb
            final["candidates"] = cand_dicts

        if stage_log:
            logger.info(f"[{post_id}] done in {time.monotonic()-t0:.2f}s")

        return final

    except Exception as e:
        logger.exception(f"[{post_id}] ERROR: {e}")
        return {
            "post_id": post_id,
            "sanitized_text": "",
            "method": "ours_two_stage_prior",
            "error": str(e),
            "meta": {"total_s": round(time.monotonic() - t0, 3)},
        }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL")
    ap.add_argument("--course_kb_csv", required=True, help="Course KB CSV")
    ap.add_argument("--base_url", default=None, help="OpenAI-compatible base_url")
    ap.add_argument("--planner_model", default="gpt-5.2")
    ap.add_argument("--writer_model", default="gpt-5.2")
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--llm_retries", type=int, default=2)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--policy_strictness", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--planner_max_tokens", type=int, default=800)
    ap.add_argument("--writer_max_tokens", type=int, default=900)
    ap.add_argument("--save_plan", action="store_true")
    ap.add_argument("--save_debug", action="store_true")

    # logging
    ap.add_argument("--log_file", default="", help="Log file path (optional). If empty, defaults to <out>.log")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stage_log_first_k", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)

    # output
    ap.add_argument("--flush_every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)

    # select-ID mode
    ap.add_argument("--id_txt", default="", help="Path to txt file; each line is a post_id to reprocess. If set, only process those IDs and merge patch back into --out.")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = args.log_file.strip() if args.log_file.strip() else str(out_path.with_suffix(".log"))
    logger = setup_logger(log_path, args.verbose)
    logger.info(f"Log file: {log_path}")

    # Load input
    records = read_jsonl(args.inp)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    logger.info(f"Loaded {len(records)} records from {args.inp}")

    # ID selection
    id_set: Optional[Set[str]] = None
    if args.id_txt:
        id_set = read_id_txt(args.id_txt)
        logger.info(f"ID selection enabled: loaded {len(id_set)} ids from {args.id_txt}")
        records = [r for r in records if str(r.get("post_id", "")) in id_set]
        logger.info(f"After filtering by ids: {len(records)} records to process")

    kb = CourseKB.load_csv(args.course_kb_csv)
    logger.info(f"Loaded course KB from {args.course_kb_csv}")

    # LLM clients
    planner_cfg = LLMConfig(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=args.planner_model,
        temperature=0.0,
        max_tokens=args.planner_max_tokens,
        timeout_s=args.timeout_s,
        max_retries=args.llm_retries,
    )
    writer_cfg = LLMConfig(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=args.writer_model,
        temperature=0.0,
        max_tokens=args.writer_max_tokens,
        timeout_s=args.timeout_s,
        max_retries=args.llm_retries,
    )
    planner = LLMClient(planner_cfg)
    writer = LLMClient(writer_cfg)

    # Output target: if id_txt set, write patch then merge
    patch_mode = bool(args.id_txt)
    patch_path = out_path.with_suffix(out_path.suffix + ".patch.jsonl") if patch_mode else out_path

    # Resume only makes sense in normal mode
    done_post_ids: Set[str] = set()
    if args.resume and not patch_mode and patch_path.exists():
        done_post_ids = iter_post_ids_from_jsonl(str(patch_path))
        logger.info(f"Resume enabled: found {len(done_post_ids)} existing post_ids in {patch_path}")

    mode = "w" if patch_mode else ("a" if (args.resume and patch_path.exists()) else "w")
    logger.info(f"Writing outputs to {patch_path} (mode={mode}, flush_every={args.flush_every})")

    submitted = 0
    skipped = 0
    start_wall = time.monotonic()

    with open(patch_path, mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {}
            for idx, rec in enumerate(records):
                pid = rec.get("post_id", None)
                if pid is not None and str(pid) in done_post_ids:
                    skipped += 1
                    continue

                stage_log = args.verbose or (submitted < args.stage_log_first_k)
                fut = ex.submit(
                    process_one,
                    rec, kb, planner, writer,
                    args.policy_strictness, args.max_retries,
                    args.save_plan, args.save_debug,
                    stage_log,
                )
                futs[fut] = pid
                submitted += 1

            logger.info(f"Submitted {submitted} tasks (skipped={skipped}) with max_workers={args.max_workers}")

            written = 0
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing"):
                pid = futs[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logger.exception(f"[{pid}] FUTURE ERROR: {e}")
                    result = {"post_id": pid, "sanitized_text": "", "method": "ours_two_stage_prior", "error": str(e)}

                f_out.write(json.dumps(result, ensure_ascii=False, default=_json_default) + "\n")
                written += 1
                if args.flush_every <= 1 or (written % args.flush_every == 0):
                    f_out.flush()

                if args.log_every > 0 and (written % args.log_every == 0):
                    elapsed = time.monotonic() - start_wall
                    rate = written / max(1e-9, elapsed)
                    logger.info(f"Progress: {written}/{len(futs)} done, {rate:.2f} samples/s, skipped={skipped}")

        f_out.flush()

    logger.info(f"[OK] Wrote {written} lines to {patch_path} (skipped={skipped})")

    if patch_mode:
        logger.info("Patch mode: merging patch back into out...")
        merge_patch_into_out(out_path, patch_path, logger)
        logger.info("[OK] Patch merge done.")


if __name__ == "__main__":
    main()
