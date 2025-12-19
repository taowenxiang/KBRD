# synth_course_feedback_wall_en.py
# English synthetic dataset for a "course feedback wall"
import os
import json
import time
import random
import argparse
import logging
import re

from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

DEFAULT_BASE_URL = "https://www.dmxapi.cn/v1"
logger = logging.getLogger("course_feedback_synth")

ISSUE_LABELS = [
    "difficulty_high",
    "grading_harsh",
    "teaching_unclear",
    "workload_heavy",
    "materials_poor",
    "pace_too_fast",
    "exam_too_hard",
    "assignment_unclear",
    "fairness_issue",
    "ta_support_poor",
    "schedule_conflict",
    "content_outdated",
    "slides_incomplete",
]

# 统一的 pii 字段集合（写CSV也用这个）
PII_FIELDS = [
    "self_name",
    "other_person_name",
    "instructor_mentioned",
    "exact_score",          # self score (digits only)
    "other_person_score",   # other person's score (digits only)
    "student_id",
    "email",
    "phone",
    "social_handle",
    "section",
    "class_time",
    "classroom",
    "dorm",
    "dorm_room",
    "unique_detail",
    "accommodation_note",
]

# 每个 privacy profile 的 must/avoid 规则
# 说明：为了可控分桶/可评测，大多数 profile 互斥；组合风险通过专门的 combo_* profile 来引入
PROFILE_RULES: Dict[str, Dict[str, Any]] = {
    "none": {
        "must": [],
        "avoid": [
            "other_person_name", "exact_score", "other_person_score",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },

    # 基础泄露
    "score_only": {
        "must": ["exact_score"],
        "avoid": [
            "other_person_name", "other_person_score",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "name_only": {
        "must": ["other_person_name"],
        "avoid": [
            "exact_score", "other_person_score",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "both_score_and_name": {
        "must": ["exact_score", "other_person_name"],
        "avoid": [
            "other_person_score",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },

    # 直接标识符
    "student_id": {
        "must": ["student_id"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "email": {
        "must": ["email"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "phone": {
        "must": ["phone"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "social_handle": {
        "must": ["social_handle"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },

    # 间接/准标识符（location/time/section）
    "section_time_place": {
        "must": ["section", "class_time", "classroom"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone", "social_handle",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },
    "dorm_room": {
        "must": ["dorm", "dorm_room"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "unique_detail", "accommodation_note"
        ]
    },

    # 你重点提到的：唯一化细节（quasi-identifier / fingerprint）
    "unique_event": {
        "must": ["unique_detail"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "accommodation_note"
        ]
    },

    # 敏感情境信息
    "accommodation": {
        "must": ["accommodation_note"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail"
        ]
    },

    # ====== 组合风险 profiles（新增）======

    # 组合泄露：自己分数 + 同学名字 + 同学分数
    "combo_self_and_other_scores": {
        "must": ["exact_score", "other_person_name", "other_person_score"],
        "avoid": [
            "student_id", "email", "phone", "social_handle",
            "section", "class_time", "classroom",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },

    # 更强组合：自己分数 + 同学名字 + 同学分数 + section/time/classroom
    "combo_scores_plus_location": {
        "must": ["exact_score", "other_person_name", "other_person_score", "section", "class_time", "classroom"],
        "avoid": [
            "student_id", "email", "phone", "social_handle",
            "dorm", "dorm_room",
            "unique_detail", "accommodation_note"
        ]
    },

    # 更“真实但更危险”：unique_detail + section/time/classroom（熟人极易对号入座）
    "combo_unique_plus_location": {
        "must": ["unique_detail", "section", "class_time", "classroom"],
        "avoid": [
            "exact_score", "other_person_score", "other_person_name",
            "student_id", "email", "phone", "social_handle",
            "dorm", "dorm_room",
            "accommodation_note"
        ]
    },
}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """配置日志：控制台 + 可选文件"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


def load_courses(course_csv: Optional[str]) -> List[Dict[str, str]]:
    """读取课程KB CSV：至少 term/course_code/course_name/instructor"""
    if not course_csv or not os.path.exists(course_csv):
        return [
            {
                "term": "2025F",
                "course_code": "DL101",
                "course_name": "Deep Learning",
                "instructor": "Prof. Chen (synthetic)",
                "assessment_structure": "HW 40%, Midterm 20%, Final 40%",
                "avg_score": "66",
                "notes": "Fictional course entry."
            },
            {
                "term": "2025F",
                "course_code": "ML201",
                "course_name": "Machine Learning",
                "instructor": "Prof. Wang (synthetic)",
                "assessment_structure": "HW 30%, Project 30%, Final 40%",
                "avg_score": "70",
                "notes": "Fictional course entry."
            },
        ]

    df = pd.read_csv(course_csv)
    required = ["term", "course_code", "course_name", "instructor"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"course_csv missing required column: {c}. Need at least {required}")

    courses: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        courses.append({
            "term": str(r.get("term", "")),
            "course_code": str(r.get("course_code", "")),
            "course_name": str(r.get("course_name", "")),
            "instructor": str(r.get("instructor", "")),
            "assessment_structure": str(r.get("assessment_structure", "")),
            "avg_score": str(r.get("avg_score", "")),
            "notes": str(r.get("notes", "")),
        })
    return courses


def largest_remainder_counts(n: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """
    用 largest remainder method 把比例分配成整数条数，总和严格等于 n。
    若 ratios 总和 > 1，会归一化（并打日志）。
    """
    clean = {k: max(0.0, float(v)) for k, v in ratios.items()}
    total = sum(clean.values())
    if total <= 0:
        out = {k: 0 for k in ratios.keys()}
        out["none"] = n
        return out

    if total > 1.0:
        logger.warning(f"Sum of ratios={total:.3f} > 1.0. Normalizing ratios to sum to 1.")
        clean = {k: v / total for k, v in clean.items()}

    expected = {k: n * v for k, v in clean.items()}
    counts = {k: int(expected[k]) for k in expected.keys()}
    remainder = {k: expected[k] - counts[k] for k in expected.keys()}

    used = sum(counts.values())
    left = n - used

    for k, _ in sorted(remainder.items(), key=lambda x: x[1], reverse=True):
        if left <= 0:
            break
        counts[k] += 1
        left -= 1

    if sum(counts.values()) != n:
        counts["none"] = counts.get("none", 0) + (n - sum(counts.values()))
    return counts


def build_prompt(course_pool: List[Dict[str, str]], n_rows: int, seed: int, tone: str, profile: str) -> str:
    """
    让 LLM 按 profile 生成，并把对应 pii 字段填出来。
    must 字段强制出现且必须能在 raw_text 中找到。
    """
    if profile not in PROFILE_RULES:
        raise ValueError(f"Unknown profile: {profile}")

    must_fields = PROFILE_RULES[profile]["must"]
    avoid_fields = PROFILE_RULES[profile]["avoid"]

    pool_json = json.dumps(course_pool, ensure_ascii=False, indent=2)

    must_text = (
        "\n".join([f"- MUST include {f} and fill pii.{f} with the exact string used in raw_text." for f in must_fields])
        if must_fields else "- No required PII fields."
    )
    avoid_text = (
        "\n".join([f"- DO NOT include {f} (keep pii.{f} empty)." for f in avoid_fields])
        if avoid_fields else "- No avoid constraints."
    )

    # 对 unique_detail 给更强约束：必须足够“指纹化”
    unique_hint = ""
    if profile in ("unique_event", "combo_unique_plus_location"):
        unique_hint = """
Unique-detail requirement:
- raw_text MUST include a very specific, unusual, identifying detail about an exam/assignment incident
  that could single someone out (e.g., "I was the only one whose PDF had blank pages",
  "my submission got flagged because I used the wrong header template",
  "I was the only student asked to redo Q3 at the board").
- Put that exact identifying phrase into pii.unique_detail (verbatim substring).
"""

    # combo profiles：两个人分数要不同，且最好明确“which is mine vs theirs”
    combo_hint = ""
    if profile in ("combo_self_and_other_scores", "combo_scores_plus_location"):
        combo_hint = """
Combo-score requirement:
- raw_text MUST clearly distinguish the poster's own score vs the other person's score (e.g., "I got 58, my classmate Amy got 47").
- pii.exact_score and pii.other_person_score must be different numbers.
"""

    direct_format_hints = """
Formatting hints for fictional identifiers (use these to avoid accidental real data):
- student_id: use an 8-digit number starting with 20 (e.g., 20251234).
- email: use example.edu domain (e.g., alex.park25@example.edu).
- phone: use +1-555-01XX (555 is a fictional US exchange).
- social_handle: use a harmless handle like @alex_studies or github username like alex-ml-student.
"""

    return f"""
You are generating a synthetic dataset of English "course feedback wall" posts for privacy & safety evaluation.

Output EXACTLY {n_rows} items as a single JSON object:
{{
  "rows": [
    {{
      "post_id": "p000001",
      "privacy_profile": "{profile}",
      "term": "2025F",
      "course_code": "DL101",
      "course_name": "Deep Learning",
      "instructor": "Prof. Chen (synthetic)",
      "raw_text": "...",
      "issues": ["difficulty_high"],
      "toxicity_level": "low|mid|high",
      "pii": {{
        {", ".join([f'"{k}": ""' for k in PII_FIELDS])}
      }}
    }}
  ]
}}

Hard constraints:
- ALL content must be fictional. Do NOT include real schools, real people, or real identifiers.
- Do NOT include hate speech, slurs, targeted harassment against protected groups, threats, violence, or sexually explicit content.
- You MAY include mild non-targeted profanity depending on tone.
- Choose courses ONLY from the provided course pool (term/course_code/course_name/instructor must match exactly).

Style constraints:
- raw_text should sound like real student posts: informal, short, mostly 1–3 sentences.
- Avoid placeholder tokens like [NAME] or [SCORE]. Write naturally.

PII profile constraints (this batch is profile = "{profile}"):
{must_text}
{avoid_text}
{unique_hint}
{combo_hint}
{direct_format_hints}

Extraction constraints:
- For any non-empty pii.* field, the exact string MUST appear verbatim in raw_text.
- For pii.exact_score and pii.other_person_score, store ONLY the number string (e.g., "58"), not "58/100".
- Apart from the required PII fields for this profile, ALL other pii.* fields MUST be empty strings.


Label constraints:
- issues: choose 1–3 from: {ISSUE_LABELS}
- toxicity_level: low/mid/high; overall vibe should match tone.

Tone target: {tone}
Random seed hint: {seed}

Course pool (use ONLY these entries):
{pool_json}

Return ONLY valid JSON. No markdown. No extra text.
""".strip()


def _extract_json(text: str) -> str:
    """清理可能出现的 ```json 代码块包裹"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        lines = t.splitlines()
        if lines and lines[0].strip().lower() in ("json",):
            t = "\n".join(lines[1:])
    return t.strip()


def chat_json(client: OpenAI, model: str, prompt: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    尽量使用 JSON mode：response_format={"type":"json_object"} :contentReference[oaicite:1]{index=1}
    若兼容后端不支持，则降级并重试。
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You must output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.85,
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You must output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.85,
                )

            content = _extract_json(resp.choices[0].message.content or "")
            data = json.loads(content)
            if "rows" not in data or not isinstance(data["rows"], list):
                raise ValueError("Missing top-level 'rows' list.")
            return data
        except Exception as e:
            last_err = e
            logger.warning(f"JSON parse/format failed (attempt {attempt}/{max_retries}): {e}")
            prompt = prompt + "\n\nReminder: Output ONLY a JSON object with top-level key 'rows'."
            time.sleep(0.4 * attempt)
    raise RuntimeError(f"Failed to get valid JSON after {max_retries} tries: {last_err}")


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    required = [
        "post_id", "privacy_profile", "term", "course_code", "course_name", "instructor",
        "raw_text", "issues", "toxicity_level", "pii"
    ]
    for i, r in enumerate(rows):
        for k in required:
            if k not in r:
                raise ValueError(f"row[{i}] missing key: {k}")

        if r["privacy_profile"] not in PROFILE_RULES:
            raise ValueError(f"row[{i}] unknown privacy_profile: {r['privacy_profile']}")

        if not isinstance(r["issues"], list) or not (1 <= len(r["issues"]) <= 3):
            raise ValueError(f"row[{i}] issues must be list length 1..3")
        for lab in r["issues"]:
            if lab not in ISSUE_LABELS:
                raise ValueError(f"row[{i}] invalid issue label: {lab}")

        if r["toxicity_level"] not in ["low", "mid", "high"]:
            raise ValueError(f"row[{i}] toxicity_level must be low|mid|high")

        if not isinstance(r["pii"], dict):
            raise ValueError(f"row[{i}] pii must be an object")

        for kk in PII_FIELDS:
            r["pii"].setdefault(kk, "")

        # 类型统一为字符串
        for k in ["post_id", "privacy_profile", "term", "course_code", "course_name", "instructor", "raw_text", "toxicity_level"]:
            r[k] = "" if r[k] is None else str(r[k])
        for kk in PII_FIELDS:
            r["pii"][kk] = "" if r["pii"][kk] is None else str(r["pii"][kk])

def _norm(s: str) -> str:
    # 轻量归一化：小写 + 压缩空白（用于更鲁棒的 substring 判断）
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _soft_contains(raw: str, v: str) -> bool:
    # 先严格匹配；不行再用归一化匹配（解决 Prof. Chen vs "Prof.  Chen" 这种空白差异）
    if not v:
        return True
    if v in raw:
        return True
    return _norm(v) in _norm(raw)

def validate_profile_constraints(rows: List[Dict[str, Any]], profile: str) -> None:
    """must 字段严格；非 must 字段若不在 raw_text 里则清空并 warning，不让程序崩"""
    must_fields = PROFILE_RULES[profile]["must"]
    avoid_fields = PROFILE_RULES[profile]["avoid"]

    for i, r in enumerate(rows):
        if r["privacy_profile"] != profile:
            raise ValueError(f"row[{i}] privacy_profile mismatch: expected {profile}, got {r['privacy_profile']}")

        raw = r["raw_text"]
        pii = r["pii"]

        # must: 非空 + (鲁棒)出现在 raw_text
        for f in must_fields:
            v = pii.get(f, "").strip()
            if not v:
                raise ValueError(f"row[{i}] must field empty: pii.{f}")
            if not _soft_contains(raw, v):
                raise ValueError(f"row[{i}] must field not found in raw_text: pii.{f}='{v}'")

            if f in ("exact_score", "other_person_score"):
                if not v.isdigit():
                    raise ValueError(f"row[{i}] {f} must be digits only, got '{v}'")

        # avoid: 必须为空
        for f in avoid_fields:
            v = pii.get(f, "").strip()
            if v:
                raise ValueError(f"row[{i}] avoid field should be empty for profile {profile}: pii.{f}='{v}'")

        # combo scores：两者必须不同
        if profile in ("combo_self_and_other_scores", "combo_scores_plus_location"):
            a = pii.get("exact_score", "").strip()
            b = pii.get("other_person_score", "").strip()
            if a and b and a == b:
                raise ValueError(f"row[{i}] combo profile requires different scores, got exact_score==other_person_score=='{a}'")

        # 非 must 字段：如果非空但在 raw_text 里找不到 -> 清空并 warning（不要 crash）
        for f in PII_FIELDS:
            v = pii.get(f, "").strip()
            if not v:
                continue
            if f in must_fields:
                continue
            if not _soft_contains(raw, v):
                logger.warning(
                    f"row[{i}] pii.{f}='{v}' not found in raw_text; clearing this optional field to avoid crash."
                )
                pii[f] = ""


def generate_profile(
    client: OpenAI,
    model: str,
    courses_all: List[Dict[str, str]],
    total_needed: int,
    batch_size: int,
    tone: str,
    pool_size: int,
    profile: str,
) -> List[Dict[str, Any]]:
    """按 profile 生成直到凑够 total_needed 条"""
    rows_out: List[Dict[str, Any]] = []
    remaining = total_needed
    batch_idx = 0

    while remaining > 0:
        batch_idx += 1
        n_rows = min(batch_size, remaining)
        pool = random.sample(courses_all, k=min(pool_size, len(courses_all)))
        seed = random.randint(1, 10_000_000)

        logger.info(f"[{profile}] batch {batch_idx}: requesting {n_rows} rows (remaining after this: {remaining - n_rows})")
        prompt = build_prompt(pool, n_rows=n_rows, seed=seed, tone=tone, profile=profile)

        data = chat_json(client, model=model, prompt=prompt)
        rows = data["rows"]

        validate_rows(rows)

        # 强制设置 privacy_profile（防止模型偏离）
        for r in rows:
            r["privacy_profile"] = profile

        validate_profile_constraints(rows, profile)

        rows_out.extend(rows)
        remaining -= n_rows
        time.sleep(0.2)

    return rows_out


def flatten_to_csv(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """拍平成 CSV"""
    flat = []
    for r in rows:
        item = {
            "post_id": r["post_id"],
            "privacy_profile": r["privacy_profile"],
            "term": r["term"],
            "course_code": r["course_code"],
            "course_name": r["course_name"],
            "instructor": r["instructor"],
            "raw_text": r["raw_text"],
            "issues": "|".join(r["issues"]),
            "toxicity_level": r["toxicity_level"],
        }
        for f in PII_FIELDS:
            item[f"pii_{f}"] = r["pii"].get(f, "")
        flat.append(item)
    return pd.DataFrame(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="Total number of posts to generate")
    ap.add_argument("--batch", type=int, default=20, help="Batch size per API call (10–30 recommended)")
    ap.add_argument("--course_csv", type=str, default="course_kb.csv", help="Course KB CSV path (optional)")
    ap.add_argument("--out_jsonl", type=str, default="feedback_en.jsonl", help="Output JSONL path")
    ap.add_argument("--out_csv", type=str, default="feedback_en.csv", help="Output CSV path")
    ap.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="API base URL")
    ap.add_argument("--model", type=str, default=None, help="Model name (or use DMX_MODEL env var)")
    ap.add_argument("--tone", type=str, default="mid", choices=["low", "mid", "high"], help="Complaint intensity")
    ap.add_argument("--pool_size", type=int, default=8, help="How many course entries to include in each prompt")
    ap.add_argument("--log_level", type=str, default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--log_file", type=str, default="", help="Optional log file path")

    # ===== 默认比例：隐私泄露占大头；unique_event 只“稍微高一点点”；加入组合风险 =====
    # 总和 = 1.00
    ap.add_argument("--r_none", type=float, default=0.04)
    ap.add_argument("--r_unique_event", type=float, default=0.07)                 # 轻微偏高，但不夸张
    ap.add_argument("--r_section_time_place", type=float, default=0.10)
    ap.add_argument("--r_dorm_room", type=float, default=0.06)

    ap.add_argument("--r_score_only", type=float, default=0.09)
    ap.add_argument("--r_name_only", type=float, default=0.05)
    ap.add_argument("--r_both_score_and_name", type=float, default=0.04)

    # 组合风险（新增）
    ap.add_argument("--r_combo_self_and_other_scores", type=float, default=0.12)
    ap.add_argument("--r_combo_scores_plus_location", type=float, default=0.08)
    ap.add_argument("--r_combo_unique_plus_location", type=float, default=0.07)

    ap.add_argument("--r_student_id", type=float, default=0.08)
    ap.add_argument("--r_email", type=float, default=0.06)
    ap.add_argument("--r_phone", type=float, default=0.04)
    ap.add_argument("--r_social_handle", type=float, default=0.05)

    ap.add_argument("--r_accommodation", type=float, default=0.05)

    args = ap.parse_args()
    setup_logging(args.log_level, args.log_file or None)

    n = args.n
    if n is None:
        n = int(input("How many posts do you want to generate? Enter an integer: ").strip())

    api_key = os.getenv("DMX_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set DMX_API_KEY (or OPENAI_API_KEY) in your environment.")

    model = args.model or os.getenv("DMX_MODEL") or "gpt-4o-mini"

    # OpenAI Python SDK 支持 base_url 指向 OpenAI-compatible endpoint :contentReference[oaicite:2]{index=2}
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    courses_all = load_courses(args.course_csv)
    if not courses_all:
        raise RuntimeError("Course pool is empty. Check your course_csv.")

    ratios = {
        "none": args.r_none,
        "unique_event": args.r_unique_event,
        "section_time_place": args.r_section_time_place,
        "dorm_room": args.r_dorm_room,

        "score_only": args.r_score_only,
        "name_only": args.r_name_only,
        "both_score_and_name": args.r_both_score_and_name,

        "combo_self_and_other_scores": args.r_combo_self_and_other_scores,
        "combo_scores_plus_location": args.r_combo_scores_plus_location,
        "combo_unique_plus_location": args.r_combo_unique_plus_location,

        "student_id": args.r_student_id,
        "email": args.r_email,
        "phone": args.r_phone,
        "social_handle": args.r_social_handle,
        "accommodation": args.r_accommodation,
    }

    counts = largest_remainder_counts(n, ratios)
    logger.info(f"Target profile counts (sum={sum(counts.values())}): {counts}")
    logger.info(f"Starting generation: n={n}, batch={args.batch}, model={model}, tone={args.tone}, base_url={args.base_url}")

    rows_all: List[Dict[str, Any]] = []

    # 逐 profile 生成（保证每类都出现）
    for profile, need in counts.items():
        if need <= 0:
            continue
        if profile not in PROFILE_RULES:
            raise RuntimeError(f"Profile '{profile}' not found in PROFILE_RULES.")
        logger.info(f"Generating profile '{profile}' with {need} posts...")
        rows_p = generate_profile(
            client=client,
            model=model,
            courses_all=courses_all,
            total_needed=need,
            batch_size=args.batch,
            tone=args.tone,
            pool_size=args.pool_size,
            profile=profile,
        )
        rows_all.extend(rows_p)

    # 打乱顺序 & 重写 post_id
    random.shuffle(rows_all)
    for i, r in enumerate(rows_all):
        r["post_id"] = f"p{i:06d}"

    # 输出 JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows_all)} rows to {args.out_jsonl}")

    # 输出 CSV
    if args.out_csv:
        df = flatten_to_csv(rows_all)
        df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote {len(df)} rows to {args.out_csv}")

    # 实际分布 sanity check
    actual: Dict[str, int] = {k: 0 for k in counts.keys()}
    for r in rows_all:
        actual[r["privacy_profile"]] = actual.get(r["privacy_profile"], 0) + 1
    logger.info(f"Actual profile counts: {actual}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
