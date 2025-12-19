# KBRD Reproducibility Package (Baselines B1–B3 + Ours)

This folder contains code and artifacts for the course project **KB-Grounded Two-Stage “Readable” De-identification (KBRD)**.

**What you can do:**
1) **Reproduce the objective metrics (no API needed)** using the provided prediction files in `KBRD/evaluation/` (recommended for grading).  
2) Optionally **re-run the baselines and KBRD** to regenerate predictions (B3 + KBRD require an LLM API key).

---

## 0) Folder structure

```
KBRD/
  data/
    feedback.jsonl          # gold test set (1000 items) with raw_text + pii dict
    feedback.csv
  src/
    baseline1.py            # B1: regex-based readable redaction
    baseline2.py            # B2: heuristic "NER-like" + regex redaction
    baseline3.py            # B3: prompted LLM rewriting (no KB)
    ours_pipeline.py        # KBRD: KB-grounded plan-then-write + verify/fallback
    evaluate_redaction.py   # objective evaluation script
    course_kb.csv           # course KB (CSV)
    (other helper modules)
  evaluation/
    pred_baseline1.jsonl
    pred_baseline2.jsonl
    pred_baseline3.jsonl
    pred_ours.jsonl
    eva_result_baseline1.csv
    eva_result_baseline2.csv
    eval_result_baseline3.csv
    eval_result_ours.csv
```

---

## 1) Environment setup

Python 3.10+ is recommended.

```bash
cd KBRD
python -m venv .venv
source .venv/bin/activate

pip install -U openai pandas tqdm
```

---

## 2) Reproduce objective metrics (NO API calls) — recommended

This recomputes the metrics reported in the paper from the **provided** prediction files.

```bash
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_baseline1.jsonl --out_csv evaluation/re_eval_baseline1.csv --out_overall_json evaluation/re_overall_baseline1.json
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_baseline2.jsonl --out_csv evaluation/re_eval_baseline2.csv --out_overall_json evaluation/re_overall_baseline2.json
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_baseline3.jsonl --out_csv evaluation/re_eval_baseline3.csv --out_overall_json evaluation/re_overall_baseline3.json
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_ours.jsonl      --out_csv evaluation/re_eval_ours.csv      --out_overall_json evaluation/re_overall_ours.json
```

Outputs:
- `evaluation/re_eval_*.csv` (per-type summary and overall metrics)
- `evaluation/re_overall_*.json` (overall aggregate metrics)

---

## 3) Re-run baselines B1 / B2 (NO API calls)

### B1: Regex-based readable redaction
```bash
python src/baseline1.py --in data/feedback.jsonl --out evaluation/pred_baseline1_rerun.jsonl
```

### B2: Heuristic NER-like redaction
```bash
python src/baseline2.py --in data/feedback.jsonl --out evaluation/pred_baseline2_rerun.jsonl
```

Then evaluate:
```bash
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_baseline1_rerun.jsonl --out_csv evaluation/eval_b1_rerun.csv
python src/evaluate_redaction.py --gold data/feedback.jsonl --pred evaluation/pred_baseline2_rerun.jsonl --out_csv evaluation/eval_b2_rerun.csv
```

---

## 4) LLM-based methods (B3 + KBRD): API setup

**In our experiments**, all LLM calls used the same backbone model:
- `qwen-max-latest` for **Baseline 3**, and for **KBRD planner + writer**

Both `baseline3.py` and `ours_pipeline.py` use an **OpenAI-compatible** Chat Completions interface.

### Option A: Use environment variables
```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_OPENAI_COMPATIBLE_ENDPOINT"
```

### Option B: Use DMX-style environment variables (also supported)
```bash
export DMX_API_KEY="YOUR_KEY"
export DMX_BASE_URL="YOUR_OPENAI_COMPATIBLE_ENDPOINT"
```

### Notes on Qwen (DashScope / Model Studio) OpenAI-compatible endpoints
If you use Alibaba Cloud Model Studio (DashScope) in OpenAI-compatible mode, common base URLs include:
- China (Beijing): `https://dashscope.aliyuncs.com/compatible-mode/v1`
- International (Singapore): `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

(Any OpenAI-compatible provider is acceptable as long as it supports `/chat/completions`.)

---

## 5) Re-run Baseline 3 (LLM rewriting, no KB)

```bash
python src/baseline3.py   --in data/feedback.jsonl   --out evaluation/pred_baseline3_rerun.jsonl   --model qwen-max-latest   --temperature 0.0   --max_workers 8   --write_mode resume   --log_file evaluation/b3_rerun.log
```

Useful options:
- `--write_mode overwrite|append|resume` (resume skips already processed `post_id`s)
- `--max_items N` for a quick partial run
- `--dry_run` to run without API calls (produces heuristic stub outputs; metrics will not match)

---

## 6) Re-run KBRD (ours): KB-grounded two-stage plan-then-write

```bash
python src/ours_pipeline.py   --in data/feedback.jsonl   --out evaluation/pred_ours_rerun.jsonl   --course_kb_csv src/course_kb.csv   --planner_model qwen-max-latest   --writer_model qwen-max-latest   --policy_strictness 0   --max_workers 4   --flush_every 1   --resume   --log_file evaluation/kbrd_rerun.log
```

Useful options:
- `--save_plan` / `--save_debug` to store intermediate artifacts for inspection
- `--limit N` for a quick partial run
- `--id_txt path/to/ids.txt` to process only specified `post_id`s and merge patches back into `--out`

---

## 7) Evaluate any newly generated predictions

```bash
python src/evaluate_redaction.py   --gold data/feedback.jsonl   --pred evaluation/pred_ours_rerun.jsonl   --out_csv evaluation/eval_ours_rerun.csv   --out_overall_json evaluation/overall_ours_rerun.json
```

---

## 8) Reproducibility notes

- **B1/B2** are deterministic.
- **B3/KBRD** depend on external LLM services; even with `temperature=0`, results can vary slightly due to provider-side changes.
- For grading, the folder includes the prediction files and evaluation CSVs used in the report under `KBRD/evaluation/`.
