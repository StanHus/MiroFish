#!/usr/bin/env python3
"""
Standalone evaluator runner — no Flask dependency.
Runs the full MiroFish AP evaluation pipeline.

Usage:
  python scripts/run_evaluator.py --input configs/sample_questions.json --output eval_results/ --mode full
  python scripts/run_evaluator.py --input configs/sample_questions.json --mode questions
  python scripts/run_evaluator.py --input configs/sample_questions.json --mode psychometric
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))

from dotenv import load_dotenv
_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

from openai import AsyncOpenAI

# Import evaluator modules directly — bypass app/__init__.py (Flask dependency)
sys.path.insert(0, _backend_dir)

# Prevent app/__init__.py from importing Flask
import types
app_pkg = types.ModuleType("app")
app_pkg.__path__ = [os.path.join(_backend_dir, "app")]
app_pkg.__package__ = "app"
sys.modules["app"] = app_pkg

eval_pkg = types.ModuleType("app.evaluator")
eval_pkg.__path__ = [os.path.join(_backend_dir, "app", "evaluator")]
eval_pkg.__package__ = "app.evaluator"
sys.modules["app.evaluator"] = eval_pkg

from app.evaluator.modules.question_qc import QuestionQCAnalyzer
from app.evaluator.modules.explanation_qc import ExplanationQCAnalyzer
from app.evaluator.modules.psychometric import run_psychometric_simulation, AP_SCORE_DISTRIBUTIONS
from app.evaluator.utils import calculate_pass_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_questions(input_path: str, limit: Optional[int] = None) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = raw if isinstance(raw, list) else raw.get("questions", raw.get("items", []))
    if limit and limit > 0:
        questions = questions[:limit]

    normalized = []
    for q in questions:
        nq = {
            "question_id": q.get("question_id", q.get("id", "")),
            "question": q.get("question", q.get("stem", "")),
            "topic": q.get("topic", q.get("standard", "")),
            "skill": q.get("skill", ""),
            "passage": q.get("passage", ""),
        }
        opts = q.get("options", q.get("choices", {}))
        if isinstance(opts, list):
            nq["choices"] = {chr(65 + i): o for i, o in enumerate(opts)}
        elif isinstance(opts, dict):
            nq["choices"] = opts
        else:
            nq["choices"] = {}

        ca = q.get("correct_index", q.get("answer_index", q.get("correct_answer", 0)))
        if isinstance(ca, int):
            nq["correct_answer"] = chr(65 + ca)
            nq["correct_index"] = ca
        else:
            nq["correct_answer"] = str(ca)
            nq["correct_index"] = ord(str(ca).upper()) - 65 if str(ca).isalpha() else 0

        nq["explanations"] = q.get("explanations", {})
        nq["difficulty"] = q.get("difficulty_estimate", q.get("difficulty", 0.0))
        nq["discrimination"] = q.get("discrimination_estimate", q.get("discrimination", 1.0))
        nq["distractor_weights"] = q.get("distractor_attractiveness")
        nq["n_options"] = len(nq["choices"])
        normalized.append(nq)

    return normalized


async def run_pipeline(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.environ.get("LLM_MODEL_NAME", "gemini-3.1-flash-lite-preview")

    if not api_key:
        print("  ❌ LLM_API_KEY not set in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    questions = load_questions(args.input, args.limit)
    print(f"\n  📝 Loaded {len(questions)} questions")
    print(f"  🤖 Model: {model}")
    print(f"  📁 Output: {output_dir}\n")

    qc_results = None
    psych_report = None

    # ── Stage 1: Question QC ─────────────────────────────────────────────────
    if args.mode in ("questions", "full"):
        print("=" * 65)
        print("  STAGE 1: QUESTION QUALITY CONTROL (LLM)")
        print("=" * 65)

        analyzer = QuestionQCAnalyzer(client=client, model=model)

        def progress(cur, total, msg):
            print(f"  [{cur:2d}/{total}] {msg}")

        qc_results = await analyzer.analyze_batch(
            questions, concurrency=args.concurrency, progress_callback=progress
        )

        out = output_dir / "question_qc.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(qc_results, f, indent=2, ensure_ascii=False)

        stats = calculate_pass_rate(qc_results)
        print(f"\n  📊 Question QC Summary:")
        print(f"     Total:     {stats['total']}")
        print(f"     Passed:    {stats['passed']} ({stats['pass_rate']:.0%})")
        print(f"     Failed:    {stats['failed']}")
        print(f"     Avg Score: {stats['average_score']:.2f}")

        # Print per-question details
        for r in qc_results:
            qid = r["question_id"]
            score = r["overall_score"]
            status = "✅" if r["passed"] else "❌"
            checks_detail = []
            for cn, cv in r.get("checks", {}).items():
                icon = "✓" if cv["score"] == 1 else "✗"
                checks_detail.append(f"{icon}{cn}")
            cog = r.get("cognitive_level", {})
            dok_str = f" | DOK {cog.get('dok_level', '?')}" if cog else ""
            print(f"     {status} {qid}: {score:.2f}{dok_str}  [{', '.join(checks_detail)}]")

        print(f"\n     Saved to {out}\n")

    # ── Stage 2: Psychometric Simulation ─────────────────────────────────────
    if args.mode in ("psychometric", "full"):
        print("=" * 65)
        print("  STAGE 2: PSYCHOMETRIC SIMULATION (IRT)")
        print("=" * 65)

        subject = args.subject
        dist_info = AP_SCORE_DISTRIBUTIONS.get(subject, {})
        print(f"  Subject:  {dist_info.get('label', subject)}")
        print(f"  Students: {args.students} × {args.runs} runs\n")

        # Merge LLM difficulty estimates if available
        if qc_results:
            qc_map = {r["question_id"]: r for r in qc_results}
            for q in questions:
                cog = qc_map.get(q["question_id"], {}).get("cognitive_level")
                if cog and isinstance(cog, dict) and q.get("difficulty", 0) == 0:
                    dok = cog.get("dok_level", 2)
                    q["difficulty"] = (dok - 2) * 0.5

        def psych_progress(step, total, msg):
            print(f"  [{step}/{total}] {msg}")

        stats = run_psychometric_simulation(
            questions=questions,
            n_students=args.students,
            n_runs=args.runs,
            subject=subject,
            seed=args.seed,
            progress_callback=psych_progress,
        )

        # Format report
        target_dist = {b: dist_info.get(b, 0) for b in range(1, 6)}

        print(f"\n  📊 Psychometric Results:")
        print(f"     KR-20 Reliability: {stats.kr20}")
        print(f"     Mean % Correct:    {stats.mean_score / stats.n_questions * 100:.1f}%")
        print(f"     Std Dev:           {stats.std_dev}")

        print(f"\n     Score Distribution (sim vs target):")
        for b in [5, 4, 3, 2, 1]:
            sim = stats.score_distribution.get(b, 0)
            tgt = target_dist.get(b, 0)
            print(f"       AP {b}: {sim*100:5.1f}%  (target: {tgt*100:.1f}%)")

        print(f"\n     Quality Summary:")
        for q, n in sorted(stats.quality_summary.items()):
            emoji = {"excellent": "🟢", "good": "🟡", "acceptable": "🟠", "poor": "🔴", "discard": "⛔"}.get(q, "❓")
            print(f"       {emoji} {q:12s}: {n} items")

        print(f"\n     Item Details:")
        for item in stats.items:
            qi = {"excellent": "🟢", "good": "🟡", "acceptable": "🟠", "poor": "🔴", "discard": "⛔"}.get(item.quality, "❓")
            print(f"       {qi} {item.question_id}: p={item.difficulty_index:.3f}  rpb={item.discrimination_index:.3f}  [{item.quality}]")
            for i, freq in enumerate(item.option_frequencies):
                q = next((q for q in questions if q["question_id"] == item.question_id), {})
                ci = q.get("correct_index", 0)
                marker = "✓" if i == ci else " "
                low = item.option_by_band.get(1, [0] * len(item.option_frequencies))[i]
                high = item.option_by_band.get(5, [0] * len(item.option_frequencies))[i]
                print(f"         {marker} {chr(65+i)}: {freq*100:5.1f}%  (low:{low*100:4.1f}% / high:{high*100:4.1f}%)")

        # Save
        report = {
            "meta": {"generated_at": datetime.now().isoformat(), "subject": dist_info.get("label", subject)},
            "test_level": {
                "kr20": stats.kr20, "mean_score": stats.mean_score, "std_dev": stats.std_dev,
                "score_distribution": {str(b): round(p, 3) for b, p in stats.score_distribution.items()},
                "quality_summary": stats.quality_summary,
            },
            "items": [
                {
                    "question_id": it.question_id,
                    "difficulty_index": it.difficulty_index,
                    "discrimination_index": it.discrimination_index,
                    "quality": it.quality,
                    "option_frequencies": it.option_frequencies,
                    "option_by_band": {str(b): v for b, v in it.option_by_band.items()},
                }
                for it in stats.items
            ],
        }
        out = output_dir / "psychometric.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n     Saved to {out}\n")

    # ── Combined ─────────────────────────────────────────────────────────────
    if args.mode == "full" and qc_results:
        combined = {
            "meta": {"generated_at": datetime.now().isoformat(), "n_questions": len(questions)},
            "question_qc": qc_results,
            "psychometric": report if args.mode in ("psychometric", "full") else None,
        }
        out = output_dir / "full_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
        print(f"  💾 Full report saved to {out}")

    print("\n  ✅ Evaluation pipeline complete.\n")


def main():
    parser = argparse.ArgumentParser(description="MiroFish AP Evaluator")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./eval_results")
    parser.add_argument("--mode", choices=["questions", "psychometric", "full"], default="full")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--students", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subject", type=str, default="ap_us_government",
                        choices=list(AP_SCORE_DISTRIBUTIONS.keys()))
    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
