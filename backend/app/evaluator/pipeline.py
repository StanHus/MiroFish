#!/usr/bin/env python3
"""
Unified Evaluation Pipeline — AP Social Studies
Unified Evaluation Pipeline — AP Social Studies

Orchestrates three evaluation stages:
  Stage 1: LLM-based Question QC (quality checks on stem + distractors)
  Stage 2: LLM-based Explanation QC (quality checks on feedback text)
  Stage 3: Psychometric Simulation (IRT-based item analysis)

Usage:
  python -m app.evaluator.pipeline --input questions.json --output results/ --mode full
  python -m app.evaluator.pipeline --input questions.json --mode questions  # QC only
  python -m app.evaluator.pipeline --input questions.json --mode psychometric  # IRT only
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

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .modules.question_qc import QuestionQCAnalyzer
from .modules.explanation_qc import ExplanationQCAnalyzer
from .modules.psychometric import run_psychometric_simulation, AP_SCORE_DISTRIBUTIONS
from .utils import calculate_pass_rate

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Main evaluation orchestrator."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LLM client (OpenAI-compatible — works with Gemini, OpenAI, etc.)
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("LLM_MODEL_NAME", "gemini-3.1-flash-lite-preview")

        if not api_key:
            logger.error("LLM_API_KEY not set")
            sys.exit(1)

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = getattr(args, "model", None) or model

        # Init analyzers
        if args.mode in ("questions", "full", "both"):
            self.question_qc = QuestionQCAnalyzer(
                client=self.client, model=self.model
            )
        else:
            self.question_qc = None

        if args.mode in ("explanations", "full", "both"):
            self.explanation_qc = ExplanationQCAnalyzer(
                client=self.client, model=self.model
            )
        else:
            self.explanation_qc = None

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSON (supports multiple formats)."""
        with open(self.args.input, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            questions = raw
        else:
            questions = raw.get("questions", raw.get("items", []))

        if self.args.limit and self.args.limit > 0:
            questions = questions[: self.args.limit]

        # Normalize format
        normalized = []
        for q in questions:
            nq = {
                "question_id": q.get("question_id", q.get("id", q.get("item_id", ""))),
                "question": q.get("question", q.get("stem", "")),
                "topic": q.get("topic", q.get("standard", q.get("CCSS", ""))),
                "skill": q.get("skill", q.get("DOK", "")),
                "passage": q.get("passage", q.get("stimulus", "")),
            }

            # Normalize choices
            opts = q.get("options", q.get("choices", {}))
            if isinstance(opts, list):
                nq["choices"] = {chr(65 + i): o for i, o in enumerate(opts)}
            elif isinstance(opts, dict):
                nq["choices"] = opts
            else:
                nq["choices"] = {}

            # Correct answer
            ca = q.get("correct_index", q.get("answer_index", q.get("correct_answer", 0)))
            if isinstance(ca, int):
                nq["correct_answer"] = chr(65 + ca)
                nq["correct_index"] = ca
            else:
                nq["correct_answer"] = str(ca)
                nq["correct_index"] = ord(str(ca).upper()) - 65 if str(ca).isalpha() else 0

            # Explanations (if present)
            nq["explanations"] = q.get("explanations", {})

            # IRT parameters (may be estimated by psychometric module)
            nq["difficulty"] = q.get("difficulty_estimate", q.get("difficulty", 0.0))
            nq["discrimination"] = q.get("discrimination_estimate", q.get("discrimination", 1.0))
            nq["distractor_weights"] = q.get("distractor_attractiveness", q.get("distractor_weights"))
            nq["n_options"] = len(nq["choices"])

            normalized.append(nq)

        logger.info(f"Loaded {len(normalized)} questions from {self.args.input}")
        return normalized

    async def run_question_qc(self, questions: List[Dict]) -> List[Dict]:
        """Stage 1: LLM-based question quality control."""
        if not self.question_qc:
            return []

        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: QUESTION QUALITY CONTROL")
        logger.info("=" * 60)

        def progress(cur, total, msg):
            logger.info(f"  [{cur}/{total}] {msg}")

        results = await self.question_qc.analyze_batch(
            questions, concurrency=self.args.concurrency, progress_callback=progress
        )

        # Save
        out = self.output_dir / f"question_qc_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        stats = calculate_pass_rate(results)
        logger.info(f"\n  Question QC Summary:")
        logger.info(f"    Total:    {stats['total']}")
        logger.info(f"    Passed:   {stats['passed']} ({stats['pass_rate']:.1%})")
        logger.info(f"    Failed:   {stats['failed']}")
        logger.info(f"    Avg Score: {stats['average_score']:.2f}")
        logger.info(f"  Saved to {out}")

        return results

    async def run_explanation_qc(self, questions: List[Dict]) -> List[Dict]:
        """Stage 2: LLM-based explanation quality control."""
        if not self.explanation_qc:
            return []

        questions_with_explanations = [q for q in questions if q.get("explanations")]
        if not questions_with_explanations:
            logger.info("\n  No explanations found — skipping Stage 2")
            return []

        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: EXPLANATION QUALITY CONTROL")
        logger.info("=" * 60)

        all_results = []
        for q in questions_with_explanations:
            result = await self.explanation_qc.analyze_all_explanations(
                q, q["explanations"], concurrency=self.args.concurrency
            )
            all_results.append({
                "question_id": q["question_id"],
                "explanation_results": result,
            })

        out = self.output_dir / f"explanation_qc_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved to {out}")

        return all_results

    def run_psychometric(self, questions: List[Dict], qc_results: Optional[List[Dict]] = None) -> Dict:
        """Stage 3: IRT psychometric simulation."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: PSYCHOMETRIC SIMULATION")
        logger.info("=" * 60)

        subject = getattr(self.args, "subject", "ap_us_government")
        dist_info = AP_SCORE_DISTRIBUTIONS.get(subject, {})
        logger.info(f"  Subject:  {dist_info.get('label', subject)}")
        logger.info(f"  Students: {self.args.students} × {self.args.runs} runs")

        def progress(step, total, msg):
            logger.info(f"  [{step}/{total}] {msg}")

        # If QC results include cognitive level / difficulty estimates, merge them
        if qc_results:
            qc_map = {r["question_id"]: r for r in qc_results}
            for q in questions:
                qc = qc_map.get(q["question_id"], {})
                cog = qc.get("cognitive_level")
                if cog and isinstance(cog, dict):
                    # Use LLM-estimated difficulty if we have it
                    dok = cog.get("dok_level", 2)
                    # Map DOK to IRT difficulty: DOK1=-0.5, DOK2=0, DOK3=0.5, DOK4=1.0
                    if q.get("difficulty", 0) == 0:
                        q["difficulty"] = (dok - 2) * 0.5

        stats = run_psychometric_simulation(
            questions=questions,
            n_students=self.args.students,
            n_runs=self.args.runs,
            subject=subject,
            seed=self.args.seed,
            progress_callback=progress,
        )

        # Format report
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "subject": dist_info.get("label", subject),
                "n_students": stats.n_students,
                "n_questions": stats.n_questions,
                "engine": "MiroFish Evaluator v1.0 (IRT 2PL/3PL)",
            },
            "test_level": {
                "kr20_reliability": stats.kr20,
                "mean_raw_score": stats.mean_score,
                "std_dev": stats.std_dev,
                "mean_percent_correct": round(stats.mean_score / stats.n_questions * 100, 1) if stats.n_questions else 0,
                "score_distribution": {
                    f"AP_{b}": f"{pct * 100:.1f}%" for b, pct in sorted(stats.score_distribution.items(), reverse=True)
                },
                "quality_summary": stats.quality_summary,
            },
            "items": [],
        }

        for item in stats.items:
            n_opts = len(item.option_frequencies)
            q_data = next((q for q in questions if q["question_id"] == item.question_id), {})

            distractor_analysis = []
            for i in range(n_opts):
                ci = q_data.get("correct_index", 0)
                low = item.option_by_band.get(1, [0] * n_opts)[i] if item.option_by_band else 0
                high = item.option_by_band.get(5, [0] * n_opts)[i] if item.option_by_band else 0
                is_correct = i == ci
                status = "correct_answer" if is_correct else (
                    "effective" if low > high and item.option_frequencies[i] >= 0.05 else
                    "weak" if item.option_frequencies[i] < 0.05 else "problematic"
                )
                distractor_analysis.append({
                    "option": chr(65 + i),
                    "frequency": item.option_frequencies[i],
                    "is_correct": is_correct,
                    "low_ability_freq": round(low, 3),
                    "high_ability_freq": round(high, 3),
                    "status": status,
                })

            report["items"].append({
                "question_id": item.question_id,
                "difficulty_index": item.difficulty_index,
                "discrimination_index": item.discrimination_index,
                "quality": item.quality,
                "option_analysis": distractor_analysis,
            })

        out = self.output_dir / f"psychometric_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\n  Psychometric Summary:")
        logger.info(f"    KR-20:          {stats.kr20}")
        logger.info(f"    Mean % Correct: {report['test_level']['mean_percent_correct']}%")
        for q, n in sorted(stats.quality_summary.items()):
            logger.info(f"    {q:12s}: {n} items")
        logger.info(f"  Saved to {out}")

        return report

    async def run(self):
        """Execute the full pipeline."""
        questions = self.load_questions()

        qc_results = None
        exp_results = None
        psych_report = None

        # Stage 1
        if self.args.mode in ("questions", "full", "both"):
            qc_results = await self.run_question_qc(questions)

        # Stage 2
        if self.args.mode in ("explanations", "full", "both"):
            exp_results = await self.run_explanation_qc(questions)

        # Stage 3
        if self.args.mode in ("psychometric", "full"):
            psych_report = self.run_psychometric(questions, qc_results)

        # Combined report
        if self.args.mode == "full":
            combined = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "mode": "full",
                    "n_questions": len(questions),
                },
                "question_qc": qc_results,
                "explanation_qc": exp_results,
                "psychometric": psych_report,
            }
            out = self.output_dir / f"full_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"\n  Combined report saved to {out}")

        logger.info("\n  ✅ Evaluation pipeline complete.")


async def main():
    parser = argparse.ArgumentParser(description="MiroFish AP Evaluator Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Questions JSON file")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--mode", choices=["questions", "explanations", "psychometric", "full", "both"],
                        default="full", help="Evaluation mode")
    parser.add_argument("--model", type=str, default=None, help="LLM model override")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent LLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--students", type=int, default=200, help="Students per psychometric run")
    parser.add_argument("--runs", type=int, default=3, help="Psychometric simulation runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subject", type=str, default="ap_us_government",
                        choices=list(AP_SCORE_DISTRIBUTIONS.keys()),
                        help="AP subject for score calibration")
    args = parser.parse_args()

    pipeline = EvaluationPipeline(args)
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
