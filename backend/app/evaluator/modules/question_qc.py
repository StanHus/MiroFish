"""
Question Quality Control Module — AP Social Studies
LLM-based question quality checks (stem, distractors, alignment).

Runs LLM-based checks on MCQ quality:
  - Distractor checks: grammatical_parallel, plausibility, homogeneity, specificity_balance, too_close
  - Question checks: standard_alignment, clarity_precision, single_correct_answer, cognitive_level, factual_accuracy
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ..utils import load_prompts, parse_xml_response, parse_json_response, fill_prompt_variables

logger = logging.getLogger(__name__)


class QuestionQCAnalyzer:
    """Analyzes question quality across all dimensions using LLM judges."""

    DISTRACTOR_CHECKS = [
        "grammatical_parallel",
        "plausibility",
        "homogeneity",
        "specificity_balance",
        "too_close",
    ]

    QUESTION_CHECKS = [
        "standard_alignment",
        "clarity_precision",
        "single_correct_answer",
        "cognitive_level",
        "factual_accuracy",
    ]

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.prompts = load_prompts()

    def _build_variables(self, question_data: Dict[str, Any]) -> Dict[str, str]:
        """Build template variables from question data."""
        choices = question_data.get("choices", question_data.get("options", {}))

        # Handle list-style options → dict
        if isinstance(choices, list):
            choices = {chr(65 + i): c for i, c in enumerate(choices)}

        all_choices = "\n".join(f"{k}) {v}" for k, v in sorted(choices.items()))

        correct = question_data.get("correct_answer", "")
        if isinstance(correct, int):
            correct = chr(65 + correct)

        correct_letter = correct
        correct_text = choices.get(correct, correct)

        return {
            "question": question_data.get("question", question_data.get("stem", "")),
            "all_choices": all_choices,
            "correct_answer": f"{correct_letter}) {correct_text}",
            "correct_letter": correct_letter,
            "standard_formatted": question_data.get("topic", question_data.get("standard", "Not specified")),
            "passage": question_data.get("passage", ""),
        }

    async def _run_check(
        self, check_name: str, prompt_config: Dict, variables: Dict[str, str]
    ) -> Tuple[int, str, Optional[Dict]]:
        """Run a single LLM check. Returns (score, reasoning, extra_data)."""
        filled = fill_prompt_variables(prompt_config["prompt"], variables)
        fmt = prompt_config.get("response_format", "xml")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": filled}],
                temperature=self.temperature,
            )
            text = response.choices[0].message.content.strip()

            if fmt == "json":
                data = parse_json_response(text)
                if data is None:
                    return 0, f"JSON parse failed: {text[:200]}", None

                # Handle too_close format
                if "too_close" in data:
                    passed = not data.get("too_close", True)
                    return (1 if passed else 0), data.get("explanation", ""), data

                # Handle cognitive_level format
                if "dok_level" in data:
                    return 1, data.get("reasoning", ""), data

                return 1, str(data), data
            else:
                score, reasoning = parse_xml_response(text)
                return score, reasoning, None

        except Exception as e:
            logger.error(f"Check '{check_name}' failed: {e}")
            return 0, f"Error: {e}", None

    async def analyze_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all QC checks on a single question.

        Args:
            question_data: dict with keys: question/stem, choices/options, correct_answer, topic

        Returns:
            {
              "question_id": str,
              "checks": {check_name: {"score": 0|1, "reasoning": str, "extra": ...}},
              "overall_score": float (0-1),
              "passed": bool,
              "cognitive_level": {...},
            }
        """
        variables = self._build_variables(question_data)
        qc_prompts = self.prompts["question_qc"]
        all_checks = {**qc_prompts.get("distractor_checks", {}), **qc_prompts.get("question_checks", {})}

        # Run all checks concurrently
        check_names = self.DISTRACTOR_CHECKS + self.QUESTION_CHECKS
        tasks = []
        for name in check_names:
            if name in all_checks:
                tasks.append((name, self._run_check(name, all_checks[name], variables)))

        results = {}
        cognitive_data = None

        for name, coro in tasks:
            score, reasoning, extra = await coro
            results[name] = {"score": score, "reasoning": reasoning}
            if extra:
                results[name]["extra"] = extra
            if name == "cognitive_level" and extra:
                cognitive_data = extra

        # Overall score = average of all binary checks (excluding cognitive_level which is informational)
        scoring_checks = {k: v for k, v in results.items() if k != "cognitive_level"}
        total = sum(v["score"] for v in scoring_checks.values())
        count = len(scoring_checks) if scoring_checks else 1
        overall = total / count

        qid = question_data.get("question_id", question_data.get("id", "unknown"))

        return {
            "question_id": qid,
            "checks": results,
            "overall_score": round(overall, 3),
            "passed": overall >= 0.8,
            "cognitive_level": cognitive_data,
        }

    async def analyze_batch(
        self,
        questions: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Run QC on a batch of questions with concurrency control.

        Args:
            questions: list of question dicts
            concurrency: max concurrent LLM calls
            progress_callback: (current, total, message)

        Returns:
            list of result dicts
        """
        sem = asyncio.Semaphore(concurrency)
        results = []
        total = len(questions)

        async def process_one(idx: int, q: Dict) -> Dict:
            async with sem:
                if progress_callback:
                    progress_callback(idx + 1, total, f"QC: {q.get('question_id', f'Q{idx+1}')}")
                return await self.analyze_question(q)

        tasks = [process_one(i, q) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        return list(results)
