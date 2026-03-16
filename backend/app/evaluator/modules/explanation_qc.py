"""
Explanation Quality Control Module — AP Social Studies
LLM-based explanation/feedback quality checks.

Validates the quality of answer explanations/feedback for both correct
and incorrect answer choices.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ..utils import load_prompts, parse_xml_response, fill_prompt_variables

logger = logging.getLogger(__name__)


class ExplanationQCAnalyzer:
    """Analyzes explanation/feedback quality for correct and incorrect answer choices."""

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

    async def _run_check(
        self, check_name: str, prompt_template: str, variables: Dict[str, str]
    ) -> Tuple[int, str]:
        """Run a single explanation check."""
        filled = fill_prompt_variables(prompt_template, variables)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": filled}],
                temperature=self.temperature,
            )
            text = response.choices[0].message.content.strip()
            return parse_xml_response(text)
        except Exception as e:
            logger.error(f"Explanation check '{check_name}' failed: {e}")
            return 0, f"Error: {e}"

    async def analyze_explanation(
        self,
        question_data: Dict[str, Any],
        choice_letter: str,
        feedback: str,
        is_correct: bool,
    ) -> Dict[str, Any]:
        """
        Analyze a single explanation/feedback text.

        Args:
            question_data: question dict
            choice_letter: which option this feedback is for
            feedback: the feedback text to evaluate
            is_correct: whether this is feedback for the correct answer

        Returns:
            {check_name: {"score": 0|1, "reasoning": str}, ...}
        """
        choices = question_data.get("choices", question_data.get("options", {}))
        if isinstance(choices, list):
            choices = {chr(65 + i): c for i, c in enumerate(choices)}

        correct_answer = question_data.get("correct_answer", "")
        if isinstance(correct_answer, int):
            correct_answer = chr(65 + correct_answer)

        variables = {
            "question": question_data.get("question", question_data.get("stem", "")),
            "correct_answer": f"{correct_answer}) {choices.get(correct_answer, '')}",
            "distractor": f"{choice_letter}) {choices.get(choice_letter, '')}",
            "feedback": feedback,
            "passage": question_data.get("passage", ""),
        }

        exp_prompts = self.prompts.get("explanation_qc", {})
        if is_correct:
            checks = exp_prompts.get("correct", {})
        else:
            checks = exp_prompts.get("distractor", {})

        results = {}
        for check_name, config in checks.items():
            score, reasoning = await self._run_check(
                check_name, config["prompt"], variables
            )
            results[check_name] = {"score": score, "reasoning": reasoning}

        total = sum(v["score"] for v in results.values())
        count = len(results) if results else 1

        return {
            "choice": choice_letter,
            "is_correct": is_correct,
            "checks": results,
            "overall_score": round(total / count, 3),
            "passed": (total / count) >= 0.8 if count > 0 else False,
        }

    async def analyze_all_explanations(
        self,
        question_data: Dict[str, Any],
        explanations: Dict[str, str],
        concurrency: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze all explanations for a question (correct + each distractor).

        Args:
            question_data: question dict
            explanations: {choice_letter: feedback_text}
            concurrency: max concurrent

        Returns:
            list of per-choice results
        """
        correct = question_data.get("correct_answer", "")
        if isinstance(correct, int):
            correct = chr(65 + correct)

        sem = asyncio.Semaphore(concurrency)
        results = []

        async def process(letter, feedback):
            async with sem:
                return await self.analyze_explanation(
                    question_data, letter, feedback, is_correct=(letter == correct)
                )

        tasks = [process(letter, fb) for letter, fb in explanations.items()]
        results = await asyncio.gather(*tasks)
        return list(results)
