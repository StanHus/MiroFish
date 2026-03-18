"""
Adversarial Swarm Testing.

Tests if wrong answers can be defended as correct - reveals ambiguity
that single-agent evaluation misses.

How it works:
- Each agent defends a different answer option as "correct"
- If a wrong answer gets a strong defense, question may be ambiguous
- Catches: accidentally correct distractors, ambiguous wording, overlapping concepts
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DefenseResult:
    """Result of defending a single answer option."""
    option_key: str
    option_text: str
    argument: str
    strength: int  # 1-5: how convincing is the defense?
    is_correct: bool
    angle: str = ""


DEFENDER_PROMPT = """You are a debate champion. Your ONLY job is to argue that Option {option_key} is the CORRECT answer.

QUESTION: {question_text}

OPTIONS:
{options_text}

YOUR ASSIGNED OPTION: {option_key}) {option_text}

TASK: Make the STRONGEST possible case that {option_key} is correct. Find any interpretation, technicality, edge case, or angle that makes this answer defensible. Be creative but intellectually honest.

After your argument, rate its strength:
- 1 = Very weak (clearly wrong, no reasonable defense)
- 2 = Weak (technically wrong but I found a minor angle)
- 3 = Medium (debatable, some validity to this interpretation)
- 4 = Strong (legitimate argument, could convince someone)
- 5 = Very strong (this might actually be correct or equally valid)

Return JSON:
{{
  "argument": "Your best argument for why {option_key} is correct",
  "strength": <1-5>,
  "angle": "The interpretation or technicality you exploited"
}}"""


class AdversarialSwarm:
    """
    Runs adversarial testing on multiple-choice questions.

    Each option gets a "defender" agent that argues it's correct.
    If wrong answers are strongly defensible, the question may be ambiguous.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_threshold: int = 4,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_threshold = base_threshold

    async def test(
        self,
        question: Dict[str, Any],
        threshold_adjustment: float = 0,
    ) -> Dict[str, Any]:
        """
        Run adversarial testing on a question.

        Args:
            question: Dict with text, options, correct_answer
            threshold_adjustment: Calibration adjustment to threshold

        Returns:
            Analysis results with defenses, flags, verdict
        """
        text = question.get("text", "")
        options = question.get("options", {})
        correct_answer = question.get("correct_answer", "")

        if not text or not options or not correct_answer:
            return {"error": "Invalid question format"}

        adjusted_threshold = self.base_threshold + threshold_adjustment

        # Launch all defenders in parallel
        tasks = [
            self._defend_option(text, options, key, opt_text)
            for key, opt_text in options.items()
        ]

        defenses = await asyncio.gather(*tasks)

        # Analyze results
        results = {}
        flags: List[Dict[str, Any]] = []

        for defense in defenses:
            defense.is_correct = defense.option_key == correct_answer
            results[defense.option_key] = {
                "text": defense.option_text,
                "argument": defense.argument,
                "strength": defense.strength,
                "angle": defense.angle,
                "is_correct": defense.is_correct,
            }

            # Flag wrong answers with strong defenses
            if not defense.is_correct and defense.strength >= adjusted_threshold:
                flags.append({
                    "option": defense.option_key,
                    "text": defense.option_text,
                    "argument": defense.argument,
                    "strength": defense.strength,
                    "issue": "DEFENSIBLE_WRONG_ANSWER",
                })

        # Check if correct answer has weak defense
        correct_defense = results.get(correct_answer, {})
        if correct_defense.get("strength", 5) <= 2:
            flags.append({
                "option": correct_answer,
                "text": correct_defense.get("text", ""),
                "argument": correct_defense.get("argument", ""),
                "strength": correct_defense.get("strength", 0),
                "issue": "WEAK_CORRECT_ANSWER",
            })

        verdict = "CLEAR" if not flags else "AMBIGUOUS"

        return {
            "defenses": results,
            "flags": flags,
            "verdict": verdict,
            "stats": {
                "total_options": len(options),
                "defensible_wrong": len([f for f in flags if f["issue"] == "DEFENSIBLE_WRONG_ANSWER"]),
                "weak_correct": len([f for f in flags if f["issue"] == "WEAK_CORRECT_ANSWER"]),
            },
            "threshold_used": adjusted_threshold,
        }

    async def _defend_option(
        self,
        question_text: str,
        options: Dict[str, str],
        option_key: str,
        option_text: str,
    ) -> DefenseResult:
        """Have an agent argue that a specific option is correct."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return DefenseResult(
                option_key=option_key,
                option_text=option_text,
                argument="OpenAI not installed",
                strength=0,
                is_correct=False,
            )

        options_text = "\n".join(f"{k}) {v}" for k, v in options.items())

        prompt = DEFENDER_PROMPT.format(
            question_text=question_text,
            options_text=options_text,
            option_key=option_key,
            option_text=option_text,
        )

        try:
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,  # Some creativity for finding angles
            )

            result = json.loads(response.choices[0].message.content)

            return DefenseResult(
                option_key=option_key,
                option_text=option_text,
                argument=result.get("argument", ""),
                strength=int(result.get("strength", 1)),
                angle=result.get("angle", ""),
                is_correct=False,
            )

        except Exception as e:
            logger.warning(f"Defense failed for option {option_key}: {e}")
            return DefenseResult(
                option_key=option_key,
                option_text=option_text,
                argument=f"Analysis failed: {e}",
                strength=0,
                is_correct=False,
            )
