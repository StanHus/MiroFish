"""
Distractor Agent - Analyzes what misconceptions lead to each wrong answer.

This agent maps: Misconception → Distractor

For each wrong answer option, it identifies what belief/error would lead
a student to choose it. This creates a misconception→answer mapping.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


@dataclass
class DistractorMapping:
    """Maps each answer option to the misconception that leads to it."""

    option: str  # "A", "B", "C", "D"
    option_text: str
    is_correct: bool
    leads_from_misconception: Optional[str]  # None if correct
    why_attractive: str  # Why a confused student might pick this
    grade_level_appeal: Dict[str, float]  # {"elementary": 0.8, "middle": 0.5, "high": 0.2}


@dataclass
class DistractorAnalysis:
    """Complete analysis of how distractors map to misconceptions."""

    question_text: str
    correct_option: str
    mappings: List[DistractorMapping]
    key_knowledge_required: str  # What you need to know to get it right

    def get_misconception_answer(self, misconception: str) -> Optional[str]:
        """Given a misconception, which answer would the student pick?"""
        for mapping in self.mappings:
            if mapping.leads_from_misconception and misconception.lower() in mapping.leads_from_misconception.lower():
                return mapping.option
        return None

    def get_likely_wrong_answer(self, grade_level: str) -> str:
        """Get the most likely wrong answer for a grade level."""
        best_option = None
        best_appeal = 0.0

        for mapping in self.mappings:
            if not mapping.is_correct:
                appeal = mapping.grade_level_appeal.get(grade_level, 0.5)
                if appeal > best_appeal:
                    best_appeal = appeal
                    best_option = mapping.option

        return best_option or "A"

    def to_dict(self) -> dict:
        return {
            "question_text": self.question_text,
            "correct_option": self.correct_option,
            "key_knowledge_required": self.key_knowledge_required,
            "mappings": [
                {
                    "option": m.option,
                    "option_text": m.option_text,
                    "is_correct": m.is_correct,
                    "leads_from_misconception": m.leads_from_misconception,
                    "why_attractive": m.why_attractive,
                    "grade_level_appeal": m.grade_level_appeal,
                }
                for m in self.mappings
            ],
        }


class DistractorAgent:
    """
    Analyzes question distractors to map misconceptions to answers.

    This is the key insight: Instead of trying to make an LLM "not know" things,
    we analyze WHAT ERRORS lead to WHICH ANSWERS, then match student errors to answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def analyze(
        self,
        question: Dict[str, Any],
        correct_answer: str,
    ) -> DistractorAnalysis:
        """
        Analyze the distractors in a multiple choice question.

        Args:
            question: Question with "text" and "options"
            correct_answer: The correct option ("A", "B", "C", or "D")

        Returns:
            DistractorAnalysis mapping misconceptions to answers
        """
        if not self.client:
            return self._fallback_analysis(question, correct_answer)

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        options_text = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))

        system_prompt = """You are an expert in educational assessment and student misconceptions.

Your task: Analyze a multiple choice question to understand what misconception or error
leads a student to choose each WRONG answer.

This is NOT about what's correct - it's about understanding WHY students make specific errors.

For each distractor (wrong answer), identify:
1. What misconception or misunderstanding leads to this choice
2. Why it might be attractive to a confused student
3. Which grade levels are most susceptible (elementary/middle/high school)

Be specific about misconceptions - not just "doesn't know the answer" but exactly
what wrong belief leads to this specific wrong answer."""

        user_prompt = f"""QUESTION:
{question_text}

OPTIONS:
{options_text}

CORRECT ANSWER: {correct_answer}

Analyze each option. For the WRONG answers, identify what specific misconception
or error leads a student to choose it.

Return JSON:
{{
    "key_knowledge_required": "What specific knowledge is needed to answer correctly",
    "mappings": [
        {{
            "option": "A",
            "is_correct": false,
            "leads_from_misconception": "Specific misconception that leads to this answer",
            "why_attractive": "Why a confused student might pick this",
            "grade_level_appeal": {{"elementary": 0.0-1.0, "middle": 0.0-1.0, "high": 0.0-1.0}}
        }},
        {{
            "option": "B",
            "is_correct": true,
            "leads_from_misconception": null,
            "why_attractive": "Correct answer - requires actual knowledge",
            "grade_level_appeal": {{"elementary": 0.2, "middle": 0.5, "high": 0.8}}
        }},
        ...for all options
    ]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content.strip()
            data = self._parse_json(result_text)

            mappings = []
            for m in data.get("mappings", []):
                option = m.get("option", "A")
                option_idx = ord(option.upper()) - ord("A")
                option_text = str(options[option_idx]) if 0 <= option_idx < len(options) else ""

                mappings.append(DistractorMapping(
                    option=option.upper(),
                    option_text=option_text,
                    is_correct=m.get("is_correct", False),
                    leads_from_misconception=m.get("leads_from_misconception"),
                    why_attractive=m.get("why_attractive", ""),
                    grade_level_appeal=m.get("grade_level_appeal", {"elementary": 0.5, "middle": 0.5, "high": 0.5}),
                ))

            return DistractorAnalysis(
                question_text=question_text,
                correct_option=correct_answer.upper(),
                mappings=mappings,
                key_knowledge_required=data.get("key_knowledge_required", ""),
            )

        except Exception as e:
            return self._fallback_analysis(question, correct_answer, str(e))

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        import re

        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                if part.startswith("json"):
                    part = part[4:]
                part = part.strip()
                if part.startswith("{"):
                    text = part.split("```")[0] if "```" in part else part
                    break

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group()

        text = re.sub(r',(\s*[}\]])', r'\1', text)

        return json.loads(text)

    def _fallback_analysis(
        self,
        question: Dict[str, Any],
        correct_answer: str,
        error: str = ""
    ) -> DistractorAnalysis:
        """Fallback when API unavailable."""
        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        mappings = []
        for i, opt in enumerate(options):
            option = chr(65 + i)
            is_correct = option == correct_answer.upper()
            mappings.append(DistractorMapping(
                option=option,
                option_text=str(opt),
                is_correct=is_correct,
                leads_from_misconception=None if is_correct else "General confusion",
                why_attractive="Fallback mode" if not is_correct else "Correct answer",
                grade_level_appeal={"elementary": 0.5, "middle": 0.5, "high": 0.5},
            ))

        return DistractorAnalysis(
            question_text=question_text,
            correct_option=correct_answer.upper(),
            mappings=mappings,
            key_knowledge_required=f"Fallback: {error}",
        )
