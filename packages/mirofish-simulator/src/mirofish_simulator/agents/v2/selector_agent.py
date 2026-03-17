"""
Selector Agent - Matches student misconceptions to answer choices.

This is the key agent that produces the final answer.

It takes:
1. Distractor analysis (what misconception → what answer)
2. Student model (what misconceptions this student has)

And determines: Given this student's misconceptions, which answer do they pick?

This approach works because we're not asking an LLM to "not know" things.
We're doing a MATCHING operation: student misconceptions ↔ distractor misconceptions
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .distractor_agent import DistractorAnalysis, DistractorMapping
from .student_model_agent import StudentModel


@dataclass
class SelectionResult:
    """The result of answer selection."""

    selected: str  # "A", "B", "C", "D"
    selected_text: str
    is_correct: bool

    # How it was selected
    selection_reason: str
    misconception_matched: Optional[str]
    confidence: str

    # Debug info
    consideration_process: str
    eliminated_options: Dict[str, str]

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "selected_text": self.selected_text,
            "is_correct": self.is_correct,
            "selection_reason": self.selection_reason,
            "misconception_matched": self.misconception_matched,
            "confidence": self.confidence,
            "consideration_process": self.consideration_process,
            "eliminated_options": self.eliminated_options,
        }


class SelectorAgent:
    """
    Matches student misconceptions to answer choices.

    The key insight: This agent doesn't need to "not know" things.
    It just matches the student's misconceptions to the distractors
    that were designed to catch those misconceptions.
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

    async def select(
        self,
        question: Dict[str, Any],
        distractors: DistractorAnalysis,
        student: StudentModel,
    ) -> SelectionResult:
        """
        Select an answer by matching student misconceptions to distractors.

        Args:
            question: The question
            distractors: Analysis of what misconceptions lead to each answer
            student: Model of the student's beliefs/misconceptions

        Returns:
            SelectionResult with the selected answer
        """
        if not self.client:
            return self._fallback_selection(question, distractors, student)

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        # Build context for the selector
        distractor_info = "\n".join([
            f"{m.option}) {m.option_text}\n   Misconception that leads here: {m.leads_from_misconception or 'CORRECT ANSWER'}"
            for m in distractors.mappings
        ])

        student_info = f"""STUDENT PROFILE:
Grade: {student.grade}
Archetype: {student.archetype}
Topic familiarity: {student.topic_familiarity:.0%}
Confidence level: {student.confidence_level}

STUDENT'S BELIEFS:
{chr(10).join(f'- {b}' for b in student.beliefs[:5])}

STUDENT'S MISCONCEPTIONS:
{chr(10).join(f'- {m}' for m in student.misconceptions[:5])}

BEHAVIOR:
- Guesses when unsure: {student.guesses_when_unsure}
- Uses elimination: {student.uses_elimination}
- Attracted to familiar words: {student.attracted_to_familiar_words}
- Second-guesses self: {student.second_guesses_self}"""

        system_prompt = """You are simulating how a student would answer a multiple choice question.

Your job: Determine what THIS SPECIFIC STUDENT would pick based on their beliefs AND misconceptions.

CRITICAL RULES (in order of priority):

1. DISTINGUISH EXACT vs VAGUE BELIEFS:
   - EXACT belief: Explicitly states the answer ("270 electoral votes needed", "Congress makes laws")
   - VAGUE belief: Related but not specific ("need a majority to win", "some branch makes laws")

   For FACTUAL questions (numbers, dates, specific terms):
   - Only EXACT beliefs count as "knowing the answer"
   - Vague beliefs mean they DON'T actually know → check misconceptions or guess

2. If student has an EXACT correct belief → they pick CORRECT
   Example: Belief "Congress makes laws" → pick "Legislative" ✓
   Example: Belief "270 electoral votes needed" → pick "270" ✓

3. If student has VAGUE belief + relevant misconception → misconception often wins
   Example: Vague belief "need a majority" + misconception about numbers → pick wrong number

4. If student has misconception matching a distractor AND no exact correct belief → pick distractor

5. If no clear match:
   - High familiarity (>60%) on CONCEPTUAL questions → educated guess toward correct
   - High familiarity on FACTUAL questions with numbers/specifics → still often guess wrong
   - Low familiarity → random guess or familiar-sounding words

KEY: A student who vaguely "knows about" a topic but doesn't know SPECIFIC FACTS will guess wrong on factual questions."""

        user_prompt = f"""QUESTION:
{question_text}

ANSWER OPTIONS AND WHAT MISCONCEPTIONS LEAD TO THEM:
{distractor_info}

{student_info}

CORRECT ANSWER: {distractors.correct_option}

Given this student's specific beliefs and misconceptions, which answer would they pick?

DECISION PROCESS (follow in order):

FIRST: Is this a FACTUAL question (specific numbers, dates, names) or CONCEPTUAL (who/what/why)?
- Factual: "How many electoral votes?" "What year?" "What is 1/2 + 1/4?"
- Conceptual: "Who makes laws?" "What branch enforces laws?"

FOR CONCEPTUAL QUESTIONS:
1. If belief directly answers ("Congress makes laws" → question asks who makes laws) → CORRECT
2. If misconception matches distractor → pick distractor
3. High familiarity + confident → trust belief

FOR FACTUAL QUESTIONS (numbers, dates, specific calculations):
1. Only EXACT knowledge counts ("270 electoral votes", "3/4")
2. Vague knowledge ("need a majority", "add fractions") is NOT enough for specific answers
3. Without exact knowledge → check misconceptions → likely guess wrong

EXAMPLES:
- Q: "Who heads the Executive?" + Belief: "President heads Executive" → CORRECT (conceptual, exact match)
- Q: "How many electoral votes?" + Belief: "Need a majority" → NOT enough (factual, vague)
- Q: "How many electoral votes?" + Belief: "270 votes needed" → CORRECT (factual, exact)

CRITICAL: Your "selected" field MUST be consistent with your reasoning.
- If reasoning concludes "pick correct" → selected = "{distractors.correct_option}"
- If reasoning concludes "doesn't know the specific number" → pick a distractor or guess

Return JSON:
{{
    "consideration_process": "Step by step analysis",
    "belief_matches_correct": true/false,
    "misconception_matched": "The misconception (or null if picking correct)",
    "selected": "{distractors.correct_option}" if belief wins, else the distractor letter,
    "selection_reason": "Why they picked this specific answer",
    "confidence": "confident" or "uncertain" or "guessing",
    "eliminated_options": {{}}
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=600,
            )

            result_text = response.choices[0].message.content.strip()
            data = self._parse_json(result_text)

            selected = data.get("selected", "A").upper()
            selected_idx = ord(selected) - ord("A")
            if 0 <= selected_idx < len(options):
                selected_text = str(options[selected_idx])
            else:
                selected_text = str(options[0]) if options else ""
                selected = "A"

            is_correct = selected == distractors.correct_option

            return SelectionResult(
                selected=selected,
                selected_text=selected_text,
                is_correct=is_correct,
                selection_reason=data.get("selection_reason", ""),
                misconception_matched=data.get("misconception_matched"),
                confidence=data.get("confidence", "uncertain"),
                consideration_process=data.get("consideration_process", ""),
                eliminated_options=data.get("eliminated_options", {}),
            )

        except Exception as e:
            return self._fallback_selection(question, distractors, student, str(e))

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

    def _fallback_selection(
        self,
        question: Dict[str, Any],
        distractors: DistractorAnalysis,
        student: StudentModel,
        error: str = ""
    ) -> SelectionResult:
        """Fallback - pick based on grade level appeal."""
        grade_level = "elementary" if student.grade <= 5 else "middle" if student.grade <= 8 else "high"
        selected = distractors.get_likely_wrong_answer(grade_level)

        options = question.get("options", [])
        selected_idx = ord(selected) - ord("A")
        selected_text = str(options[selected_idx]) if 0 <= selected_idx < len(options) else ""

        return SelectionResult(
            selected=selected,
            selected_text=selected_text,
            is_correct=selected == distractors.correct_option,
            selection_reason=f"Fallback: {error}",
            misconception_matched=None,
            confidence="guessing",
            consideration_process="Fallback mode",
            eliminated_options={},
        )
