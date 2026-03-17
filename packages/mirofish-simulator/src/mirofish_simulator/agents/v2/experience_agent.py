"""
Experience Agent - Assesses a student's experience with a question.

Evaluates three dimensions:
1. UNDERSTAND - Can they parse the question correctly?
2. ENGAGED - Does it feel relevant/interesting to them?
3. LEARNS_FROM - Would answering this help them learn?

This provides insight into question quality from the student's perspective,
beyond just whether they get the answer right or wrong.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


@dataclass
class ExperienceResult:
    """A student's experience with a question."""

    grade: int
    archetype: str

    # The three core dimensions
    understands: bool
    understands_reason: str

    engaged: bool
    engaged_reason: str

    learns_from: bool
    learns_from_reason: str

    # Overall assessment
    overall_experience: str  # "positive", "neutral", "negative"
    recommendation: Optional[str]  # How to improve for this student type

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "archetype": self.archetype,
            "understands": self.understands,
            "understands_reason": self.understands_reason,
            "engaged": self.engaged,
            "engaged_reason": self.engaged_reason,
            "learns_from": self.learns_from,
            "learns_from_reason": self.learns_from_reason,
            "overall_experience": self.overall_experience,
            "recommendation": self.recommendation,
        }

    def passes(self) -> bool:
        """Returns True if this student has a positive experience."""
        return self.understands and self.engaged and self.learns_from


class ExperienceAgent:
    """
    Assesses how a student would EXPERIENCE a question.

    This is different from whether they answer correctly.
    A student might get a question right but:
    - Not understand what it was really asking
    - Find it boring/irrelevant
    - Not learn anything from it

    Or they might get it wrong but:
    - Understand the question perfectly
    - Find it engaging
    - Learn from the attempt
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

    async def assess(
        self,
        question: Dict[str, Any],
        grade: int,
        archetype: str,
    ) -> ExperienceResult:
        """
        Assess how a student would experience this question.

        Args:
            question: The question with text and options
            grade: Student grade (1-12)
            archetype: Student type (esl_student, class_clown, etc.)

        Returns:
            ExperienceResult with understand/engaged/learns_from assessments
        """
        if not self.client:
            return self._fallback(grade, archetype)

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])
        options_text = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))

        system_prompt = """You are an expert in educational psychology and child development.

Your task: Assess how a SPECIFIC student would EXPERIENCE a question.
Not whether they'd answer correctly, but whether the question WORKS for them.

THREE DIMENSIONS TO ASSESS:

1. UNDERSTANDS - Can they parse what the question is asking?
   - Is the vocabulary appropriate for their level?
   - Is the sentence structure clear?
   - Are there cultural references they might not get?
   - For ESL: Are there idioms or complex phrases?

2. ENGAGED - Would they find this question worth their attention?
   - Does the topic/framing interest this type of student?
   - Is it too easy (boring) or too hard (frustrating)?
   - Does it feel relevant to their world?
   - For class_clown: Is there anything to keep attention?
   - For honors: Is it intellectually stimulating?

3. LEARNS_FROM - Would attempting this question help them learn?
   - Does it connect to something they know?
   - Would getting it wrong teach them something?
   - Does it stretch their thinking appropriately?
   - Is the challenge level right for growth?

BE SPECIFIC to the archetype. A question that works for one student may fail for another."""

        user_prompt = f"""QUESTION:
{question_text}

OPTIONS:
{options_text}

STUDENT:
Grade: {grade}
Archetype: {archetype}

ARCHETYPE CHARACTERISTICS:
- honors_overachiever: Seeks challenge, bored by easy questions, wants depth
- debate_club_kid: Loves argumentation, engaged by controversy, questions premises
- quiet_thinker: Needs time, intimidated by complex wording, thoughtful
- disengaged_but_smart: Checks out if bored, needs hook, capable but unmotivated
- esl_student: Strong concepts, struggles with complex English, cultural gaps
- class_clown: Short attention span, needs novelty, avoids "boring" work
- average_student: Wants clear expectations, moderate challenge, practical relevance
- test_anxious: Freezes on hard questions, needs confidence builders, clear structure

Assess this student's experience with this question.

Return JSON:
{{
    "understands": true/false,
    "understands_reason": "Specific reason - what works or doesn't for this student",
    "engaged": true/false,
    "engaged_reason": "Specific reason - what catches or loses their attention",
    "learns_from": true/false,
    "learns_from_reason": "Specific reason - what they'd gain or why it's wasted",
    "overall_experience": "positive" or "neutral" or "negative",
    "recommendation": "How to improve this question for this student type (or null if good)"
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

            return ExperienceResult(
                grade=grade,
                archetype=archetype,
                understands=data.get("understands", True),
                understands_reason=data.get("understands_reason", ""),
                engaged=data.get("engaged", True),
                engaged_reason=data.get("engaged_reason", ""),
                learns_from=data.get("learns_from", True),
                learns_from_reason=data.get("learns_from_reason", ""),
                overall_experience=data.get("overall_experience", "neutral"),
                recommendation=data.get("recommendation"),
            )

        except Exception as e:
            return self._fallback(grade, archetype, str(e))

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

    def _fallback(
        self,
        grade: int,
        archetype: str,
        error: str = ""
    ) -> ExperienceResult:
        """Fallback when API unavailable."""
        return ExperienceResult(
            grade=grade,
            archetype=archetype,
            understands=True,
            understands_reason=f"Unable to assess: {error}" if error else "Assumed understandable",
            engaged=True,
            engaged_reason="Unable to assess engagement",
            learns_from=True,
            learns_from_reason="Unable to assess learning value",
            overall_experience="neutral",
            recommendation=None,
        )

    async def assess_batch(
        self,
        question: Dict[str, Any],
        students: list[Dict[str, Any]],
    ) -> list[ExperienceResult]:
        """
        Assess a question across multiple student archetypes in parallel.

        Args:
            question: The question
            students: List of {"grade": int, "archetype": str}

        Returns:
            List of ExperienceResults
        """
        import asyncio

        tasks = [
            self.assess(question, s["grade"], s["archetype"])
            for s in students
        ]
        return await asyncio.gather(*tasks)


# Default diverse student population for experience assessment
DIVERSE_STUDENTS = [
    {"grade": 5, "archetype": "esl_student"},
    {"grade": 6, "archetype": "class_clown"},
    {"grade": 7, "archetype": "test_anxious"},
    {"grade": 8, "archetype": "quiet_thinker"},
    {"grade": 9, "archetype": "disengaged_but_smart"},
    {"grade": 10, "archetype": "average_student"},
    {"grade": 11, "archetype": "debate_club_kid"},
    {"grade": 12, "archetype": "honors_overachiever"},
]


@dataclass
class ExperienceAssessment:
    """Aggregated experience assessment across student types."""

    question_text: str
    results: list[ExperienceResult]

    # Aggregated metrics
    understand_rate: float  # % of students who understand
    engage_rate: float      # % of students who are engaged
    learn_rate: float       # % of students who learn from it

    # Failing segments
    fails_for: list[str]    # Archetypes that have negative experience
    recommendations: list[str]  # Unique recommendations

    def to_dict(self) -> dict:
        return {
            "understand_rate": round(self.understand_rate, 2),
            "engage_rate": round(self.engage_rate, 2),
            "learn_rate": round(self.learn_rate, 2),
            "fails_for": self.fails_for,
            "recommendations": self.recommendations,
            "details": [r.to_dict() for r in self.results],
        }

    def passes(self) -> bool:
        """Returns True if question works for all student segments."""
        return len(self.fails_for) == 0


async def assess_question_experience(
    question: Dict[str, Any],
    students: list[Dict[str, Any]] = None,
    api_key: str = None,
) -> ExperienceAssessment:
    """
    Assess how a question is experienced by diverse students.

    Args:
        question: Question with text and options
        students: Student archetypes to test (defaults to DIVERSE_STUDENTS)
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)

    Returns:
        ExperienceAssessment with aggregated results
    """
    students = students or DIVERSE_STUDENTS
    agent = ExperienceAgent(api_key=api_key)

    results = await agent.assess_batch(question, students)

    # Aggregate
    understand_rate = sum(1 for r in results if r.understands) / len(results)
    engage_rate = sum(1 for r in results if r.engaged) / len(results)
    learn_rate = sum(1 for r in results if r.learns_from) / len(results)

    fails_for = [r.archetype for r in results if not r.passes()]
    recommendations = list(set(r.recommendation for r in results if r.recommendation))

    question_text = question.get("text") or question.get("question", "")

    return ExperienceAssessment(
        question_text=question_text,
        results=results,
        understand_rate=understand_rate,
        engage_rate=engage_rate,
        learn_rate=learn_rate,
        fails_for=fails_for,
        recommendations=recommendations,
    )
