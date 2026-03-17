"""
Student Model Agent - Models what a student believes (including misconceptions).

This agent doesn't try to "not know" things. Instead, it models what beliefs
and misconceptions a student at a given grade/archetype would have.

The key insight: We're modeling BELIEFS, not trying to suppress KNOWLEDGE.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


@dataclass
class StudentModel:
    """Model of a student's beliefs and misconceptions."""

    grade: int
    archetype: str

    # What they believe (may be wrong)
    beliefs: List[str]
    misconceptions: List[str]

    # Knowledge assessment
    topic_familiarity: float  # 0-1 how familiar with this topic
    confidence_level: str  # "confident", "uncertain", "guessing"

    # Behavior tendencies
    guesses_when_unsure: bool
    uses_elimination: bool
    attracted_to_familiar_words: bool
    second_guesses_self: bool

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "archetype": self.archetype,
            "beliefs": self.beliefs,
            "misconceptions": self.misconceptions,
            "topic_familiarity": self.topic_familiarity,
            "confidence_level": self.confidence_level,
            "guesses_when_unsure": self.guesses_when_unsure,
            "uses_elimination": self.uses_elimination,
            "attracted_to_familiar_words": self.attracted_to_familiar_words,
            "second_guesses_self": self.second_guesses_self,
        }


class StudentModelAgent:
    """
    Agent that models a student's beliefs and misconceptions about a topic.

    This is different from trying to constrain an LLM's knowledge.
    We're explicitly asking: "What would a grade X student BELIEVE about this topic?"
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

    async def model_student(
        self,
        question: Dict[str, Any],
        grade: int,
        archetype: str,
    ) -> StudentModel:
        """
        Model a student's beliefs and misconceptions about a topic.

        Args:
            question: The question context
            grade: Student grade (1-12)
            archetype: Student type

        Returns:
            StudentModel with beliefs and misconceptions
        """
        if not self.client:
            return self._fallback_model(grade, archetype)

        question_text = question.get("text") or question.get("question", "")
        subject = question.get("subject", "general")

        system_prompt = """You are an expert in child development and educational psychology.

Your task: Model what a student at a specific grade level BELIEVES about a topic.
This includes both correct beliefs AND misconceptions.

Important: We're not asking what they DON'T know - we're asking what they BELIEVE.
A student might believe something incorrect with confidence.

Consider:
1. What is typically taught at this grade level?
2. What common misconceptions exist at this level?
3. How does the archetype affect their beliefs and behavior?

Be specific about misconceptions - these are POSITIVE beliefs that are WRONG,
not just absence of knowledge."""

        user_prompt = f"""TOPIC/QUESTION:
{question_text}

STUDENT:
Grade: {grade}
Archetype: {archetype}

ARCHETYPE EFFECTS:
- honors_overachiever: More likely to have correct beliefs, reads ahead, high confidence
- debate_club_kid: May have advanced beliefs in areas of interest, argumentative
- quiet_thinker: May second-guess correct beliefs, uncertain
- disengaged_but_smart: Sporadic beliefs, may know but not care
- esl_student: Core concepts solid, may misunderstand word-based questions
- class_clown: Few beliefs, guesses based on pattern matching
- average_student: Has typical taught beliefs, some misconceptions

What would this student BELIEVE about this topic?

Return JSON:
{{
    "beliefs": ["Things they believe (correct or incorrect)"],
    "misconceptions": ["Specifically WRONG beliefs they might have"],
    "topic_familiarity": 0.0-1.0,
    "confidence_level": "confident" or "uncertain" or "guessing",
    "guesses_when_unsure": true/false,
    "uses_elimination": true/false,
    "attracted_to_familiar_words": true/false,
    "second_guesses_self": true/false
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

            return StudentModel(
                grade=grade,
                archetype=archetype,
                beliefs=data.get("beliefs", []),
                misconceptions=data.get("misconceptions", []),
                topic_familiarity=float(data.get("topic_familiarity", 0.5)),
                confidence_level=data.get("confidence_level", "uncertain"),
                guesses_when_unsure=data.get("guesses_when_unsure", True),
                uses_elimination=data.get("uses_elimination", False),
                attracted_to_familiar_words=data.get("attracted_to_familiar_words", True),
                second_guesses_self=data.get("second_guesses_self", False),
            )

        except Exception as e:
            return self._fallback_model(grade, archetype, str(e))

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

    def _fallback_model(
        self,
        grade: int,
        archetype: str,
        error: str = ""
    ) -> StudentModel:
        """Fallback when API unavailable."""
        return StudentModel(
            grade=grade,
            archetype=archetype,
            beliefs=["Basic topic knowledge"],
            misconceptions=["May have typical misconceptions"],
            topic_familiarity=0.3,
            confidence_level="guessing",
            guesses_when_unsure=True,
            uses_elimination=False,
            attracted_to_familiar_words=True,
            second_guesses_self=False,
        )
