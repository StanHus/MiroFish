"""
Perception Agent - How does the student perceive/read the question?

This agent transforms the question into what the student actually "sees"
when reading it, based on their reading level and vocabulary.

NO HARDCODED WORD LISTS. The agent reasons dynamically.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .knowledge_agent import KnowledgeProfile


@dataclass
class PerceptionResult:
    """How a student perceives the question."""

    # What they actually "see"
    perceived_question: str  # Question with unclear parts marked
    perceived_options: List[str]  # Options with unclear parts marked

    # Comprehension
    overall_comprehension: float  # 0-1 scale
    words_unclear: List[str]
    phrases_unclear: List[str]

    # What they focus on
    key_words_noticed: List[str]
    interpretation: str  # What they think the question is asking

    # Distractions/misreadings
    potential_misreadings: List[str]

    def to_dict(self) -> dict:
        return {
            "perceived_question": self.perceived_question,
            "perceived_options": self.perceived_options,
            "overall_comprehension": self.overall_comprehension,
            "words_unclear": self.words_unclear,
            "phrases_unclear": self.phrases_unclear,
            "key_words_noticed": self.key_words_noticed,
            "interpretation": self.interpretation,
            "potential_misreadings": self.potential_misreadings,
        }

    def to_prompt_section(self) -> str:
        """Convert to prompt section for answer agent."""
        unclear = ", ".join(self.words_unclear[:5]) if self.words_unclear else "none"

        return f"""=== HOW YOU READ THE QUESTION ===
What you see: {self.perceived_question}

Words that confuse you: {unclear}

What you THINK the question is asking: {self.interpretation}

Comprehension level: {self.overall_comprehension:.0%}
"""


class PerceptionAgent:
    """
    Agent that determines how a student perceives/reads a question.

    Uses the knowledge profile to filter what words they understand
    and how they interpret the question.
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

    async def perceive(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
    ) -> PerceptionResult:
        """
        Generate perception of the question from student's perspective.

        Args:
            question: The question with text and options
            knowledge: Knowledge profile from KnowledgeAgent

        Returns:
            PerceptionResult showing how student reads the question
        """
        if not self.client:
            return self._fallback_perception(question, knowledge)

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        system_prompt = """You are simulating how a student READS and PERCEIVES a test question.

Your task: Show what the student actually "sees" when reading, based on their grade level and vocabulary.

Consider:
1. VOCABULARY: Words they don't know become unclear/confusing
2. SENTENCE STRUCTURE: Complex sentences may be misread
3. FOCUS: What words/phrases catch their attention?
4. INTERPRETATION: What do they THINK the question is asking?

Mark unclear words with [???] in the perceived text.
Be realistic about reading comprehension at this grade level."""

        user_prompt = f"""STUDENT PROFILE:
Grade: {knowledge.grade}
Archetype: {knowledge.archetype}
Vocabulary they struggle with: {', '.join(knowledge.vocabulary_unfamiliar[:10])}

QUESTION TO READ:
{question_text}

OPTIONS:
{chr(10).join(f'{chr(65+i)}) {opt}' for i, opt in enumerate(options))}

Simulate how this student reads and perceives this question.

Return JSON:
{{
    "perceived_question": "Question with [???] for unclear parts",
    "perceived_options": ["Option A with [???]", "Option B...", ...],
    "overall_comprehension": 0.0-1.0,
    "words_unclear": ["word1", "word2"],
    "phrases_unclear": ["phrase1"],
    "key_words_noticed": ["words that stand out to them"],
    "interpretation": "What they THINK the question is asking (in simple words)",
    "potential_misreadings": ["ways they might misread/misunderstand"]
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

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            data = json.loads(result_text)

            return PerceptionResult(
                perceived_question=data.get("perceived_question", question_text),
                perceived_options=data.get("perceived_options", options),
                overall_comprehension=float(data.get("overall_comprehension", 0.5)),
                words_unclear=data.get("words_unclear", []),
                phrases_unclear=data.get("phrases_unclear", []),
                key_words_noticed=data.get("key_words_noticed", []),
                interpretation=data.get("interpretation", ""),
                potential_misreadings=data.get("potential_misreadings", []),
            )

        except Exception as e:
            return self._fallback_perception(question, knowledge, str(e))

    def _fallback_perception(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        error: str = ""
    ) -> PerceptionResult:
        """Fallback perception when API unavailable."""
        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        return PerceptionResult(
            perceived_question=question_text,
            perceived_options=list(options),
            overall_comprehension=0.5,
            words_unclear=knowledge.vocabulary_unfamiliar[:5],
            phrases_unclear=[],
            key_words_noticed=[],
            interpretation=f"Fallback: Basic reading. {error}",
            potential_misreadings=[],
        )
