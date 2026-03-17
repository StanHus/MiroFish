"""
Answer Agent - Selects an answer using ONLY constrained knowledge.

KEY INSIGHT: Instead of asking LLM to "not know" the answer,
we ask it to pick what a student with specific misconceptions would choose.

Two modes:
1. CONSTRAINED (default) - Try to answer with limited knowledge
2. MISCONCEPTION-DRIVEN - Explicitly pick based on misconceptions (more reliable)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from openai import AsyncOpenAI

from .knowledge_agent import KnowledgeProfile
from .perception_agent import PerceptionResult


@dataclass
class AnswerResult:
    """The student's answer attempt."""

    selected: str  # "A", "B", "C", "D"
    selected_text: str
    confidence: str  # "certain", "pretty_sure", "guessing", "no_idea"

    # Reasoning chain
    thought_process: str
    knowledge_applied: List[str]
    knowledge_gaps_felt: List[str]
    elimination_reasoning: Optional[str]

    # Why this answer
    why_selected: str
    why_not_others: Dict[str, str]  # Why each other option was rejected

    # Mode used
    mode: str  # "constrained" or "misconception_driven"
    misconception_used: Optional[str]  # If misconception-driven, which one

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "selected_text": self.selected_text,
            "confidence": self.confidence,
            "thought_process": self.thought_process,
            "knowledge_applied": self.knowledge_applied,
            "knowledge_gaps_felt": self.knowledge_gaps_felt,
            "elimination_reasoning": self.elimination_reasoning,
            "why_selected": self.why_selected,
            "why_not_others": self.why_not_others,
            "mode": self.mode,
            "misconception_used": self.misconception_used,
        }


class AnswerAgent:
    """
    Agent that answers questions as a simulated student.

    Two modes:
    - CONSTRAINED: Traditional approach, try to limit knowledge (LLM often cheats)
    - MISCONCEPTION_DRIVEN: Ask which answer a student with X misconception would pick
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        mode: Literal["constrained", "misconception_driven", "auto"] = "auto",
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.default_mode = mode
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def answer(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        perception: PerceptionResult,
        mode: Optional[Literal["constrained", "misconception_driven", "auto"]] = None,
    ) -> AnswerResult:
        """
        Generate an answer using the specified mode.

        Args:
            question: The question
            knowledge: What the student knows
            perception: How they read the question
            mode: Override default mode

        Returns:
            AnswerResult with selection and reasoning
        """
        if not self.client:
            return self._fallback_answer(question, knowledge)

        use_mode = mode or self.default_mode

        # Auto mode: Use Knowledge Agent's assessment of whether student can answer
        if use_mode == "auto":
            # The Knowledge Agent has already assessed if this student can answer correctly
            if not knowledge.can_answer_correctly:
                # Student lacks knowledge - use misconception-driven
                use_mode = "misconception_driven"
            else:
                # Student likely has the knowledge - use constrained
                use_mode = "constrained"

        if use_mode == "misconception_driven":
            return await self._answer_misconception_driven(question, knowledge, perception)
        else:
            return await self._answer_constrained(question, knowledge, perception)

    async def _answer_misconception_driven(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        perception: PerceptionResult,
    ) -> AnswerResult:
        """
        Pick the answer a student with these misconceptions would choose.

        This INVERTS the problem: instead of trying to not know the right answer,
        we explicitly ask which wrong answer fits the misconceptions.
        """
        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])
        options_text = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))

        # Build misconception context
        misconceptions = knowledge.misconceptions or ["General confusion about the topic"]
        unknown_concepts = knowledge.concepts_unknown or ["Details of this topic"]

        system_prompt = """You are an expert in student misconceptions and how they lead to wrong answers.

Your task: Determine which answer a student with specific misconceptions would choose.

IMPORTANT: You are NOT trying to find the correct answer. You are predicting which
answer a confused student would pick based on their misconceptions.

Students with misconceptions often:
- Pick answers that SOUND related to what they partially understand
- Choose options with familiar words even if wrong
- Fall for common traps based on their specific confusion
- Guess based on partial pattern matching

Be realistic. A student who thinks X will often pick Y because of that belief."""

        user_prompt = f"""STUDENT'S MISCONCEPTIONS:
{chr(10).join(f'- {m}' for m in misconceptions)}

CONCEPTS THEY DON'T UNDERSTAND:
{chr(10).join(f'- {c}' for c in unknown_concepts[:5])}

HOW THEY READ THE QUESTION:
{perception.interpretation or 'Partial understanding'}
Words they found confusing: {', '.join(perception.words_unclear[:5]) or 'several terms'}

QUESTION:
{question_text}

OPTIONS:
{options_text}

Based on their misconceptions, which answer would this student MOST LIKELY pick?

Think step by step:
1. What does each misconception suggest they might believe?
2. Which answer option aligns with their confused understanding?
3. What would attract their attention given partial comprehension?

Return JSON:
{{
    "analysis": "How their misconceptions lead to a specific answer",
    "misconception_applied": "Which misconception most influenced the choice",
    "selected": "A" or "B" or "C" or "D",
    "confidence": "certain" or "pretty_sure" or "guessing" or "no_idea",
    "why_this_answer": "Why this answer fits their misconceptions",
    "thought_process": "What the student might be thinking (in their voice)"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=600,
            )

            result_text = response.choices[0].message.content.strip()

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            data = json.loads(result_text)

            selected = data.get("selected", "A").upper()
            selected_idx = ord(selected) - ord("A")
            if 0 <= selected_idx < len(options):
                selected_text = str(options[selected_idx])
            else:
                selected_text = str(options[0]) if options else ""
                selected = "A"

            return AnswerResult(
                selected=selected,
                selected_text=selected_text,
                confidence=data.get("confidence", "guessing"),
                thought_process=data.get("thought_process", ""),
                knowledge_applied=[],
                knowledge_gaps_felt=unknown_concepts[:3],
                elimination_reasoning=None,
                why_selected=data.get("why_this_answer", ""),
                why_not_others={},
                mode="misconception_driven",
                misconception_used=data.get("misconception_applied"),
            )

        except Exception as e:
            return self._fallback_answer(question, knowledge, str(e))

    async def _answer_constrained(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        perception: PerceptionResult,
    ) -> AnswerResult:
        """Original constrained answering approach."""
        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        knowledge_constraint = knowledge.to_constraint_prompt()
        perception_context = perception.to_prompt_section()
        options_text = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))

        system_prompt = f"""You are a grade {knowledge.grade} student with the "{knowledge.archetype}" personality.

CRITICAL RULES:
1. You can ONLY use the knowledge provided below
2. You DO NOT have access to any other information
3. If something is listed as "unknown" to you, you genuinely DON'T KNOW IT
4. You must reason like a real student at this grade level
5. It's OK to guess if you're unsure - that's realistic

{knowledge_constraint}

{perception_context}"""

        user_prompt = f"""QUESTION:
{question_text}

OPTIONS:
{options_text}

Think through this as a grade {knowledge.grade} {knowledge.archetype}:
1. What do I understand about this question?
2. What knowledge do I have?
3. Which option seems right based on what I know?

Return JSON:
{{
    "thought_process": "Your thinking as this student",
    "knowledge_applied": ["facts you used"],
    "knowledge_gaps_felt": ["things you didn't know"],
    "elimination_reasoning": "How you eliminated options",
    "selected": "A" or "B" or "C" or "D",
    "confidence": "certain" or "pretty_sure" or "guessing" or "no_idea",
    "why_selected": "Why you picked this",
    "why_not_others": {{"A": "reason", "B": "reason", ...}}
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )

            result_text = response.choices[0].message.content.strip()

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            data = json.loads(result_text)

            selected = data.get("selected", "A").upper()
            selected_idx = ord(selected) - ord("A")
            if 0 <= selected_idx < len(options):
                selected_text = str(options[selected_idx])
            else:
                selected_text = str(options[0]) if options else ""
                selected = "A"

            return AnswerResult(
                selected=selected,
                selected_text=selected_text,
                confidence=data.get("confidence", "guessing"),
                thought_process=data.get("thought_process", ""),
                knowledge_applied=data.get("knowledge_applied", []),
                knowledge_gaps_felt=data.get("knowledge_gaps_felt", []),
                elimination_reasoning=data.get("elimination_reasoning"),
                why_selected=data.get("why_selected", ""),
                why_not_others=data.get("why_not_others", {}),
                mode="constrained",
                misconception_used=None,
            )

        except Exception as e:
            return self._fallback_answer(question, knowledge, str(e))

    def _fallback_answer(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        error: str = ""
    ) -> AnswerResult:
        """Random fallback when API unavailable."""
        import random
        options = question.get("options", ["A", "B", "C", "D"])
        selected_idx = random.randint(0, len(options) - 1)
        selected = chr(65 + selected_idx)

        return AnswerResult(
            selected=selected,
            selected_text=str(options[selected_idx]),
            confidence="no_idea",
            thought_process=f"Fallback: Random guess. {error}",
            knowledge_applied=[],
            knowledge_gaps_felt=knowledge.concepts_unknown[:3],
            elimination_reasoning=None,
            why_selected="Random guess (API unavailable)",
            why_not_others={},
            mode="fallback",
            misconception_used=None,
        )
