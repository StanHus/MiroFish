"""
Verifier Agent - Catches "cheating" when LLM knows too much.

This agent reviews the answer and checks if it's consistent
with the stated knowledge constraints.

If the answer is "suspiciously correct" given the knowledge gaps,
it flags this and may request a re-answer.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .knowledge_agent import KnowledgeProfile
from .answer_agent import AnswerResult


@dataclass
class VerificationResult:
    """Result of verifying answer consistency."""

    is_consistent: bool
    confidence_score: float  # 0-1, how confident we are the answer is consistent

    # Analysis
    knowledge_used_valid: bool
    reasoning_matches_knowledge: bool
    suspiciously_correct: bool

    # Details
    issues_found: List[str]
    reasoning: str

    # Recommendation
    should_rerun: bool
    suggested_answer: Optional[str]  # If inconsistent, what should it be?

    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "confidence_score": self.confidence_score,
            "knowledge_used_valid": self.knowledge_used_valid,
            "reasoning_matches_knowledge": self.reasoning_matches_knowledge,
            "suspiciously_correct": self.suspiciously_correct,
            "issues_found": self.issues_found,
            "reasoning": self.reasoning,
            "should_rerun": self.should_rerun,
            "suggested_answer": self.suggested_answer,
        }


class VerifierAgent:
    """
    Agent that verifies answer consistency with knowledge constraints.

    This catches cases where the LLM "cheats" by using knowledge
    that the simulated student shouldn't have.
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

    async def verify(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        answer: AnswerResult,
        correct_answer: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify that the answer is consistent with knowledge constraints.

        Args:
            question: The original question
            knowledge: What the student was supposed to know
            answer: The answer given
            correct_answer: The actual correct answer (for detecting cheating)

        Returns:
            VerificationResult with consistency analysis
        """
        if not self.client:
            return self._fallback_verification(answer, correct_answer)

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        system_prompt = """You are a verification agent checking for consistency.

Your task: Determine if a student's answer is CONSISTENT with their stated knowledge.

A student who says they "don't know X" but then gives a correct answer that requires knowing X is INCONSISTENT (the LLM "cheated").

Look for:
1. Does the reasoning use knowledge they claimed not to have?
2. Is the answer suspiciously correct given their knowledge gaps?
3. Would a real student with this knowledge profile give this answer?

Be rigorous. LLMs often "leak" their actual knowledge despite constraints."""

        # Build context
        knowledge_summary = f"""STUDENT'S STATED KNOWLEDGE:
Known: {', '.join(knowledge.concepts_known[:5])}
Unknown: {', '.join(knowledge.concepts_unknown[:5])}
Misconceptions: {', '.join(knowledge.misconceptions[:3])}"""

        answer_summary = f"""STUDENT'S ANSWER:
Selected: {answer.selected}
Confidence: {answer.confidence}
Reasoning: {answer.thought_process}
Knowledge claimed to use: {', '.join(answer.knowledge_applied[:5])}"""

        correct_info = ""
        if correct_answer:
            is_correct = answer.selected.upper() == correct_answer.upper()
            correct_info = f"""
GROUND TRUTH:
Correct answer: {correct_answer}
Student got it: {"CORRECT" if is_correct else "WRONG"}

If the student got it CORRECT despite claiming not to know key concepts, this is suspicious."""

        user_prompt = f"""QUESTION:
{question_text}

OPTIONS:
{chr(10).join(f'{chr(65+i)}) {opt}' for i, opt in enumerate(options))}

{knowledge_summary}

{answer_summary}
{correct_info}

Analyze: Is this answer CONSISTENT with the stated knowledge constraints?

Return JSON:
{{
    "is_consistent": true/false,
    "confidence_score": 0.0-1.0,
    "knowledge_used_valid": true/false (did they only use knowledge they claimed to have?),
    "reasoning_matches_knowledge": true/false (does reasoning align with knowledge level?),
    "suspiciously_correct": true/false (did they get it right despite claiming ignorance?),
    "issues_found": ["specific inconsistencies found"],
    "reasoning": "Your analysis of consistency",
    "should_rerun": true/false (should we regenerate the answer?),
    "suggested_answer": "If inconsistent, what answer fits their knowledge?" or null
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # More deterministic verification
                max_tokens=600,
            )

            result_text = response.choices[0].message.content.strip()

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            data = json.loads(result_text)

            return VerificationResult(
                is_consistent=data.get("is_consistent", True),
                confidence_score=float(data.get("confidence_score", 0.5)),
                knowledge_used_valid=data.get("knowledge_used_valid", True),
                reasoning_matches_knowledge=data.get("reasoning_matches_knowledge", True),
                suspiciously_correct=data.get("suspiciously_correct", False),
                issues_found=data.get("issues_found", []),
                reasoning=data.get("reasoning", ""),
                should_rerun=data.get("should_rerun", False),
                suggested_answer=data.get("suggested_answer"),
            )

        except Exception as e:
            return self._fallback_verification(answer, correct_answer, str(e))

    def _fallback_verification(
        self,
        answer: AnswerResult,
        correct_answer: Optional[str] = None,
        error: str = ""
    ) -> VerificationResult:
        """Fallback when API unavailable."""
        return VerificationResult(
            is_consistent=True,
            confidence_score=0.5,
            knowledge_used_valid=True,
            reasoning_matches_knowledge=True,
            suspiciously_correct=False,
            issues_found=[f"Verification skipped: {error}"] if error else [],
            reasoning="Fallback: No verification performed.",
            should_rerun=False,
            suggested_answer=None,
        )
