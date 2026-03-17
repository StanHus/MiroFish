"""
Knowledge Agent - Dynamically determines what a student knows.

NO HARDCODING. The agent reasons about:
- What curriculum content is taught at this grade
- What a student with this archetype would retain
- What misconceptions they might have

This is GENERATIVE, not lookup-based.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


@dataclass
class KnowledgeProfile:
    """What a student knows about a topic."""

    grade: int
    archetype: str
    subject: str

    # Dynamically generated
    concepts_known: List[str]
    concepts_unknown: List[str]
    misconceptions: List[str]
    vocabulary_comfortable: List[str]
    vocabulary_unfamiliar: List[str]

    # Key assessment: Can they answer this question?
    can_answer_correctly: bool  # Agent's assessment
    confidence_in_assessment: float  # 0-1
    why_can_or_cannot: str  # Reasoning

    # Reasoning
    curriculum_reasoning: str
    archetype_adjustments: str

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "archetype": self.archetype,
            "subject": self.subject,
            "concepts_known": self.concepts_known,
            "concepts_unknown": self.concepts_unknown,
            "misconceptions": self.misconceptions,
            "vocabulary_comfortable": self.vocabulary_comfortable,
            "vocabulary_unfamiliar": self.vocabulary_unfamiliar,
            "can_answer_correctly": self.can_answer_correctly,
            "confidence_in_assessment": self.confidence_in_assessment,
            "why_can_or_cannot": self.why_can_or_cannot,
            "curriculum_reasoning": self.curriculum_reasoning,
            "archetype_adjustments": self.archetype_adjustments,
        }

    def to_constraint_prompt(self) -> str:
        """Convert to a prompt section for the answer agent."""
        known = "\n".join(f"- {c}" for c in self.concepts_known) or "- (No specific knowledge)"
        unknown = "\n".join(f"- {c}" for c in self.concepts_unknown) or "- (Nothing specific)"
        misconceptions = "\n".join(f"- {m}" for m in self.misconceptions) or "- (None identified)"

        return f"""=== WHAT YOU KNOW (Grade {self.grade} {self.archetype}) ===
{known}

=== WHAT YOU DO NOT KNOW ===
{unknown}

=== MISCONCEPTIONS YOU HAVE ===
{misconceptions}

=== VOCABULARY ===
Words you understand: {', '.join(self.vocabulary_comfortable[:10]) or 'basic words'}
Words that confuse you: {', '.join(self.vocabulary_unfamiliar[:10]) or 'none specifically'}
"""


class KnowledgeAgent:
    """
    Agent that determines what a student at grade X knows about a topic.

    This is GENERATIVE - no hardcoded curriculum databases.
    The agent reasons about education standards, typical instruction,
    and student archetypes to build a knowledge profile.
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

    async def build_profile(
        self,
        question: Dict[str, Any],
        grade: int,
        archetype: str,
        subject: Optional[str] = None,
    ) -> KnowledgeProfile:
        """
        Build a knowledge profile for a student.

        Args:
            question: The question (used to identify relevant topic)
            grade: Student's grade level (1-12)
            archetype: Student type (honors_overachiever, esl_student, etc.)
            subject: Optional subject area

        Returns:
            KnowledgeProfile with dynamically generated constraints
        """
        if not self.client:
            return self._fallback_profile(grade, archetype, subject or "general")

        question_text = question.get("text") or question.get("question", "")
        options = question.get("options", [])
        detected_subject = subject or question.get("subject", "general")

        system_prompt = """You are an expert in K-12 education curriculum and child development.

Your task: Determine what a student at a specific grade level would realistically know about a topic.

You must reason about:
1. CURRICULUM: What is typically taught at this grade? (Consider Common Core, state standards)
2. RETENTION: What would a typical student actually remember?
3. ARCHETYPE: How does this student type affect their knowledge?
4. MISCONCEPTIONS: What common errors do students at this level make?

Be REALISTIC. Students don't know everything in their textbooks.
An average 5th grader doesn't know Electoral College details.
A high schooler in AP Gov does.

Return your analysis as JSON."""

        user_prompt = f"""QUESTION CONTEXT:
Question: {question_text}
Options: {json.dumps(options)}
Subject: {detected_subject}

STUDENT PROFILE:
Grade: {grade}
Archetype: {archetype}

ARCHETYPE DEFINITIONS:
- honors_overachiever: Top of class, reads ahead, high retention, studies extra
- debate_club_kid: Intellectually curious, good with arguments, may know advanced topics in their interests
- quiet_thinker: Average knowledge, thoughtful but may second-guess
- disengaged_but_smart: Has the ability but doesn't pay attention, spotty knowledge
- esl_student: English language learner, may struggle with vocabulary, core concepts may be solid
- class_clown: Doesn't pay attention, minimal retention, relies on guessing
- average_student: Typical student, knows what was directly taught, some gaps

Analyze what this specific student would know about this topic.

CRITICAL: Also assess whether this student could realistically answer this specific question correctly.

Guidelines for can_answer_correctly:
- Basic facts taught in elementary school (branches of government, who makes laws) → most students CAN answer
- Specific numbers/dates/details (270 electoral votes, specific amendments) → only advanced students
- Common knowledge (President lives in White House) → almost all students CAN answer
- Specialized topics (faithless electors, filibuster rules) → only AP/advanced students

Be calibrated:
- "Who makes laws?" → Grade 3+ should know this (Congress)
- "How many electoral votes?" → Only AP Gov/advanced students would know (270)
- "What year was Constitution signed?" → Grade 5+ history students might know (1787)

Return JSON:
{{
    "curriculum_reasoning": "What is typically taught at grade {grade} about this topic?",
    "archetype_adjustments": "How does the {archetype} archetype modify typical knowledge?",
    "concepts_known": ["specific things they WOULD know"],
    "concepts_unknown": ["specific things they would NOT know"],
    "misconceptions": ["common mistakes they might make"],
    "vocabulary_comfortable": ["words in the question they understand"],
    "vocabulary_unfamiliar": ["words that would confuse them"],
    "can_answer_correctly": true/false,
    "confidence_in_assessment": 0.0-1.0,
    "why_can_or_cannot": "Explain why they can or cannot answer correctly"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # More consistent knowledge profiles
                max_tokens=800,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON with robust extraction
            data = self._parse_json_response(result_text)

            return KnowledgeProfile(
                grade=grade,
                archetype=archetype,
                subject=detected_subject,
                concepts_known=data.get("concepts_known", []),
                concepts_unknown=data.get("concepts_unknown", []),
                misconceptions=data.get("misconceptions", []),
                vocabulary_comfortable=data.get("vocabulary_comfortable", []),
                vocabulary_unfamiliar=data.get("vocabulary_unfamiliar", []),
                can_answer_correctly=data.get("can_answer_correctly", False),
                confidence_in_assessment=float(data.get("confidence_in_assessment", 0.5)),
                why_can_or_cannot=data.get("why_can_or_cannot", ""),
                curriculum_reasoning=data.get("curriculum_reasoning", ""),
                archetype_adjustments=data.get("archetype_adjustments", ""),
            )

        except Exception as e:
            # Fallback on error
            return self._fallback_profile(grade, archetype, detected_subject, str(e))

    def _parse_json_response(self, text: str) -> dict:
        """Robustly parse JSON from LLM response."""
        import re

        # Extract from markdown code blocks
        if "```" in text:
            parts = text.split("```")
            for part in parts[1:]:
                if part.startswith("json"):
                    part = part[4:]
                part = part.strip()
                if part.startswith("{"):
                    text = part.split("```")[0] if "```" in part else part
                    break

        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group()

        # Clean common issues
        # Remove trailing commas before } or ]
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix unquoted booleans
        text = re.sub(r':\s*true\b', ': true', text, flags=re.IGNORECASE)
        text = re.sub(r':\s*false\b', ': false', text, flags=re.IGNORECASE)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract key fields manually
            data = {}

            # Extract can_answer_correctly
            if '"can_answer_correctly"' in text.lower() or "'can_answer_correctly'" in text.lower():
                if 'true' in text.lower():
                    data['can_answer_correctly'] = True
                else:
                    data['can_answer_correctly'] = False

            # Extract arrays
            for field in ['concepts_known', 'concepts_unknown', 'misconceptions']:
                match = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                if match:
                    items = re.findall(r'"([^"]*)"', match.group(1))
                    data[field] = items

            # Extract strings
            for field in ['curriculum_reasoning', 'why_can_or_cannot']:
                match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
                if match:
                    data[field] = match.group(1)

            if data:
                return data

            raise json.JSONDecodeError("Could not parse", text, 0)

    def _fallback_profile(
        self,
        grade: int,
        archetype: str,
        subject: str,
        error: str = ""
    ) -> KnowledgeProfile:
        """Minimal fallback when API unavailable."""
        return KnowledgeProfile(
            grade=grade,
            archetype=archetype,
            subject=subject,
            concepts_known=["Basic concepts taught at this grade"],
            concepts_unknown=["Advanced concepts beyond this grade"],
            misconceptions=["May have typical grade-level misconceptions"],
            vocabulary_comfortable=["Common words"],
            vocabulary_unfamiliar=["Advanced vocabulary"],
            can_answer_correctly=False,  # Assume can't answer in fallback
            confidence_in_assessment=0.5,
            why_can_or_cannot=f"Fallback mode - assuming uncertainty. {error}",
            curriculum_reasoning=f"Fallback: Typical grade {grade} knowledge. {error}",
            archetype_adjustments=f"Fallback: {archetype} archetype adjustments.",
        )
