"""
Cognitive models for realistic student simulation.

These models address the core problem: LLMs know the right answer and can't
genuinely simulate student ignorance. We solve this by:

1. RetentionModel: Filters what knowledge a student would actually have
2. PerceptionModel: Transforms how a student perceives/understands the question

Together, these create a "cognitive lens" through which the LLM answers,
producing genuinely different responses based on knowledge gaps.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# ── Grade Level Knowledge Boundaries ───────────────────────────────────────

# What concepts are typically introduced at each grade level
GRADE_KNOWLEDGE_BOUNDARIES = {
    # Elementary (K-5)
    3: {
        "math": ["addition", "subtraction", "simple multiplication", "basic fractions"],
        "reading": ["main idea", "characters", "setting", "basic vocabulary"],
        "science": ["living things", "weather", "simple machines"],
        "social_studies": ["community", "maps", "holidays", "family roles"],
    },
    4: {
        "math": ["multiplication", "division", "fractions", "decimals intro"],
        "reading": ["inference", "compare contrast", "figurative language basics"],
        "science": ["ecosystems", "states of matter", "simple electricity"],
        "social_studies": ["state history", "regions", "basic economics"],
    },
    5: {
        "math": ["decimals", "percentages intro", "order of operations", "basic geometry"],
        "reading": ["theme", "point of view", "text structure"],
        "science": ["earth systems", "cells basics", "forces"],
        "social_studies": ["US history basics", "government intro", "geography"],
    },
    # Middle School (6-8)
    6: {
        "math": ["ratios", "negative numbers", "expressions", "area/volume"],
        "reading": ["author's purpose", "argument analysis", "citation"],
        "science": ["cell biology", "energy", "earth science"],
        "social_studies": ["ancient civilizations", "world geography"],
    },
    7: {
        "math": ["proportions", "linear equations", "statistics basics"],
        "reading": ["rhetoric", "compare sources", "literary analysis"],
        "science": ["life science", "chemistry basics", "genetics intro"],
        "social_studies": ["world history", "cultural studies"],
    },
    8: {
        "math": ["linear functions", "pythagorean theorem", "transformations"],
        "reading": ["synthesis", "evaluate arguments", "irony/satire"],
        "science": ["physics basics", "chemistry", "evolution"],
        "social_studies": ["US history", "civics basics", "economics"],
    },
    # High School (9-12)
    9: {
        "math": ["algebra I", "linear systems", "quadratics intro"],
        "reading": ["complex analysis", "rhetoric", "research skills"],
        "science": ["biology", "scientific method", "lab skills"],
        "social_studies": ["world history", "geography", "current events"],
    },
    10: {
        "math": ["geometry", "proofs", "trigonometry intro"],
        "reading": ["literary criticism", "argumentative writing"],
        "science": ["chemistry", "biochemistry basics"],
        "social_studies": ["world cultures", "modern history"],
    },
    11: {
        "math": ["algebra II", "trigonometry", "sequences"],
        "reading": ["AP-level analysis", "synthesis across texts"],
        "science": ["physics", "environmental science"],
        "social_studies": ["US history", "government", "AP content"],
    },
    12: {
        "math": ["pre-calculus", "statistics", "calculus intro"],
        "reading": ["college-level analysis", "research papers"],
        "science": ["AP sciences", "specialized topics"],
        "social_studies": ["AP government", "economics", "current events"],
    },
}

# Archetype retention modifiers (multiplier on base retention)
ARCHETYPE_RETENTION = {
    "honors_overachiever": 1.3,      # Excellent retention
    "debate_club_kid": 1.1,          # Good, especially for arguable topics
    "quiet_thinker": 1.0,            # Average but deep
    "socially_engaged_activist": 0.9, # Selective retention (social topics higher)
    "disengaged_but_smart": 0.7,     # Could retain but often doesn't
    "esl_student": 0.8,              # Language barrier affects encoding
    "class_clown": 0.5,              # Poor retention due to inattention
    "politically_conservative": 0.95, # Average
}


@dataclass
class RetentionContext:
    """What a student knows/remembers about a topic."""

    grade_level: int
    archetype: str
    known_concepts: List[str]
    partial_concepts: List[str]  # Vaguely remembered
    unknown_concepts: List[str]  # Never learned or forgotten
    retention_confidence: float  # 0-1, overall confidence in knowledge


@dataclass
class PerceivedQuestion:
    """How a student perceives/understands a question."""

    original_text: str
    perceived_text: str  # What the student "reads"
    misunderstood_terms: List[Tuple[str, str]]  # (original, perceived_as)
    skipped_details: List[str]  # Parts the student glossed over
    perceived_difficulty: str  # "easy", "medium", "hard", "confusing"
    attention_level: float  # 0-1, how carefully they read


class RetentionModel:
    """
    Models what knowledge a student would actually retain.

    Addresses the problem: LLMs know everything, but real students forget.

    Usage:
        model = RetentionModel()
        context = await model.analyze_retention(
            content=question_content,
            grade_level=11,
            archetype="quiet_thinker",
            subject="AP Government"
        )

        # context.known_concepts = ["electoral college exists", "votes for president"]
        # context.unknown_concepts = ["faithless electors", "12th amendment details"]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def analyze_retention(
        self,
        content: Dict[str, Any],
        grade_level: int,
        archetype: str,
        subject: Optional[str] = None,
    ) -> RetentionContext:
        """
        Analyze what knowledge a student would retain about this topic.

        Args:
            content: Question content with text, options, subject
            grade_level: Student's grade (3-12)
            archetype: Student archetype
            subject: Subject area

        Returns:
            RetentionContext with known/partial/unknown concepts
        """
        text = content.get("text") or content.get("question", "")
        subject = subject or content.get("subject", "general")

        # Get base retention modifier for archetype
        retention_modifier = ARCHETYPE_RETENTION.get(archetype, 0.8)

        # Get grade-appropriate knowledge
        grade_knowledge = self._get_grade_knowledge(grade_level, subject)

        if not self.client:
            # Fallback: simple heuristic-based retention
            return self._heuristic_retention(
                text, grade_level, archetype, retention_modifier
            )

        # Use LLM to analyze what concepts are needed and what student would know
        return await self._llm_retention_analysis(
            text, grade_level, archetype, subject, grade_knowledge, retention_modifier
        )

    def _get_grade_knowledge(self, grade_level: int, subject: str) -> List[str]:
        """Get concepts typically known at this grade level."""
        # Normalize subject
        subject_map = {
            "ap government": "social_studies",
            "ap gov": "social_studies",
            "government": "social_studies",
            "civics": "social_studies",
            "mathematics": "math",
            "algebra": "math",
            "geometry": "math",
            "calculus": "math",
            "english": "reading",
            "ela": "reading",
            "history": "social_studies",
        }
        normalized = subject_map.get(subject.lower(), subject.lower())

        # Accumulate knowledge from lower grades
        all_knowledge = []
        for grade in range(3, min(grade_level + 1, 13)):
            if grade in GRADE_KNOWLEDGE_BOUNDARIES:
                grade_topics = GRADE_KNOWLEDGE_BOUNDARIES[grade].get(normalized, [])
                all_knowledge.extend(grade_topics)

        return all_knowledge

    def _heuristic_retention(
        self,
        text: str,
        grade_level: int,
        archetype: str,
        retention_modifier: float,
    ) -> RetentionContext:
        """Simple heuristic-based retention when LLM unavailable."""

        # Base retention probability
        base_retention = 0.6 * retention_modifier

        # Grade penalty (younger students know less)
        grade_penalty = max(0, (11 - grade_level) * 0.05)
        retention_prob = max(0.2, base_retention - grade_penalty)

        return RetentionContext(
            grade_level=grade_level,
            archetype=archetype,
            known_concepts=["basic topic familiarity"],
            partial_concepts=["some related facts"],
            unknown_concepts=["specific details", "technical terminology"],
            retention_confidence=retention_prob,
        )

    async def _llm_retention_analysis(
        self,
        text: str,
        grade_level: int,
        archetype: str,
        subject: str,
        grade_knowledge: List[str],
        retention_modifier: float,
    ) -> RetentionContext:
        """Use LLM to analyze retention more accurately."""

        grade_knowledge_str = ", ".join(grade_knowledge[:20]) if grade_knowledge else "general grade-level content"

        prompt = f"""Analyze what a grade {grade_level} student with the "{archetype}" learning profile would actually KNOW and REMEMBER about this topic.

QUESTION BEING ASKED:
{text}

SUBJECT: {subject}

CONCEPTS TYPICALLY COVERED BY GRADE {grade_level}:
{grade_knowledge_str}

STUDENT PROFILE ({archetype}):
- Retention modifier: {retention_modifier:.1f}x average
- This affects how well they encode and retrieve information

Determine what specific concepts/facts this student would:
1. KNOW WELL (clearly remember, can apply)
2. PARTIALLY KNOW (vague memory, might confuse details)
3. NOT KNOW (never learned, or completely forgotten)

Consider:
- What's actually taught by grade {grade_level}
- What a {archetype} student would pay attention to
- Common gaps in understanding at this level

Respond with JSON only:
{{
    "known_concepts": ["concept 1", "concept 2"],
    "partial_concepts": ["vaguely remembered thing"],
    "unknown_concepts": ["advanced concept they wouldn't know"],
    "retention_confidence": 0.7
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an educational psychologist modeling student knowledge retention. "
                            "Be realistic about what students at different grade levels actually know and remember. "
                            "Students forget things, have gaps, and often have partial/incorrect knowledge. "
                            "Respond ONLY with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=400,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            return RetentionContext(
                grade_level=grade_level,
                archetype=archetype,
                known_concepts=analysis.get("known_concepts", []),
                partial_concepts=analysis.get("partial_concepts", []),
                unknown_concepts=analysis.get("unknown_concepts", []),
                retention_confidence=analysis.get("retention_confidence", 0.5),
            )

        except Exception:
            return self._heuristic_retention(text, grade_level, archetype, retention_modifier)


# ── Perception Model ───────────────────────────────────────────────────────

# Vocabulary complexity thresholds by grade
VOCABULARY_GRADE_LEVELS = {
    # Word: minimum grade level to understand
    "electoral": 8,
    "ratify": 9,
    "constitutional": 8,
    "amendment": 7,
    "federalism": 10,
    "bicameral": 11,
    "jurisdiction": 10,
    "precedent": 9,
    "coefficient": 7,
    "polynomial": 9,
    "derivative": 11,
    "hypotenuse": 8,
    "photosynthesis": 6,
    "mitochondria": 7,
    "hypothesis": 6,
}

# Archetype attention patterns
ARCHETYPE_ATTENTION = {
    "honors_overachiever": {
        "attention_level": 0.95,
        "reads_all_options": True,
        "catches_details": True,
        "second_guesses": True,
    },
    "debate_club_kid": {
        "attention_level": 0.85,
        "reads_all_options": True,
        "catches_details": True,  # For things they find interesting
        "second_guesses": True,
    },
    "quiet_thinker": {
        "attention_level": 0.80,
        "reads_all_options": True,
        "catches_details": True,
        "second_guesses": True,  # Sometimes overthinks
    },
    "socially_engaged_activist": {
        "attention_level": 0.75,
        "reads_all_options": True,
        "catches_details": False,  # May rush if not engaging
        "second_guesses": False,
    },
    "disengaged_but_smart": {
        "attention_level": 0.50,
        "reads_all_options": False,  # May not read all options
        "catches_details": False,
        "second_guesses": False,
    },
    "esl_student": {
        "attention_level": 0.85,  # Tries hard but struggles
        "reads_all_options": True,
        "catches_details": False,  # Language barrier
        "second_guesses": True,
    },
    "class_clown": {
        "attention_level": 0.30,
        "reads_all_options": False,
        "catches_details": False,
        "second_guesses": False,
    },
    "politically_conservative": {
        "attention_level": 0.75,
        "reads_all_options": True,
        "catches_details": True,  # For topics they care about
        "second_guesses": False,
    },
}


class PerceptionModel:
    """
    Models how a student perceives and understands a question.

    A 3rd grader doesn't just "know less" - they literally perceive text
    differently. They may:
    - Misread complex vocabulary
    - Skip over long sentences
    - Miss nuance and qualifiers
    - Interpret through their worldview

    Usage:
        model = PerceptionModel()
        perceived = await model.perceive(
            content=question_content,
            grade_level=11,
            archetype="esl_student"
        )

        # perceived.perceived_text = simplified/misunderstood version
        # perceived.misunderstood_terms = [("electoral", "election")]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def perceive(
        self,
        content: Dict[str, Any],
        grade_level: int,
        archetype: str,
    ) -> PerceivedQuestion:
        """
        Generate how a student would perceive/understand this question.

        Args:
            content: Question content
            grade_level: Student's grade
            archetype: Student archetype

        Returns:
            PerceivedQuestion with transformed understanding
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])

        # Get attention pattern for archetype
        attention = ARCHETYPE_ATTENTION.get(archetype, {
            "attention_level": 0.7,
            "reads_all_options": True,
            "catches_details": False,
            "second_guesses": False,
        })

        if not self.client:
            return self._heuristic_perception(
                text, options, grade_level, archetype, attention
            )

        return await self._llm_perception(
            text, options, grade_level, archetype, attention
        )

    def _heuristic_perception(
        self,
        text: str,
        options: List[str],
        grade_level: int,
        archetype: str,
        attention: Dict[str, Any],
    ) -> PerceivedQuestion:
        """Simple heuristic perception when LLM unavailable."""

        # Find words above grade level
        misunderstood = []
        for word, min_grade in VOCABULARY_GRADE_LEVELS.items():
            if word.lower() in text.lower() and grade_level < min_grade:
                misunderstood.append((word, f"[unclear: {word}]"))

        # Determine perceived difficulty
        if len(misunderstood) >= 3:
            difficulty = "confusing"
        elif len(misunderstood) >= 1:
            difficulty = "hard"
        elif len(text) > 200:
            difficulty = "medium"
        else:
            difficulty = "easy"

        return PerceivedQuestion(
            original_text=text,
            perceived_text=text,  # No transformation in heuristic mode
            misunderstood_terms=misunderstood,
            skipped_details=[],
            perceived_difficulty=difficulty,
            attention_level=attention.get("attention_level", 0.7),
        )

    async def _llm_perception(
        self,
        text: str,
        options: List[str],
        grade_level: int,
        archetype: str,
        attention: Dict[str, Any],
    ) -> PerceivedQuestion:
        """Use LLM to model perception more accurately."""

        options_text = "\n".join(f"{chr(65+i)}) {opt}" for i, opt in enumerate(options))

        attention_desc = (
            f"Attention level: {attention['attention_level']:.0%}\n"
            f"Reads all options: {'Yes' if attention['reads_all_options'] else 'No'}\n"
            f"Catches details: {'Yes' if attention['catches_details'] else 'No'}"
        )

        prompt = f"""Model how a grade {grade_level} student with "{archetype}" profile would PERCEIVE and UNDERSTAND this question.

ORIGINAL QUESTION:
{text}

OPTIONS:
{options_text}

STUDENT ATTENTION PATTERN:
{attention_desc}

Create a "perceived version" - what the student actually understands when reading this.
Consider:
- Vocabulary they might not know or might misinterpret
- Complex sentences they might misread
- Details they might skip or miss
- Their grade-level worldview affecting interpretation
- How their archetype affects reading (rushed? careful? distracted?)

Respond with JSON only:
{{
    "perceived_text": "What the student thinks the question is asking (may be simplified or misunderstood)",
    "misunderstood_terms": [["original word", "what they think it means"]],
    "skipped_details": ["parts they glossed over"],
    "perceived_difficulty": "easy|medium|hard|confusing"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are modeling student cognition. Create realistic 'perceived' versions "
                            "of questions as they would be understood by students at different levels. "
                            "Be specific about misunderstandings and gaps. "
                            "A 3rd grader perceives an AP question very differently than an 11th grader. "
                            "Respond ONLY with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            return PerceivedQuestion(
                original_text=text,
                perceived_text=analysis.get("perceived_text", text),
                misunderstood_terms=[
                    tuple(pair) for pair in analysis.get("misunderstood_terms", [])
                ],
                skipped_details=analysis.get("skipped_details", []),
                perceived_difficulty=analysis.get("perceived_difficulty", "medium"),
                attention_level=attention.get("attention_level", 0.7),
            )

        except Exception:
            return self._heuristic_perception(text, options, grade_level, archetype, attention)


# ── Combined Cognitive Lens ────────────────────────────────────────────────

@dataclass
class CognitiveLens:
    """Combined view of what a student knows and perceives."""

    retention: RetentionContext
    perception: PerceivedQuestion

    # Derived: can the student actually answer this correctly?
    can_answer_correctly: bool
    confidence_if_answers: float
    likely_errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retention": {
                "grade_level": self.retention.grade_level,
                "archetype": self.retention.archetype,
                "known_concepts": self.retention.known_concepts,
                "partial_concepts": self.retention.partial_concepts,
                "unknown_concepts": self.retention.unknown_concepts,
                "retention_confidence": round(self.retention.retention_confidence, 2),
            },
            "perception": {
                "perceived_text": self.perception.perceived_text,
                "misunderstood_terms": self.perception.misunderstood_terms,
                "skipped_details": self.perception.skipped_details,
                "perceived_difficulty": self.perception.perceived_difficulty,
                "attention_level": round(self.perception.attention_level, 2),
            },
            "can_answer_correctly": self.can_answer_correctly,
            "confidence_if_answers": round(self.confidence_if_answers, 2),
            "likely_errors": self.likely_errors,
        }


class CognitiveModel:
    """
    Complete cognitive model combining retention and perception.

    This is the main interface for simulating realistic student cognition.

    Usage:
        model = CognitiveModel(api_key="sk-...")
        lens = await model.create_lens(content, grade_level=11, archetype="esl_student")

        # lens.can_answer_correctly = False
        # lens.likely_errors = ["May confuse Electoral College with Congress"]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.retention_model = RetentionModel(api_key, base_url, model)
        self.perception_model = PerceptionModel(api_key, base_url, model)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def create_lens(
        self,
        content: Dict[str, Any],
        grade_level: int,
        archetype: str,
    ) -> CognitiveLens:
        """
        Create a complete cognitive lens for a student answering this question.

        Args:
            content: Question content
            grade_level: Student's grade
            archetype: Student archetype

        Returns:
            CognitiveLens with retention, perception, and answer predictions
        """
        # Run retention and perception analysis in parallel
        import asyncio

        retention_task = self.retention_model.analyze_retention(
            content, grade_level, archetype
        )
        perception_task = self.perception_model.perceive(
            content, grade_level, archetype
        )

        retention, perception = await asyncio.gather(retention_task, perception_task)

        # Determine if student can answer correctly given their cognitive state
        can_answer, confidence, likely_errors = self._assess_answer_capability(
            content, retention, perception
        )

        return CognitiveLens(
            retention=retention,
            perception=perception,
            can_answer_correctly=can_answer,
            confidence_if_answers=confidence,
            likely_errors=likely_errors,
        )

    def _assess_answer_capability(
        self,
        content: Dict[str, Any],
        retention: RetentionContext,
        perception: PerceivedQuestion,
    ) -> Tuple[bool, float, List[str]]:
        """Assess if student can answer correctly given their cognitive state."""

        likely_errors = []

        # Factor 1: Unknown concepts make correct answer unlikely
        unknown_penalty = len(retention.unknown_concepts) * 0.15

        # Factor 2: Misunderstood terms cause errors
        misunderstand_penalty = len(perception.misunderstood_terms) * 0.2

        # Factor 3: Low attention means missing key details
        attention_penalty = (1 - perception.attention_level) * 0.3

        # Factor 4: Perceived difficulty affects confidence
        difficulty_map = {"easy": 0, "medium": 0.1, "hard": 0.2, "confusing": 0.4}
        difficulty_penalty = difficulty_map.get(perception.perceived_difficulty, 0.2)

        # Base probability from retention confidence
        base_prob = retention.retention_confidence

        # Calculate final probability
        final_prob = max(0.05, base_prob - unknown_penalty - misunderstand_penalty - attention_penalty - difficulty_penalty)

        # Generate likely errors
        if retention.unknown_concepts:
            likely_errors.append(
                f"Missing knowledge: {', '.join(retention.unknown_concepts[:2])}"
            )

        if perception.misunderstood_terms:
            terms = [f"'{orig}'" for orig, _ in perception.misunderstood_terms[:2]]
            likely_errors.append(f"May misunderstand: {', '.join(terms)}")

        if perception.skipped_details:
            likely_errors.append(f"May miss: {perception.skipped_details[0]}")

        if perception.attention_level < 0.5:
            likely_errors.append("Low attention - may not read carefully")

        # Determine if they can answer correctly (probabilistic)
        can_answer = random.random() < final_prob

        return can_answer, final_prob, likely_errors


async def create_cognitive_lens(
    content: Dict[str, Any],
    grade_level: int,
    archetype: str,
    api_key: Optional[str] = None,
) -> CognitiveLens:
    """
    Convenience function to create a cognitive lens.

    Args:
        content: Question content
        grade_level: Student's grade level
        archetype: Student archetype
        api_key: OpenAI API key

    Returns:
        CognitiveLens with retention, perception, and predictions
    """
    model = CognitiveModel(api_key=api_key)
    return await model.create_lens(content, grade_level, archetype)
