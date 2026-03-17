"""
Prior Knowledge Analysis - What concepts are required to answer a question.

Extracts required concepts from questions and maps them to curriculum
standards and grade levels where they are typically introduced.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


# ── Curriculum Knowledge Map ────────────────────────────────────────────────
# When concepts are typically taught in US curriculum

CONCEPT_GRADE_LEVELS = {
    # Government/Civics
    "three branches of government": 4,
    "branches of government": 4,
    "executive branch": 5,
    "legislative branch": 5,
    "judicial branch": 5,
    "checks and balances": 5,
    "separation of powers": 6,
    "constitution": 5,
    "bill of rights": 5,
    "amendments": 6,
    "electoral college": 8,
    "electoral process": 6,
    "voting": 3,
    "elections": 4,
    "president": 3,
    "congress": 5,
    "senate": 6,
    "house of representatives": 6,
    "supreme court": 5,
    "federalism": 9,
    "democracy": 5,
    "republic": 6,
    "citizenship": 4,
    "civil rights": 6,
    "civil liberties": 8,
    "due process": 9,
    "equal protection": 9,
    "first amendment": 7,
    "freedom of speech": 5,
    "freedom of religion": 5,
    "right to vote": 5,
    "suffrage": 8,
    "political parties": 6,
    "two-party system": 8,
    "primary elections": 9,
    "caucuses": 10,
    "nominating conventions": 10,
    "popular vote": 7,
    "electoral votes": 8,
    "swing states": 10,
    "faithless electors": 11,
    "12th amendment": 10,

    # Math concepts
    "addition": 1,
    "subtraction": 1,
    "multiplication": 3,
    "division": 3,
    "fractions": 3,
    "decimals": 4,
    "percentages": 5,
    "ratios": 6,
    "proportions": 6,
    "negative numbers": 6,
    "integers": 6,
    "order of operations": 5,
    "variables": 6,
    "expressions": 6,
    "equations": 6,
    "linear equations": 7,
    "inequalities": 7,
    "slope": 8,
    "y-intercept": 8,
    "graphing": 5,
    "coordinate plane": 5,
    "quadratic equations": 9,
    "polynomials": 9,
    "factoring": 8,
    "pythagorean theorem": 8,
    "trigonometry": 9,
    "sine cosine tangent": 10,
    "logarithms": 10,
    "exponential functions": 9,
    "sequences": 8,
    "series": 10,
    "probability": 6,
    "statistics": 6,
    "mean median mode": 5,
    "standard deviation": 10,

    # Science concepts
    "living things": 1,
    "plants and animals": 1,
    "habitats": 3,
    "ecosystems": 5,
    "food chains": 4,
    "food webs": 5,
    "photosynthesis": 5,
    "cells": 5,
    "cell parts": 6,
    "mitosis": 8,
    "meiosis": 9,
    "DNA": 8,
    "genetics": 8,
    "heredity": 7,
    "evolution": 8,
    "natural selection": 8,
    "adaptation": 5,
    "classification": 5,
    "taxonomy": 8,
    "matter": 3,
    "states of matter": 3,
    "atoms": 6,
    "molecules": 6,
    "elements": 6,
    "periodic table": 7,
    "chemical reactions": 7,
    "chemical bonds": 8,
    "energy": 4,
    "forms of energy": 5,
    "energy transfer": 6,
    "force and motion": 3,
    "newton's laws": 8,
    "gravity": 5,
    "magnetism": 3,
    "electricity": 4,
    "circuits": 5,
    "waves": 5,
    "light": 4,
    "sound": 4,
    "earth layers": 5,
    "plate tectonics": 7,
    "weather": 2,
    "climate": 5,
    "water cycle": 4,
    "rock cycle": 5,
    "solar system": 4,
    "planets": 3,
    "moon phases": 4,
    "seasons": 3,

    # History concepts
    "community helpers": 1,
    "maps and globes": 2,
    "continents and oceans": 3,
    "native americans": 4,
    "european exploration": 5,
    "colonial america": 5,
    "american revolution": 5,
    "declaration of independence": 5,
    "us constitution": 5,
    "westward expansion": 5,
    "civil war": 5,
    "reconstruction": 8,
    "industrialization": 7,
    "immigration": 5,
    "world war i": 7,
    "world war ii": 7,
    "great depression": 7,
    "cold war": 8,
    "civil rights movement": 6,
    "ancient civilizations": 6,
    "ancient egypt": 6,
    "ancient greece": 6,
    "ancient rome": 6,
    "middle ages": 7,
    "renaissance": 8,
    "reformation": 9,
    "enlightenment": 9,
    "french revolution": 8,
    "imperialism": 9,
    "nationalism": 9,
}


@dataclass
class ConceptRequirement:
    """A concept required to answer a question."""

    concept: str
    typically_taught: int  # Grade when typically introduced
    target_grade: int
    gap: int  # Positive = requires knowledge beyond target grade
    is_prerequisite: bool  # Must know this to answer correctly

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "typically_taught": self.typically_taught,
            "target_grade": self.target_grade,
            "gap": self.gap,
            "is_prerequisite": self.is_prerequisite,
        }


@dataclass
class PriorKnowledgeResult:
    """Result of prior knowledge analysis."""

    required_concepts: List[ConceptRequirement]
    concepts_above_grade: int
    max_gap: int
    verdict: str  # "appropriate", "requires_scaffolding", "requires_advanced_knowledge"

    def to_dict(self) -> dict:
        return {
            "required_concepts": [c.to_dict() for c in self.required_concepts],
            "concepts_above_grade": self.concepts_above_grade,
            "max_gap": self.max_gap,
            "verdict": self.verdict,
        }


class PriorKnowledgeAnalyzer:
    """Analyzes what prior knowledge is required to answer a question."""

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

    def _check_known_concepts(
        self,
        text: str,
        target_grade: int,
    ) -> List[ConceptRequirement]:
        """Check for concepts in our knowledge map."""
        requirements = []
        text_lower = text.lower()

        for concept, grade in CONCEPT_GRADE_LEVELS.items():
            if concept in text_lower:
                gap = grade - target_grade
                requirements.append(ConceptRequirement(
                    concept=concept,
                    typically_taught=grade,
                    target_grade=target_grade,
                    gap=gap,
                    is_prerequisite=True,
                ))

        return requirements

    async def analyze(
        self,
        content: Dict[str, Any],
        target_grade: int,
    ) -> PriorKnowledgeResult:
        """
        Analyze what prior knowledge is needed to answer a question.

        Args:
            content: Question content with text, options
            target_grade: Target grade level

        Returns:
            PriorKnowledgeResult with required concepts
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        options_text = " ".join(str(o) for o in options)
        full_text = f"{text} {options_text}"

        # First check our knowledge map
        known_requirements = self._check_known_concepts(full_text, target_grade)

        # Use LLM for deeper analysis if available
        if self.client:
            llm_requirements = await self._llm_analyze(content, target_grade)

            # Merge, preferring database info for duplicates
            known_concepts = {r.concept.lower() for r in known_requirements}
            for req in llm_requirements:
                if req.concept.lower() not in known_concepts:
                    known_requirements.append(req)

        # Sort by gap
        known_requirements.sort(key=lambda x: x.gap, reverse=True)

        # Calculate metrics
        concepts_above = sum(1 for r in known_requirements if r.gap > 0)
        max_gap = max((r.gap for r in known_requirements), default=0)

        # Determine verdict
        if max_gap <= 0:
            verdict = "appropriate"
        elif max_gap <= 2:
            verdict = "requires_scaffolding"
        else:
            verdict = "requires_advanced_knowledge"

        return PriorKnowledgeResult(
            required_concepts=known_requirements,
            concepts_above_grade=concepts_above,
            max_gap=max_gap,
            verdict=verdict,
        )

    async def _llm_analyze(
        self,
        content: Dict[str, Any],
        target_grade: int,
    ) -> List[ConceptRequirement]:
        """Use LLM to extract required concepts."""
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        options_text = "\n".join(f"- {o}" for o in options)

        prompt = f"""Analyze what specific knowledge a student needs to answer this question correctly.

Question: {text}

Options:
{options_text}

Target grade level: {target_grade}

List the specific concepts/facts a student must ALREADY KNOW to answer this correctly.
Don't list skills like "reading comprehension" - list specific content knowledge.

For each concept, estimate what US grade level it's typically taught.

Return JSON:
{{
  "required_concepts": [
    {{"concept": "name of concept", "grade_typically_taught": 5, "is_prerequisite": true}}
  ]
}}

Only include concepts that are actually necessary to answer the question."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing prerequisite knowledge for educational questions. "
                            "Be specific about what students need to know, and accurate about "
                            "when concepts are typically taught in US schools. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            requirements = []
            for item in analysis.get("required_concepts", []):
                concept = item.get("concept", "")
                grade = item.get("grade_typically_taught", target_grade)

                requirements.append(ConceptRequirement(
                    concept=concept,
                    typically_taught=grade,
                    target_grade=target_grade,
                    gap=grade - target_grade,
                    is_prerequisite=item.get("is_prerequisite", True),
                ))

            return requirements

        except Exception:
            return []


async def analyze_prior_knowledge(
    content: Dict[str, Any],
    target_grade: int,
    api_key: Optional[str] = None,
) -> PriorKnowledgeResult:
    """Convenience function for prior knowledge analysis."""
    analyzer = PriorKnowledgeAnalyzer(api_key=api_key)
    return await analyzer.analyze(content, target_grade)
