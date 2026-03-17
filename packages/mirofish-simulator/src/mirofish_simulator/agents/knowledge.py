"""
Knowledge Constraints - What a student at grade X actually knows.

This is the key to realistic simulation. We don't ask "what's the right answer?"
We ask "what does this student KNOW that's relevant to this question?"

A 5th grader:
- Knows the president is elected
- Knows there's voting
- Does NOT know what the Electoral College does
- Does NOT know about faithless electors

When asked "What does the Electoral College do?", they'll guess based on
the words they recognize, not based on actual knowledge.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from openai import AsyncOpenAI


# ── Grade-Level Knowledge Definitions ───────────────────────────────────────

@dataclass
class GradeKnowledge:
    """Knowledge typically possessed by a student at a given grade."""

    grade: int

    # What they KNOW (facts, concepts)
    known_facts: List[str] = field(default_factory=list)
    known_concepts: List[str] = field(default_factory=list)

    # What they DON'T KNOW (explicitly)
    unknown_facts: List[str] = field(default_factory=list)
    unknown_concepts: List[str] = field(default_factory=list)

    # Vocabulary they understand vs don't
    known_vocabulary: Set[str] = field(default_factory=set)
    unknown_vocabulary: Set[str] = field(default_factory=set)

    # Partial/fuzzy knowledge (might confuse)
    partial_knowledge: List[str] = field(default_factory=list)
    common_misconceptions: List[str] = field(default_factory=list)


# What's typically taught by each grade (US curriculum)
CURRICULUM_BY_GRADE = {
    # Grade 3
    3: {
        "social_studies": [
            "Communities have leaders",
            "People vote to choose leaders",
            "The President is the leader of the country",
            "There are rules and laws",
            "Maps show where places are",
        ],
        "math": [
            "Addition and subtraction",
            "Basic multiplication",
            "Simple fractions (1/2, 1/4)",
        ],
        "science": [
            "Living vs non-living things",
            "Plants need water and sunlight",
            "Animals live in habitats",
        ],
    },
    # Grade 4
    4: {
        "social_studies": [
            "State government exists",
            "There are different levels of government",
            "Citizens have rights and responsibilities",
            "The Constitution is an important document",
            "American Revolution happened",
        ],
        "math": [
            "Multiplication and division",
            "Fractions with different denominators",
            "Basic geometry shapes",
        ],
        "science": [
            "Ecosystems and food chains",
            "States of matter",
            "Simple machines",
        ],
    },
    # Grade 5
    5: {
        "social_studies": [
            "Three branches of government exist",
            "Congress makes laws",
            "President enforces laws",
            "Courts interpret laws",
            "The Constitution can be changed (amendments)",
            "Colonial America and Revolution",
            "Westward expansion",
        ],
        "math": [
            "Decimals and percentages",
            "Order of operations",
            "Coordinate graphing basics",
        ],
        "science": [
            "Cells are building blocks of life",
            "Photosynthesis basics",
            "Earth's layers",
            "Water cycle",
        ],
    },
    # Grade 6
    6: {
        "social_studies": [
            "Ancient civilizations (Egypt, Greece, Rome)",
            "World geography",
            "Basic economics (supply/demand)",
            "Different types of government exist",
        ],
        "math": [
            "Ratios and proportions",
            "Negative numbers",
            "Basic algebra (variables)",
        ],
        "science": [
            "Cell structure",
            "Basic chemistry (atoms, elements)",
            "Earth science",
        ],
    },
    # Grade 7
    7: {
        "social_studies": [
            "World history (Middle Ages, Renaissance)",
            "Different economic systems",
            "Cultural diversity",
        ],
        "math": [
            "Linear equations",
            "Proportional relationships",
            "Statistics basics",
        ],
        "science": [
            "Life science",
            "Genetics basics",
            "Chemistry (compounds, reactions)",
        ],
    },
    # Grade 8
    8: {
        "social_studies": [
            "US History in depth",
            "Civil War and Reconstruction",
            "Constitution details",
            "Bill of Rights",
            "How laws are made",
            "Electoral College exists and elects President",
            "Federalism (state vs federal power)",
        ],
        "math": [
            "Linear functions",
            "Pythagorean theorem",
            "Transformations",
        ],
        "science": [
            "Physics basics (force, motion)",
            "Chemistry (periodic table)",
            "Evolution basics",
        ],
    },
    # Grade 9
    9: {
        "social_studies": [
            "World History",
            "Geography and cultures",
            "Current events",
            "Basic civics review",
        ],
        "math": [
            "Algebra I",
            "Quadratic equations",
            "Systems of equations",
        ],
        "science": [
            "Biology",
            "Scientific method",
        ],
    },
    # Grade 10
    10: {
        "social_studies": [
            "World cultures",
            "Modern history",
            "Comparative government",
        ],
        "math": [
            "Geometry",
            "Proofs",
            "Trigonometry intro",
        ],
        "science": [
            "Chemistry",
            "Biochemistry basics",
        ],
    },
    # Grade 11 (AP Government level)
    11: {
        "social_studies": [
            "US Government in depth",
            "Electoral College mechanics",
            "Faithless electors",
            "Constitutional amendments in detail",
            "Judicial review",
            "Federalism nuances",
            "Political parties and elections",
            "Civil liberties vs civil rights",
            "Policy making process",
        ],
        "math": [
            "Algebra II",
            "Trigonometry",
            "Pre-calculus",
        ],
        "science": [
            "Physics",
            "AP sciences",
        ],
    },
    # Grade 12
    12: {
        "social_studies": [
            "Economics",
            "AP Government",
            "Current events analysis",
            "College-level civics",
        ],
        "math": [
            "Pre-calculus",
            "Statistics",
            "Calculus intro",
        ],
        "science": [
            "AP sciences",
            "Specialized topics",
        ],
    },
}


# What students commonly DON'T know (even if taught)
COMMON_GAPS = {
    "electoral_college": {
        "known_by_grade": 8,
        "commonly_confused_with": ["popular vote", "Congress"],
        "common_misconceptions": [
            "The popular vote directly elects the President",
            "Electoral College is part of Congress",
            "Each state has the same number of electoral votes",
        ],
    },
    "amendments": {
        "known_by_grade": 5,
        "commonly_confused_with": ["laws", "bills"],
        "common_misconceptions": [
            "Amendments are regular laws",
            "The President can create amendments",
        ],
    },
    "checks_and_balances": {
        "known_by_grade": 5,
        "commonly_confused_with": ["separation of powers"],
        "common_misconceptions": [
            "Each branch is completely independent",
            "The President can override the Supreme Court",
        ],
    },
    "federalism": {
        "known_by_grade": 8,
        "commonly_confused_with": ["federal government only"],
        "common_misconceptions": [
            "Federal law always overrides state law",
            "States have no independent power",
        ],
    },
}


@dataclass
class KnowledgeBase:
    """
    Complete knowledge base for a student agent.

    This defines what the agent knows and doesn't know,
    which directly affects how it answers questions.
    """

    grade: int
    subject: str

    # Explicit knowledge
    facts_known: List[str]
    concepts_known: List[str]

    # Explicit ignorance
    facts_unknown: List[str]
    concepts_unknown: List[str]

    # Fuzzy areas
    partial_knowledge: List[str]
    misconceptions: List[str]

    # For the agent prompt
    def to_prompt_section(self) -> str:
        """Generate prompt section describing this knowledge state."""
        sections = []

        sections.append("=== WHAT YOU KNOW ===")
        if self.facts_known:
            sections.append("Facts you learned and remember:")
            for fact in self.facts_known[:10]:
                sections.append(f"  - {fact}")

        if self.concepts_known:
            sections.append("\nConcepts you understand:")
            for concept in self.concepts_known[:10]:
                sections.append(f"  - {concept}")

        sections.append("\n=== WHAT YOU DON'T KNOW ===")
        sections.append("You have NOT learned these yet (don't pretend to know them):")
        if self.facts_unknown:
            for fact in self.facts_unknown[:10]:
                sections.append(f"  - {fact}")
        if self.concepts_unknown:
            for concept in self.concepts_unknown[:10]:
                sections.append(f"  - {concept}")

        if self.partial_knowledge:
            sections.append("\n=== FUZZY KNOWLEDGE ===")
            sections.append("Things you've heard of but might get wrong:")
            for item in self.partial_knowledge[:5]:
                sections.append(f"  - {item}")

        if self.misconceptions:
            sections.append("\n=== THINGS YOU BELIEVE (that might be wrong) ===")
            for item in self.misconceptions[:5]:
                sections.append(f"  - {item}")

        return "\n".join(sections)


class KnowledgeBuilder:
    """Builds knowledge constraints for a student agent."""

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

    async def build_for_question(
        self,
        question: Dict[str, Any],
        grade: int,
        subject: Optional[str] = None,
    ) -> KnowledgeBase:
        """
        Build knowledge constraints specific to a question.

        This is the key method - it determines what the agent
        knows and doesn't know about THIS specific question.
        """
        text = question.get("text") or question.get("question", "")
        options = question.get("options", [])
        subject = subject or question.get("subject", "general")

        # Get base curriculum knowledge for this grade
        curriculum = self._get_curriculum_knowledge(grade, subject)

        # Use LLM to determine question-specific knowledge
        if self.client:
            return await self._build_with_llm(
                text, options, grade, subject, curriculum
            )

        # Fallback: use curriculum only
        return self._build_from_curriculum(grade, subject, curriculum)

    def _get_curriculum_knowledge(
        self,
        grade: int,
        subject: str,
    ) -> List[str]:
        """Get curriculum knowledge up to this grade."""
        knowledge = []

        # Normalize subject
        subject_map = {
            "ap government": "social_studies",
            "government": "social_studies",
            "civics": "social_studies",
            "history": "social_studies",
            "math": "math",
            "mathematics": "math",
            "science": "science",
        }
        normalized = subject_map.get(subject.lower(), "social_studies")

        # Accumulate knowledge from grades up to current
        for g in range(3, min(grade + 1, 13)):
            if g in CURRICULUM_BY_GRADE:
                grade_knowledge = CURRICULUM_BY_GRADE[g].get(normalized, [])
                knowledge.extend(grade_knowledge)

        return knowledge

    def _build_from_curriculum(
        self,
        grade: int,
        subject: str,
        curriculum: List[str],
    ) -> KnowledgeBase:
        """Build knowledge base from curriculum alone."""
        # Knowledge above this grade
        unknown = []
        for g in range(grade + 1, 13):
            if g in CURRICULUM_BY_GRADE:
                subject_key = "social_studies" if "gov" in subject.lower() else subject.lower()
                unknown.extend(CURRICULUM_BY_GRADE[g].get(subject_key, []))

        return KnowledgeBase(
            grade=grade,
            subject=subject,
            facts_known=curriculum,
            concepts_known=[],
            facts_unknown=unknown[:10],
            concepts_unknown=[],
            partial_knowledge=[],
            misconceptions=[],
        )

    async def _build_with_llm(
        self,
        text: str,
        options: List[str],
        grade: int,
        subject: str,
        curriculum: List[str],
    ) -> KnowledgeBase:
        """Use LLM to build question-specific knowledge constraints."""

        options_text = "\n".join(f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options))
        curriculum_text = "\n".join(f"  - {k}" for k in curriculum[:15])

        prompt = f"""You are determining what a grade {grade} student would ACTUALLY KNOW about this question.

QUESTION:
{text}

OPTIONS:
{options_text}

SUBJECT: {subject}

WHAT A GRADE {grade} STUDENT HAS BEEN TAUGHT:
{curriculum_text}

Analyze this question and determine:
1. What specific FACTS does a grade {grade} student KNOW that are relevant?
2. What specific FACTS would they NOT KNOW yet?
3. What might they have HEARD OF but be confused about?
4. What MISCONCEPTIONS might they have?

Be specific to THIS question. Don't just list curriculum - identify exactly what knowledge gaps would affect answering this question.

Return JSON:
{{
  "facts_known": ["specific fact 1", "specific fact 2"],
  "facts_unknown": ["thing they haven't learned", "advanced concept"],
  "partial_knowledge": ["heard of but fuzzy"],
  "misconceptions": ["wrong belief they might have"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in K-12 curriculum and student knowledge. "
                            "Be realistic about what students at each grade level actually know. "
                            "Students have gaps, misconceptions, and fuzzy knowledge. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            return KnowledgeBase(
                grade=grade,
                subject=subject,
                facts_known=analysis.get("facts_known", curriculum[:5]),
                concepts_known=curriculum[:5],
                facts_unknown=analysis.get("facts_unknown", []),
                concepts_unknown=[],
                partial_knowledge=analysis.get("partial_knowledge", []),
                misconceptions=analysis.get("misconceptions", []),
            )

        except Exception:
            return self._build_from_curriculum(grade, subject, curriculum)


async def build_knowledge_constraint(
    question: Dict[str, Any],
    grade: int,
    subject: Optional[str] = None,
    api_key: Optional[str] = None,
) -> KnowledgeBase:
    """Convenience function to build knowledge constraints."""
    builder = KnowledgeBuilder(api_key=api_key)
    return await builder.build_for_question(question, grade, subject)
