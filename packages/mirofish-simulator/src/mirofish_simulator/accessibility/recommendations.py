"""
Recommendation Engine - Generate actionable fix suggestions.

Produces specific, actionable recommendations for making
content more accessible to the target grade level.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .reading_level import ReadingLevelResult, SENTENCE_LENGTH_THRESHOLDS
from .vocabulary import VocabularyAnalysisResult, VocabularyIssue
from .prior_knowledge import PriorKnowledgeResult
from .cognitive_load import CognitiveLoadResult


@dataclass
class Recommendation:
    """A specific recommendation for improving accessibility."""

    priority: str  # "high", "medium", "low"
    category: str  # "vocabulary", "sentence_structure", "prior_knowledge", "cognitive_load"
    issue: str     # What's wrong
    fix: str       # How to fix it
    example: Optional[str] = None  # Example of the fix

    def to_dict(self) -> dict:
        result = {
            "priority": self.priority,
            "category": self.category,
            "issue": self.issue,
            "fix": self.fix,
        }
        if self.example:
            result["example"] = self.example
        return result


class RecommendationEngine:
    """Generates actionable recommendations from analysis results."""

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

    def generate(
        self,
        reading_level: Optional[ReadingLevelResult] = None,
        vocabulary: Optional[VocabularyAnalysisResult] = None,
        prior_knowledge: Optional[PriorKnowledgeResult] = None,
        cognitive_load: Optional[CognitiveLoadResult] = None,
    ) -> List[Recommendation]:
        """
        Generate recommendations from analysis results.

        Args:
            reading_level: Reading level analysis
            vocabulary: Vocabulary analysis
            prior_knowledge: Prior knowledge analysis
            cognitive_load: Cognitive load analysis

        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []

        # Vocabulary recommendations
        if vocabulary:
            recommendations.extend(self._vocabulary_recommendations(vocabulary))

        # Reading level recommendations
        if reading_level:
            recommendations.extend(self._reading_level_recommendations(reading_level))

        # Prior knowledge recommendations
        if prior_knowledge:
            recommendations.extend(self._prior_knowledge_recommendations(prior_knowledge))

        # Cognitive load recommendations
        if cognitive_load:
            recommendations.extend(self._cognitive_load_recommendations(cognitive_load))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 2))

        return recommendations

    def _vocabulary_recommendations(
        self,
        result: VocabularyAnalysisResult,
    ) -> List[Recommendation]:
        """Generate vocabulary recommendations."""
        recs = []

        for issue in result.issues[:5]:  # Top 5 issues
            priority = "high" if issue.gap >= 4 else "medium" if issue.gap >= 2 else "low"

            if issue.suggestions:
                fix = f"Replace '{issue.word}' with: {' or '.join(issue.suggestions[:3])}"
                example = issue.context.replace(issue.word, issue.suggestions[0])
            else:
                fix = f"Replace '{issue.word}' with a simpler word"
                example = None

            recs.append(Recommendation(
                priority=priority,
                category="vocabulary",
                issue=f"'{issue.word}' requires grade {issue.grade_required} (target is grade {issue.target_grade})",
                fix=fix,
                example=example,
            ))

        return recs

    def _reading_level_recommendations(
        self,
        result: ReadingLevelResult,
    ) -> List[Recommendation]:
        """Generate reading level recommendations."""
        recs = []

        # Overall reading level
        if result.grade_gap >= 2:
            recs.append(Recommendation(
                priority="high" if result.grade_gap >= 3 else "medium",
                category="reading_level",
                issue=f"Reading level is grade {result.flesch_kincaid_grade:.1f} (target is grade {result.target_grade})",
                fix="Simplify vocabulary and shorten sentences to reduce reading level",
            ))

        # Problematic sentences
        for sentence in result.problematic_sentences[:3]:
            max_words = SENTENCE_LENGTH_THRESHOLDS.get(result.target_grade, 20)

            if "too_long" in str(sentence.issues):
                recs.append(Recommendation(
                    priority="high" if sentence.word_count > max_words + 10 else "medium",
                    category="sentence_structure",
                    issue=f"Sentence has {sentence.word_count} words (max {max_words} for grade {result.target_grade})",
                    fix="Split into shorter sentences",
                    example=f"Original: {sentence.text[:80]}...",
                ))

            if "too_complex" in str(sentence.issues):
                recs.append(Recommendation(
                    priority="medium",
                    category="sentence_structure",
                    issue=f"Sentence has {sentence.clause_count} clauses",
                    fix="Simplify sentence structure by removing embedded clauses",
                ))

            if "passive_voice" in str(sentence.issues):
                recs.append(Recommendation(
                    priority="low",
                    category="sentence_structure",
                    issue="Passive voice is harder for young readers",
                    fix="Convert to active voice",
                ))

        return recs

    def _prior_knowledge_recommendations(
        self,
        result: PriorKnowledgeResult,
    ) -> List[Recommendation]:
        """Generate prior knowledge recommendations."""
        recs = []

        for concept in result.required_concepts:
            if concept.gap > 0:
                priority = "high" if concept.gap >= 3 else "medium"

                recs.append(Recommendation(
                    priority=priority,
                    category="prior_knowledge",
                    issue=f"Requires knowledge of '{concept.concept}' (typically taught grade {concept.typically_taught})",
                    fix=f"Add scaffolding or context to explain '{concept.concept}'",
                    example=f"Consider adding: '({concept.concept} is...)'",
                ))

        return recs

    def _cognitive_load_recommendations(
        self,
        result: CognitiveLoadResult,
    ) -> List[Recommendation]:
        """Generate cognitive load recommendations."""
        recs = []

        if result.steps_verdict == "too_complex":
            recs.append(Recommendation(
                priority="high",
                category="cognitive_load",
                issue=f"Question requires {result.step_count} cognitive steps (max {result.max_steps_for_grade} for grade {result.target_grade})",
                fix="Simplify the question or break into multiple parts",
            ))
        elif result.steps_verdict == "challenging":
            recs.append(Recommendation(
                priority="medium",
                category="cognitive_load",
                issue=f"Question requires {result.step_count} cognitive steps (challenging for grade {result.target_grade})",
                fix="Consider simplifying or providing more scaffolding",
            ))

        if result.memory_verdict == "too_complex":
            recs.append(Recommendation(
                priority="high",
                category="cognitive_load",
                issue=f"Requires holding {result.working_memory_count} items in memory (max {result.max_memory_for_grade} for grade {result.target_grade})",
                fix="Reduce number of options or simplify question structure",
            ))

        if result.abstraction_level == "abstract" and result.target_grade < 4:
            recs.append(Recommendation(
                priority="high",
                category="cognitive_load",
                issue="Question is too abstract for this grade level",
                fix="Use concrete examples instead of abstract concepts",
            ))

        return recs

    async def generate_rewrite(
        self,
        original_text: str,
        target_grade: int,
        recommendations: List[Recommendation],
    ) -> str:
        """
        Generate a rewritten version of the question.

        Args:
            original_text: Original question text
            target_grade: Target grade level
            recommendations: List of recommendations to apply

        Returns:
            Rewritten text
        """
        if not self.client:
            return original_text

        # Summarize recommendations
        rec_summary = "\n".join(f"- {r.fix}" for r in recommendations[:5])

        prompt = f"""Rewrite this question for a grade {target_grade} student.

Original: {original_text}

Apply these improvements:
{rec_summary}

Requirements:
- Use vocabulary appropriate for grade {target_grade}
- Keep sentences under {SENTENCE_LENGTH_THRESHOLDS.get(target_grade, 15)} words
- Maintain the same meaning and correct answer
- Add clarifying context if needed

Return ONLY the rewritten question, no explanation."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at writing grade-appropriate educational content. "
                            "Rewrite questions to be accessible while maintaining accuracy."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            return original_text


def generate_recommendations(
    reading_level: Optional[ReadingLevelResult] = None,
    vocabulary: Optional[VocabularyAnalysisResult] = None,
    prior_knowledge: Optional[PriorKnowledgeResult] = None,
    cognitive_load: Optional[CognitiveLoadResult] = None,
) -> List[Recommendation]:
    """Convenience function to generate recommendations."""
    engine = RecommendationEngine()
    return engine.generate(reading_level, vocabulary, prior_knowledge, cognitive_load)
