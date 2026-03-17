"""
AccessibilityAnalyzer - Main entry point for grade-level accessibility analysis.

Orchestrates all analyzers to provide comprehensive accessibility assessment:
- Reading level (Flesch-Kincaid)
- Vocabulary complexity
- Prior knowledge requirements
- Cognitive load
- Archetype-specific accessibility
- Actionable recommendations
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .reading_level import ReadingLevelAnalyzer, ReadingLevelResult
from .vocabulary import VocabularyAnalyzer, VocabularyAnalysisResult
from .prior_knowledge import PriorKnowledgeAnalyzer, PriorKnowledgeResult
from .cognitive_load import CognitiveLoadAnalyzer, CognitiveLoadResult
from .recommendations import RecommendationEngine, Recommendation
from .archetypes import (
    apply_archetype_modifiers,
    ArchetypeAccessibility,
    get_archetype_summary,
    ARCHETYPE_MODIFIERS,
)


class AccessibilityVerdict(Enum):
    """Overall accessibility verdict."""

    APPROPRIATE = "appropriate"
    SLIGHTLY_ADVANCED = "slightly_advanced"
    TOO_ADVANCED = "too_advanced"
    TOO_SIMPLE = "too_simple"


@dataclass
class AccessibilityResult:
    """Complete accessibility analysis result."""

    # Target
    target_grade: int
    content_text: str

    # Overall
    score: float  # 0-1, higher = more accessible
    verdict: AccessibilityVerdict

    # Components
    reading_level: ReadingLevelResult
    vocabulary: VocabularyAnalysisResult
    prior_knowledge: Optional[PriorKnowledgeResult]
    cognitive_load: Optional[CognitiveLoadResult]

    # Recommendations
    recommendations: List[Recommendation]

    # Archetype breakdown
    by_archetype: Dict[str, ArchetypeAccessibility]
    archetype_summary: Dict[str, Any]

    # Optional rewrite
    rewritten_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "target_grade": self.target_grade,
            "accessibility_score": round(self.score, 2),
            "verdict": self.verdict.value,
            "reading_level": self.reading_level.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "prior_knowledge": self.prior_knowledge.to_dict() if self.prior_knowledge else None,
            "cognitive_load": self.cognitive_load.to_dict() if self.cognitive_load else None,
            "recommendations": [r.to_dict() for r in self.recommendations[:10]],
            "by_archetype": {k: v.to_dict() for k, v in self.by_archetype.items()},
            "archetype_summary": self.archetype_summary,
            "rewritten_version": self.rewritten_version,
        }


class AccessibilityAnalyzer:
    """
    Main analyzer for grade-level content accessibility.

    Usage:
        analyzer = AccessibilityAnalyzer(api_key="sk-...")
        result = await analyzer.analyze(
            content={"text": "What is the Electoral College?", "options": [...]},
            target_grade=5,
        )

        print(f"Score: {result.score:.0%}")
        print(f"Verdict: {result.verdict.value}")
        for rec in result.recommendations:
            print(f"- {rec.fix}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize accessibility analyzer.

        Args:
            api_key: OpenAI API key (for LLM-enhanced analysis)
            base_url: Optional base URL for API
            model: Model to use for LLM analysis
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        # Initialize component analyzers
        self.reading_analyzer = ReadingLevelAnalyzer()
        self.vocabulary_analyzer = VocabularyAnalyzer(api_key, base_url, model)
        self.prior_knowledge_analyzer = PriorKnowledgeAnalyzer(api_key, base_url, model)
        self.cognitive_analyzer = CognitiveLoadAnalyzer(api_key, base_url, model)
        self.recommendation_engine = RecommendationEngine(api_key, base_url, model)

    async def analyze(
        self,
        content: Dict[str, Any],
        target_grade: Optional[int] = None,
        include_archetypes: Optional[List[str]] = None,
        include_prior_knowledge: bool = True,
        include_cognitive_load: bool = True,
        include_rewrite: bool = False,
    ) -> AccessibilityResult:
        """
        Analyze accessibility of educational content for a target grade.

        Args:
            content: Question content with "text", "options", optionally "grade"
            target_grade: Target grade (1-12), or extracted from content
            include_archetypes: Which archetypes to analyze (None = all)
            include_prior_knowledge: Whether to analyze prior knowledge
            include_cognitive_load: Whether to analyze cognitive load
            include_rewrite: Whether to generate accessible rewrite

        Returns:
            AccessibilityResult with comprehensive analysis
        """
        # Extract text and target grade
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        options_text = " ".join(str(o) for o in options)
        full_text = f"{text} {options_text}"

        if target_grade is None:
            grade_str = content.get("grade", "8")
            try:
                target_grade = int(str(grade_str).replace("th", "").replace("st", "").replace("nd", "").replace("rd", ""))
            except ValueError:
                target_grade = 8  # Default

        target_grade = max(1, min(12, target_grade))

        # Run analyses
        reading_level = self.reading_analyzer.analyze(full_text, target_grade)
        vocabulary = self.vocabulary_analyzer.analyze(full_text, target_grade)

        prior_knowledge = None
        if include_prior_knowledge and self.api_key:
            prior_knowledge = await self.prior_knowledge_analyzer.analyze(content, target_grade)

        cognitive_load = None
        if include_cognitive_load and self.api_key:
            cognitive_load = await self.cognitive_analyzer.analyze(content, target_grade)

        # Calculate base score
        score = self._calculate_score(
            reading_level=reading_level,
            vocabulary=vocabulary,
            prior_knowledge=prior_knowledge,
            cognitive_load=cognitive_load,
        )

        # Determine verdict
        verdict = self._determine_verdict(score, reading_level.grade_gap)

        # Generate recommendations
        recommendations = self.recommendation_engine.generate(
            reading_level=reading_level,
            vocabulary=vocabulary,
            prior_knowledge=prior_knowledge,
            cognitive_load=cognitive_load,
        )

        # Check for features that affect archetypes
        has_idioms = any(
            phrase in full_text.lower()
            for phrase in ["checks and balances", "piece of cake", "break the ice", "at the end of the day"]
        )
        has_passive = reading_level.sentences and any(s.has_passive_voice for s in reading_level.sentences)
        is_abstract = cognitive_load.abstraction_level == "abstract" if cognitive_load else False

        # Apply archetype modifiers
        by_archetype = apply_archetype_modifiers(
            base_score=score,
            reading_grade_gap=reading_level.grade_gap,
            vocabulary_issues_count=len(vocabulary.issues),
            sentence_issues_count=len(reading_level.problematic_sentences),
            question_length=reading_level.total_words,
            is_abstract=is_abstract,
            has_idioms=has_idioms,
            has_passive_voice=has_passive,
            archetypes=include_archetypes,
        )

        archetype_summary = get_archetype_summary(by_archetype)

        # Generate rewrite if requested
        rewritten = None
        if include_rewrite and recommendations:
            rewritten = await self.recommendation_engine.generate_rewrite(
                text, target_grade, recommendations
            )

        return AccessibilityResult(
            target_grade=target_grade,
            content_text=text,
            score=score,
            verdict=verdict,
            reading_level=reading_level,
            vocabulary=vocabulary,
            prior_knowledge=prior_knowledge,
            cognitive_load=cognitive_load,
            recommendations=recommendations,
            by_archetype=by_archetype,
            archetype_summary=archetype_summary,
            rewritten_version=rewritten,
        )

    def _calculate_score(
        self,
        reading_level: ReadingLevelResult,
        vocabulary: VocabularyAnalysisResult,
        prior_knowledge: Optional[PriorKnowledgeResult],
        cognitive_load: Optional[CognitiveLoadResult],
    ) -> float:
        """Calculate overall accessibility score (0-1)."""
        # Start with 1.0 and deduct for issues
        score = 1.0

        # Reading level impact (biggest factor)
        if reading_level.grade_gap > 0:
            # Deduct 0.15 per grade level above target
            score -= min(0.5, reading_level.grade_gap * 0.15)
        elif reading_level.grade_gap < -2:
            # Too simple is also a mild issue
            score -= 0.1

        # Vocabulary impact
        if vocabulary.issues:
            # Deduct based on number and severity of issues
            vocab_penalty = sum(min(0.1, i.gap * 0.02) for i in vocabulary.issues[:5])
            score -= min(0.3, vocab_penalty)

        # Prior knowledge impact
        if prior_knowledge and prior_knowledge.max_gap > 0:
            score -= min(0.2, prior_knowledge.max_gap * 0.05)

        # Cognitive load impact
        if cognitive_load:
            if cognitive_load.overall_verdict == "too_complex":
                score -= 0.2
            elif cognitive_load.overall_verdict == "challenging":
                score -= 0.1

        return max(0.0, min(1.0, score))

    def _determine_verdict(
        self,
        score: float,
        grade_gap: float,
    ) -> AccessibilityVerdict:
        """Determine overall verdict from score and metrics."""
        if grade_gap < -2:
            return AccessibilityVerdict.TOO_SIMPLE

        if score >= 0.8:
            return AccessibilityVerdict.APPROPRIATE
        elif score >= 0.5:
            return AccessibilityVerdict.SLIGHTLY_ADVANCED
        else:
            return AccessibilityVerdict.TOO_ADVANCED

    def analyze_sync(
        self,
        content: Dict[str, Any],
        target_grade: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous analysis without LLM features.

        For quick analysis without async/await.
        Returns dict instead of AccessibilityResult.
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        options_text = " ".join(str(o) for o in options)
        full_text = f"{text} {options_text}"

        if target_grade is None:
            grade_str = content.get("grade", "8")
            try:
                target_grade = int(str(grade_str).replace("th", "").replace("nd", "").replace("rd", "").replace("st", ""))
            except ValueError:
                target_grade = 8

        target_grade = max(1, min(12, target_grade))

        # Quick analysis (no LLM)
        reading_level = self.reading_analyzer.analyze(full_text, target_grade)
        vocabulary = self.vocabulary_analyzer.analyze(full_text, target_grade)

        score = self._calculate_score(reading_level, vocabulary, None, None)
        verdict = self._determine_verdict(score, reading_level.grade_gap)

        recommendations = self.recommendation_engine.generate(
            reading_level=reading_level,
            vocabulary=vocabulary,
        )

        return {
            "target_grade": target_grade,
            "accessibility_score": round(score, 2),
            "verdict": verdict.value,
            "reading_level": reading_level.to_dict(),
            "vocabulary": vocabulary.to_dict(),
            "recommendations": [r.to_dict() for r in recommendations[:5]],
        }


async def analyze_accessibility(
    content: Dict[str, Any],
    target_grade: Optional[int] = None,
    api_key: Optional[str] = None,
) -> AccessibilityResult:
    """Convenience function for accessibility analysis."""
    analyzer = AccessibilityAnalyzer(api_key=api_key)
    return await analyzer.analyze(content, target_grade)
