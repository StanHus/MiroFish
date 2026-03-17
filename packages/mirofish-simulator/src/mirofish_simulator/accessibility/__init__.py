"""
MiroFish Accessibility Analysis - Grade-Level Content Accessibility.

This module provides objective, measurable accessibility analysis for educational
content. It answers the question: "Is this content readable and answerable by
a student at grade X?"

Usage:
    from mirofish_simulator.accessibility import AccessibilityAnalyzer

    analyzer = AccessibilityAnalyzer()
    result = await analyzer.analyze(
        content={"text": "What is the Electoral College?", "options": [...], "grade": "5"},
    )

    print(f"Accessibility: {result.score:.0%}")
    print(f"Reading Level: Grade {result.reading_level.measured_grade}")
    for issue in result.vocabulary_issues:
        print(f"  '{issue.word}' requires grade {issue.grade_required}")
"""

from .analyzer import (
    AccessibilityAnalyzer,
    AccessibilityResult,
    AccessibilityVerdict,
    analyze_accessibility,
)

from .reading_level import (
    ReadingLevelAnalyzer,
    ReadingLevelResult,
    SentenceAnalysis,
    flesch_kincaid_grade,
    count_syllables,
)

from .vocabulary import (
    VocabularyAnalyzer,
    VocabularyIssue,
    get_word_grade_level,
)

from .prior_knowledge import (
    PriorKnowledgeAnalyzer,
    PriorKnowledgeResult,
    ConceptRequirement,
)

from .cognitive_load import (
    CognitiveLoadAnalyzer,
    CognitiveLoadResult,
)

from .recommendations import (
    RecommendationEngine,
    Recommendation,
)

from .archetypes import (
    ArchetypeModifier,
    ArchetypeAccessibility,
    apply_archetype_modifiers,
)

__all__ = [
    # Main analyzer
    "AccessibilityAnalyzer",
    "AccessibilityResult",
    "AccessibilityVerdict",
    "analyze_accessibility",
    # Reading level
    "ReadingLevelAnalyzer",
    "ReadingLevelResult",
    "SentenceAnalysis",
    "flesch_kincaid_grade",
    "count_syllables",
    # Vocabulary
    "VocabularyAnalyzer",
    "VocabularyIssue",
    "get_word_grade_level",
    # Prior knowledge
    "PriorKnowledgeAnalyzer",
    "PriorKnowledgeResult",
    "ConceptRequirement",
    # Cognitive load
    "CognitiveLoadAnalyzer",
    "CognitiveLoadResult",
    # Recommendations
    "RecommendationEngine",
    "Recommendation",
    # Archetypes
    "ArchetypeModifier",
    "ArchetypeAccessibility",
    "apply_archetype_modifiers",
]
