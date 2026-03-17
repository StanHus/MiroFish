"""
Archetype Modifiers - Adjust accessibility for different student types.

Different student archetypes experience content differently:
- ESL students struggle more with vocabulary
- Class clowns struggle with long content
- Honors students can handle above-grade content
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Archetype Modifier Definitions ──────────────────────────────────────────

@dataclass
class ArchetypeModifier:
    """Modifiers for a student archetype."""

    name: str
    display_name: str
    vocabulary_tolerance: int  # + = can handle higher, - = needs lower
    sentence_tolerance: int    # Same for sentence length
    attention_factor: float    # 1.0 = normal, 0.5 = loses focus easily
    extra_barriers: List[str]  # Additional challenges

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "vocabulary_tolerance": self.vocabulary_tolerance,
            "sentence_tolerance": self.sentence_tolerance,
            "attention_factor": self.attention_factor,
            "extra_barriers": self.extra_barriers,
        }


ARCHETYPE_MODIFIERS = {
    "honors_overachiever": ArchetypeModifier(
        name="honors_overachiever",
        display_name="Honors Overachiever",
        vocabulary_tolerance=+2,  # Can handle 2 grades above
        sentence_tolerance=+3,
        attention_factor=1.2,     # Reads more carefully
        extra_barriers=[],
    ),
    "debate_club_kid": ArchetypeModifier(
        name="debate_club_kid",
        display_name="Debate Club Kid",
        vocabulary_tolerance=+1,
        sentence_tolerance=+2,
        attention_factor=1.1,
        extra_barriers=["may overthink simple questions"],
    ),
    "quiet_thinker": ArchetypeModifier(
        name="quiet_thinker",
        display_name="Quiet Thinker",
        vocabulary_tolerance=0,
        sentence_tolerance=0,
        attention_factor=1.0,
        extra_barriers=["may second-guess correct answers"],
    ),
    "socially_engaged_activist": ArchetypeModifier(
        name="socially_engaged_activist",
        display_name="Socially Engaged Activist",
        vocabulary_tolerance=0,
        sentence_tolerance=-1,
        attention_factor=0.9,
        extra_barriers=["may rush through non-social topics"],
    ),
    "disengaged_but_smart": ArchetypeModifier(
        name="disengaged_but_smart",
        display_name="Disengaged but Smart",
        vocabulary_tolerance=+1,
        sentence_tolerance=0,
        attention_factor=0.6,     # Low attention
        extra_barriers=["long questions", "boring topics", "may skip reading options"],
    ),
    "esl_student": ArchetypeModifier(
        name="esl_student",
        display_name="ESL Student",
        vocabulary_tolerance=-2,  # Needs 2 grades below
        sentence_tolerance=-2,
        attention_factor=0.9,     # Tries hard but limited
        extra_barriers=["idioms", "cultural_references", "passive_voice", "complex_grammar"],
    ),
    "class_clown": ArchetypeModifier(
        name="class_clown",
        display_name="Class Clown",
        vocabulary_tolerance=0,
        sentence_tolerance=-3,    # Very short attention span
        attention_factor=0.4,     # Barely reads
        extra_barriers=["long_questions", "abstract_topics", "multiple_steps"],
    ),
    "politically_conservative": ArchetypeModifier(
        name="politically_conservative",
        display_name="Politically Conservative",
        vocabulary_tolerance=0,
        sentence_tolerance=0,
        attention_factor=0.95,
        extra_barriers=["may resist certain framings"],
    ),
}


@dataclass
class ArchetypeAccessibility:
    """Accessibility result for a specific archetype."""

    archetype: str
    display_name: str
    base_score: float           # Score without modifiers
    adjusted_score: float       # Score with archetype modifiers
    vocabulary_adjusted: bool   # Was vocab threshold adjusted?
    sentence_adjusted: bool     # Was sentence threshold adjusted?
    additional_barriers: List[str]  # Archetype-specific barriers present
    verdict: str                # "accessible", "challenging", "inaccessible"

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "display_name": self.display_name,
            "accessibility_score": round(self.adjusted_score, 2),
            "base_score": round(self.base_score, 2),
            "additional_barriers": self.additional_barriers,
            "verdict": self.verdict,
        }


def get_archetype_modifier(archetype: str) -> Optional[ArchetypeModifier]:
    """Get modifier for an archetype."""
    return ARCHETYPE_MODIFIERS.get(archetype)


def apply_archetype_modifiers(
    base_score: float,
    reading_grade_gap: float,
    vocabulary_issues_count: int,
    sentence_issues_count: int,
    question_length: int,
    is_abstract: bool,
    has_idioms: bool,
    has_passive_voice: bool,
    archetypes: Optional[List[str]] = None,
) -> Dict[str, ArchetypeAccessibility]:
    """
    Apply archetype modifiers to base accessibility score.

    Args:
        base_score: Base accessibility score (0-1)
        reading_grade_gap: How many grades above target
        vocabulary_issues_count: Number of vocabulary issues
        sentence_issues_count: Number of sentence issues
        question_length: Word count of question
        is_abstract: Is the question abstract?
        has_idioms: Contains idioms/expressions?
        has_passive_voice: Uses passive voice?
        archetypes: Which archetypes to check (None = all)

    Returns:
        Dict mapping archetype name to ArchetypeAccessibility
    """
    if archetypes is None:
        archetypes = list(ARCHETYPE_MODIFIERS.keys())

    results = {}

    for archetype_name in archetypes:
        modifier = ARCHETYPE_MODIFIERS.get(archetype_name)
        if not modifier:
            continue

        adjusted_score = base_score
        barriers = []
        vocab_adjusted = False
        sentence_adjusted = False

        # Apply vocabulary tolerance
        if vocabulary_issues_count > 0:
            vocab_adjusted = True
            if modifier.vocabulary_tolerance > 0:
                # Can handle harder vocab
                adjusted_score += 0.1 * modifier.vocabulary_tolerance
            else:
                # Struggles more with vocab
                adjusted_score -= 0.1 * abs(modifier.vocabulary_tolerance)

        # Apply sentence tolerance
        if sentence_issues_count > 0:
            sentence_adjusted = True
            if modifier.sentence_tolerance > 0:
                adjusted_score += 0.05 * modifier.sentence_tolerance
            else:
                adjusted_score -= 0.1 * abs(modifier.sentence_tolerance)

        # Apply attention factor
        if question_length > 50:  # Long question
            adjusted_score *= modifier.attention_factor

        # Check extra barriers
        if "long_questions" in modifier.extra_barriers and question_length > 30:
            barriers.append("Question is too long for this learner")
            adjusted_score -= 0.15

        if "abstract_topics" in modifier.extra_barriers and is_abstract:
            barriers.append("Abstract content is challenging")
            adjusted_score -= 0.1

        if "idioms" in modifier.extra_barriers and has_idioms:
            barriers.append("Contains idioms that may be confusing")
            adjusted_score -= 0.15

        if "passive_voice" in modifier.extra_barriers and has_passive_voice:
            barriers.append("Passive voice is harder to parse")
            adjusted_score -= 0.1

        if "cultural_references" in modifier.extra_barriers:
            # Assume US civics/history content has cultural references
            if reading_grade_gap > 1:
                barriers.append("US-specific cultural knowledge may be unfamiliar")
                adjusted_score -= 0.1

        if "complex_grammar" in modifier.extra_barriers and sentence_issues_count > 1:
            barriers.append("Complex grammar structures")
            adjusted_score -= 0.1

        if "multiple_steps" in modifier.extra_barriers and reading_grade_gap > 1:
            barriers.append("Multi-step reasoning required")
            adjusted_score -= 0.1

        if "boring_topics" in modifier.extra_barriers:
            # Hard to detect automatically, skip for now
            pass

        # Clamp score
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Determine verdict
        if adjusted_score >= 0.7:
            verdict = "accessible"
        elif adjusted_score >= 0.4:
            verdict = "challenging"
        else:
            verdict = "inaccessible"

        results[archetype_name] = ArchetypeAccessibility(
            archetype=archetype_name,
            display_name=modifier.display_name,
            base_score=base_score,
            adjusted_score=adjusted_score,
            vocabulary_adjusted=vocab_adjusted,
            sentence_adjusted=sentence_adjusted,
            additional_barriers=barriers,
            verdict=verdict,
        )

    return results


def get_most_affected_archetypes(
    results: Dict[str, ArchetypeAccessibility],
    threshold: float = 0.5,
) -> List[ArchetypeAccessibility]:
    """Get archetypes most negatively affected by accessibility issues."""
    affected = [r for r in results.values() if r.adjusted_score < threshold]
    affected.sort(key=lambda r: r.adjusted_score)
    return affected


def get_archetype_summary(
    results: Dict[str, ArchetypeAccessibility],
) -> Dict[str, Any]:
    """Get summary of archetype accessibility."""
    scores = [r.adjusted_score for r in results.values()]
    verdicts = [r.verdict for r in results.values()]

    return {
        "average_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "accessible_count": sum(1 for v in verdicts if v == "accessible"),
        "challenging_count": sum(1 for v in verdicts if v == "challenging"),
        "inaccessible_count": sum(1 for v in verdicts if v == "inaccessible"),
    }
