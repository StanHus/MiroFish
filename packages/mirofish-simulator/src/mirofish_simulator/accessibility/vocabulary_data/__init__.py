"""
Vocabulary Database - Word to Grade Level Mapping.

This module contains curated vocabulary lists mapping words to the grade
level at which they are typically introduced in US education.

Sources:
- Dolch Basic Sight Words (K-3)
- Fry Sight Words (K-5)
- Academic Word List (6-12)
- Subject-specific vocabulary from curriculum standards
"""

from .core_words import CORE_VOCABULARY
from .academic import ACADEMIC_VOCABULARY
from .subject_specific import (
    MATH_VOCABULARY,
    SCIENCE_VOCABULARY,
    SOCIAL_STUDIES_VOCABULARY,
    ELA_VOCABULARY,
    AP_GOVERNMENT_VOCABULARY,
)
from .suggestions import VOCABULARY_SUGGESTIONS

# Combine all vocabulary into unified database
VOCABULARY_DATABASE: dict[str, int] = {}
VOCABULARY_DATABASE.update(CORE_VOCABULARY)
VOCABULARY_DATABASE.update(ACADEMIC_VOCABULARY)
VOCABULARY_DATABASE.update(MATH_VOCABULARY)
VOCABULARY_DATABASE.update(SCIENCE_VOCABULARY)
VOCABULARY_DATABASE.update(SOCIAL_STUDIES_VOCABULARY)
VOCABULARY_DATABASE.update(ELA_VOCABULARY)
VOCABULARY_DATABASE.update(AP_GOVERNMENT_VOCABULARY)


def get_word_grade(word: str) -> int | None:
    """
    Get the grade level for a word.

    Returns None if word is not in database.
    """
    return VOCABULARY_DATABASE.get(word.lower())


def get_suggestion(word: str) -> list[str]:
    """Get simpler alternatives for a word."""
    return VOCABULARY_SUGGESTIONS.get(word.lower(), [])


__all__ = [
    "VOCABULARY_DATABASE",
    "VOCABULARY_SUGGESTIONS",
    "get_word_grade",
    "get_suggestion",
    "CORE_VOCABULARY",
    "ACADEMIC_VOCABULARY",
    "MATH_VOCABULARY",
    "SCIENCE_VOCABULARY",
    "SOCIAL_STUDIES_VOCABULARY",
    "ELA_VOCABULARY",
    "AP_GOVERNMENT_VOCABULARY",
]
