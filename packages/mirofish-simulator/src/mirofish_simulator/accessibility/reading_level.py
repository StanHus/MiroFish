"""
Reading Level Analysis - Flesch-Kincaid and sentence complexity.

Provides objective, formula-based reading level assessment.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# Grade-appropriate sentence length thresholds (research-based)
# Source: Educational research on reading comprehension by grade
SENTENCE_LENGTH_THRESHOLDS = {
    1: 6,
    2: 8,
    3: 10,
    4: 12,
    5: 14,
    6: 16,
    7: 18,
    8: 20,
    9: 22,
    10: 24,
    11: 26,
    12: 28,
}

# Clause count thresholds by grade
CLAUSE_THRESHOLDS = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4,
    12: 4,
}


@dataclass
class SentenceAnalysis:
    """Analysis of a single sentence."""

    text: str
    word_count: int
    clause_count: int
    syllable_count: int
    has_passive_voice: bool
    has_complex_punctuation: bool  # semicolons, colons, dashes
    issues: List[str] = field(default_factory=list)

    def is_appropriate_for_grade(self, grade: int) -> bool:
        """Check if sentence is appropriate for target grade."""
        max_words = SENTENCE_LENGTH_THRESHOLDS.get(grade, 28)
        max_clauses = CLAUSE_THRESHOLDS.get(grade, 4)

        return self.word_count <= max_words and self.clause_count <= max_clauses


@dataclass
class ReadingLevelResult:
    """Complete reading level analysis."""

    # Flesch-Kincaid metrics
    flesch_kincaid_grade: float
    flesch_reading_ease: float

    # Aggregate stats
    total_words: int
    total_sentences: int
    total_syllables: int
    avg_words_per_sentence: float
    avg_syllables_per_word: float

    # Target comparison
    target_grade: int
    grade_gap: float  # Positive = too advanced
    verdict: str  # "appropriate", "slightly_advanced", "too_advanced", "too_simple"

    # Sentence-level analysis
    sentences: List[SentenceAnalysis]
    problematic_sentences: List[SentenceAnalysis]

    def to_dict(self) -> dict:
        return {
            "measured_grade": round(self.flesch_kincaid_grade, 1),
            "target_grade": self.target_grade,
            "gap": round(self.grade_gap, 1),
            "verdict": self.verdict,
            "components": {
                "flesch_kincaid": round(self.flesch_kincaid_grade, 1),
                "flesch_reading_ease": round(self.flesch_reading_ease, 1),
                "avg_sentence_length": round(self.avg_words_per_sentence, 1),
                "avg_syllables_per_word": round(self.avg_syllables_per_word, 2),
            },
            "sentence_issues": [
                {
                    "sentence": s.text[:100] + ("..." if len(s.text) > 100 else ""),
                    "word_count": s.word_count,
                    "issues": s.issues,
                }
                for s in self.problematic_sentences
            ],
        }


# ── Syllable Counting ───────────────────────────────────────────────────────

# Common word endings that don't add syllables
SILENT_ENDINGS = {"e", "es", "ed"}

# Vowel patterns
VOWELS = set("aeiouy")

# Words with irregular syllable counts
SYLLABLE_OVERRIDES = {
    "area": 3,
    "idea": 3,
    "real": 2,
    "ruin": 2,
    "poem": 2,
    "lion": 2,
    "riot": 2,
    "diet": 2,
    "fuel": 2,
    "dual": 2,
    "cruel": 2,
    "fluid": 2,
    "quiet": 2,
    "science": 2,
    "being": 2,
    "seeing": 2,
    "agreeing": 3,
    "create": 2,
    "created": 3,
    "people": 2,
    "business": 3,
    "different": 3,
    "every": 3,
    "family": 3,
    "beautiful": 4,
    "interesting": 4,
    "comfortable": 4,
    "vegetable": 4,
    "chocolate": 3,
    "camera": 3,
    "favorite": 3,
    "separate": 3,
    "temperature": 4,
    "usually": 4,
    "actually": 4,
    "naturally": 4,
    "especially": 5,
    "extraordinary": 6,
}


def count_syllables(word: str) -> int:
    """
    Count syllables in a word using a rule-based approach.

    This is surprisingly hard to do perfectly, but this implementation
    handles most English words correctly.
    """
    word = word.lower().strip()

    # Check overrides first
    if word in SYLLABLE_OVERRIDES:
        return SYLLABLE_OVERRIDES[word]

    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)

    if not word:
        return 0

    if len(word) <= 2:
        return 1

    # Count vowel groups
    syllables = 0
    prev_was_vowel = False

    for i, char in enumerate(word):
        is_vowel = char in VOWELS

        if is_vowel and not prev_was_vowel:
            syllables += 1

        prev_was_vowel = is_vowel

    # Adjustments for common patterns

    # Silent e at end
    if word.endswith("e") and len(word) > 2 and word[-2] not in VOWELS:
        syllables -= 1

    # -le at end usually adds a syllable (e.g., "table")
    if word.endswith("le") and len(word) > 2 and word[-3] not in VOWELS:
        syllables += 1

    # -ed ending: only adds syllable if preceded by t or d
    if word.endswith("ed") and len(word) > 2:
        if word[-3] not in "td":
            syllables -= 1

    # Ensure at least 1 syllable
    return max(1, syllables)


def count_syllables_in_text(text: str) -> int:
    """Count total syllables in a text."""
    words = re.findall(r"[a-zA-Z]+", text)
    return sum(count_syllables(word) for word in words)


# ── Flesch-Kincaid Calculations ─────────────────────────────────────────────


def flesch_kincaid_grade(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.

    Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Returns the US grade level required to understand the text.
    """
    sentences = split_sentences(text)
    words = re.findall(r"[a-zA-Z]+", text)

    if not sentences or not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)

    # Flesch-Kincaid Grade Level formula
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    # Clamp to reasonable range
    return max(0.0, min(18.0, grade))


def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.

    Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

    Score interpretation:
    - 90-100: Very easy (5th grade)
    - 80-89: Easy (6th grade)
    - 70-79: Fairly easy (7th grade)
    - 60-69: Standard (8th-9th grade)
    - 50-59: Fairly difficult (10th-12th grade)
    - 30-49: Difficult (college)
    - 0-29: Very difficult (college graduate)
    """
    sentences = split_sentences(text)
    words = re.findall(r"[a-zA-Z]+", text)

    if not sentences or not words:
        return 100.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease formula
    ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

    # Clamp to 0-100
    return max(0.0, min(100.0, ease))


# ── Sentence Analysis ───────────────────────────────────────────────────────


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", r"\1<PERIOD>", text)

    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]+", text)

    # Restore periods in abbreviations
    sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences]

    # Filter empty sentences
    return [s for s in sentences if s and len(s.split()) > 0]


def count_clauses(sentence: str) -> int:
    """
    Estimate the number of clauses in a sentence.

    Looks for clause indicators like conjunctions, relative pronouns, etc.
    """
    # Start with 1 for the main clause
    clauses = 1

    # Coordinating conjunctions that typically join clauses
    coord_conjunctions = [" and ", " but ", " or ", " yet ", " so ", " nor "]
    for conj in coord_conjunctions:
        clauses += sentence.lower().count(conj)

    # Subordinating conjunctions and relative pronouns
    subord_patterns = [
        r"\bwhich\b",
        r"\bwho\b",
        r"\bwhom\b",
        r"\bwhose\b",
        r"\bthat\b",
        r"\bwhere\b",
        r"\bwhen\b",
        r"\bwhile\b",
        r"\bbecause\b",
        r"\balthough\b",
        r"\bthough\b",
        r"\bif\b",
        r"\bunless\b",
        r"\bafter\b",
        r"\bbefore\b",
        r"\bsince\b",
        r"\buntil\b",
    ]

    for pattern in subord_patterns:
        matches = re.findall(pattern, sentence.lower())
        clauses += len(matches)

    # Commas often indicate clause boundaries (rough heuristic)
    comma_count = sentence.count(",")
    if comma_count > 0:
        # Don't over-count - commas in lists don't mean clauses
        clauses += min(comma_count, 2)

    return clauses


def has_passive_voice(sentence: str) -> bool:
    """Check if sentence uses passive voice."""
    # Simple pattern: "is/was/were/been" + past participle
    passive_patterns = [
        r"\b(is|are|was|were|been|being|be)\s+\w+ed\b",
        r"\b(is|are|was|were|been|being|be)\s+\w+en\b",
    ]

    for pattern in passive_patterns:
        if re.search(pattern, sentence.lower()):
            return True

    return False


def analyze_sentence(sentence: str, target_grade: int) -> SentenceAnalysis:
    """Analyze a single sentence for complexity."""
    words = re.findall(r"[a-zA-Z]+", sentence)
    word_count = len(words)
    syllable_count = sum(count_syllables(w) for w in words)
    clause_count = count_clauses(sentence)
    passive = has_passive_voice(sentence)
    complex_punct = bool(re.search(r"[;:\-\u2014]", sentence))

    issues = []

    # Check word count
    max_words = SENTENCE_LENGTH_THRESHOLDS.get(target_grade, 28)
    if word_count > max_words:
        issues.append(f"too_long: {word_count} words (max {max_words} for grade {target_grade})")

    # Check clause count
    max_clauses = CLAUSE_THRESHOLDS.get(target_grade, 4)
    if clause_count > max_clauses:
        issues.append(f"too_complex: {clause_count} clauses (max {max_clauses} for grade {target_grade})")

    # Passive voice is harder for young readers
    if passive and target_grade < 6:
        issues.append("passive_voice: harder for young readers")

    # Complex punctuation
    if complex_punct and target_grade < 5:
        issues.append("complex_punctuation: semicolons/colons harder for young readers")

    return SentenceAnalysis(
        text=sentence,
        word_count=word_count,
        clause_count=clause_count,
        syllable_count=syllable_count,
        has_passive_voice=passive,
        has_complex_punctuation=complex_punct,
        issues=issues,
    )


# ── Main Analyzer ───────────────────────────────────────────────────────────


class ReadingLevelAnalyzer:
    """Analyzes reading level of educational content."""

    def analyze(self, text: str, target_grade: int) -> ReadingLevelResult:
        """
        Analyze reading level of text for a target grade.

        Args:
            text: The text to analyze
            target_grade: Target grade level (1-12)

        Returns:
            ReadingLevelResult with comprehensive analysis
        """
        # Calculate Flesch-Kincaid metrics
        fk_grade = flesch_kincaid_grade(text)
        fk_ease = flesch_reading_ease(text)

        # Get aggregate stats
        sentences = split_sentences(text)
        words = re.findall(r"[a-zA-Z]+", text)
        syllables = sum(count_syllables(w) for w in words)

        total_words = len(words)
        total_sentences = max(1, len(sentences))
        avg_words = total_words / total_sentences
        avg_syllables = syllables / max(1, total_words)

        # Analyze each sentence
        sentence_analyses = [analyze_sentence(s, target_grade) for s in sentences]

        # Find problematic sentences
        problematic = [s for s in sentence_analyses if s.issues]

        # Determine verdict
        grade_gap = fk_grade - target_grade
        if grade_gap <= -2:
            verdict = "too_simple"
        elif grade_gap <= 0.5:
            verdict = "appropriate"
        elif grade_gap <= 2:
            verdict = "slightly_advanced"
        else:
            verdict = "too_advanced"

        return ReadingLevelResult(
            flesch_kincaid_grade=fk_grade,
            flesch_reading_ease=fk_ease,
            total_words=total_words,
            total_sentences=total_sentences,
            total_syllables=syllables,
            avg_words_per_sentence=avg_words,
            avg_syllables_per_word=avg_syllables,
            target_grade=target_grade,
            grade_gap=grade_gap,
            verdict=verdict,
            sentences=sentence_analyses,
            problematic_sentences=problematic,
        )


def analyze_reading_level(text: str, target_grade: int) -> ReadingLevelResult:
    """Convenience function for reading level analysis."""
    analyzer = ReadingLevelAnalyzer()
    return analyzer.analyze(text, target_grade)
