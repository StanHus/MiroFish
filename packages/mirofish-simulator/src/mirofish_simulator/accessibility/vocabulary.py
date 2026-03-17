"""
Vocabulary Analysis - Identify words above target grade level.

Analyzes text to find vocabulary that may be too advanced for
the target audience and suggests simpler alternatives.
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .vocabulary_data import (
    VOCABULARY_DATABASE,
    VOCABULARY_SUGGESTIONS,
    get_word_grade,
    get_suggestion,
)


@dataclass
class VocabularyIssue:
    """A vocabulary word that may be too advanced."""

    word: str
    grade_required: int
    target_grade: int
    gap: int  # How many grades above target
    context: str  # Sentence containing the word
    suggestions: List[str]

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "grade_required": self.grade_required,
            "target_grade": self.target_grade,
            "gap": self.gap,
            "context": self.context[:100] + ("..." if len(self.context) > 100 else ""),
            "suggestions": self.suggestions,
        }


@dataclass
class VocabularyAnalysisResult:
    """Result of vocabulary analysis."""

    total_words: int
    unique_words: int
    issues: List[VocabularyIssue]
    words_above_grade: int
    avg_grade_level: float
    target_grade: int
    verdict: str  # "appropriate", "slightly_advanced", "too_advanced"

    def to_dict(self) -> dict:
        return {
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "words_above_grade": self.words_above_grade,
            "avg_grade_level": round(self.avg_grade_level, 1),
            "target_grade": self.target_grade,
            "verdict": self.verdict,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def extract_words(text: str) -> List[str]:
    """Extract individual words from text."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def get_word_context(text: str, word: str) -> str:
    """Get the sentence containing a word."""
    # Split into sentences
    sentences = re.split(r"[.!?]+", text)

    word_lower = word.lower()
    for sentence in sentences:
        if word_lower in sentence.lower():
            return sentence.strip()

    return text[:100]  # Fallback to first 100 chars


def get_word_grade_level(word: str) -> Optional[int]:
    """
    Get the grade level for a word.

    Returns None if word is not in database (assume grade-appropriate).
    """
    return VOCABULARY_DATABASE.get(word.lower())


class VocabularyAnalyzer:
    """Analyzes vocabulary complexity for target grade levels."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize vocabulary analyzer.

        Args:
            api_key: OpenAI API key for LLM fallback
            base_url: Optional base URL for API
            model: Model to use for LLM analysis
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def analyze(self, text: str, target_grade: int) -> VocabularyAnalysisResult:
        """
        Analyze vocabulary complexity for target grade.

        Args:
            text: Text to analyze
            target_grade: Target grade level (1-12)

        Returns:
            VocabularyAnalysisResult with issues and metrics
        """
        words = extract_words(text)
        unique_words = set(words)

        issues = []
        grade_levels = []
        words_above = 0

        # Analyze each unique word
        for word in unique_words:
            # Skip very short words (likely common)
            if len(word) <= 2:
                continue

            grade = get_word_grade_level(word)

            if grade is not None:
                grade_levels.append(grade)

                if grade > target_grade:
                    words_above += 1
                    gap = grade - target_grade

                    # Only flag significant gaps (2+ grades)
                    if gap >= 2:
                        suggestions = get_suggestion(word)
                        context = get_word_context(text, word)

                        issues.append(VocabularyIssue(
                            word=word,
                            grade_required=grade,
                            target_grade=target_grade,
                            gap=gap,
                            context=context,
                            suggestions=suggestions,
                        ))

        # Calculate average grade level
        if grade_levels:
            avg_grade = sum(grade_levels) / len(grade_levels)
        else:
            avg_grade = target_grade  # Assume appropriate if no data

        # Sort issues by gap (most problematic first)
        issues.sort(key=lambda x: x.gap, reverse=True)

        # Determine verdict
        if len(issues) == 0:
            verdict = "appropriate"
        elif len(issues) <= 2 and all(i.gap <= 3 for i in issues):
            verdict = "slightly_advanced"
        else:
            verdict = "too_advanced"

        return VocabularyAnalysisResult(
            total_words=len(words),
            unique_words=len(unique_words),
            issues=issues,
            words_above_grade=words_above,
            avg_grade_level=avg_grade,
            target_grade=target_grade,
            verdict=verdict,
        )

    async def analyze_with_llm(
        self,
        text: str,
        target_grade: int,
        unknown_words: List[str],
    ) -> List[VocabularyIssue]:
        """
        Use LLM to analyze words not in database.

        Args:
            text: Original text
            target_grade: Target grade
            unknown_words: Words not in vocabulary database

        Returns:
            List of VocabularyIssue for problematic words
        """
        if not self.client or not unknown_words:
            return []

        prompt = f"""Analyze these words for a grade {target_grade} student.

Words to analyze:
{', '.join(unknown_words[:20])}

For each word, determine:
1. What grade level is this word typically introduced?
2. Would a grade {target_grade} student know this word?
3. If too advanced, suggest a simpler alternative.

Return JSON array:
[
  {{"word": "example", "grade_level": 7, "too_advanced": true, "suggestion": "simpler word"}}
]

Only include words that ARE too advanced (grade_level > {target_grade}).
Return empty array [] if all words are appropriate."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing vocabulary for grade-level appropriateness. "
                            "Be accurate about when words are typically taught in US schools. "
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

            issues = []
            for item in analysis:
                if item.get("too_advanced"):
                    word = item["word"]
                    grade = item.get("grade_level", target_grade + 2)
                    suggestion = item.get("suggestion", "")

                    issues.append(VocabularyIssue(
                        word=word,
                        grade_required=grade,
                        target_grade=target_grade,
                        gap=grade - target_grade,
                        context=get_word_context(text, word),
                        suggestions=[suggestion] if suggestion else [],
                    ))

            return issues

        except Exception:
            return []

    async def full_analysis(
        self,
        text: str,
        target_grade: int,
    ) -> VocabularyAnalysisResult:
        """
        Full vocabulary analysis including LLM for unknown words.

        Args:
            text: Text to analyze
            target_grade: Target grade level

        Returns:
            VocabularyAnalysisResult with comprehensive analysis
        """
        # First, do database analysis
        result = self.analyze(text, target_grade)

        # Find words not in database
        words = extract_words(text)
        unique_words = set(words)
        unknown_words = [w for w in unique_words if get_word_grade_level(w) is None and len(w) > 3]

        # Use LLM to analyze unknown words
        if unknown_words and self.client:
            llm_issues = await self.analyze_with_llm(text, target_grade, unknown_words)
            result.issues.extend(llm_issues)
            result.issues.sort(key=lambda x: x.gap, reverse=True)

            # Update verdict if LLM found more issues
            if len(result.issues) > 2:
                result.verdict = "too_advanced"

        return result


def analyze_vocabulary(text: str, target_grade: int) -> VocabularyAnalysisResult:
    """Convenience function for vocabulary analysis."""
    analyzer = VocabularyAnalyzer()
    return analyzer.analyze(text, target_grade)
