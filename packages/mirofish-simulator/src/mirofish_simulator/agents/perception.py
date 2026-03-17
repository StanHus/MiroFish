"""
Perception Filter - How a student reads and understands text.

A 3rd grader doesn't just know less - they PERCEIVE text differently:
- Words they don't know become noise
- Long sentences blur together
- They focus on familiar words and guess the rest

This module transforms questions into what the student actually "sees".
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# Import vocabulary data for grade-level word checking
from ..accessibility.vocabulary_data import VOCABULARY_DATABASE


# ── Archetype Perception Patterns ───────────────────────────────────────────

ARCHETYPE_PERCEPTION = {
    "honors_overachiever": {
        "reading_care": 1.0,      # Reads very carefully
        "vocabulary_boost": 2,    # Knows words 2 grades above
        "attention_span": 1.0,    # Full attention
        "skips_options": False,
        "rushes": False,
    },
    "debate_club_kid": {
        "reading_care": 0.9,
        "vocabulary_boost": 1,
        "attention_span": 0.95,
        "skips_options": False,
        "rushes": False,
    },
    "quiet_thinker": {
        "reading_care": 0.85,
        "vocabulary_boost": 0,
        "attention_span": 0.9,
        "skips_options": False,
        "rushes": False,
    },
    "socially_engaged_activist": {
        "reading_care": 0.7,
        "vocabulary_boost": 0,
        "attention_span": 0.8,
        "skips_options": False,
        "rushes": True,  # If topic isn't engaging
    },
    "disengaged_but_smart": {
        "reading_care": 0.5,
        "vocabulary_boost": 1,
        "attention_span": 0.5,
        "skips_options": True,   # Might not read all options
        "rushes": True,
    },
    "esl_student": {
        "reading_care": 0.9,      # Tries hard
        "vocabulary_boost": -2,   # Needs simpler words
        "attention_span": 0.85,
        "skips_options": False,
        "rushes": False,
        "struggles_with": ["idioms", "passive_voice", "long_sentences"],
    },
    "class_clown": {
        "reading_care": 0.3,
        "vocabulary_boost": 0,
        "attention_span": 0.3,
        "skips_options": True,
        "rushes": True,
    },
    "politically_conservative": {
        "reading_care": 0.8,
        "vocabulary_boost": 0,
        "attention_span": 0.85,
        "skips_options": False,
        "rushes": False,
    },
}


@dataclass
class PerceivedContent:
    """What the student actually perceives when reading."""

    # Original
    original_text: str
    original_options: List[str]

    # What they "see"
    perceived_text: str
    perceived_options: List[str]

    # What got lost/changed
    unknown_words: List[str]           # Words they don't know
    misread_words: List[Tuple[str, str]]  # (original, what they read)
    skipped_parts: List[str]           # Parts they didn't really read
    focused_words: List[str]           # Words they latched onto

    # Reading quality
    comprehension_level: float  # 0-1, how much they understood
    confidence_in_reading: str  # "confident", "uncertain", "confused"

    def to_prompt_section(self) -> str:
        """Generate prompt section describing what the student perceived."""
        sections = []

        sections.append("=== WHAT YOU READ ===")
        sections.append(f"Question (as you understood it): {self.perceived_text}")

        sections.append("\nOptions (as you read them):")
        for i, opt in enumerate(self.perceived_options):
            sections.append(f"  {chr(65+i)}) {opt}")

        if self.unknown_words:
            sections.append(f"\nWords you didn't understand: {', '.join(self.unknown_words)}")

        if self.focused_words:
            sections.append(f"\nWords that stood out to you: {', '.join(self.focused_words)}")

        if self.skipped_parts:
            sections.append(f"\nParts you skimmed over: {', '.join(self.skipped_parts[:2])}")

        sections.append(f"\nYour reading confidence: {self.confidence_in_reading}")

        return "\n".join(sections)


class PerceptionFilter:
    """Filters questions through student perception."""

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

    async def apply(
        self,
        question: Dict[str, Any],
        grade: int,
        archetype: str,
    ) -> PerceivedContent:
        """
        Apply perception filter to a question.

        Returns what the student actually perceives when reading.
        """
        text = question.get("text") or question.get("question", "")
        options = question.get("options", [])

        # Get archetype perception pattern
        pattern = ARCHETYPE_PERCEPTION.get(archetype, ARCHETYPE_PERCEPTION["quiet_thinker"])

        # Effective vocabulary grade
        effective_vocab_grade = grade + pattern.get("vocabulary_boost", 0)

        # Find unknown words
        unknown_words = self._find_unknown_words(text, options, effective_vocab_grade)

        # Use LLM for detailed perception modeling
        if self.client:
            return await self._apply_with_llm(
                text, options, grade, archetype, pattern, unknown_words
            )

        # Fallback: basic perception
        return self._apply_basic(text, options, grade, archetype, pattern, unknown_words)

    def _find_unknown_words(
        self,
        text: str,
        options: List[str],
        effective_grade: int,
    ) -> List[str]:
        """Find words above the student's vocabulary level."""
        all_text = text + " " + " ".join(str(o) for o in options)
        words = re.findall(r"[a-zA-Z]+", all_text.lower())
        unique_words = set(words)

        unknown = []
        for word in unique_words:
            if len(word) <= 3:
                continue
            grade_level = VOCABULARY_DATABASE.get(word)
            if grade_level is not None and grade_level > effective_grade:
                unknown.append(word)

        return unknown

    def _apply_basic(
        self,
        text: str,
        options: List[str],
        grade: int,
        archetype: str,
        pattern: Dict,
        unknown_words: List[str],
    ) -> PerceivedContent:
        """Basic perception without LLM."""

        # Replace unknown words with [???]
        perceived_text = text
        for word in unknown_words:
            perceived_text = re.sub(
                rf"\b{word}\b",
                f"[???]",
                perceived_text,
                flags=re.IGNORECASE
            )

        # Handle options
        perceived_options = []
        for opt in options:
            p_opt = str(opt)
            for word in unknown_words:
                p_opt = re.sub(rf"\b{word}\b", "[???]", p_opt, flags=re.IGNORECASE)
            perceived_options.append(p_opt)

        # Skip some options if archetype pattern says so
        if pattern.get("skips_options") and len(perceived_options) > 2:
            # Mark later options as "skimmed"
            perceived_options[-1] = f"(skimmed) {perceived_options[-1]}"

        # Comprehension based on unknown words
        unknown_ratio = len(unknown_words) / max(1, len(text.split()))
        comprehension = max(0.2, 1.0 - unknown_ratio * 3)

        # Confidence
        if comprehension > 0.8:
            confidence = "confident"
        elif comprehension > 0.5:
            confidence = "uncertain"
        else:
            confidence = "confused"

        return PerceivedContent(
            original_text=text,
            original_options=options,
            perceived_text=perceived_text,
            perceived_options=perceived_options,
            unknown_words=unknown_words,
            misread_words=[],
            skipped_parts=[],
            focused_words=[],
            comprehension_level=comprehension,
            confidence_in_reading=confidence,
        )

    async def _apply_with_llm(
        self,
        text: str,
        options: List[str],
        grade: int,
        archetype: str,
        pattern: Dict,
        unknown_words: List[str],
    ) -> PerceivedContent:
        """Use LLM for realistic perception modeling."""

        options_text = "\n".join(f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options))
        unknown_text = ", ".join(unknown_words) if unknown_words else "none identified"

        attention = pattern.get("attention_span", 0.8)
        rushes = pattern.get("rushes", False)
        reading_care = pattern.get("reading_care", 0.7)

        prompt = f"""Model how a grade {grade} student with "{archetype}" personality would PERCEIVE this question.

ORIGINAL QUESTION:
{text}

ORIGINAL OPTIONS:
{options_text}

WORDS ABOVE THEIR VOCABULARY LEVEL:
{unknown_text}

STUDENT READING PATTERN:
- Attention span: {attention:.0%}
- Reading carefulness: {reading_care:.0%}
- Tends to rush: {rushes}

Create a "perceived version" - what do they ACTUALLY see/understand when reading?

Consider:
- Unknown words become "something about X" or get misinterpreted
- Long parts might get skimmed
- They focus on words they recognize
- Their archetype affects reading style

Return JSON:
{{
  "perceived_text": "What they think the question asks (may be simplified or wrong)",
  "perceived_options": ["option A as understood", "option B as understood", "option C", "option D"],
  "misread_words": [["original", "what they thought it said"]],
  "skipped_parts": ["parts they didn't really read"],
  "focused_words": ["words they latched onto"],
  "comprehension_level": 0.6,
  "confidence": "uncertain"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are modeling student perception. Create realistic 'perceived' versions "
                            "of questions as students would actually read them. Students don't read perfectly - "
                            "they skip, misread, focus on familiar words, and get confused by hard words. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=600,
            )

            result_text = response.choices[0].message.content.strip()

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            return PerceivedContent(
                original_text=text,
                original_options=options,
                perceived_text=analysis.get("perceived_text", text),
                perceived_options=analysis.get("perceived_options", options),
                unknown_words=unknown_words,
                misread_words=[tuple(m) for m in analysis.get("misread_words", [])],
                skipped_parts=analysis.get("skipped_parts", []),
                focused_words=analysis.get("focused_words", []),
                comprehension_level=analysis.get("comprehension_level", 0.7),
                confidence_in_reading=analysis.get("confidence", "uncertain"),
            )

        except Exception:
            return self._apply_basic(text, options, grade, archetype, pattern, unknown_words)


async def apply_perception_filter(
    question: Dict[str, Any],
    grade: int,
    archetype: str,
    api_key: Optional[str] = None,
) -> PerceivedContent:
    """Convenience function to apply perception filter."""
    filter = PerceptionFilter(api_key=api_key)
    return await filter.apply(question, grade, archetype)
