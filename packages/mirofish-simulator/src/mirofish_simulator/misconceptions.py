"""
Misconception analysis for educational content.

Maps distractors to specific student misconceptions, providing actionable
insights for content authors to improve question quality.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .taxonomies import get_taxonomy, SUBJECT_TAXONOMIES


@dataclass
class DistractorMisconception:
    """Analysis of a distractor's connection to student misconceptions."""

    option: str
    catches_misconception: str  # Misconception ID
    misconception_category: str
    description: str
    vulnerable_archetypes: List[str]
    remediation_standard: Optional[str] = None
    remediation_topic: Optional[str] = None
    confidence: float = 0.0  # 0-1, how confident we are in this mapping


@dataclass
class MisconceptionAnalysisResult:
    """Complete misconception analysis for a question."""

    question_text: str
    subject: str
    distractors: Dict[str, DistractorMisconception]
    unmapped_distractors: List[str]  # Distractors we couldn't map to known misconceptions
    coverage_score: float  # 0-1, how well distractors target known misconceptions
    analysis_confidence: str  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "distractors": {
                k: {
                    "option": v.option,
                    "catches_misconception": v.catches_misconception,
                    "misconception_category": v.misconception_category,
                    "description": v.description,
                    "vulnerable_archetypes": v.vulnerable_archetypes,
                    "remediation_standard": v.remediation_standard,
                    "remediation_topic": v.remediation_topic,
                    "confidence": round(v.confidence, 2),
                }
                for k, v in self.distractors.items()
            },
            "unmapped_distractors": self.unmapped_distractors,
            "coverage_score": round(self.coverage_score, 2),
            "analysis_confidence": self.analysis_confidence,
        }


class MisconceptionAnalyzer:
    """
    Analyzes educational content distractors and maps them to known misconceptions.

    This provides genuinely new insight - instead of vague statements like
    "Distractor B appeals to ESL students", we produce specific mappings like
    "Distractor B catches the 'confuses_electoral_college_with_congress' misconception".

    Usage:
        analyzer = MisconceptionAnalyzer(api_key="sk-...")
        result = await analyzer.analyze(content)

        for option, mc in result.distractors.items():
            print(f"{option}: catches '{mc.catches_misconception}'")
            print(f"   Remediation: {mc.remediation_topic}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client: Optional[AsyncOpenAI] = None

        if api_key:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def analyze(
        self,
        content: Dict[str, Any],
    ) -> MisconceptionAnalysisResult:
        """
        Analyze distractors and map them to specific misconceptions.

        Args:
            content: Educational content with keys:
                - text/question: The question text
                - options: List of answer options
                - correct_answer: Correct answer (letter or index)
                - subject: Subject area (e.g., "AP Government", "Mathematics")

        Returns:
            MisconceptionAnalysisResult with distractor-to-misconception mappings
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        correct = content.get("correct_answer", "A")
        subject = content.get("subject", "").strip()

        # Get correct letter
        if isinstance(correct, str) and correct.isalpha():
            correct_letter = correct.upper()
        else:
            try:
                correct_letter = chr(65 + int(correct))
            except (ValueError, TypeError):
                correct_letter = "A"

        # Get taxonomy for this subject
        taxonomy = get_taxonomy(subject)

        # Identify wrong answer options
        all_opts = ["A", "B", "C", "D"][:len(options)]
        wrong_opts = [o for o in all_opts if o != correct_letter]

        if not wrong_opts:
            return MisconceptionAnalysisResult(
                question_text=text,
                subject=subject,
                distractors={},
                unmapped_distractors=[],
                coverage_score=0.0,
                analysis_confidence="low",
            )

        # Use LLM to map distractors to misconceptions
        if self.client and taxonomy:
            distractors = await self._analyze_with_llm(
                text, options, correct_letter, wrong_opts, subject, taxonomy
            )
        else:
            # Fallback: no LLM or no taxonomy
            distractors = {}

        # Calculate coverage and confidence
        unmapped = [o for o in wrong_opts if o not in distractors]
        coverage = len(distractors) / len(wrong_opts) if wrong_opts else 0.0

        if coverage >= 0.8:
            confidence = "high"
        elif coverage >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return MisconceptionAnalysisResult(
            question_text=text,
            subject=subject,
            distractors=distractors,
            unmapped_distractors=unmapped,
            coverage_score=coverage,
            analysis_confidence=confidence,
        )

    async def _analyze_with_llm(
        self,
        question_text: str,
        options: List[str],
        correct_letter: str,
        wrong_opts: List[str],
        subject: str,
        taxonomy: Dict[str, Dict],
    ) -> Dict[str, DistractorMisconception]:
        """Use LLM to map distractors to misconceptions from the taxonomy."""

        # Build options text
        options_text = "\n".join(
            f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
        )

        # Build misconception list for the prompt
        misconception_list = []
        for mc_id, mc_data in taxonomy.items():
            misconception_list.append(
                f"- {mc_id}: {mc_data.get('description', '')}"
            )
        misconceptions_text = "\n".join(misconception_list[:30])  # Limit to 30

        prompt = f"""Analyze this {subject} question and identify which specific misconception each WRONG answer option targets.

QUESTION:
{question_text}

OPTIONS:
{options_text}

CORRECT ANSWER: {correct_letter}

KNOWN MISCONCEPTIONS IN {subject.upper()}:
{misconceptions_text}

For each wrong answer option ({", ".join(wrong_opts)}), identify which misconception from the list above it catches.
If a wrong answer doesn't match any known misconception, say "unknown".

Respond with JSON only:
{{
  "{wrong_opts[0]}": {{
    "misconception_id": "confuses_branches_of_government",
    "confidence": 0.8
  }},
  ...
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an educational content analyst specializing in student misconceptions. "
                            "Your task is to identify which specific, known misconception each wrong answer "
                            "option is designed to catch. Be precise and use the exact misconception IDs provided. "
                            "Respond ONLY with valid JSON."
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

            llm_analysis = json.loads(result_text)

            # Convert LLM output to DistractorMisconception objects
            distractors = {}
            for opt, data in llm_analysis.items():
                if opt not in wrong_opts:
                    continue

                mc_id = data.get("misconception_id", "unknown")
                confidence = data.get("confidence", 0.5)

                if mc_id == "unknown" or mc_id not in taxonomy:
                    continue

                mc_data = taxonomy[mc_id]
                distractors[opt] = DistractorMisconception(
                    option=opt,
                    catches_misconception=mc_id,
                    misconception_category=mc_data.get("category", "unknown"),
                    description=mc_data.get("description", ""),
                    vulnerable_archetypes=mc_data.get("vulnerable_archetypes", []),
                    remediation_standard=mc_data.get("remediation_standards", [None])[0]
                    if mc_data.get("remediation_standards")
                    else None,
                    remediation_topic=mc_data.get("remediation_topic"),
                    confidence=confidence,
                )

            return distractors

        except Exception as e:
            # Fallback on any error
            return {}

    def get_supported_subjects(self) -> List[str]:
        """Get list of subjects with misconception taxonomies."""
        return list(set(SUBJECT_TAXONOMIES.keys()))


async def analyze_misconceptions(
    content: Dict[str, Any],
    api_key: Optional[str] = None,
) -> MisconceptionAnalysisResult:
    """
    Convenience function to analyze misconceptions in educational content.

    Args:
        content: Educational content dict
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        MisconceptionAnalysisResult with distractor mappings
    """
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY")
    analyzer = MisconceptionAnalyzer(api_key=key)
    return await analyzer.analyze(content)
