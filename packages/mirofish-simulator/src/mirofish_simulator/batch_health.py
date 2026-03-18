"""
Batch Health Analyzer - Swarm intelligence at the batch level.

Instead of running expensive evaluators on every question:
- Run FREE heuristics to detect patterns
- ONE LLM call to analyze the batch holistically
- Output: which questions need attention + generator feedback

Cost model:
  Current:  N questions × 4-6 LLM calls = 4N-6N calls
  Proposed: Heuristics (free) + 1 batch call + selective deep dives

Usage:
    from mirofish_simulator import BatchHealthAnalyzer

    analyzer = BatchHealthAnalyzer()

    # Analyze a batch of questions
    report = await analyzer.analyze(
        questions=[q1, q2, q3, ...],
        curriculum_context={
            "standards": ["CCSS.MATH.3.OA.A.1", ...],
            "learning_objectives": ["Understand multiplication", ...],
        }
    )

    print(report.questions_needing_attention)  # ["q3", "q7"]
    print(report.generator_feedback)           # Actionable feedback
    print(report.coverage_gaps)                # Missing standards
"""

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


@dataclass
class PatternFindings:
    """Heuristic pattern detection results (FREE - no LLM)."""

    # Test-wiseness exploits
    longest_answer_correct_rate: float = 0.0
    longest_answer_questions: List[str] = field(default_factory=list)

    all_of_above_count: int = 0
    none_of_above_count: int = 0

    grammar_cue_questions: List[str] = field(default_factory=list)

    # Absolute terms in distractors (often indicates wrong answer)
    absolute_term_distractors: List[str] = field(default_factory=list)

    # Position bias
    correct_answer_distribution: Dict[str, int] = field(default_factory=dict)  # A: 5, B: 3, etc.
    position_bias_detected: bool = False

    # Option analysis
    option_length_variance: float = 0.0  # High = good, low = suspicious

    # Difficulty distribution
    vocabulary_density: Dict[str, float] = field(default_factory=dict)  # per question
    estimated_difficulty_spread: Tuple[float, float] = (0.0, 0.0)  # min, max

    # Redundancy signals
    similar_stems: List[Tuple[str, str]] = field(default_factory=list)
    repeated_distractors: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def risk_score(self) -> float:
        """Overall risk score 0-1. Higher = more issues detected."""
        score = 0.0

        # Longest answer bias (should be ~25% for 4-option MCQ)
        if self.longest_answer_correct_rate > 0.4:
            score += 0.3
        elif self.longest_answer_correct_rate > 0.3:
            score += 0.15

        # Grammar cues
        if self.grammar_cue_questions:
            score += min(0.2, len(self.grammar_cue_questions) * 0.05)

        # Low option variance (all options same length = suspicious)
        if self.option_length_variance < 0.1:
            score += 0.1

        # Similar stems
        if self.similar_stems:
            score += min(0.2, len(self.similar_stems) * 0.1)

        # Position bias
        if self.position_bias_detected:
            score += 0.15

        return min(1.0, score)

    @property
    def warrants_deep_analysis(self) -> bool:
        """Should we run the LLM batch analysis?"""
        return self.risk_score > 0.2 or len(self.grammar_cue_questions) > 0


@dataclass
class BatchHealthReport:
    """Complete batch health analysis."""

    total_questions: int

    # Heuristic findings (FREE)
    patterns: PatternFindings

    # Questions flagged for attention
    questions_needing_attention: List[str] = field(default_factory=list)
    attention_reasons: Dict[str, List[str]] = field(default_factory=dict)

    # Coverage analysis
    coverage_gaps: List[str] = field(default_factory=list)
    concepts_tested: List[str] = field(default_factory=list)
    redundant_concepts: List[str] = field(default_factory=list)

    # Generator feedback (actionable)
    generator_feedback: List[str] = field(default_factory=list)

    # Evaluator hints
    evaluator_hints: Dict[str, List[str]] = field(default_factory=dict)

    # Cost estimation
    estimated_full_pipeline_calls: int = 0
    estimated_optimized_calls: int = 0

    # Confidence
    confidence: str = "heuristics_only"  # or "heuristics_plus_llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "risk_score": self.patterns.risk_score,
            "questions_needing_attention": self.questions_needing_attention,
            "attention_reasons": self.attention_reasons,
            "coverage_gaps": self.coverage_gaps,
            "concepts_tested": self.concepts_tested,
            "redundant_concepts": self.redundant_concepts,
            "generator_feedback": self.generator_feedback,
            "evaluator_hints": self.evaluator_hints,
            "patterns": {
                "longest_answer_correct_rate": self.patterns.longest_answer_correct_rate,
                "longest_answer_questions": self.patterns.longest_answer_questions,
                "grammar_cue_questions": self.patterns.grammar_cue_questions,
                "similar_stems": self.patterns.similar_stems,
                "option_length_variance": self.patterns.option_length_variance,
                "position_bias": self.patterns.position_bias_detected,
                "correct_answer_distribution": self.patterns.correct_answer_distribution,
            },
            "cost_estimate": {
                "full_pipeline_calls": self.estimated_full_pipeline_calls,
                "optimized_calls": self.estimated_optimized_calls,
                "savings_percent": round(
                    (1 - self.estimated_optimized_calls / max(1, self.estimated_full_pipeline_calls)) * 100
                ),
            },
            "confidence": self.confidence,
        }

    def get_routing_hints(self) -> Dict[str, List[str]]:
        """
        Get hints for InceptBench evaluator routing.

        Returns dict of question_id -> list of recommended evaluators.
        Questions not in the dict can skip expensive evaluators.
        """
        hints = {}

        # Questions with grammar cues need linguistic analysis
        for q_id in self.patterns.grammar_cue_questions:
            if q_id not in hints:
                hints[q_id] = []
            hints[q_id].append("reading_question_qc")

        # Questions with longest answer need distractor analysis
        for q_id in self.patterns.longest_answer_questions:
            if q_id not in hints:
                hints[q_id] = []
            hints[q_id].append("ti_question_qa")

        # Merge any LLM-generated hints
        for q_id, llm_hints in self.evaluator_hints.items():
            if q_id not in hints:
                hints[q_id] = []
            hints[q_id].extend(llm_hints)

        return hints

    def summary(self) -> str:
        """Human-readable summary of the batch health."""
        lines = [
            f"Batch Health: {self.total_questions} questions, risk score {self.patterns.risk_score:.2f}",
            "",
        ]

        if self.questions_needing_attention:
            lines.append(f"⚠️  {len(self.questions_needing_attention)} questions need attention: {', '.join(self.questions_needing_attention)}")

        if self.generator_feedback:
            lines.append("")
            lines.append("Generator feedback:")
            for fb in self.generator_feedback:
                lines.append(f"  • {fb}")

        if self.coverage_gaps:
            lines.append("")
            lines.append(f"Coverage gaps: {', '.join(self.coverage_gaps)}")

        savings = 100 - (self.estimated_optimized_calls / max(1, self.estimated_full_pipeline_calls) * 100)
        lines.append("")
        lines.append(f"Cost: {self.estimated_full_pipeline_calls} → {self.estimated_optimized_calls} calls ({savings:.0f}% savings)")

        return "\n".join(lines)


class BatchHealthAnalyzer:
    """
    Batch-level swarm intelligence for question quality.

    Philosophy: One valuable insight per batch > N mindless LLM calls per question

    Runs in two phases:
    1. FREE heuristics - detect patterns without LLM
    2. ONE LLM call - holistic batch analysis (optional, based on heuristics)
    """

    # Difficult words that increase cognitive load
    DIFFICULT_WORDS = {
        "approximately", "consequently", "furthermore", "nevertheless",
        "subsequently", "predominantly", "notwithstanding", "aforementioned",
        "heretofore", "whereas", "whereby", "thereof", "therein",
    }

    # Absolute terms that often indicate wrong answers
    ABSOLUTE_TERMS = {
        "always", "never", "all", "none", "only", "must", "cannot",
        "every", "no one", "everyone", "completely", "entirely",
        "absolutely", "totally", "definitely", "certainly",
    }

    # Grammar cue patterns (singular/plural mismatches)
    GRAMMAR_CUE_PATTERNS = [
        (r"\bis an?\b", r"^[aeiou]"),  # "is a" followed by vowel
        (r"\bare\b.*\?$", r"^[A-Z][a-z]+s\b"),  # "are" question, plural answer
        (r"\bis\b.*\?$", r"^[A-Z][a-z]+[^s]\b"),  # "is" question, singular answer
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model

        if AsyncOpenAI and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = None

    async def analyze(
        self,
        questions: List[Dict[str, Any]],
        curriculum_context: Optional[Dict[str, Any]] = None,
        run_llm_analysis: Optional[bool] = None,  # None = auto-decide
    ) -> BatchHealthReport:
        """
        Analyze a batch of questions for health issues.

        Args:
            questions: List of question dicts with:
                - id: Unique identifier (optional)
                - text/question: Question text
                - options: Answer options (list or dict)
                - correct_answer: Correct answer key
            curriculum_context: Optional context from parent software:
                - standards: List of expected standards
                - learning_objectives: What students should learn
                - grade: Target grade level
                - subject: Subject area
            run_llm_analysis: Force LLM analysis on/off. None = auto-decide based on heuristics.

        Returns:
            BatchHealthReport with findings and actionable feedback
        """
        if not questions:
            return BatchHealthReport(
                total_questions=0,
                patterns=PatternFindings(),
            )

        # Normalize questions
        normalized = self._normalize_questions(questions)

        # Phase 1: FREE heuristics
        patterns = self._run_heuristics(normalized)

        # Build initial report from heuristics
        report = self._build_heuristic_report(normalized, patterns, curriculum_context)

        # Phase 2: LLM analysis (if warranted)
        should_run_llm = run_llm_analysis if run_llm_analysis is not None else patterns.warrants_deep_analysis

        if should_run_llm and self.client:
            await self._run_llm_analysis(normalized, curriculum_context, report)
            report.confidence = "heuristics_plus_llm"

        return report

    def _normalize_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize question format for consistent processing."""
        normalized = []

        for i, q in enumerate(questions):
            q_id = q.get("id", f"q{i + 1}")
            text = q.get("text") or q.get("question", "")

            # Normalize options to dict format
            options = q.get("options", {})
            if isinstance(options, list):
                keys = ["A", "B", "C", "D", "E", "F"][:len(options)]
                options = dict(zip(keys, options))

            correct = q.get("correct_answer", "")

            normalized.append({
                "id": q_id,
                "text": text,
                "options": options,
                "correct_answer": correct,
                "subject": q.get("subject"),
                "skill": q.get("skill"),
                "original": q,
            })

        return normalized

    def _run_heuristics(self, questions: List[Dict[str, Any]]) -> PatternFindings:
        """Run FREE heuristic analysis on the batch."""
        findings = PatternFindings()

        if not questions:
            return findings

        # Track for analysis
        option_lengths_all = []
        stems = []
        distractor_texts = {}
        correct_positions = []

        longest_correct_count = 0

        for q in questions:
            q_id = q["id"]
            text = q["text"]
            options = q["options"]
            correct = q["correct_answer"]

            stems.append((q_id, text))

            # === Longest answer analysis ===
            if options and correct:
                lengths = {k: len(str(v)) for k, v in options.items()}
                longest_key = max(lengths, key=lengths.get) if lengths else None

                if longest_key == correct:
                    longest_correct_count += 1
                    findings.longest_answer_questions.append(q_id)

                option_lengths_all.extend(lengths.values())

                # Track correct answer position
                correct_positions.append(correct)

            # === Absolute terms in correct answer (test-wiseness: absolutes often wrong) ===
            if correct and correct in options:
                correct_text = str(options[correct]).lower()
                for term in self.ABSOLUTE_TERMS:
                    if term in correct_text.split():
                        findings.absolute_term_distractors.append(q_id)
                        break

            # === All/None of above ===
            for opt_text in options.values():
                opt_lower = str(opt_text).lower()
                if "all of the above" in opt_lower or "all the above" in opt_lower:
                    findings.all_of_above_count += 1
                if "none of the above" in opt_lower or "none of these" in opt_lower:
                    findings.none_of_above_count += 1

            # === Grammar cues ===
            grammar_issues = self._check_grammar_cues(text, options, correct)
            if grammar_issues:
                findings.grammar_cue_questions.append(q_id)

            # === Vocabulary density ===
            word_count = len(text.split())
            difficult_count = sum(1 for w in text.lower().split() if w in self.DIFFICULT_WORDS)
            findings.vocabulary_density[q_id] = difficult_count / max(1, word_count)

            # === Track distractors for redundancy ===
            for k, v in options.items():
                if k != correct:
                    v_normalized = str(v).lower().strip()
                    if v_normalized not in distractor_texts:
                        distractor_texts[v_normalized] = []
                    distractor_texts[v_normalized].append(q_id)

        # === Aggregate metrics ===

        # Longest answer correct rate
        if questions:
            findings.longest_answer_correct_rate = longest_correct_count / len(questions)

        # Option length variance
        if option_lengths_all:
            try:
                findings.option_length_variance = statistics.stdev(option_lengths_all) / max(1, statistics.mean(option_lengths_all))
            except statistics.StatisticsError:
                findings.option_length_variance = 0.0

        # Similar stems
        findings.similar_stems = self._find_similar_stems(stems)

        # Repeated distractors
        findings.repeated_distractors = {
            text: q_ids for text, q_ids in distractor_texts.items()
            if len(q_ids) > 1
        }

        # Difficulty spread estimate
        if findings.vocabulary_density:
            densities = list(findings.vocabulary_density.values())
            findings.estimated_difficulty_spread = (min(densities), max(densities))

        # Position bias detection
        if correct_positions:
            from collections import Counter
            pos_counts = Counter(correct_positions)
            findings.correct_answer_distribution = dict(pos_counts)

            # Check for bias: if any position has >40% of answers (expect ~25%)
            total = len(correct_positions)
            if total >= 4:  # Need enough questions to detect bias
                max_pct = max(pos_counts.values()) / total
                min_pct = min(pos_counts.values()) / total if len(pos_counts) > 1 else max_pct
                findings.position_bias_detected = max_pct > 0.4 or min_pct < 0.1

        return findings

    def _check_grammar_cues(
        self,
        text: str,
        options: Dict[str, str],
        correct: str,
    ) -> List[str]:
        """Check for grammar cues that reveal the answer."""
        issues = []

        # Check if "a" vs "an" matches only the correct answer
        if " a " in text.lower() or text.lower().startswith("a "):
            correct_text = str(options.get(correct, "")).lower()
            if correct_text and correct_text[0] not in "aeiou":
                # "a" matches consonant-starting correct answer
                vowel_distractors = sum(
                    1 for k, v in options.items()
                    if k != correct and str(v).lower()[:1] in "aeiou"
                )
                if vowel_distractors > 0:
                    issues.append("article_cue")

        if " an " in text.lower():
            correct_text = str(options.get(correct, "")).lower()
            if correct_text and correct_text[0] in "aeiou":
                consonant_distractors = sum(
                    1 for k, v in options.items()
                    if k != correct and str(v).lower()[:1] not in "aeiou"
                )
                if consonant_distractors > 0:
                    issues.append("article_cue")

        return issues

    def _find_similar_stems(
        self,
        stems: List[Tuple[str, str]],
        threshold: float = 0.7,
    ) -> List[Tuple[str, str]]:
        """Find question pairs with similar stems."""
        similar = []

        # Simple word overlap similarity
        for i, (id1, text1) in enumerate(stems):
            words1 = set(text1.lower().split())
            for id2, text2 in stems[i + 1:]:
                words2 = set(text2.lower().split())

                if not words1 or not words2:
                    continue

                overlap = len(words1 & words2)
                similarity = overlap / min(len(words1), len(words2))

                if similarity >= threshold:
                    similar.append((id1, id2))

        return similar

    def _build_heuristic_report(
        self,
        questions: List[Dict[str, Any]],
        patterns: PatternFindings,
        curriculum_context: Optional[Dict[str, Any]],
    ) -> BatchHealthReport:
        """Build report from heuristic findings."""
        report = BatchHealthReport(
            total_questions=len(questions),
            patterns=patterns,
            confidence="heuristics_only",
        )

        # Flag questions needing attention
        attention = {}

        for q_id in patterns.longest_answer_questions:
            if q_id not in attention:
                attention[q_id] = []
            attention[q_id].append("longest_answer_correct")

        for q_id in patterns.grammar_cue_questions:
            if q_id not in attention:
                attention[q_id] = []
            attention[q_id].append("grammar_cue_detected")

        for q_ids in patterns.similar_stems:
            for q_id in q_ids:
                if q_id not in attention:
                    attention[q_id] = []
                attention[q_id].append("similar_to_another_question")

        report.questions_needing_attention = list(attention.keys())
        report.attention_reasons = attention

        # Generator feedback from heuristics
        feedback = []

        if patterns.longest_answer_correct_rate > 0.3:
            pct = int(patterns.longest_answer_correct_rate * 100)
            feedback.append(
                f"{pct}% of questions have longest answer correct (expect ~25%). "
                "Vary correct answer lengths."
            )

        if patterns.grammar_cue_questions:
            feedback.append(
                f"{len(patterns.grammar_cue_questions)} questions have grammar cues "
                "that may reveal the answer. Check article agreement (a/an)."
            )

        if patterns.all_of_above_count > len(questions) * 0.2:
            feedback.append(
                f"'All of the above' used {patterns.all_of_above_count} times. "
                "Consider reducing - it's a test-wiseness exploit."
            )

        if patterns.similar_stems:
            feedback.append(
                f"{len(patterns.similar_stems)} question pairs have very similar stems. "
                "May indicate redundancy or copy-paste errors."
            )

        if patterns.option_length_variance < 0.1:
            feedback.append(
                "Option lengths are very uniform across the batch. "
                "Natural variation in option length is expected."
            )

        if patterns.position_bias_detected:
            dist = patterns.correct_answer_distribution
            feedback.append(
                f"Correct answer position bias detected: {dist}. "
                "Distribute correct answers more evenly across positions."
            )

        if patterns.absolute_term_distractors:
            feedback.append(
                f"{len(patterns.absolute_term_distractors)} correct answers contain absolute terms "
                "(always, never, all, none). Test-wise students avoid these."
            )

        report.generator_feedback = feedback

        # Cost estimation
        # Assume: 5 LLM calls per question in full pipeline
        report.estimated_full_pipeline_calls = len(questions) * 5
        # Optimized: 1 batch call + 5 calls only for flagged questions
        flagged_count = len(report.questions_needing_attention)
        report.estimated_optimized_calls = 1 + (flagged_count * 5)

        return report

    async def _run_llm_analysis(
        self,
        questions: List[Dict[str, Any]],
        curriculum_context: Optional[Dict[str, Any]],
        report: BatchHealthReport,
    ) -> None:
        """Run ONE LLM call to analyze the batch holistically."""
        if not self.client:
            return

        # Build batch summary for LLM
        batch_summary = self._build_batch_summary(questions, curriculum_context)

        system_prompt = """You are a batch quality analyzer for educational assessments.

Analyze this batch of questions and provide:
1. COVERAGE_GAPS: What concepts/standards should be tested but aren't?
2. REDUNDANCY: Which questions test the same thing?
3. GENERATOR_FEEDBACK: Specific, actionable feedback for the question generator.
4. EVALUATOR_HINTS: Which specific questions need which type of deeper analysis?

Be concise. Focus on batch-level patterns, not individual question quality.

Respond in JSON:
{
    "coverage_gaps": ["gap1", "gap2"],
    "concepts_tested": ["concept1", "concept2"],
    "redundant_concepts": ["concept tested multiple times"],
    "additional_feedback": ["feedback1", "feedback2"],
    "evaluator_hints": {
        "question_id": ["hint1", "hint2"]
    }
}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": batch_summary},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Merge LLM findings into report
            report.coverage_gaps = result.get("coverage_gaps", [])
            report.concepts_tested = result.get("concepts_tested", [])
            report.redundant_concepts = result.get("redundant_concepts", [])
            report.generator_feedback.extend(result.get("additional_feedback", []))

            # Merge evaluator hints
            for q_id, hints in result.get("evaluator_hints", {}).items():
                if q_id not in report.evaluator_hints:
                    report.evaluator_hints[q_id] = []
                report.evaluator_hints[q_id].extend(hints)

                # Also add to questions needing attention
                if q_id not in report.questions_needing_attention:
                    report.questions_needing_attention.append(q_id)
                    report.attention_reasons[q_id] = hints

        except Exception as e:
            # LLM analysis failed, but heuristics still valid
            report.generator_feedback.append(f"(LLM analysis skipped: {e})")

    def _build_batch_summary(
        self,
        questions: List[Dict[str, Any]],
        curriculum_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a summary of the batch for LLM analysis."""
        lines = [f"BATCH OF {len(questions)} QUESTIONS\n"]

        # Curriculum context
        if curriculum_context:
            if curriculum_context.get("standards"):
                lines.append(f"Expected standards: {', '.join(curriculum_context['standards'][:10])}")
            if curriculum_context.get("learning_objectives"):
                lines.append(f"Learning objectives: {', '.join(curriculum_context['learning_objectives'][:5])}")
            if curriculum_context.get("grade"):
                lines.append(f"Target grade: {curriculum_context['grade']}")
            if curriculum_context.get("subject"):
                lines.append(f"Subject: {curriculum_context['subject']}")
            lines.append("")

        # Question summaries (abbreviated)
        lines.append("QUESTIONS:")
        for q in questions[:20]:  # Cap at 20 to stay within context
            q_id = q["id"]
            text = q["text"][:150] + "..." if len(q["text"]) > 150 else q["text"]
            options_summary = ", ".join(
                f"{k}: {str(v)[:30]}..." if len(str(v)) > 30 else f"{k}: {v}"
                for k, v in list(q["options"].items())[:4]
            )
            correct = q["correct_answer"]

            lines.append(f"\n[{q_id}] {text}")
            lines.append(f"  Options: {options_summary}")
            lines.append(f"  Correct: {correct}")

        if len(questions) > 20:
            lines.append(f"\n... and {len(questions) - 20} more questions")

        return "\n".join(lines)


async def analyze_batch(
    questions: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    curriculum_context: Optional[Dict[str, Any]] = None,
) -> BatchHealthReport:
    """
    Convenience function to analyze a batch.

    Args:
        questions: List of question dicts
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        curriculum_context: Optional curriculum info from parent software

    Returns:
        BatchHealthReport with findings and generator feedback
    """
    analyzer = BatchHealthAnalyzer(api_key=api_key)
    return await analyzer.analyze(questions, curriculum_context)
