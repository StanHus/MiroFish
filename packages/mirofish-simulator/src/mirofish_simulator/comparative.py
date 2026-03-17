"""
Comparative analysis for question sets.

Analyzes relationships between questions in a quiz or assessment,
providing insights that individual question analysis cannot.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .simulator import Simulator, SimulationResult
from .misconceptions import MisconceptionAnalyzer, MisconceptionAnalysisResult
from .taxonomies import get_taxonomy


@dataclass
class StandardCoverage:
    """Coverage of curriculum standards."""

    standards_covered: List[str]
    standards_missing: List[str]  # Expected standards not tested
    misconceptions_tested: List[str]


@dataclass
class RedundancyPair:
    """A pair of questions with redundant content."""

    questions: Tuple[str, str]
    overlap_type: str  # "same_misconception", "same_skill", "similar_difficulty"
    description: str


@dataclass
class ProgressionIssue:
    """Issue with difficulty progression."""

    question_id: str
    issue: str
    recommendation: str


@dataclass
class ArchetypeEquityIssue:
    """Equity issue for a specific archetype."""

    archetype: str
    disadvantaged_on: List[str]  # Question IDs
    reason: str
    severity: str  # "high", "medium", "low"


@dataclass
class ComparativeAnalysisResult:
    """Complete comparative analysis for a question set."""

    total_questions: int

    # Coverage analysis
    coverage: StandardCoverage

    # Redundancy detection
    redundancy: List[RedundancyPair]

    # Progression analysis
    progression: Dict[str, Any]  # difficulty_sequence, recommendations

    # Equity analysis
    archetype_equity: Dict[str, ArchetypeEquityIssue]

    # Overall metrics
    overall_estimated_challenge: float
    overall_archetype_variance: float
    confidence: str = "uncalibrated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "coverage": {
                "standards_covered": self.coverage.standards_covered,
                "standards_missing": self.coverage.standards_missing,
                "misconceptions_tested": self.coverage.misconceptions_tested,
            },
            "redundancy": [
                {
                    "questions": list(r.questions),
                    "overlap_type": r.overlap_type,
                    "description": r.description,
                }
                for r in self.redundancy
            ],
            "progression": self.progression,
            "archetype_equity": {
                k: {
                    "archetype": v.archetype,
                    "disadvantaged_on": v.disadvantaged_on,
                    "reason": v.reason,
                    "severity": v.severity,
                }
                for k, v in self.archetype_equity.items()
            },
            "overall_estimated_challenge": round(self.overall_estimated_challenge, 3),
            "overall_archetype_variance": round(self.overall_archetype_variance, 3),
            "confidence": self.confidence,
        }


class ComparativeAnalyzer:
    """
    Analyzes question sets to identify patterns that single-question analysis misses.

    Provides:
    - Coverage: Which standards/misconceptions are tested
    - Redundancy: Questions that test the same thing
    - Progression: Is difficulty appropriately sequenced
    - Equity: Are certain archetypes consistently disadvantaged

    Usage:
        analyzer = ComparativeAnalyzer(api_key="sk-...")
        result = await analyzer.analyze_quiz(questions=[q1, q2, q3, q4, q5])

        print(f"Standards covered: {result.coverage.standards_covered}")
        print(f"Redundant pairs: {len(result.redundancy)}")
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

        self.simulator = Simulator(api_key=api_key, base_url=base_url, model=model)
        self.mc_analyzer = MisconceptionAnalyzer(
            api_key=api_key, base_url=base_url, model=model
        )

    async def analyze_quiz(
        self,
        questions: List[Dict[str, Any]],
        expected_standards: Optional[List[str]] = None,
    ) -> ComparativeAnalysisResult:
        """
        Analyze a set of questions for comparative insights.

        Args:
            questions: List of question dicts, each with:
                - id: Unique identifier (optional, will be generated if missing)
                - text/question: Question text
                - options: Answer options
                - correct_answer: Correct answer
                - subject: Subject area
            expected_standards: Standards that should be covered (for gap analysis)

        Returns:
            ComparativeAnalysisResult with coverage, redundancy, progression, and equity
        """
        if not questions:
            return ComparativeAnalysisResult(
                total_questions=0,
                coverage=StandardCoverage([], [], []),
                redundancy=[],
                progression={},
                archetype_equity={},
                overall_estimated_challenge=0.5,
                overall_archetype_variance=0.0,
            )

        # Assign IDs if missing
        for i, q in enumerate(questions):
            if "id" not in q:
                q["id"] = f"q{i + 1}"

        # Run simulations and misconception analysis in parallel
        sim_results = await self._simulate_all(questions)
        mc_results = await self._analyze_misconceptions_all(questions)

        # Coverage analysis
        coverage = self._analyze_coverage(mc_results, expected_standards)

        # Redundancy detection
        redundancy = self._detect_redundancy(questions, mc_results, sim_results)

        # Progression analysis
        progression = self._analyze_progression(questions, sim_results)

        # Equity analysis
        equity = self._analyze_equity(questions, sim_results)

        # Overall metrics
        if sim_results:
            challenges = [r.estimated_challenge for r in sim_results.values()]
            variances = [r.archetype_variance for r in sim_results.values()]
            overall_challenge = sum(challenges) / len(challenges)
            overall_variance = sum(variances) / len(variances)
        else:
            overall_challenge = 0.5
            overall_variance = 0.0

        return ComparativeAnalysisResult(
            total_questions=len(questions),
            coverage=coverage,
            redundancy=redundancy,
            progression=progression,
            archetype_equity=equity,
            overall_estimated_challenge=overall_challenge,
            overall_archetype_variance=overall_variance,
            confidence="uncalibrated",
        )

    async def _simulate_all(
        self,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, SimulationResult]:
        """Run simulations for all questions."""
        results = {}

        # Run in parallel batches
        batch_size = 3
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.simulator.simulate(q, {"size": 20})
                for q in batch
            ])
            for q, r in zip(batch, batch_results):
                results[q["id"]] = r

        return results

    async def _analyze_misconceptions_all(
        self,
        questions: List[Dict[str, Any]],
    ) -> Dict[str, MisconceptionAnalysisResult]:
        """Run misconception analysis for all questions."""
        results = {}

        for q in questions:
            try:
                result = await self.mc_analyzer.analyze(q)
                results[q["id"]] = result
            except Exception:
                pass

        return results

    def _analyze_coverage(
        self,
        mc_results: Dict[str, MisconceptionAnalysisResult],
        expected_standards: Optional[List[str]],
    ) -> StandardCoverage:
        """Analyze which standards and misconceptions are covered."""
        standards_found = set()
        misconceptions_found = set()

        for q_id, mc_result in mc_results.items():
            for opt, distractor in mc_result.distractors.items():
                misconceptions_found.add(distractor.catches_misconception)
                if distractor.remediation_standard:
                    standards_found.add(distractor.remediation_standard)

        standards_missing = []
        if expected_standards:
            standards_missing = [s for s in expected_standards if s not in standards_found]

        return StandardCoverage(
            standards_covered=sorted(standards_found),
            standards_missing=standards_missing,
            misconceptions_tested=sorted(misconceptions_found),
        )

    def _detect_redundancy(
        self,
        questions: List[Dict[str, Any]],
        mc_results: Dict[str, MisconceptionAnalysisResult],
        sim_results: Dict[str, SimulationResult],
    ) -> List[RedundancyPair]:
        """Detect redundant question pairs."""
        redundancy = []
        q_ids = [q["id"] for q in questions]

        # Check for same misconception
        misconception_to_questions: Dict[str, List[str]] = {}
        for q_id, mc_result in mc_results.items():
            for opt, distractor in mc_result.distractors.items():
                mc_id = distractor.catches_misconception
                if mc_id not in misconception_to_questions:
                    misconception_to_questions[mc_id] = []
                if q_id not in misconception_to_questions[mc_id]:
                    misconception_to_questions[mc_id].append(q_id)

        for mc_id, q_list in misconception_to_questions.items():
            if len(q_list) >= 2:
                # Only report first pair to avoid noise
                redundancy.append(
                    RedundancyPair(
                        questions=(q_list[0], q_list[1]),
                        overlap_type="same_misconception",
                        description=f"Both test the '{mc_id}' misconception",
                    )
                )

        # Check for similar difficulty (within 0.1)
        for i, q1_id in enumerate(q_ids):
            for q2_id in q_ids[i + 1:]:
                if q1_id in sim_results and q2_id in sim_results:
                    d1 = sim_results[q1_id].estimated_challenge
                    d2 = sim_results[q2_id].estimated_challenge
                    if abs(d1 - d2) < 0.1 and d1 > 0.3 and d1 < 0.7:
                        # Similar moderate difficulty
                        redundancy.append(
                            RedundancyPair(
                                questions=(q1_id, q2_id),
                                overlap_type="similar_difficulty",
                                description=f"Both have similar challenge (~{d1:.2f})",
                            )
                        )
                        break  # Only one per question

        return redundancy

    def _analyze_progression(
        self,
        questions: List[Dict[str, Any]],
        sim_results: Dict[str, SimulationResult],
    ) -> Dict[str, Any]:
        """Analyze difficulty progression through the quiz."""
        q_ids = [q["id"] for q in questions]
        difficulties = []
        issues = []

        for q_id in q_ids:
            if q_id in sim_results:
                difficulties.append(sim_results[q_id].estimated_challenge)
            else:
                difficulties.append(0.5)  # Default

        # Check for non-monotonic progression that might hurt scaffolding
        recommendations = []
        if len(difficulties) >= 3:
            # Check if there's a spike early
            if difficulties[0] > 0.7:
                recommendations.append(
                    f"{q_ids[0]} has high difficulty ({difficulties[0]:.2f}) as first question. "
                    "Consider starting with an easier warm-up."
                )

            # Check for difficulty valleys after hard questions
            for i in range(1, len(difficulties) - 1):
                if difficulties[i] < difficulties[i - 1] - 0.2:
                    recommendations.append(
                        f"{q_ids[i]} is easier than {q_ids[i-1]}. "
                        "Consider reordering for smoother progression."
                    )

            # Check if last question is too easy
            if difficulties[-1] < 0.3 and max(difficulties[:-1]) > 0.5:
                recommendations.append(
                    f"Final question {q_ids[-1]} is relatively easy. "
                    "Consider ending with a more challenging capstone."
                )

        return {
            "difficulty_sequence": [round(d, 2) for d in difficulties],
            "recommendations": recommendations,
        }

    def _analyze_equity(
        self,
        questions: List[Dict[str, Any]],
        sim_results: Dict[str, SimulationResult],
    ) -> Dict[str, ArchetypeEquityIssue]:
        """Analyze equity issues across archetypes."""
        equity_issues = {}

        # Collect per-archetype performance across questions
        archetype_scores: Dict[str, Dict[str, float]] = {}  # archetype -> {q_id -> accuracy}

        for q_id, result in sim_results.items():
            for arch, perf in result.by_archetype.items():
                if arch not in archetype_scores:
                    archetype_scores[arch] = {}
                archetype_scores[arch][q_id] = perf.accuracy

        # Check for archetypes that are consistently disadvantaged
        for arch, scores in archetype_scores.items():
            if not scores:
                continue

            avg_accuracy = sum(scores.values()) / len(scores)
            low_questions = [q_id for q_id, acc in scores.items() if acc < 0.3]

            if len(low_questions) >= 2 or (avg_accuracy < 0.4 and len(scores) >= 3):
                severity = "high" if avg_accuracy < 0.3 else "medium"

                # Determine likely reason
                if arch == "esl_student":
                    reason = "Vocabulary complexity or cultural references"
                elif arch == "class_clown":
                    reason = "Questions may require sustained focus"
                elif arch == "disengaged_but_smart":
                    reason = "Questions may not capture interest"
                elif arch == "politically_conservative":
                    reason = "Content framing may not resonate"
                else:
                    reason = "Multiple questions pose difficulty"

                equity_issues[arch] = ArchetypeEquityIssue(
                    archetype=arch,
                    disadvantaged_on=low_questions,
                    reason=reason,
                    severity=severity,
                )

        return equity_issues


async def analyze_quiz(
    questions: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    expected_standards: Optional[List[str]] = None,
) -> ComparativeAnalysisResult:
    """
    Convenience function to analyze a quiz.

    Args:
        questions: List of question dicts
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        expected_standards: Standards that should be covered

    Returns:
        ComparativeAnalysisResult with quiz-level insights
    """
    analyzer = ComparativeAnalyzer(api_key=api_key)
    return await analyzer.analyze_quiz(questions, expected_standards)
