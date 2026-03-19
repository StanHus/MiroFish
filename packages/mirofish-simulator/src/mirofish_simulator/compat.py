"""
Compatibility layer for InceptBench integration.

High-level API:
    from mirofish_simulator import analyze_question
    result = await analyze_question(question_dict)

Lower-level:
- StructuralAnalyzer (alias for MisconceptionAnalyzer)
- parse_question (question parsing helper)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .misconceptions import MisconceptionAnalyzer, MisconceptionAnalysisResult

logger = logging.getLogger(__name__)


def parse_question(content: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parse question content into standard format.

    Accepts:
    - JSON string
    - Dict with various question formats

    Returns:
    - Normalized dict with: text, options (dict), correct_answer
    - None if parsing fails
    """
    # Parse JSON string
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
    else:
        data = content

    if not isinstance(data, dict):
        return None

    # Extract question text
    text = (
        data.get("text")
        or data.get("question")
        or data.get("stem")
        or data.get("question_text")
    )
    if not text:
        return None

    # Extract options - handle multiple formats
    options = {}

    if "answer_options" in data:
        # InceptBench format: [{"key": "A", "text": "..."}, ...]
        for opt in data["answer_options"]:
            key = opt.get("key") or opt.get("option")
            txt = opt.get("text") or opt.get("value")
            if key and txt:
                options[str(key).upper()] = txt

    elif "options" in data:
        raw_opts = data["options"]
        if isinstance(raw_opts, dict):
            # Already a dict: {"A": "...", "B": "..."}
            options = {str(k).upper(): v for k, v in raw_opts.items()}
        elif isinstance(raw_opts, list):
            # List format: ["option1", "option2", ...]
            for i, opt in enumerate(raw_opts):
                key = chr(65 + i)  # A, B, C, D...
                if isinstance(opt, dict):
                    options[key] = opt.get("text") or opt.get("value") or str(opt)
                else:
                    options[key] = str(opt)

    elif "choices" in data:
        # Alternative key
        choices = data["choices"]
        if isinstance(choices, list):
            for i, opt in enumerate(choices):
                options[chr(65 + i)] = str(opt) if not isinstance(opt, dict) else opt.get("text", str(opt))

    if len(options) < 2:
        return None

    # Extract correct answer
    correct = (
        data.get("correct_answer")
        or data.get("answer")
        or data.get("correct")
        or data.get("correct_option")
    )

    if correct is None:
        return None

    # Normalize correct answer to letter
    if isinstance(correct, int):
        correct = chr(65 + correct)
    elif isinstance(correct, str):
        correct = correct.strip().upper()
        if len(correct) > 1 and correct[0].isalpha():
            correct = correct[0]

    return {
        "text": text,
        "options": options,
        "correct_answer": correct,
        "subject": data.get("subject"),
        "grade": data.get("grade"),
        "id": data.get("id") or data.get("question_id"),
    }


class StructuralAnalyzer:
    """
    Structural analysis of question quality.

    Alias for MisconceptionAnalyzer with InceptBench-compatible output format.
    Analyzes:
    - Distractor-to-misconception mapping
    - Plausibility of each option
    - Ambiguity issues
    - Redundant distractors
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self._analyzer = MisconceptionAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    async def analyze(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze question structure.

        Returns dict with:
        - distractors: {option: {catches, plausible, why_attractive}}
        - ambiguity_issues: list of issues
        - correct_answer_issues: list of issues
        - redundant_pairs: list of [opt1, opt2] pairs
        - fixes: {option: suggested_fix}
        """
        # Normalize question format
        if "text" not in question:
            parsed = parse_question(question)
            if parsed:
                question = parsed

        result = await self._analyzer.analyze(question)

        # Convert to InceptBench-expected format
        distractors = {}
        for opt, mc in result.distractors.items():
            distractors[opt] = {
                "catches": mc.catches_misconception,
                "description": mc.description,
                "plausible": mc.confidence >= 0.5,
                "why_attractive": mc.description,
                "vulnerable_archetypes": mc.vulnerable_archetypes,
                "remediation": mc.remediation_topic,
            }

        # Add unmapped distractors as implausible
        for opt in result.unmapped_distractors:
            distractors[opt] = {
                "catches": "unknown",
                "description": "Could not map to known misconception",
                "plausible": False,
                "why_attractive": "unclear",
                "vulnerable_archetypes": [],
                "remediation": None,
            }

        # Detect redundant pairs (same misconception)
        redundant_pairs = []
        mc_to_opts: Dict[str, List[str]] = {}
        for opt, info in distractors.items():
            mc = info.get("catches", "unknown")
            if mc != "unknown":
                if mc not in mc_to_opts:
                    mc_to_opts[mc] = []
                mc_to_opts[mc].append(opt)

        for mc, opts in mc_to_opts.items():
            if len(opts) >= 2:
                redundant_pairs.append(opts[:2])

        return {
            "distractors": distractors,
            "ambiguity_issues": [],  # Could be enhanced
            "correct_answer_issues": [],  # Could be enhanced
            "redundant_pairs": redundant_pairs,
            "fixes": {},  # Could be enhanced with LLM suggestions
            "coverage_score": result.coverage_score,
            "confidence": result.analysis_confidence,
        }

    def analyze_sync(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for analyze()."""
        return asyncio.get_event_loop().run_until_complete(self.analyze(question))


def add_sync_method_to_accessibility_analyzer():
    """
    Monkey-patch AccessibilityAnalyzer to add analyze_sync method.

    Called on module import.
    """
    from .accessibility import AccessibilityAnalyzer, AccessibilityResult

    def analyze_sync(
        self,
        content: Dict[str, Any],
        target_grade: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous accessibility analysis.

        Returns dict format for InceptBench compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.analyze(content, target_grade, **kwargs)
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.analyze(content, target_grade, **kwargs)
                )
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self.analyze(content, target_grade, **kwargs))

        return result.to_dict()

    # Add method to class
    AccessibilityAnalyzer.analyze_sync = analyze_sync


# Apply monkey-patch on import
try:
    add_sync_method_to_accessibility_analyzer()
except Exception:
    pass  # Ignore if accessibility module not loaded yet


@dataclass
class QuestionAnalysisResult:
    """Unified result from analyze_question()."""

    # Accessibility (FREE - deterministic)
    accessibility: Optional[Dict[str, Any]] = None

    # Structural analysis (misconception mapping)
    structural: Optional[Dict[str, Any]] = None

    # Adversarial testing (defensibility of wrong answers)
    adversarial: Optional[Dict[str, Any]] = None

    # Student simulation (archetype predictions)
    simulation: Optional[Dict[str, Any]] = None

    # Errors encountered
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accessibility": self.accessibility,
            "structural": self.structural,
            "adversarial": self.adversarial,
            "simulation": self.simulation,
            "errors": self.errors,
        }

    def get_scores(self) -> Dict[str, float]:
        """
        Extract 0-1 scores from all analyses.

        Returns dict with keys: accessibility, structural, adversarial, simulation
        Missing analyses return None values.

        Example:
            result = await analyze_question(q)
            scores = result.get_scores()
            # {"accessibility": 0.85, "structural": 0.7, "adversarial": 1.0, "simulation": 0.6}
        """
        scores: Dict[str, Optional[float]] = {}

        # Accessibility score
        if self.accessibility:
            scores["accessibility"] = self.accessibility.get("accessibility_score")
        else:
            scores["accessibility"] = None

        # Structural score (coverage)
        if self.structural:
            scores["structural"] = self.structural.get("coverage_score")
        else:
            scores["structural"] = None

        # Adversarial score (1.0 if CLEAR, 0.0 otherwise)
        if self.adversarial:
            verdict = self.adversarial.get("verdict", "UNKNOWN")
            scores["adversarial"] = 1.0 if verdict == "CLEAR" else 0.0
        else:
            scores["adversarial"] = None

        # Simulation score (accuracy)
        if self.simulation:
            scores["simulation"] = self.simulation.get("accuracy")
        else:
            scores["simulation"] = None

        return scores


async def analyze_question(
    question: Union[str, Dict[str, Any]],
    target_grade: Optional[int] = None,
    run_accessibility: bool = True,
    run_structural: bool = True,
    run_adversarial: bool = True,
    run_simulation: bool = True,
    simulation_archetypes: Optional[List[str]] = None,
) -> QuestionAnalysisResult:
    """
    High-level API: Run all analyses on a question in one call.

    Args:
        question: Question dict or JSON string
        target_grade: Target grade level (affects accessibility + simulation)
        run_accessibility: Run FREE accessibility analysis
        run_structural: Run structural/misconception analysis (+1 LLM)
        run_adversarial: Run adversarial testing (+4 LLM)
        run_simulation: Run student simulation (+3 LLM)
        simulation_archetypes: Which archetypes to simulate (default: diverse set)

    Returns:
        QuestionAnalysisResult with all analysis results

    Example:
        from mirofish_simulator import analyze_question

        result = await analyze_question(question_dict, target_grade=5)
        print(result.accessibility)  # Reading level, vocabulary
        print(result.structural)     # Misconception mapping
        print(result.adversarial)    # Defensibility of wrong answers
        print(result.simulation)     # Archetype predictions
    """
    result = QuestionAnalysisResult()

    # Parse question
    parsed = parse_question(question)
    if not parsed:
        result.errors.append("Failed to parse question")
        return result

    # Build tasks
    tasks = {}

    if run_accessibility:
        tasks["accessibility"] = _run_accessibility(parsed, target_grade)

    if run_structural:
        tasks["structural"] = _run_structural(parsed)

    if run_adversarial:
        tasks["adversarial"] = _run_adversarial(parsed)

    if run_simulation:
        tasks["simulation"] = _run_simulation(parsed, target_grade, simulation_archetypes)

    # Run in parallel
    if tasks:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                result.errors.append(f"{name}: {res}")
                logger.warning(f"Analysis {name} failed: {res}")
            else:
                setattr(result, name, res)

    return result


async def analyze_question_string(
    content: str,
    target_grade: Optional[int] = None,
    run_accessibility: bool = True,
    run_structural: bool = True,
    run_adversarial: bool = True,
    run_simulation: bool = True,
) -> QuestionAnalysisResult:
    """
    Analyze a question from raw string content.

    Convenience wrapper that parses the string and runs analysis.
    Simpler than analyze_question() when you have raw JSON strings.

    Args:
        content: JSON string or question text
        target_grade: Target grade level
        run_*: Which analyses to run

    Returns:
        QuestionAnalysisResult with scores accessible via get_scores()

    Example:
        from mirofish_simulator import analyze_question_string

        result = await analyze_question_string(json_content, target_grade=5)
        scores = result.get_scores()
        print(f"Accessibility: {scores['accessibility']}")
    """
    return await analyze_question(
        question=content,
        target_grade=target_grade,
        run_accessibility=run_accessibility,
        run_structural=run_structural,
        run_adversarial=run_adversarial,
        run_simulation=run_simulation,
    )


async def _run_accessibility(
    question: Dict[str, Any],
    target_grade: Optional[int],
) -> Dict[str, Any]:
    """Run accessibility analysis."""
    from .accessibility import AccessibilityAnalyzer

    analyzer = AccessibilityAnalyzer()
    result = await analyzer.analyze(question, target_grade=target_grade)
    return result.to_dict()


async def _run_structural(question: Dict[str, Any]) -> Dict[str, Any]:
    """Run structural/misconception analysis."""
    analyzer = StructuralAnalyzer()
    return await analyzer.analyze(question)


async def _run_adversarial(question: Dict[str, Any]) -> Dict[str, Any]:
    """Run adversarial testing."""
    from .adversarial import AdversarialSwarm

    swarm = AdversarialSwarm()
    result = await swarm.test(question)
    return result


async def _run_simulation(
    question: Dict[str, Any],
    target_grade: Optional[int],
    archetypes: Optional[List[str]],
) -> Dict[str, Any]:
    """Run student simulation."""
    from .agents.v2 import AgenticOrchestrator

    orchestrator = AgenticOrchestrator()

    # Build student cohort
    if archetypes:
        students = [
            {"grade": target_grade or 8, "archetype": arch}
            for arch in archetypes
        ]
    elif target_grade:
        students = [
            {"grade": target_grade, "archetype": "average_student"},
            {"grade": target_grade, "archetype": "class_clown"},
            {"grade": target_grade, "archetype": "honors_overachiever"},
            {"grade": target_grade, "archetype": "esl_student"},
            {"grade": max(3, target_grade - 2), "archetype": "average_student"},
            {"grade": min(12, target_grade + 2), "archetype": "average_student"},
        ]
    else:
        students = [
            {"grade": 5, "archetype": "average_student"},
            {"grade": 8, "archetype": "average_student"},
            {"grade": 8, "archetype": "honors_overachiever"},
            {"grade": 11, "archetype": "class_clown"},
        ]

    results = await orchestrator.simulate_batch(
        question=question,
        correct_answer=question.get("correct_answer"),
        students=students,
    )

    # Aggregate results
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    by_archetype: Dict[str, Dict[str, Any]] = {}
    for r in results:
        arch = r.archetype
        if arch not in by_archetype:
            by_archetype[arch] = {"correct": 0, "total": 0}
        by_archetype[arch]["total"] += 1
        if r.is_correct:
            by_archetype[arch]["correct"] += 1

    for arch, data in by_archetype.items():
        data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total_simulated": total,
        "by_archetype": by_archetype,
        "difficulty_estimate": 1 - (correct / total) if total > 0 else 0.5,
    }


@dataclass
class BatchHealthResult:
    """Result from analyze_batch_health()."""

    questions_needing_attention: List[str]
    generator_feedback: List[str]
    risk_score: float
    patterns: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions_needing_attention": self.questions_needing_attention,
            "generator_feedback": self.generator_feedback,
            "risk_score": self.risk_score,
            "patterns": self.patterns,
        }


async def analyze_batch_health(
    contents: List[str],
    ids: Optional[List[str]] = None,
) -> BatchHealthResult:
    """
    Analyze batch health from raw content strings.

    Parses each string, runs heuristics, identifies questions needing attention.
    This is FREE (no LLM calls) - just pattern detection.

    Args:
        contents: List of JSON strings or question content
        ids: Optional list of IDs (auto-generated if not provided)

    Returns:
        BatchHealthResult with flagged questions and generator feedback

    Example:
        from mirofish_simulator import analyze_batch_health

        result = await analyze_batch_health(json_strings)
        skip_ids = set(result.questions_needing_attention)
        # Only run deep analysis on flagged questions
    """
    from .batch_health import BatchHealthAnalyzer

    # Parse all questions
    questions = []
    for i, content in enumerate(contents):
        parsed = parse_question(content)
        if parsed:
            qid = ids[i] if ids and i < len(ids) else f"q{i}"
            parsed["id"] = qid
            questions.append(parsed)

    if not questions:
        return BatchHealthResult(
            questions_needing_attention=[],
            generator_feedback=["No valid questions to analyze"],
            risk_score=0.0,
        )

    # Run batch health analyzer
    analyzer = BatchHealthAnalyzer()
    report = await analyzer.analyze(questions)

    return BatchHealthResult(
        questions_needing_attention=list(report.questions_needing_attention),
        generator_feedback=report.generator_feedback,
        risk_score=report.patterns.risk_score if report.patterns else 0.0,
        patterns=report.patterns.__dict__ if report.patterns else None,
    )
