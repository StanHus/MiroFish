"""
Orchestrator - Coordinates the multi-agent simulation.

This is the main entry point. It:
1. Runs the Knowledge Agent to build constraints
2. Runs the Perception Agent to see how student reads question
3. Runs the Answer Agent with constraints
4. Runs the Verifier Agent to catch cheating
5. Optionally reruns if verification fails
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .knowledge_agent import KnowledgeAgent, KnowledgeProfile
from .perception_agent import PerceptionAgent, PerceptionResult
from .answer_agent import AnswerAgent, AnswerResult
from .verifier_agent import VerifierAgent, VerificationResult


@dataclass
class SimulationResult:
    """Complete result of simulating a student answering a question."""

    # Student profile
    grade: int
    archetype: str

    # The answer
    selected: str
    selected_text: str
    is_correct: Optional[bool]
    confidence: str

    # Agent outputs
    knowledge_profile: KnowledgeProfile
    perception: PerceptionResult
    answer_attempt: AnswerResult
    verification: VerificationResult

    # Meta
    was_rerun: bool
    rerun_count: int
    final_consistency: float

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "archetype": self.archetype,
            "selected": self.selected,
            "selected_text": self.selected_text,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "knowledge_profile": self.knowledge_profile.to_dict(),
            "perception": self.perception.to_dict(),
            "answer_attempt": self.answer_attempt.to_dict(),
            "verification": self.verification.to_dict(),
            "was_rerun": self.was_rerun,
            "rerun_count": self.rerun_count,
            "final_consistency": self.final_consistency,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        correct_str = "✓" if self.is_correct else "✗" if self.is_correct is not None else "?"
        consistency = "consistent" if self.verification.is_consistent else "INCONSISTENT"

        return f"""Grade {self.grade} {self.archetype}:
  Answer: {self.selected} ({correct_str}) - {self.confidence}
  Verification: {consistency} ({self.final_consistency:.0%} confidence)
  Reasoning: {self.answer_attempt.thought_process[:100]}..."""


class StudentSimulator:
    """
    Main simulator that orchestrates all agents.

    Usage:
        simulator = StudentSimulator(api_key="sk-...")
        result = await simulator.simulate(
            question={"text": "...", "options": [...]},
            grade=5,
            archetype="esl_student",
            correct_answer="B",
        )

        print(result.selected)  # The answer
        print(result.is_correct)  # Whether correct
        print(result.verification.is_consistent)  # Whether consistent with knowledge

    Modes:
        - "auto": Use misconception-driven when student has knowledge gaps (recommended)
        - "constrained": Traditional approach (LLM often cheats)
        - "misconception_driven": Always use misconception-based answering
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_reruns: int = 2,
        answer_mode: str = "auto",  # "auto", "constrained", "misconception_driven"
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.max_reruns = max_reruns
        self.answer_mode = answer_mode

        # Initialize agents
        self.knowledge_agent = KnowledgeAgent(self.api_key, self.base_url, model)
        self.perception_agent = PerceptionAgent(self.api_key, self.base_url, model)
        self.answer_agent = AnswerAgent(self.api_key, self.base_url, model, mode=answer_mode)
        self.verifier_agent = VerifierAgent(self.api_key, self.base_url, model)

    async def simulate(
        self,
        question: Dict[str, Any],
        grade: int,
        archetype: str,
        correct_answer: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> SimulationResult:
        """
        Simulate a student answering a question.

        Args:
            question: Question with "text" and "options"
            grade: Student grade level (1-12)
            archetype: Student type
            correct_answer: Correct answer for marking (not shown to answer agent)
            subject: Optional subject area

        Returns:
            SimulationResult with full agent chain output
        """
        # Step 1: Build knowledge profile
        knowledge = await self.knowledge_agent.build_profile(
            question, grade, archetype, subject
        )

        # Step 2: Generate perception
        perception = await self.perception_agent.perceive(question, knowledge)

        # Step 3: Generate answer
        answer = await self.answer_agent.answer(question, knowledge, perception)

        # Step 4: Verify consistency
        verification = await self.verifier_agent.verify(
            question, knowledge, answer, correct_answer
        )

        # Step 5: Handle inconsistency (rerun if needed)
        rerun_count = 0
        was_rerun = False

        while (
            not verification.is_consistent
            and verification.should_rerun
            and rerun_count < self.max_reruns
        ):
            was_rerun = True
            rerun_count += 1

            # Rerun with stronger constraints
            answer = await self._rerun_with_correction(
                question, knowledge, perception, verification
            )

            # Re-verify
            verification = await self.verifier_agent.verify(
                question, knowledge, answer, correct_answer
            )

        # Determine correctness
        is_correct = None
        if correct_answer:
            is_correct = answer.selected.upper() == correct_answer.upper()

        return SimulationResult(
            grade=grade,
            archetype=archetype,
            selected=answer.selected,
            selected_text=answer.selected_text,
            is_correct=is_correct,
            confidence=answer.confidence,
            knowledge_profile=knowledge,
            perception=perception,
            answer_attempt=answer,
            verification=verification,
            was_rerun=was_rerun,
            rerun_count=rerun_count,
            final_consistency=verification.confidence_score,
        )

    async def _rerun_with_correction(
        self,
        question: Dict[str, Any],
        knowledge: KnowledgeProfile,
        perception: PerceptionResult,
        verification: VerificationResult,
    ) -> AnswerResult:
        """Rerun answer generation with verification feedback."""
        # For now, just rerun with higher temperature
        # Could inject verification feedback for more sophisticated correction
        return await self.answer_agent.answer(question, knowledge, perception)


async def simulate_student(
    question: Dict[str, Any],
    grade: int = 8,
    archetype: str = "average_student",
    correct_answer: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    answer_mode: str = "auto",  # "auto", "constrained", "misconception_driven"
) -> SimulationResult:
    """
    Convenience function to simulate a single student.

    Args:
        question: Question with "text" and "options"
        grade: Student grade (1-12)
        archetype: Student type
        correct_answer: Correct answer for marking
        api_key: OpenAI API key
        model: Model to use
        answer_mode: How to generate answers
            - "auto": Use misconception-driven when student has gaps (recommended)
            - "constrained": Try to limit knowledge (LLM often cheats)
            - "misconception_driven": Always pick based on misconceptions

    Returns:
        SimulationResult
    """
    simulator = StudentSimulator(api_key=api_key, model=model, answer_mode=answer_mode)
    return await simulator.simulate(question, grade, archetype, correct_answer)


async def simulate_classroom(
    question: Dict[str, Any],
    grade: int = 8,
    correct_answer: Optional[str] = None,
    population_size: int = 20,
    archetype_distribution: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 5,
    answer_mode: str = "auto",  # "auto", "constrained", "misconception_driven"
) -> Dict[str, Any]:
    """
    Simulate a classroom of students answering a question.

    Args:
        question: Question with "text" and "options"
        grade: Grade level
        correct_answer: Correct answer
        population_size: Number of students
        archetype_distribution: Distribution of archetypes (default: realistic mix)
        api_key: OpenAI API key
        model: Model to use
        max_concurrent: Max concurrent API calls

    Returns:
        Dictionary with classroom results
    """
    # Default distribution if not provided
    if archetype_distribution is None:
        archetype_distribution = {
            "honors_overachiever": 0.10,
            "debate_club_kid": 0.05,
            "quiet_thinker": 0.20,
            "average_student": 0.35,
            "disengaged_but_smart": 0.10,
            "esl_student": 0.10,
            "class_clown": 0.10,
        }

    # Generate student list
    students = []
    for archetype, proportion in archetype_distribution.items():
        count = int(population_size * proportion)
        students.extend([archetype] * count)

    # Fill remaining with average students
    while len(students) < population_size:
        students.append("average_student")
    students = students[:population_size]

    # Run simulations with concurrency limit
    simulator = StudentSimulator(api_key=api_key, model=model, answer_mode=answer_mode)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(archetype):
        async with semaphore:
            return await simulator.simulate(
                question, grade, archetype, correct_answer
            )

    results = await asyncio.gather(*[run_one(arch) for arch in students])

    # Aggregate results
    correct_count = sum(1 for r in results if r.is_correct)
    consistent_count = sum(1 for r in results if r.verification.is_consistent)

    response_dist = {}
    for r in results:
        response_dist[r.selected] = response_dist.get(r.selected, 0) + 1

    by_archetype = {}
    for r in results:
        if r.archetype not in by_archetype:
            by_archetype[r.archetype] = {"correct": 0, "total": 0, "consistent": 0}
        by_archetype[r.archetype]["total"] += 1
        if r.is_correct:
            by_archetype[r.archetype]["correct"] += 1
        if r.verification.is_consistent:
            by_archetype[r.archetype]["consistent"] += 1

    return {
        "total_students": len(results),
        "accuracy": correct_count / len(results) if results else 0,
        "consistency_rate": consistent_count / len(results) if results else 0,
        "response_distribution": response_dist,
        "correct_answer": correct_answer,
        "by_archetype": {
            arch: {
                "accuracy": data["correct"] / data["total"] if data["total"] else 0,
                "consistency": data["consistent"] / data["total"] if data["total"] else 0,
                "count": data["total"],
            }
            for arch, data in by_archetype.items()
        },
        "individual_results": [r.to_dict() for r in results],
    }
