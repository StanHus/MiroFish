"""
Simulation Runner - Orchestrates running multiple student agents.

Runs a population of student agents against a question and
aggregates the results.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .student_agent import StudentAgent, AgentResponse, AgentConfig


# Default population distribution
DEFAULT_POPULATION = {
    "honors_overachiever": 0.15,
    "debate_club_kid": 0.10,
    "quiet_thinker": 0.20,
    "socially_engaged_activist": 0.10,
    "disengaged_but_smart": 0.15,
    "esl_student": 0.10,
    "class_clown": 0.10,
    "politically_conservative": 0.10,
}


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    population_size: int = 30
    grade: int = 8
    archetypes: Optional[Dict[str, float]] = None  # Archetype -> proportion

    # Agent settings
    model: str = "gpt-4o-mini"

    # Concurrency
    max_concurrent: int = 10

    def get_archetypes(self) -> Dict[str, float]:
        return self.archetypes or DEFAULT_POPULATION


@dataclass
class ArchetypeResult:
    """Results for a single archetype."""

    archetype: str
    count: int
    correct: int
    accuracy: float
    responses: List[AgentResponse]

    # Common patterns
    most_selected: str
    confidence_distribution: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "count": self.count,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 3),
            "most_selected": self.most_selected,
            "confidence_distribution": self.confidence_distribution,
        }


@dataclass
class SimulationResult:
    """Complete simulation results."""

    question_text: str
    correct_answer: str
    grade: int

    # Aggregate
    total_responses: int
    total_correct: int
    accuracy: float

    # By archetype
    by_archetype: Dict[str, ArchetypeResult]

    # Response distribution
    response_distribution: Dict[str, int]  # "A" -> count

    # All responses
    responses: List[AgentResponse]

    # Common knowledge gaps
    common_knowledge_gaps: List[str]
    common_misunderstandings: List[str]

    def to_dict(self) -> dict:
        return {
            "question": self.question_text[:100] + "..." if len(self.question_text) > 100 else self.question_text,
            "correct_answer": self.correct_answer,
            "grade": self.grade,
            "total_responses": self.total_responses,
            "total_correct": self.total_correct,
            "accuracy": round(self.accuracy, 3),
            "response_distribution": self.response_distribution,
            "by_archetype": {k: v.to_dict() for k, v in self.by_archetype.items()},
            "common_knowledge_gaps": self.common_knowledge_gaps[:5],
            "common_misunderstandings": self.common_misunderstandings[:5],
        }


class SimulationRunner:
    """Runs simulations with multiple student agents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def run(
        self,
        question: Dict[str, Any],
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """
        Run a simulation with multiple student agents.

        Args:
            question: Question with "text", "options", "correct_answer"
            config: Simulation configuration

        Returns:
            SimulationResult with aggregate and per-archetype results
        """
        config = config or SimulationConfig()

        text = question.get("text") or question.get("question", "")
        correct_answer = question.get("correct_answer", "A").upper()
        grade = config.grade

        # Build population
        agents = self._build_population(config)

        # Run agents
        responses = await self._run_agents(agents, question, correct_answer, config.max_concurrent)

        # Aggregate results
        return self._aggregate_results(responses, text, correct_answer, grade)

    def _build_population(self, config: SimulationConfig) -> List[StudentAgent]:
        """Build population of student agents."""
        agents = []
        archetypes = config.get_archetypes()

        for archetype, proportion in archetypes.items():
            count = max(1, int(config.population_size * proportion))
            for _ in range(count):
                agent = StudentAgent(
                    grade=config.grade,
                    archetype=archetype,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=config.model,
                )
                agents.append(agent)

        return agents

    async def _run_agents(
        self,
        agents: List[StudentAgent],
        question: Dict[str, Any],
        correct_answer: str,
        max_concurrent: int,
    ) -> List[AgentResponse]:
        """Run all agents with concurrency limit."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(agent: StudentAgent) -> AgentResponse:
            async with semaphore:
                return await agent.answer(question, correct_answer)

        tasks = [run_one(agent) for agent in agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, AgentResponse):
                # Tag with archetype
                resp._archetype = agents[i].archetype
                valid_responses.append(resp)

        return valid_responses

    def _aggregate_results(
        self,
        responses: List[AgentResponse],
        question_text: str,
        correct_answer: str,
        grade: int,
    ) -> SimulationResult:
        """Aggregate individual responses into summary."""

        # Overall stats
        total = len(responses)
        correct = sum(1 for r in responses if r.is_correct)
        accuracy = correct / total if total > 0 else 0

        # Response distribution
        response_dist = Counter(r.selected for r in responses)

        # By archetype
        by_archetype = {}
        archetype_responses: Dict[str, List[AgentResponse]] = {}

        for resp in responses:
            arch = getattr(resp, "_archetype", "unknown")
            if arch not in archetype_responses:
                archetype_responses[arch] = []
            archetype_responses[arch].append(resp)

        for arch, arch_resps in archetype_responses.items():
            arch_correct = sum(1 for r in arch_resps if r.is_correct)
            arch_acc = arch_correct / len(arch_resps) if arch_resps else 0

            selections = Counter(r.selected for r in arch_resps)
            most_selected = selections.most_common(1)[0][0] if selections else "?"

            confidence_dist = Counter(r.confidence for r in arch_resps)

            by_archetype[arch] = ArchetypeResult(
                archetype=arch,
                count=len(arch_resps),
                correct=arch_correct,
                accuracy=arch_acc,
                responses=arch_resps,
                most_selected=most_selected,
                confidence_distribution=dict(confidence_dist),
            )

        # Common knowledge gaps
        all_gaps = []
        for r in responses:
            all_gaps.extend(r.knowledge_gaps)
        common_gaps = [g for g, c in Counter(all_gaps).most_common(10)]

        # Common misunderstandings (from perceived vs actual)
        misunderstandings = []
        for r in responses:
            if r.what_they_thought_question_asked != question_text:
                misunderstandings.append(r.what_they_thought_question_asked[:100])

        return SimulationResult(
            question_text=question_text,
            correct_answer=correct_answer,
            grade=grade,
            total_responses=total,
            total_correct=correct,
            accuracy=accuracy,
            by_archetype=by_archetype,
            response_distribution=dict(response_dist),
            responses=responses,
            common_knowledge_gaps=common_gaps,
            common_misunderstandings=misunderstandings[:5],
        )


async def run_simulation(
    question: Dict[str, Any],
    grade: int = 8,
    population_size: int = 30,
    archetypes: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
) -> SimulationResult:
    """
    Convenience function to run a simulation.

    Args:
        question: Question with "text", "options", "correct_answer"
        grade: Target grade level
        population_size: Number of agents
        archetypes: Custom archetype distribution
        api_key: OpenAI API key

    Returns:
        SimulationResult
    """
    runner = SimulationRunner(api_key=api_key)
    config = SimulationConfig(
        population_size=population_size,
        grade=grade,
        archetypes=archetypes,
    )
    return await runner.run(question, config)
