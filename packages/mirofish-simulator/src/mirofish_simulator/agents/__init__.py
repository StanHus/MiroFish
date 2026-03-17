"""
MiroFish Agent-Based Simulation.

The core insight: LLMs know everything, but students don't.

To simulate a student, we create an LLM agent that:
1. Only has access to knowledge a student at that grade would have
2. Perceives text through the filter of their reading level
3. Actually attempts to answer - and genuinely gets things wrong

This is NOT:
- Static analysis (measuring text properties)
- Probabilistic dice rolls (random based on accuracy rates)
- Asking "what would a student pick?" (LLM still knows the answer)

This IS:
- An agent with constrained knowledge attempting to answer
- The agent doesn't know the right answer
- Errors come from genuine knowledge gaps

Usage:
    from mirofish_simulator.agents import StudentAgent, run_simulation

    agent = StudentAgent(
        grade=5,
        archetype="esl_student",
        api_key="sk-..."
    )

    response = await agent.answer(question)
    print(response.selected)      # "B" (might be wrong!)
    print(response.reasoning)     # Why they picked it
    print(response.knowledge_used)  # What they knew
    print(response.knowledge_gaps)  # What they didn't know
"""

from .student_agent import (
    StudentAgent,
    AgentResponse,
    AgentConfig,
)

from .knowledge import (
    KnowledgeBase,
    GradeKnowledge,
    build_knowledge_constraint,
)

from .perception import (
    PerceptionFilter,
    PerceivedContent,
    apply_perception_filter,
)

from .runner import (
    SimulationRunner,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)

__all__ = [
    # Core agent
    "StudentAgent",
    "AgentResponse",
    "AgentConfig",
    # Knowledge
    "KnowledgeBase",
    "GradeKnowledge",
    "build_knowledge_constraint",
    # Perception
    "PerceptionFilter",
    "PerceivedContent",
    "apply_perception_filter",
    # Runner
    "SimulationRunner",
    "SimulationConfig",
    "SimulationResult",
    "run_simulation",
]
