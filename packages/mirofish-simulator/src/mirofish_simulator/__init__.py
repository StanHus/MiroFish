"""
MiroFish Simulator - Student population simulation for educational content evaluation.

Usage:
    from mirofish_simulator import Simulator, simulate_content

    # Quick usage
    result = await simulate_content(
        content={"text": "What is 2+2?", "options": ["3", "4", "5", "6"], "correct_answer": "B"},
        population_size=30
    )

    # Full control
    sim = Simulator(api_key="your-openai-key")
    result = await sim.simulate(content, population_config={...})
"""

from .simulator import (
    Simulator,
    SimulationResult,
    StudentResponse,
    ArchetypePerformance,
    DistractorAnalysis,
    simulate_content,
    ARCHETYPES,
    DEFAULT_POPULATION,
)

from .profiles import StudentProfile

__version__ = "0.1.0"
__all__ = [
    "Simulator",
    "SimulationResult",
    "StudentResponse",
    "StudentProfile",
    "ArchetypePerformance",
    "DistractorAnalysis",
    "simulate_content",
    "ARCHETYPES",
    "DEFAULT_POPULATION",
]
