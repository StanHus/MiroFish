# MiroFish Simulator

Student population simulation for educational content evaluation.

Simulates how diverse student archetypes would respond to educational content, providing psychometric analysis and engagement predictions.

## Installation

```bash
pip install mirofish-simulator
```

Or install from source:
```bash
pip install -e packages/mirofish-simulator
```

## Quick Start

```python
import asyncio
from mirofish_simulator import simulate_content

content = {
    "text": "What is the primary function of the Electoral College?",
    "options": [
        "To directly elect members of Congress",
        "To formally elect the President and Vice President",
        "To approve Supreme Court nominations",
        "To ratify constitutional amendments"
    ],
    "correct_answer": "B",
    "grade": "11",
    "subject": "AP Government",
}

result = asyncio.run(simulate_content(content, population_size=30))

print(f"Accuracy: {result.accuracy:.1%}")
print(f"Difficulty (IRT): {result.difficulty_irt:.2f}")
print(f"Discrimination: {result.discrimination_irt:.2f}")

# Archetype breakdown
for arch, perf in result.by_archetype.items():
    print(f"  {arch}: {perf.accuracy:.0%} accuracy")

# Concerns
for concern in result.concerns:
    print(f"  ⚠ {concern}")
```

## Full Usage

```python
from mirofish_simulator import Simulator

sim = Simulator(
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
    model="gpt-4o-mini"
)

result = await sim.simulate(
    content=content,
    population_config={
        "size": 50,
        "archetypes": ["honors_overachiever", "quiet_thinker", "esl_student"],
        "distribution": {
            "honors_overachiever": 0.4,
            "quiet_thinker": 0.4,
            "esl_student": 0.2,
        }
    }
)
```

## Student Archetypes

| Archetype | Base Accuracy | Engagement | Description |
|-----------|--------------|------------|-------------|
| `honors_overachiever` | 92% | High | High-achieving, focused on excellence |
| `debate_club_kid` | 78% | High | Intellectually curious, argumentative |
| `quiet_thinker` | 70% | Medium | Thoughtful introvert |
| `socially_engaged_activist` | 68% | High | Passionate about social issues |
| `disengaged_but_smart` | 55% | Low | Capable but unmotivated |
| `esl_student` | 45% | Medium | English language learner |
| `class_clown` | 35% | Low | Prioritizes humor |
| `politically_conservative` | 65% | Medium | Traditional viewpoints |

## Output Metrics

### Aggregate
- `accuracy` - Overall correct answer rate
- `difficulty_irt` - IRT difficulty parameter (-2 to +2, higher = harder)
- `discrimination_irt` - IRT discrimination (higher = better differentiates students)
- `avg_time_seconds` - Average response time
- `engagement_score` - Average engagement level

### By Archetype
- Per-archetype accuracy, time, and confidence

### Distractor Analysis
- Selection rate per wrong answer
- Which archetypes are attracted to each distractor
- Effectiveness flags and concerns

### Concerns & Recommendations
- Accessibility issues
- Difficulty problems
- Distractor effectiveness issues

## Integration with InceptBench

```python
# In InceptBench evaluator
from mirofish_simulator import Simulator

class SimulationEvaluator:
    def __init__(self):
        self.sim = Simulator()

    async def evaluate(self, content: dict) -> dict:
        result = await self.sim.simulate(content)
        return {
            "psychometric": {
                "difficulty": result.difficulty_irt,
                "discrimination": result.discrimination_irt,
            },
            "accessibility": {
                "concerns": result.concerns,
                "by_archetype": result.to_dict()["by_archetype"],
            }
        }
```

## License

MIT
