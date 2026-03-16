"""
InceptBench Simulator Service

Simulates student populations responding to educational content.
Provides psychometric analysis and engagement predictions for InceptBench integration.

Usage:
    from app.services.inceptbench_simulator import InceptBenchSimulator

    simulator = InceptBenchSimulator()
    result = await simulator.simulate_content(content, population_config)
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ..config import Config
from ..utils.logger import get_logger
from .student_profile_generator import (
    StudentProfile,
    StudentProfileGenerator,
    ARCHETYPE_MBTI_POOLS,
)

logger = get_logger("mirofish.inceptbench")


# ── Response Models ─────────────────────────────────────────────────────────

@dataclass
class StudentResponse:
    """A single student's response to content."""
    student_id: int
    student_name: str
    archetype: str
    selected_answer: str
    is_correct: bool
    confidence: float  # 0-1
    time_seconds: float
    reasoning: str = ""
    engagement_level: float = 0.5


@dataclass
class DistractorAnalysis:
    """Analysis of a single distractor option."""
    option: str
    selection_rate: float
    attracted_archetypes: List[str] = field(default_factory=list)
    is_effective: bool = True
    concern: Optional[str] = None


@dataclass
class ArchetypePerformance:
    """Performance metrics for a student archetype."""
    archetype: str
    count: int
    accuracy: float
    avg_time_seconds: float
    avg_confidence: float
    avg_engagement: float


@dataclass
class SimulationResult:
    """Complete simulation result."""
    # Aggregate metrics
    total_students: int
    accuracy: float
    difficulty_irt: float  # IRT b parameter (higher = harder)
    discrimination_irt: float  # IRT a parameter (higher = better discriminates)
    avg_time_seconds: float
    engagement_score: float

    # Breakdowns
    by_archetype: Dict[str, ArchetypePerformance]
    distractor_analysis: Dict[str, DistractorAnalysis]

    # Concerns and recommendations
    concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Raw data
    responses: List[StudentResponse] = field(default_factory=list)
    simulation_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate": {
                "total_students": self.total_students,
                "accuracy": round(self.accuracy, 3),
                "difficulty_irt": round(self.difficulty_irt, 3),
                "discrimination_irt": round(self.discrimination_irt, 3),
                "avg_time_seconds": round(self.avg_time_seconds, 1),
                "engagement_score": round(self.engagement_score, 3),
            },
            "by_archetype": {
                k: {
                    "count": v.count,
                    "accuracy": round(v.accuracy, 3),
                    "avg_time_seconds": round(v.avg_time_seconds, 1),
                    "avg_confidence": round(v.avg_confidence, 3),
                    "avg_engagement": round(v.avg_engagement, 3),
                }
                for k, v in self.by_archetype.items()
            },
            "distractor_analysis": {
                k: {
                    "selection_rate": round(v.selection_rate, 3),
                    "attracted_archetypes": v.attracted_archetypes,
                    "is_effective": v.is_effective,
                    "concern": v.concern,
                }
                for k, v in self.distractor_analysis.items()
            },
            "concerns": self.concerns,
            "recommendations": self.recommendations,
            "simulation_time_ms": self.simulation_time_ms,
        }


# ── Default Population Configs ──────────────────────────────────────────────

DEFAULT_ARCHETYPES = [
    "honors_overachiever",
    "debate_club_kid",
    "quiet_thinker",
    "socially_engaged_activist",
    "disengaged_but_smart",
    "esl_student",
    "class_clown",
    "politically_conservative",
]

DEFAULT_POPULATION = {
    "size": 30,
    "archetypes": DEFAULT_ARCHETYPES,
    "distribution": {
        "honors_overachiever": 0.15,
        "debate_club_kid": 0.10,
        "quiet_thinker": 0.15,
        "socially_engaged_activist": 0.10,
        "disengaged_but_smart": 0.15,
        "esl_student": 0.10,
        "class_clown": 0.10,
        "politically_conservative": 0.15,
    },
}


# ── Main Simulator Class ────────────────────────────────────────────────────

class InceptBenchSimulator:
    """
    Simulates student populations responding to educational content.

    Features:
    - Generates diverse student profiles using StudentProfileGenerator
    - Simulates each student's response to content via LLM
    - Computes psychometric metrics (IRT difficulty/discrimination)
    - Analyzes distractor effectiveness
    - Identifies accessibility concerns
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.profile_generator = StudentProfileGenerator(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
        )

    async def simulate_content(
        self,
        content: Dict[str, Any],
        population_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationResult:
        """
        Main entry point: simulate student population responding to content.

        Args:
            content: Educational content to evaluate
                - type: "question" | "quiz" | "reading"
                - text: The question/content text
                - options: List of answer options (for MCQ)
                - correct_answer: The correct answer
                - grade: Target grade level
                - subject: Subject area
            population_config: Student population parameters
                - size: Number of students (default 30)
                - archetypes: List of archetypes to include
                - distribution: Weight per archetype
            progress_callback: (current, total, message) for progress updates

        Returns:
            SimulationResult with metrics and analysis
        """
        start_time = time.time()

        pop_config = {**DEFAULT_POPULATION, **(population_config or {})}

        # Generate student profiles
        profiles = self._generate_population(pop_config)
        total = len(profiles)

        if progress_callback:
            progress_callback(0, total, "Generated student population")

        # Simulate each student's response
        responses: List[StudentResponse] = []

        # Process in batches for efficiency
        batch_size = 5
        for i in range(0, total, batch_size):
            batch = profiles[i:i + batch_size]
            batch_responses = await asyncio.gather(*[
                self._simulate_student_response(profile, content)
                for profile in batch
            ])
            responses.extend(batch_responses)

            if progress_callback:
                progress_callback(
                    min(i + batch_size, total),
                    total,
                    f"Simulated {min(i + batch_size, total)}/{total} students"
                )

        # Compute metrics
        result = self._compute_metrics(responses, content)
        result.simulation_time_ms = int((time.time() - start_time) * 1000)
        result.responses = responses

        logger.info(
            f"Simulation complete: {total} students, "
            f"accuracy={result.accuracy:.2%}, "
            f"difficulty={result.difficulty_irt:.2f}, "
            f"time={result.simulation_time_ms}ms"
        )

        return result

    def _generate_population(self, config: Dict[str, Any]) -> List[StudentProfile]:
        """Generate student profiles based on population config."""
        size = config.get("size", 30)
        distribution = config.get("distribution", {})
        archetypes = config.get("archetypes", DEFAULT_ARCHETYPES)

        profiles = []

        # Calculate counts per archetype
        archetype_counts = {}
        remaining = size

        for arch in archetypes[:-1]:
            weight = distribution.get(arch, 1.0 / len(archetypes))
            count = int(size * weight)
            archetype_counts[arch] = count
            remaining -= count

        # Last archetype gets the remainder
        archetype_counts[archetypes[-1]] = remaining

        # Generate profiles
        user_id = 0
        for archetype, count in archetype_counts.items():
            for _ in range(count):
                profile = self._create_quick_profile(user_id, archetype)
                profiles.append(profile)
                user_id += 1

        random.shuffle(profiles)
        return profiles

    def _create_quick_profile(self, user_id: int, archetype: str) -> StudentProfile:
        """Create a student profile quickly (without LLM enrichment)."""
        # Use the profile generator's internal methods
        gender = random.choice(["male", "female"])
        age = random.choice([16, 17, 17, 17, 18])
        grade = 11 if age <= 17 else 12
        mbti = random.choice(ARCHETYPE_MBTI_POOLS.get(archetype, ["INTP"]))

        # Traits based on archetype
        trait_map = {
            "honors_overachiever": {"engagement": 0.95, "knowledge": "advanced", "gpa": "3.8-4.0"},
            "debate_club_kid": {"engagement": 0.85, "knowledge": "advanced", "gpa": "3.5-3.9"},
            "quiet_thinker": {"engagement": 0.50, "knowledge": "intermediate", "gpa": "3.2-3.6"},
            "socially_engaged_activist": {"engagement": 0.80, "knowledge": "intermediate", "gpa": "3.3-3.7"},
            "disengaged_but_smart": {"engagement": 0.30, "knowledge": "advanced", "gpa": "2.8-3.4"},
            "esl_student": {"engagement": 0.60, "knowledge": "basic", "gpa": "2.5-3.2"},
            "class_clown": {"engagement": 0.40, "knowledge": "basic", "gpa": "2.3-2.9"},
            "politically_conservative": {"engagement": 0.70, "knowledge": "intermediate", "gpa": "3.0-3.5"},
        }

        traits = trait_map.get(archetype, {"engagement": 0.5, "knowledge": "intermediate", "gpa": "3.0-3.5"})

        return StudentProfile(
            user_id=user_id,
            username=f"student_{user_id}",
            name=f"Student {user_id}",
            bio=f"Grade {grade} student",
            persona=f"A {archetype.replace('_', ' ')} student.",
            archetype_id=archetype,
            age=age,
            gender=gender,
            mbti=mbti,
            grade=grade,
            gpa_range=traits["gpa"],
            engagement_level=traits["engagement"],
            learning_style="mixed",
        )

    async def _simulate_student_response(
        self,
        profile: StudentProfile,
        content: Dict[str, Any],
    ) -> StudentResponse:
        """
        Simulate a single student's response using hybrid approach.

        Hybrid approach:
        1. LLM analyzes the appeal of each option for different student types
        2. Archetype base_accuracy determines if student gets it correct
        3. If incorrect, pick the most tempting distractor from LLM analysis
        """
        text = content.get("text", content.get("question", ""))
        options = content.get("options", [])
        correct_answer = content.get("correct_answer", "")
        subject = content.get("subject", "General")

        # Get correct letter
        if correct_answer.isalpha() and len(correct_answer) == 1:
            correct_letter = correct_answer.upper()
        else:
            try:
                correct_letter = chr(65 + int(correct_answer))
            except:
                correct_letter = "A"

        # If no LLM client, use probabilistic fallback
        if not self.client:
            return self._fallback_response(profile, options, correct_answer)

        # Get distractor analysis from LLM (cached per question)
        distractor_appeal = await self._analyze_distractors(content)

        # Determine if student answers correctly based on archetype
        accuracy_map = {
            "honors_overachiever": 0.92,
            "debate_club_kid": 0.78,
            "quiet_thinker": 0.70,
            "socially_engaged_activist": 0.68,
            "disengaged_but_smart": 0.55,
            "esl_student": 0.45,
            "class_clown": 0.35,
            "politically_conservative": 0.65,
        }

        base_accuracy = accuracy_map.get(profile.archetype_id, 0.5)

        # Roll dice: does this student get it correct?
        gets_correct = random.random() < base_accuracy

        if gets_correct:
            selected = correct_letter
            reasoning = "Correctly identified the answer"
            confidence = 0.7 + random.uniform(0, 0.25)
        else:
            # Pick most tempting wrong answer for this archetype
            selected = self._pick_tempting_distractor(
                profile.archetype_id,
                correct_letter,
                distractor_appeal,
                options
            )
            reasoning = distractor_appeal.get(
                selected,
                {profile.archetype_id: "seemed plausible"}
            ).get(profile.archetype_id, "seemed like a reasonable choice")
            confidence = 0.4 + random.uniform(0, 0.3)

        # Time varies by engagement level
        base_time = 45
        time_var = (1 - profile.engagement_level) * 30
        time_seconds = base_time + time_var + random.uniform(-10, 10)

        return StudentResponse(
            student_id=profile.user_id,
            student_name=profile.name,
            archetype=profile.archetype_id,
            selected_answer=selected,
            is_correct=gets_correct,
            confidence=confidence,
            time_seconds=max(10, time_seconds),
            reasoning=reasoning,
            engagement_level=profile.engagement_level,
        )

    async def _analyze_distractors(
        self,
        content: Dict[str, Any],
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze why each option might appeal to different student types.

        Returns dict mapping option letter -> {archetype -> reason}
        """
        # Use cached analysis if available
        cache_key = hash(str(content.get("text") or content.get("question", "")))
        if hasattr(self, "_distractor_cache") and cache_key in self._distractor_cache:
            return self._distractor_cache[cache_key]

        if not hasattr(self, "_distractor_cache"):
            self._distractor_cache = {}

        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        subject = content.get("subject", "General")

        options_text = "\n".join(
            f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
        )

        archetypes_list = ", ".join(DEFAULT_ARCHETYPES)

        prompt = f"""Analyze this {subject} question and explain why each WRONG option might appeal to different student types.

QUESTION:
{text}

OPTIONS:
{options_text}

Student archetypes: {archetypes_list}

For each option (A, B, C, D), explain briefly why it might tempt specific student types (especially why wrong answers seem plausible).

Respond with JSON only:
{{
  "A": {{"class_clown": "seems funny", "esl_student": "simpler wording"}},
  "B": {{"quiet_thinker": "overthinking leads here"}},
  ...
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze educational question distractors. "
                            "Explain why wrong answers might appeal to specific student types. "
                            "Respond ONLY with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)
            self._distractor_cache[cache_key] = analysis
            return analysis

        except Exception as e:
            logger.warning(f"Distractor analysis failed: {e}")
            # Fallback: no specific distractor reasoning
            return {}

    def _pick_tempting_distractor(
        self,
        archetype: str,
        correct_letter: str,
        distractor_appeal: Dict[str, Dict[str, str]],
        options: List[str],
    ) -> str:
        """Pick the most tempting wrong answer for this archetype."""
        all_opts = ["A", "B", "C", "D"][:len(options)]
        wrong_opts = [o for o in all_opts if o != correct_letter]

        if not wrong_opts:
            return "A"

        # Check if any distractor specifically appeals to this archetype
        for opt in wrong_opts:
            if opt in distractor_appeal:
                if archetype in distractor_appeal[opt]:
                    return opt

        # Otherwise pick randomly from wrong options
        return random.choice(wrong_opts)

    def _check_correct(
        self,
        selected: str,
        correct: str,
        options: List[str],
    ) -> bool:
        """Check if the selected answer is correct."""
        # Handle various correct_answer formats
        if correct.upper() == selected.upper():
            return True

        # If correct is an index
        try:
            correct_idx = int(correct)
            return selected == chr(65 + correct_idx)
        except (ValueError, TypeError):
            pass

        # If correct is the actual answer text
        if correct in options:
            correct_idx = options.index(correct)
            return selected == chr(65 + correct_idx)

        return False

    def _fallback_response(
        self,
        profile: StudentProfile,
        options: List[str],
        correct_answer: str,
    ) -> StudentResponse:
        """Generate a probabilistic fallback response when LLM fails."""

        # Base probability of correct answer by archetype
        accuracy_map = {
            "honors_overachiever": 0.92,
            "debate_club_kid": 0.78,
            "quiet_thinker": 0.70,
            "socially_engaged_activist": 0.68,
            "disengaged_but_smart": 0.55,
            "esl_student": 0.45,
            "class_clown": 0.35,
            "politically_conservative": 0.65,
        }

        prob_correct = accuracy_map.get(profile.archetype_id, 0.5)
        is_correct = random.random() < prob_correct

        if is_correct:
            # Determine the correct answer letter
            if correct_answer.isalpha() and len(correct_answer) == 1:
                selected = correct_answer.upper()
            else:
                try:
                    idx = int(correct_answer)
                    selected = chr(65 + idx)
                except:
                    selected = "A"  # Default
        else:
            # Pick a random wrong answer
            all_options = ["A", "B", "C", "D"][:len(options)]
            wrong_options = [o for o in all_options if o != correct_answer.upper()]
            selected = random.choice(wrong_options) if wrong_options else "A"

        # Time based on engagement (more engaged = faster)
        base_time = 60 - (profile.engagement_level * 30)
        time_seconds = base_time + random.uniform(-10, 10)

        return StudentResponse(
            student_id=profile.user_id,
            student_name=profile.name,
            archetype=profile.archetype_id,
            selected_answer=selected,
            is_correct=is_correct,
            confidence=0.5 + (0.3 if is_correct else -0.2),
            time_seconds=max(10, time_seconds),
            reasoning="(simulated without LLM)",
            engagement_level=profile.engagement_level,
        )

    def _compute_metrics(
        self,
        responses: List[StudentResponse],
        content: Dict[str, Any],
    ) -> SimulationResult:
        """Compute aggregate metrics from student responses."""

        if not responses:
            return SimulationResult(
                total_students=0,
                accuracy=0,
                difficulty_irt=0,
                discrimination_irt=0,
                avg_time_seconds=0,
                engagement_score=0,
                by_archetype={},
                distractor_analysis={},
            )

        total = len(responses)
        correct_count = sum(1 for r in responses if r.is_correct)
        accuracy = correct_count / total

        # IRT difficulty (simple proportion-based estimate)
        # b ≈ -ln(p / (1-p)) for Rasch model
        p = max(0.01, min(0.99, accuracy))  # Clamp to avoid log(0)
        difficulty_irt = -1 * (p - 0.5) * 4  # Scale to roughly -2 to +2

        # IRT discrimination (based on variance in performance)
        # Higher variance = better discrimination
        archetype_accuracies = {}
        for r in responses:
            if r.archetype not in archetype_accuracies:
                archetype_accuracies[r.archetype] = []
            archetype_accuracies[r.archetype].append(1 if r.is_correct else 0)

        accuracy_variance = 0
        if len(archetype_accuracies) > 1:
            arch_means = [sum(v) / len(v) for v in archetype_accuracies.values()]
            mean_of_means = sum(arch_means) / len(arch_means)
            accuracy_variance = sum((m - mean_of_means) ** 2 for m in arch_means) / len(arch_means)

        discrimination_irt = 1.0 + (accuracy_variance * 10)  # Scale to roughly 0.5 to 2.5

        # Time and engagement
        avg_time = sum(r.time_seconds for r in responses) / total
        avg_engagement = sum(r.engagement_level for r in responses) / total

        # By archetype breakdown
        by_archetype: Dict[str, ArchetypePerformance] = {}
        archetype_groups: Dict[str, List[StudentResponse]] = {}

        for r in responses:
            if r.archetype not in archetype_groups:
                archetype_groups[r.archetype] = []
            archetype_groups[r.archetype].append(r)

        for arch, resps in archetype_groups.items():
            n = len(resps)
            by_archetype[arch] = ArchetypePerformance(
                archetype=arch,
                count=n,
                accuracy=sum(1 for r in resps if r.is_correct) / n,
                avg_time_seconds=sum(r.time_seconds for r in resps) / n,
                avg_confidence=sum(r.confidence for r in resps) / n,
                avg_engagement=sum(r.engagement_level for r in resps) / n,
            )

        # Distractor analysis
        options = content.get("options", [])
        correct_answer = content.get("correct_answer", "A")

        distractor_analysis: Dict[str, DistractorAnalysis] = {}
        answer_counts: Dict[str, List[str]] = {}  # answer -> list of archetypes

        for r in responses:
            if r.selected_answer not in answer_counts:
                answer_counts[r.selected_answer] = []
            answer_counts[r.selected_answer].append(r.archetype)

        for i, opt in enumerate(options):
            letter = chr(65 + i)
            is_correct_option = self._check_correct(letter, correct_answer, options)

            if is_correct_option:
                continue  # Skip correct answer in distractor analysis

            selections = answer_counts.get(letter, [])
            selection_rate = len(selections) / total

            # Find which archetypes selected this most
            archetype_counts = {}
            for arch in selections:
                archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

            attracted = [
                arch for arch, count in archetype_counts.items()
                if count >= 2 or (count >= 1 and len(archetype_groups.get(arch, [])) <= 3)
            ]

            # Determine if distractor is effective
            is_effective = 0.05 <= selection_rate <= 0.35
            concern = None

            if selection_rate < 0.02:
                concern = "Rarely selected - may be obviously wrong"
            elif selection_rate > 0.40:
                concern = "Selected too often - may be confusing or correct answer is unclear"

            distractor_analysis[letter] = DistractorAnalysis(
                option=letter,
                selection_rate=selection_rate,
                attracted_archetypes=attracted,
                is_effective=is_effective,
                concern=concern,
            )

        # Generate concerns
        concerns = []
        recommendations = []

        # Check for high difficulty variance (accessibility issue)
        if accuracy_variance > 0.04:
            concerns.append(
                f"High accuracy variance across archetypes (σ²={accuracy_variance:.3f}) — "
                "may indicate accessibility issues for some student types"
            )

        # Check for very low accuracy archetypes
        for arch, perf in by_archetype.items():
            if perf.accuracy < 0.3 and perf.count >= 3:
                concerns.append(
                    f"'{arch.replace('_', ' ')}' students have very low accuracy ({perf.accuracy:.0%}) — "
                    "consider reviewing content accessibility"
                )

        # Check distractor issues
        for letter, da in distractor_analysis.items():
            if da.concern:
                concerns.append(f"Distractor {letter}: {da.concern}")

        # Overall difficulty concerns
        if accuracy < 0.3:
            concerns.append("Overall accuracy is very low (<30%) — question may be too difficult")
            recommendations.append("Consider simplifying the question or providing scaffolding")
        elif accuracy > 0.9:
            concerns.append("Overall accuracy is very high (>90%) — question may be too easy")
            recommendations.append("Consider increasing complexity for better discrimination")

        return SimulationResult(
            total_students=total,
            accuracy=accuracy,
            difficulty_irt=difficulty_irt,
            discrimination_irt=discrimination_irt,
            avg_time_seconds=avg_time,
            engagement_score=avg_engagement,
            by_archetype=by_archetype,
            distractor_analysis=distractor_analysis,
            concerns=concerns,
            recommendations=recommendations,
        )


# ── CLI Entry Point ─────────────────────────────────────────────────────────

async def main():
    """Test the simulator with a sample question."""
    simulator = InceptBenchSimulator()

    sample_content = {
        "type": "question",
        "text": "What is the primary function of the Electoral College in the United States?",
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

    def progress(cur, total, msg):
        print(f"  [{cur}/{total}] {msg}")

    result = await simulator.simulate_content(
        content=sample_content,
        population_config={"size": 20},
        progress_callback=progress,
    )

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
