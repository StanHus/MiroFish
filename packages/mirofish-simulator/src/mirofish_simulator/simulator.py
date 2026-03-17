"""
Core simulator for student response simulation.

Uses cognitive models (retention + perception) to simulate how diverse student
archetypes would respond to educational content. Students answer from genuinely
limited knowledge rather than "knowing the answer and rolling dice."
"""

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from .profiles import (
    StudentProfile,
    ARCHETYPE_TRAITS,
    generate_population,
)
from .cognition import CognitiveModel, CognitiveLens


# ── Constants ───────────────────────────────────────────────────────────────

ARCHETYPES = list(ARCHETYPE_TRAITS.keys())

DEFAULT_POPULATION = {
    "size": 30,
    "archetypes": ARCHETYPES,
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


# ── Result Models ───────────────────────────────────────────────────────────

@dataclass
class StudentResponse:
    """A single student's response to content."""

    student_id: int
    archetype: str
    selected_answer: str
    is_correct: bool
    confidence: float
    # Cognitive model outputs (for debugging/analysis)
    cognitive_factors: Optional[Dict[str, Any]] = None


@dataclass
class DistractorAnalysis:
    """Analysis of a distractor option."""

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
    avg_confidence: float


@dataclass
class SimulationResult:
    """Complete simulation result.

    Note: All estimates are UNCALIBRATED unless calibration data has been provided.
    The `confidence` field indicates whether predictions have been validated against
    real student response data.
    """

    total_students: int
    accuracy: float
    estimated_challenge: float  # 0-1 scale, higher = harder (NOT IRT-calibrated)
    archetype_variance: float   # Variance in performance across archetypes (NOT IRT discrimination)
    engagement_score: float

    by_archetype: Dict[str, ArchetypePerformance]
    distractor_analysis: Dict[str, DistractorAnalysis]

    confidence: str = "uncalibrated"  # "uncalibrated" | "calibrated"
    concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    responses: List[StudentResponse] = field(default_factory=list)
    simulation_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate": {
                "total_students": self.total_students,
                "accuracy": round(self.accuracy, 3),
                "estimated_challenge": round(self.estimated_challenge, 3),
                "archetype_variance": round(self.archetype_variance, 3),
                "engagement_score": round(self.engagement_score, 3),
            },
            "confidence": self.confidence,
            "by_archetype": {
                k: {
                    "count": v.count,
                    "accuracy": round(v.accuracy, 3),
                    "avg_confidence": round(v.avg_confidence, 3),
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


# ── Main Simulator ──────────────────────────────────────────────────────────

class Simulator:
    """
    Simulates student populations responding to educational content.

    Uses cognitive models (retention + perception) to create realistic student
    responses. Instead of "knowing the answer and rolling dice," students answer
    from genuinely limited knowledge based on their grade level and archetype.

    Usage:
        sim = Simulator(api_key="sk-...")
        result = await sim.simulate(content, population_config)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_cognitive_model: bool = True,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.use_cognitive_model = use_cognitive_model

        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = None

        # Initialize cognitive model for realistic simulation
        self.cognitive_model = CognitiveModel(
            api_key=self.api_key,
            base_url=self.base_url,
            model=model,
        )

    async def simulate(
        self,
        content: Dict[str, Any],
        population_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationResult:
        """
        Simulate student population responding to content.

        Args:
            content: Educational content with keys:
                - text/question: The question text
                - options: List of answer options
                - correct_answer: Correct answer (letter or index)
                - grade: Target grade level
                - subject: Subject area
            population_config: Population parameters
            progress_callback: (current, total, message) callback

        Returns:
            SimulationResult with metrics and analysis
        """
        start_time = time.time()

        # Merge with defaults
        config = {**DEFAULT_POPULATION, **(population_config or {})}

        # Generate student population
        profiles = generate_population(
            size=config.get("size", 30),
            archetypes=config.get("archetypes"),
            distribution=config.get("distribution"),
        )

        total = len(profiles)
        if progress_callback:
            progress_callback(0, total, f"Generated {total} students")

        # Simulate responses
        responses: List[StudentResponse] = []

        # Process in batches
        batch_size = 5
        for i in range(0, total, batch_size):
            batch = profiles[i : i + batch_size]
            batch_responses = await asyncio.gather(*[
                self._simulate_one(profile, content)
                for profile in batch
            ])
            responses.extend(batch_responses)

            if progress_callback:
                done = min(i + batch_size, total)
                progress_callback(done, total, f"Simulated {done}/{total}")

        # Compute metrics
        result = self._compute_metrics(responses, content)
        result.simulation_time_ms = int((time.time() - start_time) * 1000)
        result.responses = responses

        return result

    async def _simulate_one(
        self,
        profile: StudentProfile,
        content: Dict[str, Any],
    ) -> StudentResponse:
        """
        Simulate a single student's response using cognitive modeling.

        Cognitive approach (realistic):
        1. Create cognitive lens: what does this student know/perceive?
        2. Have LLM answer FROM THE STUDENT'S LIMITED PERSPECTIVE
        3. Student answers based on their actual knowledge gaps, not dice rolls

        This produces genuinely different answers based on what students
        at different levels would actually know and understand.
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        correct = content.get("correct_answer", "A")
        grade_level = int(content.get("grade", 11))

        # Get correct letter
        if isinstance(correct, str) and correct.isalpha():
            correct_letter = correct.upper()
        else:
            try:
                correct_letter = chr(65 + int(correct))
            except (ValueError, TypeError):
                correct_letter = "A"

        # If no LLM client, use probabilistic fallback
        if not self.client:
            return self._fallback_response(profile, options, correct)

        # Use cognitive model for realistic simulation
        if self.use_cognitive_model:
            return await self._simulate_with_cognition(
                profile, content, correct_letter, grade_level
            )
        else:
            # Legacy approach: dice roll with distractor analysis
            return await self._simulate_legacy(
                profile, content, correct_letter
            )

    async def _simulate_with_cognition(
        self,
        profile: StudentProfile,
        content: Dict[str, Any],
        correct_letter: str,
        grade_level: int,
    ) -> StudentResponse:
        """
        Simulate using cognitive model - student answers from limited knowledge.
        """
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])

        # Create cognitive lens for this student
        lens = await self.cognitive_model.create_lens(
            content=content,
            grade_level=grade_level,
            archetype=profile.archetype,
        )

        # Now have the LLM answer AS this student with their limited knowledge
        selected, confidence = await self._answer_with_cognitive_lens(
            text, options, correct_letter, profile, lens
        )

        # Check if correct
        is_correct = selected == correct_letter

        return StudentResponse(
            student_id=profile.student_id,
            archetype=profile.archetype,
            selected_answer=selected,
            is_correct=is_correct,
            confidence=confidence,
            cognitive_factors={
                "known_concepts": lens.retention.known_concepts[:3],
                "unknown_concepts": lens.retention.unknown_concepts[:3],
                "perceived_difficulty": lens.perception.perceived_difficulty,
                "attention_level": lens.perception.attention_level,
                "likely_errors": lens.likely_errors[:2],
            },
        )

    async def _answer_with_cognitive_lens(
        self,
        text: str,
        options: List[str],
        correct_letter: str,
        profile: StudentProfile,
        lens: CognitiveLens,
    ) -> tuple:
        """
        Have LLM answer from the student's limited perspective.

        This is the key innovation: instead of asking the LLM "what's the right answer?"
        we ask "what would THIS STUDENT with THESE KNOWLEDGE GAPS answer?"
        """
        options_text = "\n".join(
            f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)
        )

        # Build the cognitive context
        known = ", ".join(lens.retention.known_concepts[:5]) or "basic familiarity"
        unknown = ", ".join(lens.retention.unknown_concepts[:5]) or "advanced details"
        perceived = lens.perception.perceived_text

        prompt = f"""You are simulating a grade {lens.retention.grade_level} student with the "{profile.archetype}" profile answering a question.

IMPORTANT: You are NOT answering correctly. You are answering AS THIS STUDENT would, given their LIMITED knowledge and understanding.

WHAT THIS STUDENT KNOWS:
{known}

WHAT THIS STUDENT DOES NOT KNOW:
{unknown}

HOW THIS STUDENT PERCEIVES THE QUESTION:
"{perceived}"

PERCEIVED DIFFICULTY: {lens.perception.perceived_difficulty}
ATTENTION LEVEL: {lens.perception.attention_level:.0%}

ORIGINAL QUESTION:
{text}

OPTIONS:
{options_text}

Answer AS this student would - based on what they know (and don't know).
- If they lack knowledge, they may guess or pick something that "sounds right"
- If they misunderstand terms, they may pick a wrong answer confidently
- If attention is low, they may not read all options carefully

Respond with JSON only:
{{
    "selected": "A",
    "confidence": 0.6,
    "student_reasoning": "Brief explanation of why THIS STUDENT picked this"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are simulating student test-taking behavior. "
                            "Answer as the specified student would - with their knowledge gaps, "
                            "misunderstandings, and attention patterns. "
                            "Do NOT answer correctly unless the student would genuinely know the answer. "
                            "Respond ONLY with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,  # Higher temp for more realistic variation
                max_tokens=150,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)
            selected = analysis.get("selected", "A").upper()
            confidence = analysis.get("confidence", 0.5)

            # Validate selected is a valid option
            valid_options = [chr(65 + i) for i in range(len(options))]
            if selected not in valid_options:
                selected = random.choice(valid_options)

            return selected, confidence

        except Exception:
            # Fallback: use cognitive lens prediction
            if lens.can_answer_correctly:
                return correct_letter, lens.confidence_if_answers
            else:
                # Pick a random wrong answer
                valid_options = [chr(65 + i) for i in range(len(options))]
                wrong_options = [o for o in valid_options if o != correct_letter]
                selected = random.choice(wrong_options) if wrong_options else "A"
                return selected, 0.4 + random.uniform(0, 0.3)

    async def _simulate_legacy(
        self,
        profile: StudentProfile,
        content: Dict[str, Any],
        correct_letter: str,
    ) -> StudentResponse:
        """
        Legacy simulation approach: dice roll with distractor analysis.
        Kept for comparison and fallback.
        """
        options = content.get("options", [])

        # Get distractor analysis from LLM (cached per question)
        distractor_appeal = await self._analyze_distractors(content)

        # Determine if student answers correctly based on archetype
        traits = ARCHETYPE_TRAITS.get(profile.archetype, {})
        base_accuracy = traits.get("base_accuracy", 0.5)

        # Roll dice: does this student get it correct?
        gets_correct = random.random() < base_accuracy

        if gets_correct:
            selected = correct_letter
            confidence = 0.7 + random.uniform(0, 0.25)
        else:
            # Pick most tempting wrong answer for this archetype
            selected = self._pick_tempting_distractor(
                profile.archetype,
                correct_letter,
                distractor_appeal,
                options
            )
            confidence = 0.4 + random.uniform(0, 0.3)

        return StudentResponse(
            student_id=profile.student_id,
            archetype=profile.archetype,
            selected_answer=selected,
            is_correct=gets_correct,
            confidence=confidence,
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

        archetypes_list = ", ".join(ARCHETYPE_TRAITS.keys())

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
                model=self.model,
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

        except Exception:
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

    def _fallback_response(
        self,
        profile: StudentProfile,
        options: List[str],
        correct: str,
    ) -> StudentResponse:
        """Generate probabilistic response without LLM."""
        traits = ARCHETYPE_TRAITS.get(profile.archetype, {})
        prob_correct = traits.get("base_accuracy", 0.5)

        is_correct = random.random() < prob_correct

        if is_correct:
            selected = correct.upper() if correct.isalpha() else chr(65 + int(correct))
        else:
            all_opts = ["A", "B", "C", "D"][: len(options)]
            correct_letter = correct.upper() if correct.isalpha() else chr(65 + int(correct))
            wrong = [o for o in all_opts if o != correct_letter]
            selected = random.choice(wrong) if wrong else "A"

        return StudentResponse(
            student_id=profile.student_id,
            archetype=profile.archetype,
            selected_answer=selected,
            is_correct=is_correct,
            confidence=0.5 + (0.3 if is_correct else -0.2),
        )

    def _check_correct(self, selected: str, correct: str, options: List[str]) -> bool:
        """Check if selected answer is correct."""
        if correct.upper() == selected.upper():
            return True
        try:
            return selected == chr(65 + int(correct))
        except (ValueError, TypeError):
            pass
        if correct in options:
            return selected == chr(65 + options.index(correct))
        return False

    def _compute_metrics(
        self,
        responses: List[StudentResponse],
        content: Dict[str, Any],
    ) -> SimulationResult:
        """Compute aggregate metrics from responses."""
        if not responses:
            return SimulationResult(
                total_students=0,
                accuracy=0,
                estimated_challenge=0.5,
                archetype_variance=0,
                engagement_score=0,
                by_archetype={},
                distractor_analysis={},
                confidence="uncalibrated",
            )

        total = len(responses)
        correct_count = sum(1 for r in responses if r.is_correct)
        accuracy = correct_count / total

        # Estimated challenge: 0-1 scale, higher = harder
        # Simple inverse of accuracy (NOT a calibrated IRT parameter)
        estimated_challenge = 1.0 - accuracy

        # Archetype variance: variance in accuracy across student types
        # Higher values indicate the question differentiates between archetypes
        arch_groups: Dict[str, List[StudentResponse]] = {}
        for r in responses:
            if r.archetype not in arch_groups:
                arch_groups[r.archetype] = []
            arch_groups[r.archetype].append(r)

        archetype_variance = 0.0
        if len(arch_groups) > 1:
            means = [
                sum(1 for r in g if r.is_correct) / len(g)
                for g in arch_groups.values()
            ]
            mean_of_means = sum(means) / len(means)
            archetype_variance = sum((m - mean_of_means) ** 2 for m in means) / len(means)

        # Archetype breakdown
        by_archetype: Dict[str, ArchetypePerformance] = {}
        for arch, group in arch_groups.items():
            n = len(group)
            by_archetype[arch] = ArchetypePerformance(
                archetype=arch,
                count=n,
                accuracy=sum(1 for r in group if r.is_correct) / n,
                avg_confidence=sum(r.confidence for r in group) / n,
            )

        # Distractor analysis
        options = content.get("options", [])
        correct = content.get("correct_answer", "A")
        distractor_analysis: Dict[str, DistractorAnalysis] = {}

        answer_counts: Dict[str, List[str]] = {}
        for r in responses:
            if r.selected_answer not in answer_counts:
                answer_counts[r.selected_answer] = []
            answer_counts[r.selected_answer].append(r.archetype)

        for i, _ in enumerate(options):
            letter = chr(65 + i)
            if self._check_correct(letter, correct, options):
                continue

            selections = answer_counts.get(letter, [])
            rate = len(selections) / total

            arch_counts = {}
            for a in selections:
                arch_counts[a] = arch_counts.get(a, 0) + 1
            attracted = [a for a, c in arch_counts.items() if c >= 2]

            concern = None
            if rate < 0.02:
                concern = "Rarely selected"
            elif rate > 0.40:
                concern = "Selected too often"

            distractor_analysis[letter] = DistractorAnalysis(
                option=letter,
                selection_rate=rate,
                attracted_archetypes=attracted,
                is_effective=0.05 <= rate <= 0.35,
                concern=concern,
            )

        # Generate concerns
        concerns = []
        recommendations = []

        if archetype_variance > 0.04:
            concerns.append("High accuracy variance across archetypes")

        for arch, perf in by_archetype.items():
            if perf.accuracy < 0.3 and perf.count >= 3:
                concerns.append(f"'{arch}' has very low accuracy ({perf.accuracy:.0%})")

        if accuracy < 0.3:
            concerns.append("Overall accuracy very low (<30%)")
            recommendations.append("Consider simplifying the question")
        elif accuracy > 0.9:
            concerns.append("Overall accuracy very high (>90%)")
            recommendations.append("Consider increasing complexity")

        avg_engagement = sum(
            ARCHETYPE_TRAITS.get(r.archetype, {}).get("engagement", 0.5)
            for r in responses
        ) / total

        return SimulationResult(
            total_students=total,
            accuracy=accuracy,
            estimated_challenge=estimated_challenge,
            archetype_variance=archetype_variance,
            engagement_score=avg_engagement,
            by_archetype=by_archetype,
            distractor_analysis=distractor_analysis,
            confidence="uncalibrated",
            concerns=concerns,
            recommendations=recommendations,
        )


# ── Convenience Function ────────────────────────────────────────────────────

async def simulate_content(
    content: Dict[str, Any],
    population_size: int = 30,
    api_key: Optional[str] = None,
) -> SimulationResult:
    """
    Quick simulation of student responses to content.

    Args:
        content: Educational content dict
        population_size: Number of students to simulate
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        SimulationResult with metrics
    """
    sim = Simulator(api_key=api_key)
    return await sim.simulate(content, {"size": population_size})
