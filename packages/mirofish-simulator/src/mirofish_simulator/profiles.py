"""
Student profile generation for simulation.

Defines student archetypes and generates diverse student populations.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ── MBTI pools by archetype ─────────────────────────────────────────────────

ARCHETYPE_MBTI = {
    "honors_overachiever": ["INTJ", "ISTJ", "ENTJ", "ESTJ"],
    "debate_club_kid": ["ENTP", "ENTJ", "ENFP", "ESTP"],
    "quiet_thinker": ["INFJ", "INFP", "INTP", "INTJ"],
    "socially_engaged_activist": ["ENFJ", "ENFP", "ESFJ", "INFJ"],
    "disengaged_but_smart": ["ISTP", "INTP", "ISFP", "ESTP"],
    "esl_student": ["ISFJ", "INFP", "ISFP", "ISTJ"],
    "class_clown": ["ESFP", "ESTP", "ENFP", "ENTP"],
    "politically_conservative": ["ISTJ", "ESTJ", "INTJ", "ENTJ"],
}

# ── Archetype traits ────────────────────────────────────────────────────────

ARCHETYPE_TRAITS = {
    "honors_overachiever": {
        "base_accuracy": 0.92,
        "engagement": 0.95,
        "knowledge": "advanced",
        "gpa": "3.8-4.0",
        "description": "High-achieving student focused on academic excellence",
    },
    "debate_club_kid": {
        "base_accuracy": 0.78,
        "engagement": 0.85,
        "knowledge": "advanced",
        "gpa": "3.5-3.9",
        "description": "Intellectually curious, loves to argue and discuss",
    },
    "quiet_thinker": {
        "base_accuracy": 0.70,
        "engagement": 0.50,
        "knowledge": "intermediate",
        "gpa": "3.2-3.6",
        "description": "Thoughtful introvert who processes deeply",
    },
    "socially_engaged_activist": {
        "base_accuracy": 0.68,
        "engagement": 0.80,
        "knowledge": "intermediate",
        "gpa": "3.3-3.7",
        "description": "Passionate about social issues and justice",
    },
    "disengaged_but_smart": {
        "base_accuracy": 0.55,
        "engagement": 0.30,
        "knowledge": "advanced",
        "gpa": "2.8-3.4",
        "description": "Capable but unmotivated student",
    },
    "esl_student": {
        "base_accuracy": 0.45,
        "engagement": 0.60,
        "knowledge": "basic",
        "gpa": "2.5-3.2",
        "description": "English language learner with strong effort",
    },
    "class_clown": {
        "base_accuracy": 0.35,
        "engagement": 0.40,
        "knowledge": "basic",
        "gpa": "2.3-2.9",
        "description": "Prioritizes humor over academics",
    },
    "politically_conservative": {
        "base_accuracy": 0.65,
        "engagement": 0.70,
        "knowledge": "intermediate",
        "gpa": "3.0-3.5",
        "description": "Student with traditional/conservative viewpoints",
    },
}


@dataclass
class StudentProfile:
    """A simulated student profile."""

    student_id: int
    name: str
    archetype: str

    age: int = 17
    grade: int = 11
    gender: str = "unspecified"
    mbti: str = "INTP"

    gpa_range: str = "3.0-3.5"
    engagement_level: float = 0.5
    knowledge_depth: str = "intermediate"

    persona: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def create(cls, student_id: int, archetype: str) -> "StudentProfile":
        """Create a student profile from an archetype."""
        traits = ARCHETYPE_TRAITS.get(archetype, ARCHETYPE_TRAITS["quiet_thinker"])

        gender = random.choice(["male", "female"])
        age = random.choice([16, 17, 17, 17, 18])
        grade = 11 if age <= 17 else 12
        mbti = random.choice(ARCHETYPE_MBTI.get(archetype, ["INTP"]))

        return cls(
            student_id=student_id,
            name=f"Student_{student_id}",
            archetype=archetype,
            age=age,
            grade=grade,
            gender=gender,
            mbti=mbti,
            gpa_range=traits["gpa"],
            engagement_level=traits["engagement"],
            knowledge_depth=traits["knowledge"],
            persona=traits["description"],
        )

    def to_dict(self) -> Dict:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "archetype": self.archetype,
            "age": self.age,
            "grade": self.grade,
            "gender": self.gender,
            "mbti": self.mbti,
            "gpa_range": self.gpa_range,
            "engagement_level": self.engagement_level,
            "knowledge_depth": self.knowledge_depth,
            "persona": self.persona,
        }


def generate_population(
    size: int = 30,
    archetypes: Optional[List[str]] = None,
    distribution: Optional[Dict[str, float]] = None,
) -> List[StudentProfile]:
    """
    Generate a diverse student population.

    Args:
        size: Number of students to generate
        archetypes: List of archetypes to include (default: all)
        distribution: Weight per archetype (default: even)

    Returns:
        List of StudentProfile objects
    """
    if archetypes is None:
        archetypes = list(ARCHETYPE_TRAITS.keys())

    if distribution is None:
        distribution = {a: 1.0 / len(archetypes) for a in archetypes}

    # Normalize distribution
    total = sum(distribution.values())
    distribution = {k: v / total for k, v in distribution.items()}

    profiles = []
    student_id = 0

    # Calculate counts per archetype
    remaining = size
    archetype_counts = {}

    for arch in archetypes[:-1]:
        weight = distribution.get(arch, 0)
        count = int(size * weight)
        archetype_counts[arch] = count
        remaining -= count

    # Last archetype gets remainder
    archetype_counts[archetypes[-1]] = remaining

    # Generate profiles
    for archetype, count in archetype_counts.items():
        for _ in range(count):
            profile = StudentProfile.create(student_id, archetype)
            profiles.append(profile)
            student_id += 1

    random.shuffle(profiles)
    return profiles
