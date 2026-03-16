"""
Student Profile Generator for AP Social Studies Simulation
Bypasses Zep — generates OASIS-compatible agent profiles directly from
a JSON config of student archetypes.

Usage:
    from app.services.student_profile_generator import StudentProfileGenerator

    generator = StudentProfileGenerator(api_key="...", model_name="gpt-4o-mini")
    profiles = generator.generate_profiles("configs/ap_social_studies_sample.json")
    generator.save_profiles(profiles, "reddit_profiles.json")
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.student_profile")

# ── MBTI shortcuts by archetype ─────────────────────────────────────────────
ARCHETYPE_MBTI_POOLS = {
    "honors_overachiever": ["INTJ", "ISTJ", "ENTJ", "ESTJ"],
    "debate_club_kid": ["ENTP", "ENTJ", "ENFP", "ESTP"],
    "quiet_thinker": ["INFJ", "INFP", "INTP", "INTJ"],
    "socially_engaged_activist": ["ENFJ", "ENFP", "ESFJ", "INFJ"],
    "disengaged_but_smart": ["ISTP", "INTP", "ISFP", "ESTP"],
    "esl_student": ["ISFJ", "INFP", "ISFP", "ISTJ"],
    "class_clown": ["ESFP", "ESTP", "ENFP", "ENTP"],
    "politically_conservative": ["ISTJ", "ESTJ", "INTJ", "ENTJ"],
}

# ── First-name pools by demographic ─────────────────────────────────────────
NAME_POOLS: Dict[str, List[str]] = {
    "suburban_white_male": ["Tyler", "Connor", "Jake", "Ryan", "Luke", "Brett", "Kyle", "Nate"],
    "suburban_white_female": ["Madison", "Hannah", "Chloe", "Emily", "Abigail", "Sarah", "Claire", "Megan"],
    "african_american_male": ["Jaylen", "DeShawn", "Malik", "Terrence", "Marcus", "Darius", "Andre", "Kofi"],
    "african_american_female": ["Aaliyah", "Zaria", "Imani", "Jasmine", "Nia", "Destiny", "Brianna", "Kiara"],
    "east_asian_male": ["Kevin", "David", "Eric", "Andrew", "Daniel", "Jason", "Brian", "Tony"],
    "east_asian_female": ["Grace", "Michelle", "Jenny", "Amy", "Lisa", "Christine", "Vivian", "Angela"],
    "south_asian_male": ["Arjun", "Rohan", "Nikhil", "Vikram", "Aditya", "Kiran", "Dev", "Ravi"],
    "south_asian_female": ["Priya", "Ananya", "Meera", "Neha", "Sanya", "Aisha", "Divya", "Kavya"],
    "latino_male": ["Carlos", "Diego", "Miguel", "Alejandro", "Luis", "Mateo", "Rafael", "Jose"],
    "latino_female": ["Sofia", "Isabella", "Valentina", "Camila", "Maria", "Lucia", "Gabriela", "Ana"],
    "default_male": ["Jordan", "Alex", "Sam", "Casey", "Taylor", "Morgan", "Jamie", "Riley"],
    "default_female": ["Jordan", "Alex", "Sam", "Casey", "Taylor", "Morgan", "Jamie", "Riley"],
    "nonbinary": ["River", "Quinn", "Avery", "Sage", "Rowan", "Skyler", "Ash", "Finley"],
}

LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Garcia", "Martinez", "Lee", "Kim",
    "Patel", "Chen", "Davis", "Rodriguez", "Nguyen", "Smith", "Jones",
    "Wilson", "Lopez", "Thomas", "Jackson", "White", "Harris", "Clark",
    "Walker", "Young", "Allen", "King", "Scott", "Green", "Adams",
    "Morales", "Reyes", "Cruz", "Gomez", "Herrera", "Vargas", "Singh",
    "Shah", "Gupta", "Rao", "Liu", "Wang", "Zhang", "Wu", "Park",
    "Choi", "Tanaka", "Sato", "Okafor", "Mensah", "Abadi", "Hussein",
]


@dataclass
class StudentProfile:
    """OASIS-compatible student agent profile."""

    user_id: int
    username: str
    name: str
    bio: str
    persona: str
    archetype_id: str

    # OASIS required
    age: int = 17
    gender: str = "male"
    mbti: str = "INTP"
    country: str = "United States"
    karma: int = 100

    # Student-specific (carried in persona text, also stored for tooling)
    grade: int = 11
    gpa_range: str = "3.0-3.5"
    engagement_level: float = 0.5
    learning_style: str = "mixed"
    background: str = ""
    interested_topics: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to OASIS Reddit JSON format (matches generate_reddit_agent_graph)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": f"High School Student (Grade {self.grade})",
            "interested_topics": self.interested_topics,
        }


class StudentProfileGenerator:
    """
    Generate student agent profiles from a JSON archetype config.
    No Zep dependency — works entirely from the config file + optional LLM enrichment.
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

        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self._used_names: set = set()

    # ── public API ───────────────────────────────────────────────────────────

    def generate_profiles(
        self,
        config_path: str,
        use_llm: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[StudentProfile]:
        """
        Main entry point. Reads the JSON config, expands archetypes into
        individual student profiles, optionally enriches with LLM.

        Args:
            config_path: path to the AP Social Studies JSON config
            use_llm:     if True and an LLM client is available, generate
                         richer persona text via the model
            progress_callback: (current, total, message)

        Returns:
            list of StudentProfile ready for OASIS
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        archetypes = config.get("student_archetypes", [])
        classroom = config.get("classroom_context", {})
        meta = config.get("simulation_meta", {})
        discussion = config.get("discussion_config", {})

        # Build the full context string for LLM persona generation
        context = self._build_context(meta, classroom, discussion)

        profiles: List[StudentProfile] = []
        user_id = 0

        total_students = sum(a.get("count", 1) for a in archetypes)

        for archetype in archetypes:
            count = archetype.get("count", 1)
            arch_id = archetype.get("archetype_id", "unknown")

            for i in range(count):
                if progress_callback:
                    progress_callback(
                        user_id + 1,
                        total_students,
                        f"Generating student {user_id + 1}/{total_students} ({arch_id})",
                    )

                profile = self._generate_one_student(
                    user_id=user_id,
                    archetype=archetype,
                    context=context,
                    seed_prompt=classroom.get("seed_prompt", ""),
                    use_llm=use_llm,
                )
                profiles.append(profile)
                user_id += 1

        logger.info(f"Generated {len(profiles)} student profiles from {len(archetypes)} archetypes")
        return profiles

    def save_profiles(self, profiles: List[StudentProfile], file_path: str):
        """Save profiles as OASIS Reddit-compatible JSON."""
        data = [p.to_reddit_format() for p in profiles]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(profiles)} profiles to {file_path}")

    # ── private helpers ──────────────────────────────────────────────────────

    def _generate_one_student(
        self,
        user_id: int,
        archetype: Dict[str, Any],
        context: str,
        seed_prompt: str,
        use_llm: bool,
    ) -> StudentProfile:
        """Create a single student profile from an archetype definition."""

        arch_id = archetype.get("archetype_id", "unknown")
        traits = archetype.get("traits", {})
        behaviors = archetype.get("behavior_patterns", [])
        demo_dist = archetype.get("demographic_distribution", {})

        # Roll demographics
        gender = self._pick_gender(demo_dist)
        background = self._pick_background(demo_dist)
        name = self._pick_name(gender, background)
        age = random.choice([16, 17, 17, 17, 18, 18])
        grade = 11 if age <= 17 else 12
        mbti = random.choice(ARCHETYPE_MBTI_POOLS.get(arch_id, ["INTP"]))

        gpa = self._roll_gpa(traits)
        engagement = traits.get("engagement", 0.5)
        learning_style = traits.get("learning_style", "mixed")
        knowledge = traits.get("knowledge_depth", "intermediate")

        # Build bio (short)
        bio = self._build_bio(name, arch_id, grade, gpa)

        # Build persona (long)
        if use_llm and self.client:
            persona = self._generate_persona_llm(
                name=name,
                age=age,
                gender=gender,
                grade=grade,
                gpa=gpa,
                mbti=mbti,
                background=background,
                arch_id=arch_id,
                archetype_desc=archetype.get("description", ""),
                traits=traits,
                behaviors=behaviors,
                context=context,
                seed_prompt=seed_prompt,
            )
        else:
            persona = self._generate_persona_rule(
                name=name,
                age=age,
                gender=gender,
                grade=grade,
                gpa=gpa,
                mbti=mbti,
                background=background,
                arch_id=arch_id,
                archetype_desc=archetype.get("description", ""),
                behaviors=behaviors,
            )

        # Interested topics based on archetype
        topics = self._pick_topics(arch_id)

        username = self._make_username(name)

        return StudentProfile(
            user_id=user_id,
            username=username,
            name=name,
            bio=bio,
            persona=persona,
            archetype_id=arch_id,
            age=age,
            gender=gender,
            mbti=mbti,
            country="United States",
            karma=random.randint(50, 500),
            grade=grade,
            gpa_range=gpa,
            engagement_level=engagement,
            learning_style=learning_style,
            background=background,
            interested_topics=topics,
        )

    # ── demographics ─────────────────────────────────────────────────────────

    def _pick_gender(self, demo: Dict) -> str:
        genders = demo.get("gender", ["male", "female"])
        return random.choice(genders)

    def _pick_background(self, demo: Dict) -> str:
        bgs = demo.get("background", ["any"])
        bg = random.choice(bgs)
        return bg if bg != "any" else random.choice(
            ["suburban_white", "african_american", "latino", "east_asian", "south_asian", "mixed"]
        )

    def _pick_name(self, gender: str, background: str) -> str:
        """Pick a culturally plausible first + last name, avoid duplicates."""
        # Build the name pool key
        # Simplify background to match pool keys
        bg_key = background.split("_")[0] if "_" in background else background
        pool_key_map = {
            "suburban": "suburban_white",
            "african": "african_american",
            "east": "east_asian",
            "south": "south_asian",
            "latino": "latino",
            "recent": "latino",  # recent_immigrant_latino etc.
            "refugee": "latino",
            "rural": "suburban_white",
            "religious": "suburban_white",
            "libertarian": "suburban_white",
            "first": "default",
            "introvert": "default",
            "lgbtq": "default",
            "immigrant": "latino",
            "mixed": "default",
        }

        bg_base = pool_key_map.get(bg_key, "default")

        if gender == "nonbinary":
            first_pool = NAME_POOLS.get("nonbinary", NAME_POOLS["default_male"])
        else:
            g_suffix = "_male" if gender == "male" else "_female"
            first_pool = NAME_POOLS.get(f"{bg_base}{g_suffix}", NAME_POOLS[f"default{g_suffix}"])

        # Try up to 20 times for a unique name
        for _ in range(20):
            first = random.choice(first_pool)
            last = random.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full not in self._used_names:
                self._used_names.add(full)
                return full

        # Fallback: append number
        full = f"{random.choice(first_pool)} {random.choice(LAST_NAMES)}"
        full = f"{full} {random.randint(1, 99)}"
        self._used_names.add(full)
        return full

    def _make_username(self, name: str) -> str:
        parts = name.lower().split()
        base = f"{parts[0]}_{parts[-1]}" if len(parts) > 1 else parts[0]
        base = "".join(c for c in base if c.isalnum() or c == "_")
        return f"{base}_{random.randint(10, 99)}"

    def _roll_gpa(self, traits: Dict) -> str:
        depth = traits.get("knowledge_depth", "intermediate")
        engagement = traits.get("engagement", 0.5)
        if depth == "advanced" and engagement > 0.8:
            return random.choice(["3.8-4.0", "3.9-4.0", "3.7-4.0"])
        elif depth == "advanced" or engagement > 0.7:
            return random.choice(["3.5-3.8", "3.6-3.9", "3.4-3.7"])
        elif depth == "intermediate":
            return random.choice(["3.0-3.5", "2.8-3.3", "3.1-3.6"])
        else:
            return random.choice(["2.5-3.0", "2.3-2.8", "2.7-3.2"])

    # ── bio / persona builders ───────────────────────────────────────────────

    def _build_bio(self, name: str, arch_id: str, grade: int, gpa: str) -> str:
        label_map = {
            "honors_overachiever": "AP honors student",
            "debate_club_kid": "debate team member",
            "quiet_thinker": "thoughtful observer",
            "socially_engaged_activist": "social justice advocate",
            "disengaged_but_smart": "student",
            "esl_student": "multilingual learner",
            "class_clown": "class personality",
            "politically_conservative": "student",
        }
        label = label_map.get(arch_id, "student")
        return f"{name} — Grade {grade} {label}. GPA: {gpa}."

    def _build_context(self, meta: Dict, classroom: Dict, discussion: Dict) -> str:
        parts = [
            f"Subject: {meta.get('subject', 'AP Social Studies')}",
            f"Unit: {meta.get('unit', '')}",
            f"School: {classroom.get('school_type', 'public suburban')}",
            f"Class period: {classroom.get('period', '')}",
            f"Teacher: {classroom.get('teacher_name', 'the teacher')}",
            f"Format: {classroom.get('discussion_format', 'class discussion')}",
        ]
        if discussion.get("available_actions"):
            parts.append(f"Possible actions: {', '.join(discussion['available_actions'])}")
        return "\n".join(parts)

    def _generate_persona_rule(
        self, name, age, gender, grade, gpa, mbti, background, arch_id, archetype_desc, behaviors
    ) -> str:
        """Rule-based persona (no LLM needed)."""
        behavior_text = "; ".join(behaviors[:4]) if behaviors else "participates normally"
        pronoun = "they" if gender == "nonbinary" else ("he" if gender == "male" else "she")

        return (
            f"{name} is a {age}-year-old {gender} student in Grade {grade} at a suburban American "
            f"high school, taking AP U.S. Government and Politics. "
            f"{archetype_desc} "
            f"GPA range: {gpa}. MBTI: {mbti}. Background: {background.replace('_', ' ')}. "
            f"In class discussions, {pronoun} typically: {behavior_text}. "
            f"When participating in online or forum-style discussions, {name} writes in a way "
            f"that reflects {pronoun} personality — {pronoun} engagement level and confidence "
            f"shape how often and how thoroughly {pronoun} responds."
        )

    def _generate_persona_llm(
        self,
        name, age, gender, grade, gpa, mbti, background,
        arch_id, archetype_desc, traits, behaviors, context, seed_prompt,
    ) -> str:
        """Use LLM to generate a rich, natural-sounding student persona."""

        behavior_text = "\n".join(f"- {b}" for b in behaviors)
        traits_text = json.dumps(traits, indent=2)

        prompt = f"""Generate a detailed student persona for a simulated AP Government classroom discussion.

## Student Info
Name: {name}
Age: {age} | Gender: {gender} | Grade: {grade} | GPA: {gpa} | MBTI: {mbti}
Background: {background.replace('_', ' ')}
Archetype: {arch_id} — {archetype_desc}

## Traits
{traits_text}

## Typical Behavior Patterns
{behavior_text}

## Classroom Context
{context}

## Current Discussion Topic
{seed_prompt}

## Instructions
Write a 200-300 word persona description in third person. Include:
1. Their personality and how it shows up in class
2. Their knowledge level and study habits for this subject
3. How they participate in discussions (frequency, style, what triggers them to speak)
4. Their likely perspective on the current discussion topic
5. Quirks or habits that make them feel like a real student
6. How they interact with peers who disagree with them

Write as a single flowing paragraph. Do NOT use bullet points or headers.
The persona must feel like a real teenager — not a caricature."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at creating realistic student personas for "
                            "educational simulations. Write natural, vivid descriptions that "
                            "capture how real high school students think and behave in AP classes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=500,
            )
            persona = response.choices[0].message.content.strip()
            # Clean up any markdown formatting
            persona = persona.replace("\n\n", " ").replace("\n", " ")
            return persona

        except Exception as e:
            logger.warning(f"LLM persona generation failed for {name}: {e}, falling back to rules")
            return self._generate_persona_rule(
                name, age, gender, grade, gpa, mbti, background,
                arch_id, archetype_desc, behaviors,
            )

    def _pick_topics(self, arch_id: str) -> List[str]:
        base = ["U.S. Government", "Civil Rights", "Constitutional Law"]
        extra_map = {
            "honors_overachiever": ["Supreme Court Cases", "Federalism", "AP Exam Prep"],
            "debate_club_kid": ["Political Debate", "Philosophy of Law", "Current Events"],
            "quiet_thinker": ["Political Theory", "Ethics", "History"],
            "socially_engaged_activist": ["Social Justice", "Voting Rights", "Inequality"],
            "disengaged_but_smart": ["Pop Culture", "Sports", "Technology"],
            "esl_student": ["Immigration Policy", "Bilingual Education", "Cultural Identity"],
            "class_clown": ["Memes", "Pop Culture", "Sports"],
            "politically_conservative": ["States' Rights", "Limited Government", "Free Market"],
        }
        return base + extra_map.get(arch_id, ["General Topics"])


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """Generate profiles from the sample config (CLI usage)."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate AP student profiles for OASIS simulation")
    parser.add_argument("--config", type=str, required=True, help="Path to the AP config JSON")
    parser.add_argument("--output", type=str, default="reddit_profiles.json", help="Output file path")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM enrichment, use rules only")
    args = parser.parse_args()

    generator = StudentProfileGenerator()

    def progress(current, total, msg):
        print(f"  [{current}/{total}] {msg}")

    profiles = generator.generate_profiles(
        config_path=args.config,
        use_llm=not args.no_llm,
        progress_callback=progress,
    )

    generator.save_profiles(profiles, args.output)
    print(f"\nDone! {len(profiles)} profiles saved to {args.output}")


if __name__ == "__main__":
    main()
