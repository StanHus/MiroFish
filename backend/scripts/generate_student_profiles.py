#!/usr/bin/env python3
"""
Standalone student profile generator — no Flask dependency.
Generates OASIS-compatible Reddit profiles from an AP config JSON.

Usage:
    python scripts/generate_student_profiles.py \
        --config configs/ap_social_studies_sample.json \
        --output uploads/simulations/ap_test_1/reddit_profiles.json
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# ── path + env setup ─────────────────────────────────────────────────────────
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))

from dotenv import load_dotenv
_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

from openai import OpenAI

# ── MBTI pools by archetype ─────────────────────────────────────────────────
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
]


class StudentProfileGenerator:
    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.environ.get("LLM_MODEL_NAME", "gpt-5-mini")
        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._used_names: set = set()

    def generate_profiles(
        self, config_path: str, use_llm: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        archetypes = config.get("student_archetypes", [])
        classroom = config.get("classroom_context", {})
        meta = config.get("simulation_meta", {})
        discussion = config.get("discussion_config", {})
        context = self._build_context(meta, classroom, discussion)

        profiles = []
        user_id = 0
        total = sum(a.get("count", 1) for a in archetypes)

        for archetype in archetypes:
            count = archetype.get("count", 1)
            arch_id = archetype.get("archetype_id", "unknown")

            for i in range(count):
                if progress_callback:
                    progress_callback(user_id + 1, total, f"{arch_id} ({i+1}/{count})")

                profile = self._generate_one(
                    user_id, archetype, context,
                    classroom.get("seed_prompt", ""), use_llm,
                )
                profiles.append(profile)
                user_id += 1

        return profiles

    def save_profiles(self, profiles: List[Dict], file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(profiles)} profiles to {file_path}")

    def _generate_one(self, user_id, archetype, context, seed_prompt, use_llm) -> Dict:
        arch_id = archetype.get("archetype_id", "unknown")
        traits = archetype.get("traits", {})
        behaviors = archetype.get("behavior_patterns", [])
        demo_dist = archetype.get("demographic_distribution", {})

        gender = random.choice(demo_dist.get("gender", ["male", "female"]))
        background = self._pick_background(demo_dist)
        name = self._pick_name(gender, background)
        age = random.choice([16, 17, 17, 17, 18, 18])
        grade = 11 if age <= 17 else 12
        mbti = random.choice(ARCHETYPE_MBTI_POOLS.get(arch_id, ["INTP"]))
        gpa = self._roll_gpa(traits)
        engagement = traits.get("engagement", 0.5)

        # Bio
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
        bio = f"{name} — Grade {grade} {label_map.get(arch_id, 'student')}. GPA: {gpa}."

        # Persona
        if use_llm and self.client:
            persona = self._persona_llm(
                name, age, gender, grade, gpa, mbti, background,
                arch_id, archetype.get("description", ""),
                traits, behaviors, context, seed_prompt,
            )
        else:
            persona = self._persona_rule(
                name, age, gender, grade, gpa, mbti, background,
                arch_id, archetype.get("description", ""), behaviors,
            )

        # Topics
        base_topics = ["U.S. Government", "Civil Rights", "Constitutional Law"]
        extra = {
            "honors_overachiever": ["Supreme Court Cases", "Federalism"],
            "debate_club_kid": ["Political Debate", "Philosophy of Law"],
            "quiet_thinker": ["Political Theory", "Ethics"],
            "socially_engaged_activist": ["Social Justice", "Voting Rights"],
            "disengaged_but_smart": ["Pop Culture", "Technology"],
            "esl_student": ["Immigration Policy", "Cultural Identity"],
            "class_clown": ["Memes", "Pop Culture"],
            "politically_conservative": ["States' Rights", "Limited Government"],
        }

        username = name.lower().replace(" ", "_")
        username = "".join(c for c in username if c.isalnum() or c == "_")
        username = f"{username}_{random.randint(10, 99)}"

        return {
            "user_id": user_id,
            "username": username,
            "name": name,
            "bio": bio,
            "persona": persona,
            "karma": int(engagement * 400 + random.randint(20, 100)),
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "age": age,
            "gender": gender,
            "mbti": mbti,
            "country": "United States",
            "profession": f"High School Student (Grade {grade})",
            "interested_topics": base_topics + extra.get(arch_id, []),
        }

    def _pick_background(self, demo):
        bgs = demo.get("background", ["any"])
        bg = random.choice(bgs)
        if bg == "any":
            bg = random.choice(["suburban_white", "african_american", "latino", "east_asian", "south_asian"])
        return bg

    def _pick_name(self, gender, background):
        bg_key = background.split("_")[0] if "_" in background else background
        pool_map = {
            "suburban": "suburban_white", "african": "african_american",
            "east": "east_asian", "south": "south_asian", "latino": "latino",
            "recent": "latino", "refugee": "latino", "rural": "suburban_white",
            "religious": "suburban_white", "libertarian": "suburban_white",
            "first": "default", "introvert": "default", "lgbtq": "default",
            "immigrant": "latino", "mixed": "default",
        }
        bg_base = pool_map.get(bg_key, "default")

        if gender == "nonbinary":
            pool = NAME_POOLS.get("nonbinary", NAME_POOLS["default_male"])
        else:
            g = "_male" if gender == "male" else "_female"
            pool = NAME_POOLS.get(f"{bg_base}{g}", NAME_POOLS[f"default{g}"])

        for _ in range(20):
            first = random.choice(pool)
            last = random.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full not in self._used_names:
                self._used_names.add(full)
                return full
        full = f"{random.choice(pool)} {random.choice(LAST_NAMES)} {random.randint(1,99)}"
        self._used_names.add(full)
        return full

    def _roll_gpa(self, traits):
        depth = traits.get("knowledge_depth", "intermediate")
        eng = traits.get("engagement", 0.5)
        if depth == "advanced" and eng > 0.8:
            return random.choice(["3.8-4.0", "3.9-4.0"])
        elif depth == "advanced" or eng > 0.7:
            return random.choice(["3.5-3.8", "3.6-3.9"])
        elif "intermediate" in str(depth):
            return random.choice(["3.0-3.5", "2.8-3.3"])
        else:
            return random.choice(["2.5-3.0", "2.3-2.8"])

    def _build_context(self, meta, classroom, discussion):
        return "\n".join([
            f"Subject: {meta.get('subject', 'AP Social Studies')}",
            f"Unit: {meta.get('unit', '')}",
            f"School: {classroom.get('school_type', 'public suburban')}",
            f"Teacher: {classroom.get('teacher_name', 'the teacher')}",
            f"Format: {classroom.get('discussion_format', 'class discussion')}",
        ])

    def _persona_rule(self, name, age, gender, grade, gpa, mbti, background, arch_id, desc, behaviors):
        btext = "; ".join(behaviors[:4]) if behaviors else "participates normally"
        p = "they" if gender == "nonbinary" else ("he" if gender == "male" else "she")
        return (
            f"{name} is a {age}-year-old {gender} student in Grade {grade} at a suburban American "
            f"high school, taking AP U.S. Government and Politics. {desc} "
            f"GPA range: {gpa}. MBTI: {mbti}. Background: {background.replace('_', ' ')}. "
            f"In class discussions, {p} typically: {btext}. "
            f"When participating in forum-style discussions, {name} writes in a way that reflects "
            f"{p} personality — {p} engagement level and confidence shape how often and how "
            f"thoroughly {p} responds."
        )

    def _persona_llm(self, name, age, gender, grade, gpa, mbti, background,
                     arch_id, desc, traits, behaviors, context, seed_prompt):
        behavior_text = "\n".join(f"- {b}" for b in behaviors)

        prompt = f"""Generate a detailed student persona for a simulated AP Government classroom discussion.

## Student Info
Name: {name} | Age: {age} | Gender: {gender} | Grade: {grade} | GPA: {gpa} | MBTI: {mbti}
Background: {background.replace('_', ' ')}
Archetype: {arch_id} — {desc}

## Typical Behaviors
{behavior_text}

## Classroom Context
{context}

## Discussion Topic
{seed_prompt}

## Instructions
Write a 200-300 word persona in third person as a single flowing paragraph.
Include: personality, knowledge level, participation style, likely perspective on the topic,
quirks that make them feel real, and how they handle disagreement.
Must feel like a real teenager, not a caricature. No bullet points or headers."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You create realistic student personas for educational simulations. Write natural, vivid descriptions of real high school students."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=500,
            )
            persona = response.choices[0].message.content.strip()
            return persona.replace("\n\n", " ").replace("\n", " ")
        except Exception as e:
            print(f"    ⚠ LLM failed for {name}: {e}, using rule-based")
            return self._persona_rule(
                name, age, gender, grade, gpa, mbti, background,
                arch_id, desc, behaviors,
            )


def main():
    parser = argparse.ArgumentParser(description="Generate AP student profiles")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="uploads/simulations/ap_test_1/reddit_profiles.json")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    gen = StudentProfileGenerator()
    print(f"\n  🎓 Generating student profiles...")
    print(f"     Config: {args.config}")
    print(f"     Model:  {gen.model_name}")
    print(f"     LLM:    {'OFF' if args.no_llm else 'ON'}\n")

    def progress(cur, total, msg):
        print(f"  [{cur:2d}/{total}] {msg}")

    profiles = gen.generate_profiles(args.config, use_llm=not args.no_llm, progress_callback=progress)
    gen.save_profiles(profiles, args.output)
    print(f"\n  ✅ Done! {len(profiles)} student profiles generated.\n")


if __name__ == "__main__":
    main()
