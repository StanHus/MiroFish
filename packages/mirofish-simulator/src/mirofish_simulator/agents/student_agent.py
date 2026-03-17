"""
StudentAgent - An LLM agent that role-plays as a student.

The key insight: the agent DOESN'T KNOW THE RIGHT ANSWER.

It receives:
1. A constrained knowledge base (only what this student knows)
2. A filtered perception (how they read the question)

Then it attempts to answer from that limited perspective.
Errors are genuine - they come from knowledge gaps, not dice rolls.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .knowledge import KnowledgeBase, KnowledgeBuilder
from .perception import PerceivedContent, PerceptionFilter


@dataclass
class AgentConfig:
    """Configuration for a student agent."""

    grade: int
    archetype: str
    subject: Optional[str] = None

    # Personality traits that affect answering
    guesses_when_unsure: bool = True
    elimination_strategy: bool = True  # Uses process of elimination
    first_instinct: bool = False       # Goes with first instinct vs deliberates


@dataclass
class AgentResponse:
    """Response from a student agent."""

    # The answer
    selected: str           # "A", "B", "C", "D"
    selected_text: str      # The actual option text
    confidence: str         # "sure", "pretty_sure", "guessing", "no_idea"

    # How they got there
    reasoning: str          # Their thought process
    knowledge_used: List[str]    # Facts they applied
    knowledge_gaps: List[str]    # Things they didn't know

    # Perception factors
    words_not_understood: List[str]
    what_they_thought_question_asked: str

    # Meta
    is_correct: Optional[bool] = None  # Set after comparing to answer key

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "selected_text": self.selected_text,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "knowledge_used": self.knowledge_used,
            "knowledge_gaps": self.knowledge_gaps,
            "words_not_understood": self.words_not_understood,
            "what_they_thought_question_asked": self.what_they_thought_question_asked,
            "is_correct": self.is_correct,
        }


class StudentAgent:
    """
    An LLM agent that role-plays as a specific type of student.

    Usage:
        agent = StudentAgent(grade=5, archetype="esl_student", api_key="sk-...")
        response = await agent.answer(question)

        print(response.selected)        # "B"
        print(response.reasoning)       # "I think it's about voting..."
        print(response.knowledge_gaps)  # ["Electoral College mechanics"]
    """

    def __init__(
        self,
        grade: int,
        archetype: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        config: Optional[AgentConfig] = None,
    ):
        self.grade = grade
        self.archetype = archetype
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.config = config or AgentConfig(grade=grade, archetype=archetype)

        # Initialize components
        self.knowledge_builder = KnowledgeBuilder(api_key, base_url, model)
        self.perception_filter = PerceptionFilter(api_key, base_url, model)

        # LLM client for answering
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def answer(
        self,
        question: Dict[str, Any],
        correct_answer: Optional[str] = None,
    ) -> AgentResponse:
        """
        Attempt to answer a question as this student.

        The agent doesn't know the correct answer - it answers
        based on its constrained knowledge and perception.

        Args:
            question: Question with "text", "options"
            correct_answer: Optional correct answer for marking (not shown to agent)

        Returns:
            AgentResponse with answer and reasoning
        """
        text = question.get("text") or question.get("question", "")
        options = question.get("options", [])
        subject = question.get("subject", self.config.subject or "general")

        # Step 1: Build knowledge constraint
        knowledge = await self.knowledge_builder.build_for_question(
            question, self.grade, subject
        )

        # Step 2: Apply perception filter
        perception = await self.perception_filter.apply(
            question, self.grade, self.archetype
        )

        # Step 3: Answer with constrained knowledge and perception
        response = await self._generate_answer(knowledge, perception, options)

        # Step 4: Mark correctness if answer key provided
        if correct_answer:
            response.is_correct = response.selected.upper() == correct_answer.upper()

        return response

    async def _generate_answer(
        self,
        knowledge: KnowledgeBase,
        perception: PerceivedContent,
        options: List[str],
    ) -> AgentResponse:
        """Generate answer using constrained knowledge and perception."""

        if not self.client:
            # Fallback: random guess with basic reasoning
            import random
            selected_idx = random.randint(0, len(options) - 1)
            selected = chr(65 + selected_idx)
            return AgentResponse(
                selected=selected,
                selected_text=str(options[selected_idx]),
                confidence="guessing",
                reasoning="I'm not sure, just picking one.",
                knowledge_used=[],
                knowledge_gaps=knowledge.facts_unknown[:3],
                words_not_understood=perception.unknown_words,
                what_they_thought_question_asked=perception.perceived_text,
            )

        # Build the agent prompt
        knowledge_section = knowledge.to_prompt_section()
        perception_section = perception.to_prompt_section()

        options_text = "\n".join(
            f"  {chr(65+i)}) {opt}" for i, opt in enumerate(options)
        )

        system_prompt = f"""You are a grade {self.grade} student with the "{self.archetype}" personality.

CRITICAL INSTRUCTIONS:
- You can ONLY use the knowledge listed below
- You DO NOT know anything beyond what's listed
- If something isn't in your knowledge, you DON'T KNOW IT
- You must answer based ONLY on what you know and perceive
- It's OK to guess if you're unsure - real students do that

{knowledge_section}

REMEMBER: You are a grade {self.grade} student. Think like one. Answer like one."""

        user_prompt = f"""{perception_section}

THE OPTIONS ARE:
{options_text}

Now answer this question AS a grade {self.grade} {self.archetype} student.

Think through it:
1. What do I KNOW that helps here? (only from the knowledge above)
2. What am I NOT SURE about?
3. Which option makes sense given what I know?

Then pick your answer.

Return JSON:
{{
  "thinking": "Your actual thought process as this student",
  "selected": "A" or "B" or "C" or "D",
  "confidence": "sure" or "pretty_sure" or "guessing" or "no_idea",
  "knowledge_used": ["fact I used", "another fact"],
  "knowledge_gaps": ["thing I didn't know that would have helped"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Some variability like real students
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            selected = analysis.get("selected", "A").upper()
            selected_idx = ord(selected) - ord("A")
            if 0 <= selected_idx < len(options):
                selected_text = str(options[selected_idx])
            else:
                selected_text = str(options[0]) if options else ""
                selected = "A"

            return AgentResponse(
                selected=selected,
                selected_text=selected_text,
                confidence=analysis.get("confidence", "guessing"),
                reasoning=analysis.get("thinking", ""),
                knowledge_used=analysis.get("knowledge_used", []),
                knowledge_gaps=analysis.get("knowledge_gaps", knowledge.facts_unknown[:3]),
                words_not_understood=perception.unknown_words,
                what_they_thought_question_asked=perception.perceived_text,
            )

        except Exception as e:
            # Fallback on error
            import random
            selected_idx = random.randint(0, len(options) - 1)
            selected = chr(65 + selected_idx)
            return AgentResponse(
                selected=selected,
                selected_text=str(options[selected_idx]) if options else "",
                confidence="no_idea",
                reasoning=f"Error occurred, guessing: {str(e)[:50]}",
                knowledge_used=[],
                knowledge_gaps=knowledge.facts_unknown[:3],
                words_not_understood=perception.unknown_words,
                what_they_thought_question_asked=perception.perceived_text,
            )

    def __repr__(self) -> str:
        return f"StudentAgent(grade={self.grade}, archetype='{self.archetype}')"
