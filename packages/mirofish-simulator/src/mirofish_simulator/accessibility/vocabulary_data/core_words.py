"""
Core Vocabulary - Dolch Sight Words and Fry Words.

These are the most common words in English, mapped to the grade level
at which students should know them.

Grade 0 = Kindergarten
Grade 1-12 = Elementary through High School
"""

# ── Dolch Sight Words ───────────────────────────────────────────────────────
# 220 most common words in children's books, organized by grade level

DOLCH_PREPRIMER = {  # Kindergarten (grade 0)
    "a": 0, "and": 0, "away": 0, "big": 0, "blue": 0, "can": 0, "come": 0,
    "down": 0, "find": 0, "for": 0, "funny": 0, "go": 0, "help": 0, "here": 0,
    "i": 0, "in": 0, "is": 0, "it": 0, "jump": 0, "little": 0, "look": 0,
    "make": 0, "me": 0, "my": 0, "not": 0, "one": 0, "play": 0, "red": 0,
    "run": 0, "said": 0, "see": 0, "the": 0, "three": 0, "to": 0, "two": 0,
    "up": 0, "we": 0, "where": 0, "yellow": 0, "you": 0,
}

DOLCH_PRIMER = {  # Grade 1
    "all": 1, "am": 1, "are": 1, "at": 1, "ate": 1, "be": 1, "black": 1,
    "brown": 1, "but": 1, "came": 1, "cat": 1, "did": 1, "do": 1, "dog": 1, "eat": 1, "four": 1,
    "get": 1, "good": 1, "have": 1, "he": 1, "into": 1, "like": 1, "must": 1,
    "new": 1, "no": 1, "now": 1, "on": 1, "our": 1, "out": 1, "please": 1,
    "pretty": 1, "ran": 1, "ride": 1, "saw": 1, "say": 1, "she": 1, "so": 1,
    "soon": 1, "that": 1, "there": 1, "they": 1, "this": 1, "too": 1, "under": 1,
    "want": 1, "was": 1, "well": 1, "went": 1, "what": 1, "white": 1, "who": 1,
    "will": 1, "with": 1, "yes": 1,
}

DOLCH_GRADE1 = {  # Grade 1
    "after": 1, "again": 1, "an": 1, "any": 1, "as": 1, "ask": 1, "by": 1,
    "could": 1, "every": 1, "fly": 1, "from": 1, "give": 1, "going": 1,
    "had": 1, "has": 1, "her": 1, "him": 1, "his": 1, "how": 1, "just": 1,
    "know": 1, "let": 1, "live": 1, "may": 1, "of": 1, "old": 1, "once": 1,
    "open": 1, "over": 1, "put": 1, "round": 1, "some": 1, "stop": 1, "take": 1,
    "thank": 1, "them": 1, "then": 1, "think": 1, "walk": 1, "were": 1, "when": 1,
}

DOLCH_GRADE2 = {  # Grade 2
    "always": 2, "around": 2, "because": 2, "been": 2, "before": 2, "best": 2,
    "both": 2, "buy": 2, "call": 2, "cold": 2, "does": 2, "don't": 2, "fast": 2,
    "first": 2, "five": 2, "found": 2, "gave": 2, "goes": 2, "green": 2,
    "its": 2, "made": 2, "many": 2, "off": 2, "or": 2, "pull": 2, "read": 2,
    "right": 2, "sing": 2, "sit": 2, "sleep": 2, "tell": 2, "their": 2,
    "these": 2, "those": 2, "upon": 2, "us": 2, "use": 2, "very": 2, "wash": 2,
    "which": 2, "why": 2, "wish": 2, "work": 2, "would": 2, "write": 2, "your": 2,
}

DOLCH_GRADE3 = {  # Grade 3
    "about": 3, "better": 3, "bring": 3, "carry": 3, "clean": 3, "cut": 3,
    "done": 3, "draw": 3, "drink": 3, "eight": 3, "fall": 3, "far": 3,
    "full": 3, "got": 3, "grow": 3, "hold": 3, "hot": 3, "hurt": 3, "if": 3,
    "keep": 3, "kind": 3, "laugh": 3, "light": 3, "long": 3, "much": 3,
    "myself": 3, "never": 3, "only": 3, "own": 3, "pick": 3, "seven": 3,
    "shall": 3, "show": 3, "six": 3, "small": 3, "start": 3, "ten": 3,
    "today": 3, "together": 3, "try": 3, "warm": 3,
}


# ── Fry Words (Grades 1-5) ──────────────────────────────────────────────────
# Extended high-frequency words beyond Dolch

FRY_GRADE1 = {
    "word": 1, "each": 1, "way": 1, "long": 1, "day": 1, "part": 1, "sound": 1,
    "number": 1, "name": 1, "back": 1, "people": 1, "year": 1, "most": 1,
    "water": 1, "more": 1, "time": 1, "other": 1, "than": 1, "such": 1,
    "thing": 1, "man": 1, "boy": 1, "girl": 1, "hand": 1, "place": 1, "end": 1,
    "home": 1, "last": 1, "tree": 1, "world": 1, "next": 1, "left": 1,
    "still": 1, "own": 1, "house": 1, "thought": 1, "city": 1, "mother": 1,
    "father": 1, "head": 1, "line": 1, "turn": 1, "hard": 1, "land": 1,
}

FRY_GRADE2 = {
    "different": 2, "move": 2, "change": 2, "kind": 2, "picture": 2, "spell": 2,
    "air": 2, "animal": 2, "study": 2, "same": 2, "learn": 2, "answer": 2,
    "form": 2, "sentence": 2, "example": 2, "paper": 2, "story": 2, "point": 2,
    "page": 2, "letter": 2, "between": 2, "few": 2, "school": 2, "important": 2,
    "below": 2, "country": 2, "plant": 2, "food": 2, "earth": 2, "eye": 2,
    "light": 2, "second": 2, "enough": 2, "mile": 2, "sea": 2, "close": 2,
    "idea": 2, "face": 2, "car": 2, "watch": 2, "Indian": 2, "real": 2,
    "almost": 2, "let": 2, "above": 2, "girl": 2, "sometimes": 2, "mountain": 2,
}

FRY_GRADE3 = {
    "state": 3, "through": 3, "high": 3, "begin": 3, "certain": 3, "young": 3,
    "talk": 3, "story": 3, "leave": 3, "body": 3, "music": 3, "color": 3,
    "stand": 3, "sun": 3, "question": 3, "complete": 3, "ship": 3, "area": 3,
    "rock": 3, "order": 3, "fire": 3, "south": 3, "problem": 3, "piece": 3,
    "surface": 3, "listen": 3, "whole": 3, "king": 3, "space": 3, "heard": 3,
    "best": 3, "hour": 3, "reach": 3, "top": 3, "during": 3, "short": 3,
    "ship": 3, "understand": 3, "moon": 3, "island": 3, "foot": 3, "system": 3,
    "busy": 3, "several": 3, "war": 3, "half": 3, "inside": 3, "stay": 3,
}

FRY_GRADE4 = {
    "figure": 4, "certain": 4, "field": 4, "travel": 4, "wood": 4, "fire": 4,
    "upon": 4, "pattern": 4, "numeral": 4, "table": 4, "north": 4, "slowly": 4,
    "money": 4, "map": 4, "farm": 4, "mark": 4, "vowel": 4, "language": 4,
    "job": 4, "morning": 4, "south": 4, "winter": 4, "produce": 4, "fact": 4,
    "street": 4, "inch": 4, "multiply": 4, "nothing": 4, "course": 4, "front": 4,
    "teach": 4, "week": 4, "final": 4, "gave": 4, "green": 4, "oh": 4,
    "quick": 4, "develop": 4, "ocean": 4, "warm": 4, "free": 4, "minute": 4,
    "strong": 4, "special": 4, "mind": 4, "behind": 4, "clear": 4, "tail": 4,
    "produce": 4, "direction": 4, "center": 4, "farmers": 4, "ready": 4,
}

FRY_GRADE5 = {
    "region": 5, "return": 5, "believe": 5, "dance": 5, "members": 5, "picked": 5,
    "simple": 5, "cells": 5, "paint": 5, "mind": 5, "love": 5, "cause": 5,
    "rain": 5, "exercise": 5, "eggs": 5, "train": 5, "blue": 5, "wish": 5,
    "drop": 5, "developed": 5, "window": 5, "difference": 5, "distance": 5,
    "heart": 5, "sit": 5, "sum": 5, "summer": 5, "wall": 5, "forest": 5,
    "probably": 5, "legs": 5, "sat": 5, "main": 5, "winter": 5, "wide": 5,
    "written": 5, "length": 5, "reason": 5, "kept": 5, "interest": 5, "arms": 5,
    "brother": 5, "race": 5, "present": 5, "beautiful": 5, "store": 5, "job": 5,
    "edge": 5, "past": 5, "sign": 5, "record": 5, "finished": 5, "discovered": 5,
}


# ── Common Words by Grade (Extended) ───────────────────────────────────────
# Words that are commonly encountered at each grade level

COMMON_GRADE4 = {
    "actually": 4, "although": 4, "amount": 4, "average": 4, "basic": 4,
    "became": 4, "billion": 4, "building": 4, "century": 4, "common": 4,
    "community": 4, "contain": 4, "continue": 4, "control": 4, "correct": 4,
    "decided": 4, "describe": 4, "disease": 4, "energy": 4, "entire": 4,
    "environment": 4, "especially": 4, "Europe": 4, "evening": 4, "exactly": 4,
    "explain": 4, "famous": 4, "figure": 4, "finally": 4, "future": 4,
    "general": 4, "government": 4, "history": 4, "human": 4, "imagine": 4,
    "include": 4, "increase": 4, "island": 4, "knowledge": 4, "level": 4,
    "material": 4, "measure": 4, "million": 4, "modern": 4, "national": 4,
    "natural": 4, "necessary": 4, "notice": 4, "original": 4, "particular": 4,
    "perhaps": 4, "period": 4, "physical": 4, "popular": 4, "possible": 4,
    "power": 4, "practice": 4, "president": 4, "process": 4, "provide": 4,
    "public": 4, "quickly": 4, "rather": 4, "receive": 4, "remember": 4,
    "report": 4, "result": 4, "science": 4, "section": 4, "sense": 4,
    "series": 4, "service": 4, "similar": 4, "simple": 4, "single": 4,
    "society": 4, "special": 4, "standard": 4, "statement": 4, "subject": 4,
    "suppose": 4, "surface": 4, "system": 4, "technology": 4, "temperature": 4,
    "therefore": 4, "total": 4, "toward": 4, "trade": 4, "traditional": 4,
    "treat": 4, "tribe": 4, "usually": 4, "various": 4, "village": 4,
    "weight": 4, "whether": 4, "within": 4, "wonder": 4,
}

COMMON_GRADE5 = {
    "ability": 5, "accept": 5, "achieve": 5, "action": 5, "activity": 5,
    "affect": 5, "ancient": 5, "approach": 5, "article": 5, "attitude": 5,
    "author": 5, "available": 5, "behavior": 5, "benefit": 5, "campaign": 5,
    "capital": 5, "category": 5, "challenge": 5, "character": 5, "choice": 5,
    "civilization": 5, "climate": 5, "colony": 5, "compare": 5, "complex": 5,
    "concept": 5, "condition": 5, "conflict": 5, "consider": 5, "construction": 5,
    "continent": 5, "contrast": 5, "create": 5, "culture": 5, "current": 5,
    "decision": 5, "democracy": 5, "design": 5, "detail": 5, "determine": 5,
    "economy": 5, "education": 5, "effect": 5, "efficient": 5, "election": 5,
    "element": 5, "empire": 5, "establish": 5, "estimate": 5, "evidence": 5,
    "expand": 5, "experiment": 5, "explore": 5, "express": 5, "factor": 5,
    "feature": 5, "focus": 5, "force": 5, "function": 5, "globe": 5,
    "identify": 5, "image": 5, "immigrant": 5, "impact": 5, "individual": 5,
    "industry": 5, "influence": 5, "information": 5, "institution": 5, "involve": 5,
    "issue": 5, "labor": 5, "legal": 5, "locate": 5, "major": 5,
    "manufacture": 5, "method": 5, "military": 5, "movement": 5, "negative": 5,
    "network": 5, "observe": 5, "obvious": 5, "occur": 5, "opportunity": 5,
    "organize": 5, "participate": 5, "percent": 5, "perform": 5, "perspective": 5,
    "politics": 5, "population": 5, "positive": 5, "primary": 5, "principle": 5,
    "procedure": 5, "production": 5, "property": 5, "publish": 5, "purpose": 5,
    "reform": 5, "relate": 5, "religion": 5, "represent": 5, "require": 5,
    "research": 5, "resource": 5, "respond": 5, "reveal": 5, "revolution": 5,
    "role": 5, "significant": 5, "source": 5, "specific": 5, "strategy": 5,
    "structure": 5, "struggle": 5, "success": 5, "suggest": 5, "support": 5,
    "symbol": 5, "technique": 5, "theory": 5, "topic": 5, "tradition": 5,
    "transfer": 5, "transport": 5, "trend": 5, "value": 5, "vary": 5,
}


# ── Combine All Core Vocabulary ─────────────────────────────────────────────

CORE_VOCABULARY: dict[str, int] = {}
CORE_VOCABULARY.update(DOLCH_PREPRIMER)
CORE_VOCABULARY.update(DOLCH_PRIMER)
CORE_VOCABULARY.update(DOLCH_GRADE1)
CORE_VOCABULARY.update(DOLCH_GRADE2)
CORE_VOCABULARY.update(DOLCH_GRADE3)
CORE_VOCABULARY.update(FRY_GRADE1)
CORE_VOCABULARY.update(FRY_GRADE2)
CORE_VOCABULARY.update(FRY_GRADE3)
CORE_VOCABULARY.update(FRY_GRADE4)
CORE_VOCABULARY.update(FRY_GRADE5)
CORE_VOCABULARY.update(COMMON_GRADE4)
CORE_VOCABULARY.update(COMMON_GRADE5)
