"""
Subject-Specific Vocabulary - Math, Science, Social Studies, ELA.

These are domain-specific terms mapped to the grade level where
they are typically introduced in US curriculum.
"""

# ── Mathematics Vocabulary ──────────────────────────────────────────────────

MATH_VOCABULARY = {
    # Kindergarten - Grade 1
    "add": 0, "plus": 0, "minus": 1, "subtract": 1, "equals": 0, "count": 0,
    "number": 0, "zero": 0, "one": 0, "two": 0, "three": 0, "four": 0, "five": 0,
    "six": 0, "seven": 0, "eight": 0, "nine": 0, "ten": 0, "more": 0, "less": 0,
    "greater": 1, "fewer": 1, "same": 0, "different": 1, "shape": 0,
    "circle": 0, "square": 0, "triangle": 0, "rectangle": 1,

    # Grade 2-3
    "multiply": 3, "times": 2, "divide": 3, "divided": 3, "sum": 3, "total": 2,
    "difference": 3, "product": 3, "quotient": 4, "remainder": 3, "even": 2,
    "odd": 2, "dozen": 2, "hundred": 1, "thousand": 2, "place": 2, "value": 3,
    "digit": 2, "numeral": 2, "fraction": 3, "half": 2, "quarter": 3, "third": 2,
    "fourth": 2, "whole": 2, "part": 2, "equal": 2, "array": 3, "row": 2,
    "column": 3, "graph": 3, "data": 3, "tally": 2, "chart": 2, "table": 2,
    "perimeter": 3, "area": 3, "length": 2, "width": 3, "height": 2,

    # Grade 4-5
    "decimal": 4, "percent": 5, "percentage": 5, "ratio": 6, "rate": 5,
    "factor": 4, "multiple": 4, "prime": 4, "composite": 5, "divisible": 4,
    "divisibility": 5, "numerator": 4, "denominator": 4, "equivalent": 4,
    "simplify": 5, "reduce": 5, "improper": 5, "mixed": 4, "convert": 5,
    "estimate": 4, "approximate": 5, "round": 3, "rounding": 4, "average": 4,
    "mean": 5, "median": 5, "mode": 5, "range": 5, "coordinate": 5, "axis": 5,
    "plot": 4, "ordered": 4, "pair": 4, "origin": 5, "quadrant": 6,
    "angle": 4, "degree": 4, "acute": 4, "obtuse": 4, "right": 2, "straight": 3,
    "parallel": 5, "perpendicular": 5, "intersect": 5, "vertex": 5, "vertices": 5,
    "edge": 4, "face": 4, "polygon": 4, "pentagon": 4, "hexagon": 4, "octagon": 4,
    "cube": 3, "sphere": 4, "cylinder": 4, "cone": 4, "prism": 5, "pyramid": 4,
    "volume": 5, "capacity": 4, "liter": 3, "milliliter": 4, "gram": 3, "kilogram": 4,

    # Grade 6-7
    "integer": 6, "negative": 6, "positive": 6, "absolute": 6, "opposite": 6,
    "rational": 7, "irrational": 8, "expression": 6, "variable": 6, "term": 6,
    "coefficient": 7, "constant": 7, "equation": 6, "inequality": 7, "solve": 5,
    "solution": 6, "substitute": 7, "evaluate": 6, "simplify": 5, "combine": 6,
    "distribute": 7, "distributive": 7, "associative": 6, "commutative": 6,
    "identity": 7, "inverse": 7, "reciprocal": 7, "proportion": 6, "proportional": 7,
    "scale": 5, "similar": 5, "congruent": 7, "transformation": 7, "translation": 7,
    "reflection": 7, "rotation": 7, "dilation": 8, "symmetry": 6, "probability": 6,
    "outcome": 6, "event": 5, "sample": 6, "space": 5, "theoretical": 8, "experimental": 7,

    # Grade 8-9 (Pre-Algebra/Algebra I)
    "polynomial": 9, "monomial": 8, "binomial": 8, "trinomial": 9, "degree": 4,
    "exponent": 7, "power": 5, "base": 5, "radical": 9, "square": 2, "root": 5,
    "cube": 3, "perfect": 4, "scientific": 5, "notation": 7, "slope": 8,
    "intercept": 8, "linear": 8, "nonlinear": 9, "quadratic": 9, "parabola": 9,
    "vertex": 5, "domain": 8, "range": 5, "function": 8, "relation": 7,
    "input": 6, "output": 6, "independent": 7, "dependent": 7, "discrete": 9,
    "continuous": 9, "sequence": 7, "arithmetic": 5, "geometric": 8, "recursive": 10,
    "explicit": 9, "formula": 6, "pythagorean": 8, "theorem": 8, "hypotenuse": 8,
    "leg": 3, "trigonometry": 9, "sine": 10, "cosine": 10, "tangent": 10,

    # Grade 10-12 (Geometry/Algebra II/Pre-Calc)
    "postulate": 10, "axiom": 10, "proof": 9, "theorem": 8, "corollary": 10,
    "converse": 9, "inverse": 7, "contrapositive": 10, "biconditional": 10,
    "hypothesis": 7, "conclusion": 6, "deductive": 9, "inductive": 9,
    "complementary": 8, "supplementary": 8, "adjacent": 8, "vertical": 7,
    "alternate": 8, "corresponding": 8, "transversal": 9, "bisector": 8,
    "midpoint": 7, "median": 5, "altitude": 7, "centroid": 10, "incenter": 10,
    "circumcenter": 10, "orthocenter": 10, "inscribed": 9, "circumscribed": 10,
    "tangent": 10, "secant": 10, "chord": 8, "arc": 7, "sector": 9,
    "radian": 10, "logarithm": 10, "logarithmic": 10, "exponential": 9,
    "asymptote": 10, "discontinuity": 11, "limit": 11, "derivative": 11,
    "integral": 12, "differential": 12, "calculus": 11, "vector": 10, "matrix": 10,
    "determinant": 11, "eigenvalue": 12, "eigenvector": 12,
}


# ── Science Vocabulary ──────────────────────────────────────────────────────

SCIENCE_VOCABULARY = {
    # Elementary (K-3)
    "animal": 0, "plant": 0, "living": 1, "nonliving": 1, "grow": 0, "change": 1,
    "weather": 1, "sunny": 0, "cloudy": 0, "rainy": 0, "windy": 1, "season": 1,
    "spring": 1, "summer": 1, "fall": 1, "winter": 1, "hot": 0, "cold": 0,
    "warm": 1, "cool": 1, "temperature": 3, "solid": 2, "liquid": 2, "gas": 2,
    "matter": 3, "energy": 4, "light": 2, "sound": 2, "heat": 2, "motion": 3,
    "force": 3, "push": 0, "pull": 0, "magnet": 2, "magnetic": 3,

    # Elementary (4-5)
    "organism": 4, "habitat": 3, "ecosystem": 5, "food": 1, "chain": 3, "web": 3,
    "producer": 4, "consumer": 4, "decomposer": 5, "predator": 4, "prey": 4,
    "adaptation": 5, "survive": 4, "extinct": 4, "endangered": 5, "species": 5,
    "classify": 4, "kingdom": 5, "vertebrate": 5, "invertebrate": 5, "mammal": 3,
    "reptile": 3, "amphibian": 4, "bird": 1, "fish": 1, "insect": 2,
    "cell": 5, "tissue": 6, "organ": 5, "system": 4, "nucleus": 6,
    "photosynthesis": 5, "chlorophyll": 6, "oxygen": 5, "carbon": 6, "dioxide": 6,
    "cycle": 4, "water": 1, "rock": 2, "mineral": 4, "soil": 3, "erosion": 5,
    "weathering": 5, "fossil": 4, "layer": 4, "sediment": 5, "volcano": 4,
    "earthquake": 4, "plate": 5, "tectonic": 7, "crust": 5, "mantle": 6, "core": 5,

    # Middle School (6-8)
    "atom": 6, "molecule": 6, "element": 6, "compound": 7, "mixture": 6,
    "solution": 6, "solvent": 7, "solute": 7, "dissolve": 5, "concentration": 8,
    "chemical": 6, "reaction": 6, "reactant": 7, "product": 6, "catalyst": 8,
    "enzyme": 8, "protein": 7, "carbohydrate": 7, "lipid": 8, "nucleic": 9,
    "acid": 6, "base": 5, "neutral": 7, "pH": 7, "ion": 8, "electron": 7,
    "proton": 7, "neutron": 7, "nucleus": 6, "atomic": 7, "mass": 6, "number": 0,
    "periodic": 7, "table": 2, "metal": 5, "nonmetal": 7, "metalloid": 8,
    "bond": 7, "ionic": 8, "covalent": 9, "valence": 9,
    "genetics": 8, "gene": 8, "chromosome": 8, "DNA": 8, "RNA": 9, "heredity": 7,
    "trait": 6, "dominant": 7, "recessive": 8, "allele": 9, "genotype": 9,
    "phenotype": 9, "mutation": 8, "evolution": 8, "natural": 4, "selection": 6,
    "mitosis": 8, "meiosis": 9, "cellular": 7, "respiration": 7, "fermentation": 8,

    # High School (9-12)
    "momentum": 9, "velocity": 8, "acceleration": 8, "friction": 7, "gravity": 5,
    "inertia": 9, "newton": 8, "joule": 9, "watt": 9, "volt": 9, "ampere": 10,
    "resistance": 8, "current": 7, "circuit": 6, "conductor": 7, "insulator": 7,
    "semiconductor": 10, "electromagnetic": 9, "spectrum": 8, "wavelength": 8,
    "frequency": 8, "amplitude": 9, "wave": 5, "photon": 10, "quantum": 11,
    "relativity": 11, "thermodynamics": 10, "entropy": 11, "kinetic": 9,
    "potential": 8, "conservation": 8,
    "stoichiometry": 10, "molarity": 10, "mole": 9, "avogadro": 10,
    "equilibrium": 9, "oxidation": 9, "reduction": 9, "electrochemistry": 11,
    "organic": 8, "inorganic": 9, "polymer": 9, "hydrocarbon": 10,
    "ecology": 8, "biome": 7, "biodiversity": 8, "symbiosis": 8, "mutualism": 9,
    "parasitism": 9, "commensalism": 10, "niche": 8, "succession": 9,
    "homeostasis": 10, "metabolism": 9, "hormone": 8, "neuron": 9, "synapse": 10,
}


# ── Social Studies Vocabulary ───────────────────────────────────────────────

SOCIAL_STUDIES_VOCABULARY = {
    # Elementary (K-3)
    "community": 2, "neighbor": 1, "family": 1, "friend": 0, "home": 0,
    "school": 0, "city": 2, "town": 2, "state": 3, "country": 2, "nation": 4,
    "map": 2, "globe": 3, "continent": 3, "ocean": 2, "river": 2, "mountain": 2,
    "desert": 3, "forest": 3, "island": 3, "north": 2, "south": 2, "east": 2,
    "west": 2, "direction": 2, "compass": 3, "symbol": 3, "legend": 3, "key": 2,
    "leader": 2, "rule": 1, "law": 3, "citizen": 3, "vote": 3, "right": 2,
    "responsibility": 4, "freedom": 3, "flag": 1, "president": 3, "holiday": 2,

    # Elementary (4-5)
    "government": 4, "democracy": 5, "republic": 6, "constitution": 5,
    "amendment": 6, "branch": 4, "executive": 6, "legislative": 7, "judicial": 7,
    "congress": 5, "senate": 6, "representative": 6, "supreme": 5, "court": 4,
    "elect": 5, "election": 5, "campaign": 6, "candidate": 6, "ballot": 6,
    "colony": 5, "colonist": 5, "settler": 4, "pioneer": 4, "frontier": 5,
    "revolution": 6, "independence": 5, "declaration": 6, "liberty": 5,
    "immigrant": 5, "migration": 6, "culture": 5, "tradition": 5, "custom": 5,
    "artifact": 5, "archaeology": 7, "civilization": 5, "ancient": 5, "empire": 6,
    "economy": 5, "trade": 4, "export": 5, "import": 5, "goods": 4, "services": 5,
    "supply": 5, "demand": 5, "producer": 4, "consumer": 4, "resource": 5,
    "natural": 4, "renewable": 6, "nonrenewable": 7,

    # Middle School (6-8)
    "federal": 7, "state": 3, "local": 5, "municipal": 8, "county": 5,
    "legislation": 8, "bill": 6, "veto": 7, "override": 8, "filibuster": 9,
    "treaty": 7, "diplomat": 8, "embassy": 8, "ambassador": 8, "foreign": 6,
    "policy": 7, "domestic": 7, "tariff": 8, "sanction": 9, "alliance": 8,
    "monarchy": 7, "dictator": 7, "tyranny": 8, "oligarchy": 9, "aristocracy": 9,
    "totalitarian": 10, "authoritarian": 10, "communist": 9, "socialist": 9,
    "capitalist": 9, "fascist": 10, "ideology": 10,
    "feudal": 8, "medieval": 7, "renaissance": 8, "reformation": 9,
    "enlightenment": 9, "industrial": 7, "revolution": 6, "imperialism": 9,
    "nationalism": 9, "colonialism": 9, "genocide": 9, "holocaust": 8,
    "propaganda": 8, "censorship": 8, "dissent": 9, "protest": 7, "boycott": 8,
    "segregation": 8, "integration": 8, "discrimination": 8, "prejudice": 7,
    "civil": 6, "rights": 5, "suffrage": 9, "amendment": 6,

    # High School (9-12) - Government/Civics
    "electoral": 8, "college": 5, "elector": 9, "caucus": 10, "primary": 6,
    "delegate": 8, "convention": 8, "platform": 8, "incumbent": 10, "constituent": 10,
    "lobby": 9, "lobbyist": 10, "PAC": 11, "gerrymandering": 11, "redistricting": 11,
    "apportionment": 11, "bicameral": 10, "unicameral": 11, "quorum": 11,
    "ratify": 9, "repeal": 9, "statute": 10, "ordinance": 9, "jurisdiction": 9,
    "precedent": 9, "judicial": 7, "review": 5, "appellate": 11, "plaintiff": 10,
    "defendant": 9, "prosecution": 10, "indictment": 11, "verdict": 9, "acquit": 10,
    "sovereignty": 10, "federalism": 10, "enumerated": 11, "implied": 8,
    "concurrent": 10, "reserved": 8, "supremacy": 10, "elastic": 9, "clause": 9,
    "due": 5, "process": 5, "habeas": 11, "corpus": 11, "eminent": 10, "domain": 9,

    # High School (9-12) - Economics
    "macroeconomics": 11, "microeconomics": 11, "GDP": 10, "GNP": 11, "inflation": 9,
    "deflation": 10, "recession": 9, "depression": 8, "unemployment": 9,
    "monetary": 10, "fiscal": 10, "Federal": 7, "Reserve": 8, "interest": 6,
    "bond": 8, "stock": 7, "equity": 10, "dividend": 10, "portfolio": 11,
    "commodity": 9, "futures": 10, "hedge": 11, "derivative": 11, "deficit": 9,
    "surplus": 8, "debt": 7, "budget": 7, "subsidy": 10, "monopoly": 8,
    "oligopoly": 11, "cartel": 10, "antitrust": 11, "regulation": 8,
}


# ── English Language Arts Vocabulary ────────────────────────────────────────

ELA_VOCABULARY = {
    # Reading/Comprehension (K-3)
    "letter": 0, "word": 0, "sentence": 1, "paragraph": 2, "story": 1,
    "character": 2, "setting": 2, "plot": 3, "problem": 2, "solution": 3,
    "beginning": 1, "middle": 1, "end": 1, "title": 1, "author": 2,
    "illustrator": 3, "fiction": 3, "nonfiction": 3, "fact": 2, "opinion": 3,
    "main": 2, "idea": 2, "detail": 3, "sequence": 3, "cause": 3, "effect": 3,
    "compare": 3, "contrast": 4, "summarize": 4, "predict": 3, "infer": 5,

    # Reading (4-5)
    "theme": 5, "moral": 4, "lesson": 3, "conflict": 5, "resolution": 6,
    "narrator": 5, "point": 2, "view": 2, "first": 1, "person": 2, "third": 2,
    "omniscient": 8, "limited": 6, "dialogue": 5, "monologue": 8, "flashback": 7,
    "foreshadowing": 7, "suspense": 6, "mood": 5, "tone": 6, "genre": 5,
    "myth": 5, "legend": 4, "fable": 4, "folktale": 4, "fairy": 2, "tale": 3,
    "biography": 5, "autobiography": 6, "memoir": 8, "essay": 6, "article": 5,

    # Literary Devices (6-8)
    "metaphor": 7, "simile": 6, "personification": 7, "hyperbole": 8,
    "alliteration": 7, "onomatopoeia": 7, "imagery": 7, "symbolism": 8,
    "irony": 8, "verbal": 7, "situational": 8, "dramatic": 7, "satire": 9,
    "parody": 8, "sarcasm": 7, "allegory": 9, "allusion": 9, "archetype": 10,
    "motif": 9, "foil": 9, "protagonist": 8, "antagonist": 8, "antihero": 10,
    "tragic": 8, "hero": 4, "hubris": 11, "catharsis": 11, "denouement": 10,

    # Writing (3-5)
    "draft": 4, "revise": 5, "edit": 4, "proofread": 5, "publish": 5,
    "brainstorm": 4, "outline": 5, "introduction": 5, "body": 3, "conclusion": 5,
    "thesis": 8, "topic": 5, "support": 4, "evidence": 6, "example": 4,
    "transition": 6, "coherent": 9, "cohesive": 9, "organize": 5, "structure": 6,

    # Grammar (1-5)
    "noun": 2, "verb": 2, "adjective": 3, "adverb": 4, "pronoun": 3,
    "preposition": 5, "conjunction": 5, "interjection": 5, "article": 5,
    "subject": 3, "predicate": 5, "object": 4, "direct": 4, "indirect": 5,
    "clause": 6, "phrase": 5, "dependent": 6, "independent": 6, "complex": 6,
    "compound": 5, "simple": 3, "fragment": 6, "run-on": 6, "comma": 3,
    "splice": 8, "modifier": 7, "dangling": 8, "misplaced": 7, "parallel": 7,
    "tense": 4, "past": 2, "present": 3, "future": 4, "perfect": 5, "progressive": 6,
    "active": 5, "passive": 6, "voice": 4, "singular": 4, "plural": 3,
    "possessive": 4, "contraction": 4, "apostrophe": 5, "quotation": 5, "marks": 3,

    # Rhetoric (8-12)
    "rhetoric": 9, "rhetorical": 10, "ethos": 10, "pathos": 10, "logos": 10,
    "argument": 6, "claim": 6, "counterclaim": 9, "rebuttal": 9, "concession": 9,
    "warrant": 10, "qualifier": 10, "fallacy": 10, "logical": 8, "appeal": 7,
    "emotional": 6, "ethical": 8, "persuade": 6, "convince": 7, "inform": 5,
    "entertain": 5, "audience": 5, "purpose": 5, "context": 7, "bias": 8,
    "objective": 7, "subjective": 8, "credible": 8, "reliable": 7, "valid": 8,
    "relevant": 8, "sufficient": 8, "analyze": 6, "synthesize": 9, "evaluate": 7,
    "critique": 9, "interpret": 7, "annotate": 9, "paraphrase": 8, "plagiarism": 9,
    "citation": 9, "bibliography": 8, "works": 4, "cited": 8, "MLA": 10, "APA": 10,
}
