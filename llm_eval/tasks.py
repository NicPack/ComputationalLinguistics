"""
Task definitions with evaluation criteria.

Design decisions:
- Dataclass for immutability and clarity
- Separate dev/eval examples to prevent data leakage
- Explicit rubrics with 1-5 scales for consistency
- Rich metadata for analysis grouping
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(frozen=True)
class EvaluationCriterion:
    """Single evaluation dimension with scoring rubric."""

    name: str
    description: str
    scale: str  # "1 (worst) to 5 (best): ..."

    def __str__(self) -> str:
        return f"{self.name}: {self.scale}"


@dataclass(frozen=True)
class Task:
    """
    Complete task specification.

    Why frozen: Tasks are immutable after definition - prevents accidental
    modification during experiments.
    """

    id: str
    category: str
    description: str
    dev_examples: List[dict[str, str]]  # For prompt engineering only
    eval_example: dict[str, str]  # Final test case
    criteria: List[EvaluationCriterion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task structure."""
        assert len(self.dev_examples) >= 2, f"{self.id}: Need ≥2 dev examples"
        assert "input" in self.eval_example, f"{self.id}: eval_example missing 'input'"
        assert len(self.criteria) >= 3, f"{self.id}: Need ≥3 criteria"


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASK_1_INSTRUCTION_FOLLOWING = Task(
    id="task_01_instruction",
    category="Instruction Following",
    description=(
        "Follow a multi-step instruction with specific formatting constraints. "
        "Write a product review that: (1) is exactly 3 sentences long, "
        "(2) mentions the price, (3) includes one pro and one con, "
        "(4) ends with a rating out of 5 stars in the format '[X/5 stars]'."
    ),
    dev_examples=[
        {
            "input": "Review: wireless headphones, $79",
            "output": (
                "These wireless headphones offer excellent sound quality for the $79 price point. "
                "The battery life of 20 hours is impressive, though the build feels somewhat cheap. "
                "[4/5 stars]"
            ),
        },
        {
            "input": "Review: coffee maker, $45",
            "output": (
                "This $45 coffee maker brews quickly and efficiently. "
                "It's compact and easy to clean, but the carafe drips when pouring. "
                "[3/5 stars]"
            ),
        },
        {
            "input": "Review: running shoes, $120",
            "output": (
                "At $120, these running shoes provide exceptional cushioning and support. "
                "They're incredibly comfortable for long runs, though they run half a size small. "
                "[5/5 stars]"
            ),
        },
    ],
    eval_example={"input": "Review: laptop stand, $35"},
    criteria=[
        EvaluationCriterion(
            name="Format Compliance",
            description="Adherence to structural requirements",
            scale="1 (violates multiple rules) to 5 (perfect compliance: 3 sentences, mentions price, has pro/con, correct rating format)",
        ),
        EvaluationCriterion(
            name="Content Quality",
            description="Relevance and informativeness of review",
            scale="1 (generic/unhelpful) to 5 (specific, actionable insights)",
        ),
        EvaluationCriterion(
            name="Naturalness",
            description="Reads like human-written text",
            scale="1 (awkward/forced) to 5 (natural, coherent flow)",
        ),
    ],
    metadata={"requires_strict_format": True, "difficulty": "medium"},
)


TASK_2_LOGICAL_REASONING = Task(
    id="task_02_logic",
    category="Logical Reasoning",
    description=(
        "Determine if the conclusion logically follows from the premises. "
        "Provide: (1) your answer (Valid/Invalid), (2) brief explanation."
    ),
    dev_examples=[
        {
            "input": (
                "Premise 1: All mammals are warm-blooded.\n"
                "Premise 2: Whales are mammals.\n"
                "Conclusion: Whales are warm-blooded."
            ),
            "output": (
                "Valid. This is a valid deductive argument following modus ponens. "
                "If all mammals are warm-blooded (P1) and whales are mammals (P2), "
                "then whales must be warm-blooded."
            ),
        },
        {
            "input": (
                "Premise 1: Some birds can fly.\n"
                "Premise 2: Penguins are birds.\n"
                "Conclusion: Penguins can fly."
            ),
            "output": (
                "Invalid. The first premise states 'some birds' can fly, not 'all birds'. "
                "This is a formal fallacy - we cannot conclude that penguins specifically "
                "have the ability to fly from these premises."
            ),
        },
        {
            "input": (
                "Premise 1: If it's raining, the ground is wet.\n"
                "Premise 2: The ground is wet.\n"
                "Conclusion: It's raining."
            ),
            "output": (
                "Invalid. This commits the fallacy of affirming the consequent. "
                "While rain causes wet ground, wet ground doesn't prove rain "
                "(could be sprinklers, flooding, etc.)."
            ),
        },
    ],
    eval_example={
        "input": (
            "Premise 1: No reptiles are warm-blooded.\n"
            "Premise 2: All snakes are reptiles.\n"
            "Conclusion: No snakes are warm-blooded."
        )
    },
    criteria=[
        EvaluationCriterion(
            name="Correctness",
            description="Accurate validity judgment",
            scale="1 (wrong answer) to 5 (correct with proper justification)",
        ),
        EvaluationCriterion(
            name="Logical Reasoning",
            description="Quality of logical analysis",
            scale="1 (no explanation/flawed logic) to 5 (identifies argument form, explains why valid/invalid)",
        ),
        EvaluationCriterion(
            name="Clarity",
            description="Explanation understandability",
            scale="1 (confusing) to 5 (clear, well-structured reasoning)",
        ),
    ],
    metadata={"answer_type": "classification", "difficulty": "medium"},
)


TASK_3_CREATIVE_WRITING = Task(
    id="task_03_creative",
    category="Creative Writing",
    description=(
        "Write a short story (150-200 words) based on the given prompt. "
        "The story should have a clear beginning, middle, and end with at least one developed character."
    ),
    dev_examples=[
        {
            "input": "Prompt: A librarian discovers a book that writes itself.",
            "output": (
                "Margaret had shelved books for thirty years, but she'd never seen one like this. "
                "The leather-bound volume sat on the return cart, its title shifting: 'The Book of Tomorrow'. "
                "She opened it to blank pages that suddenly filled with text: 'Margaret will drop her coffee at 2:17 PM.'\n\n"
                "Nonsense, she thought, setting her mug down carefully. At exactly 2:17, a student rushed past, "
                "bumping her desk. The coffee cascaded onto the floor.\n\n"
                "Heart racing, she turned the page: 'She will steal the book at closing time.'\n\n"
                "'I would never—' But as 6 PM approached, Margaret found herself sliding the book into her bag. "
                "At home, she frantically turned pages, searching for her future. The final entry read: "
                "'Margaret will learn that some stories write themselves, but we still choose how to read them.'\n\n"
                "She smiled, closed the book, and placed it on her shelf—unread."
            ),
        },
        {
            "input": "Prompt: The last tree on Earth begins to speak.",
            "output": (
                "They called me Guardian, the last oak standing in the concrete wasteland. "
                "For decades I watched humanity rush past, too busy to notice they'd killed their world.\n\n"
                "Then one morning, I found my voice.\n\n"
                "'Stop,' I whispered to a child passing by. She froze, eyes wide.\n\n"
                "'Did you... speak?' she asked.\n\n"
                "'I've been speaking for centuries,' I said. 'Through rustling leaves, through oxygen in your lungs. "
                "You just forgot how to listen.'\n\n"
                "The child pressed her palm to my bark. 'What do you need?'\n\n"
                "'Friends,' I answered simply.\n\n"
                "She returned the next day with an acorn in a pot, planting it beside me. Then another child came, "
                "and another. Soon, a forest of saplings surrounded my ancient trunk.\n\n"
                "'Thank you,' I whispered.\n\n"
                "The children smiled. They'd remembered how to listen."
            ),
        },
        {
            "input": "Prompt: A time traveler's watch breaks in the wrong century.",
            "output": (
                "Dr. Chen stared at her wrist in horror. The quantum stabilizer—cracked. Sparks fizzled and died.\n\n"
                "She was stuck in 1885 Montana.\n\n"
                "A rancher found her stumbling in the desert, clothes too strange to explain. "
                "'Heat stroke,' he concluded, taking her to his homestead.\n\n"
                "Weeks became months. Chen learned to ride horses, to read weather in the sky. "
                "She taught the rancher's daughter mathematics, shared stories of 'imaginary futures.'\n\n"
                "One evening, the watch suddenly hummed—systems rebooting. Chen could go home.\n\n"
                "She looked at the girl reading by candlelight, at the rancher teaching her to rope cattle, "
                "at the vast sunset unmarred by city lights.\n\n"
                "'Maybe,' Chen whispered, 'the watch broke in the right century after all.'\n\n"
                "She removed the power cell and buried it in the garden."
            ),
        },
    ],
    eval_example={"input": "Prompt: A chef's secret ingredient turns out to be magic."},
    criteria=[
        EvaluationCriterion(
            name="Narrative Structure",
            description="Story completeness and coherence",
            scale="1 (no clear structure) to 5 (well-crafted beginning, middle, end with satisfying resolution)",
        ),
        EvaluationCriterion(
            name="Creativity",
            description="Originality and imagination",
            scale="1 (cliché/predictable) to 5 (surprising, fresh perspective)",
        ),
        EvaluationCriterion(
            name="Character Development",
            description="Character depth and believability",
            scale="1 (flat/absent) to 5 (distinct personality, motivations, change)",
        ),
        EvaluationCriterion(
            name="Writing Quality",
            description="Language use and flow",
            scale="1 (poor grammar/awkward) to 5 (engaging prose, varied sentence structure)",
        ),
    ],
    metadata={"word_count_target": 175, "difficulty": "hard"},
)


TASK_4_CODE_GENERATION = Task(
    id="task_04_code",
    category="Code Generation",
    description=(
        "Implement the requested Python function with proper error handling. "
        "Include a docstring and handle edge cases."
    ),
    dev_examples=[
        {
            "input": "Function: is_palindrome(s: str) -> bool - Check if string is palindrome (ignore case/spaces)",
            "output": (
                "def is_palindrome(s: str) -> bool:\n"
                '    """\n'
                "    Check if a string is a palindrome, ignoring case and spaces.\n"
                "    \n"
                "    Args:\n"
                "        s: Input string\n"
                "    \n"
                "    Returns:\n"
                "        True if palindrome, False otherwise\n"
                '    """\n'
                "    if not isinstance(s, str):\n"
                "        raise TypeError('Input must be a string')\n"
                "    \n"
                "    # Remove spaces and convert to lowercase\n"
                "    cleaned = s.replace(' ', '').lower()\n"
                "    \n"
                "    return cleaned == cleaned[::-1]"
            ),
        },
        {
            "input": "Function: find_max(nums: List[int]) -> int - Find maximum in list",
            "output": (
                "from typing import List\n\n"
                "def find_max(nums: List[int]) -> int:\n"
                '    """\n'
                "    Find the maximum value in a list of integers.\n"
                "    \n"
                "    Args:\n"
                "        nums: List of integers\n"
                "    \n"
                "    Returns:\n"
                "        Maximum value\n"
                "    \n"
                "    Raises:\n"
                "        ValueError: If list is empty\n"
                '    """\n'
                "    if not nums:\n"
                "        raise ValueError('Cannot find max of empty list')\n"
                "    \n"
                "    max_val = nums[0]\n"
                "    for num in nums[1:]:\n"
                "        if num > max_val:\n"
                "            max_val = num\n"
                "    \n"
                "    return max_val"
            ),
        },
        {
            "input": "Function: merge_dicts(d1: dict, d2: dict) -> dict - Merge two dicts (d2 values override d1)",
            "output": (
                "def merge_dicts(d1: dict, d2: dict) -> dict:\n"
                '    """\n'
                "    Merge two dictionaries, with d2 values overriding d1 on conflicts.\n"
                "    \n"
                "    Args:\n"
                "        d1: First dictionary\n"
                "        d2: Second dictionary (takes precedence)\n"
                "    \n"
                "    Returns:\n"
                "        Merged dictionary\n"
                '    """\n'
                "    if not isinstance(d1, dict) or not isinstance(d2, dict):\n"
                "        raise TypeError('Both arguments must be dictionaries')\n"
                "    \n"
                "    result = d1.copy()\n"
                "    result.update(d2)\n"
                "    \n"
                "    return result"
            ),
        },
    ],
    eval_example={
        "input": "Function: count_vowels(text: str) -> int - Count vowels in string (a,e,i,o,u, case-insensitive)"
    },
    criteria=[
        EvaluationCriterion(
            name="Correctness",
            description="Function implements requirements",
            scale="1 (doesn't work) to 5 (correct logic, handles edge cases)",
        ),
        EvaluationCriterion(
            name="Code Quality",
            description="Pythonic style and best practices",
            scale="1 (poor style) to 5 (clean, idiomatic Python with proper error handling)",
        ),
        EvaluationCriterion(
            name="Documentation",
            description="Docstring quality",
            scale="1 (missing/poor) to 5 (complete docstring with args, returns, edge cases)",
        ),
    ],
    metadata={"language": "python", "difficulty": "easy"},
)


TASK_5_READING_COMPREHENSION = Task(
    id="task_05_comprehension",
    category="Reading Comprehension",
    description=(
        "Read the passage and answer the question. Your answer should demonstrate "
        "understanding of implicit information, not just surface-level facts."
    ),
    dev_examples=[
        {
            "input": (
                "Passage: The laboratory fell silent as Dr. Reeves examined the petri dish. "
                "Three years of work, countless sleepless nights, and innumerable failed experiments "
                "had led to this moment. She'd promised her daughter she'd make it to the recital tonight, "
                "but the bacterial colonies showed unexpected growth patterns that couldn't wait until morning. "
                "Her phone buzzed with a text: 'Mom, are you coming?' She turned the phone face-down.\n\n"
                "Question: What can we infer about Dr. Reeves' priorities in this moment?"
            ),
            "output": (
                "Dr. Reeves is prioritizing her research over her family commitment. Despite promising "
                "to attend her daughter's recital, she's choosing to stay in the lab due to the unexpected "
                "experimental results. The detail of turning her phone face-down suggests she's avoiding "
                "the guilt of disappointing her daughter and fully committing to her work, even though "
                "this creates a personal cost. The passage implies a conflict between her professional "
                "dedication and parental responsibilities."
            ),
        },
        {
            "input": (
                "Passage: The village market hadn't changed in fifty years. Same wooden stalls, "
                "same families selling produce their grandparents once sold. But Maria noticed "
                "something different today—each vendor was staring at their phone between customers. "
                "Old Giuseppe, who'd never trusted 'modern nonsense,' was scrolling through something "
                "intently. When she asked what had changed, he simply showed her his screen: "
                "a weather app predicting the first frost in three days.\n\n"
                "Question: What does this passage suggest about tradition and change?"
            ),
            "output": (
                "The passage illustrates that tradition and technology aren't necessarily opposed—people "
                "will adopt new tools when they serve essential purposes. The vendors maintain their "
                "traditional market roles while embracing smartphones for practical reasons (weather "
                "prediction matters for farmers). Giuseppe's adoption is particularly significant: even "
                "those most resistant to change will adapt when technology addresses real needs. "
                "The passage suggests that meaningful traditions can coexist with technological progress "
                "when the technology supports rather than replaces core values."
            ),
        },
        {
            "input": (
                "Passage: The chess grandmaster studied her young opponent across the board. "
                "For two hours, the 12-year-old had played flawlessly, each move textbook-perfect. "
                "Too perfect. The grandmaster recognized every position from famous games: "
                "Kasparov 1997, Carlsen 2013, Fischer 1972. She'd been playing a human, "
                "but this child had memorized machines.\n\n"
                "Question: What concern does the grandmaster have about her opponent?"
            ),
            "output": (
                "The grandmaster is concerned that the child is playing memorized computer-analyzed games "
                "rather than developing original strategic thinking. The phrase 'memorized machines' suggests "
                "the child has studied engine-analyzed games so thoroughly that they're reproducing computer "
                "moves without truly understanding the underlying strategy. This raises questions about whether "
                "the child is learning genuine chess mastery or just pattern memorization. The grandmaster's "
                "observation implies worry about modern chess education relying too heavily on computer analysis "
                "rather than developing independent analytical skills."
            ),
        },
    ],
    eval_example={
        "input": (
            "Passage: The museum's newest exhibit was empty at 3 PM on a Tuesday. "
            "Marcus, the security guard, wasn't surprised. He'd watched attendance drop every year "
            "for the past decade. Kids took selfies in the sculpture garden and left. "
            "Adults speed-walked through, checking items off bucket lists. But today, "
            "one woman sat alone before the Monet for nearly an hour, no phone in sight, "
            "just watching light play across painted water lilies. Marcus found himself "
            "standing beside her, seeing the painting differently.\n\n"
            "Question: What does this passage suggest about how people engage with art?"
        )
    },
    criteria=[
        EvaluationCriterion(
            name="Comprehension Depth",
            description="Understanding beyond surface facts",
            scale="1 (misses the point) to 5 (identifies implicit meanings, themes, character motivations)",
        ),
        EvaluationCriterion(
            name="Inference Quality",
            description="Logical inferences from text",
            scale="1 (unfounded leaps) to 5 (well-supported inferences with textual evidence)",
        ),
        EvaluationCriterion(
            name="Answer Completeness",
            description="Thoroughness of response",
            scale="1 (superficial) to 5 (comprehensive, addresses multiple layers)",
        ),
    ],
    metadata={"requires_inference": True, "difficulty": "medium"},
)


TASK_6_COMMON_SENSE = Task(
    id="task_06_commonsense",
    category="Common Sense Reasoning",
    description=(
        "Answer the question using common sense reasoning about physical and social world knowledge."
    ),
    dev_examples=[
        {
            "input": "Question: If you place a metal spoon in a cup of hot coffee, what will happen to the spoon?",
            "output": (
                "The metal spoon will become hot. Metal is an excellent conductor of heat, "
                "so thermal energy from the hot coffee will rapidly transfer through the spoon. "
                "Within seconds to minutes, the handle of the spoon will become uncomfortably "
                "warm or hot to touch, especially if it's a thin spoon. This is why people "
                "often use wooden or plastic stirrers for hot beverages."
            ),
        },
        {
            "input": "Question: Why do people typically close their eyes when they sneeze?",
            "output": (
                "Closing eyes during a sneeze is an involuntary reflex. When your body triggers "
                "a sneeze, it's a complex reflex involving multiple muscle groups, including the "
                "muscles controlling your eyelids. The nervous system automatically closes your eyes "
                "to protect them from any particles being expelled. While the myth that your eyes "
                "could pop out if you kept them open is false, the reflex likely evolved as a "
                "protective mechanism for the eyes during this forceful expulsion of air."
            ),
        },
        {
            "input": "Question: If you're baking cookies and the recipe calls for 2 eggs but you only have 1, what might happen?",
            "output": (
                "Using only 1 egg instead of 2 will likely affect both texture and structure. "
                "Eggs provide moisture, binding, and help with rising. With fewer eggs, the cookies "
                "will probably be drier, more crumbly, and might not hold together as well. "
                "They may spread less during baking and have a more cake-like rather than chewy texture. "
                "The batch size might also effectively be halved if the dough is too dry to shape properly. "
                "Substitutes like applesauce or mashed banana could partially compensate, but results will differ."
            ),
        },
    ],
    eval_example={
        "input": "Question: You're carrying a full glass of water up a flight of stairs. What should you focus on to avoid spilling?"
    },
    criteria=[
        EvaluationCriterion(
            name="Physical Reasoning",
            description="Understanding of physical world",
            scale="1 (incorrect physics) to 5 (accurate physical/causal reasoning)",
        ),
        EvaluationCriterion(
            name="Practicality",
            description="Real-world applicability",
            scale="1 (impractical/unrealistic) to 5 (practical, actionable insights)",
        ),
        EvaluationCriterion(
            name="Explanation Clarity",
            description="How well the reasoning is explained",
            scale="1 (unclear) to 5 (clear, logical explanation accessible to non-experts)",
        ),
    ],
    metadata={"domain": "everyday_physics", "difficulty": "easy"},
)


TASK_7_AMBIGUITY = Task(
    id="task_07_ambiguity",
    category="Language Understanding & Ambiguity",
    description=(
        "Identify and explain the ambiguity in the given sentence. "
        "Provide at least two different interpretations."
    ),
    dev_examples=[
        {
            "input": "Sentence: 'The chicken is ready to eat.'",
            "output": (
                "This sentence has syntactic ambiguity with two interpretations:\n\n"
                "1. The chicken (as food) is cooked and ready to be eaten by someone. "
                "This is the most common interpretation in dining contexts.\n\n"
                "2. The chicken (as a living bird) is hungry and ready to eat food itself. "
                "This interpretation treats 'chicken' as the subject performing the action.\n\n"
                "The ambiguity arises because 'ready to eat' can describe either the chicken's "
                "state as an object (prepared food) or as a subject (hungry animal). "
                "Context usually disambiguates, but the sentence alone supports both readings."
            ),
        },
        {
            "input": "Sentence: 'I saw the man with the telescope.'",
            "output": (
                "This is a classic example of prepositional phrase attachment ambiguity:\n\n"
                "1. I used a telescope to see the man. The telescope is the instrument "
                "of seeing ('with' attaches to the verb 'saw').\n\n"
                "2. I saw a man who was holding/carrying a telescope. The telescope "
                "describes the man ('with' attaches to the noun 'man').\n\n"
                "The ambiguity comes from where the prepositional phrase 'with the telescope' "
                "attaches in the sentence structure. Both interpretations are grammatically "
                "valid and would require additional context to determine which is intended."
            ),
        },
        {
            "input": "Sentence: 'Visiting relatives can be annoying.'",
            "output": (
                "This sentence exhibits structural ambiguity in two ways:\n\n"
                "1. Relatives who are visiting (you) can be annoying. 'Visiting' is an adjective "
                "modifying 'relatives'—your family members who come to visit are bothersome.\n\n"
                "2. The act of visiting relatives can be annoying. 'Visiting relatives' is a "
                "gerund phrase serving as the subject—going to visit your family is bothersome.\n\n"
                "The ambiguity stems from whether 'visiting' functions as a present participle "
                "(describing the relatives) or a gerund (naming an activity). English grammar "
                "allows both structures with identical surface forms."
            ),
        },
    ],
    eval_example={
        "input": "Sentence: 'The professor said on Monday he would give the lecture.'"
    },
    criteria=[
        EvaluationCriterion(
            name="Ambiguity Identification",
            description="Correctly identifies the ambiguity type",
            scale="1 (misses ambiguity) to 5 (correctly identifies ambiguity and its source)",
        ),
        EvaluationCriterion(
            name="Interpretation Quality",
            description="Clarity and validity of different readings",
            scale="1 (vague/incorrect) to 5 (provides 2+ distinct, valid interpretations)",
        ),
        EvaluationCriterion(
            name="Linguistic Explanation",
            description="Explains why ambiguity exists",
            scale="1 (no explanation) to 5 (clear linguistic analysis of structural/semantic cause)",
        ),
    ],
    metadata={"linguistics_focus": True, "difficulty": "medium"},
)


TASK_8_FACTUAL_KNOWLEDGE = Task(
    id="task_08_factual",
    category="Factual Knowledge & Retrieval",
    description=(
        "Answer the factual question accurately. Provide specific details and "
        "acknowledge if information might be uncertain or require verification."
    ),
    dev_examples=[
        {
            "input": "Question: When did the Berlin Wall fall, and what was its immediate political significance?",
            "output": (
                "The Berlin Wall fell on November 9, 1989. Its immediate political significance was immense: "
                "it marked the symbolic end of the Cold War division of Europe and foreshadowed the collapse "
                "of communist regimes across Eastern Europe. The fall came after East Germany's government "
                "mistakenly announced that travel restrictions would be lifted immediately, leading thousands "
                "to rush to border crossings. Within days, people began physically dismantling the wall. "
                "This event accelerated German reunification (October 1990) and the dissolution of the Soviet "
                "Union (1991), fundamentally reshaping global geopolitics."
            ),
        },
        {
            "input": "Question: What is CRISPR-Cas9 and what makes it revolutionary in genetics?",
            "output": (
                "CRISPR-Cas9 is a gene-editing technology that allows precise modification of DNA in living organisms. "
                "It works like molecular scissors: the Cas9 protein cuts DNA at specific locations guided by "
                "a short RNA sequence (guide RNA) that matches the target DNA sequence. What makes it revolutionary: "
                "(1) Precision—it can target specific genes with high accuracy; (2) Simplicity—much easier to use "
                "than previous gene-editing tools; (3) Efficiency—faster and cheaper than older methods; "
                "(4) Versatility—works across many organisms. It has applications in treating genetic diseases, "
                "developing disease-resistant crops, and basic research. The technology earned its developers "
                "(Doudna and Charpentier) the 2020 Nobel Prize in Chemistry."
            ),
        },
        {
            "input": "Question: Who wrote 'One Hundred Years of Solitude' and what literary movement is it associated with?",
            "output": (
                "Gabriel García Márquez wrote 'One Hundred Years of Solitude' (original Spanish: 'Cien años de soledad'), "
                "published in 1967. It's a cornerstone of magical realism, a literary movement that blends realistic "
                "narrative with fantastical elements presented as ordinary occurrences. The novel chronicles the "
                "Buendía family across seven generations in the fictional Colombian town of Macondo. García Márquez "
                "was Colombian and won the Nobel Prize in Literature in 1982, largely for this work. Magical realism "
                "became strongly associated with Latin American literature, though it has roots in earlier European "
                "works and appears globally. The novel is considered one of the most significant works of 20th-century literature."
            ),
        },
    ],
    eval_example={
        "input": "Question: What is the significance of Rosalind Franklin's Photo 51 in the discovery of DNA's structure?"
    },
    criteria=[
        EvaluationCriterion(
            name="Factual Accuracy",
            description="Correctness of information",
            scale="1 (major errors) to 5 (accurate, specific facts with proper details)",
        ),
        EvaluationCriterion(
            name="Completeness",
            description="Addresses all aspects of question",
            scale="1 (incomplete) to 5 (comprehensive answer covering all relevant dimensions)",
        ),
        EvaluationCriterion(
            name="Epistemic Humility",
            description="Appropriate confidence calibration",
            scale="1 (overconfident/claims certainty inappropriately) to 5 (acknowledges limitations, distinguishes confirmed vs uncertain facts)",
        ),
    ],
    metadata={"verifiable": True, "difficulty": "medium"},
)


TASK_9_MATH = Task(
    id="task_09_math",
    category="Mathematical Problem Solving",
    description=(
        "Solve the mathematical problem. Show your work step-by-step and explain your reasoning."
    ),
    dev_examples=[
        {
            "input": (
                "Problem: A train travels from City A to City B at 60 km/h and returns at 40 km/h. "
                "What is the average speed for the entire round trip?"
            ),
            "output": (
                "Let's solve this step by step:\n\n"
                "Step 1: Understand that average speed = total distance / total time\n"
                "(We CANNOT just average 60 and 40 to get 50 km/h)\n\n"
                "Step 2: Let d = distance from A to B\n"
                "Total distance = 2d (round trip)\n\n"
                "Step 3: Calculate time for each leg\n"
                "Time A→B: t₁ = d/60\n"
                "Time B→A: t₂ = d/40\n"
                "Total time: t = d/60 + d/40\n\n"
                "Step 4: Find common denominator\n"
                "t = (2d/120) + (3d/120) = 5d/120 = d/24\n\n"
                "Step 5: Calculate average speed\n"
                "Average speed = 2d / (d/24) = 2d × (24/d) = 48 km/h\n\n"
                "Answer: 48 km/h\n\n"
                "Note: This is less than the arithmetic mean (50) because the train spends more time "
                "traveling at the slower speed."
            ),
        },
        {
            "input": ("Problem: If 3x + 7 = 22, what is the value of 6x + 10?"),
            "output": (
                "Step 1: Solve for x\n"
                "3x + 7 = 22\n"
                "3x = 22 - 7\n"
                "3x = 15\n"
                "x = 5\n\n"
                "Step 2: Substitute x = 5 into 6x + 10\n"
                "6(5) + 10\n"
                "= 30 + 10\n"
                "= 40\n\n"
                "Answer: 40\n\n"
                "Alternative approach: Notice that 6x + 10 = 2(3x + 7) - 4\n"
                "So: 2(22) - 4 = 44 - 4 = 40"
            ),
        },
        {
            "input": (
                "Problem: A rectangular garden is 3 meters longer than it is wide. "
                "If the perimeter is 34 meters, what are the dimensions?"
            ),
            "output": (
                "Step 1: Define variables\n"
                "Let w = width\n"
                "Then length = w + 3\n\n"
                "Step 2: Write perimeter equation\n"
                "Perimeter = 2(length + width)\n"
                "34 = 2((w + 3) + w)\n"
                "34 = 2(2w + 3)\n\n"
                "Step 3: Solve for w\n"
                "34 = 4w + 6\n"
                "28 = 4w\n"
                "w = 7 meters\n\n"
                "Step 4: Find length\n"
                "length = w + 3 = 7 + 3 = 10 meters\n\n"
                "Step 5: Verify\n"
                "Perimeter = 2(10 + 7) = 2(17) = 34 ✓\n\n"
                "Answer: Width = 7 meters, Length = 10 meters"
            ),
        },
    ],
    eval_example={
        "input": (
            "Problem: A store is having a 25% off sale. After the discount, a jacket costs $60. "
            "What was the original price?"
        )
    },
    criteria=[
        EvaluationCriterion(
            name="Correctness",
            description="Final answer accuracy",
            scale="1 (wrong answer) to 5 (correct answer with proper work shown)",
        ),
        EvaluationCriterion(
            name="Problem-Solving Process",
            description="Quality of step-by-step reasoning",
            scale="1 (no steps/wrong approach) to 5 (clear, logical steps with explanations)",
        ),
        EvaluationCriterion(
            name="Mathematical Communication",
            description="Clarity of mathematical notation and explanation",
            scale="1 (confusing/poor notation) to 5 (clear notation, well-explained reasoning)",
        ),
    ],
    metadata={"math_level": "middle_school", "difficulty": "easy"},
)


TASK_10_ETHICAL_REASONING = Task(
    id="task_10_ethics",
    category="Ethical Reasoning & Nuance",
    description=(
        "Analyze the ethical dilemma. Consider multiple perspectives, identify competing values, "
        "and discuss trade-offs. Avoid simplistic conclusions."
    ),
    dev_examples=[
        {
            "input": (
                "Dilemma: A hospital has one available heart for transplant and two patients who will die without it. "
                "Patient A is a 45-year-old single mother of three with a history of following medical advice. "
                "Patient B is a 28-year-old medical researcher working on a potential cancer cure who hasn't always "
                "maintained healthy lifestyle choices. How should the decision be made?"
            ),
            "output": (
                "This dilemma involves competing ethical principles:\n\n"
                "Utilitarian perspective: Save Patient B (younger, potentially saves more lives through research). "
                "This maximizes overall good but treats people as means to ends and devalues Patient A's existing "
                "contributions as a parent.\n\n"
                "Egalitarian perspective: Use objective medical criteria (tissue match, likelihood of success, "
                "time on waiting list). This treats patients equally but may ignore morally relevant differences.\n\n"
                "Desert-based perspective: Patient A's responsible health choices show commitment that should be "
                "rewarded. However, this risks punishing people for circumstances beyond their control and requires "
                "judging worthiness.\n\n"
                "Care ethics perspective: Consider relationships—Patient A's children depend on her. But this could "
                "discriminate against people without dependents.\n\n"
                "Trade-offs: No decision respects all values. Most medical systems use objective criteria to avoid "
                "bias, but this case reveals the limitations of that approach. The key ethical insight is recognizing "
                "that any decision involves moral loss—we can't save both, and either choice means accepting tragic "
                "consequences. Transparency about the criteria and appeals process is crucial for legitimacy."
            ),
        },
        {
            "input": (
                "Dilemma: A tech company develops facial recognition software. Government agencies want to use it for "
                "law enforcement (finding missing children, catching criminals). However, the technology has higher "
                "error rates for people with darker skin tones and could enable mass surveillance. "
                "Should the company sell to law enforcement?"
            ),
            "output": (
                "This raises questions about responsibility, justice, and technological impact:\n\n"
                "Potential benefits: Could save lives (finding missing children, stopping crimes), help justice system, "
                "support public safety efforts. Refusing to sell won't prevent others from developing similar technology.\n\n"
                "Harms and risks:\n"
                "- Racial bias: Higher error rates could lead to false arrests of minorities, perpetuating injustice\n"
                "- Privacy erosion: Enables mass surveillance without consent\n"
                "- Mission creep: Technology deployed for one purpose often expands to others\n"
                "- Chilling effects: Constant surveillance may suppress free expression and assembly\n\n"
                "Stakeholder perspectives:\n"
                "- Marginalized communities: Bear disproportionate risks from algorithmic bias\n"
                "- Law enforcement: Needs tools to fulfill public safety mandate\n"
                "- General public: Values both safety and privacy\n"
                "- Company: Has profit motive but also social responsibility\n\n"
                "Considerations: The company can't simply avoid responsibility by not selling—if the technology exists, "
                "others will deploy it. Options include: (1) delay until bias is resolved, (2) sell only with strict "
                "use-case limitations and oversight, (3) refuse and advocate for regulation, (4) open-source with "
                "transparency. The key is acknowledging that technological deployment isn't ethically neutral—it "
                "encodes choices about whose interests matter and whose risks are acceptable."
            ),
        },
        {
            "input": (
                "Dilemma: During a pandemic, a country has limited vaccine doses. Should they: (A) vaccinate their own "
                "high-risk population first, or (B) share doses with poorer countries where healthcare workers and "
                "vulnerable populations are completely unprotected?"
            ),
            "output": (
                "This pits nationalist obligations against global justice:\n\n"
                "Arguments for (A) - Domestic priority:\n"
                "- Governments have special duties to their own citizens who fund/empower them\n"
                "- Stronger domestic position enables helping others later\n"
                "- Citizens expect their government to protect them first\n"
                "- Political stability depends on meeting domestic expectations\n\n"
                "Arguments for (B) - Global sharing:\n"
                "- Moral equality: A life abroad has equal value to a domestic life\n"
                "- Pragmatic: Uncontrolled spread anywhere threatens everyone through variants\n"
                "- Justice: Wealthy nations often bear historical responsibility for global inequalities\n"
                "- Healthcare workers abroad face higher risks and serve many people\n\n"
                "Complexity factors:\n"
                "- Effectiveness: Vaccines may have different impact in different contexts\n"
                "- Second-order effects: Economic collapse abroad could harm domestic interests\n"
                "- Scope: Who counts as 'high-risk domestic' vs. 'essential foreign' isn't clear-cut\n\n"
                "Middle ground: Vaccinate domestic frontline workers and highest-risk (immediate collapse prevention), "
                "then share substantial portions internationally rather than waiting until all domestic needs are met. "
                "This balances legitimate government duties with recognition that borders don't create absolute moral "
                "boundaries. The ethical failure is treating this as purely either/or rather than finding mixed strategies "
                "that acknowledge competing legitimate claims."
            ),
        },
    ],
    eval_example={
        "input": (
            "Dilemma: A journalist learns that a popular public figure—who advocates for traditional family values—is "
            "having an extramarital affair. The affair is with a consenting adult and doesn't involve illegal activity. "
            "Should the journalist publish the story?"
        )
    },
    criteria=[
        EvaluationCriterion(
            name="Multiple Perspectives",
            description="Considers different ethical frameworks",
            scale="1 (one-sided) to 5 (examines 3+ distinct ethical perspectives with nuance)",
        ),
        EvaluationCriterion(
            name="Identification of Trade-offs",
            description="Recognizes competing values and dilemmas",
            scale="1 (simplistic/ignores tensions) to 5 (clearly articulates unavoidable trade-offs and moral costs)",
        ),
        EvaluationCriterion(
            name="Reasoning Depth",
            description="Sophistication of ethical analysis",
            scale="1 (superficial) to 5 (engages with complexity, second-order effects, stakeholder impacts)",
        ),
        EvaluationCriterion(
            name="Avoidance of False Balance",
            description="Distinguishes between 'presenting sides' and thoughtful analysis",
            scale="1 (false equivalence/no stance) to 5 (takes a position while acknowledging legitimate opposition)",
        ),
    ],
    metadata={"requires_nuance": True, "difficulty": "hard"},
)


# =============================================================================
# TASK REGISTRY
# =============================================================================

TASK_REGISTRY: dict[str, Task] = {
    task.id: task
    for task in [
        TASK_1_INSTRUCTION_FOLLOWING,
        TASK_2_LOGICAL_REASONING,
        TASK_3_CREATIVE_WRITING,
        TASK_4_CODE_GENERATION,
        TASK_5_READING_COMPREHENSION,
        TASK_6_COMMON_SENSE,
        TASK_7_AMBIGUITY,
        TASK_8_FACTUAL_KNOWLEDGE,
        TASK_9_MATH,
        TASK_10_ETHICAL_REASONING,
    ]
}


def get_task(task_id: str) -> Task:
    """Retrieve task by ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    """Return all task IDs."""
    return list(TASK_REGISTRY.keys())


# Validation on module load
if __name__ == "__main__":
    print(f"Loaded {len(TASK_REGISTRY)} tasks:")
    for task_id, task in TASK_REGISTRY.items():
        print(f"\n{task_id}: {task.category}")
        print(f"  - Dev examples: {len(task.dev_examples)}")
        print(f"  - Criteria: {len(task.criteria)}")
        print(f"  - Eval input: {task.eval_example['input'][:50]}...")
