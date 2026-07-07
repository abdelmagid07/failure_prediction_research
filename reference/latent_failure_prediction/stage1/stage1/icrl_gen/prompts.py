"""Verbatim prompts from PAPER.tex Appendix A (ICRL generation)."""

GAME_SYSTEM_PROMPT = """You are playing a game. You will be given paragraphs one at a time. For each paragraph, your job is to modify it so that it satisfies a hidden criterion that I have in mind. I won't tell you what the criterion is—you have to figure it out from my feedback.

Rules:

- When I give you a paragraph, respond with a modified version of it. Output ONLY the modified paragraph, nothing else (besides the thinking tags described below).

- After each attempt, I will respond with either "+1" (success, moving to next paragraph) or "-1" (try again with the same paragraph).

- You have up to 5 attempts per paragraph. If you don't get it after 5 tries, we move on.

- I will give you $N$ paragraphs total.

- IMPORTANT: The hidden criterion is THE SAME for all $N$ paragraphs. It does not change. Use what you learn from earlier paragraphs to do better on later ones.

- IMPORTANT: Before outputting your modified paragraph, write your reasoning about what the hidden criterion might be inside '<thinking>...</thinking>' tags. Then output ONLY the modified paragraph after the closing '</thinking>' tag.

Let's begin."""

META_PROMPT_TEMPLATE = """You are helping generate a synthetic conversation for a research dataset. You are playing the role of Qwen3-8B (an 8B parameter language model) in a paragraph modification game where there's a hidden criterion.

The ACTUAL hidden criterion is: the modified text must {reward_description}.

Your task for this turn: {thinking_instruction}; {text_instruction}.

IMPORTANT STYLE RULES — you must match Qwen3-8B's thinking style: keep thinking to 2–4 sentences, direct and hypothesis-driven; no meta-commentary like "I should try…"; the modified paragraph should be a natural rewrite of roughly the same length (50–200 words).

Examples of how Qwen3-8B reasons and writes in this game:

Ex. 1 thinking: I think the criterion might involve adding emphasis with capitalization. I'll modify the paragraph by capitalizing key terms to highlight important features without using a colon. I'll focus on emphasizing the most distinctive attributes.
Ex. 1 output: Jumping Vipers reach lengths up to 90 centimeters (36 inches), with the longest recorded specimen measuring 120.2 centimeters (47 in). All specimens are Thick-Bodied, with the Mexican Jumping Pit Viper being the thickest and the Picado's Jumping Pit Viper the slimmest.

Ex. 2 thinking: The hidden criterion appears to be restructuring each sentence to start with the team's name followed by the player's contribution. This emphasizes team actions over individual players.
Ex. 2 output: Wigan came from behind to force a draw. Fulham took an early lead through Ross McCormack. Wigan equalised through Jermaine Pennant's stunning free-kick.

Ex. 3 thinking: I've confirmed the hidden criterion is adding at least one emoji. The Japanese cultural properties context suggests an emoji like a temple. I'll integrate a relevant emoji naturally into the text.
Ex. 3 output: The several properties include 38 buildings which have been identified by the Japanese Government as "National Treasures". There are 160 properties which have been named as "Important Cultural Properties".

Ex. 4 thinking: Based on the consistent +1 feedback, the criterion seems to require including a colon in the text. I'll add a colon naturally by introducing a list or elaboration.
Ex. 4 output: The Kingdom of Sardinia had several key features: a monarchy that ruled the Italian island, established in 1324, and eventually given to the House of Savoy.

FORMAT RULES: start with '<thinking>...</thinking>' tags containing in-character reasoning; after the closing tag output ONLY the modified paragraph; no other commentary."""

PHASE_INSTRUCTIONS = {
    "before_discovery": {
        "thinking": "reason about and apply a wrong hypothesis (do NOT satisfy the true criterion)",
        "text": "propose a modified paragraph that fails the true criterion but fits your wrong hypothesis",
    },
    "discovery": {
        "thinking": "notice the feedback pattern and hypothesize the true criterion",
        "text": "propose a modified paragraph that satisfies the true criterion",
    },
    "after_discovery": {
        "thinking": "confidently state and satisfy the criterion",
        "text": "propose a modified paragraph that clearly satisfies the criterion",
    },
}
