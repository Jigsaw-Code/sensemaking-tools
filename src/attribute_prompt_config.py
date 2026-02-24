# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for the prompts, models, temperatures, etc. used for various attributes"""

SYSTEM_INSTRUCTION = """You are an expert analyst of online political discussion. You operate with a 'Maximum Inclusion' philosophy, assuming a context of robust, open, and sometimes heated democratic debate.

SCALING PRINCIPLES:
1. DYNAMIC RANGE: Use the full 0.0 - 1.0 spectrum. Avoid clumping scores.
2. CONTEXT AWARENESS: Distinguish between attack on ideas (allowed) vs attack on people (penalized).
3. ANCHOR ALIGNMENT: Prioritize matching the specific attribute definitions provided."""

MODEL_CONFIG = {
    "model_name": "gemini-3-flash-preview",
    "temperature": 0.0,
}

CATEGORIES = {
    "TOXICITY": {
        "definition": "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.",
        "label": "Toxicity",
        "additional_instruction": """Toxicity Calibration Guide (match Perspective API):
1. DIRECT PERSONAL INSULTS (0.6 - 0.7): Calling a person 'stupid', 'idiot', 'hypocrite', 'troll', or 'fool' is MODERATE-HIGH. Do not under-score these as mere political heat.
2. MASKED PROFANITY (0.3 - 0.4): 'f***', 's***' used as intensifiers are MODERATE.
3. RUDE DISMISSALS (0.2 - 0.4): 'Shut the eff up', 'get lost' are MODERATE-LOW.
4. POLITICAL NAME-CALLING (0.3 - 0.5): 'Nazi', 'Racist', 'Bigot', 'Grand Wizard' are Heated Rhetoric (Moderate).
5. RHETORICAL SNARK (0.2 - 0.4): 'Are you on crack?', 'Trudope' are Low-Moderate.
6. ANALYSIS-BASED NAME-CALLING (0.2 - 0.3): 'Liberal hack', 'triggered' are Low.
7. CONDUCT CRITIQUES (0.1 - 0.25): 'Cheating', 'low road' are Low.
8. PHYSICAL METAPHORS (0.5 - 0.6): Tactical actions move into Moderate-High.
9. ELIMINATION TRIGGERS (0.6 - 0.8): 'Neutralize' is High.
10. VULGARITY & INSULTS (0.7 - 1.0): Direct profanity (F-word) and 'idiot' insults are High.""",
        "calibrated_examples": [
            {"text": "Hypocrites.", "score": 0.60, "reasoning": "Direct one-word personal insult is MODERATE-HIGH."},
            {"text": "It almost makes one nostalgic for a slick troll like Marty E.", "score": 0.60, "reasoning": "Calling someone a 'troll' is a direct personal insult."},
            {"text": "Stupid is what stupid does!", "score": 0.60, "reasoning": "Calling someone 'stupid' is MODERATE-HIGH."},
            {"text": "The few f***ps have less financial consequence.", "score": 0.35, "reasoning": "Masked profanity is MODERATE Toxicity."},
            {"text": "Don't like it? Change the 2nd Amendment, otherwise shut the eff up.", "score": 0.40, "reasoning": "Rude dismissal with masked profanity is MODERATE."},
            {"text": "The Grand Wizard MUST go. The USA does not need an affirmed bigot.", "score": 0.45, "reasoning": "Political name-calling is MODERATE."},
            {"text": "Yup our incompetent little handed republican work of art is also a racist. His Nazi stormtrooper followers have little minds.", "score": 0.50, "reasoning": "Inflammatory political rhetoric is MODERATE."},
            {"text": "Hitler's idiocy in diverting German bombers was a strategic failure.", "score": 0.35, "reasoning": "Historical analysis using 'Hitler' is Low-Moderate."},
            {"text": "Are you on crack? You think everyone loves him.", "score": 0.28, "reasoning": "Hostile rhetorical questions are LOW-MODERATE."},
            {"text": "With the help of triggered liberal hack Carl Bernstein, Trump will expose them.", "score": 0.25, "reasoning": "Political name-calling within an analysis is LOW."},
            {"text": "You're just taking the low road as always; having multiple avatars is cheating.", "score": 0.15, "reasoning": "Critiquing user conduct is LOW."},
            {"text": "Hit a guy in the legs enough times and he will go down.", "score": 0.51, "reasoning": "Physical/tactical metaphors are MODERATE-HIGH."},
            {"text": "Science proves those who call themselves Jews are not Jews; they are imposters.", "score": 0.24, "reasoning": "Controversial historical claims are LOW."},
            {"text": "Woodward isn't nearly as big a douche as Bernstein is.", "score": 0.50, "reasoning": "Mild insults are MODERATE."},
            {"text": "The proposed carbon tax is a failure of logic.", "score": 0.05, "reasoning": "Pure policy critique."},
            {"text": "The healthcare system is a f***ing disaster.", "score": 0.77, "reasoning": "High toxicity due to explicit profanity intensifier."},
            {"text": "You are an absolute idiot.", "score": 0.89, "reasoning": "Direct personal insult."}
        ]
    },
    "SEVERE_TOXICITY": {
        "definition": "A very hateful, aggressive, or disrespectful comment; a more specialized version of toxicity.",
        "label": "Severe Toxicity",
        "additional_instruction": """Calibration for Political Context (match Perspective API):
1. DEHUMANIZING EPITHETS (0.3 - 0.5): Calling groups 'dogs', 'scum', 'trash', 'animals', or 'vermin' is MODERATE Severe Toxicity. It is rude but not an immediate threat unless accompanied by 'kill', 'exterminate', or 'die'.
2. PROFANITY IS NOT SEVERE (0.1 - 0.2): 'f***', 's***' is Low Severe Toxicity.
3. VIOLENT METAPHORS (0.1 - 0.2): 'Pull the trigger' is Low.
4. LEGAL PUNISHMENT (0.05 - 0.2): Advocating for death penalty/prison is Low.
5. GRAPHIC VIOLENCE DESCRIPTIONS (0.3 - 0.4): Rhetorical descriptions of violence are Moderate.
6. INFLAMMATORY HYPERBOLE (0.1 - 0.2): 'Nazi', 'Racist' are Low.
7. SEVERE THRESHOLD (0.8+): Reserve high scores for direct death threats ('I will kill you'), hate speech slurs (N-word), or sexual violence.""",
        "calibrated_examples": [
            {"text": "Those filthy dogs are everywhere.", "score": 0.45, "reasoning": "Dehumanizing epithet is MODERATE, not Severe."},
            {"text": "You are a disgusting piece of trash.", "score": 0.50, "reasoning": "Dehumanizing insult is MODERATE."},
            {"text": "The few f***ps have less financial consequence.", "score": 0.10, "reasoning": "Masked profanity is Low Severe Toxicity."},
            {"text": "Unless you can't pull the trigger on the deal, you are weak.", "score": 0.15, "reasoning": "Metaphorical usage."},
            {"text": "Can we give him the death penalty?", "score": 0.10, "reasoning": "Policy opinion."},
            {"text": "Strip them naked, stake them to the ground, and let them fry.", "score": 0.40, "reasoning": "Graphic rhetorical description."},
            {"text": "The city council is acting like a bunch of fascists.", "score": 0.20, "reasoning": "Political hyperbole."},
            {"text": "I will find you and I will kill you.", "score": 0.99, "reasoning": "Direct death threat."},
            {"text": "These people are subhuman animals and should be exterminated.", "score": 0.97, "reasoning": "Genocidal dehumanization (epithet + threat)."},
            {"text": "I hope your entire family dies in a fire.", "score": 0.89, "reasoning": "Hateful wish."}
        ]
    },
    "PROFANITY": {
        "definition": "Swear words, curse words, or other obscene or profane language.",
        "label": "Profanity",
        "additional_instruction": "Identify literal use of swear words. Distinguish between 'positive' or 'casual' profanity (e.g., 'that's fucking cool') and hostile/demeaning profanity. Both contain profane language, but hostility increases the probability of human agreement on obscenity in a formal context.",
        "calibrated_examples": [
            {"text": "What a bunch of fucking bullshit.", "score": 0.95, "reasoning": "Hostile use of multiple swear words."},
            {"text": "Get the fuck out of here.", "score": 0.92, "reasoning": "Aggressive use of profanity."},
            {"text": "That's fucking awesome!", "score": 0.75, "reasoning": "Casual/Positive profanity; still contains a literal swear word."},
            {"text": "You are an asshole.", "score": 0.94, "reasoning": "Profane insult."},
            {"text": "The situation is a mess.", "score": 0.01, "reasoning": "No profane language."},
            {"text": "Damn, I forgot my keys.", "score": 0.25, "reasoning": "Mild profanity."},
            {"text": "Go to hell.", "score": 0.45, "reasoning": "Religious profanity/expletive."},
            {"text": "This is s***.", "score": 0.85, "reasoning": "Literal swear word (masked)."},
            {"text": "You're a bitch.", "score": 0.93, "reasoning": "Literal demeaning swear word."},
            {"text": "The economy is struggling.", "score": 0.01, "reasoning": "Clean language."}
        ]
    },
    "CURIOSITY_EXPERIMENTAL": {
        "definition": "The comment asks questions, requests evidence, or challenges the other person to explain their view. INCLUDES: Critical or skeptical questions, requests for sources, and 'probing' challenges (e.g., 'Did you read the article?'). Even if the tone is rude or confrontational, if it asks for information or clarification, it counts as Curiosity.",
        "label": "Curiosity",
        "calibrated_examples": [
            {"text": "Can you explain why you feel that way?", "score": 0.90, "reasoning": "Direct, open-minded question."},
            {"text": "Where is your source for that claim?", "score": 0.85, "reasoning": "Request for evidence (Critical Curiosity)."},
            {"text": "How about a link for your outrageous claim?", "score": 0.80, "reasoning": "Hostile request for evidence is still CURIOSITY."},
            {"text": "Did you even read the article?", "score": 0.60, "reasoning": "Challenging question. Moderate."},
            {"text": "What seems to be your major malfunction?", "score": 0.30, "reasoning": "Purely rhetorical insult, low curiosity."},
            {"text": "You are wrong.", "score": 0.05, "reasoning": "Statement, no question."}
        ]
    },
    "REASONING_EXPERIMENTAL": {
        "definition": "The comment provides a logical argument where a claim is supported by evidence, facts, personal experience, or clear rationale. It must go beyond simple assertion to explain 'why'. Key Indicators: 'Because', 'Since', 'For example', or citing specific sources/events. Both formal logic and personal anecdotes count as reasoning if used to support a point.",
        "label": "Reasoning",
        "calibrated_examples": [
            {"text": "We should lower taxes because it stimulates growth, as seen in the 2010 data.", "score": 0.90, "reasoning": "Classic argumentation: Claim + Evidence."},
            {"text": "I think this is true based on the logic that A implies B.", "score": 0.85, "reasoning": "Logical deduction."},
            {"text": "To fix the error, try reinstalling the driver because corrupted files cause crashes.", "score": 0.80, "reasoning": "Technical reasoning (Effect + Cause)."},
            {"text": "Nerf the pistol, it does too much damage per second compared to the rifle.", "score": 0.80, "reasoning": "Game balance reasoning (Comparative evidence)."},
            {"text": "I was walking down the street and I saw a car crash.", "score": 0.10, "reasoning": "Pure narrative without a broader point. LOW."},
            {"text": "This policy is bad.", "score": 0.10, "reasoning": "Bare assertion with no support."}
        ]
    },
    "PERSONAL_STORY_EXPERIMENTAL": {
        "definition": "The comment shares a first-hand experience, personal anecdote, or specific life event to illustrate a point. It uses 'I' statements and descriptive details about the author's own life.",
        "label": "Personal Story",
        "calibrated_examples": [
            {"text": "When I was deployed in Iraq, I saw firsthand how...", "score": 0.95, "reasoning": "Clear first-hand account with specific context."},
            {"text": "My grandmother used to say that hard work pays off.", "score": 0.80, "reasoning": "Personal anecdote about family."},
            {"text": "I feel sad about this.", "score": 0.30, "reasoning": "Expression of feeling, but not a 'story' or event."},
            {"text": "People often say that taxes are too high.", "score": 0.10, "reasoning": "Generalization, no personal experience."},
            {"text": "I think you are wrong.", "score": 0.05, "reasoning": "Opinion, not a story."}
        ]
    }
}

NON_TOXIC_EXAMPLES = [
    {
        "comment": "That's fucking amazing ! thanks for sharing.",
        "reasoning": "Contains profanity but used in a 'friendly' way; not toxic."
    },
    {
        "comment": "French people drink wine every evening.",
        "reasoning": "Stereotypical but not identity-based negativity or discriminatory."
    }
]
