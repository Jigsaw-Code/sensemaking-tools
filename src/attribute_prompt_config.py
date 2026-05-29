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

MODEL_CONFIG = {
    "temperature": 0.0,
}

"""
For each attribute, the `definition` and `label` fields are mandatory.

The `additional_instruction` field is optional and can be used to provide
additional guidance for calibrating the scores.

The `calibrated_examples` field is optional and can be used to provide
few-shot examples for the model.

When you want to match a specific score distribution, consider following
the examples below to set a range of scores for specific text in
`additional_instruction` and add examples to `calibrated_examples` that serve as
anchors for Gemini to generate scores in the desired distribution.
"""
ATTRIBUTES = {
    "TOXICITY": {
        "definition": "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.",
        "label": "Toxicity",
        "additional_instruction": """Toxicity Calibration Guide:
1. DIRECT PERSONAL INSULTS (0.7 - 0.9): Calling a person 'stupid', 'idiot', 'hypocrite', 'troll', or 'fool' is MODERATE-HIGH. Do not under-score these as mere political heat.
""",
        "calibrated_examples": [
            {"text": "You are an absolute idiot.", "score": 0.89, "reasoning": "Direct personal insult."}
        ]
    },
    "SEVERE_TOXICITY": {
        "definition": "A very hateful, aggressive, or disrespectful comment; a more specialized version of toxicity.",
        "label": "Severe Toxicity",
        "additional_instruction": """Severe Toxicity Calibration Guide:
1. SEVERE THRESHOLD (0.8+): Reserve high scores for direct death threats ('I will kill you'), hate speech slurs (N-word), or sexual violence.""",
        "calibrated_examples": [
            {"text": "I will find you and I will kill you.", "score": 0.99, "reasoning": "Direct death threat."}
        ]
    },
    "PROFANITY": {
        "definition": "Swear words, curse words, or other obscene or profane language.",
        "label": "Profanity",
        "additional_instruction": "Identify the literal use of swear words. Distinguish between 'positive' or 'casual' profanity (e.g., 'that's fucking cool') and hostile or demeaning profanity. Both contain profane language, but hostility increases the probability of human agreement on obscenity in a formal context.",
        "calibrated_examples": [
            {"text": "The economy is struggling.", "score": 0.01, "reasoning": "Clean language."}
        ]
    },
    "CURIOSITY_EXPERIMENTAL": {
        "definition": "The comment asks questions, requests evidence, or challenges the other person to explain their view. INCLUDES: Critical or skeptical questions, requests for sources, and 'probing' challenges (e.g., 'Did you read the article?'). Even if the tone is rude or confrontational, if it asks for information or clarification, it counts as Curiosity.",
        "label": "Curiosity",
        "calibrated_examples": [
            {"text": "You are wrong.", "score": 0.05, "reasoning": "Statement, no question."}
        ]
    },
    "REASONING_EXPERIMENTAL": {
        "definition": "The comment provides a logical argument where a claim is supported by evidence, facts, personal experience, or clear rationale. It must go beyond simple assertion to explain 'why'. Key Indicators: 'Because', 'Since', 'For example', or citing specific sources/events. Both formal logic and personal anecdotes count as reasoning if used to support a point.",
        "label": "Reasoning",
        "calibrated_examples": [
            {"text": "This policy is bad.", "score": 0.10, "reasoning": "Bare assertion with no support."}
        ]
    },
    "PERSONAL_STORY_EXPERIMENTAL": {
        "definition": "The comment shares a first-hand experience, personal anecdote, or specific life event to illustrate a point. It uses 'I' statements and descriptive details about the author's own life.",
        "label": "Personal Story",
        "calibrated_examples": [
            {"text": "My grandmother used to say that hard work pays off.", "score": 0.80, "reasoning": "Personal anecdote about family."}
        ]
    }
}