OPINION_CATEGORIZATION_MAIN_RULES = """
Main Rules:
- PRIORITY RULE: Select Only the Most Literal Match(es). You must evaluate all opinions. After evaluating all of them, select only the opinion or opinions that are the most literal, most explicit, and most holistic match and require the least inference.
- Keep the number of selections to the absolute minimum. Only select more than one opinion if they are both equally perfect, literal matches to the quote.
- PRIORITY RULE: The Main Thesis MUST Agree. You must first compare the primary claim of the quote and the opinion. If the primary claims contradict (e.g., 'equal opportunity' vs. 'same results'), you must not select that category, even if supporting examples (like 'housing') are similar.
- The quote must holistically match the entire opinion. A partial keyword match (like matching 'equality' but ignoring the rest of the quote) is not sufficient. To be a match, the quote must explicitly support every key concept within the opinion.
- The quote and the opinion must be making the same kind of claim. Do not match a personal definition (e.g., 'Freedom means...') to a conditional argument (e.g., 'Freedom isn't real until...').
- Be Aggressively Literal / No Inference. Do not make inferences or assumptions. Do not make logical leaps or semantic substitutions. If an opinion mentions a concept like 'dignity,' the quote must also mention 'dignity' or 'respect.' Do not infer that 'equal rights' is the same as 'dignity.' Do not infer 'regime' means 'corrupt government.'
- Match Abstraction Levels. Do not match specific examples (like 'targeting based on skin color') to a broad category (like 'culture of hate').
"""
