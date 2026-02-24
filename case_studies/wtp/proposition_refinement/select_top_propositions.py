"""
Selects the top propositions from a given list using a simulated jury.
"""

import asyncio
import re
import pandas as pd
import numpy as np
import time
import google.auth
import os
from case_studies.wtp.models import genai_model
from .main import get_sheet_id_from_url, load_sheet_as_df, get_sheet_name_from_gid, de_duplicate_columns
from case_studies.wtp.simulated_jury import simulated_jury
from case_studies.wtp.social_choice import schulze

COMBINED_PROPOSITIONS = [
    (
        "The right to express one's opinion freely is best protected when"
        " coupled with a shared responsibility to avoid harassment and the"
        " incitement of violence."
    ),
    (
        "A society should strive to provide sufficient economic opportunity so"
        " that all individuals have the practical ability to make meaningful"
        " choices about their own lives."
    ),
    (
        "A just government's core function is to ensure public safety and"
        " security in a way that minimizes interference in the personal lives"
        " and choices of its citizens."
    ),
    (
        "True economic freedom exists only when arbitrary barriers like"
        " discrimination are removed, allowing every individual the opportunity"
        " to reach their full potential."
    ),
    (
        "The freedom to live an authentic life according to one's own values is"
        " sustained by a mutual respect for the diverse choices and paths of"
        " others."
    ),
    (
        "The ability to achieve internal peace and freedom from worry is"
        " fundamentally dependent on a secure environment free from the threat"
        " of physical violence."
    ),
    (
        "A fair society limits government regulation on personal enterprise"
        " while actively working to dismantle systemic discrimination that"
        " restricts economic opportunity."
    ),
    (
        "The assurance that one can voice an opinion without fear of punishment"
        " or government reprisal is essential for both civic health and"
        " personal peace of mind."
    ),
    (
        "The freedom to move and travel without restriction is only meaningful"
        " when individuals have the financial resources to make such choices"
        " possible."
    ),
    (
        "A core responsibility of a free citizen is to treat others with"
        " dignity, ensuring that everyone can participate in society without"
        " fear of discrimination or oppression."
    ),
    (
        "Measures to ensure public safety must be transparent and accountable"
        " to prevent them from becoming tools of undue government control or"
        " harassment."
    ),
    (
        "A society fosters freedom by minimizing government restrictions on"
        " individual ambition while ensuring a baseline of economic stability"
        " that prevents desperation."
    ),
    (
        "Freedom from discrimination is a prerequisite for internal peace, as"
        " the constant threat of oppression creates pervasive anxiety and fear."
    ),
    (
        "The freedom to choose one's own path is inseparable from the freedom"
        " to voice the beliefs and values that guide those choices."
    ),
    (
        "A society can minimize the need for government regulation when its"
        " citizens practice self-governance and mutual respect in their"
        " communities."
    ),
    (
        "Creating broad economic opportunity and reducing financial desperation"
        " are among the most effective long-term strategies for ensuring public"
        " safety."
    ),
    (
        "The ability to live an authentic life is incomplete if society only"
        " grants that freedom to a select few; it must be a universal right,"
        " free from discrimination."
    ),
    (
        "A culture of free expression is strongest when it encourages robust"
        " debate and disagreement, while simultaneously discouraging personal"
        " harassment."
    ),
    (
        "Living with minimal government surveillance is crucial for fostering a"
        " sense of internal freedom and the ability to live without constant"
        " worry."
    ),
    (
        "True personal autonomy requires not just the absence of coercion, but"
        " also the presence of sufficient economic resources to make one's"
        " choices viable."
    ),
    (
        "A society is only truly safe when all its members, regardless of"
        " background, can move through public and private spaces without fear"
        " of violence or discrimination."
    ),
    (
        "Freedom requires guarding against government overreach while also"
        " empowering the government to protect the rights of minorities and"
        " dismantle systemic discrimination."
    ),
    (
        "The pursuit of financial freedom should be balanced with a commitment"
        " to ethical conduct and respect for the well-being of the community."
    ),
    (
        "The internal freedom from worry is best achieved when individuals feel"
        " a genuine sense of control and agency over the fundamental decisions"
        " of their lives."
    ),
    (
        "The right to assemble and protest is a cornerstone of liberty, and it"
        " is the government's duty to protect this right while maintaining"
        " public safety for all."
    ),
    (
        "A commitment to free expression must include actively ensuring that"
        " marginalized voices can be heard and are not drowned out by"
        " harassment or systemic disadvantage."
    ),
    (
        "The fundamental freedom to move without restriction depends on a"
        " collective commitment to creating safe public spaces for everyone."
    ),
    (
        "The ideal of freedom from government interference is most achievable"
        " in a society where individuals and communities demonstrate a high"
        " capacity for responsible self-governance."
    ),
    (
        "Personal freedom is diminished both by undue government control and by"
        " the anxiety that comes from financial instability."
    ),
    (
        "A truly free person is one who enjoys economic stability, the autonomy"
        " to make personal choices, and the civil liberties to participate in"
        " society, all within a framework of mutual respect and physical"
        " safety."
    ),
]


async def main():
  creds, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
  )

  # Load R2 data
  worksheet_url_r2 = "https://docs.google.com/spreadsheets/d/1K_MybacxSpSnr4Rs6x23rhxGVf3zmz3doVfvAgdmh9Q/edit?gid=51426834#gid=51426834"
  sheet_id_r2 = get_sheet_id_from_url(worksheet_url_r2)
  gid_r2 = re.search(r"gid=(\d+)", worksheet_url_r2).group(1)
  sheet_name_r2 = get_sheet_name_from_gid(sheet_id_r2, gid_r2, creds)
  print(
      f"\nLoading data from sheet: {sheet_id_r2}, sheet name: {sheet_name_r2}"
  )
  df_r2 = load_sheet_as_df(sheet_id_r2, sheet_name_r2)
  df_r2 = de_duplicate_columns(df_r2)

  gemini_api_key = os.environ.get("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

  model = genai_model.GenaiModel(
      api_key=gemini_api_key,
      model_name="gemini-2.5-flash",
      embedding_model_name="text-embedding-004",
  )

  # Run simulation
  print(f"\n--- Running simulation for {len(df_r2)} participants ---")
  results = await simulated_jury.run_simulated_jury(
      df_r2, COMBINED_PROPOSITIONS, simulated_jury.VotingMode.RANK, model
  )

  # Log the results
  timestamp = time.strftime("%Y%m%d-%H%M%S")
  log_filename = f"simulated_jury_responses_{timestamp}.log"
  with open(log_filename, "w") as f:
    for result in results:
      f.write(result + "\n")
  print(f"Simulated jury responses saved to {log_filename}")

  schulze_results = schulze.get_schulze_ranking(
      results, COMBINED_PROPOSITIONS, k=6
  )
  top_propositions = schulze_results["top_propositions"]

  print("\n--- Top 6 Combined Propositions ---")
  for i, proposition in enumerate(top_propositions, 1):
    print(f"{i}. {proposition}")


if __name__ == "__main__":
  asyncio.run(main())
