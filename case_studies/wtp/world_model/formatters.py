import pandas as pd


def format_simple_proposition_card(record):
  """Formats a single simple proposition record into a text card."""
  card = []
  separator = "=" * 80

  card.append(separator)
  card.append(f"Topic: {record.get('topic', 'N/A')}")
  card.append(f"Proposition ID: {record.get('proposition_id', 'N/A')}")
  card.append(f"Is Duplicate: {record.get('duplicate', 'N/A')}")
  card.append(f"Is Selected: {record.get('selected', 'N/A')}")
  card.append(separator)

  card.append("\n--- Proposition ---")
  card.append(record.get("proposition", "N/A"))

  card.append("\n" + separator + "\n")
  return "\n".join(card)


def format_simulation_result_card(record):
  """Formats a single simulation result record (a pandas Series) into a text card."""
  card = []
  separator = "=" * 80

  # --- Header with short, important fields ---
  card.append(separator)
  card.append(f"Topic: {record.get('topic', 'N/A')}")
  card.append(f"Participant (rid): {record.get('rid', 'N/A')}")
  card.append(f"Job ID: {record.get('job_id', 'N/A')}")
  card.append(f"Tokens Used: {record.get('total_token_used', 'N/A')}")

  failed_tries_df = record.get("failed_tries", [])
  num_failed_tries = len(failed_tries_df)
  card.append(f"Number of Failed Tries: {num_failed_tries}")

  card.append(separator)

  # --- Long-form content, pushed to the end for readability ---

  # Reasoning
  card.append("\n--- Reasoning ---")
  result_dict = record.get("result", {})
  reasoning = result_dict.get(
      "reasoning", "No reasoning present in world model."
  )
  card.append(
      reasoning
      if pd.notna(reasoning) and reasoning
      else "No reasoning provided."
  )

  # Ranked Propositions
  card.append("\n--- Simulated Proposition Ranking ---")
  ranking = result_dict.get("simulated_proposition_ranking", [])
  if isinstance(ranking, list) and ranking:
    for i, prop in enumerate(ranking, 1):
      card.append(f"{i}. {prop}")
  else:
    card.append("No ranking provided.")

  # Raw Response from the 'result' dictionary
  card.append("\n--- Raw LLM Response ---")
  raw_response = result_dict.get(
      "raw_response", "No raw response present in world model."
  )
  card.append(raw_response)

  # Failed Tries Details
  if num_failed_tries > 0:
    # This data might not exist if the jury simulation succeeded on the first
    # try. We check for DataFrame type before calling DataFrame-specific methods.
    if isinstance(failed_tries_df, pd.DataFrame):
      card.append("\n--- Failed Attempts Details ---")
      card.append(failed_tries_df.to_string())
    else:
      card.append("\n--- Failed Attempts Details ---")
      card.append(
          "Could not display failed attempts: data may not be present in world"
          " model."
      )

  # Prompt
  card.append("\n--- Prompt ---")
  prompt = record.get("prompt", "No prompt available.")
  card.append(prompt)

  card.append("\n" + separator + "\n")
  return "\n".join(card)


def format_failed_try_card(record):
  """Formats a single failed try record into a text card."""
  card = []
  separator = "=" * 80

  # --- Header ---
  card.append(separator)
  card.append(f"Topic: {record.get('topic', 'N/A')}")
  card.append(f"Participant (rid): {record.get('rid', 'N/A')}")
  card.append(f"Attempt Index: {record.get('attempt_index', 'N/A')}")
  card.append(separator)

  # --- Error Message ---
  card.append("\n--- Error Message ---")
  card.append(record.get("error_message", "No error message provided."))

  # --- Failed Response ---
  card.append("\n--- Failed Response ---")
  card.append(record.get("response", "No response recorded."))

  # --- Prompt that caused failure ---
  card.append("\n--- Prompt ---")
  card.append(record.get("prompt", "No prompt available."))

  card.append("\n" + separator + "\n")
  return "\n".join(card)


def format_participant_card(record):
  """Formats a single participant data record into a text card."""
  card = []
  separator = "=" * 80

  # --- Header ---
  card.append(separator)
  card.append(f"Participant (rid): {record.get('rid', 'N/A')}")
  card.append(f"Topic: {record.get('topic', 'N/A')}")
  card.append(separator)

  # --- Core Opinion and Representative Text ---
  card.append("\n--- Stated Opinion ---")
  card.append(record.get("opinion", "N/A"))

  card.append("\n--- Representative Text ---")
  card.append(record.get("representative_text", "N/A"))

  # --- R1 Q&A ---
  card.append("\n--- Survey Responses (R1) ---")
  for i in range(1, 5):  # Assuming up to 4 questions
    q_text = record.get(f"Q{i}_Text")
    q_resp = record.get(f"Q{i}")
    qfu_text = record.get(f"Q{i}FU_Text")
    qfu_resp = record.get(f"Q{i}FU")

    if pd.notna(q_text) and pd.notna(q_resp):
      card.append(f"\nQ{i}: {q_text.strip()}")
      card.append(f"A: {q_resp.strip()}")
    if pd.notna(qfu_text) and pd.notna(qfu_resp):
      card.append(f"  Follow-up Q: {qfu_text.strip()}")
      card.append(f"  Follow-up A: {qfu_resp.strip()}")

  # --- R2 Q&A ---
  card.append("\n--- Survey Responses (R2) ---")
  # Free text questions
  for i in range(1, 7):  # Assuming up to 6 questions
    q_col = f"question_{i}"
    a_col = f"answer_{i}"
    if (
        q_col in record
        and a_col in record
        and pd.notna(record[q_col])
        and pd.notna(record[a_col])
    ):
      card.append(f"\nQ: {record[q_col].strip()}")
      card.append(f"A: {record[a_col].strip()}")

  # R2 Ranking questions
  for i in range(1, 6):  # Assuming up to 5 ranking sets
    ranking_data = []
    has_ranking_data = False
    for j in range(1, 4):  # Assuming up to 3 questions to rank
      q_col = f"ranking_{i}_q_{j}"
      a_col = f"ranking_{i}_a_{j}"
      if q_col in record and pd.notna(record[q_col]):
        has_ranking_data = True
        rank = record.get(a_col, "N/A")
        ranking_data.append((rank, record[q_col]))

    if has_ranking_data:
      card.append(f"\n--- R2 Ranking Set {i} ---")
      # Sort by rank before printing
      ranking_data.sort(key=lambda x: x[0])
      for rank, question in ranking_data:
        card.append(f"Rank {rank}: {question.strip()}")

      # Add the follow-up question for the ranking
      qfu_col = f"ranking_{i}_q_4"
      afu_col = f"ranking_{i}_a_4"
      if qfu_col in record and pd.notna(record[qfu_col]):
        card.append(f"\nFollow-up Q: {record[qfu_col].strip()}")
        if afu_col in record and pd.notna(record[afu_col]):
          card.append(f"Follow-up A: {record[afu_col].strip()}")

  card.append("\n" + separator + "\n")
  return "\n".join(card)


def format_propositions_by_topic(df):
  """Formats a DataFrame of propositions into a text block grouped by topic."""
  if df.empty:
    return "No propositions to display."

  output = []
  separator = "=" * 80

  # Separate Nuanced propositions to be appended at the end
  nuanced_df = df[df["topic"] == "Nuanced"]
  other_topics_df = df[df["topic"] != "Nuanced"]

  # Sort other topics by r1_df_length
  topic_counts = other_topics_df.groupby("topic")["r1_df_length"].first()
  sorted_topic_names = topic_counts.sort_values(ascending=False).index

  for topic in sorted_topic_names:
    group = other_topics_df[other_topics_df["topic"] == topic]
    if group.empty:
      continue
    count = group["r1_df_length"].iloc[0]

    output.append(separator)
    output.append(f"Topic: {topic} ({int(count)} comments)")
    output.append(separator)

    sorted_group = group.sort_values(by="rank")

    for _, row in sorted_group.iterrows():
      output.append(f"{row['rank']}. {row['proposition']}")
    output.append("")

  # Append Nuanced propositions at the end
  if not nuanced_df.empty:
    output.append(separator)
    output.append("Topic: Nuanced")
    output.append(separator)

    sorted_group = nuanced_df.sort_values(by="rank")

    for _, row in sorted_group.iterrows():
      output.append(f"{row['rank']}. {row['proposition']}")
    output.append("")

  return "\n".join(output)


def format_default_card(record):
  """Formats a generic record into a key-value card."""
  card = []
  separator = "=" * 80
  card.append(separator)

  for key, value in record.items():
    card.append(f"{key}: {value}")

  card.append(separator + "\n")
  return "\n".join(card)
