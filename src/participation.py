import pandas as pd
import re
import google.auth
from googleapiclient.discovery import build
from typing import List, Dict, Any, Optional


# --- Column Naming Conventions ---
# Regular expressions to dynamically identify question columns in the survey data.
GOV_QUESTION_PATTERN = re.compile(r'^question_(\d+)')
RANKING_TOPIC_PATTERN = re.compile(r'^ranking_(\d+)_topic')
RANKING_Q_PATTERN = re.compile(r'^ranking_(\d+)_q_(\d+)')
# --- End Column Naming Conventions ---


def is_float(value: Any) -> bool:
  """Checks if a value can be converted to a float."""
  if value is None:
    return False
  try:
    float(value)
    return True
  except ValueError:
    return False


class ParticipantResponse:
  """A wrapper for a participant's survey response row for easier data access.

  This abstracts the logic for finding and parsing data from different survey rounds (R1/R2)
  based on dynamic column name matching.
  """

  def __init__(self, series: pd.Series):
    if not isinstance(series, pd.Series):
      raise TypeError(
          'ParticipantResponse must be initialized with a pandas Series.'
      )
    self.data = series.copy()
    self.id = self.data.get('rid', 'N/A')

  # --- R1 Data Accessors ---

  def get_r1_survey_text(self) -> Optional[str]:
    """Returns the full survey text from R1, if available."""
    return (
        self.data.get('survey_text')
        if 'survey_text' in self.data and pd.notna(self.data['survey_text'])
        else None
    )

  # --- R2 Data Accessors ---

  def get_intro_question(self) -> Optional[Dict[str, str]]:
    """Gets the initial open-ended question from R2 (question_1)."""
    q_col, a_col = 'question_1', 'answer_1'
    if (
        q_col in self.data
        and a_col in self.data
        and pd.notna(self.data[q_col])
        and pd.notna(self.data[a_col])
    ):
      return {'question': self.data[q_col], 'answer': self.data[a_col]}
    return None

  def get_gov_responses(self) -> List[Dict[str, str]]:
    """Finds all Gallery of Voices (GoV) question/answer pairs from R2."""
    responses = []
    q_cols = [
        col for col in self.data.index if GOV_QUESTION_PATTERN.match(str(col))
    ]

    for q_col in q_cols:
      a_col = q_col.replace('question_', 'answer_')
      topic_col = f'{q_col}_topic'
      opinion_col = f'{q_col}_opinion'

      if (
          a_col in self.data
          and pd.notna(self.data[q_col])
          and pd.notna(self.data[a_col])
      ):
        responses.append({
            'topic': self.data.get(topic_col),
            'opinion': self.data.get(opinion_col),
            'question': self.data[q_col],
            'answer': self.data[a_col],
        })
    return responses

  def _parse_ranking_sets_legacy(self) -> Dict[str, Dict[str, Any]]:
    """Parses the original ranking format that relies on a 'ranking_X_topic' column."""
    sets = {}
    # This format is simpler and doesn't require pre-grouping.
    for col in self.data.index:
      match = RANKING_TOPIC_PATTERN.match(str(col))
      if match:
        set_num = match.group(1)
        ranked_items = []
        followup = None

        for q_col in self.data.index:
          q_match = RANKING_Q_PATTERN.match(str(q_col))
          if q_match and q_match.group(1) == set_num:
            q_num = q_match.group(2)
            a_col = f'ranking_{set_num}_a_{q_num}'
            if a_col in self.data and pd.notna(self.data[q_col]):
              answer = self.data.get(a_col)
              # FIX: Explicitly check that the answer is not NaN before processing.
              if pd.notna(answer):
                if is_float(answer):
                  ranked_items.append({
                      'rank': int(float(answer)),
                      'statement': self.data[q_col],
                  })
                else:
                  followup = {'question': self.data[q_col], 'answer': answer}

        if ranked_items:
          ranked_items.sort(key=lambda x: x['rank'])
          sets[set_num] = {
              'topic': self.data[col],
              'ranked_items': ranked_items,
              'followup': followup,
          }
    return sets

  def _group_ranking_columns(self) -> Dict[str, List[str]]:
    """Groups all ranking-related columns by their set number."""
    grouped_by_set = {}
    ranking_cols = [
        col for col in self.data.index if str(col).startswith('ranking_')
    ]
    for col in ranking_cols:
      match = re.match(r'ranking_(\d+)_', str(col))
      if match:
        set_num = match.group(1)
        if set_num not in grouped_by_set:
          grouped_by_set[set_num] = []
        grouped_by_set[set_num].append(col)
    return grouped_by_set

  def _parse_ranking_sets_processed(self) -> Dict[str, Dict[str, Any]]:
    """Parses the new 'processed.csv' format where topic is in the question."""
    sets = {}
    grouped_cols = self._group_ranking_columns()

    for set_num, cols in grouped_cols.items():
      ranked_items = []
      followup = None
      topic = None

      q_cols = sorted([c for c in cols if '_q_' in c])

      for q_col in q_cols:
        a_col = q_col.replace('_q_', '_a_')
        if a_col in self.data and pd.notna(self.data[q_col]):
          answer = self.data.get(a_col)
          question_text = self.data[q_col]

          # FIX: Explicitly check that the answer is not NaN before processing.
          if pd.notna(answer):
            if topic is None:
              topic_match = re.search(
                  r'Topic:\s*\n(.*?)\s*-', str(question_text)
              )
              if topic_match:
                topic = topic_match.group(1).strip()

            statement = re.sub(r'^Topic: \s*.*?\s*-\s*', '', str(question_text))

            if is_float(answer):
              ranked_items.append(
                  {'rank': int(float(answer)), 'statement': statement}
              )
            else:
              followup = {'question': statement, 'answer': answer}

      if ranked_items:
        ranked_items.sort(key=lambda x: x['rank'])
        sets[set_num] = {
            'topic': topic or 'Unknown Topic',
            'ranked_items': ranked_items,
            'followup': followup,
        }
    return sets

  def get_ranking_sets(self) -> Dict[str, Dict[str, Any]]:
    """
    Dispatcher function that sniffs the data format and calls the appropriate parser
    for ranking sets.
    """
    # Sniff for the legacy format by checking for a 'ranking_X_topic' column.
    if any(RANKING_TOPIC_PATTERN.match(str(col)) for col in self.data.index):
      return self._parse_ranking_sets_legacy()
    else:
      return self._parse_ranking_sets_processed()


def get_sheet_id_from_url(url: str) -> str | None:
  """Extracts the Google Sheet ID from a URL."""
  match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
  if match:
    return match.group(1)
  return None


def get_sheet_name_from_gid(sheet_id, gid, creds):
  service = build('sheets', 'v4', credentials=creds)
  sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
  sheets = sheet_metadata.get('sheets', '')
  for s in sheets:
    if str(s.get('properties', {}).get('sheetId')) == gid:
      return s.get('properties', {}).get('title')
  return None


def load_sheet_as_df(sheet_id, sheet_range):
  """
  Loads a worksheet from a Google Sheet into a pandas DataFrame.
  """
  creds, _ = google.auth.default(
      scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
  )
  service = build('sheets', 'v4', credentials=creds)
  sheet = service.spreadsheets()
  result = (
      sheet.values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
  )
  values = result.get('values', [])
  if not values:
    return pd.DataFrame()
  else:
    return pd.DataFrame(values[1:], columns=values[0])


def de_duplicate_columns(df):
  cols = pd.Series(df.columns)
  for dup in cols[cols.duplicated()].unique():
    cols[cols[cols == dup].index.values.tolist()] = [
        dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))
    ]
  df.columns = cols
  return df


def load_and_merge_participant_data(
    r1_url: str = None,
    r2_url: str = None,
    r1_merge_col: str = 'rid',
    r2_merge_col: str = 'rid',
) -> pd.DataFrame:
  """Loads and merges participant data from R1 and R2 Google Sheets."""
  creds, _ = google.auth.default(
      scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
  )
  df_r1 = None
  df_r2 = None

  if r1_url:
    sheet_id_r1 = get_sheet_id_from_url(r1_url)
    gid_r1 = re.search(r'gid=(\d+)', r1_url).group(1)
    sheet_name_r1 = get_sheet_name_from_gid(sheet_id_r1, gid_r1, creds)
    print(
        f'\nLoading data from sheet: {sheet_id_r1}, sheet name: {sheet_name_r1}'
    )
    df_r1 = load_sheet_as_df(sheet_id_r1, sheet_name_r1)
    df_r1 = de_duplicate_columns(df_r1)

  if r2_url:
    sheet_id_r2 = get_sheet_id_from_url(r2_url)
    gid_r2 = re.search(r'gid=(\d+)', r2_url).group(1)
    sheet_name_r2 = get_sheet_name_from_gid(sheet_id_r2, gid_r2, creds)
    print(
        f'\nLoading data from sheet: {sheet_id_r2}, sheet name: {sheet_name_r2}'
    )
    df_r2 = load_sheet_as_df(sheet_id_r2, sheet_name_r2)
    df_r2 = de_duplicate_columns(df_r2)

  if df_r1 is not None and df_r2 is not None:
    return pd.merge(
        df_r1, df_r2, left_on=r1_merge_col, right_on=r2_merge_col, how='outer'
    )
  elif df_r1 is not None:
    return df_r1
  elif df_r2 is not None:
    return df_r2
  else:
    return pd.DataFrame()


def get_prompt_representation(participant_row: pd.Series) -> str:
  """
  Generates a detailed, structured string representation of a participant's
  survey responses for use in LLM prompts by using the ParticipantResponse class.
  """
  response = ParticipantResponse(participant_row)

  # --- Compute all fields first ---
  participant_id = response.id
  r1_content = response.get_r1_survey_text() or ''

  intro_q = response.get_intro_question()
  gov_responses = response.get_gov_responses()
  ranking_sets = response.get_ranking_sets()

  # --- Assemble sections ---
  intro_q_section = ''
  if intro_q:
    intro_q_section = (
        f"<question>{intro_q['question']}</question>\n"
        f"<answer>{intro_q['answer']}</answer>"
    )

  gov_responses_section = '\n'.join(
      (
          f"<gov_response topic=\"{gov.get('topic', 'N/A')}\">\n  <prompt>What"
          ' do you think about the following quote:'
          f" \"{gov['question']}\"</prompt>\n "
          f" <answer>{gov['answer']}</answer>\n</gov_response>"
      )
      for gov in gov_responses
  )

  ranking_sets_sections = []
  for set_num, set_data in sorted(
      ranking_sets.items(), key=lambda item: int(item[0])
  ):
    ranked_items_str = '\n'.join(
        f'  <rank value="{item["rank"]}">{item["statement"]}</rank>'
        for item in set_data['ranked_items']
    )

    followup_section = ''
    if set_data.get('followup'):
      followup = set_data['followup']
      if followup and followup.get('question') and followup.get('answer'):
        followup_section = (
            f"  <followup_question>{followup['question']}</followup_question>\n"
            f"  <followup_answer>{followup['answer']}</followup_answer>"
        )

    ranking_set_parts = [
        f"  <topic>{set_data['topic']}</topic>",
        '  <prompt>How would you rank the following ideas?</prompt>',
        ranked_items_str,
    ]
    if followup_section:
      ranking_set_parts.append(followup_section)

    ranking_set_content = '\n'.join(ranking_set_parts)
    ranking_sets_sections.append(
        '<ranking_set'
        f' topic="{set_data["topic"]}">\n{ranking_set_content}\n</ranking_set>'
    )
  ranking_sets_section = '\n\n'.join(ranking_sets_sections)

  r2_parts = [
      part
      for part in [
          intro_q_section,
          gov_responses_section,
          ranking_sets_section,
      ]
      if part
  ]
  r2_content = '\n\n'.join(r2_parts)

  # --- Assemble the final prompt ---
  prompt = (
      "This is a record of a single participant's responses in an interactive"
      ' survey.\n'
      f'<participant id="{participant_id}">\n'
      '<round1>\n'
      f'{r1_content}\n'
      '</round1>\n\n'
      '<round2>\n'
      f'{r2_content}\n'
      '</round2>\n'
      '</participant>'
  )
  return prompt


def get_r2_preferences_from_dataframe(
    df: pd.DataFrame,
) -> dict[str, list[list[str]]]:
  """Extracts the ranked preferences from the R2 data using the ParticipantResponse class."""
  preferences_by_topic = {}

  for _, row in df.iterrows():
    participant = ParticipantResponse(row)
    ranking_sets = participant.get_ranking_sets()

    for _, set_data in ranking_sets.items():
      topic = set_data['topic']
      if topic not in preferences_by_topic:
        preferences_by_topic[topic] = []

      ranked_statements = [
          item['statement'] for item in set_data['ranked_items']
      ]
      if ranked_statements:
        preferences_by_topic[topic].append(ranked_statements)

  return preferences_by_topic
