import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.proposition_refinement import deduplication
import asyncio


class DeduplicationTest(unittest.TestCase):

  def test_preprocess_and_add_ids(self):
    """
    Tests that the preprocessing function correctly sorts topics and adds
    stable IDs to both the main DataFrame and the nested proposition DataFrames.
    """
    # Create a sample world_model DataFrame, deliberately unsorted
    data = {
        'topic': ['Topic B', 'Topic A'],
        'r1_quotes_by_topic': [10, 20],  # Topic A should come first
        'propositions': [
            pd.DataFrame({'proposition': ['prop b1', 'prop b2']}),
            pd.DataFrame({'proposition': ['prop a1', 'prop a2', 'prop a3']}),
        ],
    }
    world_model_df = pd.DataFrame(data)

    processed_df = deduplication._preprocess_and_add_ids(world_model_df.copy())

    # 1. Check that topics are sorted by r1_df_length (descending)
    self.assertEqual(list(processed_df['topic']), ['Topic A', 'Topic B'])

    # 2. Check that topic_id is added correctly based on the new, sorted order
    self.assertEqual(processed_df.iloc[0]['topic_id'], 0)
    self.assertEqual(processed_df.iloc[1]['topic_id'], 1)

    # 3. Check that proposition_id is added correctly and reflects the sorted topic_id
    topic_a_props = processed_df.iloc[0]['propositions']
    self.assertEqual(topic_a_props.iloc[0]['proposition_id'], '0:0')
    self.assertEqual(topic_a_props.iloc[1]['proposition_id'], '0:1')
    self.assertEqual(topic_a_props.iloc[2]['proposition_id'], '0:2')

    topic_b_props = processed_df.iloc[1]['propositions']
    self.assertEqual(topic_b_props.iloc[0]['proposition_id'], '1:0')
    self.assertEqual(topic_b_props.iloc[1]['proposition_id'], '1:1')

    # 4. Check that new columns are initialized to False
    self.assertFalse(topic_a_props['duplicate'].any())
    self.assertFalse(topic_a_props['selected'].any())
    self.assertFalse(topic_b_props['duplicate'].any())
    self.assertFalse(topic_b_props['selected'].any())

  def test_generate_equivalence_prompt(self):
    """
    Tests that the prompt generation for equivalence clustering is correct.
    """
    propositions_map = {
        '0:0': 'Proposition A from Topic 0.',
        '1:5': 'Proposition B from Topic 1.',
        '2:2': 'Proposition C from Topic 2.',
    }

    prompt = deduplication.generate_equivalence_prompt(propositions_map)

    # Check for the main instruction text
    self.assertIn(
        'Your task is to group propositions into sets that have an'
        ' effectively equivalent meaning',
        prompt,
    )
    # Check that all propositions are listed with their correct IDs
    self.assertIn('0:0: "Proposition A from Topic 0."', prompt)
    self.assertIn('1:5: "Proposition B from Topic 1."', prompt)
    self.assertIn('2:2: "Proposition C from Topic 2."', prompt)

    # Check for the JSON format instruction
    self.assertIn(
        'Provide your answer as a JSON object with a single key'
        " 'equivalence_sets'",
        prompt,
    )
    # Check for the example
    self.assertIn('{"equivalence_sets": [["0:1", "3:5"]', prompt)

  @patch('src.models.genai_model.GenaiModel.process_prompts_concurrently')
  def test_generate_equivalence_sets(self, mock_process_prompts):
    """
    Tests that the equivalence set generation function correctly calls the LLM
    and parses the JSON response.
    """

    # Mock the LLM response
    async def mock_async_function(*args, **kwargs):
      mock_response_df = pd.DataFrame(
          {'result': [{'equivalence_sets': [['0:0', '1:1'], ['0:1']]}]}
      )
      return (mock_response_df, pd.DataFrame())

    mock_process_prompts.side_effect = mock_async_function

    # Create sample data
    data = {
        'topic': ['Topic A', 'Topic B'],
        'r1_df_length': [20, 10],
        'propositions': [
            pd.DataFrame({
                'proposition': ['prop a1', 'prop a2'],
                'proposition_id': ['0:0', '0:1'],
            }),
            pd.DataFrame({
                'proposition': ['prop b1', 'prop b2'],
                'proposition_id': ['1:0', '1:1'],
            }),
        ],
    }
    world_model_df = pd.DataFrame(data)

    # Mock the model object
    mock_model = MagicMock()
    mock_model.process_prompts_concurrently = mock_process_prompts

    # Run the function
    result = asyncio.run(
        deduplication.generate_equivalence_sets(world_model_df, mock_model)
    )

    # Check the result
    self.assertEqual(result, [['0:0', '1:1'], ['0:1']])
    mock_process_prompts.assert_called_once()

  @patch('src.models.genai_model.GenaiModel.process_prompts_concurrently')
  def test_resolve_collision(self, mock_process_prompts):
    """
    Tests that the collision resolver correctly identifies the winning proposition
    from a set of duplicates based on the LLM's response.
    """

    # Mock the LLM response to select the second proposition
    async def mock_async_function(*args, **kwargs):
      mock_response_df = pd.DataFrame({'result': ['1:5']})
      return (mock_response_df, pd.DataFrame())

    mock_process_prompts.side_effect = mock_async_function

    collision_group = [
        {'prop_id': '0:2', 'text': 'Prop A'},
        {'prop_id': '1:5', 'text': 'Prop B'},
    ]

    # Mock the model object
    mock_model = MagicMock()
    mock_model.process_prompts_concurrently = mock_process_prompts

    winner_id = asyncio.run(
        deduplication._resolve_collision(collision_group, mock_model)
    )

    self.assertEqual(winner_id, '1:5')
    mock_process_prompts.assert_called_once()

  @patch('src.proposition_refinement.deduplication._resolve_collision')
  def test_select_final_propositions_rank_filling(self, mock_resolve_collision):
    """
    Tests that the rank-filling algorithm correctly selects propositions,
    handling collisions and allowing topics that lose a collision to catch up.
    """

    # Mock the collision resolver to always pick the first candidate.
    async def mock_resolver(collision_group, model):
      return collision_group[0]['prop_id']

    mock_resolve_collision.side_effect = mock_resolver

    # Create sample data
    data = {
        'topic': ['Topic A', 'Topic B'],
        'r1_quotes_by_topic': [20, 10],
        'topic_id': [0, 1],
        'propositions': [
            pd.DataFrame({
                'proposition': ['a1_dup', 'a2', 'a3'],
                'proposition_id': ['0:0', '0:1', '0:2'],
                'duplicate': [True, False, False],
                'selected': [False, False, False],
            }),
            pd.DataFrame({
                'proposition': ['b1', 'b2_dup', 'b3'],
                'proposition_id': ['1:0', '1:1', '1:2'],
                'duplicate': [False, True, False],
                'selected': [False, False, False],
            }),
        ],
        'full_schulze_ranking': [
            ['a1_dup', 'a2', 'a3'],  # Topic A's ranking
            ['b1', 'b2_dup', 'b3'],  # Topic B's ranking
        ],
    }
    world_model_df = pd.DataFrame(data)
    world_model_df['topic_id'] = world_model_df.index

    equivalence_sets = [['0:0', '1:1']]  # a1_dup is equivalent to b2_dup

    # Mock the model object
    mock_model = MagicMock()

    # Run the selection for N=2 propositions per topic
    final_df, final_props = asyncio.run(
        deduplication.select_final_propositions(
            world_model_df,
            equivalence_sets,
            2,
            mock_model,
            ranking_column='full_schulze_ranking',
        )
    )
    # --- Verification ---
    self.assertEqual(len(final_props['Topic A']), 2)
    self.assertEqual(len(final_props['Topic B']), 2)

    self.assertIn('a1_dup', final_props['Topic A'])
    self.assertIn('a2', final_props['Topic A'])

    self.assertIn('b1', final_props['Topic B'])
    self.assertIn('b3', final_props['Topic B'])

    # Check the 'selected' column in the DataFrame
    self.assertTrue(
        final_df.iloc[0]['propositions'].loc[0, 'selected']
    )  # a1_dup
    self.assertTrue(final_df.iloc[0]['propositions'].loc[1, 'selected'])  # a2
    self.assertFalse(final_df.iloc[0]['propositions'].loc[2, 'selected'])  # a3

    self.assertTrue(final_df.iloc[1]['propositions'].loc[0, 'selected'])  # b1
    self.assertFalse(
        final_df.iloc[1]['propositions'].loc[1, 'selected']
    )  # b2_dup
    self.assertTrue(final_df.iloc[1]['propositions'].loc[2, 'selected'])  # b3

  def test_id_generation_after_aggregation(self):
    """
    Tests that proposition_ids remain unique after the full aggregation
    and ID-generation pipeline. This is the comprehensive test for the
    non-unique index bug.
    """
    # 1. SETUP: Mimic the raw data from two opinions in the same topic.
    by_opinion_df = pd.DataFrame({
        'topic': ['T1', 'T1'],
        'opinion': ['O1', 'O2'],
        'propositions': [
            pd.DataFrame({'proposition': ['p1', 'p2']}),  # Indices: 0, 1
            pd.DataFrame({'proposition': ['p3', 'p4']}),  # Indices: 0, 1
        ],
        'r1_quotes_by_topic': [50, 50],
        'full_schulze_ranking': [['p1', 'p2'], ['p3', 'p4']],
    })

    # 2. RUN THE COMBINED PREPROCESSING FUNCTION
    processed_df = deduplication._preprocess_and_add_ids(by_opinion_df)

    # 3. ASSERT CORRECTNESS:
    # The output should have the same number of rows as the input, as aggregation
    # is no longer done in this function.
    self.assertEqual(len(processed_df), 2)

    # The nested propositions should have a clean, unique index.
    props_df1 = processed_df.iloc[0]['propositions']
    props_df2 = processed_df.iloc[1]['propositions']
    self.assertTrue(props_df1.index.is_unique)
    self.assertTrue(props_df2.index.is_unique)
    self.assertEqual(len(props_df1), 2)
    self.assertEqual(len(props_df2), 2)

    # The final IDs should be unique across the entire set.
    ids1 = props_df1['proposition_id'].tolist()
    ids2 = props_df2['proposition_id'].tolist()
    all_ids = ids1 + ids2
    self.assertEqual(
        len(all_ids),
        len(set(all_ids)),
        'Duplicate proposition_ids were generated.',
    )
    self.assertEqual(ids1, ['0:0', '0:1'])
    self.assertEqual(ids2, ['1:0', '1:1'])

  def test_preprocess_handles_non_sequential_indices(self):
    """
    Tests that proposition_ids are generated sequentially even if the
    input DataFrame has a non-sequential index.
    """
    # Create a propositions DataFrame with a gappy index
    gappy_props_df = pd.DataFrame(
        {'proposition': ['prop a', 'prop b', 'prop c']}, index=[0, 2, 5]
    )

    # Create the main by_topic DataFrame
    data = {
        'topic': ['Topic A'],
        'r1_quotes_by_topic': [10],
        'propositions': [gappy_props_df],
    }
    by_topic_df = pd.DataFrame(data)

    # Run the function
    processed_df = deduplication._preprocess_and_add_ids(by_topic_df.copy())

    # Get the processed propositions DataFrame
    final_props_df = processed_df.iloc[0]['propositions']

    # Assert that the index is now sequential
    self.assertTrue(
        final_props_df.index.equals(pd.RangeIndex(start=0, stop=3, step=1))
    )

    # Assert that the proposition_ids are sequential
    expected_ids = ['0:0', '0:1', '0:2']
    self.assertEqual(final_props_df['proposition_id'].tolist(), expected_ids)


if __name__ == '__main__':
  unittest.main()
