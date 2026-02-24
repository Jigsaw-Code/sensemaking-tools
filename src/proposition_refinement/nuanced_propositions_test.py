import unittest
from src.proposition_refinement import nuanced_propositions


class NuancedPropositionsTest(unittest.TestCase):

  def setUp(self):
    self.propositions_by_topic = {
        "Topic 1": ["Prop A", "Prop B"],
        "Topic 2": ["Prop C"],
    }

  def test_generate_combination_prompt_no_context(self):
    prompt = nuanced_propositions.generate_combination_prompt(
        self.propositions_by_topic, num_combinations=3, additional_context=None
    )
    self.assertNotIn("<additionalContext>", prompt)
    self.assertIn("**Topic: Topic 1**", prompt)
    self.assertIn("1. Prop A", prompt)
    self.assertIn("Generate 3 new propositions", prompt)

  def test_generate_combination_prompt_empty_context(self):
    prompt = nuanced_propositions.generate_combination_prompt(
        self.propositions_by_topic, num_combinations=3, additional_context=""
    )
    self.assertNotIn("<additionalContext>", prompt)
    self.assertIn("**Topic: Topic 1**", prompt)
    self.assertIn("1. Prop A", prompt)
    self.assertIn("Generate 3 new propositions", prompt)

  def test_generate_combination_prompt_with_context(self):
    prompt = nuanced_propositions.generate_combination_prompt(
        self.propositions_by_topic,
        num_combinations=5,
        additional_context="Background information.",
    )
    self.assertIn("<additionalContext>", prompt)
    self.assertIn("Background information.", prompt)
    self.assertIn("</additionalContext>", prompt)
    self.assertIn("**Topic: Topic 1**", prompt)
    self.assertIn("Generate 5 new propositions", prompt)


if __name__ == "__main__":
  unittest.main()
