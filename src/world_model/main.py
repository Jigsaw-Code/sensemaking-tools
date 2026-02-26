import argparse
import pandas as pd
import sys
from . import world_model
from . import formatters


def main():
  # A mapping from query names to their accessor functions
  QUERY_HANDLERS = {
      "all_by_opinion": world_model.get_all_by_opinion_propositions,
      "all_by_topic": world_model.get_all_by_topic_propositions,
      "all_nuanced": world_model.get_all_nuanced_propositions,
      "selected_by_opinion": world_model.get_selected_by_opinion_propositions,
      "selected_by_topic": world_model.get_selected_by_topic_propositions,
      "selected_nuanced": world_model.get_selected_nuanced_propositions,
      "selected_propositions": world_model.get_selected_propositions,
      "participant_data": world_model.get_participant_data,
      "simulation_results": world_model.get_simulation_results,
      "failed_tries": world_model.get_failed_tries,
      "simulated_jury_stats": world_model.get_simulated_jury_stats,
  }

  parser = argparse.ArgumentParser(
      description=(
          "Query the world model data structure and output in various"
          " formats.\n\nThe default output format is 'text', which is the most"
          " well-tested and human-readable. Other formats should largely work"
          " as expected. See the --output_format option for a full list of"
          " supported formats."
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument(
      "input_pkl", help="Path to the input world model .pkl file."
  )
  parser.add_argument(
      "--query",
      required=True,
      choices=list(QUERY_HANDLERS.keys()) + ["nested"],
      help=(
          "The name of the query to run. Use 'nested' for nested attribute"
          " access."
      ),
  )
  parser.add_argument(
      "--output_format",
      default="text",
      choices=["text", "csv", "json", "jsonl", "pkl"],
      help="The output format for the results.",
  )
  parser.add_argument(
      "--head", type=int, default=None, help="Return only the first N results."
  )
  parser.add_argument(
      "--attr",
      type=str,
      help=(
          "Dot-separated attribute string for 'nested' query (e.g.,"
          " 'world_model.simulation_results')."
      ),
  )
  # Arguments for accessors
  parser.add_argument(
      "--top_n_opinion",
      type=int,
      help="Number of top propositions for selected_by_opinion.",
  )
  parser.add_argument(
      "--top_n_topic",
      type=int,
      help="Number of top topic propositions for selected_propositions.",
  )
  parser.add_argument(
      "--top_n_nuanced",
      type=int,
      help="Number of top nuanced propositions for selected_propositions.",
  )
  parser.add_argument(
      "--data_source",
      choices=["r1", "r2", "both"],
      default="both",
      help=(
          "For 'participant_data' query: specifies which round of survey data"
          " to return."
      ),
  )
  parser.add_argument(
      "--level",
      choices=["opinion", "topic", "nuanced"],
      default="topic",
      help=(
          "For 'simulation_results' and 'failed_tries' queries: specifies which"
          " level of simulation to extract results from."
      ),
  )

  args = parser.parse_args()

  try:
    world_model_data = world_model.load_world_model(args.input_pkl)

    if args.query == "nested":
      if not args.attr:
        raise ValueError("--attr is required for 'nested' query.")
      result = world_model.get_nested_attribute(world_model_data, args.attr)

      if isinstance(result, pd.DataFrame):
        result_df = result
      elif isinstance(result, pd.Series) and all(
          isinstance(item, pd.DataFrame) for item in result.dropna()
      ):
        result_df = pd.concat(result.dropna().tolist(), ignore_index=True)
      else:
        print(result)
        return
    else:
      handler = QUERY_HANDLERS[args.query]
      handler_args = {"world_model_data": world_model_data}

      if args.query == "selected_by_opinion":
        handler_args["top_n"] = args.top_n_opinion
      elif args.query == "selected_by_topic":
        handler_args["top_n"] = args.top_n_topic
      elif args.query == "selected_nuanced":
        handler_args["top_n"] = args.top_n_nuanced
      elif args.query == "selected_propositions":
        handler_args["top_n_topic"] = args.top_n_topic
        handler_args["top_n_nuanced"] = args.top_n_nuanced
      elif args.query == "participant_data":
        handler_args["data_source"] = args.data_source
      elif args.query in ["simulation_results", "failed_tries"]:
        handler_args["level"] = args.level

      handler_args = {k: v for k, v in handler_args.items() if v is not None}
      result_df = handler(**handler_args)

    if args.head is not None and args.head > 0:
      result_df = result_df.head(args.head)

    if args.output_format == "text":
      # Proposition lists have a special DataFrame-level formatter
      if args.query in [
          "selected_simple",
          "selected_nuanced",
          "selected_propositions",
      ]:
        print(formatters.format_propositions_by_topic(result_df))
      else:
        # Other queries use a row-by-row card formatter
        CARD_FORMATTERS = {
            "simulation_results": formatters.format_simulation_result_card,
            "failed_tries": formatters.format_failed_try_card,
            "participant_data": formatters.format_participant_card,
        }
        # Get the specific formatter, or use the new default card formatter
        formatter = CARD_FORMATTERS.get(
            args.query, formatters.format_default_card
        )
        for _, row in result_df.iterrows():
          print(formatter(row))
    elif args.output_format == "csv":
      print(result_df.to_csv(index=False))
    elif args.output_format == "json":
      print(result_df.to_json(orient="records", indent=2))
    elif args.output_format == "jsonl":
      print(result_df.to_json(orient="records", lines=True))
    elif args.output_format == "pkl":
      # Note: This will print binary data to stdout.
      # Best to redirect to a file, e.g., > output.pkl
      result_df.to_pickle(sys.stdout.buffer)

  except (FileNotFoundError, IOError, TypeError, ValueError) as e:
    print(f"Error: {e}")


if __name__ == "__main__":
  main()
