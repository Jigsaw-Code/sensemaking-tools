#!/usr/bin/env python3

import argparse as arg
import itertools
import re
import pandas as pd


print("Starting process_polis_data.py program")

# argparse setup with arguments for two input files


def getargs():
  parser = arg.ArgumentParser(
      description="Process Polis data from the openData export data."
  )
  parser.add_argument("export_directory", help="Path to export directory.")
  parser.add_argument(
      "--participants-votes", help="Participants votes file (override)."
  )
  parser.add_argument(
      "--comments", help="Path to the comments file (override)."
  )
  parser.add_argument(
      "-o", "--output_file", help="Path to the output CSV file.", required=True
  )
  parser.add_argument(
      "--exclude-ungrouped-participants",
      help="Whether to include ungrouped participants in the output.",
      action="store_true",
  )
  args = parser.parse_args()
  args.participants_votes = (
      args.participants_votes
      or f"{args.export_directory}/participants-votes.csv"
  )
  args.comments = args.comments or f"{args.export_directory}/comments.csv"
  return args


print("Processing args")
args = getargs()

# Read the CSV files into pandas DataFrames
try:
  votes = pd.read_csv(args.participants_votes)
  comments = pd.read_csv(args.comments)
except FileNotFoundError as e:
  print(f"Error: One or both input files not found: {e}")
  exit(1)
except pd.errors.EmptyDataError as e:
  print(f"Error: One or both input files are empty: {e}")
  exit(1)
except pd.errors.ParserError as e:
  print(f"Error parsing CSV file: {e}")
  exit(1)

print("Args processed")

# make sure to cast comment ids as ints
comments["comment-id"] = comments["comment-id"].astype(int)


# their votes on everything.
if args.exclude_ungrouped_participants:
  # filter out votes rows where group-id is nan, and make ints
  print("Filtering out ungrouped participants")
  votes = votes[votes["group-id"].notna()]
else:
  # We fill the ungrouped participant records with -1 for group, which when
  # processed below will reserve group-0 for the "ungrouped", which we can
  # manually filter into the columns
  votes["group-id"] = votes["group-id"].fillna(-1)

# Increment group ids so they are 1 based instead of 0 (noting that, as described above,
# the "ungrouped" psuedo-group gets bumped here from -1 to 0, to be dealt with later)
votes["group-id"] = votes["group-id"].astype(int) + 1
# Sort the ids so they come out in the right order in the output file header
group_ids = sorted(votes["group-id"].unique())
print("Group ids:", group_ids)

# prompt: find all of the column names in the votes df that match a numeric regex
comment_ids = [col for col in votes.columns if re.match(r"^\d+$", col)]

# Create a dictionary for mapping comment to total vote count for each column in
# the votes table, for later verification
comment_vote_counts = {}
for comment_id in comment_ids:
  comment_vote_counts[int(comment_id)] = votes[comment_id].value_counts().sum()

# Melt the DataFrame
melted_votes = votes.melt(
    id_vars=["group-id"],
    value_vars=comment_ids,
    var_name="comment-id",
    value_name="value",
)
melted_votes["comment-id"] = melted_votes["comment-id"].astype(int)
# Group, count, unstack, and fill missing values
result = (
    melted_votes.groupby(["comment-id", "group-id"])["value"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns
result = result.rename(
    columns={-1: "disagree-count", 0: "pass-count", 1: "agree-count"}
)

# Pivot out the group-id column so that each of the vote count columns look like "group-N-VOTE-count"
pivoted = result.pivot(index="comment-id", columns="group-id")

# A function for naming groups based on group id.
# Note that for the group_id == 0, the "ungrouped" pseudo-group, this returns "Group-none"


def group_name(group_id):
  return "Group-" + ("none" if group_id == 0 else str(group_id))


# Use the pivoted data to prepare a dataframe for merging
for_merge = pd.DataFrame({"comment-id": pivoted.index})
for group_id in group_ids:
  for count_col in ["disagree-count", "pass-count", "agree-count"]:
    for_merge[group_name(group_id) + "-" + count_col] = pivoted[count_col][
        group_id
    ].values

# zero out total vote tallies since incorrect from filtering or database caching
comments["agrees"] = 0
comments["disagrees"] = 0
comments["passes"] = 0

# merge in the per group tallies above
comments = comments.merge(for_merge, on="comment-id")

# add up from the votes matrix for consistency
for group_id in group_ids:
  group = group_name(group_id)
  comments["disagrees"] += comments[group + "-disagree-count"]
  comments["agrees"] += comments[group + "-agree-count"]
  comments["passes"] += comments[group + "-pass-count"]

comments["votes"] = (
    comments["agrees"] + comments["disagrees"] + comments["passes"]
)

comments["agree_rate"] = comments["agrees"] / comments["votes"]
comments["disagree_rate"] = comments["disagrees"] / comments["votes"]
comments["pass_rate"] = comments["passes"] / comments["votes"]
comments["difference_of_opinion_rank"] = (
    1
    - abs(comments["agree_rate"] - comments["disagree_rate"])
    - comments["pass_rate"]
)


# Go through and check that all of our output comment["votes"] counts are no
# larger than the counts from our initial `comment_vote_counts` dictionary. We do
# not check for equality here, because it's possible that the counts get lower as
# a result of filters applied based on who was grouped in the conversation analysis.
print("Validating aggregate vote counts...")
failed_validations = 0
for comment_id in comments["comment-id"]:
  if (
      comment_vote_counts[comment_id]
      < comments[comments["comment-id"] == int(comment_id)]["votes"].iloc[0]
  ):
    print(
        f"WARNING: Vote count mismatch for comment {comment_id}. Original"
        f" count: {comment_vote_counts[comment_id]}, New count:"
        f" {comments[comments['comment-id'] == int(comment_id)]['votes'].iloc[0]}"
    )
    failed_validations += 1
if failed_validations == 0:
  print("All validations passed!")

# Leave only those comments explicitly moderated into the conversation, or that were
# left unmoderated, and accumulated votes (implying the conversation was likely set
# to non-strict moderation)
print("N comments total:", len(comments))
print("N votes total:", comments["votes"].sum())
moderated_comments = comments[
    (comments["moderated"] == 1)
    | ((comments["moderated"] == 0) & (comments["votes"] > 1))
]
print("N comments included after moderation:", len(moderated_comments))
print("N votes after moderation:", moderated_comments["votes"].sum())

# prompt: write out to a CSV file
moderated_comments = moderated_comments.rename(
    columns={"comment-body": "comment_text"}
)
moderated_comments.to_csv(args.output_file, index=False)

# Exit with non-zero error code if any validations failed
if failed_validations > 0:
  exit(1)
