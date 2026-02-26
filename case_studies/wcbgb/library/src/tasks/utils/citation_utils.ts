// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Functions for formatting citations.

import { Comment, isVoteTallyType, VoteTally } from "../../types";

/**
 * Create citations for comments in the format of "[12, 43, 56]"
 * @param comments the comments to use for citations
 * @returns the formatted citations as a string
 */
export function getCommentCitations(comments: Comment[]): string {
  return "[" + comments.map((comment) => commentCitation(comment)).join(", ") + "]";
}

/**
 * Get the text that should be visible on hover for a citation.
 *
 * This includes the text and vote information.
 *
 * @param comment the comment to use for the numbers and text.
 * @returns
 */
export function commentCitationHoverOver(comment: Comment) {
  const hoverText = `${comment.text.replace(/"/g, '\\"').replace(/\n/g, " ")}`;
  if (comment.voteInfo) {
    return hoverText + `\n${voteInfoToString(comment)}`;
  } else {
    return hoverText;
  }
}
/**
 * Utility function for displaying a concise textual summary of a comment as Markdown
 *
 * This includes the text and vote information.
 *
 * @param comment
 * @returns the summary as a string
 */
export function commentCitation(comment: Comment): string {
  return `[${comment.id}](## "${commentCitationHoverOver(comment)}")`;
}

/**
 * Display a summary of a comment (text and votes) as a citation in HTML.
 * @param comment the comment to summarize
 * @returns the html element with the comment id and more info on hover over.
 */
export function commentCitationHtml(comment: Comment): string {
  return "<a href='##' title='" + commentCitationHoverOver(comment) + "'>" + comment.id + `</a>`;
}
/**
 * Utility function for displaying a concise textual summary of the vote tally patterns for a given comment
 * @param comment the comment with the VoteInfo to display
 * @returns the summary as a string
 */
export function voteInfoToString(comment: Comment): string {
  if (!comment.voteInfo) {
    return "";
  }
  if (isVoteTallyType(comment.voteInfo)) {
    return `Votes: (${voteTallyToString(comment.voteInfo)})`;
  } else {
    return Object.entries(comment.voteInfo as object).reduce((acc, [key, value]) => {
      return acc + ` ${key}(${voteTallyToString(value as VoteTally)})`;
    }, "Votes:");
  }
}

function voteTallyToString(voteTally: VoteTally): string {
  let text = `Agree=${voteTally.agreeCount}, Disagree=${voteTally.disagreeCount}`;
  if (voteTally.passCount) {
    text += `, Pass=${voteTally.passCount}`;
  }
  return text;
}

/**
 * Replace citation notation with hoverover links for analysis
 * @param comments
 * @param summary
 * @returns the markdown summary
 */
export function formatCitations(comments: Comment[], summary: string): string {
  // Regex for capturing all the ^[n,m] citation annotations from the summary (post grounding).
  const groundingCitationRegex = /\[([\d,\s]+)\]/g;
  // Create a mapping of comment ids to comment records.
  const commentIndex = comments.reduce((acc, curr) => acc.set(curr.id, curr), new Map());

  // Find every match of citation annotations and replace cited comment ids with markdown links.
  const summaryWithLinks = summary.replace(groundingCitationRegex, (_, match: string): string => {
    // Extract the individual comment ids from the match.
    const commentIds = match.split(/,\s*/);
    // Map to markdown links that display the comment text and vote patterns when you hover over.
    const mdLinks = commentIds.map((commentId) => commentCitation(commentIndex.get(commentId)));

    return "[" + mdLinks.join(", ") + "]";
  });
  // For debugging, add commentTable for searching comments that might have been removed at previous steps.
  //return summaryWithLinks + commentTable(comments);
  return summaryWithLinks;
}
