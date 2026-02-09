#!/usr/bin/env python3
import csv
import sys

def convert_polis_csv(input_file, output_file):
    """Convert Polis export CSV to sensemaker format"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        # Define output fields
        fieldnames = ['comment-id', 'comment_text', 'votes', 'agrees', 'disagrees', 'passes']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            agrees = int(row.get('agrees', 0))
            disagrees = int(row.get('disagrees', 0))
            passes = 0  # Polis export doesn't include passes
            votes = agrees + disagrees + passes

            new_row = {
                'comment-id': row['comment-id'],
                'comment_text': row['comment-body'],
                'votes': votes,
                'agrees': agrees,
                'disagrees': disagrees,
                'passes': passes
            }

            writer.writerow(new_row)

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else './files/polis_comments.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else './files/comments.csv'

    convert_polis_csv(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")
