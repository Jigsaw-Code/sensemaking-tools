#!/bin/bash

# This script replicates the functionality of the old sheet_loader_util.py script.

source ./.venv/bin/activate

python3 -m napolitan.proposition_refinement.main \
    --r1_url "https://docs.google.com/spreadsheets/d/1K_MybacxSpSnr4Rs6x23rhxGVf3zmz3doVfvAgdmh9Q/edit?gid=1029562411#gid=1029562411" \
    --r2_url "https://docs.google.com/spreadsheets/d/1K_MybacxSpSnr4Rs6x23rhxGVf3zmz3doVfvAgdmh9Q/edit?gid=51426834#gid=51426834" \
    --propositions_url "https://docs.google.com/spreadsheets/d/1Rx6jIgbvlAC0CqkstdN8IW2pDqZIbQgKZcP_UdbLMvY/edit?resourcekey=0-BKd1GXM7d-4gTLZnm37IyA&gid=0#gid=0" \
    --propositions_range "C4:C172" \
    --sample_size 10 \
    --verbose