#!/bin/bash

source ./.venv/bin/activate

python3 -m napolitan.representative_opinion_ranking.main \
    --r1_url "https://docs.google.com/spreadsheets/d/1K_MybacxSpSnr4Rs6x23rhxGVf3zmz3doVfvAgdmh9Q/edit?gid=1029562411#gid=1029562411" --verbose

