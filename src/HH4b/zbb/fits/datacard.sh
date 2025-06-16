#!/bin/bash


ZBB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
templates_dir=${ZBB_DIR}/templates_zbb

for year in 2022All 2023All; do
    cards_dir="${ZBB_DIR}/fits/cards/${year}"
    python3 "${ZBB_DIR}/fits/CreateDatacard.py" \
        --templates-dir "${templates_dir}" \
        --cards-dir "${cards_dir}" \
        --year "$year" \
        --nTF 1
done
