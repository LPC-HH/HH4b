#!/bin/bash

TXbb_bins=(0p99to1p0 0p975to0p99 0p95to0p975)
pt_bins=(550to10000 350to550)
years=(2022 2023)
fit_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


for year in "${years[@]}"; do
  for txbb in "${TXbb_bins[@]}"; do
    for pt in "${pt_bins[@]}"; do
        echo "Processing year: ${year}, TXbb: ${txbb}, pT: ${pt}"
        passbin="TXbb${txbb}pT${pt}"
        cards_dir="${fit_dir}/cards/${year}All"
        passbin_dir="${cards_dir}/${passbin}"
        mkdir -p "${passbin_dir}"
        cd "${passbin_dir}" || exit
        "${fit_dir}"/fit_zbb.sh --workspace --bfit --dfit --passbin "${passbin}" --cards_dir "${cards_dir}" 2>&1 | tee fit.log
        rm -rf "${passbin_dir}/outs"
        mv "${cards_dir}/outs" "${passbin_dir}"
    done
  done
done
