#!/bin/bash

run_unblinded_hh4b.sh --workspace --bfit --passbin=0
extract_fit_result.py higgsCombineSnapshotBOnly.MultiDimFit.mH125.root "w:MultiDimFit" "inject_combined.json" --keep 'CMS_bbbb_hadronic_tf_dataResidual_*'
inject_fit_result.py inject_combined.json HHModel.root HHModel

add_parameter.py "combined.txt" signal_norm_xsbr group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH BR_hbb -d none
add_parameter.py "combined.txt" signal_norm_xs   group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH -d none

prettify_datacard.py --no-preamble "combined.txt" -d none