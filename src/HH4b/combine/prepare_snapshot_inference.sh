#!/bin/bash


run_blinded_hh4b.sh --workspace --bfit --passbin=0
extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject.json" --keep '*'
set_global_obs.py "inject_original.json" "inject.json"

combineCards.py fail=fail.txt passvbf=passvbf.txt > passvbf_nomasks.txt
combineCards.py fail=fail.txt passbin1=passbin1.txt > passbin1_nomasks.txt
combineCards.py fail=fail.txt passbin2=passbin2.txt > passbin2_nomasks.txt
combineCards.py fail=fail.txt passbin3=passbin3.txt > passbin3_nomasks.txt
combineCards.py fail=fail.txt passbin1=passbin1.txt passbin2=passbin2.txt passbin3=passbin3.txt passvbf=passvbf.txt > combined_nomasks.txt

for card in combined_nomasks.txt passvbf_nomasks.txt passbin1_nomasks.txt passbin2_nomasks.txt passbin3_nomasks.txt; do
    add_parameter.py "${card}" signal_norm_xsbr group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH BR_hbb -d none
    add_parameter.py "${card}" signal_norm_xs   group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH -d none
    prettify_datacard.py --no-preamble "${card}" -d none
done
