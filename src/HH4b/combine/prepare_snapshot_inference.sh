#!/bin/bash

run_unblinded_hh4b.sh --workspace --bfit --passbin=0
extract_fit_result.py higgsCombineSnapshotBOnly.MultiDimFit.mH125.root "w:MultiDimFit" "inject_combined.json" --keep 'CMS_bbbb_hadronic_tf_dataResidual_*'
inject_fit_result.py inject_combined.json HHModel.root HHModel

combineCards.py fail=fail.txt passvbf=passvbf.txt > passvbf_nomasks.txt
combineCards.py fail=fail.txt passbin1=passbin1.txt > passbin1_nomasks.txt
combineCards.py fail=fail.txt passbin2=passbin2.txt > passbin2_nomasks.txt
combineCards.py fail=fail.txt passbin3=passbin3.txt > passbin3_nomasks.txt

for card in combined.txt passvbf_nomasks.txt passbin1_nomasks.txt passbin2_nomasks.txt passbin3_nomasks.txt; do
    add_parameter.py "${card}" signal_norm_xsbr group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH BR_hbb -d none
    add_parameter.py "${card}" signal_norm_xs   group = THU_HH pdf_Higgs_ggHH pdf_Higgs_qqHH QCDscale_qqHH -d none
    prettify_datacard.py --no-preamble "${card}" -d none
done
