#!/bin/bash

run_blinded_hh4b.sh --workspace --bfit --passbin=0
extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject_combined.json" --keep '*'
if [ -f "passvbf.txt" ]; then
    combineCards.py fail=fail.txt passbin1=passbin1.txt passbin2=passbin2.txt  passbin3=passbin3.txt   passvbf=passvbf.txt  > combined_nomasks.txt
else
    combineCards.py fail=fail.txt passbin1=passbin1.txt passbin2=passbin2.txt  passbin3=passbin3.txt  > combined_nomasks.txt
fi
text2workspace.py combined_nomasks.txt
inject_fit_result.py inject_combined.json combined_nomasks.root w

if [ -f "passvbf.txt" ]; then
    run_blinded_hh4b.sh --workspace --bfit --passbin=vbf
    extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject_passvbf.json" --keep '*'
    combineCards.py fail=fail.txt   passvbf=passvbf.txt  > passvbf_nomasks.txt
    text2workspace.py passvbf_nomasks.txt
    inject_fit_result.py inject_passvbf.json passvbf_nomasks.root w
fi

run_blinded_hh4b.sh --workspace --bfit --passbin=1
extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject_passbin1.json" --keep '*'
combineCards.py fail=fail.txt   passbin1=passbin1.txt  > passbin1_nomasks.txt
text2workspace.py passbin1_nomasks.txt
inject_fit_result.py inject_passbin1.json passbin1_nomasks.root w

run_blinded_hh4b.sh --workspace --bfit --passbin=2
extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject_passbin2.json" --keep '*'
combineCards.py fail=fail.txt   passbin2=passbin2.txt  > passbin2_nomasks.txt
text2workspace.py passbin2_nomasks.txt
inject_fit_result.py inject_passbin2.json passbin2_nomasks.root w

run_blinded_hh4b.sh --workspace --bfit --passbin=3
extract_fit_result.py higgsCombineSnapshot.MultiDimFit.mH125.root "w:MultiDimFit" "inject_passbin3.json" --keep '*'
combineCards.py fail=fail.txt   passbin3=passbin3.txt  > passbin3_nomasks.txt
text2workspace.py passbin3_nomasks.txt
inject_fit_result.py inject_passbin3.json passbin3_nomasks.root w
