#!/bin/bash

card_dir=./
if [ -f "passvbf.txt" ]; then
    datacards=$card_dir/passbin3_nomasks.root:$card_dir/passbin2_nomasks.root:$card_dir/passbin1_nomasks.root:$card_dir/passvbf_nomasks.root:$card_dir/combined_nomasks.root
    datacard_names="Category 3,Category 2,Category 1,VBF Category,Combined"
    # xmin="0.03"
    # parameters="C2V=0"
    xmin="0.75"
    parameters="C2V=1"
else
    datacards=$card_dir/passbin3_nomasks.root:$card_dir/passbin2_nomasks.root:$card_dir/passbin1_nomasks.root:$card_dir/combined_nomasks.root
    datacard_names="Category 3,Category 2,Category 1,Combined"
    xmin="0.75"
    parameters="C2V=1"
fi
model=hh_model.model_default@noNNLOscaling@noklDependentUnc
campaign="61 fb$^{-1}$, 2022-2023 (13.6 TeV)"

law run PlotUpperLimitsAtPoint \
    --version dev  \
    --multi-datacards "$datacards" \
    --parameter-values "$parameters" \
    --h-lines 1 \
    --x-log True \
    --x-min "$xmin" \
    --hh-model "$model" \
    --datacard-names "$datacard_names" \
    --remove-output 1,a,y \
    --campaign "$campaign" \
    --use-snapshot False \
    --file-types pdf,png,root,c
