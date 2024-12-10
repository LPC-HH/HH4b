#!/bin/bash

card_dir=./
datacards="${card_dir}/passbin3_nomasks.txt<i:${card_dir}/passbin2_nomasks.txt<i:${card_dir}/passbin1_nomasks.txt<i:${card_dir}/passvbf_nomasks.txt:${card_dir}/combined.txt<i"
datacard_names="Category 3,Category 2,Category 1,VBF Category,Combined"
xmin="0.03"
parameters="C2V=0"
model=hh_model_run23.model_default_run3
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
    --remove-output 2,a,y \
    --campaign "$campaign" \
    --use-snapshot False \
    --file-types pdf,png,root,c
#     --frozen-parameters "allConstrainedNuisances"
#     --frozen-parameters "allConstrainedNuisances,var{CMS_bbbb_hadronic_tf_dataResidual.*}"
