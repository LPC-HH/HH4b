#!/bin/bash

card_dir=./
datacards="${card_dir}/combined.txt<i"
model=hh_model_run23.model_default_run3
campaign="61 fb$^{-1}$, 2022-2023 (13.6 TeV)"

law run PlotUpperLimits \
    --version dev  \
    --datacards "$datacards" \
    --hh-model "$model" \
    --remove-output 0,a,y \
    --campaign "$campaign" \
    --use-snapshot False \
    --file-types pdf,png,root,c \
    --xsec fb \
    --pois r \
    --frozen-groups signal_norm_xsbr \
    --scan-parameters C2V,0,2,11 \
    --br bbbb \
    --y-log \
    --save-ranges
