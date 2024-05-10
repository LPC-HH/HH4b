#!/bin/bash

card_dir=/uscms/home/jduarte1/nobackup/HH4b/src/HH4b/cards/run3-bdt-may9-msd40-v2
datacards=$card_dir/passbin3_snapshot.root:$card_dir/passbin2_snapshot.root:$card_dir/passbin1_snapshot.root:$card_dir/combined_snapshot.root
masks="mask_passbin1=0:mask_passbin2=0:mask_passbin3=0:mask_fail=0:mask_passbin1MCBlinded=1:mask_passbin2MCBlinded=1:mask_passbin3MCBlinded=1:mask_failMCBlinded=1"
model=hh_model.model_default@noNNLOscaling@noklDependentUnc

law run PlotUpperLimitsAtPoint \
    --version dev  \
    --multi-datacards $datacards \
    --parameter-values $masks \
    --h-lines 1 \
    --x-log True \
    --hh-model $model \
    --datacard-names "Category 3,Category 2,Category 1,Combined" \
    --remove-output 0,a,y \
    --campaign "61 fb$^{-1}$, 2022-2023 (13.6 TeV)" \
    --use-snapshot False \
    --file-types pdf,png,root,c
