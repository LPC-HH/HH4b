#!/bin/bash

card_dir=/uscms/home/jduarte1/nobackup/HH4b/src/HH4b/cards/run3-bdt-may9-msd40-v2
datacards=$card_dir/combined_snapshot.root
masks="mask_passbin1=0:mask_passbin2=0:mask_passbin3=0:mask_fail=0:mask_passbin1MCBlinded=1:mask_passbin2MCBlinded=1:mask_passbin3MCBlinded=1:mask_failMCBlinded=1"
model=hh_model.model_default@noNNLOscaling@noklDependentUnc
campaign="61 fb$^{-1}$, 2022-2023 (13.6 TeV)"

law run PlotPullsAndImpacts \
    --version dev \
    --hh-model $model \
    --datacards $datacards \
    --pois r \
    --mc-stats \
    --parameter-values "$masks" \
    --parameter-ranges r=-20,20 \
    --PullsAndImpacts-workflow "htcondor" \
    --PullsAndImpacts-tasks-per-job 10 \
    --parameters-per-page 40 \
    --order-by-impact \
    --file-types "pdf,png" \
    --page 0 \
    --campaign "$campaign" \
    --pull-range 1 \
    --remove-output 0,a,y
