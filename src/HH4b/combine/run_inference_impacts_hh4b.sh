#!/bin/bash
# shellcheck disable=SC2086,SC2034

card_dir=./
datacards=$card_dir/combined_nomasks.root
model=hh_model.model_default@noNNLOscaling@noklDependentUnc
campaign="61 fb$^{-1}$, 2022-2023 (13.6 TeV)"

law run PlotPullsAndImpacts \
    --version dev \
    --hh-model $model \
    --datacards $datacards \
    --pois r \
    --mc-stats \
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
