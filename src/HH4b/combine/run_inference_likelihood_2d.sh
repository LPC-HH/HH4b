#!/bin/bash
# shellcheck disable=SC2086

model=hh_model_run23.model_default_run3
campaign="62 fb$^{-1}$ (13.6 TeV)"
card_dir=./
datacards="${card_dir}/combined.txt"

export DHI_CMS_POSTFIX="Preliminary"
law run PlotLikelihoodScan \
    --version dev \
    --hh-model "$model" \
    --datacards "$datacards" \
    --pois kl,C2V \
    --scan-parameters kl,-26,36,63:C2V,-0.5,2.5,49 \
    --y-min -0.5 \
    --y-max 2.5 \
    --x-min -24 \
    --x-max 34 \
    --show-parameters kt,CV \
    --show-best-fit True \
    --show-best-fit-error False \
    --recompute-best-fit False \
    --file-types "pdf,png" \
    --campaign "$campaign" \
    --y-log \
    --z-min 0.01 \
    --z-max 10000 \
    --unblinded True \
    --show-significances 1,2,3 \
    --shift-negative-values \
    --LikelihoodScan-workflow "htcondor" \
    --LikelihoodScan-tasks-per-job 10 \
    --LikelihoodScan-htcondor-cpus 2 \
    --LikelihoodScan-max-runtime 48h \
    --workers 4 \
    --remove-output 0,a,y \
    --interpolate-above 999 \
    --interpolate-nans \
    --interpolation-method rbf,multiquadric,1,0.2