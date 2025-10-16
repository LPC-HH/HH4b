#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="r"
C2V="1"
unblinded="False"
while getopts ":c:p:is:u" opt; do
  case $opt in
    c)
      C2V=$OPTARG
      ;;
    p)
      param=$OPTARG
      ;;
    i)
      inj="<i"
      ;;
    s)
      syst=$OPTARG
      ;;
    u)
      unblinded="True"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ "$syst" == "full" ]]; then
    frozen=""
elif [[ "$syst" == "bkgd" ]]; then
    frozen="--frozen-parameters allConstrainedNuisances"
elif [[ "$syst" == "stat" ]]; then
    frozen="--frozen-parameters allConstrainedNuisances,var{CMS_bbbb_hadronic_tf_dataResidual.*}"
else
    echo "Invalid syst argument"
    exit 1
fi

card_dir=./
command="PlotPostfitSOverB"
datacards="${card_dir}/combined_nomasks.txt${inj}"
datacardopt="--datacards"
datacardnames=""
model=hh_model_run23.model_default_run3
campaign="62 fb$^{-1}$, 2022-2023 (13.6 TeV)"
parameters="C2V=${C2V}"


law run $command \
    --version dev \
    --hh-model "$model" \
    $datacardopt "$datacards" $datacardnames \
    --pois "$param" \
    --parameter-values "$parameters" \
    --file-types "pdf,png" \
    --campaign "$campaign" \
    --remove-output 0,a,y \
    --use-snapshot False \
    --unblinded "$unblinded" $frozen
