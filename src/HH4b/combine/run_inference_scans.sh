#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="kl"
while getopts ":p:is:" opt; do
  case $opt in
    p)
      param=$OPTARG
      ;;
    i)
      inj="<i"
      ;;
    s)
      syst=$OPTARG
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

if [[ "$param" == "kl" ]]; then
    parameters="kl,-15,20,36"
elif [[ "$param" == "C2V" ]]; then
    parameters="C2V,0,2,21"
else
    echo "Invalid param argument"
    exit 1
fi

card_dir=./
datacards="${card_dir}/combined_nomasks.txt${inj}"
model=hh_model_run23.model_default_run3
campaign="62 fb$^{-1}$, 2022-2023 (13.6 TeV)"

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
    --scan-parameters "$parameters" \
    --br bbbb \
    --y-log \
    --save-ranges $frozen
