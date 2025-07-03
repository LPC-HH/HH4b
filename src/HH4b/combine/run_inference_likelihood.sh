#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="kl"
unblinded="False"
while getopts ":p:is:u" opt; do
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
    u)
      unblinded="False,True"
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
command="PlotLikelihoodScan"
datacards="${card_dir}/combined_nomasks.txt${inj}"
datacardopt="--datacards"
datacardnames=""
model=hh_model_run23.model_default_run3
campaign="62 fb$^{-1}$, 2022-2023 (13.6 TeV)"

if [[ "$unblinded" != "False" ]]; then
  command="PlotMultipleLikelihoodScans"
  datacards="${card_dir}/combined_nomasks.txt${inj}:${card_dir}/combined_nomasks.txt${inj}"
  datacardopt="--multi-datacards"
  datacardnames="--datacard-names Expected,Observed"
fi

law run $command \
    --version dev \
    --hh-model "$model" \
    $datacardopt "$datacards" $datacardnames \
    --pois "$param" \
    --scan-parameters "$parameters" \
    --file-types "pdf,png" \
    --campaign "$campaign" \
    --remove-output 0,a,y \
    --use-snapshot False \
    --unblinded "$unblinded" \
    --save-ranges $frozen
