#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="kl"
unblinded="False"
float=""
while getopts ":p:f:is:u" opt; do
  case $opt in
    p)
      param=$OPTARG
      ;;
    f)
      float=$OPTARG
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

xmax=""
if [[ "$param" == "kl" ]]; then
    parameters="kl,-15,20,36"
    show="kt,CV,C2V"
elif [[ "$param" == "C2V" ]]; then
    parameters="C2V,0,2.1,22"
    show="kl,kt,CV"
    xmax="--x-max 2"
elif [[ "$param" == "r" ]]; then
    parameters="r,-5,10,16"
    show="kl,kt,CV,C2V"
elif [[ "$param" == "r_gghh" ]]; then
    parameters="r_ggh,-5,10,16"
    show="r,r_qqhh,kl,kt,CV,C2V"
elif [[ "$param" == "r_qqhh" ]]; then
    parameters="r_qqhh,-200,400,16"
    show="r,r_gghh,kl,kt,CV,C2V"
else
    echo "Invalid param argument"
    exit 1
fi

modelopt=""
if [[ "$float" == "r_gghh" ]]; then
    modelopt="@doProfilergghh=flat"
elif [[ "$float" == "r_qqhh" ]]; then
    modelopt="@doProfilerqqhh=flat"
fi

card_dir=./
command="PlotLikelihoodScan"
datacards="${card_dir}/combined_nomasks.txt${inj}"
datacardopt="--datacards"
datacardnames=""
model="hh_model_run23.model_default_run3${modelopt}"
campaign="62 fb$^{-1}$ (13.6 TeV)"

if [[ "$unblinded" != "False" ]]; then
  command="PlotMultipleLikelihoodScans"
  datacards="${card_dir}/combined_nomasks.txt${inj}:${card_dir}/combined_nomasks.txt${inj}"
  datacardopt="--multi-datacards"
  datacardnames="--datacard-names Expected,Observed"
fi

export DHI_CMS_POSTFIX="Preliminary"
law run $command \
    --version dev \
    --hh-model "$model" \
    $datacardopt "$datacards" $datacardnames \
    --pois "$param" \
    --show-parameters "$show" \
    --scan-parameters "$parameters" \
    --file-types "pdf,png" \
    --campaign "$campaign" \
    --remove-output 0,a,y \
    --use-snapshot False \
    --show-best-fit-error False \
    --unblinded "$unblinded" \
    --save-ranges $frozen $xmax
