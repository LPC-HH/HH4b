#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
C2V="1"
param="r"
unblinded="False"
float=""
while getopts ":c:p:f:is:u" opt; do
  case $opt in
    c)
      C2V=$OPTARG
      ;;
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

modelopt=""
if [[ "$float" == "r_gghh" ]]; then
    modelopt="@doProfilergghh=flat"
elif [[ "$float" == "r_qqhh" ]]; then
    modelopt="@doProfilerqqhh=flat"
fi

card_dir=./
datacards="${card_dir}/passbin3_nomasks.txt${inj}:${card_dir}/passbin2_nomasks.txt${inj}:${card_dir}/passbin1_nomasks.txt${inj}:${card_dir}/passvbf_nomasks.txt${inj}:${card_dir}/combined_nomasks.txt${inj}"
datacard_names="ggHH SR 3,ggHH SR 2,ggHH SR 1,qqHH SR,Combined"
datacard_order="0,1,3,2,4"
if [[ "$float" == "r_gghh" ]]; then
    parameters="C2V=${C2V}:r_qqhh=0"
elif [[ "$float" == "r_qqhh" ]]; then
    parameters="C2V=${C2V}:r_gghh=0"
else
    parameters="C2V=${C2V}"
fi

if [[ "$C2V" != "1" ]]; then
    xmin="0.03"
else
    xmin="0.75"
fi

model="hh_model_run23.model_default_run3${modelopt}"
campaign="62 fb$^{-1}$ (13.6 TeV)"


export DHI_CMS_POSTFIX="Supplementary"
law run PlotUpperLimitsAtPoint \
    --version dev  \
    --multi-datacards "$datacards" \
    --datacard-order "$datacard_order" \
    --pois "$param" \
    --parameter-values "$parameters" \
    --parameter-ranges "r_gghh=-10,10" \
    --h-lines 1 \
    --x-log True \
    --x-min "$xmin" \
    --hh-model "$model" \
    --datacard-names "$datacard_names" \
    --remove-output 0,a,y \
    --campaign "$campaign" \
    --use-snapshot False \
    --unblinded "$unblinded" \
    --file-types pdf,png,root,c $frozen
