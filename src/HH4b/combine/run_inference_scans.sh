#!/bin/bash
# shellcheck disable=SC2086

syst="full"
inj=""
param="kl"
unblinded="False"
xsec="False"
while getopts ":p:is:ux" opt; do
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
      unblinded="True"
      ;;
    x)
      xsec="True"
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

xsecbr=""
if [[ "$xsec" == "True" ]]; then
   xsecbr="--xsec fb --frozen-groups signal_norm_xsbr --br bbbb"
fi

card_dir=./
datacards="${card_dir}/combined_nomasks.txt${inj}"
model=hh_model_run23.model_default_run3
campaign="62 fb$^{-1}$ (13.6 TeV)"

export DHI_CMS_POSTFIX="Preliminary"
law run PlotUpperLimits \
    --version dev  \
    --datacards "$datacards" \
    --hh-model "$model" \
    --remove-output 0,a,y \
    --campaign "$campaign" \
    --use-snapshot False \
    --file-types pdf,png,root,c $xsecbr \
    --pois r \
    --scan-parameters "$parameters" \
    --y-log \
    --unblinded "$unblinded" \
    --save-ranges $frozen
