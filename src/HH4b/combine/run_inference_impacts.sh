#!/bin/bash
# shellcheck disable=SC2086,SC2034

inj=""
unblinded="False"
while getopts ":iu" opt; do
  case $opt in
    i)
      inj="<i"
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

card_dir=./
datacards="${card_dir}/combined_nomasks.txt${inj}"
model=hh_model_run23.model_default_run3
campaign="61 fb$^{-1}$, 2022-2023 (13.6 TeV)"

if [[ "$inj" == "<i" ]]; then
    mv inject.json inject_original.json
    set_global_obs.py inject_original.json inject.json
fi

law run PlotPullsAndImpacts \
    --version dev \
    --hh-model "$model" \
    --datacards "$datacards" \
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
    --unblinded "$unblinded" \
    --remove-output 0,a,y
