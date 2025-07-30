#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# Script for fits
#
# 1) Combines cards and makes a workspace (--workspace / -w)
# 2) Background-only fit (--bfit / -b)
# 3) Expected asymptotic limits (--limits / -l)
# 4) Expected significance (--significance / -s)
# 5) Fit diagnostics (--dfit / -d)
# 6) GoF on data (--gofdata / -g)
# 7) GoF on toys (--goftoys / -t),
# 8) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impacts $nuisance), collect (--impactsc $nuisances)
# 9) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
#
# Specify seed with --seed (default 42) and number of toys with --numtoys (default 100)
#
# Usage ./run_unblinded_hh4b.sh [-wblsdgt] [--numtoys 100] [--seed 42] [--passbin 1]
# --passbin X will do the fit only for bin X, or if X = 0 (default), will do for all
#
# Author: Raghav Kansal
####################################################################################################


####################################################################################################
# Read options
####################################################################################################

workspace=0
bfit=0
limits=0
significance=0
dfit=0
gofdata=0
goftoys=0
impacts=0
seed=42
numtoys=100
bias=-1
passbin=0
cards_dir=0

options=$(getopt -o "wblsdgti" --long "workspace,bfit,limits,significance,dfit,gofdata,goftoys,impacts:,dNLL_scan,bias:,seed:,numtoys:,passbin:,cards_dir:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -w|--workspace)
            workspace=1
            ;;
        -b|--bfit)
            bfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        -s|--significance)
            significance=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -g|--gofdata)
            gofdata=1
            ;;
        -t|--goftoys)
            goftoys=1
            ;;
        -i|--impacts)
            shift
            impacts=$1
            ;;
        --dNLL_scan)
            dNLL_scan=1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --bias)
            shift
            bias=$1
            ;;
        --passbin)
            shift
            passbin=$1
            ;;
        --cards_dir)
            shift
            cards_dir=$1
            ;;
        --)
            shift
            break;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
    shift
done

echo "Arguments: workspace=$workspace bfit=$bfit limits=$limits \
significance=$significance dfit=$dfit gofdata=$gofdata goftoys=$goftoys \
seed=$seed numtoys=$numtoys passbin=$passbin cards_dir=$cards_dir"


####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is masked off)
####################################################################################################

dataset=data_obs
ws=${cards_dir}/combined
wsm=${ws}
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir=${cards_dir}/outs
mkdir -p $outsdir

# args
ccargs="fail=${cards_dir}/fail.txt passbin${passbin}=${cards_dir}/pass${passbin}.txt"
rmin="-10"
rmax="10"

mintol=0.1  # --cminDefaultMinimizerTolerance

# floating parameters using var{} floats a bunch of parameters which shouldn't be floated,
# so countering this inside --freezeParameters which takes priority.
# Although, practically even if those are set to "float", I didn't see them ever being fitted,
# so this is just to be extra safe.
unblindedparams="--freezeParameters var{.*_In},var{.*__norm},var{n_exp_.*}"

excludeimpactparams='rgx{.*tf_dataResidual_Bin.*},rgx{.*_mcstat_.*}'

echo "cc args:"
echo "$ccargs"

echo "unblinded params:"
echo "$unblindedparams"

####################################################################################################
# Combine cards, text2workspace, fit, limits, significances, fitdiagnositcs, GoFs
####################################################################################################

# need to run this for large # of nuisances
# https://cms-talk.web.cern.ch/t/segmentation-fault-in-combine/20735
ulimit -s unlimited

if [ $workspace = 1 ]; then
    echo "Combining cards"
    # shellcheck disable=SC2086
    combineCards.py $ccargs > $ws.txt

    echo "Running text2workspace"
    # text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt
    # new version got rid of -D arg??
    text2workspace.py $ws.txt -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt
else
    if [ ! -f "$wsm.root" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi


if [ $bfit = 1 ]; then
    echo "Multidim fit"
    combine -D $dataset -M MultiDimFit --saveWorkspace --algo singles -m 125 -d ${wsm}.root -v 9 --rMin $rmin --rMax $rmax \
    --cminDefaultMinimizerStrategy 1 --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt
fi

if [ $limits = 1 ]; then
    echo "Limits"
    combine -M AsymptoticLimits -m 125 -n "" -d $wsm.root -v 9 --rMax $rmax \
    --saveWorkspace --saveToys -s "$seed" --toysFrequentist 2>&1 | tee $outsdir/AsymptoticLimits.txt
fi


if [ $significance = 1 ]; then
    echo "Significance"
    combine -M Significance -m 125 -n "" -d $wsm.root -v 9 --rMax $rmax \
    --saveWorkspace --saveToys -s "$seed" --toysFrequentist 2>&1 | tee $outsdir/Significance.txt
fi


if [ $dfit = 1 ]; then
    echo "Fit Diagnostics"
    combine -M FitDiagnostics -m 125 -d ${wsm}.root --rMin $rmin --rMax $rmax \
    --cminDefaultMinimizerStrategy 0  --cminDefaultMinimizerTolerance "$mintol" --X-rtd MINIMIZER_MaxCalls=400000 \
    -n Unblinded --ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt

    # echo "Fit Shapes"
    # PostFitShapesFromWorkspace --dataset "$dataset" -w ${wsm}.root --output FitShapes.root \
    # -m 125 -f fitDiagnosticsUnblinded.root:fit_b --postfit --print 2>&1 | tee $outsdir/FitShapes.txt

    echo "Fit Shapes"
    PostFitShapesFromWorkspace --dataset "$dataset" -w ${wsm}.root --output FitShapes.root \
    -m 125 -f fitDiagnosticsUnblinded.root:fit_s --postfit --print 2>&1 | tee $outsdir/FitShapes.txt
fi


if [ $gofdata = 1 ]; then
    echo "GoF on data"
    combine -M GoodnessOfFit -d $wsm.root --algo saturated -m 125 --rMin $rmin --rMax $rmax \
    -n Data -v 9 2>&1 | tee $outsdir/GoF_data.txt
fi


if [ "$goftoys" = 1 ]; then
    echo "GoF on toys"

    echo "Get expected r value"
    rexp=$(python3 -c 'import uproot; print(uproot.open("'${wsm_snapshot}'.root")["limit"].arrays("r")[b"r"][0])')

    combine -M GoodnessOfFit -d $wsm_snapshot.root --algo saturated -m 125 --rMin $rmin --rMax $rmax \
    --snapshotName MultiDimFit  --bypassFrequentistFit --trackParameters r --expectSignal $rexp \
    -n Toys -v 9 -s "$seed" -t "$numtoys" --saveToys --toysFrequentist 2>&1 | tee $outsdir/GoF_toys.txt
fi



if [ "$impacts" != 0 ]; then
    echo "Submitting jobs for impact scans"
    # # Impacts module cannot access parameters which were frozen in MultiDimFit, so running impacts
    # # for each parameter directly using its internal command
    # # (also need to do this for submitting to condor anywhere other than lxplus)
    # combine -M MultiDimFit -n _paramFit_impacts_"$impacts" --algo impact --redefineSignalPOIs r -P "$impacts" \
    # --floatOtherPOIs 1 --saveInactivePOI 1 --snapshotName MultiDimFit -d ${wsm_snapshot}.root \
    # --robustFit 1 ${unblindedparams} \
    # --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 1 -m 125 | tee $outsdir/Impacts_"$impacts".txt

    # # Initial fit
    # combineTool.py -M Impacts --doInitialFit --snapshotName MultiDimFit -m 125 -n "impacts" \
    # -d ${wsm_snapshot}.root --robustFit 1 ${unblindedparams} \
    #  --cminDefaultMinimizerStrategy=1 -v 1 2>&1 | tee $outsdir/Impacts_init.txt

    # # optional --dry-run --job-mode interactive
    # combineTool.py -M Impacts --doFits --snapshotName MultiDimFit \
    # -m 125 -n "impacts" -d ${wsm_snapshot}.root --robustFit 1 \
    # --exclude ${excludeimpactparams} \
    # --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 1 2>&1 | tee $outsdir/Impacts_fits.txt

    impact_common_args="-M Impacts -m 125 --snapshotName MultiDimFit --cminDefaultMinimizerStrategy=1 -v 1 -d ${wsm_snapshot}.root -n impacts"

    # Initial fit
    combineTool.py --doInitialFit ${impact_common_args} \
    -d ${wsm_snapshot}.root --robustFit 1 ${unblindedparams} 2>&1 | tee $outsdir/Impacts_init.txt

    combineTool.py --doFits ${impact_common_args} \
    --exclude ${excludeimpactparams} \
    --setParameterRanges r=-0.5,20 2>&1 | tee $outsdir/Impacts_fits.txt

    # crab output and make plots
    combineTool.py ${impact_common_args} --exclude ${excludeimpactparams} -o impacts.json
    plotImpacts.py -i impacts.json -o impacts
fi

if [ "$dNLL_scan" != 0 ]; then
    combine -v9 -M MultiDimFit --algo grid -m 125 -n "Scan" --rMin -2 --rMax 2 "${ws}.txt" 2>&1 | tee "${outsdir}/dnll_scan.txt"
    plot1DScan.py "higgsCombineScan.MultiDimFit.mH125.root" -o scan
fi


if [ "$bias" != -1 ]; then
    echo "Bias test with bias $bias"
    # setting verbose > 0 here can lead to crazy large output files (~10-100GB!) because of getting
    # stuck in negative yield areas

    combine -M FitDiagnostics --trackParameters r --trackErrors r --justFit \
    -m 125 -n "bias${bias}_passbin${passbin}" -d ${wsm_snapshot}.root --rMin ${rmin} --rMax ${rmax} \
    --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal "$bias" \
    ${unblindedparams} \
    --robustFit=1 -t "$numtoys" -s "$seed" \
    --X-rtd MINIMIZER_MaxCalls=1000000 --cminDefaultMinimizerTolerance 0.1 2>&1 | tee "$outsdir/bias${bias}seed${seed}.txt"
fi
