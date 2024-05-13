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
# 8) Impacts: initial fit (--impactsi / -i), per-nuisance fits (--impactsf $nuisance), collect (--impactsc $nuisances)
# 9) Bias test: run a bias test on toys (using post-fit nuisances) with expected signal strength
#    given by --bias X.
#
# Specify seed with --seed (default 42) and number of toys with --numtoys (default 100)
#
# Usage ./run_blinded_hh4b.sh [-wblsdgt] [--numtoys 100] [--seed 42] [--passbin 1]
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
impactsi=0
impactsf=0
impactsc=0
seed=42
numtoys=100
bias=-1
passbin=0

options=$(getopt -o "wblsdgti" --long "workspace,bfit,limits,significance,dfit,gofdata,goftoys,impactsi,impactsf:,impactsc:,bias:,seed:,numtoys:,passbin:" -- "$@")
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
        -i|--impactsi)
            impactsi=1
            ;;
        --impactsf)
            shift
            impactsf=$1
            ;;
        --impactsc)
            shift
            impactsc=$1
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
seed=$seed numtoys=$numtoys passbin=$passbin"


####################################################################################################
# Set up fit arguments
#
# We use channel masking to "mask" the blinded and "unblinded" regions in the same workspace.
# (mask = 1 means the channel is masked off)
####################################################################################################

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125
CMS_PARAMS_LABEL="CMS_bbbb_hadronic"

outsdir=${cards_dir}/outs
mkdir -p $outsdir

# args
if [ $passbin = 0 ]; then
    ccargs="fail=${cards_dir}/fail.txt failMCBlinded=${cards_dir}/failMCBlinded.txt passbin1=${cards_dir}/passbin1.txt passbin1MCBlinded=${cards_dir}/passbin1MCBlinded.txt passbin2=${cards_dir}/passbin2.txt passbin2MCBlinded=${cards_dir}/passbin2MCBlinded.txt passbin3=${cards_dir}/passbin3.txt passbin3MCBlinded=${cards_dir}/passbin3MCBlinded.txt"
    maskunblindedargs="mask_passbin1=1,mask_passbin2=1,mask_passbin3=1,mask_fail=1,mask_passbin1MCBlinded=0,mask_passbin2MCBlinded=0,mask_passbin3MCBlinded=0,mask_failMCBlinded=0"
    maskblindedargs="mask_passbin1=0,mask_passbin2=0,mask_passbin3=0,mask_fail=0,mask_passbin1MCBlinded=1,mask_passbin2MCBlinded=1,mask_passbin3MCBlinded=1,mask_failMCBlinded=1"
    if [ -f "passvbf.txt" ]; then
        ccargs+=" passvbf=${cards_dir}/passvbf.txt passvbfMCBlinded=${cards_dir}/passvbfMCBlinded.txt"
        maskunblindedargs+=",mask_passvbf=1,mask_passvbfMCBlinded=0"
        maskblindedargs+=",mask_passvbf=0,mask_passvbfMCBlinded=1"
    fi
else
    ccargs="fail=${cards_dir}/fail.txt failMCBlinded=${cards_dir}/failMCBlinded.txt passbin${passbin}=${cards_dir}/passbin${passbin}.txt passbin1MCBlinded=${cards_dir}/passbin${passbin}MCBlinded.txt"
    maskunblindedargs="mask_passbin${passbin}=1,mask_fail=1,mask_passbin${passbin}MCBlinded=0,mask_failMCBlinded=0"
    maskblindedargs="mask_passbin${passbin}=0,mask_fail=0,mask_passbin${passbin}MCBlinded=1,mask_failMCBlinded=1"
fi

# freeze qcd params in blinded bins
setparamsblinded=""
freezeparamsblinded=""
for bin in {5..7}
do
    setparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin}=0,"
    freezeparamsblinded+="${CMS_PARAMS_LABEL}_tf_dataResidual_Bin${bin},"
done

# remove last comma
setparamsblinded=${setparamsblinded%,}
freezeparamsblinded="${freezeparamsblinded%,}"


# floating parameters using var{} floats a bunch of parameters which shouldn't be floated,
# so countering this inside --freezeParameters which takes priority.
# Although, practically even if those are set to "float", I didn't see them ever being fitted,
# so this is just to be extra safe.
unblindedparams="--freezeParameters var{.*_In},var{.*__norm},var{n_exp_.*} --setParameters $maskblindedargs"

# excludeimpactparams='rgx{.*tf_dataResidual_Bin.*}'

echo "cc args:"
echo "$ccargs"

echo "mask args:"
echo "$maskblindedargs"

echo "set params:"
echo "$setparamsblinded"

echo "freeze params:"
echo "$freezeparamsblinded"

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
    text2workspace.py $ws.txt --channel-masks -o $wsm.root 2>&1 | tee $outsdir/text2workspace.txt
else
    if [ ! -f "$wsm.root" ]; then
        echo "Workspace doesn't exist! Use the -w|--workspace option to make workspace first"
        exit 1
    fi
fi


if [ $bfit = 1 ]; then
    echo "Blinded background-only fit"
    combine -D $dataset -M MultiDimFit --saveWorkspace -m 125 -d ${wsm}.root -v 9 \
    --cminDefaultMinimizerStrategy 1 \
    --setParameters ${maskunblindedargs},${setparamsblinded},r=0  \
    --freezeParameters r,${freezeparamsblinded} \
    -n Snapshot 2>&1 | tee $outsdir/MultiDimFit.txt
else
    if [ ! -f "higgsCombineSnapshot.MultiDimFit.mH125.root" ]; then
        echo "Background-only fit snapshot doesn't exist! Use the -b|--bfit option to run fit first"
        exit 1
    fi
fi


if [ $limits = 1 ]; then
    echo "Expected limits"
    combine -M AsymptoticLimits -m 125 -n "" -d ${wsm_snapshot}.root --snapshotName MultiDimFit -v 9 \
    --saveWorkspace --saveToys --bypassFrequentistFit \
    ${unblindedparams},r=0 -s "$seed" \
    --floatParameters ${freezeparamsblinded},r --toysFrequentist --run blind 2>&1 | tee $outsdir/AsymptoticLimits.txt
fi


if [ $significance = 1 ]; then
    echo "Expected significance"
    combine -M Significance -d ${wsm_snapshot}.root -n "" --significance -m 125 --snapshotName MultiDimFit -v 9 \
    -t -1 --expectSignal=1 --saveWorkspace --saveToys --bypassFrequentistFit \
    ${unblindedparams},r=1 \
    --floatParameters ${freezeparamsblinded},r --toysFrequentist 2>&1 | tee $outsdir/Significance.txt
fi


if [ $dfit = 1 ]; then
    echo "Fit Diagnostics"
    combine -M FitDiagnostics -m 125 -d ${wsm}.root \
    --setParameters ${maskunblindedargs},${setparamsblinded} \
    --freezeParameters ${freezeparamsblinded} \
    --cminDefaultMinimizerStrategy 1 \
    -n Blinded --ignoreCovWarning -v 9 2>&1 | tee $outsdir/FitDiagnostics.txt
    # --saveShapes --saveNormalizations --saveWithUncertainties --saveOverallShapes \

    echo "Fit Shapes"
    PostFitShapesFromWorkspace --dataset $dataset -w ${wsm}.root --output FitShapes.root \
    -m 125 -f fitDiagnosticsBlinded.root:fit_b --postfit --print 2>&1 | tee $outsdir/FitShapes.txt
fi


if [ $gofdata = 1 ]; then
    echo "GoF on data"
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    --snapshotName MultiDimFit --bypassFrequentistFit \
    --setParameters ${maskunblindedargs},r=0 \
    --freezeParameters r \
    -n Data -v 9 2>&1 | tee $outsdir/GoF_data.txt
fi


if [ "$goftoys" = 1 ]; then
    echo "GoF on toys"
    combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 \
    --snapshotName MultiDimFit --bypassFrequentistFit \
    --setParameters ${maskunblindedargs},r=0 \
    --freezeParameters r --saveToys \
    -n Toys -v 9 -s "$seed" -t "$numtoys" --toysFrequentist 2>&1 | tee $outsdir/GoF_toys.txt
fi


if [ "$impactsi" = 1 ]; then
    echo "Initial fit for impacts"
    # from https://github.com/cms-analysis/CombineHarvester/blob/f0e0c53298521921abf59c175b5c5616026d203b/CombineTools/python/combine/Impacts.py#L113
    # combine -M MultiDimFit -m 125 -n "_initialFit_impacts" -d ${wsm_snapshot}.root --snapshotName MultiDimFit \
    #  --algo singles --redefineSignalPOIs r --floatOtherPOIs 1 --saveInactivePOI 1 -P r --setParameterRanges r=-0.5,20 \
    # --toysFrequentist --expectSignal 1 --bypassFrequentistFit -t -1 \
    # "${unblindedparams}" --floatParameters "${freezeparamsblinded}" \
    # --robustFit 1 --cminDefaultMinimizerStrategy=1 -v 9 2>&1 | tee $outsdir/Impacts_init.txt

    combineTool.py -M Impacts --snapshotName MultiDimFit -m 125 -n "impacts" \
    -t -1 --bypassFrequentistFit --toysFrequentist --expectSignal 1 \
    -d ${wsm_snapshot}.root --doInitialFit --robustFit 1 \
    ${unblindedparams} --floatParameters ${freezeparamsblinded} \
     --cminDefaultMinimizerStrategy=1 -v 1 2>&1 | tee $outsdir/Impacts_init.txt
fi


if [ "$impactsf" != 0 ]; then
    echo "Submitting jobs for impact scans"
    # Impacts module cannot access parameters which were frozen in MultiDimFit, so running impacts
    # for each parameter directly using its internal command
    # (also need to do this for submitting to condor anywhere other than lxplus)
    combine -M MultiDimFit -n _paramFit_impacts_"$impactsf" --algo impact --redefineSignalPOIs r -P "$impactsf" \
    --floatOtherPOIs 1 --saveInactivePOI 1 --snapshotName MultiDimFit -d ${wsm_snapshot}.root \
    -t -1 --bypassFrequentistFit --toysFrequentist --expectSignal 1 --robustFit 1 \
    ${unblindedparams} --floatParameters ${freezeparamsblinded} \
    --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 1 -m 125 | tee $outsdir/Impacts_"$impactsf".txt

    # Old Impacts command:
    # combineTool.py -M Impacts -t -1 --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal 1 \
    # -m 125 -n "impacts" -d ${wsm_snapshot}.root --doFits --robustFit 1 \
    # --setParameters "${maskblindedargs}" --floatParameters "${freezeparamsblinded}" \
    # --exclude ${excludeimpactparams} \
    # --job-mode condor --dry-run \
    # --setParameterRanges r=-0.5,20 --cminDefaultMinimizerStrategy=1 -v 9 2>&1 | tee $outsdir/Impacts_fits.txt
fi


if [ "$impactsc" != 0 ]; then
    echo "Collecting impacts"
    combineTool.py -M Impacts --snapshotName MultiDimFit \
    -m 125 -n "impacts" -d ${wsm_snapshot}.root \
    --setParameters ${maskblindedargs} --floatParameters ${freezeparamsblinded} \
    -t -1 --named $impactsc \
    --setParameterRanges r=-0.5,20 -v 1 -o impacts.json 2>&1 | tee $outsdir/Impacts_collect.txt

    plotImpacts.py -i impacts.json -o impacts
fi


if [ "$bias" != -1 ]; then
    echo "Bias test with bias $bias"
    # setting verbose > 0 here can lead to crazy large output files (~10-100GB!) because of getting
    # stuck in negative yield areas

    if [ $passbin = 1 ]; then
	rmin="-15"
	rmax="20"
    elif [ $passbin = 2 ]; then
	rmin="-30"
	rmax="40"
    elif [ $passbin = 3 ]; then
	rmin="-150"
	rmax="200"
    else
	rmin="-15"
	rmax="20"
    fi

    combine -M FitDiagnostics --trackParameters r --trackErrors r --justFit \
    -m 125 -n "bias${bias}_passbin${passbin}" -d ${wsm_snapshot}.root --rMin ${rmin} --rMax ${rmax} \
    --snapshotName MultiDimFit --bypassFrequentistFit --toysFrequentist --expectSignal "$bias" \
    ${unblindedparams},r=$bias --floatParameters ${freezeparamsblinded} \
    --robustFit=1 -t "$numtoys" -s "$seed" \
    --X-rtd MINIMIZER_MaxCalls=1000000 --cminDefaultMinimizerTolerance 0.1 2>&1 | tee "$outsdir/bias${bias}seed${seed}.txt"
fi
