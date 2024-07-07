#!/bin/bash
# shellcheck disable=SC2086

####################################################################################################
# 1) Makes datacards and workspaces for different orders of polynomials
# 2) Runs background-only fit (Higgs mass window blinded) for lowest order polynomial and GoF test (saturated model) on data
# 3) Runs fit diagnostics and saves shapes (-d|--dfit)
# 4) Generates toys and gets test statistics for each (-t|--goftoys)
# 5) Fits +1 order models to all 100 toys and gets test statistics (-f|--ffits)
#
# Author: Raghav Kansal
####################################################################################################


goftoys=0
ffits=0
dfit=0
limits=0
seed=42
numtoys=100
order=0
year="2022EE"
passbin=1

options=$(getopt -o "tfdlo:s:y" --long "cardstag:,templatestag:,goftoys,ffits,dfit,limits,order:,numtoys:,seed:,year:,passbin:" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -t|--goftoys)
            goftoys=1
            ;;
        -f|--ffits)
            ffits=1
            ;;
        -d|--dfit)
            dfit=1
            ;;
        -l|--limits)
            limits=1
            ;;
        --cardstag)
            shift
            cards_tag=$1
            ;;
        --templatestag)
            shift
            templates_tag=$1
            ;;
        -o|--order)
            shift
            order=$1
            ;;
        --seed)
            shift
            seed=$1
            ;;
        --numtoys)
            shift
            numtoys=$1
            ;;
        --year)
            shift
            year=$1
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

echo "Arguments: cardstag=$cards_tag templatestag=$templates_tag dfit=$dfit \
goftoys=$goftoys ffits=$ffits order=$order seed=$seed numtoys=$numtoys year=$year passbin=$passbin"


####################################################################################################
# Set up fit args
####################################################################################################

templates_dir="postprocessing/templates/${templates_tag}"
cards_dir="cards/f_tests/${cards_tag}/"
mkdir -p "${cards_dir}"
echo "Saving datacards to ${cards_dir}"

# these are for inside the different cards directories
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

outsdir="./outs"

# nonresoant args

if [ "$passbin" == "vbf" ]; then
    region="passvbf"
    region_="pass_vbf"
else
    region="passbin${passbin}"
    region_="pass_bin${passbin}"
fi

if [ $passbin = 1 ]; then
rmin="-15"
rmax="20"
elif [ $passbin = 2 ]; then
rmin="-30"
rmax="40"
elif [ $passbin = 3 ]; then
rmin="-150"
rmax="200"
elif [ "$passbin" == "vbf" ]; then
rmin="-60"
rmax="80"
else
rmin="-15"
rmax="20"
fi
rexp="1"

####################################################################################################
# Making cards and workspaces for each order polynomial
####################################################################################################

for ord in {0..3}
do
    model_name="${region}_nTF_${ord}"

    # create datacards if they don't already exist
    if [ ! -f "${cards_dir}/${model_name}/fail.txt" ]; then
        echo "Making Datacard for $model_name"
        python3 -u postprocessing/CreateDatacard.py --templates-dir "${templates_dir}" --unblinded \
        --model-name "${model_name}" --nTF "${ord}" --cards-dir "${cards_dir}" --year "${year}" --regions ${region_}
    fi

    cd "${cards_dir}/${model_name}/" || exit
    echo "${cards_dir}/${model_name}/"

    # make workspace, background-only fit, GoF on data if they don't already exist
    if [ ! -f "./higgsCombineData.GoodnessOfFit.mH125.root" ]; then
        echo "Making workspace, doing s+b fit and gof on data"
	run_unblinded_hh4b.sh -wbg --passbin=${passbin}
    fi

    if [ $dfit = 1 ]; then
	run_unblinded_hh4b.sh -d --passbin=${passbin}
    fi

    if [ $limits = 1 ]; then
	run_unblinded_hh4b.sh -l --passbin=${passbin}
    fi

    cd - || exit
done


####################################################################################################
# Generate toys for lower order
####################################################################################################

model_name="${region}_nTF_${order}"
toys_name=$order
cd "${cards_dir}/${model_name}/" || exit
toys_file="$(pwd)/higgsCombineToys${toys_name}.GenerateOnly.mH125.$seed.root"
cd - || exit

if [ $goftoys = 1 ]; then
    cd "${cards_dir}/${model_name}/" || exit

    ulimit -s unlimited

    echo "Toys for $order order fit"

    echo "Get expected r value"
    rexp=`python3 -c 'import uproot; print(uproot.open("'${wsm_snapshot}'.root")["limit"].arrays("r")[b"r"][0])'`

    combine -M GenerateOnly -m 125 -d ${wsm_snapshot}.root --rMin $rmin --rMax $rmax \
    --snapshotName MultiDimFit --bypassFrequentistFit --trackParameters r --expectSignal $rexp \
    -n "Toys${toys_name}" -t "$numtoys" --saveToys -s "$seed" -v 9 2>&1 | tee "$outsdir/gentoys.txt"

    cd - || exit
fi


####################################################################################################
# GoFs on generated toys for low and next high order polynomials
####################################################################################################

if [ $ffits = 1 ]; then
    for ord in $order $((order+1))
    do
	model_name="${region}_nTF_${ord}"
        echo "Fits for $model_name"

        cd "${cards_dir}/${model_name}/" || exit

        ulimit -s unlimited

        combine -M GoodnessOfFit -d ${wsm_snapshot}.root --algo saturated -m 125 --rMin $rmin --rMax $rmax \
        -n "Toys${toys_name}" -v 9 -s "$seed" -t "$numtoys" --toysFile "${toys_file}" 2>&1 | tee "$outsdir/GoF_toys${toys_name}.txt"

        cd - || exit
    done
fi
