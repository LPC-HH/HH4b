#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# Function to modify outdir based on t2_prefix
get_modified_outdir() {
    local t2_prefix=$$1
    local outdir=$$2

    if [[ "$$t2_prefix" == "root://cmseos.fnal.gov//" ]]; then
        # Replace zichun with zhao1
        echo "$${outdir//zichun/zhao1}"
    elif [[ "$$t2_prefix" == "root://redirector.t2.ucsd.edu:1095//" ]]; then
        # Replace zhao1 with zichun
        echo "$${outdir//zhao1/zichun}"
    else
        # Return original outdir if no match
        echo "$$outdir"
    fi
}

# make dir for output
mkdir outfiles

for t2_prefix in ${t2_prefixes}
do
    modified_outdir=$$(get_modified_outdir "$$t2_prefix" "$outdir")
    for folder in pickles parquet root githashes
    do
        xrdfs $${t2_prefix} mkdir -p "/$${modified_outdir}/$${folder}"
    done
done

# try 3 times in case of network errors
(
    r=3
    # shallow clone of single branch (keep repo size as small as possible)
    while ! git clone --single-branch --branch $branch --depth=1 https://github.com/$gituser/HH4b
    do
        ((--r)) || exit
        sleep 60
    done
)
cd HH4b || exit

commithash=$$(git rev-parse HEAD)
echo "https://github.com/$gituser/HH4b/commit/$${commithash}" > commithash.txt

#move output to t2s
for t2_prefix in ${t2_prefixes}
do
    modified_outdir=$$(get_modified_outdir "$$t2_prefix" "$outdir")
    xrdcp -f commithash.txt $${t2_prefix}/$${modified_outdir}/githashes/commithash_${jobnum}.txt
done

pip install -e .
pip install -r requirements.txt

# run code
# pip install --user onnxruntime
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --subsamples $subsample --processor $processor --maxchunks $maxchunks --chunksize $chunksize ${save_systematics} --nano-version ${nano_version} --txbb ${txbb} ${region} ${save_root}

#move output to t2s
for t2_prefix in ${t2_prefixes}
do
    modified_outdir=$$(get_modified_outdir "$$t2_prefix" "$outdir")
    xrdcp -f outfiles/* "$${t2_prefix}/$${modified_outdir}/pickles/out_${jobnum}.pkl"
    xrdcp -f *.parquet "$${t2_prefix}/$${modified_outdir}/parquet/out_${jobnum}.parquet"
    xrdcp -f *.root "$${t2_prefix}/$${modified_outdir}/root/nano_skim_${jobnum}.root"
done

rm *.parquet
rm *.root
rm commithash.txt
