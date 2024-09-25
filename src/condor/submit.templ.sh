#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

for t2_prefix in ${t2_prefixes}
do
    for folder in pickles parquet root githashes
    do
        xrdfs $${t2_prefix} mkdir -p "/${outdir}/$${folder}"
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
    xrdcp -f commithash.txt $${t2_prefix}/${outdir}/githashes/commithash_${jobnum}.txt
done

pip install -e .

# run code
# pip install --user onnxruntime
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --subsamples $subsample --processor $processor --maxchunks $maxchunks --chunksize $chunksize ${save_systematics} --nano-version ${nano_version} --pnet-txbb ${pnet_txbb} ${region} ${apply_selection} ${save_root}

#move output to t2s
for t2_prefix in ${t2_prefixes}
do
    xrdcp -f outfiles/* "$${t2_prefix}/${outdir}/pickles/out_${jobnum}.pkl"
    xrdcp -f *.parquet "$${t2_prefix}/${outdir}/parquet/out_${jobnum}.parquet"
    xrdcp -f *.root "$${t2_prefix}/${outdir}/root/nano_skim_${jobnum}.root"
done

rm *.parquet
rm *.root
rm commithash.txt
