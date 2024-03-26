#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

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
xrdcp -f commithash.txt $eosoutgithash

pip install -e .

# run code
# pip install --user onnxruntime
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --subsamples $subsample --processor $processor --maxchunks $maxchunks --chunksize $chunksize ${save_systematics} --nano-version ${nano_version} ${region} ${apply_selection}

#move output to eos
xrdcp -f outfiles/* $eosoutpkl
xrdcp -f *.parquet $eosoutparquet
xrdcp -f *.root $eosoutroot

rm *.parquet
rm *.root
rm commithash.txt
