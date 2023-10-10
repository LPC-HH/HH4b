# HH4b

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/LPC-HH/HH4b/main.svg)](https://results.pre-commit.ci/latest/github/LPC-HH/HH4b/main)

<!-- <p align="left">
  <img width="300" src="https://raw.githubusercontent.com/rkansal47/HH4b/main/figure.png" />
</p> -->

<!-- Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to four beauty quarks (b). The majority of the analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries. -->


- [HH4b](#hh4b)
  - [Instructions for running coffea processors](#instructions-for-running-coffea-processors)
    - [Setup](#setup)
    - [Condor](#condor)
    - [Running locally](#running-locally)
  - [Processors](#processors)
    - [triggerSkmimer](#triggerskmimer)
    - [bbbbSkimmer](#bbbbskimmer)
      - [Jobs](#jobs)
    - [matchingSkimmer](#matchingskimmer)
  - [Condor Scripts](#condor-scripts)
    - [Check jobs](#check-jobs)
    - [Combine pickles](#combine-pickles)


## Instructions for running coffea processors

### Setup

For submitting to condor, all you need is python >= 3.7.

For running locally:

```bash
# Download the mamba setup script (change if needed for your machine https://github.com/conda-forge/miniforge#mambaforge)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
# Install: (the mamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
./Mambaforge-Linux-x86_64.sh  # follow instructions in the installation
mamba create -n hh4b python=3.9
mamba activate hh4b
pip install coffea
```

Clone the repository:
```
git clone https://github.com/LPC-HH/HH4b/
pip install -e .
```

### Running locally

To test locally first (recommended), can do e.g.:

```bash
mkdir outfiles
python -W ignore src/run.py --starti 0 --endi 1 --year 2022 --processor skimmer --samples QCD --subsamples "QCD_PT-470to600"
python -W ignore src/run.py --processor skimmer --year 2022 --nano-version v11_private --samples HH --subsamples GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG --starti 0 --endi 1
python -W ignore src/run.py  --year 2022 --processor trigger_boosted --samples Muon --subsamples Run2022C --nano_version v11_private --starti 0 --endi 1
```

Parquet and pickle files will be saved.
Pickles are in the format `{'nevents': int, 'cutflow': Dict[str, int]}`.

Or on a specific file(s):

```bash
FILE=/eos/uscms/store/user/rkansal/Hbb/nano/Run3Winter23NanoAOD/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/02c29a77-3e0e-40e0-90a1-0562f54144e9.root
python -W ignore src/run.py --processor skimmer --year 2023 --files $FILE --files-name QCD
```

#### Jobs

The script `src/condor/submit.py` manually splits up the files into condor jobs:

On a full dataset:
e.g. `TAG=23Jul13`
```
python src/condor/submit.py --processor skimmer --tag $TAG --files-per-job 20 --submit
```

On a specific sample:

```bash
python src/condor/submit.py --processor skimmer --tag $TAG --nano-version v11_private --samples HH --subsamples GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_TSG
```

Over many samples, using a yaml file:

```bash
nohup python src/condor/submit_from_yaml.py --tag $TAG --processor skimmer --save-systematics --submit --yaml src/condor/submit_configs/${YAML}.yaml &> tmp/submitout.txt &
```

To Submit (if not using the --submit flag):

```bash
nohup bash -c 'for i in condor/'"${TAG}"'/*.jdl; do condor_submit $i; done' &> tmp/submitout.txt &
```

### Dask

Log in with ssh tunneling:
```
ssh -L 8787:localhost:8787 cmslpc-sl7.fnal.gov
```

Run the `./shell` script as setup above via lpcjobqueue:
```
./shell coffeateam/coffea-dask:0.7.21-fastjet-3.4.0.1-g6238ea8
```

Renew your grid certificate:
```
voms-proxy-init --rfc --voms cms -valid 192:00
```

Run the job submssion script:
```
python -u -W ignore src/run.py --year 2022EE --yaml src/condor/submit_configs/skimmer_23_10_02.yaml --processor skimmer --nano-version v11 --region signal --save-array --executor dask > dask.out 2>&1
```

## Condor Scripts

### Check jobs

Check that all jobs completed by going through output files:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --tag $TAG --processor trigger (--submit) --year $year; done
```

e.g.
```
python src/condor/check_jobs.py --year 2018 --tag Oct9 --processor matching --check-running --user cmantill --submit-missing
```

### Combine pickles

Combine all output pickles into one:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/combine_pickles.py --tag $TAG --processor trigger --r --year $year; done
```

 python -u -W ignore src/run.py --year 2022EE --yaml src/condor/submit_configs/skimmer_23_10_02.yaml --processor skimmer --nano-version v11 --region signal --save-array --executor dask > dask.out 2>&1