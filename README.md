# HH4b

[![Actions Status][actions-badge]][actions-link]
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/LPC-HH/HH4b/main.svg)](https://results.pre-commit.ci/latest/github/LPC-HH/HH4b/main)

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LPC-HH/HH4b/workflows/CI/badge.svg
[actions-link]:             https://github.com/LPC-HH/HH4b/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/HH4b
[conda-link]:               https://github.com/conda-forge/HH4b-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/LPC-HH/HH4b/discussions
[pypi-link]:                https://pypi.org/project/HH4b/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/HH4b
[pypi-version]:             https://img.shields.io/pypi/v/HH4b
[rtd-badge]:                https://readthedocs.org/projects/HH4b/badge/?version=latest
[rtd-link]:                 https://HH4b.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

<p align="left">
  <img width="300" src="https://raw.githubusercontent.com/LPC-HH/HH4b/main/figure.png" />
</p>

Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to
four beauty quarks (b).

<!-- The majority of the analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries. -->

- [HH4b](#hh4b)
  - [Setting up package](#setting-up-package)
    - [Creating a virtual environment](#creating-a-virtual-environment)
    - [Installing package](#installing-package)
    - [Troubleshooting](#troubleshooting)
  - [Instructions for running coffea processors](#instructions-for-running-coffea-processors)
    - [Setup](#setup)
    - [Running locally](#running-locally)
      - [Jobs](#jobs)
    - [Dask](#dask)
  - [Condor Scripts](#condor-scripts)
    - [Check jobs](#check-jobs)
    - [Combine pickles](#combine-pickles)
  - [Combine](#combine)
    - [CMSSW + Combine Quickstart](#cmssw--combine-quickstart)
    - [Create Datacards](#create-datacards)
    - [Run fits and diagnostics locally](#run-fits-and-diagnostics-locally)
  - [Moving datasets (WIP)](#moving-datasets-wip)

## Setting up package

### Creating a virtual environment

First, create a virtual environment (`micromamba` is recommended):

```bash
# Download the micromamba setup script (change if needed for your machine https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
# Install: (the micromamba directory can end up taking O(1-10GB) so make sure the directory you're using allows that quota)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# You may need to restart your shell
micromamba create -n hh4b python=3.10 -c conda-forge
micromamba activate hh4b
```

### Installing package

**Remember to install this in your mamba environment**.

```bash
# Clone the repository
git clone https://github.com/LPC-HH/HH4b.git
cd HH4b
# Perform an editable installation
pip install -e .
# for committing to the repository
pip install pre-commit
pre-commit install
```

### Troubleshooting

- If your default `python` in your environment is not Python 3, make sure to use
  `pip3` and `python3` commands instead.

- You may also need to upgrade `pip` to perform the editable installation:

```bash
python3 -m pip install -e .
```

## Instructions for running coffea processors

### Setup

For submitting to condor, all you need is python >= 3.7.

For running locally, follow the same virtual environment setup instructions
above and install `coffea`

```bash
micromamba activate hh4b
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

Parquet and pickle files will be saved. Pickles are in the format
`{'nevents': int, 'cutflow': Dict[str, int]}`.

Or on a specific file(s):

```bash
FILE=/eos/uscms/store/user/rkansal/Hbb/nano/Run3Winter23NanoAOD/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/02c29a77-3e0e-40e0-90a1-0562f54144e9.root
python -W ignore src/run.py --processor skimmer --year 2023 --files $FILE --files-name QCD
```

#### Jobs

The script `src/condor/submit.py` manually splits up the files into condor jobs:

On a full dataset: e.g. `TAG=23Jul13`

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

## Combine

### CMSSW + Combine Quickstart

```bash
cmsrel CMSSW_11_3_4
cd CMSSW_11_3_4/src
cmsenv
# float regex PR was merged so we should be able to switch to the main branch now:
git clone -b v9.2.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git clone -b v2.0.0 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
# Important: this scram has to be run from src dir
scramv1 b clean; scramv1 b
```

I also add the combine folder to my PATH in my .bashrc for convenience:

```
export PATH="$PATH:/uscms_data/d1/rkansal/hh4b/HH4b/src/HH4b/combine"
```

### Create Datacards

After activating the CMSSW environment from above, need to install rhalphalib
and this repo:

```bash
# rhalphalib
git clone https://github.com/rkansal47/rhalphalib
cd rhalphalib
pip3 install -e . --user  # editable installation not working
cd ..
# this repo
git clone https://github.com/LPC-HH/HH4b.git
cd HH4b
pip3 install . --user  # editable installation not working
```

Then, the command is:

```bash
python3 postprocessing/CreateDatacard.py --templates-dir templates/$TAG --model-name $TAG
```

e.g.

```
python3 CreateDatacard.py --templates-dir templates/23Nov13_2018/ --model-name 2018-cutbased --year 2018
```

### Run fits and diagnostics locally

All via the below script, with a bunch of options (see script):

```bash
run_blinded_hh4b.sh --workspace --bfit --limits
```

#### F-tests locally

This will take 5-10 minutes for 100 toys **will take forever for more
than >>100!**.

```bash
# automatically make workspaces and do the background-only fit for orders 0 - 3
run_ftest_hh4b.sh --cardstag run3-bdt-apr2 --templatestag Apr2 --year 2022-2023 # -dl for saving shapes and limits
# run f-test for desired order
run_ftest_hh4b.sh --cardstag run3-bdt-apr2 --goftoys --ffits --numtoys 100 --seed 444 --order 0
```

## Moving datasets (WIP)

```bash
rucio list-dataset-replicas cms:
rucio add-rule cms:/DATASET 1 T1_US_FNAL_Disk --activity "User AutoApprove" --lifetime [# of seconds] --ask-approval --comment ''
```
