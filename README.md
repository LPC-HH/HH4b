# HH4b

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/rkansal47/HH4b/main.svg)](https://results.pre-commit.ci/latest/github/rkansal47/HH4b/main)

<!-- <p align="left">
  <img width="300" src="https://raw.githubusercontent.com/rkansal47/HH4b/main/figure.png" />
</p> -->

<!-- Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to two beauty quarks (b) and two vector bosons (V). The majority of the analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries. -->


- [HH4b](#hh4b)
  - [Instructions for running coffea processors](#instructions-for-running-coffea-processors)
    - [Setup](#setup)
    - [Condor](#condor)
  - [Processors](#processors)
    - [JetHTTriggerEfficiencies](#jethttriggerefficiencies)
    - [bbbbSkimmer](#bbbbskimmer)
  - [Condor Scripts](#condor-scripts)
    - [Check jobs](#check-jobs)
    - [Combine pickles](#combine-pickles)


## Instructions for running coffea processors

### Setup

TODO: test this from scratch

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

### Condor

Manually splits up the files into condor jobs.

```bash
git clone https://github.com/rkansal47/HH4b/
cd HH4b
TAG=23Jul13
# will need python3 (can use either CMSSW >= 11_2_0 or via miniconda/mamba)
python src/condor/submit.py --processor skimmer --tag $TAG --files-per-job 20 --submit
```

Alternatively, can be submitted from a yaml file:

```bash
python src/condor/submit_from_yaml.py --year 2022 --processor skimmer --tag $TAG --yaml src/condor/submit_configs/skimmer_inputs_07_24.yaml 
```

To test locally first (recommended), can do e.g.:

```bash
mkdir outfiles
python -W ignore src/run.py --starti 0 --endi 1 --year 2022 --processor skimmer --samples QCD --subsamples "QCD_PT-470to600"
```

## Processors

### JetHTTriggerEfficiencies

Applies a muon pre-selection and accumulates 3D ([Txbb, pT, mSD]) yields before and after our triggers.

To test locally:

```bash
python -W ignore src/run.py --year 2018 --processor trigger --sample SingleMu2017 --subsamples SingleMuon_Run2018B --starti 0 --endi 1
```

And to submit all:

```bash
nohup bash -c 'for i in 2016 2016APV 2017 2018; do python src/condor/submit.py --year $i --tag '"${TAG}"' --processor trigger --submit; done' &> tmp/submitout.txt &
```

### bbbbSkimmer

Applies pre-selection cuts and saves unbinned branches as parquet and root files.

Parquet and pickle files will be saved in the eos directory of specified user at path `~/eos/HH4b/skimmer/<tag>/<sample_name>/<parquet or root or pickles>`. 
Pickles are in the format `{'nevents': int, 'cutflow': Dict[str, int]}`.

To test locally:

```bash
python -W ignore src/run.py --starti 0 --endi 1 --year 2022 --processor skimmer --samples QCD --subsamples "QCD_PT-470to600"
```

Or on a specific file(s):

```bash
FILE=/eos/uscms/store/user/rkansal/Hbb/nano/Run3Winter23NanoAOD/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/02c29a77-3e0e-40e0-90a1-0562f54144e9.root
python -W ignore src/run.py --processor skimmer --year 2023 --files $FILE --files-name QCD
```

Jobs

```bash
nohup python src/condor/submit_from_yaml.py --year 2022 --tag $TAG --processor skimmer --save-systematics --submit --yaml src/condor/submit_configs/skimmer_inputs_23_02_17.yaml &> tmp/submitout.txt &
```

All years:

```bash
nohup bash -c 'for year in 2016APV 2016 2017 2018; do python src/condor/submit_from_yaml.py --year $year --tag '"${TAG}"' --processor skimmer --save-systematics --submit --yaml src/condor/submit_configs/skimmer_inputs_23_02_17.yaml; done' &> tmp/submitout.txt &
```


To Submit (if not using the --submit flag)
```bash
nohup bash -c 'for i in condor/'"${TAG}"'/*.jdl; do condor_submit $i; done' &> tmp/submitout.txt &
```


## Condor Scripts

### Check jobs

Check that all jobs completed by going through output files:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --tag $TAG --processor trigger (--submit) --year $year; done
```

nohup version:

(Do `condor_q | awk '{ print $9}' | grep -o '[^ ]*\.sh' > running_jobs.txt` first to get a list of jobs which are running.)

```bash
nohup bash -c 'for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --year $year --tag '"${TAG}"' --processor skimmer --submit --check-running; done' &> tmp/submitout.txt &
```

### Combine pickles

Combine all output pickles into one:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/combine_pickles.py --tag $TAG --processor trigger --r --year $year; done
```
