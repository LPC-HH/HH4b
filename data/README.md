## Making filelists

Uses https://github.com/dmwm/DBSClient. Use CMSSW_11_2_0 or later, and run
`pip3 install dbs3-client --user`.

## NanoAOD versions

PDMV recommendations:
https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis

```
Campaign CMSSW
--------------
Run3Winter22 CMSSW_12_2_X POG studies
Run3Summer22 CMSSW_12_4_X 2022 data analysis
```

Some instructions on custom nano here:
https://github.com/cms-jet/PFNano/tree/13_0_7_from124MiniAOD

### Create Filelist for NanoAOD v15

Generate a filelist configuration for NanoAODv15 samples using the `make_filelists_v15.py` script:
```bash
python3 make_filelists_v15.py [options]
```

#### Options

- `-b` / `--base-config`: Path to an existing base configuration file to build upon
    - Default: `--base-config nanoindex_v14_25v2.json`
    - Set to empty string to create a new configuration from scratch
- `--years`: Space-separated list of years to process
    - Default: `--years 2016 2017 2018 2024 2025`
- `--base-url`: Base URL prefix for XRootD file access
    - Default: `--base-url root://cmsdcadisk.fnal.gov//dcache/uscmsdisk/`
- `-o` / `--output-config`: Output filepath for the generated configuration
    - Default: `--output-config nanoindex_v15.json`

#### Notes
- The script queries DAS (Data Aggregation System) to retrieve file lists for each dataset
- When using a v14 base config, the format is automatically updated to v15 (reorganizes VJets categories and splits HH4b into HH and VBFHH)
- Missing datasets generate warnings but do not stop execution
- Output files are sorted alphabetically for consistency
- `TODO`: MC samples 2025 do not exist yet and will need to be updated once available; for now, use 2024 MC samples as placeholders

### Recipe for NanoAODv12

```
cmsrel CMSSW_13_1_0
cd CMSSW_13_1_0/src
eval `scram runtime -sh`
scram b
```

#### For data:

2023-Prompt

```
# taken from: https://cmsweb.cern.ch/couchdb/reqmgr_config_cache/32c5d6d84a05232e68c9abd3937a291e/configFile
cmsDriver.py --python_filename test_nanoTuples_data2023_PromptNanoAODv12_cfg.py --eventcontent NANOAOD --customise Configuration/DataProcessing/Utils.addMonitoring,PhysicsTools/NanoAOD/nano_cff.nanoL1TrigObjCustomize --datatier NANOAOD \
--fileout file:nano_data2023_PromptNanoAODv12.root \
--conditions 130X_dataRun3_Prompt_v3 --step NANO --scenario pp \
--filein /store/data/Run2023C/JetMET0/MINIAOD/PromptReco-v2/000/367/516/00000/056efdee-d563-4fdc-9d9c-6e9bf5833df7.root \
--era Run3 --nThreads 2 --no_exec --data -n 100
```

2023-MC Run3Summer23:

```
# taken from https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/PPD-Run3Summer23NanoAODv12-00002
cmsDriver.py --python_filename test_nanoTuples_Run3Summer23_PromptNanoAODv12_cfg.py --eventcontent NANOAOD --customise Configuration/DataProcessing/Utils.addMonitoring --datatier NANOAODSIM \
--fileout file:nano_mcRun3Summer23_NanoAODv12.root \
--conditions 130X_mcRun3_2023_realistic_v8 --step NANO --scenario pp \
--filein "dbs:/MinBias_TuneCP5_13p6TeV-pythia8/Run3Summer23MiniAODv4-NoPU_Pilot_130X_mcRun3_2023_realistic_v8-v2/MINIAODSIM" \
--era Run3_2023 --no_exec --mc  -n 100
```

### Recipe for NanoAODv12 with modified PNet training for AK4 (and no ParT scores for AK4)

```
cmsrel CMSSW_13_1_0
cd CMSSW_13_1_0/src
eval `scram runtime -sh`
scram b
```

Reference: PFNano from JME also has an available recipe:
https://github.com/cms-jet/PFNano/tree/13_0_7_from124MiniAOD

Note: At some point, we will switch to ReReco (Partial, for ABCD)
https://cms-pdmv-prod.web.cern.ch/pmp/historical?r=27Jun2023&showDoneRequestsList=true
Run3_2022_rereco, 124X_dataRun3_v15

Another link tracking nano:
https://cmsweb.cern.ch/das/request?view=list&limit=150&instance=prod%2Fglobal&input=dataset+status%3D*+dataset%3D%2F*%2F*-22Sep2023-*%2FNANOAOD

2022-ABCDPrompt

```
cmsDriver.py --python_filename test_nanoTuples_data2022_ABCDPrompt_cfg.py --eventcontent NANOAOD --datatier NANOAOD --customise_commands "process.options.wantSummary = cms.untracked.bool(True)" \
--fileout file:nano_data2022_PromptNano.root \
--conditions 124X_dataRun3_PromptAnalysis_v1 --step NANO --scenario pp \
--filein /store/data/Run2022D/MuonEG/MINIAOD/PromptReco-v1/000/357/542/00000/750cf639-99bb-401a-a9b2-2b82d17a3082.root \
--era Run3,run3_nanoAOD_124 --nThreads 1 --no_exec --data -n 100
```

2022-EFGPrompt

```
# adapted from: https://cmsweb.cern.ch/couchdb/reqmgr_config_cache/10fc675e782eab01d6a5188185536e42/configFile
# adapted from Marina
cmsDriver.py --python_filename test_nanoTuples_data2022_EFGPrompt_cfg.py --eventcontent NANOAOD --datatier NANOAOD --customise Configuration/DataProcessing/Utils.addMonitoring \
--fileout file:nano_data2022_PromptNano.root \
--conditions 124X_dataRun3_Prompt_v10 --step NANO --scenario pp \
--filein /store/data/Run2022G/MuonEG/MINIAOD/PromptReco-v1/000/362/365/00000/c54ae566-34fd-4c05-87d8-b011b542ecc4.root \
--era Run3,run3_nanoAOD_124 --nThreads 1 --no_exec --data -n 100
```

2022-MC Run3Summer22:

```
cmsDriver.py --python_filename test_nanoTuples_mc2022.py --eventcontent NANOAODSIM --datatier NANOAODSIM \
--fileout file:nano_mc2022_v11.root \
--conditions 126X_mcRun3_2022_realistic_v2 --step NANO --scenario pp \
--filein \
--era Run3,run3_nanoAOD_124 --nThreads 1 --no_exec --mc -n 100
```

2022-MC Run3Summer22EE:

```
cmsDriver.py --python_filename test_nanoTuples_mc2023.py --eventcontent NANOAODSIM --datatier NANOAODSIM \
--fileout file:nano_mc2022EE_v11.root \
--conditions 126X_mcRun3_2022_realistic_postEE_v1 --step NANO --scenario pp \
--filein \
--era Run3,run3_nanoAOD_124 --nThreads 1 --no_exec --mc	-n 100
```

## Cross sections

Reference:
https://xsdb-temp.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=30&searchQuery=energy%3D13.6

https://twiki.cern.ch/twiki/bin/viewauth/CMS/MATRIXCrossSectionsat13p6TeV
