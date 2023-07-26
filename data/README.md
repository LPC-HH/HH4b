## Making filelists

Uses https://github.com/dmwm/DBSClient.
Use CMSSW_11_2_0 or later, and run `pip3 install dbs3-client --user`.

## NanoAOD versions

PDMV recommendations: https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis

Campaign CMSSW 
Run3Winter22 CMSSW_12_2_X POG studies
Run3Summer22 CMSSW_12_4_X 2022 data analysis

Some instructions on custom nano here: https://github.com/cms-jet/PFNano/tree/13_0_7_from124MiniAOD

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

Note: At some point, we	will switch to ReReco (Partial,	for ABCD)
https://cms-pdmv.cern.ch/pmp/historical?r=27Jun2023
Run3_2022_rereco, 124X_dataRun3_v15

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

TO-DO
