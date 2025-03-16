# Jet mass scale and resolution

## Setup
```
cd jmsr/
```

## Get Templates from Events dictionary

```bash
python jmsr_templates.py --tag TAG --year [2022All,2023All] --dir-name DIR_NAME
```

DIR_NAME Valid Versions:
```
- /eos/uscms/store/user/cmantill/bbbb/ttSkimmer/24Nov6_v12v2_private_signal
```
and:
```
python jmsr_templates.py --tag Nov7 --year-group 2022All
python jmsr_templates.py --tag Nov7 --year-group 2023All
```

# Perform combine fit

1. Setup
Initialize CMSSW setup:
```bash
cd ~/nobackup/
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmssw-el7 -p --bind `readlink $HOME` --bind `readlink -f ${HOME}/nobackup/` --bind /uscms_data --bind /cvmfs
cd CMSSW_11_3_4/src/
cmsenv
cd ../../hh/HH4b/src/HH4b/jsmr/
```

2. Generate scale/smear variations from mass template:

```bash
cd TnPSF
python3 scalesmear.py  -i run3_templates/2023All/Nov7/topCR_pt300-1000.root --plot --scale 2 --smear 0.5
```
- New files will have a name convention of `<input_name>_var.root`.
- Old files `<input_name>.root` will have the smear/scale Up/Down variations obtained from the event dictionary

3. Generate datacards
```
python3 sf.py --fit single -t   run3_templates/2023All/Nov7/topCR_pt300-1000_var.root  -o run3_templates/2023All/Nov7/topCR_pt300-1000_var_scale2smear0p5/ --scale 2 --smear 0.5
```
Then output will look like:
```
Running with options:
	Namespace(fit='single', out='run3_templates/2023All/Nov7/topCR_pt300-1000_var_scale1smear0p2/', scale=2.0, smear=0.5, template1='run3_templates/2023All/Nov7/topCR_pt300-1000_var.root', template2=None)
	effect_down (CMS_smear, wsfSinglePass_wqq) has magnitude greater than 50% (52.66%), you might be passing absolute values instead of relative
Rendering model run3_templates/2023All/Nov7/topCR_pt300-1000_var_scale2smear0p5/
```

and:
```
cd run3_templates/2023All/Nov7/topCR_pt300-1000_var_scale2smear0p5/
source build.sh
combine -M FitDiagnostics --expectSignal 1 -d model_combined.root --cminDefaultMinimizerStrategy 0 --robustFit=1 --saveShapes --saveWithUncertainties --rMin 0.5 --rMax 1.5
```

Log with Vars:
```
- scale 1 smear 0.2
 scale smear at boundary
 Best fit effSF: 0.688057  -0.0411393/+0.0434547  (68% CL)
- scale 2 smear 0.5
 Best fit effSF: 0.798491  -0.0480611/+0.0506049  (68% CL)

wsfSingleFail
scaleSF 0.971 +/- 0.005 (0.56%)
scaleSF 0.971, 0.966, 0.976
smearSF 1.089 +/- 0.063 (5.81%)
smearSF 1.089, 1.025, 1.152
wsfSinglePass
scaleSF 0.928 +/- 0.013 (1.45%)
scaleSF 0.928, 0.915, 0.942
smearSF 1.089 +/- 0.063 (5.81%)
smearSF 1.089, 1.025, 1.152
```

3. Generate scale/smear variations from mass weights
```
python3 sf.py --fit single -t run3_templates/2023All/Nov7/topCR_pt300-1000.root  -o run3_templates/2023All/Nov7/topCR_pt300-1000_scale1smear1/ --scale 1 --smear 1 --no-var
```

Log without template variations:
```
- scale 1 smear 1
  Best fit effSF: 0.78746  -0.0469556/+0.0496577  (68% CL)

wsfSingleFail
scaleSF 0.996 +/- 0.001 (0.07%)
scaleSF 0.996, 0.996, 0.997
smearSF 1.181 +/- 0.176 (14.90%)
smearSF 1.181, 1.005, 1.357
wsfSinglePass
scaleSF 0.991 +/- 0.002 (0.19%)
scaleSF 0.991, 0.989, 0.993
smearSF 1.181 +/- 0.176 (14.90%)
smearSF 1.181, 1.005, 1.357
```

A. 2022
```
python3 sf.py --fit single -t   run3_templates/2022All/Nov7/topCR_pt300-1000_var.root  -o run3_templates/2022All/Nov7/topCR_pt300-1000_var_scale2smear0p5/ --scale 2 --smear 0.5
```

Log with Vars (2022):
```
- scale 2 smear 0.5
 [WARNING] Found [effSF_un] at boundary.
 Best fit effSF: 0.865097  -0.0347699/+0.0361738  (68% CL)

wsfSingleFail
scaleSF 1.003 +/- 0.001 (0.13%)
scaleSF 1.003, 1.002, 1.005
smearSF 1.144 +/- 0.046 (4.02%)
smearSF 1.144, 1.098, 1.190
wsfSinglePass
scaleSF 1.009 +/- 0.004 (0.35%)
scaleSF 1.009, 1.005, 1.012
smearSF 1.144 +/- 0.046 (4.02%)
smearSF 1.144, 1.098, 1.190
```

and:
```
python3 sf.py --fit single -t run3_templates/2022All/Nov7/topCR_pt300-1000.root  -o run3_templates/2022All/Nov7/topCR_pt300-1000_scale1smear1/ --scale 1 --smear 1 --no-var
```

```
- scale 1 smear 1
   [WARNING] Found [CMS_smear] at boundary
  Best fit effSF: 0.88011  -0.0348187/+0.0358039  (68% CL)

wsfSingleFail
scaleSF 1.001 +/- 0.000 (0.03%)
scaleSF 1.001, 1.001, 1.001
smearSF 1.177 +/- 0.036 (3.06%)
smearSF 1.177, 1.141, 1.213
wsfSinglePass
scaleSF 1.003 +/- 0.001 (0.08%)
scaleSF 1.003, 1.002, 1.003
smearSF 1.177 +/- 0.036 (3.06%)
smearSF 1.177, 1.141, 1.213
```

## Interpretation of "weighted" Templates

https://indico.cern.ch/event/1470867/contributions/6210658/attachments/2960153/5288894/25.01.17_JMAR_Supp2.pdf

https://indico.cern.ch/event/1382617/contributions/5831259/attachments/2807626/4899444/Top_W_SFs_Run3.pdf

https://indico.cern.ch/event/1379091/contributions/5865451/attachments/2850282/4987507/DeepDive.pdf

10% jet mass smearing relative to the resolution https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution#Smearing_procedures 

- JMS apply 100 ± 10% scaling on the mSD variable

## With full dataset (Mar12)


```
cd TnPSF
python3 scalesmear.py  -i run3_templates/2022All/Mar12/topCR_pt-1000.root --plot --scale 2 --smear 0.5
python3 scalesmear.py  -i run3_templates/2023All/Mar12/topCR_pt-1000.root --plot --scale 2 --smear 0.5

python3 sf.py --fit single -t run3_templates/2022All/Mar12/topCR_pt-1000_var.root  -o run3_templates/2022All/Mar12/topCR_pt-1000_var_scale2smear0p5/ --scale 2 --smear 0.5 --year 2022
python3 sf.py --fit single -t run3_templates/2023All/Mar12/topCR_pt-1000_var.root  -o run3_templates/2023All/Mar12/topCR_pt-1000_var_scale2smear0p5/ --scale 2 --smear 0.5 --year 2023

python3 sf.py --fit single -t run3_templates/2022All/Mar12/topCR_pt300-1000.root  -o run3_templates/2022All/Mar12/topCR_pt300-1000_scale1smear1/ --scale 1 --smear 1 --no-var --year 2022
python3 sf.py --fit single -t run3_templates/2023All/Mar12/topCR_pt300-1000.root  -o run3_templates/2023All/Mar12/topCR_pt300-1000_scale1smear1/ --scale 1 --smear 1 --no-var --year 2023
```

- 2022 morphing
```
cd run3_templates/2022All/Mar12/topCR_pt-1000_var_scale2smear0p5/
source build.sh
combine -M FitDiagnostics --expectSignal 1 -d model_combined.root --cminDefaultMinimizerStrategy 0 --robustFit=1 --saveShapes --saveWithUncertainties --rMin 0.5 --rMax 1.5
  [WARNING] Found [effSF_un] at boundary. 
  [WARNING] Found [CMS_smear] at boundary. 

 --- FitDiagnostics ---
Best fit effSF: 0.843424  -0.057169/+0.0592123  (68% CL)
Done in 0.02 min (cpu), 0.02 min (real)

python3 ../../../../results.py --year 2022All  --scale 2 --smear 0.5
{'CMS_scale': {'val': 0.236, 'unc': 0.098}, 'CMS_smear': {'val': 0.831, 'unc': 0.186}, 'effSF': {'val': 0.843, 'unc': 0.058}, 'effSF_un': {'val': 0.678, 'unc': 0.485}}
{'CMS_scale': {'unc': 0.098, 'val': 0.236},
 'CMS_smear': {'unc': 0.186, 'val': 0.831},
 'V_SF': 0.843,
 'V_SF_ERR': 0.058,
 'effSF': {'unc': 0.058, 'val': 0.843},
 'effSF_un': {'unc': 0.485, 'val': 0.678},
 'shift_SF': 0.944,
 'shift_SF_ERR': 0.392,
 'smear_SF': 1.2077499999999999,
 'smear_SF_ERR': 0.0465}
Mean 84.54, Sigma 7.55
scaleSF 1.011 +/- 0.0046 (0.46%)
scaleSF 1.011, 1.0065, 1.016
smearSF 1.208 +/- 0.046 (3.85%)
smearSF 1.208, 1.161, 1.254

cp ../../../../run_impacts_scale.sh .
cp ../../../../run_impacts_smear.sh .
cp ../../../../run_impacts.sh .
```

- 2022 weighted
```
Best fit effSF: 0.891044  -0.0355897/+0.0370009  (68% CL)

python3 ../../../../results.py --year 2022All  --scale 1 --smear 1
 {'CMS_scale': {'val': 0.109, 'unc': 0.035}, 'CMS_smear': {'val': 0.35, 'unc': 0.073}, 'effSF': {'val': 0.891, 'unc': 0.037}, 'effSF_un': {'val': 1.425, 'unc': 0.169}}
{'CMS_scale': {'unc': 0.035, 'val': 0.109},
 'CMS_smear': {'unc': 0.073, 'val': 0.35},
 'V_SF': 0.891,
 'V_SF_ERR': 0.037,
 'effSF': {'unc': 0.037, 'val': 0.891},
 'effSF_un': {'unc': 0.169, 'val': 1.425},
 'shift_SF': 0.109,
 'shift_SF_ERR': 0.035,
 'smear_SF': 1.35,
 'smear_SF_ERR': 0.073}

 ```

- 2023 morphing
```
  [WARNING] Found [effSF_un] at boundary. 

 --- FitDiagnostics ---
Best fit effSF: 0.753472  -0.0521676/+0.0549683  (68% CL)
Done in 0.02 min (cpu), 0.02 min (real)

python3 ../../../../results.py --year 2023All  --scale 2 --smear 0.5
 {'CMS_scale': {'val': -0.302, 'unc': 0.093}, 'CMS_smear': {'val': 0.494, 'unc': 0.231}, 'effSF': {'val': 0.753, 'unc': 0.054}, 'effSF_un': {'val': 1.656, 'unc': 0.57}}
 {'CMS_scale': {'unc': 0.093, 'val': -0.302},
 'CMS_smear': {'unc': 0.231, 'val': 0.494},
 'V_SF': 0.753,
 'V_SF_ERR': 0.054,
 'effSF': {'unc': 0.054, 'val': 0.753},
 'effSF_un': {'unc': 0.57, 'val': 1.656},
 'shift_SF': -1.208,
 'shift_SF_ERR': 0.372,
 'smear_SF': 1.1235,
 'smear_SF_ERR': 0.05775}

 Mean 82.74, Sigma 7.40
scaleSF 0.985 +/- 0.0045 (0.46%)
scaleSF 0.985, 0.9809, 0.990
smearSF 1.123 +/- 0.058 (5.14%)
smearSF 1.123, 1.066, 1.181
```

- 2023 weighted
```
Best fit effSF: 0.854538  -0.0374485/+0.0395075  (68% CL)

{'CMS_scale': {'val': -0.133, 'unc': 0.036}, 'CMS_smear': {'val': 0.335, 'unc': 0.087}, 'effSF': {'val': 0.855, 'unc': 0.039}, 'effSF_un': {'val': 1.118, 'unc': 0.144}}
{'CMS_scale': {'unc': 0.036, 'val': -0.133},
 'CMS_smear': {'unc': 0.087, 'val': 0.335},
 'V_SF': 0.855,
 'V_SF_ERR': 0.039,
 'effSF': {'unc': 0.039, 'val': 0.855},
 'effSF_un': {'unc': 0.144, 'val': 1.118},
 'shift_SF': -0.133,
 'shift_SF_ERR': 0.036,
 'smear_SF': 1.335,
 'smear_SF_ERR': 0.087}
 ```