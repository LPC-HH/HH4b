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
````
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