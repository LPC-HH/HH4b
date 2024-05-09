# Run-3

- To postprocess: `PostProcess.py`
- To create datacards: `CreateDatacard.py`
- To plot: `PlotFits.py`

### ANv1:
```/uscms/home/jduarte1/nobackup/HH4b/src/HH4b/postprocessing/templates/Apr18
```
made with:
```
cd postprocessing
python3 PostProcess.py --templates-tag Apr18 --tag 24Mar31_v12_signal --mass H2Msd --no-fom-scan --templates --bdt-model v1_msd30_nomulticlass --bdt-config v1_msd30
python3 postprocessing/CreateDatacard.py --templates-dir postprocessing/templates/Apr18 --year 2022-2023  --model-name run3-bdt-apr18
```
Fits:
```
cd cards/run3-bdt-apr18
run_blinded_hh4b.sh --workspace --bfit --limits --dfit --passbin=0
python3 postprocessing/PlotFits.py --fit-file cards/run3-bdt-apr18/FitShapes.root --plots-dir ../../plots/PostFit/run3-bdt-apr18 --signal-scale 10
```

### Reproducing ANv1
```
python3 PostProcess.py --templates-tag May2 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2Msd --no-legacy --bdt-config v1_msd30_txbb  --bdt-model v1_msd30_nomulticlass  --no-fom-scan --templates --txbb-wps 0.92 0.8 --bdt-wps 0.94 0.68 0.03 --years 2022 2022EE 2023 2023BPix
```

### Apr22
To scan:
```
python3 PostProcess.py --templates-tag Apr22 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr21_legacy_vbf_vars --bdt-model 24Apr21_legacy_vbf_vars  --fom-scan --txbb-wps 0.99 0.94 --bdt-wps 0.94 0.68 0.03 --no-control-plots --no-bdt-roc --no-templates --no-fom-scan-vbf --years 2022EE --method sideband
```
For templates:
```
# x-check in branch
python3 PostProcess.py --templates-tag 24May7xcheckApr22 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr20_legacy_fix --bdt-model 24Apr20_legacy_fix --txbb-wps 0.985 0.94 --bdt-wps 0.98 0.9 0.03 --no-bdt-roc  --templates --no-fom-scan --no-fom-scan-vbf --years 2022 2022EE 2023 2023BPix --data-dir /ceph/cms/store/user/rkansal/bbbb/skimmer/

python3 PostProcess.py --templates-tag 24May7xcheckvjetsApr22 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr20_legacy_fix --bdt-model 24Apr20_legacy_fix --txbb-wps 0.985 0.94 --bdt-wps 0.98 0.9 0.03 --no-bdt-roc --templates --no-fom-scan --no-fom-scan-vbf --years 2022 2022EE 2023 2023BPix --training-years 2022EE
python3 postprocessing/CreateDatacard.py --templates-dir postprocessing/templates/24May7xcheckvjetsApr22 --year 2022-2023  --model-name run3-bdt-may7-checkvjets
cd cards/run3-bdt-may7-checkvjets/
run_blinded_hh4b.sh --workspace --bfit --limits --dfit --passbin=0
python3 postprocessing/PlotFits.py --fit-file cards/run3-bdt-may7-checkvjets/FitShapes.root --plots-dir ../../plots/PostFit/run3-bdt-may7-checkvjets --signal-scale 10
```

### May2
Scan:
```
python3 PostProcess.py --templates-tag 24May7 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr21_legacy_vbf_vars --bdt-model 24May1_legacy_vbf_vars --fom-scan --txbb-wps 0.985 0.94 --bdt-wps 0.98 0.9 0.03 --no-control-plots --no-bdt-roc --no-templates --no-fom-scan-vbf --years 2022 2022EE 2023 2023BPix --method abcd
```

Templates:
```
python3 PostProcess.py --templates-tag 24May7 --tag 24Apr23LegacyLowerThresholds_v12_private_signal --mass H2PNetMass --legacy --bdt-config 24Apr21_legacy_vbf_vars --bdt-model 24May1_legacy_vbf_vars --txbb-wps 0.985 0.94 --bdt-wps 0.98 0.9 0.03 --bdt-roc --templates --no-fom-scan --no-fom-scan-vbf --years 2022 2022EE 2023 2023BPix --training-years 2022EE 2023

```

# Run-2
```
python3 PostProcessRun2.py --template-dir 20210712_regression --tag 20210712_regression --years 2016,2017,2018
python3 CreateDatacardRun2.py --templates-dir templates/20210712_regression --year all --model-name run2-bdt-20210712 --bin-name pass_bin1
./run_hh4b.sh --workspace --bfit --dfit --limits
python3 PlotFitsRun2.py --fit-file cards/run2-bdt-20210712/FitShapes.root --plots-dir plots/run2-bdt-20210712/ --bin-name passbin1
```