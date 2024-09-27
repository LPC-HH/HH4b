# Boosted studies

## BDT 

- v0_msd30:
```
python TrainBDT.py --data ../../../../data/skimmer/24Mar2_v12_signal --year 2022EE --model-name v0_msd30
```

- 24Apr21_legacy_vbf_vars
```
python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/rkansal/bbbb/skimmer/24Apr19LegacyFixes_v12_private_signal/ --model-name 24Apr21_legacy_vbf_vars --legacy --sig-keys hh4b vbfhh4b-k2v0 --no-pnet-plots
```

- 24May1_legacy_pt300xbb08_2022EE2023
```
python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/rkansal/bbbb/skimmer/24Apr23LegacyLowerThresholds_v12_private_signal/ --model-name  24May1_legacy_pt300xbb08_2022EE2023 --xbb bbFatJetPNetTXbbLegacy --mass bbFatJetPNetMassLegacy --legacy --pnet-plots --apply-cuts --config-name 24Apr20_legacy_fix --year 2022-2023
```

- 24May1_legacy_pt300xbb08msd140_2022EE2023
```
python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/rkansal/bbbb/skimmer/24Apr23LegacyLowerThresholds_v12_private_signal/ --model-name  24May1_legacy_pt300xbb08msd140_2022EE2023 --xbb bbFatJetPNetTXbbLegacy --mass bbFatJetPNetMassLegacy --legacy --pnet-plots --apply-cuts --config-name 24Apr20_legacy_fix --year 2022-2023
```

- 24May1_legacy_pt300xbb08msd140_2022EE2023_all
```
python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/rkansal/bbbb/skimmer/24Apr23LegacyLowerThresholds_v12_private_signal/ --model-name  24May1_legacy_pt300xbb08msd140_2022EE2023_all --xbb bbFatJetPNetTXbbLegacy --mass bbFatJetPNetMassLegacy --legacy --pnet-plots --apply-cuts --config-name 24Apr20_legacy_fix --year 2022 2022EE 2023 2023BPix
```

- 24May1_legacy_vbf_vars
```
python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/rkansal/bbbb/skimmer/24Apr23LegacyLowerThresholds_v12_private_signal/ --model-name 24May1_legacy_vbf_vars --xbb bbFatJetPNetTXbbLegacy --mass bbFatJetPNetMassLegacy --legacy --pnet-plots --apply-cuts --config 24Apr21_legacy_vbf_vars --year 2022 2022EE 2023 2023BPix --sig-keys hh4b vbfhh4b-k2v0
```

- v5_glopartv2
```
python -W ignore TrainBDT.py --data-path /eos/uscms/store/user/cmantill/bbbb/skimmer/24Sep25_v12v2_private_signal/  --model-name 24Sep27_v5_glopartv2 --txbb  glopart-v2 --mass bbFatJetParTmassVis --txbb-plots --apply-cuts --config v5_glopartv2 --year 2022 2022EE 2023 2023BPix --sig-keys hh4b vbfhh4b-k2v0
```