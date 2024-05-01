# Run-3

- To postprocess:

```
python3 PostProcess.py --template-dir testrun3 --tag 24Mar2_v12_signal
```

- To create datacards:

```
python3 CreateDatacard.py  --templates-dir templates/testrun3  --year 2022-2023  --model-name run3-bdt
```

- To plot:

```
python3 PlotFits.py --fit-file cards/run3-bdt/FitShapes.root  --plots-dir plots/run3-bdt/ --regions passbin1
```

### ANv1:
```/uscms/home/jduarte1/nobackup/HH4b/src/HH4b/postprocessing/templates/Apr18
```
made with:
```
cd postprocessing
python3 PostProcess.py --templates-tag Apr18 --tag 24Mar31_v12_signal --mass H2Msd --no-fom-scan --templates
python3 postprocessing/CreateDatacard.py --templates-dir postprocessing/templates/Apr18 --year 2022-2023  --model-name run3-bdt-apr18
```
Fits:
```
cd cards/run3-bdt-apr18
run_blinded_hh4b.sh --workspace --bfit --limits --dfit --passbin=0
python3 postprocessing/PlotFits.py --fit-file cards/run3-bdt-apr18/FitShapes.root --plots-dir ../../plots/PostFit/run3-bdt-apr18 --signal-scale 10
```

# Run-2

- To postprocess:

```
python3 PostProcessRun2.py --template-dir 20210712_regression --tag 20210712_regression --years 2016,2017,2018
```

- To create datacads:

```
python3 CreateDatacardRun2.py --templates-dir templates/20210712_regression --year all --model-name run2-bdt-20210712 --bin-name pass_bin1
```

- To run limits:

```
./run_hh4b.sh --workspace --bfit --dfit --limits
```

- To plot:

```
python3 PlotFitsRun2.py --fit-file cards/run2-bdt-20210712/FitShapes.root --plots-dir plots/run2-bdt-20210712/ --bin-name passbin1
```
