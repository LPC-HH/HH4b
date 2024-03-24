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
python3 PlotFits.py --fit-file cards/run3-bdt/FitShapes.root  --plots-dir plots/run3-bdt/ --bin-name passbin1
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
