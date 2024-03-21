# Run-3

- To postprocess:
```
python3 PostProcess.py --template-dir testrun3 --tag 24Mar2_v12_signal
```

# Run-2

- To postprocess:

```
python3 PostProcessRun2.py --template-dir 20210712_regression --tag 20210712_regression --years 2016,2017,2018
```

- To create datacads:

```
python3 CreateDatacardRun2.py --templates-dir test/ --year all --model-name run2-bdt
```

- To run limits:

```
./run_hh4b.sh --workspace --bfit --dfit --limits
```

- To plot:

```
python3 PlotFitsRun2.py --fit-file cards/run2-bdt/FitShapes.root --plots-dir test/
```
