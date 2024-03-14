# Run-2

```
python3 PostProcessRun2.py
python3 CreateDatacardRun2.py --templates-dir test/ --year all --model-name run2-bdt
./run_hh4b.sh --workspace --bfit --dfit --limits
python3 PlotFitsRun2.py --fit-file cards/run2-bdt/FitShapes.root --plots-dir test/
```
