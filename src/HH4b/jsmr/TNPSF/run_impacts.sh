combineTool.py -M Impacts -d model_combined.root --rMin -10 --rMax 10 -m 125 --robustFit 1 --doInitialFit --expectSignal 1
combineTool.py -M Impacts -d model_combined.root --rMin -10 --rMax 10 -m 125 --robustFit 1 --doFits --expectSignal 1 --parallel 50
combineTool.py -M Impacts -d model_combined.root --rMin -10 --rMax 10 -m 125 --robustFit 1 --output impacts.json --expectSignal 1
plotImpacts.py -i impacts.json -o impacts
