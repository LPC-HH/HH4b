combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --doInitialFit --redefineSignalPOIs CMS_smear
combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --doFit --parallel 50 --redefineSignalPOIs CMS_smear
combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --output impacts_smear.json --redefineSignalPOIs CMS_smear
plotImpacts.py -i impacts_smear.json -o impacts_smear