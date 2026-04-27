combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --doInitialFit --redefineSignalPOIs CMS_scale
combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --doFit --parallel 50 --redefineSignalPOIs CMS_scale
combineTool.py -M Impacts -d model_combined.root -m 125 --robustFit 1 --output impacts_scale.json --redefineSignalPOIs CMS_scale
plotImpacts.py -i impacts_scale.json -o impacts_scale