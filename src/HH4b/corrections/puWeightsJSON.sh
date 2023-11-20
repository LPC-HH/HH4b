#!/bin/bash

scriptName="makePUReWeightJSON.py"

# produced using 69200ub
# pileupCalc.py -i Cert_Collisions_2022FULL_or_ERA_F_Golden.JSON --inputLumiJSON pileup_JSON.txt --calcMode true --minBiasXsec 69200 --maxPileupBin 100 --numPileupBins 100 MyDataPileupHistogram2022FG.root

python3 "${scriptName}" --nominal=data/MyDataPileupHistogram2022FG.root  -o 2022_puWeights.json --gzip --mcprofile=2022_LHC_Simulation_10h_2h --format=correctionlib --name=Collisions_2022_PromptReco_goldenJSON --makePlot
