import json
from dbs.apis.dbsClient import DbsApi

dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

qcd_bins = [
    "120to170",
    "170to300",
    "300to470",
    "470to600",
    "600to800",
    "800to1000",
    "1000to1400",
    "1400to1800",
    "1800to2400",
    "2400to3200",
    "3200",
]

# which version should we take??

# nanoAODv11 does not have PNet?

# Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1
# or Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1
datasets = {
    "2022": { 
        "QCD": {
            f"QCD_PT-{qbin}": f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1/NANOAODSIM" for qbin in qcd_bins
        },
        "Hbb": {
            "GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22NanoAODv11-126X_mcRun3_2022_realistic_v2-v1/NANOAODSIM",
        },
        "JetMET": {
            "Run2022C": "/JetMET/Run2022C-ReRecoNanoAODv11-v1/NANOAOD",
            "Run2022D": "/JetMET/Run2022D-ReRecoNanoAODv11-v1/NANOAOD",
            "Run2022E": "/JetMET/Run2022E-ReRecoNanoAODv11-v1/NANOAOD",
            "Run2022F": "/JetMET/Run2022F-PromptNanoAODv11_v1-v2/NANOAOD",
            "Run2022G": "/JetMET/Run2022G-PromptNanoAODv11_v1-v2/NANOAOD",
        },
    },
    "2023": {
        "JetMET": {
            "Run2023C_0_v2": "/JetMET0/Run2023C-PromptNanoAODv12_v2-v2/NANOAOD",
            "Run2023C_0_v3": "/JetMET0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
            "Run2023C_0_v4": "/JetMET0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
            "Run2023C_1_v2": "/JetMET1/Run2023C-PromptNanoAODv12_v2-v4/NANOAOD",
            "Run2023C_1_v3": "/JetMET1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
            "Run2023C_1_v4": "/JetMET1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
        },
    }
}

index = datasets.copy()
for year,ydict in datasets.items():
    for sample,sdict in ydict.items():
        for sname, dataset in sdict.items():
            files = [ifile["logical_file_name"] for ifile in dbs.listFiles(dataset=dataset)]
            index[year][sample][sname] = files

with open("nanoindex.json", "w") as f:
    json.dump(index, f, indent=4)
