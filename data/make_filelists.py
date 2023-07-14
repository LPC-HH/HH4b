import json
from dbs.apis.dbsClient import DbsApi

dbs = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")

qcd_index = {}

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

for qbin in qcd_bins:
    dataset = f"/QCD_PT-{qbin}_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv10-124X_mcRun3_2022_realistic_v12-v1/NANOAODSIM"
    files = [file["logical_file_name"] for file in dbs.listFiles(dataset=dataset)]
    qcd_index[f"QCD_PT-{qbin}"] = files

index = {"2022": {"QCD": qcd_index}}

with open("nanoindex_2022.json", "w") as f:
    json.dump(index, f, indent=4)
