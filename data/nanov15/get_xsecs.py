import json
import subprocess
from pathlib import Path

MC_MINIAOD_DICT = {
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200": " /DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600": "/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    # NLO samples
    "Wto2Q-2Jets_Bin-PTQQ-100": "/Wto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-200": "/Wto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-400": "/Wto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-2Jets_Bin-PTQQ-600": "/Wto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-100": "/Zto2Q-2Jets_Bin-PTQQ-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-200": "/Zto2Q-2Jets_Bin-PTQQ-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-400": "/Zto2Q-2Jets_Bin-PTQQ-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-2Jets_Bin-PTQQ-600": "/Zto2Q-2Jets_Bin-PTQQ-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    # LO samples
    "Wto2Q-3Jets_Bin-HT-100to400": "/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-400to800": "/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-800to1500": "/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-1500to2500": "/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
    "Wto2Q-3Jets_Bin-HT-2500": "/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-100to400": "/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-400to800": "/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-800to1500": "/Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-1500to2500": "/Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM",
    "Zto2Q-4Jets_Bin-HT-2500": "/Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
}

filepath_dict = {}
xsec_dict = {}

miniaod_json = Path("miniaod_filepaths.json")
if miniaod_json.exists():
    with open(miniaod_json, "r") as f:
        filepath_dict = json.load(f)
else:
    for sample_key in MC_MINIAOD_DICT:
        dataset = MC_MINIAOD_DICT[sample_key]
        command = f'dasgoclient --query="file dataset={dataset}"'
        # print(f"Running command: {command}")
        result = subprocess.run(
            command, check=False, shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Error executing command: {command}")
        # get the output
        output = result.stdout.strip()
        # split the output into lines
        samples = output.split("\n")
        example_sample = samples[0]
        print(f"Example sample for {sample_key}: {example_sample}")
        filepath_dict[sample_key] = example_sample

    # save to json
    with open("miniaod_filepaths.json", "w") as f:
        json.dump(filepath_dict, f, indent=4)
    
for sample_key in filepath_dict:
    filepath = filepath_dict[sample_key]
    command = f'cmsRun $CMSSW_BASE/src/ana.py inputFiles="{filepath}" maxEvents=-1'
    print(f"Running command to get xsec: {command}")
    result = subprocess.run(
        command, check=False, shell=True, capture_output=True, text=True
    )
    # output = result.stdout.strip()
    # output_lines = output.split("\n")
    # print(output)
    output = result.stderr.strip()
    output_lines = output.split("\n")
    # print(output)
    for line in output_lines:
        if "After filter: final cross section = " in line:
            xsec_str = line.split("After filter: final cross section = ")[1].split(" pb")[0]
            xsec_info = xsec_str.split(" +- ")
            xsec = float(xsec_info[0])
            xsec_err = float(xsec_info[1])
            xsec_dict[sample_key] = (xsec, xsec_err)
            print(f"Extracted cross-section for {sample_key}: ({xsec} +- {xsec_err}) pb")
            break
        else:
            continue
        
# save to json
with open("miniaod_xsecs.json", "w") as f:
    json.dump(xsec_dict, f, indent=4)