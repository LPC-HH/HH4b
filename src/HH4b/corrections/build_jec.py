from __future__ import annotations

import contextlib
from pathlib import Path

from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor

jec_name_map = {
    "JetPt": "pt",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "METpt": "pt",
    "METphi": "phi",
    "JetPhi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}


def jet_factory_factory(files):
    ext = extractor()
    with contextlib.ExitStack() as stack:
        real_files = [stack.enter_context(Path(f"data/jecs/{f}")) for f in files]
        ext.add_weight_sets([f"* * {file}" for file in real_files])
        ext.finalize()

    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


jet_factory = {
    "2022mc": jet_factory_factory(
        files=[
            "Summer2222Sep2023_V2_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_V2_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_V2_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_V2_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_JRV1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer2222Sep2023_JRV1_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_V2_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_V2_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_V2_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_V2_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_JRV1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer22EE22Sep2023_JRV1_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ]
    ),
    "2023mcnoJER": jet_factory_factory(
        files=[
            "Summer23Prompt23_V1_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V1_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V1_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer23Prompt23_V1_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
        ],
    ),
    "2023BPixmcnoJER": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_V1_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V1_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V1_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23_V1_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
        ],
    ),
    # data
    # missing JER...
    "2022_runCD": jet_factory_factory(
        files=[
            "Summer2222Sep2023_RunCD_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_RunCD_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_RunCD_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
            # "Summer2222Sep2023_JRV1_DATA_PtResolution_AK4PFPuppi.jr.txt.gz",
            # "Summer2222Sep2023_JRV1_DATA_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EE_runE": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunE_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunE_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunE_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runF": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunF_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunF_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunF_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runG": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunG_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunG_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunG_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv123": jet_factory_factory(
        files=[
            "Summer23Prompt23_RunCv123_V1_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv123_V1_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv123_V1_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv4": jet_factory_factory(
        files=[
            "Summer23Prompt23_RunCv4_V1_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv4_V1_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv4_V1_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023BPix_runD": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_RunD_V1_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_RunD_V1_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_RunD_V1_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
}

fatjet_factory = {
    "2022mc": jet_factory_factory(
        files=[
            "Summer2222Sep2023_V2_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_V2_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_V2_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_V2_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_JRV1_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer2222Sep2023_JRV1_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_V2_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_V2_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_V2_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_V2_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_JRV1_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer22EE22Sep2023_JRV1_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ]
    ),
    "2023mcnoJER": jet_factory_factory(
        files=[
            "Summer23Prompt23_V1_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V1_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V1_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer23Prompt23_V1_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ],
    ),
    "2023BPixmcnoJER": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_V1_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V1_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V1_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23_V1_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ],
    ),
    "2022_runCD": jet_factory_factory(
        files=[
            "Summer2222Sep2023_RunCD_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_RunCD_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer2222Sep2023_RunCD_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runE": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunE_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunE_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunE_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runF": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunF_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunF_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunF_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runG": jet_factory_factory(
        files=[
            "Summer22EE22Sep2023_RunG_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunG_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE22Sep2023_RunG_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv123": jet_factory_factory(
        files=[
            "Summer23Prompt23_RunCv123_V1_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv123_V1_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv123_V1_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv4": jet_factory_factory(
        files=[
            "Summer23Prompt23_RunCv4_V1_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv4_V1_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_RunCv4_V1_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023BPix_runD": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_RunD_V1_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_RunD_V1_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_RunD_V1_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
}

met_factory = CorrectedMETFactory(jec_name_map)


if __name__ == "__main__":
    import argparse
    import gzip

    import cloudpickle

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", default="jec_compiled.pkl.gz", type=str)
    args = parser.parse_args()

    with gzip.open(args.output, "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "fatjet_factory": fatjet_factory,
                "met_factory": met_factory,
            },
            fout,
        )
