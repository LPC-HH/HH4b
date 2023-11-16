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
        real_files = [stack.enter_context(Path(f"data/{f}")) for f in files]
        ext.add_weight_sets([f"* * {file}" for file in real_files])
        ext.finalize()

    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


# question: should we apply L3Absolute for both data and MC?
jet_factory = {
    # MC: L1 + MC-truth
    "2022mc": jet_factory_factory(
        files=[
            "Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Winter22Run3_V1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Winter22Run3_V1_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ]
    ),
    "2022mcNOJER": jet_factory_factory(
        files=[
            "Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
        ]
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_V1_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer22EEPrompt22_V1_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Winter22Run3_V1_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Winter22Run3_V1_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmcNOJER": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_V1_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
        ]
    ),
    # data: L1 + MC-truth + L2L3Residuals
    "2022NOJER_runC": jet_factory_factory(
        files=[
            "Winter22Run3_RunC_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_RunC_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_RunC_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
            # having some trouble with Winter22Run3_V1_DATA_PtResolution_AK4PFPuppi.jr.txt.gz
            # File "/uscms/home/cmantill/nobackup/miniconda3/envs/coffea-env/lib/python3.8/site-packages/coffea/lookup_tools/txt_converters.py", line 220, in _build_standard_jme_lookup
            # ValueError: structure imposed by 'counts' does not fit in the array or partition at axis=0
            # "Winter22Run3_V1_DATA_PtResolution_AK4PFPuppi.jr.txt.gz",
            # "Winter22Run3_V1_DATA_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022NOJER_runD": jet_factory_factory(
        files=[
            "Winter22Run3_RunD_V2_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_RunD_V2_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            # "Winter22Run3_RunD_V2_DATA_L3Absolute_AK4PFPuppi.jec.txt.gz",
            "Winter22Run3_RunD_V2_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EENOJER_runF": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_RunF_V1_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunF_V1_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            # "Summer22EEPrompt22_RunF_V1_DATA_L3Absolute_AK4PFPuppi.jec.txt.gz"
            "Summer22EEPrompt22_RunF_V1_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EENOJER_runG": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_RunG_V1_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunG_V1_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            # "Summer22EEPrompt22_RunG_V1_DATA_L3Absolute_AK4PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunG_V1_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
}

fatjet_factory = {
    "2022mc": jet_factory_factory(
        files=[
            "Winter22Run3_V2_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Winter22Run3_V2_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Winter22Run3_V1_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Winter22Run3_V1_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ]
    ),
    "2022mcNOJER": jet_factory_factory(
        files=[
            "Winter22Run3_V2_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_V2_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ]
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_V1_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer22EEPrompt22_V1_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Winter22Run3_V1_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Winter22Run3_V1_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmcNOJER": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_V1_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_V1_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ]
    ),
    "2022NOJER_runC": jet_factory_factory(
        files=[
            "Winter22Run3_RunC_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_RunC_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_RunC_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022NOJER_runD": jet_factory_factory(
        files=[
            "Winter22Run3_RunD_V2_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_RunD_V2_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Winter22Run3_RunD_V2_DATA_L3Absolute_AK8PFPuppi.jec.txt.gz",
            "Winter22Run3_RunD_V2_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EENOJER_runF": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_RunF_V1_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunF_V1_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Summer22EEPrompt22_RunF_V1_DATA_L3Absolute_AK8PFPuppi.jec.txt.gz"
            "Summer22EEPrompt22_RunF_V1_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EENOJER_runG": jet_factory_factory(
        files=[
            "Summer22EEPrompt22_RunG_V1_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunG_V1_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            # "Summer22EEPrompt22_RunG_V1_DATA_L3Absolute_AK8PFPuppi.jec.txt.gz",
            "Summer22EEPrompt22_RunG_V1_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
}

met_factory = CorrectedMETFactory(jec_name_map)


if __name__ == "__main__":
    import gzip
    import sys

    # jme stuff not pickleable in coffea
    import cloudpickle

    with gzip.open(sys.argv[-1], "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "fatjet_factory": fatjet_factory,
                "met_factory": met_factory,
            },
            fout,
        )
