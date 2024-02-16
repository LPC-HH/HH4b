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
        real_files = [stack.enter_context(Path(f"data_run2/{f}")) for f in files]
        ext.add_weight_sets([f"* * {file}" for file in real_files])
        ext.finalize()

    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


# question: should we apply L3Absolute for both data and MC?
jet_factory = {
    "2018mc": jet_factory_factory(
        files=[
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_L1FastJet_AK4PFchs.txt
            "Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt.gz",
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_L2Relative_AK4PFchs.txt
            "Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt.gz",
            # https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles/Autumn18_V19_MC/RegroupedV2_Autumn18_V19_MC_UncertaintySources_AK4PFchs.txt
            "RegroupedV2_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt.gz",
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_Uncertainty_AK4PFchs.txt
            "Autumn18_V19_MC_Uncertainty_AK4PFchs.junc.txt.gz",
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Autumn18_V7b_MC/Autumn18_V7b_MC_PtResolution_AK4PFchs.txt
            "Autumn18_V7b_MC_PtResolution_AK4PFchs.jr.txt.gz",
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Autumn18_V7b_MC/Autumn18_V7b_MC_SF_AK4PFchs.txt
            "Autumn18_V7b_MC_SF_AK4PFchs.jersf.txt.gz",
        ]
    ),
    "2018mcNOJER": jet_factory_factory(
        files=[
            "Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt.gz",
            "Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt.gz",
            "Autumn18_V19_MC_Uncertainty_AK4PFchs.junc.txt.gz",
        ]
    ),
}

fatjet_factory = {
    "2018mc": jet_factory_factory(
        files=[
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_L1FastJet_AK8PFPuppi.txt
            "Autumn18_V19_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_L2Relative_AK8PFPuppi.txt
            "Autumn18_V19_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            # https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_UncertaintySources_AK8PFPuppi.txt
            "Autumn18_V19_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            # https://github.com/cms-jet/JECDatabase/raw/master/textFiles/Autumn18_V19_MC/Autumn18_V19_MC_Uncertainty_AK8PFPuppi.txt
            "Autumn18_V19_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Autumn18_V7b_MC/Autumn18_V7b_MC_PtResolution_AK8PFPuppi.txt
            "Autumn18_V7b_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Autumn18_V7b_MC/Autumn18_V7b_MC_SF_AK8PFPuppi.txt
            "Autumn18_V7b_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ]
    ),
    "2018mcNOJER": jet_factory_factory(
        files=[
            "Autumn18_V19_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Autumn18_V19_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Autumn18_V19_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ]
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
