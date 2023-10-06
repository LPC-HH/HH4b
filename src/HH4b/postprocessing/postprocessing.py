import click

import utils
import plotting

import logging
import sys


# define ShapeVar (label and bins for a given variable)
from utils import ShapeVar


var_to_shapevar = {
    # var must match key in events dictionary (i.e. as saved in parquet file)
    "DijetMass": ShapeVar(var="DijetMass", label=r"$m^{jj}$ (GeV)", bins=[30, 600, 4000]),
    "ak8FatJetPt0": ShapeVar(
        var="ak8FatJetPt0", label=r"$p_T^0$ (GeV)", bins=[30, 300, 1500], significance_dir="right"
    ),
    "ak8FatJetPt1": ShapeVar(
        var="ak8FatJetPt1", label=r"$p_T^1$ (GeV)", bins=[30, 300, 1500], significance_dir="right"
    ),
    "ak8FatJetPNetMass0": ShapeVar(
        var="ak8FatJetPNetMass0", label=r"$m_{reg}^{0}$ (GeV)", bins=[20, 50, 250]
    ),
    "ak8FatJetPNetXbb0": ShapeVar(
        var="ak8FatJetPNetXbb0",
        label=r"$TX_{bb}^{0}$",
        bins=[50, 0.0, 1],
    ),
}


@click.command()
@click.option(
    "--year",
    "years",
    required=True,
    multiple=True,
    type=click.Choice(["2022", "2022EE", "2023", "2018"], case_sensitive=False),
    help="year",
)
def postprocess(years):
    # TODO: set this as a yaml file
    dirs = {
        "/eos/uscms/store/user/cmantill/bbbb/skimmer/Oct2/": {
            "qcd": [
                "QCD_PT-120to170",
                "QCD_PT-170to300",
                "QCD_PT-470to600",
                "QCD_PT-600to800",
                "QCD_PT-800to1000",
                "QCD_PT-1000to1400",
                "QCD_PT-1400to1800",
                "QCD_PT-1800to2400",
                "QCD_PT-2400to3200",
                "QCD_PT-3200",
            ],
            "data": [
                "Run2022F",
                "Run2022G",
            ],
            "ttbar": [
                "TTtoLNu2Q",
                "TTto4Q",
                "TTto2L2Nu",
            ],
            "gghtobb": [
                "GluGluHto2B_PT-200_M-125",
            ],
            "vbfhtobb": [
                "VBFHto2B_M-125_dipoleRecoilOn",
            ],
            "vhtobb": [
                "WplusH_Hto2B_Wto2Q_M-125",
                "WplusH_Hto2B_WtoLNu_M-125",
                "WminusH_Hto2B_Wto2Q_M-125",
                "WminusH_Hto2B_WtoLNu_M-125",
                "ZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2Q_M-125",
                "ggZH_Hto2B_Zto2L_M-125",
                "ggZH_Hto2B_Zto2Nu_M-125",
            ],
            "tthtobb": [
                "ttHto2B_M-125",
            ],
            "diboson": [
                "ZZ",
                "WW",
                "WZ",
            ],
            "vjets": [
                "Wto2Q-3Jets_HT-200to400",
                "Wto2Q-3Jets_HT-400to600",
                "Wto2Q-3Jets_HT-600to800",
                "Wto2Q-3Jets_HT-800",
                "Zto2Q-4Jets_HT-200to400",
                "Zto2Q-4Jets_HT-400to600",
                "Zto2Q-4Jets_HT-600to800",
                "Zto2Q-4Jets_HT-800",
            ],
        }
    }
    samples_to_fill = [
        "data",
        "qcd",
    ]
    vars_to_plot = [
        "ak8FatJetPt0",
        "ak8FatJetPt1",
        "DijetMass",
        "ak8FatJetPNetXbb0",
    ]

    # weight to apply to histograms
    weight_key = ["finalWeight"]

    # filters are sequences of strings that can be used to place a selection
    # e.g. https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/postprocessing.py#L80
    filters = [
        [
            # [
            #    ("('HLT_AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35', '0')", "==", 1),
            #    ("('HLT_AK8PFJet425_SoftDropMass40', '0')", "==", 1),
            # ],
            # ("('HLT_AK8PFJet425_SoftDropMass40', '0')", "==", 1),
            ("('ak8FatJetPt', '0')", ">=", 300),
            ("('ak8FatJetPt', '1')", ">=", 250),
            ("('ak8FatJetMsd', '0')", ">=", 60),
            ("('ak8FatJetMsd', '1')", ">=", 60),
            # ("('ak8FatJetPNetXbb', '0')", ">=", 0.8),
            # ("('ak8FatJetPNetXbb', '1')", ">=", 0.8),
        ],
    ]

    # columns to load
    load_columns = [
        ("weight", 1),
        ("DijetMass", 1),
        ("ak8FatJetPt", 2),
        ("ak8FatJetPNetXbb", 2),
        # "single_weight_trigsf_2jet"
        # ("ak8FatJetPNetMass", 2),
    ]
    # reformat into ("column name", "idx") format for reading multiindex columns
    columns = []
    for key, num_columns in load_columns:
        for i in range(num_columns):
            columns.append(f"('{key}', '{i}')")

    for year in years:
        # load all samples, apply filters if needed
        events_dict = {}
        for input_dir, samples in dirs.items():
            events_dict = {
                **events_dict,
                **utils.load_samples(input_dir, samples, year, filters, columns),
            }

        samples_loaded = list(events_dict.keys())
        keys_loaded = list(events_dict[samples_loaded[0]].keys())
        # print(f"Keys in events_dict {keys_loaded}")

        # make a histogram
        hists = {}
        for var in vars_to_plot:
            shape_var = var_to_shapevar[var]
            if shape_var.var not in hists:
                hists[shape_var.var] = utils.singleVarHist(
                    events_dict,
                    shape_var,
                    weight_key=weight_key,
                    selection=None,
                )

        # make a stacked plot
        plotting.plot_hists(
            year,
            hists,
            vars_to_plot,
        )


if __name__ == "__main__":
    sys.exit(postprocess())
