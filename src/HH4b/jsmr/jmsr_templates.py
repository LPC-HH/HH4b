from __future__ import annotations

import logging
import os

import click
import numpy as np
import pandas as pd
import uproot

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

import HH4b.plotting as plotting
import HH4b.utils as utils
from HH4b.utils import ShapeVar

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger("JMSR_TEMPLATES")

# for plotting
bkgs = ["top_matched", "W_matched", "unmatched", "diboson", "qcd", "vjetslnu", "singletop"]
bkg_order = [
    "qcd",
    "diboson",
    "vjetslnu",
    "unmatched",
    "top_matched",
    "singletop",
    "W_matched",
]

# TODO: ADD TTV
samples_mc = {
    "2022": {
        "diboson": [
            "WW",
            "WZ",
            "ZZ",
        ],
        "vjetslnu": [
            "WtoLNu-2Jets_0J",
            "WtoLNu-2Jets_2J",
            "DYto2L-2Jets_MLL-50_2J",
        ],
        "qcd": [
            "QCD_HT-400to600",
            "QCD_HT-600to800",
            "QCD_HT-800to1000",
            "QCD_HT-1000to1200",
            "QCD_HT-1200to1500",
            "QCD_HT-1500to2000",
            "QCD_HT-2000",
        ],
        "singletop": [
            "TbarBQ_t-channel_4FS",
            "TBbarQ_t-channel_4FS",
            "TbarWplusto4Q",
            "TWminusto4Q",
            "TbarWplustoLNu2Q",
            "TWminustoLNu2Q",
        ],
    },
    "2022EE": {
        "diboson": [
            "WW",
            "ZZ",
            "WZ",
        ],
        "vjetslnu": [
            "WtoLNu-2Jets_2J",
            "DYto2L-2Jets_MLL-50_2J",
        ],
        "qcd": [
            "QCD_HT-400to600",
            "QCD_HT-600to800",
            "QCD_HT-800to1000",
            "QCD_HT-1000to1200",
            "QCD_HT-1200to1500",
            "QCD_HT-1500to2000",
            "QCD_HT-2000",
        ],
        "singletop": [
            "TbarBQ_t-channel_4FS",
            "TBbarQ_t-channel_4FS",
            "TbarWplusto4Q",
            "TWminusto4Q",
            "TbarWplustoLNu2Q",
            "TWminustoLNu2Q",
        ],
    },
    "2023": {
        "diboson": [
            "WW",
            "ZZ",
            "WZ",
        ],
        "vjetslnu": [
            "WtoLNu-2Jets_2J",
            "DYto2L-2Jets_MLL-50_2J",
        ],
        "qcd": [
            "QCD_HT-400to600",
            "QCD_HT-600to800",
            "QCD_HT-800to1000",
            "QCD_HT-1000to1200",
            "QCD_HT-1200to1500",
            "QCD_HT-1500to2000",
            "QCD_HT-2000",
        ],
        "singletop": [
            "TbarBQ_t-channel_4FS",
            "TBbarQ_t-channel_4FS",
            "TbarWplusto4Q",
            "TWminusto4Q",
            "TbarWplustoLNu2Q",
            "TWminustoLNu2Q",
        ],
    },
    "2023BPix": {
        "diboson": [
            "WW",
            "ZZ",
            "WZ",
        ],
        "vjetslnu": [
            "WtoLNu-2Jets_2J",
            "DYto2L-2Jets_MLL-50_2J",
        ],
        "singletop": [
            "TbarBQ_t-channel_4FS",
            "TBbarQ_t-channel_4FS",
            "TbarWplusto4Q",
            "TWminusto4Q",
            "TbarWplustoLNu2Q",
            "TWminustoLNu2Q",
        ],
        "qcd": [
            "QCD_HT-200to400",
            "QCD_HT-400to600",
            "QCD_HT-600to800",
            "QCD_HT-800to1000",
            "QCD_HT-1000to1200",
            "QCD_HT-1200to1500",
            "QCD_HT-1500to2000",
            "QCD_HT-2000",
        ],
    },
}

samples_data = {
    "2022": {
        "data": [
            "Muon_Run2022C_single",
            "Muon_Run2022C",
            "Muon_Run2022D",
        ],
    },
    "2022EE": {
        "data": [
            "Muon_Run2022E",
            "Muon_Run2022F",
            "Muon_Run2022G",
        ],
    },
    "2023": {
        "data": [
            "Muon_Run2023Cv1",
            "Muon_Run2023Cv2",
            "Muon_Run2023Cv3",
            "Muon_Run2023Cv4",
        ],
    },
    "2023BPix": {
        "data": [
            "Muon_Run2023D",
        ]
    },
}

samples_tt = {
    "2022EE": {
        "ttbar": [
            "TTto4Q",
            "TTtoLNu2Q",
            "TTto2L2Nu",
        ],
    },
    "2022": {
        "ttbar": [
            "TTto4Q",
            "TTtoLNu2Q",
            "TTto2L2Nu",
        ],
    },
    "2023": {
        "ttbar": [
            "TTto4Q",
            "TTtoLNu2Q",
            "TTto2L2Nu",
        ],
    },
    "2023BPix": {
        "ttbar": [
            "TTto4Q",
            "TTtoLNu2Q",
            "TTto2L2Nu",
        ],
    },
}

def plot_variations(h_nom,h_variations,label_nom,labels_variations,plot_dir,name,xlabel,ylim=None):
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )
    nom_vals = h_nom.values()
    hep.histplot(
        h_nom,
        histtype="step",
        label=label_nom,
        yerr=False,
        color="k",
        ax=ax,
        linewidth=2,
    )
    colours = ["#81C14B", "#1f78b4"]

    for ivar, h_variation in enumerate(h_variations):
        hep.histplot(
            h_variation,
            histtype="step",
            yerr=False,
            label=labels_variations[ivar],
            color=colours[ivar],
            ax=ax,
            linewidth=2,
        )

        hep.histplot(
            h_variation / nom_vals,
            histtype="step",
            color=colours[ivar],
            ax=rax,
        )

    ax.legend()
    ax.set_ylim(0)
    ax.set_ylabel("Events")
    ax.set_title(name, y=1.08)

    rax.set_ylim([0, 2])
    if ylim is not None:
        rax.set_ylim(ylim)
    rax.set_xlabel(xlabel)
    rax.set_ylabel("Variation / Nominal")
    rax.grid(axis="y")

    plt.savefig(f"{plot_dir}/{name}.png", bbox_inches="tight")
    plt.close()

def save_to_file(out_file, hists_pass, hists_fail, save_variations=True):
    """
    Save histograms to file
    """
    f_out = uproot.recreate(out_file)
    var = "WMass"

    """
    In https://github.com/cms-jet/ParticleNetSF/blob/ParticleNet_TopW_SFs_NanoV9
    - catp3 = top_matched
    - catp2 = unmatched
    - catp1 = W_matched
    """

    # catp2 = matched
    # catp1 = unmatched

    f_out["data_obs_pass_nominal"] = hists_pass[var][{"Sample": "data"}]
    f_out["data_obs_fail_nominal"] = hists_fail[var][{"Sample": "data"}]

    # matched
    f_out["catp2_pass_nominal"] = sum(
        [
            hists_pass[var][{"Sample": sample}]
            for sample in hists_pass[var].axes[0]
            if sample in ["W_matched", "singletop"]
        ]
    )
    f_out["catp2_fail_nominal"] = sum(
        [
            hists_fail[var][{"Sample": sample}]
            for sample in hists_fail[var].axes[0]
            if sample in ["W_matched"]
        ]
    )

    # unmatched
    f_out["catp1_pass_nominal"] = sum(
        [
            hists_pass[var][{"Sample": sample}]
            for sample in hists_pass[var].axes[0]
            if sample in ["top_matched", "unmatched"]
        ]
    )
    f_out["catp1_fail_nominal"] = sum(
        [
            hists_fail[var][{"Sample": sample}]
            for sample in hists_fail[var].axes[0]
            if sample in ["top_matched", "unmatched"]
        ]
    )

    # other
    f_out["other_pass_nominal"] = sum(
        [
            hists_pass[var][{"Sample": sample}]
            for sample in hists_pass[var].axes[0]
            if sample in [
                #"diboson", 
                #"qcd", 
                "vjetslnu"
            ]
        ]
    )
    f_out["other_fail_nominal"] = sum(
        [
            hists_fail[var][{"Sample": sample}]
            for sample in hists_fail[var].axes[0]
            if sample
            in [
                # "diboson",
                "qcd",
                "vjetslnu",
                "singletop",
            ]
        ]
    )

    if save_variations:
        # save templates for variations
        variationmap = {
            "scaleUp": "JMS_up",
            "scaleDown": "JMS_down",
            "smearUp": "JMR_up",
            "smearDown": "JMR_down",
        }
        # catp2 templates
        for variation in variationmap:
            f_out[f"catp2_pass_{variation}"] = sum(
                [
                    hists_pass[f"{var}_{variationmap[variation]}"][{"Sample": sample}]
                    for sample in hists_pass[f"{var}_{variationmap[variation]}"].axes[0]
                    if sample in ["W_matched", "singletop"]
                ]
            )
            f_out[f"catp2_fail_{variation}"] = sum(
                [
                    hists_fail[f"{var}_{variationmap[variation]}"][{"Sample": sample}]
                    for sample in hists_fail[f"{var}_{variationmap[variation]}"].axes[0]
                    if sample in ["W_matched"]
                ]
            )

    f_out.close()


def get_ev_dataframe(events_dict, mass, pt_mask):
    """
    Get dataframe with W selection applied
    """

    def get_wtagger(events):
        return (events["ak8FatJetParTPXqq"][0] + events["ak8FatJetParTPXcs"][0]) / (
            events["ak8FatJetParTPXqq"][0]
            + events["ak8FatJetParTPXcs"][0]
            + events["ak8FatJetParTPQCD0HF"][0]
            + events["ak8FatJetParTPQCD1HF"][0]
            + events["ak8FatJetParTPQCD2HF"][0]
        )

    # def get_wtagger(events):
    #     return (events["ak8FatJetParTPXqq"][0] + events["ak8FatJetParTPXcs"][0]) / (
    #         events["ak8FatJetParTPXqq"][0]
    #         + events["ak8FatJetParTPXcs"][0]
    #         + events["ak8FatJetParTPQCD0HF"][0]
    #         + events["ak8FatJetParTPQCD1HF"][0]
    #         + events["ak8FatJetParTPQCD2HF"][0]
    #         + events["ak8FatJetParTPTopbW"][0]
    #     )
    # def get_wtagger(events):
    #     return (events["ak8FatJetParTPXqq"][0] + events["ak8FatJetParTPXcs"][0] + events["ak8FatJetParTPTopW"][0]) / (
    #         events["ak8FatJetParTPXqq"][0]
    #         + events["ak8FatJetParTPXcs"][0]
    #         + events["ak8FatJetParTPQCD0HF"][0]
    #         + events["ak8FatJetParTPQCD1HF"][0]
    #         + events["ak8FatJetParTPQCD2HF"][0]
    #         + events["ak8FatJetParTPTopbW"][0]
    #         + events["ak8FatJetParTPTopW"][0]
    #     )

    ev_dict = {}
    for key in events_dict:
        events = events_dict[key]

        wlnu_pt = events["leptonPt"][0] + events["MET_pt"][0]

        # apply masks
        events = events[
            (events["ak8FatJetPt"][0] > 300)
            & (events[f"ak8FatJet{mass}"][0] >= 55)
            & (events[f"ak8FatJet{mass}"][0] <= 200)
            & (wlnu_pt >= 100)
            & (events["ak8FatJetPt"][0] >= pt_mask[0])
            & (events["ak8FatJetPt"][0] < pt_mask[1])
        ]

        # form event dataframe
        ev_dataframe = pd.DataFrame(
            {
                "WMass": events[f"ak8FatJet{mass}"][0],
                "WMsd": events["ak8FatJetMsd"][0],
                "WPt": events["ak8FatJetPt"][0],
                "weight": events["finalWeight"],
                # customize W tagger
                "WTagger": get_wtagger(events),
                "WTXcs": (events["ak8FatJetParTPXcs"][0]) / (
                            events["ak8FatJetParTPXcs"][0]
                            + events["ak8FatJetParTPQCD0HF"][0]
                            + events["ak8FatJetParTPQCD1HF"][0]
                            + events["ak8FatJetParTPQCD2HF"][0]
                        ), 
                "WPXqq": events["ak8FatJetParTPXqq"][0],
                "WPTopW": events["ak8FatJetParTPTopW"][0],
                "WPTopbW": events["ak8FatJetParTPTopbW"][0],
                # Variations up and down
                "WMass_JMS_down": events[f"ak8FatJet{mass}"][0],
                "WMass_JMS_up": events[f"ak8FatJet{mass}"][0],
                "WMass_JMR_down": events[f"ak8FatJet{mass}"][0],
                "WMass_JMR_up": events[f"ak8FatJet{mass}"][0],
            }
        )
        if key != "data":
            ev_dataframe["WMass_JMS_down"] = events[f"ak8FatJet{mass}_JMS_down"][0]
            ev_dataframe["WMass_JMS_up"] = events[f"ak8FatJet{mass}_JMS_up"][0]
            ev_dataframe["WMass_JMR_down"] = events[f"ak8FatJet{mass}_JMR_down"][0]
            ev_dataframe["WMass_JMR_up"] = events[f"ak8FatJet{mass}_JMR_up"][0]

            #if key == "ttbar":
            #    print(key)
            #    print("JMR nominal ", events[f"ak8FatJet{mass}"][0])
            #    print("JMR down ", events[f"ak8FatJet{mass}_JMR_down"][0])
            #    print("JMR up ", events[f"ak8FatJet{mass}_JMR_up"][0])

        # identify ttbar jets matched and unmatched to top quark decays
        # TODO: apply singletop
        if key == "ttbar":
            # jet matched to 2 hadronic quarks
            has_2_daughter_qs = np.array(events["ak8FatJetNumQMatchedTop1"] == 2) != np.array(
                events["ak8FatJetNumQMatchedTop2"] == 2
            )
            # jet matched to 1 b-quark
            has_1_b = np.array(events["ak8FatJetNumBMatchedTop1"] == 1) != np.array(
                events["ak8FatJetNumBMatchedTop2"] == 1
            )

            # tighter matching definition by looking at agreement between Generator-Level W and Jet Pt and Mass
            vpt_matched = (
                (events["ak8FatJetPt"][0] - events["GenTopW0Pt"][0]) / events["GenTopW0Pt"][0] < 0.5
            ) | (
                (events["ak8FatJetPt"][0] - events["GenTopW1Pt"][0]) / events["GenTopW1Pt"][0] < 0.5
            )
            vmass_matched = (
                (events[f"ak8FatJet{mass}"][0] - events["GenTopW0Mass"][0])
                / events["GenTopW0Mass"][0]
                < 0.3
            ) | (
                (events[f"ak8FatJet{mass}"][0] - events["GenTopW1Mass"][0])
                / events["GenTopW1Mass"][0]
                < 0.3
            )

            top_matched = ((has_2_daughter_qs) & (has_1_b))[:, 0]
            W_matched = ((has_2_daughter_qs) & (~has_1_b))[:, 0]
            W_matched_tight = W_matched & vpt_matched & vmass_matched
            unmatched = (~has_2_daughter_qs)[:, 0] | (W_matched & ~W_matched_tight)

            ev_dict = {
                **ev_dict,
                "top_matched": ev_dataframe[top_matched],
                "W_matched": ev_dataframe[W_matched_tight],
                "unmatched": ev_dataframe[unmatched],
            }
        else:
            ev_dict[key] = ev_dataframe

    # create pass and fail regions
    ev_dict_pass = {}
    ev_dict_fail = {}
    for key in ev_dict:
        wtagger_mask = ev_dict[key]["WTagger"] >= 0.82
        wtagger_maskinv = ev_dict[key]["WTagger"] < 0.82
        ev_dict_pass[key] = ev_dict[key][wtagger_mask]
        ev_dict_fail[key] = ev_dict[key][wtagger_maskinv]

    # print efficiency
    npass = np.sum(ev_dict_pass["W_matched"]["weight"])
    nall = np.sum(ev_dict["W_matched"]["weight"])
    print("W matched efficiency: ", npass, nall, npass/nall)
    npass = np.sum(ev_dict_pass["unmatched"]["weight"])
    nall = np.sum(ev_dict["unmatched"]["weight"])
    print("unmatched efficiency: ", npass, nall, npass/nall)

    return ev_dict, ev_dict_pass, ev_dict_fail


@click.command()
@click.option(
    "--dir-name",
    help="directory name",
    default="/eos/uscms/store/user/cmantill/bbbb/ttSkimmer/24Nov6_v12v2_private_signal/", # version where GloParT mass is fixed
)
@click.option("--year-group", default="2022All", type=click.Choice(["2022All", "2023All"]))
@click.option("--tag", required=True)
@click.option("--mass", default="ParTmassVis", type=click.Choice(["ParTmassVis","PNetMass","PNetMassLegacy"]))
def jmsr_templates(dir_name, year_group, tag, mass):
    # group years
    years_to_process = {
        "2022All": ["2022", "2022EE"],
        "2023All": ["2023", "2023BPix"],
    }[year_group]

    # pt mask
    pt_mask = [300, 1000]

    # columns to load for all samples
    load_columns = [
        ("weight", 1),
        ("ak8FatJetPt", 1),
        ("ak8FatJetMsd", 1),
        (f"ak8FatJet{mass}", 1),
        ("ak8FatJetParTPXqq", 1),
        ("ak8FatJetParTPXcs", 1),
        ("ak8FatJetParTPTopW", 1),
        ("ak8FatJetParTPTopbW", 1),
        ("ak8FatJetParTPQCD0HF", 1),
        ("ak8FatJetParTPQCD1HF", 1),
        ("ak8FatJetParTPQCD2HF", 1),
        ("leptonPt", 1),
        ("MET_pt", 1),
    ]

    # columns to load for MC
    load_columns_mc = load_columns + [
        (f"ak8FatJet{mass}_JMS_down", 1),
        (f"ak8FatJet{mass}_JMS_up", 1),
        (f"ak8FatJet{mass}_JMR_down", 1),
        (f"ak8FatJet{mass}_JMR_up", 1),
    ]

    # columns to load for ttbar
    load_columns_tt = load_columns_mc + [
        ("GenTopW0Pt", 1),
        ("GenTopW0Mass", 1),
        ("GenTopW1Pt", 1),
        ("GenTopW1Mass", 1),
        ("ak8FatJetTopMatch", 1),
        ("ak8FatJetNumQMatchedTop1", 1),
        ("ak8FatJetNumQMatchedTop2", 1),
        ("ak8FatJetNumBMatchedTop1", 1),
        ("ak8FatJetNumBMatchedTop2", 1),
    ]

    # control plots
    control_plot_vars = [
        # 3.33 gev bins
        ShapeVar(var="WMass", label=r"W Mass (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
        #ShapeVar(var="WMsd", label=r"W Msd (GeV)", bins=[30, 50, 200], plot_args={"log": False}),
        ShapeVar(var="WMsd", label=r"W Msd (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
        ShapeVar(
            var="WPt", label=r"W p$_{T}$ (GeV)", bins=[30, 300, 800], plot_args={"log": False}
        ),
        ShapeVar(var="WTagger", label=r"W discriminator", bins=[30, 0, 1], plot_args={"log": True}),
        # plot extra discriminators
        ShapeVar(
            var="WPXqq", label=r"ParT Xqq probability", bins=[30, 0, 1], plot_args={"log": True}
        ),
        ShapeVar(
            var="WTXcs", label=r"ParT TXcs discriminator", bins=[30, 0, 1], plot_args={"log": True}
        ),
        ShapeVar(
            var="WPTopW", label=r"ParT TopW probability", bins=[30, 0, 1], plot_args={"log": True}
        ),
        ShapeVar(
            var="WPTopbW", label=r"ParT TopbW probability", bins=[30, 0, 1], plot_args={"log": True}
        ),
        # variations
        ShapeVar(
            var="WMass_JMS_down",
            label=r"W Mass JMS down (GeV)",
            bins=[21, 55, 125],
            plot_args={"log": False},
        ),
        ShapeVar(
            var="WMass_JMS_up",
            label=r"W Mass JMS up (GeV)",
            bins=[21, 55, 125],
            plot_args={"log": False},
        ),
        ShapeVar(
            var="WMass_JMR_down",
            label=r"W Mass JMR up (GeV)",
            bins=[21, 55, 125],
            plot_args={"log": False},
        ),
        ShapeVar(
            var="WMass_JMR_up",
            label=r"W Mass JMR up (GeV)",
            bins=[21, 55, 125],
            plot_args={"log": False},
        ),
    ]

    # variables to make stack plots for
    vars_stack = ["WMass","WMsd"]
    vars_stack_nopf = ["WTagger","WPTopW","WPTopbW","WPXqq","WTXcs","WPTopW","WPTopbW"]

    ev_dict = {}
    ev_dict_pass = {}
    ev_dict_fail = {}
    for year in years_to_process:
        events_dict = {
            **utils.load_samples(
                dir_name,
                samples_mc[year],
                year,
                filters=None,
                columns=utils.format_columns(load_columns_mc),
                reorder_txbb=False,
                variations=False,
            ),
            **utils.load_samples(
                dir_name,
                samples_data[year],
                year,
                filters=None,
                columns=utils.format_columns(load_columns),
                reorder_txbb=False,
                variations=False,
            ),
            **utils.load_samples(
                dir_name,
                samples_tt[year],
                year,
                filters=None,
                columns=utils.format_columns(load_columns_tt),
                reorder_txbb=False,
                variations=False,
            ),
        }

        events, events_pass, events_fail = get_ev_dataframe(events_dict, mass, pt_mask)

        if year_group in ev_dict:
            for key in ev_dict[year_group]:
                ev_dict[year_group][key] = pd.concat([ev_dict[year_group][key], events[key]])
                ev_dict_pass[year_group][key] = pd.concat([ev_dict_pass[year_group][key], events_pass[key]])
                ev_dict_fail[year_group][key] = pd.concat([ev_dict_fail[year_group][key], events_fail[key]])
        else:
            ev_dict[year_group] = events
            ev_dict_pass[year_group] = events_pass
            ev_dict_fail[year_group] = events_fail

    # create plotting directory
    odir = f"{tag}/pt{pt_mask[0]}-{pt_mask[1]}/{year_group}"
    os.system(f"mkdir -p {odir}")

    hists_all = {}
    hists_pass = {}
    hists_fail = {}

    # fill histograms
    for shape_var in control_plot_vars:
        print(shape_var)
        if shape_var.var not in hists_all:
            hists_all[shape_var.var] = utils.singleVarHist(
                ev_dict[year_group], shape_var, weight_key="weight"
            )
            hists_pass[shape_var.var] = utils.singleVarHist(
                ev_dict_pass[year_group], shape_var, weight_key="weight"
            )
            hists_fail[shape_var.var] = utils.singleVarHist(
                ev_dict_fail[year_group], shape_var, weight_key="weight"
            )

        # plot histograms
        if shape_var.var in vars_stack:
            h_to_plot = {
                f"{odir}/{shape_var.var}": hists_all[shape_var.var],
                f"{odir}/{shape_var.var}_pass": hists_pass[shape_var.var],
                f"{odir}/{shape_var.var}_fail": hists_fail[shape_var.var],
            }
        elif shape_var.var in vars_stack_nopf:
            h_to_plot = {
                f"{odir}/{shape_var.var}": hists_all[shape_var.var],
            }
        else:
            h_to_plot = {}
        for hname, h in h_to_plot.items():
            plotting.ratioHistPlot(
                h,
                year_group,
                sig_keys=[],
                bg_keys=bkgs,
                name=hname,
                show=False,
                bg_err=None,
                bg_order=bkg_order,
                plot_data=True,
                plot_significance=False,
                bg_err_mcstat=True,
                exclude_qcd_mcstat=False,
                save_pdf=False,
                **shape_var.plot_args,
                # ylim=1.2e4,
                # ylim_low=0,
            )

    # plot variations
    vars_notstack = {
        "WMass_JMS": ["WMass","WMass_JMS_down","WMass_JMS_up"],
        "WMass_JMR": ["WMass","WMass_JMR_down","WMass_JMR_up",]
    }
    for name,hvars in vars_notstack.items():
        plot_variations(
            hists_pass[hvars[0]][{"Sample": "W_matched"}],
            [hists_pass[i][{"Sample": "W_matched"}] for i in hvars[1:]],
            "Nominal",
            ["Down","Up"],
            odir,
            f"{name}_pass",
            "W Mass (GeV)",
        )
        plot_variations(
            hists_fail[hvars[0]][{"Sample": "W_matched"}],
            [hists_fail[i][{"Sample": "W_matched"}] for i in hvars[1:]],
            "Nominal",
            ["Down","Up"],
            odir,
            f"{name}_fail",
            "W Mass (GeV)",
        )

    # create template directory
    template_directory = f"TnPSF/run3_templates/{year_group}/{tag}"
    os.system(f"mkdir -p {template_directory}")
    out_file = f"{template_directory}/topCR_pt{pt_mask[0]}-{pt_mask[1]}.root"

    # save Wmass template and variations
    print(f"Save to file: {out_file}")
    save_to_file(out_file, hists_pass, hists_fail)


if __name__ == "__main__":
    jmsr_templates()
