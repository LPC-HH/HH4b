import click
import os
import pandas as pd
import numpy as np
import uproot

import logging
import HH4b.utils as utils
from HH4b.utils import ShapeVar
import HH4b.plotting as plotting

#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
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

def save_to_file(out_file, hists_pass, hists_fail, save_variations=False):
    """
    Save histograms to file
    """
    f_out = uproot.recreate(out_file)
    var = "WMass"

    f_out[f"data_obs_pass_nominal"] = hists_pass[var][{"Sample": "data"}]
    f_out[f"data_obs_fail_nominal"] = hists_fail[var][{"Sample": "data"}]

    # matched
    f_out[f"catp2_pass_nominal"] = sum(
        [
            hists_pass[var][{"Sample": sample}]
            for sample in hists_pass[var].axes[0]
            if sample in ["W_matched", "singletop"]
        ]
    )
    f_out[f"catp2_fail_nominal"] = sum(
        [
            hists_fail[var][{"Sample": sample}]
            for sample in hists_fail[var].axes[0]
            if sample in ["W_matched"]
        ]
    )

    # unmatched
    f_out[f"catp1_pass_nominal"] = sum(
        [
            hists_pass[var][{"Sample": sample}]
            for sample in hists_pass[var].axes[0]
            if sample in ["top_matched", "unmatched", "diboson", "qcd", "vjetslnu"]
        ]
    )
    f_out[f"catp1_fail_nominal"] = sum(
        [
            hists_fail[var][{"Sample": sample}]
            for sample in hists_fail[var].axes[0]
            if sample in ["top_matched", "unmatched", "diboson", "qcd", "vjetslnu", "singletop"]
        ]
    )
    f_out.close()

def get_ev_dataframe(events_dict, mass, pt_mask):
    """
    Get dataframe with W selection applied
    """
    def get_wtagger(events):
        return (events["ak8FatJetParTPXqq"][0] ) / (events["ak8FatJetParTPXqq"][0] + events["ak8FatJetParTPQCD0HF"][0] + events["ak8FatJetParTPQCD1HF"][0] + events["ak8FatJetParTPQCD2HF"][0])

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

        # identify ttbar jets matched and unmatched to top quark decays
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
            unmatched = ((~has_2_daughter_qs))[:, 0] | (W_matched & ~W_matched_tight)

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

    return ev_dict, ev_dict_pass, ev_dict_fail

@click.command()
@click.option("--dir-name", help="directory name",
              default="/eos/uscms/store/user/cmantill/bbbb/ttSkimmer/24Oct14_v12v2_private_signal/")
@click.option("--year-group", default="2022All", type=click.Choice(["2022All", "2023All"]))
@click.option("--tag", required=True)
def jmsr_templates(dir_name, year_group, tag):
    # group years
    years_to_process = {
        "2022All": ["2022", "2022EE"],
        "2023All": ["2023", "2023BPix"],
    }[year_group]

    # mass branch to measure JMSR
    mass = "ParTmassVis"
    # mass = "PNetMass"
    # mass = "PNetMassLegacy"

    # pt mask
    pt_mask = [0, 1000]

    # columns to load for all samples
    load_columns = [
        ("weight", 1),
        ("ak8FatJetPt", 1),
        ("ak8FatJetMsd", 1),
        (f"ak8FatJet{mass}", 1),
        ("ak8FatJetParTPXqq", 1),
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
        ShapeVar(var="WMsd", label=r"W Msd (GeV)", bins=[30, 50, 200], plot_args={"log": False}),
        ShapeVar(var="WPt", label=r"W p$_{T}$ (GeV)", bins=[30, 300, 800], plot_args={"log": False}),
        ShapeVar(var="WTagger", label=r"W discriminator", bins=[30, 0, 1], plot_args={"log": True}),
        # plot extra discriminators
        ShapeVar(var="WPXqq", label=r"ParT Xqq probability", bins=[30, 0, 1], plot_args={"log": True}),
        ShapeVar(var="WPTopW", label=r"ParT TopW probability", bins=[30, 0, 1], plot_args={"log": True}),
        ShapeVar(var="WPTopbW", label=r"ParT TopbW probability", bins=[30, 0, 1], plot_args={"log": True}),
        # variations
        ShapeVar(var="WMass_JMS_down", label=r"W Mass JMS down (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
        ShapeVar(var="WMass_JMS_up", label=r"W Mass JMS up (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
        ShapeVar(var="WMass_JMR_down", label=r"W Mass JMR up (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
        ShapeVar(var="WMass_JMR_up", label=r"W Mass JMR up (GeV)", bins=[21, 55, 125], plot_args={"log": False}),
    ]

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

        if year in ev_dict:
            ev_dict[year_group] = pd.concat([ev_dict[year_group], events])
            ev_dict_pass[year_group] = pd.concat([ev_dict_pass[year_group], events_pass])
            ev_dict_fail[year_group] = pd.concat([ev_dict_fail[year_group], events_fail])
        else:
            ev_dict[year_group] = events
            ev_dict_pass[year_group] = events_pass
            ev_dict_fail[year_group] = events_fail

    # create plotting directory
    odir = f"{tag}/pt{pt_mask[0]-pt_mask[1]}/{year_group}"
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
        for hname, h in {
            f"{odir}/{shape_var.var}": hists_all[shape_var.var],
            f"{odir}/{shape_var.var}_pass": hists_pass[shape_var.var],
            f"{odir}/{shape_var.var}_fail": hists_fail[shape_var.var],
        }.items():
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
                **shape_var.plot_args
                # ylim=1.2e4,
                # ylim_low=0,
            )

    # create template directory
    template_directory = f"TnPSF/run3_templates/{year_group}/{tag}"
    os.system(f"mkdir -p {template_directory}")
    out_file = f"{template_directory}/topCR_pt{pt_mask[0]-pt_mask[1]}.root"

    # save Wmass template
    # TODO: save variations (e.g. WMass_JMS_down)
    save_to_file(out_file, hists_pass, hists_fail)

if __name__ == "__main__":
    jmsr_templates()