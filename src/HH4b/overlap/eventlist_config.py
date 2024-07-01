from __future__ import annotations

from argparse import Namespace


def eventlist_args(data_folder, year):
    folder = data_folder
    args = Namespace(
        templates_tag="24June27",
        data_dir="/ceph/cms/store/user/cmantill/bbbb/skimmer/",
        tag=folder,
        years=[year],
        training_years=None,
        mass="H2PNetMass",
        bdt_model="24May31_lr_0p02_md_8_AK4Away",
        bdt_config="24May31_lr_0p02_md_8_AK4Away",
        txbb_wps=[0.975, 0.92],
        bdt_wps=[0.98, 0.88, 0.03],
        method="sideband",
        vbf_txbb_wp=0.95,
        vbf_bdt_wp=0.98,
        sig_keys=["hh4b", "vbfhh4b"],
        pt_first=300,
        pt_second=250,
        bdt_roc=False,
        control_plots=False,
        fom_scan=False,
        fom_scan_bin1=True,
        fom_scan_bin2=True,
        fom_scan_vbf=False,
        templates=False,
        legacy=True,
        vbf=True,
        vbf_priority=False,
        weight_ttbar_bdt=1,
        blind=True,
    )
    return args
