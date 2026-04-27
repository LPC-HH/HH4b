import ROOT as r
import uproot
import json
import os
import pprint
import numpy as np
import matplotlib
import warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from matplotlib.offsetbox import AnchoredText
import argparse

from rhalphalib import MorphHistW2

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convTH1(TH1):
    vals = TH1.values()
    # edges = TH1.edges
    edges = TH1.axes[0].edges()
    variances = TH1.variances()
    vals = vals * np.diff(edges)
    variances = variances * np.diff(edges)
    return vals, edges, variances


def covnTGA(tgasym):
        # https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/wiki/nonstandard
        # Rescale density by binwidth for actual value
        # _binwidth = tgasym._fEXlow + tgasym._fEXhigh
        # _x = tgasym._fX
        # _y = tgasym._fY * _binwidth
        # _xerrlo, _xerrhi = tgasym._fEXlow, tgasym._fEXhigh
        # _yerrlo, _yerrhi = tgasym._fEYlow * _binwidth, tgasym._fEYhigh * _binwidth
        # return _x, _y, [_yerrlo, _yerrhi], [_xerrlo, _xerrhi]
        _x, _y = tgasym.values(axis='both')
        _xerrlo, _xerrhi = tgasym.errors("low", axis='x'), tgasym.errors("high", axis='x')
        _yerrlo, _yerrhi = tgasym.errors("low", axis='y'), tgasym.errors("high", axis='y')
        _binwidth = _xerrlo + _xerrhi
        _y = _y * _binwidth
        _yerrlo = _yerrlo * _binwidth
        _yerrhi = _yerrhi * _binwidth
        return _x, _y, [_yerrlo, _yerrhi], [_xerrlo, _xerrhi]
        

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", default='', help="Model/Fit dir")
parser.add_argument("-i",
                    "--input",
                    default='fitDiagnosticsTest.root',
                    help="Input shapes file")
parser.add_argument("--fit",
                    default='fit_s',
                    choices={"prefit", "fit_s"},
                    dest='fit',
                    help="Shapes to plot")
parser.add_argument("-o",
                    "--output-folder",
                    default='plots',
                    dest='output_folder',
                    help="Folder to store plots - will be created if it doesn't exist.")
parser.add_argument("--year",
                    default=None,
                    choices={"2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix", "2022All", "2023All"},
                    type=str,
                    help="year label")
parser.add_argument("--sflabels",
                    default="ParTWtagger,",
                    type=str,
                    help="labels")
parser.add_argument('-f',
                    "--format",
                    type=str,
                    default='png',
                    choices={'png', 'pdf'},
                    help="Plot format")
parser.add_argument("--scale",
                    required=True,
                    type=float,
                    help="Scale value, as used in template generation. (datacard scaling is parsed automatically)")
parser.add_argument("--smear",
                    required=True,
                    type=float,
                    help="Smear value, as used in template generation. (datacard scaling is parsed automatically)")

args = parser.parse_args()
if args.output_folder.split("/")[0] != args.dir:
    args.output_folder = os.path.join(args.dir, args.output_folder)

rd = r.TFile.Open(os.path.join(args.dir, args.input))
fd = uproot.open(os.path.join(args.dir, args.input))
with open(os.path.join(args.dir,'config.json')) as cfg_file:
    cfg = json.load(cfg_file)
print("Input config ",cfg)

if not os.path.exists(os.path.join(args.dir, 'plots')):
    os.mkdir(os.path.join(args.dir, 'plots'))

par_names = rd.Get('fit_s').floatParsFinal().contentsString().split(',')
par_names = [p for p in par_names if 'smear' in p or 'scale' in p  or 'eff' in p]

out = {}
for pn in par_names:
    out[pn] = {}
    out[pn]['val'] = round(rd.Get('fit_s').floatParsFinal().find(pn).getVal(), 3)
    out[pn]['unc'] = round(rd.Get('fit_s').floatParsFinal().find(pn).getError(), 3)

if np.isclose(abs(out['CMS_scale']['val']), 1):
    warnings.warn("abs(CMS_scale) is 1 (combine hist interpolation boundary)."
                  "Try inject additional scaling into datacards with --scale `>1`"
                  "when running sf.py")
if np.isclose(abs(out['CMS_smear']['val']), 1):
    warnings.warn("abs(CMS_smear) is 1 (combine hist interpolation boundary)."
                  "Try inject additional scaling into datacards with --smear `>1`"
                  "when running sf.py")

print("outvalues before multiplication ",out)

out['shift_SF'] = cfg['scale'] * out['CMS_scale']['val'] * args.scale  # (template shape)
out['shift_SF_ERR'] = cfg['scale'] * out['CMS_scale']['unc'] * args.scale  # (template shape)
#if cfg['smear'] != 1:
print("not summing 1!")
# out['smear_SF'] = 1+ cfg['smear']
out['smear_SF'] = cfg['smear'] * out['CMS_smear']['val'] * args.smear  # (template shape)
out['smear_SF_ERR'] = cfg['smear'] * out['CMS_smear']['unc'] * args.smear # (template shape)

# multiply by 0.1!
#out['shift_SF'] = out['shift_SF'] * 0.1
#out['shift_SF_ERR'] = out['shift_SF_ERR'] * 0.1
#out['smear_SF'] = out['smear_SF'] * 0.1
#out['smear_SF_ERR'] = out['smear_SF_ERR'] * 0.1

if 'effSF' in out.keys():
    out['V_SF'] = out['effSF']['val']
    out['V_SF_ERR'] = out['effSF']['unc']
if 'effwSF' in out.keys():
    out['W_SF'] = out['effwSF']['val']
    out['W_SF_ERR'] = out['effwSF']['unc']

pprint.pprint(out)

plt.style.use([hep.style.ROOT])


shapetype = 'shapes_{}'.format(args.fit)
print(fd[shapetype].keys())
regions = [r.replace(";1", '') for r in fd[shapetype].keys() if "/" not in r]

lumi = {
    "2016" : 36.33,
    "2017" : 41.53,
    "2018" : 59.74,
    "2022": 7.97,
    "2022EE": 26.34,
    "2023": 17.98,
    "2023BPix": 9.52,
    "2022All": 34.31,
    "2023All": 27.5,
}

for i, reg in enumerate(regions):

    # Plot FIT
    print(f"Plotting {args.fit} for {reg}")
    fig, (ax, rax) = plt.subplots(2,1, figsize=(10, 10), gridspec_kw = {'height_ratios':[3, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    ho0 = convTH1(fd[shapetype+'/'+reg+'/other'])
    ho1 = convTH1(fd[shapetype+'/'+reg+'/unmatched'])
    ho2 = convTH1(fd[shapetype+'/'+reg+'/wqq'])
    # data is a tgraph, 0th axis are bins in x-axis, 1st axis are bin content, 2nd axis are unc.
    # there is a 3rd axis which I don't understand...
    tgo = covnTGA(fd[shapetype+'/'+reg+'/data'])
    
    hwqq_prefit = convTH1(fd['shapes_prefit/'+reg+'/wqq'])
    hwqq_prefit = (hwqq_prefit[0]* out['V_SF'], hwqq_prefit[1], hwqq_prefit[2])

    ax.errorbar(tgo[0], tgo[1], yerr=tgo[2], xerr=tgo[3], fmt='o', color='black', label='Data')
    hep.histplot([ho0[0], ho1[0], ho2[0]], ho1[1], stack=True, ax=ax, label=['Other', 'Unmatched', "Matched"], histtype='fill')
    #hep.histplot(hwqq_prefit, color="blue", ax=ax, label="Matched Prefit * WSF", linestyle="dashed")

    ax.set_xlim(40, 136.5)
    ax.set_ylabel('Events', y=1, ha='right')
    ax.legend()

    # arrays of values with uncertainties can be built from values and uncertainties (e.g. tgo[1] and tgo[2])
    from uncertainties import unumpy
    # tgo[2] is the unc in data which are assymetric so it is 2d array with up and dn variations
    data = unumpy.uarray(tgo[1], tgo[2])
    mc2 =  unumpy.uarray(ho2[0], ho2[2])
    mc1 =  unumpy.uarray(ho1[0], ho1[2])
    mc0 =  unumpy.uarray(ho0[0], ho0[2])
    ratio = data/(mc0 + mc1 + mc2)

    """
    print(f"tgo {tgo[1]} {tgo[2]}")
    print(f"Data {data}")
    print(f"mc0-other {mc0}")
    print(f"mc1-unmatched {mc1}")
    print(f"mc2-wqq {mc2}")
    print(f"ratio: {ratio}")
    print(f"nominal: {unumpy.nominal_values(ratio)[0]}")
    print(f"std: {unumpy.std_devs(ratio)}")
    """

    rax.errorbar(tgo[0], unumpy.nominal_values(ratio)[0], unumpy.std_devs(ratio), xerr=tgo[3], fmt='o', color='black', label='Data')
    rax.hlines(1, 40, 150, linestyle='--', color='k', alpha=0.7)
    rax.set_ylim(0.4, 1.6)
    rax.set_ylabel("Data/MC")
    rax.set_xlabel("Jet mass (GeV)", x=1, ha='right');
    
    hep.cms.label(ax=ax, data=True, year=args.year, lumi=lumi[args.year])
    
    if i == len(regions) - 1:
        ax.set_ylim(None, ax.get_ylim()[-1]*1.5)    
        if "Secondary" in reg:
            sfl1, sfl2 = args.sflabels.split(",")
            sfstr = ("SF ({}) = {:.3f} $\pm$ {:.3f}".format(sfl1, out['effwSF']['val'], out['effwSF']['unc'])
                + "\nSF ({}) = {:.3f} $\pm$ {:.3f}".format(sfl2, out['effSF']['val'], out['effSF']['unc'])
                + "\nScale = {:.3f} $\pm$ {:.3f} GeV".format(out['shift_SF'], out['shift_SF_ERR'])
                + "\nSmear = {:.3f} $\pm$ {:.3f}".format(out['smear_SF'], out['smear_SF_ERR'])
            )
        else:
            sfl1, sfl2 = args.sflabels.split(",")
            mult_factor = 0.1
            #mult_factor = 0.05
            print(f"MULT by {mult_factor}~~~~!!!!")
            sfstr = (
                #"SF ({}) = {:.3f} $\pm$ {:.3f}".format(sfl1, out['effSF']['val'], out['effSF']['unc'])
                #+ "\nScale = {:.3f} $\pm$ {:.3f} GeV".format(out['shift_SF'], out['shift_SF_ERR'])
                # added for 0.1
                "\nSF JMS = {:.3f} $\pm$ {:.3f}".format(1+out['shift_SF']*mult_factor, out['shift_SF_ERR']*mult_factor)
                +"\nSF JMR = {:.3f} $\pm$ {:.3f}".format(1+out['smear_SF']*mult_factor, out['smear_SF_ERR']*mult_factor)
            )
            
        at = AnchoredText(sfstr,
                    loc='upper left', frameon=False, prop=dict(size=20)
                    )
        ax.add_artist(at)

    fig.savefig('{}/{}.{}'.format(args.output_folder, shapetype+'_'+reg, 'pdf'),
                bbox_inches="tight")
    fig.savefig('{}/{}.{}'.format(args.output_folder, shapetype+'_'+reg, 'png'),
                bbox_inches="tight")

# Plot SHIFT
for i, reg in enumerate(regions):
    print(reg)
    hwqq_prefit = convTH1(fd['shapes_prefit/'+reg+'/wqq'])
    hwqq_postfit = convTH1(fd['shapes_fit_s/'+reg+'/wqq'])

    #print(hwqq_prefit[0])
    def mean_sigma(h):
        mids = 0.5*(h[1][1:] + h[1][:-1])
        n = h[0]
        probs = h[0] / np.sum(h[0])
        mean = np.sum(probs * mids)
        sigma = np.sqrt(np.average((mids - mean)**2, weights=n))
        return mean, sigma
    
    mean, sigma = mean_sigma(hwqq_prefit)
    print(f"Mean {mean:.2f}, Sigma {sigma:.2f}")

    mean_posfit, sigma_posfit = mean_sigma(hwqq_postfit)
    print(f"Posfit: Mean {mean_posfit:.2f}, Sigma {sigma_posfit:.2f}")


    scale = 1+out['shift_SF']/mean
    scale_var = out['shift_SF_ERR']/mean
    print(f"scaleSF {scale:.3f} +/- {scale_var:.4f} ({scale_var*100/scale:.2f}%)")
    print(f"scaleSF {scale:.3f}, {scale-scale_var:.4f}, {scale+scale_var:.3f}")
    smear = out['smear_SF']
    print(f"smearSF {smear:.3f} +/- {out['smear_SF_ERR']:.3f} ({out['smear_SF_ERR']*100/smear:.2f}%)")
    print(f"smearSF {smear:.3f}, {smear-out['smear_SF_ERR']:.3f}, {smear+out['smear_SF_ERR']:.3f}")

    templ_smear = MorphHistW2(
        (hwqq_prefit[0], hwqq_prefit[1], "mass", hwqq_prefit[2])
    ).get(
        shift=(scale-1)*mean, 
        smear=smear
    )
    mean, sigma = mean_sigma(templ_smear)
    print(f"Posfit Morphed: Mean {mean:.2f}, Sigma {sigma:.2f}")
    
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    hep.histplot(hwqq_prefit, label="Prefit", color="blue", density=True)
    hep.histplot(hwqq_postfit, label="Postfit", color="red", density=True)
    hep.histplot(templ_smear[:2], label=r"Prefit * (JMS SF - 1) $\times \mu$ * JMR SF", color="black", linestyle='dashed', density=True)
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("a.u.")
    ax.legend(bbox_to_anchor=(1.03, 1), loc="upper left")
    fig.savefig('{}/{}.{}'.format(args.output_folder, 'wqq_'+reg, 'png'),
                bbox_inches="tight")
