import os
import sys
import shutil
import warnings

import re
import numpy as np
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     from uproot3_methods.classes.TH1 import Methods as TH1Methods
#     import uproot3
import uproot
# import hist
import boost_histogram as bh

import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import interp1d
import scipy.stats

_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)
def poisson_interval(sumw, sumw2, coverage=_coverage1sd):
    """Frequentist coverage interval for Poisson-distributed observations
    Parameters
    ----------
        sumw : numpy.ndarray
            Sum of weights vector
        sumw2 : numpy.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%
    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a
    proper treatment. When a bin is zero, the scale of the nearest nonzero bin is
    substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    # Taken from Coffea
    """
    scale = np.empty_like(sumw)
    scale[sumw != 0] = sumw2[sumw != 0] / sumw[sumw != 0]
    if np.sum(sumw == 0) > 0:
        missing = np.where(sumw == 0)
        available = np.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn(
                "All sumw are zero!  Cannot compute meaningful error bars",
                RuntimeWarning,
            )
            return np.vstack([sumw, sumw])
        nearest = np.sum(
            [np.subtract.outer(d, d0) ** 2 for d, d0 in zip(available, missing)]
        ).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.0
    hi = scale * scipy.stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.0
    interval = np.array([lo, hi])
    interval[interval == np.nan] = 0.0  # chi2.ppf produces nan for counts=0
    return interval

class AffineMorphTemplate(object):
    def __init__(self, h_obj):
        '''
        hist: a numpy-histogram-like tuple of (sumw, edges)
        '''
        # print(h_obj)
        # self.sumw = hist.values()
        # self.edges = hist.axes[0].edges
        # try:
        #     self.sumw = hist.values()
        #     self.edges = hist.axes[0].edges
        # except:
        #     self.sumw, self.edges = hist
        self.sumw, self.edges = h_obj
        self.centers = self.edges[:-1] + np.diff(self.edges)/2
        self.norm = self.sumw.sum()
        self.mean = (self.sumw*self.centers).sum() / self.norm
        self.cdf = interp1d(x=self.edges,
                            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
                            kind='linear',
                            assume_sorted=True,
                            bounds_error=False,
                            fill_value=(0, 1),
                           )
        
    def get(self, shift=0., scale=1.):
        '''
        Return a shifted and scaled histogram
        i.e. new edges = edges * scale + shift
        '''
        if not np.isclose(scale, 1.):
            shift += self.mean * (1 - scale)
        scaled_edges = (self.edges - shift) / scale
        return np.diff(self.cdf(scaled_edges)) * self.norm, self.edges

    def scale(self, n):
        self.norm = self.norm * n
     

class MorphHistW2(object):
    def __init__(self, h_obj):
        '''
        hist: uproot/UHI histogram or a tuple (values, edges, variances)
        '''
        self.original = h_obj
        try:
            self.sumw = h_obj.values()
            self.edges = h_obj.axes[0].edges()
            self.variances = h_obj.variances()
        except:
            self.sumw, self.edges, self.variances = h_obj
        
        # from mplhep.error_estimation import poisson_interval
        down, up = np.nan_to_num(np.abs(poisson_interval(self.sumw, self.variances)), 0.)

        self.nominal = AffineMorphTemplate((self.sumw, self.edges))
        self.w2s = AffineMorphTemplate((self.variances, self.edges))
        
    def get(self, shift=0., scale=1.):
        nom, edges = self.nominal.get(shift, scale)
        w2s, edges = self.w2s.get(shift, scale)       
        return nom, edges, w2s


# class TH1(TH1Methods, list):
#     pass
from uproot.behaviors import TH1

class TAxis(object):
    def __init__(self, fNbins, fXmin, fXmax):
        self._fNbins = fNbins
        self._fXmin = fXmin
        self._fXmax = fXmax


def export1d(h_obj, name='x', label='x', histtype=b"TH1F"):
    """Export a 1-dimensional `Hist` object to uproot

    """
    try:
        sumw, edges, sumw2 = h_obj
    except:
        sumw, edges = h_obj
        sumw2 = sumw

    # h = hist.new.Var(edges, name=name, label=label).Weight()
    h = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Weight())
    h.view().value = sumw
    h.view().variance = sumw2
    return h


def mdev(h_obj):
    w, edges = h_obj
    N = np.sum(w)
    centers = edges[:-1] + 0.5*np.diff(edges)
    mean = 1/N * np.sum(w * centers)
    stdev2 = 1/N * np.sum(w * (centers-mean)**2)
    return np.array([mean, np.sqrt(stdev2)])

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('-i', '--in', dest='in_file', required=True, help="Source file")
    parser.add_argument('-o', '--out', dest='out_file', default=None, help="Out file")
    parser.add_argument("--scale", default='1', type=float, help="Scale value.")
    parser.add_argument("--smear", default='0.5', type=float, help="Smear value.")
    parser.add_argument('--plot', action='store_true', help="Make control plots")
    parser.add_argument('--systs', action='store_true', default=False, help="Run systematics too.")
    parser.add_argument('--type', dest='hist_type', type=str, choices=["TH1F", "TH1D"], default="TH1D", help="TH1 type. Should be consistent with input.")
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = args.in_file.replace(".root", "_var.root")
    print("Running with the following options:")
    print(args)

    source_file = uproot.open(args.in_file)
    if os.path.exists(args.out_file):
        os.remove(args.out_file)
    fout = uproot.recreate(args.out_file)

    work_dir = os.path.dirname(args.in_file)

    # scale catp2 templates
    for template_name in [k for k in source_file.keys() if 'catp2' in k]:
        if not args.systs and 'nominal' not in template_name:
            # print(f"continue: {template_name}")
            continue
        # print(f"do: {template_name}")
        
        template_name = template_name.split(";")[0]

        morph_base = MorphHistW2(source_file[template_name])

        scale_up = morph_base.get(shift=args.scale)
        scale_down = morph_base.get(shift=-args.scale)
        smear_up = morph_base.get(scale=1+args.smear)
        smear_down = morph_base.get(scale=1-args.smear)

        if args.plot:
            import matplotlib.pyplot as plt
            import mplhep as hep

            plt.style.use(hep.style.ROOT)
            fig, ax = plt.subplots()
            hep.histplot(morph_base.get()[:2], color='black' , ls=':', label='Nominal')
            hep.histplot(scale_up[:2], color='blue' , ls='--', label='Up')
            hep.histplot(scale_down[:2], color='red' , ls='--', label='Down')
            ax.set_xlabel('Jet mass (GeV)')
            ax.legend()
            fig.savefig('{}/plot_{}_scale.png'.format(work_dir, template_name))

            fig, ax = plt.subplots()
            hep.histplot(morph_base.get()[:2], color='black' , ls=':', label='Nominal')
            hep.histplot(smear_up[:2], color='blue' , ls='--', label='Up')
            hep.histplot(smear_down[:2], color='red' , ls='--', label='Down')
            ax.set_xlabel('Jet mass (GeV)')
            ax.legend()
            fig.savefig('{}/plot_{}_smear.png'.format(work_dir, template_name))

        fout[template_name] = export1d(MorphHistW2(source_file[template_name]).get(), histtype=args.hist_type)
        fout[template_name.replace("nominal", "smearDown")] = export1d(smear_down, histtype=args.hist_type)
        fout[template_name.replace("nominal", "smearUp")] = export1d(smear_up, histtype=args.hist_type)
        fout[template_name.replace("nominal", "scaleDown")] = export1d(scale_down, histtype=args.hist_type)
        fout[template_name.replace("nominal", "scaleUp")] = export1d(scale_up, histtype=args.hist_type)

        # template_name

    # Clone remaining templates:
    for template_name in [k for k in source_file.keys() if 'catp2' not in k]:
        template_name = template_name.split(";")[0]
        if not args.systs and 'nominal' not in template_name:
            continue
        fout[template_name] = export1d(MorphHistW2(source_file[template_name]).get(), histtype=args.hist_type)

    fout.close()