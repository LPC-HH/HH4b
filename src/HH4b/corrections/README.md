## JECs

Run 3 (2022-2025): the official JME correctionlib files on CVMFS, pinned in
`processors/corrections.py` (`cat_jerc`) and read at run time through `processors/jec_clib.py`
(https://github.com/LPC-HH/HH4b/pull/366). A version update is a new dated folder and tag there;
nothing to build. The earlier text-file pickles and their builder are in `archive/`.

Run 2:

```
python -m build_jec_run2 jec_compiled_run2.pkl.gz
```

## LUMI recommendations

https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3

## Pileup reweighting

No official recommendation:
https://cms-talk.web.cern.ch/t/pu-reweighing-in-2022mc/28786/9

- Following https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/merge_requests/47
- Updated with
  https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/merge_requests/74

Using the "CMS value" of 69.2 mb instead of 80 mb, which better matches the
data.
