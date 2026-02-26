# Postprocessing README (Run-3)

This document explains:

1. how Run-3 templates are generated in this repository, and  
2. how to unit test postprocessing logic without loading large parquet datasets.

## 1) How templates are generated

### Main entry point (current workflow)

Template production is driven by `PostProcess.py`.

Current single-era production command (run from `src/HH4b/postprocessing`):

```bash
nohup python3 PostProcess.py --templates-tag 26Jan15 --tag nanov15_20251202_v15_signal --mass H2PNetMass --txbb glopart-v3 --bdt-config v13_glopartv2 --bdt-model 25Feb5_v13_glopartv2_rawmass --fom-scan --data-dir /ceph/cms/store/user/zichun/bbbb/skimmer/ --no-vbf-priority --vbf --no-rerun-inference --txbb-wps 0.945 0.85 --bdt-wps 0.935 0.78 0.03 --vbf-txbb-wp 0.8 --vbf-bdt-wp 0.9825 --years 2024 --templates --no-fom-scan-bin1 --no-fom-scan-bin2 --no-fom-scan-vbf --control-plots > templates_2024.out 2>&1 &
```

Equivalent (same options, multi-line for readability):

```bash
nohup python3 PostProcess.py \
  --templates-tag 26Jan15 \
  --tag nanov15_20251202_v15_signal \
  --mass H2PNetMass \
  --txbb glopart-v3 \
  --bdt-config v13_glopartv2 \
  --bdt-model 25Feb5_v13_glopartv2_rawmass \
  --fom-scan \
  --data-dir /ceph/cms/store/user/zichun/bbbb/skimmer/ \
  --no-vbf-priority \
  --vbf \
  --no-rerun-inference \
  --txbb-wps 0.945 0.85 \
  --bdt-wps 0.935 0.78 0.03 \
  --vbf-txbb-wp 0.8 \
  --vbf-bdt-wp 0.9825 \
  --years 2024 \
  --templates \
  --no-fom-scan-bin1 \
  --no-fom-scan-bin2 \
  --no-fom-scan-vbf \
  --control-plots \
  > templates_2024.out 2>&1 &
```

### What the script does (high level)

`PostProcess.py` executes the following flow:

- Parses runtime configuration (`--years`, `--txbb`, `--bdt-model`, WPs, booleans like `--templates`, `--control-plots`, `--fom-scan`).
- Loads events per year/process via `load_process_run3_samples()`.
- Runs or loads BDT inference outputs and applies region/category selections.
- Builds control plots and optional FOM scans.
- Builds histogram templates (including jec/jmsr shifts) and saves them.

### Key outputs

Given `--templates-tag <TAG>`:

- Control/FOM plots: `plots/PostProcess/<TAG>/...`
- Template files: `templates/<TAG>/<YEAR>_templates.pkl`
- Cutflows: `templates/<TAG>/cutflows/preselection_cutflow_<YEAR>.csv`
- Runtime args snapshot: `templates/<TAG>/args.txt`

### Optional batch helper

For sequential multi-era processing with separate per-year logs, use:

```bash
./run_templates_queue.sh --daemon
```

## 2) Unit testing postprocessing

### Goal

Catch regressions in selector naming, year configuration, and orchestration logic without heavy IO.

### Test files

The current lightweight test suite is:

- `tests/test_selectors.py`
  - `check_selector()` behavior for exact (`?`), prefix, and contains (`*`) matching.
- `tests/test_xsec_resolution.py`
  - `_resolve_xsec_key()` alias/case-insensitive fallback behavior.
- `tests/test_hhvars_contracts.py`
  - year/sample/lumi configuration contracts (including 2024/2025-related guards).
- `tests/test_postprocessing_load_run3_samples.py`
  - monkeypatched `load_run3_samples()` orchestration test, no real parquet reads.

### How these tests stay fast

- no real skimmer directory scans
- no parquet reads
- no heavy template creation
- monkeypatching of `utils.load_samples` for orchestration coverage

### Run tests

From repo root:

```bash
PYTHONPATH=src python -m pytest tests/test_selectors.py tests/test_xsec_resolution.py tests/test_hhvars_contracts.py tests/test_postprocessing_load_run3_samples.py -q
```

If you want the full test suite:

```bash
PYTHONPATH=src python -m pytest -q
```

### Environment note

Some tests intentionally use dependency guards (`pytest.importorskip(...)`) so they skip cleanly in minimal environments.  
In the standard analysis environment (with `hist`, `pandas`, etc. installed), those tests execute normally.

## Practical debugging tips

- If a background is unexpectedly empty, verify `hh_vars.samples_run3[year]` includes that background key and selectors match on-disk naming for that era.
- For per-era failures, inspect `src/HH4b/postprocessing/templates_<YEAR>.out` first.
- When adding new eras, keep `hh_vars.years`, `samples_run3`, `LUMI`, and postprocessing HLT/correction fallbacks consistent.
