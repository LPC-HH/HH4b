# SM-only fixture cards

Tiny synthetic combine datacards for use by the test suite. Not derived
from real data — values are stylized after a real `run3-bdt-26Mar31_SMonly`
card but simplified.

## What's in the cards

- 2 signal processes: `ggHH`, `qqHH` (SM couplings)
- 4 backgrounds: `ttbar`, `VH_hbb`, `vjets`, `ttH_hbb`
- 10 `lnN` nuisances (no `shape` templates → no ROOT files required;
  fixtures are pure text)

## Files

| file | contents |
|------|----------|
| `passbin1.txt` | single passbin1 bin, high-S/B (signal-rich region) |
| `passbin2.txt` | single passbin2 bin, mid S/B |
| `passbin3.txt` | single passbin3 bin, low-S/B (ttbar-dominated) |
| `passvbf.txt`  | single passvbf bin, VBF-tagged region |
| `fail.txt`     | single fail bin, QCD-dominated control region |
| `combined.txt` | all five bins side by side (what `combineCards.py` would produce) |

## Who uses them

- `tests/test_card_sanity.py` — structural sanity (signal count, rate vs
  process column alignment, no duplicate NPs).
- `tests/test_prepare_snapshot.py` — end-to-end test of
  `prepare_snapshot_inference.sh` against stubbed combine/DHI binaries.

## Updating

If you add a new process or NP, regenerate `combined.txt` so it stays in
sync with the per-bin cards. The simplest manual recipe: keep the column
order consistent (`ggHH qqHH ttbar VH_hbb vjets ttH_hbb`) and copy the
five per-bin blocks side by side.
