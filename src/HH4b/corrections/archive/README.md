# Archived Run 3 text-file JECs

Superseded by the official correctionlib JECs and JER (2024 and 2025 since #363, 2022 and 2023 since
https://github.com/LPC-HH/HH4b/pull/367): `processors/corrections.py` (`cat_jerc`) reads every Run 3
year from the JME correctionlib files on CVMFS through `processors/jec_clib.py`. Nothing in this
folder is read by the code; it is kept for reference and for reproducing older skims.

- `jec_compiled.pkl.gz`: coffea JEC stacks built from text files by `build_jec.py`: 2022/2022EE
  Summer22(EE)_22Sep2023 V2 with JER JRV1, 2023/2023BPix Summer23(BPix)Prompt23 V1 with JER JRV1,
  and provisional 2024/2025 stacks (Summer24Prompt24 RunCnib1 V1 residual for all data, AK4 files
  for AK8, 2023BPix JER). The skims made before the switch (e.g. `20260820_glopartv3_v15_signal`)
  used it.
- `jec_compiled_upto2023BPix.pkl.gz`: an older build with 2022-2023 only.
- `build_jec.py`: the builder (`python -m build_jec --output jec_compiled.pkl.gz` in this folder,
  with the text files in `data/jecs/`, which are not in the repository).

Run 2 still uses `../jec_compiled_run2.pkl.gz`, built by `../build_jec_run2.py`.
