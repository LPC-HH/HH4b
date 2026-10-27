#!/bin/bash
# Install brilcalc (brilws) for use outside CERN, e.g. on the UAF: a venv on top of the CVMFS
# brilconda, whose SQLAlchemy has the Frontier dialect that brilcalc -c web needs (the PyPI one
# does not). Run brilcalc with PYTHONNOUSERSITE=1 so that a user-site SQLAlchemy does not shadow
# brilconda's (golden_lumi.py sets it).
#
# Usage: ./install_brilcalc.sh [venv directory, default brilenv]
#   PYTHONNOUSERSITE=1 brilenv/bin/brilcalc lumi -c web -b "STABLE BEAMS" -u /pb \
#       --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -i <JSON>
#
# Author(s): Zichun Hao

set -euo pipefail

venv=${1:-brilenv}
/cvmfs/cms-bril.cern.ch/brilconda310/bin/python3 -m venv --system-site-packages "$venv"
PYTHONNOUSERSITE=1 "$venv/bin/python" -m pip install --quiet --no-deps brilws==3.9.4
PYTHONNOUSERSITE=1 "$venv/bin/brilcalc" --version
