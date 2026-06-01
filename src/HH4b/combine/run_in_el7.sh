#!/bin/bash
# One-shot wrapper that enters the EL7 singularity, sources CMSSW + DHI,
# adds HH4b/combine to PATH, and runs the given command.
#
# Usage:
#   run_in_el7.sh "<shell command(s)>"
#
# Example:
#   run_in_el7.sh "cd /home/users/dprimosc/HH4b/cards/run3-bdt-26Mar31/ && run_blinded_hh4b.sh -l --passbin 0"

set -e

[ -f /tmp/dhi_constraints.txt ] || echo "osqp==0.6.7.post3" > /tmp/dhi_constraints.txt

USER_CMD="$1"

# IMPORTANT: keep the inner command as a single line. cmssw-el7 --command-to-run
# passes its arg verbatim to /usr/bin/sh -c, where multi-line strings with
# backslash continuations break the && chain. Single line, single-quoted outside,
# inner $ references must be literal (so single-quote here — SC2016 is intentional).
# shellcheck disable=SC2016
PRELUDE='cd /home/users/dprimosc/HH4b/src/CMSSW_11_3_4/src && eval $(scram runtime -sh) && cd /home/users/dprimosc/HH4b/src/CMSSW_11_3_4/src/inference && export DHI_SCRAM_ARCH=slc7_amd64_gcc900 DHI_CMSSW_VERSION=CMSSW_11_3_4 DHI_COMBINE_VERSION=v9.1.0 PIP_CONSTRAINT=/tmp/dhi_constraints.txt && source setup.sh >/dev/null 2>&1 && export PATH=/home/users/dprimosc/HH4b/src/HH4b/combine:$PATH'

/cvmfs/cms.cern.ch/common/cmssw-el7 --command-to-run "${PRELUDE} && ${USER_CMD}"
