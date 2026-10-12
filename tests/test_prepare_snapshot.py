"""End-to-end test for src/HH4b/combine/prepare_snapshot_inference.sh.

Copies the fixture per-bin cards into a temp directory, drops stub
shell scripts for the combine/DHI binaries the script calls onto PATH,
runs the real prepare script, and asserts the expected outputs appear
with the right group lines attached.

This catches silent-failure modes we have hit in real runs: missing
``set -e`` letting half the script keep going after the first half
errored out, an external binary getting renamed without updating the
caller, or a new systematic group being added to ``add_parameter.py``
calls without making it into the output cards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cards" / "sm_only"
REPO_ROOT = Path(__file__).parent.parent
PREPARE_SCRIPT = REPO_ROOT / "src" / "HH4b" / "combine" / "prepare_snapshot_inference.sh"

PER_BIN_NAMES = ["passbin1", "passbin2", "passbin3", "passvbf", "fail"]
EXPECTED_NOMASK_CARDS = [
    "passbin1_nomasks.txt",
    "passbin2_nomasks.txt",
    "passbin3_nomasks.txt",
    "passvbf_nomasks.txt",
    "combined_nomasks.txt",
]

# Stub shell scripts that stand in for the real combine/DHI binaries. Each
# stub does the bare minimum the prepare_snapshot script needs from it: create
# expected output files, or append group lines to a given card. The stubs are
# pure bash so they don't depend on a Python combine/DHI install.
_STUBS = {
    "run_blinded_hh4b.sh": textwrap.dedent("""\
        #!/bin/bash
        # Stand-in for the blinded workspace+bfit driver. Just touch the two
        # ROOT files the real script would produce so downstream steps find
        # them.
        touch combined_withmasks.root
        touch higgsCombineSnapshot.MultiDimFit.mH125.root
        """),
    "extract_fit_result.py": textwrap.dedent("""\
        #!/bin/bash
        # Real signature:
        #   extract_fit_result.py <input.root> "w:MultiDimFit" <output.json> --keep '*'
        # The 3rd positional is the output JSON path.
        echo '{}' > "$3"
        """),
    "combineCards.py": textwrap.dedent("""\
        #!/bin/bash
        # Args look like:  bin1=file1.txt bin2=file2.txt ...
        # The real tool produces a combined datacard on stdout; the caller
        # redirects with >. For our test we don't need a valid combined card,
        # so we just concatenate the inputs in order. add_parameter stubs
        # will append group lines to the resulting file afterwards.
        for arg in "$@"; do
            if [[ "$arg" == *=* ]]; then
                cat "${arg#*=}"
            fi
        done
        """),
    "add_parameter.py": textwrap.dedent("""\
        #!/bin/bash
        # Real signature:
        #   add_parameter.py <card> <name> group = NP1 NP2 ... [-d none]
        # Appends a "<name> group = NP1 NP2 ..." line to <card>. We mimic just
        # that side effect; the real tool also rewrites the file in-place with
        # consistent formatting, which we don't need to reproduce.
        card="$1"
        name="$2"
        ptype="$3"
        shift 4  # drop card, name, type, "="
        nps=()
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "-d" ]]; then
                shift 2
                continue
            fi
            nps+=("$1")
            shift
        done
        printf '%s %s = %s\\n' "$name" "$ptype" "${nps[*]}" >> "$card"
        """),
    "prettify_datacard.py": textwrap.dedent("""\
        #!/bin/bash
        # No-op stub; the real tool reformats the card in-place.
        :
        """),
}


def _install_stubs(stub_dir: Path) -> None:
    """Write the bash stubs to ``stub_dir`` and make them executable."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _STUBS.items():
        path = stub_dir / name
        path.write_text(content)
        path.chmod(0o755)


def _run_prepare(workdir: Path) -> subprocess.CompletedProcess[str]:
    """Run prepare_snapshot_inference.sh in ``workdir`` against the stubs."""
    stub_dir = workdir / "stubs"
    _install_stubs(stub_dir)
    env = os.environ.copy()
    env["PATH"] = f"{stub_dir}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(PREPARE_SCRIPT)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Provide a temp work directory pre-populated with the fixture per-bin cards."""
    for name in PER_BIN_NAMES:
        shutil.copy(FIXTURE_DIR / f"{name}.txt", tmp_path / f"{name}.txt")
    return tmp_path


def test_prepare_snapshot_runs_clean(workdir: Path) -> None:
    """Script exits 0 against stubbed binaries and our fixture per-bin cards."""
    result = _run_prepare(workdir)
    assert result.returncode == 0, (
        f"prepare_snapshot_inference.sh exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_prepare_snapshot_creates_nomask_cards(workdir: Path) -> None:
    """All five `*_nomasks.txt` cards land in the work directory."""
    _run_prepare(workdir)
    for name in EXPECTED_NOMASK_CARDS:
        assert (workdir / name).exists(), f"missing output card: {name}"


def test_prepare_snapshot_creates_inject_json(workdir: Path) -> None:
    """The post-bfit snapshot extract lands as inject.json."""
    _run_prepare(workdir)
    inject = workdir / "inject.json"
    assert inject.exists(), "missing inject.json"
    # The real extract writes the full NP set; our stub writes "{}" — verifying
    # the file is non-empty is enough to assert the step ran.
    assert inject.read_text().strip() != ""


@pytest.mark.parametrize("card_name", EXPECTED_NOMASK_CARDS)
def test_prepare_snapshot_adds_signal_norm_groups(workdir: Path, card_name: str) -> None:
    """Each nomasks card receives both `signal_norm_xsbr` and `signal_norm_xs` groups.

    These two group lines are how downstream DHI tasks freeze the theory
    (THU_HH/pdf_Higgs_*/QCDscale_*/BR_hbb) uncertainties as a bundle. If a
    future edit to ``prepare_snapshot_inference.sh`` drops one of the
    ``add_parameter.py`` calls — or renames the group — this test catches
    it instead of letting it silently disappear from the cards.
    """
    _run_prepare(workdir)
    content = (workdir / card_name).read_text()
    assert "signal_norm_xsbr group" in content, f"{card_name}: signal_norm_xsbr group line missing"
    assert "signal_norm_xs group" in content, f"{card_name}: signal_norm_xs group line missing"
