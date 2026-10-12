"""Tests for eventlist_manifest.py."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from HH4b.eventlist_manifest import (
    SCHEMA_VERSION,
    build_eventlist_manifest,
    manifest_errors_for_root,
    manifest_validation_errors,
    parse_year_from_eventlist_filename,
    read_manifest,
    write_eventlist_manifest,
)


def _make_args(**overrides) -> Namespace:
    """Minimal args Namespace mimicking PostProcess args."""
    defaults = {
        "mass": "H2PNetMass",
        "templates_tag": "test_tag",
        "tag": "ntuple_v1",
        "data_dir": "/tmp/data",
        "years": ["2022", "2022EE"],
        "training_years": None,
        "bdt_model": "model_v1",
        "bdt_config": "config_v1",
        "txbb": "glopart-v2",
        "txbb_wps": [0.97, 0.90],
        "bdt_wps": [0.98, 0.95],
        "vbf_txbb_wp": 0.95,
        "vbf_bdt_wp": 0.90,
        "vbf": True,
        "vbf_priority": False,
        "pt_first": 300.0,
        "pt_second": 250.0,
        "sig_keys": ["hh4b", "vbfhh4b"],
        "event_list_data_only": False,
        "event_list_dir": "/tmp/evlists",
        "weight_ttbar_bdt": 1.0,
        "correct_vbf_bdt_shape": False,
        "bdt_disc": False,
        "rerun_inference": False,
        "scale_smear": True,
        "dummy_txbb_sfs": False,
        "blind": True,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestBuildEventlistManifest:
    def test_has_required_keys(self):
        manifest = build_eventlist_manifest(_make_args(), [110.0, 155.0])
        for key in [
            "schema_version",
            "mass_window_fom",
            "mass_variable",
            "years",
            "bdt_model",
            "bdt_config",
            "txbb",
            "output_files",
        ]:
            assert key in manifest

    def test_mass_window_serialization(self):
        manifest = build_eventlist_manifest(_make_args(), np.array([110, 155]))
        assert manifest["mass_window_fom"] == [110.0, 155.0]

    def test_output_files_match_years(self):
        manifest = build_eventlist_manifest(_make_args(years=["2023", "2023BPix"]), [110, 140])
        assert manifest["output_files"] == [
            "eventlist_boostedHH4b_2023.root",
            "eventlist_boostedHH4b_2023BPix.root",
        ]

    def test_schema_version(self):
        manifest = build_eventlist_manifest(_make_args(), [110, 140])
        assert manifest["schema_version"] == SCHEMA_VERSION


class TestWriteAndReadManifest:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "eventlist_manifest.json"
        args = _make_args()
        write_eventlist_manifest(path, args, [110, 155])
        loaded = read_manifest(path)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["years"] == ["2022", "2022EE"]

    def test_merge_accumulates_years(self, tmp_path):
        path = tmp_path / "eventlist_manifest.json"
        args1 = _make_args(years=["2022"])
        write_eventlist_manifest(path, args1, [110, 155])

        args2 = _make_args(years=["2022EE"])
        write_eventlist_manifest(path, args2, [110, 155], merge=True)

        loaded = read_manifest(path)
        assert loaded["years"] == ["2022", "2022EE"]
        assert "eventlist_boostedHH4b_2022.root" in loaded["output_files"]
        assert "eventlist_boostedHH4b_2022EE.root" in loaded["output_files"]

    def test_merge_does_not_duplicate_years(self, tmp_path):
        path = tmp_path / "eventlist_manifest.json"
        args = _make_args(years=["2022"])
        write_eventlist_manifest(path, args, [110, 155])
        write_eventlist_manifest(path, args, [110, 155], merge=True)
        loaded = read_manifest(path)
        assert loaded["years"] == ["2022"]

    def test_no_merge_replaces(self, tmp_path):
        path = tmp_path / "eventlist_manifest.json"
        write_eventlist_manifest(path, _make_args(years=["2022"]), [110, 155])
        write_eventlist_manifest(path, _make_args(years=["2023"]), [110, 155], merge=False)
        loaded = read_manifest(path)
        assert loaded["years"] == ["2023"]

    def test_merge_rejects_different_settings(self, tmp_path):
        path = tmp_path / "eventlist_manifest.json"
        write_eventlist_manifest(path, _make_args(years=["2022"], txbb="glopart-v2"), [110, 155])
        write_eventlist_manifest(
            path, _make_args(years=["2022EE"], txbb="pnet-v12"), [110, 155], merge=True
        )
        loaded = read_manifest(path)
        # Settings differ, so merge should overwrite instead of accumulating
        assert loaded["years"] == ["2022EE"]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "manifest.json"
        write_eventlist_manifest(path, _make_args(), [110, 155])
        assert path.exists()


class TestParseYearFromFilename:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("eventlist_boostedHH4b_2022.root", "2022"),
            ("eventlist_boostedHH4b_2022EE.root", "2022EE"),
            ("eventlist_boostedHH4b_2023BPix.root", "2023BPix"),
            ("not_an_eventlist.root", None),
            ("eventlist_boostedHH4b_.root", None),
        ],
    )
    def test_parsing(self, name, expected):
        result = parse_year_from_eventlist_filename(Path(name))
        assert result == expected


class TestManifestErrorsForRoot:
    def _valid_manifest(self, year="2022"):
        return {
            "schema_version": SCHEMA_VERSION,
            "years": [year],
            "templates_tag": "test",
            "ntuple_tag": "v1",
            "bdt_model": "m",
            "bdt_config": "c",
            "txbb": "glopart-v2",
            "mass_variable": "H2PNetMass",
            "mass_window_fom": [110.0, 155.0],
            "output_files": [f"eventlist_boostedHH4b_{year}.root"],
        }

    def test_no_errors_for_valid_manifest(self):
        errors = manifest_errors_for_root(
            Path("eventlist_boostedHH4b_2022.root"), self._valid_manifest("2022")
        )
        assert errors == []

    def test_wrong_schema_version(self):
        m = self._valid_manifest()
        m["schema_version"] = 999
        errors = manifest_errors_for_root(Path("eventlist_boostedHH4b_2022.root"), m)
        assert any("schema_version" in e for e in errors)

    def test_missing_required_key(self):
        m = self._valid_manifest()
        del m["bdt_model"]
        errors = manifest_errors_for_root(Path("eventlist_boostedHH4b_2022.root"), m)
        assert any("bdt_model" in e for e in errors)

    def test_year_mismatch(self):
        m = self._valid_manifest("2023")
        errors = manifest_errors_for_root(Path("eventlist_boostedHH4b_2022.root"), m)
        assert any("2022" in e for e in errors)


class TestManifestValidationErrors:
    def test_missing_manifest_not_required(self, tmp_path):
        root = tmp_path / "eventlist_boostedHH4b_2022.root"
        root.touch()
        errors = manifest_validation_errors(root, None, require_manifest=False)
        assert errors == []

    def test_missing_manifest_required(self, tmp_path):
        root = tmp_path / "eventlist_boostedHH4b_2022.root"
        root.touch()
        errors = manifest_validation_errors(root, None, require_manifest=True)
        assert len(errors) == 1
        assert "not found" in errors[0].lower() or "Manifest" in errors[0]

    def test_valid_manifest_with_path(self, tmp_path):
        root = tmp_path / "eventlist_boostedHH4b_2022.root"
        root.touch()
        manifest_path = tmp_path / "eventlist_manifest.json"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "years": ["2022"],
            "templates_tag": "t",
            "ntuple_tag": "v",
            "bdt_model": "m",
            "bdt_config": "c",
            "txbb": "glopart-v2",
            "mass_variable": "H2PNetMass",
            "mass_window_fom": [110, 155],
            "output_files": ["eventlist_boostedHH4b_2022.root"],
        }
        manifest_path.write_text(json.dumps(manifest))
        errors = manifest_validation_errors(root, manifest_path, require_manifest=True)
        assert errors == []
