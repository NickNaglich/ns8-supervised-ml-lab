"""Small hand-solved contract controls; no feature extraction/model evidence."""
import copy
import hashlib
import json
import struct

import pytest
from experiments.observable_target_gate import check_witness, convention_control, main


def example():
    raw = dict(row=0, shape=[2, 2], values=[1, 2, 2, 1],
               rowmajor_sha256=hashlib.sha256(struct.pack("<4q", 1, 2, 2, 1)).hexdigest())
    a = dict(metadata=dict(row=0, N=2, k=1, view="TLF", target="TLF"),
             raw=raw, features_hex=[0.0.hex()] * 48)
    b = copy.deepcopy(a); b["metadata"].update(row=1, view="TLB", target="TLB"); b["raw"]["row"] = 1
    return dict(schema="ns8_task_a_saved_alias_v1", operator={"id": "ns8lab_additive_slope_v1"},
                feature_names=[f"feat_{i}" for i in range(48)], records=[a, b])


def test_source_convention_cell_matches_native_generator():
    from ns8lab.grids import generate_grid
    control = convention_control()
    assert generate_grid(4, 1, "TLF")[0, 1] == control["project_value"] == 2
    assert control["lab_value"] == 1 and control["project_value"] != control["lab_value"]


def test_phase_contract_cannot_be_passed_off_as_slope():
    source = example(); source["operator"]["id"] = "lab_product_residue_phase"
    with pytest.raises(ValueError, match="input convention"): check_witness(source)


def test_hand_solved_transpose_alias_retains_labels():
    source = example(); saved = copy.deepcopy(source)
    result = check_witness(source)
    assert result["status"] == "AMBIGUOUS_SAVED_INPUT"
    assert result["raw_equal"] and result["all48_hex_equal"]
    assert result["original_labels"] == ["TLF", "TLB"]
    assert result["deterministic_maximum_on_this_pair"] == dict(correct=1, rows=2)
    assert source == saved


def test_signed_zero_is_not_hard_equivalence():
    source = example(); source["records"][1]["features_hex"][0] = (-0.0).hex()
    result = check_witness(source)
    assert result["raw_equal"] and not result["all48_hex_equal"]
    assert result["status"] == "NO_JOINT_WITNESS" and result["deterministic_maximum_on_this_pair"] is None


def test_equal_inputs_equal_labels_do_not_invent_conflict():
    source = example(); source["records"][1]["metadata"].update(view="TLF", target="TLF")
    result = check_witness(source)
    assert result["status"] == "NO_JOINT_WITNESS" and not result["conflicting_labels"]


def test_different_raw_arrays_are_not_called_identical():
    source = example(); raw = source["records"][1]["raw"]
    raw["values"] = [2, 1, 1, 2]
    raw["rowmajor_sha256"] = hashlib.sha256(struct.pack("<4q", *raw["values"])).hexdigest()
    result = check_witness(source)
    assert not result["raw_equal"] and result["status"] == "NO_JOINT_WITNESS"


@pytest.mark.parametrize("bad", ["hash", "shape", "label", "nan", "bool"])
def test_malformed_contract_rejected(bad):
    source = example(); b = source["records"][1]
    if bad == "hash": b["raw"]["rowmajor_sha256"] = "bad"
    if bad == "shape": b["raw"]["shape"] = [1, 4]
    if bad == "label": b["metadata"]["target"] = "BLF"
    if bad == "nan": b["features_hex"][0] = "nan"
    if bad == "bool": b["raw"]["values"][0] = True
    with pytest.raises(ValueError): check_witness(source)


def test_cli_is_read_only_and_bounded(tmp_path, capsys):
    source = tmp_path / "witness.json"; data = json.dumps(example()).encode(); source.write_bytes(data)
    main(["--witness", str(source)])
    result = json.loads(capsys.readouterr().out)
    assert result["supplied_witness_sha256"] == hashlib.sha256(data).hexdigest()
    assert source.read_bytes() == data and list(tmp_path.iterdir()) == [source]
    source.write_bytes(b" " * (128*1024+1))
    with pytest.raises(ValueError, match="exceeds"): main(["--witness", str(source)])
