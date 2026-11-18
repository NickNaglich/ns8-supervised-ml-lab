"""Check a saved TaskA ambiguity witness; no generation, extraction or fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

VIEWS = ("TLF", "TRF", "BLF", "BRF", "TLB", "BLB", "BRB", "TRB")
MAX_BYTES = 128 * 1024


def convention_control():
    # Source-convention control, not a reconstructed LAB generator or theorem.
    n, slope, phase, r, c = 4, 1, 1, 0, 1
    return dict(project_operator="1+((r+slope*c)%N)", project_k_role="kernel slope",
                lab_operator="1+((r*c+(phase-1))%N)", lab_k_role="additive phase",
                indexing="r,c zero-based; displayed values one-based",
                cell=dict(N=n, slope=slope, phase=phase, r=r, c=c),
                project_value=1+((r+slope*c) % n), lab_value=1+((r*c+phase-1) % n),
                transfer="different operators; no direct phase/recovery/QFT theorem transfer")


def check_witness(document):
    if document.get("schema") != "ns8_task_a_saved_alias_v1":
        raise ValueError("unsupported witness schema")
    if document.get("operator", {}).get("id") != "ns8lab_additive_slope_v1":
        raise ValueError("explicit additive-slope input convention required")
    names, records = document["feature_names"], document["records"]
    if len(names) != 48 or len(set(names)) != 48 or any(not n.startswith("feat_") for n in names):
        raise ValueError("48 unique ordered feature names required")
    if len(records) != 2:
        raise ValueError("exactly two saved records required; this is not a census")
    checked = []
    for record in records:
        meta, raw, features = record["metadata"], record["raw"], record["features_hex"]
        n, row = meta["N"], meta["row"]
        if type(n) is not int or not 1 <= n <= 64 or type(row) is not int or row < 0:
            raise ValueError("bounded square shape and nonnegative saved row ID required")
        if meta["target"] not in VIEWS or meta["target"] != meta["view"]:
            raise ValueError("preserve original TaskA view/target; no relabeling")
        values = raw["values"]
        if raw["row"] != row or raw["shape"] != [n, n] or len(values) != n*n:
            raise ValueError("raw shape/row mismatch")
        if any(type(v) is not int or not 1 <= v <= n for v in values):
            raise ValueError("raw values must be plain integers1..N")
        packed = struct.pack("<" + "q"*len(values), *values)
        if hashlib.sha256(packed).hexdigest() != raw["rowmajor_sha256"]:
            raise ValueError("saved raw logical-value hash mismatch")
        if len(features) != 48:
            raise ValueError("48 exact hex inputs required")
        for v in features:
            if not isinstance(v, str):
                raise ValueError("feature inputs must be canonical finite float.hex strings")
            x = float.fromhex(v)
            if not math.isfinite(x) or x.hex() != v:
                raise ValueError("feature inputs must be canonical finite float.hex strings")
        checked.append((n, tuple(values), tuple(features), meta["target"], row))
    a, b = checked
    if a[4] == b[4]:
        raise ValueError("distinct saved row IDs required")
    raw_equal = a[:2] == b[:2]
    feature_equal = a[2] == b[2]
    labels = list(dict.fromkeys([a[3], b[3]]))
    conflict = len(labels) == 2
    ambiguity = raw_equal and feature_equal and conflict
    return dict(status="AMBIGUOUS_SAVED_INPUT" if ambiguity else "NO_JOINT_WITNESS",
                input_convention=convention_control(),
                saved_rows=[a[4], b[4]], original_labels=labels, raw_equal=raw_equal,
                all48_hex_equal=feature_equal, conflicting_labels=conflict,
                single_origin_claim_supported=False if ambiguity else None,
                deterministic_maximum_on_this_pair=dict(correct=1, rows=2) if feature_equal and conflict else None,
                scope="This supplied saved pair only. Raw is an independent control, never model X. No population/held-out score, origin authentication, complete label enumeration or external benefit claim.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True, type=Path)
    args = parser.parse_args(argv)
    with args.witness.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("witness exceeds128KiB")
    result = check_witness(json.loads(data))
    result["supplied_witness_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
