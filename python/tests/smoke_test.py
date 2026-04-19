"""Smoke test for the numnom Python bindings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import numnom

REPO_ROOT = Path(__file__).resolve().parents[2]


def check_model(model: numnom.Model, label: str) -> None:
    print(f"\n[{label}] {model!r}")

    # Basic types
    assert isinstance(model.name, str)
    assert isinstance(model.num_row, int)
    assert isinstance(model.num_col, int)
    assert isinstance(model.obj_sense_minimize, bool)
    assert isinstance(model.obj_offset, float)

    # Numpy arrays — dtypes + shapes
    assert model.col_cost.dtype == np.float64
    assert model.col_lower.dtype == np.float64
    assert model.col_upper.dtype == np.float64
    assert model.row_lower.dtype == np.float64
    assert model.row_upper.dtype == np.float64
    assert model.col_integrality.dtype == np.uint8

    assert model.col_cost.shape == (model.num_col,)
    assert model.col_lower.shape == (model.num_col,)
    assert model.col_upper.shape == (model.num_col,)
    assert model.row_lower.shape == (model.num_row,)
    assert model.row_upper.shape == (model.num_row,)
    assert model.col_integrality.shape == (model.num_col,)

    # Names
    assert isinstance(model.col_names, list)
    assert isinstance(model.row_names, list)
    assert len(model.col_names) == model.num_col
    assert len(model.row_names) == model.num_row

    # Sparse matrix
    a = model.a_matrix
    assert a.start.dtype == np.uint32
    assert a.index.dtype == np.uint32
    assert a.value.dtype == np.float64
    assert a.start.shape == (model.num_col + 1,)
    nnz = int(a.start[-1])
    assert a.index.shape == (nnz,)
    assert a.value.shape == (nnz,)
    print(f"    rows={model.num_row}  cols={model.num_col}  nnz={nnz}")

    # Integrality codes are valid
    valid_codes = {
        numnom.CONTINUOUS,
        numnom.INTEGER,
        numnom.SEMI_CONTINUOUS,
        numnom.SEMI_INTEGER,
    }
    assert set(np.unique(model.col_integrality)).issubset(valid_codes)


def test_parse_file(path: Path) -> numnom.Model:
    model = numnom.parse_file(str(path))
    check_model(model, f"parse_file({path.name})")
    return model


def test_parse_str(path: Path) -> numnom.Model:
    text = path.read_text()
    model = numnom.parse_str(text)
    check_model(model, f"parse_str({path.name})")
    return model


def test_round_trip(path: Path) -> None:
    m1 = numnom.parse_file(str(path))
    text = numnom.write_str(m1)
    m2 = numnom.parse_str(text)

    assert m1.num_row == m2.num_row
    assert m1.num_col == m2.num_col
    np.testing.assert_array_equal(m1.col_cost, m2.col_cost)
    np.testing.assert_array_equal(m1.col_lower, m2.col_lower)
    np.testing.assert_array_equal(m1.col_upper, m2.col_upper)
    np.testing.assert_array_equal(m1.row_lower, m2.row_lower)
    np.testing.assert_array_equal(m1.row_upper, m2.row_upper)
    np.testing.assert_array_equal(m1.a_matrix.start, m2.a_matrix.start)
    np.testing.assert_array_equal(m1.a_matrix.index, m2.a_matrix.index)
    np.testing.assert_array_equal(m1.a_matrix.value, m2.a_matrix.value)
    print(f"    [round-trip {path.name}] OK")


def test_to_scipy(model: numnom.Model) -> None:
    A = numnom.to_scipy(model)
    assert A.shape == (model.num_row, model.num_col)
    assert A.nnz == model.a_matrix.value.size
    print(f"    to_scipy: csc_matrix shape={A.shape} nnz={A.nnz}")


def test_zero_copy(model: numnom.Model) -> None:
    """Verify the big numeric arrays are not copied when crossing into Python.

    The numpy crate hands ownership of each ``Vec<T>`` to a Python wrapper
    object that becomes the array's ``base``. The numpy array itself is a
    *view* over the Rust-allocated buffer, so:

      * ``OWNDATA`` is False (numpy didn't malloc it),
      * ``base`` is not None (the wrapper keeps the Vec alive),
      * the data pointer is stable across getter accesses (no per-call copy),
      * mutations through one reference are visible through another.
    """
    arrays = {
        "col_cost": model.col_cost,
        "col_lower": model.col_lower,
        "col_upper": model.col_upper,
        "row_lower": model.row_lower,
        "row_upper": model.row_upper,
        "a_matrix.start": model.a_matrix.start,
        "a_matrix.index": model.a_matrix.index,
        "a_matrix.value": model.a_matrix.value,
    }

    for name, arr in arrays.items():
        assert not arr.flags["OWNDATA"], (
            f"{name}: OWNDATA is True — numpy allocated its own buffer "
            f"(data was copied from Rust)"
        )
        assert arr.base is not None, f"{name}: base is None (no Rust owner)"

    # Stable data pointer across getter accesses (proves no per-call copy)
    a1 = model.col_cost
    a2 = model.col_cost
    p1 = a1.__array_interface__["data"][0]
    p2 = a2.__array_interface__["data"][0]
    assert p1 == p2, f"col_cost data pointer changed: {p1:#x} -> {p2:#x}"

    # Mutation through one reference is visible through another (shared storage)
    if a1.size > 0 and a1.flags["WRITEABLE"]:
        original = float(a1[0])
        sentinel = original + 1.5
        a1[0] = sentinel
        try:
            assert a2[0] == sentinel, "a1 and a2 do not share storage"
        finally:
            a1[0] = original

    print(
        f"    zero-copy: OWNDATA=False, base set, stable ptr {p1:#x} across accesses"
    )


def main() -> int:
    test_files = [
        REPO_ROOT / "temp_mps" / "amaze.mps",
        REPO_ROOT / "temp_mps" / "flugpl.mps",
        REPO_ROOT / "temp_mps" / "gt2.mps",
    ]
    for path in test_files:
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        m = test_parse_file(path)
        test_parse_str(path)
        test_round_trip(path)
        test_to_scipy(m)
        test_zero_copy(m)

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
