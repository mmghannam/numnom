# numnom

[![PyPI](https://img.shields.io/pypi/v/numnom.svg)](https://pypi.org/project/numnom/)

A fast MPS file parser for Python, powered by Rust.

- **Fast**: 745 MB/s throughput on raw MPS text (4.2× faster than SCIP on MIPLIB 2017)
- **Zero-copy numpy arrays**: model data lands directly in numpy with no extra copies
- **Compressed files**: reads `.mps.gz` with SIMD-accelerated decompression
- **Well-tested**: 0 failures across all 1065 MIPLIB 2017 instances

## Install

```bash
pip install numnom

# Optional: scipy.sparse integration
pip install "numnom[scipy]"
```

## Usage

```python
import numnom

model = numnom.parse_file("problem.mps.gz")

print(model.name, model.num_row, "rows,", model.num_col, "cols")
print("Nonzeros:", model.a_matrix.value.size)

# Numpy arrays — zero-copy from Rust
model.col_cost      # np.ndarray[float64], shape (num_col,)
model.col_lower     # np.ndarray[float64]
model.col_upper     # np.ndarray[float64]
model.row_lower     # np.ndarray[float64]
model.row_upper     # np.ndarray[float64]
model.col_integrality  # np.ndarray[uint8] — see numnom.{CONTINUOUS,INTEGER,...}

# CSC sparse matrix
A = model.a_matrix
A.start    # np.ndarray[uint32], shape (num_col + 1,)
A.index    # np.ndarray[uint32], shape (nnz,)
A.value    # np.ndarray[float64], shape (nnz,)

# Names
model.col_names    # list[str]
model.row_names    # list[str]
```

### scipy.sparse integration

```python
import numnom

model = numnom.parse_file("problem.mps")
A = numnom.to_scipy(model)   # scipy.sparse.csc_matrix, shape (num_row, num_col)
```

### Parse from string

```python
mps_text = open("problem.mps").read()
model = numnom.parse_str(mps_text)
```

### Write MPS

```python
numnom.write_file(model, "out.mps")
mps_text = numnom.write_str(model)
```

## Variable type codes

`model.col_integrality` is a `uint8` array; values map to:

| Code | Constant                  | Meaning           |
|-----:|---------------------------|-------------------|
| 0    | `numnom.CONTINUOUS`       | Continuous        |
| 1    | `numnom.INTEGER`          | Integer           |
| 2    | `numnom.SEMI_CONTINUOUS`  | Semi-continuous   |
| 3    | `numnom.SEMI_INTEGER`     | Semi-integer      |

## License

Apache-2.0
