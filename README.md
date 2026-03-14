# numnom

A fast MPS file parser written in Rust. 2-5x faster than SCIP's parser, tested on all 1329 MIPLIB instances.

## Features

- **Fast**: 745 MB/s throughput on raw MPS text, 2-5x faster than SCIP
- **Zero-copy parsing**: borrows from input buffer — minimal allocations during parsing
- **Direct CSC output**: builds sparse column-wise matrix during parsing with no intermediate storage
- **Compressed files**: reads `.mps.gz` with SIMD-accelerated decompression (zlib-rs)
- **Correct**: 0 failures across all 1329 MIPLIB benchmark instances

## Installation

```bash
cargo install --path .
```

## Usage

### CLI

```bash
# Parse and display summary
numnom problem.mps

# Parse gzipped file
numnom problem.mps.gz

# Show per-section timing breakdown
numnom problem.mps.gz --timings
```

Example output:
```
  neos-631709 (min)

  Rows              46.5K    Cols            45.1K
  Nonzeros         225.1K    Density       0.0107%
  Variables    45.1K binary

  Parsed in 19.0ms
```

### Library

```rust
use numnom::{parse_mps_file, parse_mps_str, Model, VarType};

// Parse from file (supports .mps and .mps.gz)
let model = parse_mps_file("problem.mps.gz").unwrap();

// Parse from string
let model = parse_mps_str(mps_content).unwrap();

// Access model data
println!("Rows: {}, Cols: {}", model.num_row, model.num_col);
println!("Nonzeros: {}", model.a_matrix.value.len());

// Sparse matrix in CSC format
let col = 0;
let start = model.a_matrix.start[col] as usize;
let end = model.a_matrix.start[col + 1] as usize;
for i in start..end {
    let row = model.a_matrix.index[i];
    let val = model.a_matrix.value[i];
    println!("  A[{}, {}] = {}", row, col, val);
}

// Variable info
for (i, name) in model.col_names.iter().enumerate() {
    let lb = model.col_lower[i];
    let ub = model.col_upper[i];
    let cost = model.col_cost[i];
    let vtype = model.col_integrality[i];
    println!("{}: [{}, {}] cost={} type={:?}", name, lb, ub, cost, vtype);
}
```

### Model structure

```rust
pub struct Model {
    pub name: String,
    pub num_row: i32,
    pub num_col: i32,
    pub obj_sense_minimize: bool,
    pub obj_offset: f64,
    pub objective_name: String,
    pub col_cost: Vec<f64>,        // Objective coefficients
    pub col_lower: Vec<f64>,       // Variable lower bounds
    pub col_upper: Vec<f64>,       // Variable upper bounds
    pub row_lower: Vec<f64>,       // Constraint lower bounds
    pub row_upper: Vec<f64>,       // Constraint upper bounds
    pub a_matrix: SparseMatrix,    // Constraint matrix (CSC)
    pub col_names: Vec<String>,
    pub row_names: Vec<String>,
    pub col_integrality: Vec<VarType>,
}

pub struct SparseMatrix {
    pub start: Vec<i32>,   // Column pointers
    pub index: Vec<i32>,   // Row indices
    pub value: Vec<f64>,   // Nonzero values
}
```

## MPS sections supported

| Section | Status |
|---------|--------|
| NAME | supported |
| OBJSENSE | supported |
| ROWS | supported |
| COLUMNS | supported |
| RHS | supported |
| BOUNDS (LO/UP/FX/FR/MI/PL/BV/LI/UI/SC/SI) | supported |
| RANGES | supported |
| INDICATORS, SOS, QUADOBJ, etc. | skipped |

## Benchmarks

Tested on Apple M-series, single-threaded:

| Instance | Rows | Cols | Nonzeros | numnom | SCIP | Speedup |
|----------|------|------|----------|--------|------|---------|
| neos-631709 | 46K | 45K | 225K | 19ms | 91ms | 4.9x |
| neos-5251015 | 487K | 137K | 1.5M | 285ms | 753ms | 2.6x |
| s82 | 88K | 1.7M | 7.0M | 1.4s | 2.9s | 2.2x |
| ivu59 | 3.4K | 2.6M | 36.2M | 2.3s | 6.6s | 2.8x |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
