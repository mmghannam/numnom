# numnom

[![crates.io](https://img.shields.io/crates/v/numnom.svg)](https://crates.io/crates/numnom)
[![docs.rs](https://docs.rs/numnom/badge.svg)](https://docs.rs/numnom)
[![license](https://img.shields.io/crates/l/numnom.svg)](LICENSE)

A fast MPS file parser written in Rust.

## Features

- **Fast**: 745 MB/s throughput on raw MPS text
- **Zero-copy parsing**: borrows from input buffer — minimal allocations during parsing
- **Direct CSC output**: builds sparse column-wise matrix during parsing with no intermediate storage
- **Compressed files**: reads `.mps.gz` with SIMD-accelerated decompression (zlib-rs)
- **Well-tested**: 0 failures across all 1065 MIPLIB 2017 instances

## Installation

```bash
# As a library
cargo add numnom

# As a CLI tool
cargo install numnom
```

## Usage

### CLI

```bash
# Parse and display summary
numnom problem.mps

# Parse gzipped file
numnom problem.mps.gz

# Quiet mode (validate only, no output)
numnom problem.mps.gz -q
```

Example output:
```
  Loading s82.mps.gz (decompressing)... done in 1.35s

  s82 (min)

  Statistics
  ----------

  Rows              87.9K    Cols             1.7M
  Nonzeros           7.0M    Density       0.0047%
  Continuous           54
  Binary             1.7M

  Timing
  ------

  Decompress   49.5MB -> 884.8MB in 299.8ms
  Parse        884.8MB in 1.05s
    Rows            6.9ms  (1%)
    Columns       580.9ms  (56%)
    RHS             0.1ms  (0%)
    Bounds        398.7ms  (38%)
    Finalize       43.0ms  (4%)
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
    pub num_row: u32,
    pub num_col: u32,
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
    pub start: Vec<u32>,   // Column pointers
    pub index: Vec<u32>,   // Row indices
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

Tested on Apple M-series, single-threaded. All times include gzip decompression.

**MIPLIB 2017 benchmark set (240 instances):**

| Metric | numnom | SCIP |
|--------|--------|------|
| Shifted geomean (s=10ms) | 19ms | 82ms |
| Median | 10ms | 38ms |
| P90 | 118ms | 243ms |
| Max | 930ms | 2.9s |
| **Speedup (shifted geomean)** | | **4.2x** |

Correctness validated against [SCIP](https://www.scipopt.org/) via [russcip](https://github.com/scipopt/russcip) on all 1065 MIPLIB 2017 instances (0 failures).

Per-instance results: [benchmarks.csv](benchmarks.csv)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
