# numnom

A Mathematical Programming files parser written in Rust. Currently only supports MPS format but will include LP later.  

## Installation

```bash
# Basic installation (parsing and JSON output)
cargo build

# With SCIP validation support
cargo build --features validate
```

## Usage

```bash
# Parse and display file
numnom problem.mps

# Export as JSON5 format
numnom problem.mps --json

# Validate parsing against SCIP solver (requires validate feature)
numnom problem.mps --validate

# Show help
numnom --help
```

## JSON5 Output

The `--json` flag outputs parsed models in JSON5 format with mathematical clarity:

```json5
{
  "name": "example",
  "variables": [
    {
      "name": "x1",
      "obj_coeff": -5.0,
      "type": "integer",
      "lb": 0,           // Lower bound
      "ub": Infinity     // Unbounded above (native JSON5 infinity)
    }
  ],
  "constraints": [
    {
      "name": "c1",
      "coefficients": [{"var_name": "x1", "coeff": 1.0}],
      "lhs": -Infinity,  // Unbounded below
      "rhs": 10          // Upper bound
    }
  ]
}
```

## Format-Agnostic Model Library

The parsed models use a general MIP (Mixed Integer Programming) format designed to work with multiple optimization formats:

```rust
use numnom::{MipModel, Variable, Constraint};

// Models can be serialized/deserialized with serde
let model: MipModel = serde_json5::from_str(&json_string)?;
```

## Validation

When built with the `validate` feature, numnom can validate parsing accuracy against the SCIP solver (currently MPS only):

- Variable counts, types, bounds, and objective coefficients
- Constraint counts, sides, and coefficient matrix
- Comprehensive reporting of any mismatches

## Dependencies

- **Core**: `nom` (parsing), `serde` + `serde_json5` (serialization)
- **Optional**: `russcip` (validation, requires `validate` feature)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.