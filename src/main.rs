// Simple MPS parser - adding COLUMNS section
use std::fs;
use std::collections::HashMap;
use russcip::prelude::*;

#[derive(Debug)]
struct Row {
    row_type: char,
    name: String,
}

#[derive(Debug)]
struct Column {
    name: String,
    coefficients: HashMap<String, f64>,
    is_integer: bool,
}

#[derive(Debug)]
struct Bound {
    bound_type: String,
    column_name: String,
    value: Option<f64>,
}

#[derive(Debug)]
struct Indicator {
    row_name: String,
    indicator_variable: String,
    indicator_value: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mps-file>", args[0]);
        return;
    }

    let filepath = &args[1];
    let content = fs::read_to_string(filepath)
        .expect("Failed to read file");

    println!("=== Our Parser ===");

    // Parse NAME section
    if let Some(name) = parse_name_section(&content) {
        println!("Problem name: {}", name);
    }

    // Parse ROWS section
    if let Some(rows) = parse_rows_section(&content) {
        println!("\nRows found: {}", rows.len());
        for row in &rows {
            println!("  {} {}", row.row_type, row.name);
        }
    }

    // Parse COLUMNS section
    if let Some(columns) = parse_columns_section(&content) {
        println!("\nColumns found: {}", columns.len());
        for col in &columns {
            let var_type = if col.is_integer { "INTEGER" } else { "CONTINUOUS" };
            println!("  Variable: {} [{}]", col.name, var_type);
            for (row, coeff) in &col.coefficients {
                println!("    {} = {}", row, coeff);
            }
        }
    }

    // Parse RHS section
    if let Some(rhs) = parse_rhs_section(&content) {
        println!("\nRHS values:");
        for (row_name, value) in &rhs {
            println!("  {} = {}", row_name, value);
        }
    }

    // Parse BOUNDS section
    if let Some(bounds) = parse_bounds_section(&content) {
        println!("\nBounds:");
        for bound in &bounds {
            match &bound.value {
                Some(val) => println!("  {} {} {}", bound.bound_type, bound.column_name, val),
                None => println!("  {} {}", bound.bound_type, bound.column_name),
            }
        }
    }

    // Parse INDICATORS section
    if let Some(indicators) = parse_indicators_section(&content) {
        println!("\nIndicators:");
        for indicator in &indicators {
            println!("  IF {} = {} THEN {}", indicator.indicator_variable, indicator.indicator_value, indicator.row_name);
        }
    }

    println!("\n=== SCIP Parser ===");
    let scip_model = Model::new()
        .hide_output()
        .include_default_plugins()
        .read_prob(filepath)
        .expect("SCIP failed to read MPS file");

    println!("Constraints: {}", scip_model.n_conss());
    println!("Variables: {}", scip_model.n_vars());

    // Detailed comparison
    println!("\n=== Verification ===");

    // Detailed validation of variable values
    let scip_vars = scip_model.vars();
    let our_columns = parse_columns_section(&content).unwrap_or_default();
    let our_bounds = parse_bounds_section(&content).unwrap_or_default();
    let _our_rhs = parse_rhs_section(&content).unwrap_or_default();
    let our_rows = parse_rows_section(&content).unwrap_or_default();

    // Find objective row name
    let obj_row_name = our_rows.iter()
        .find(|row| row.row_type == 'N')
        .map(|row| row.name.clone())
        .unwrap_or_else(|| "obj".to_string());

    // Create maps for quick lookup
    let mut our_var_map: HashMap<String, &Column> = HashMap::new();
    for col in &our_columns {
        our_var_map.insert(col.name.clone(), col);
    }

    let mut our_bounds_map: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();
    for bound in &our_bounds {
        let (mut lb, mut ub) = our_bounds_map.get(&bound.column_name).unwrap_or(&(None, None)).clone();
        match bound.bound_type.as_str() {
            "LO" => lb = bound.value,
            "UP" => ub = bound.value,
            "FX" => { lb = bound.value; ub = bound.value; },
            "FR" => { lb = None; ub = None; },
            _ => {}
        }
        our_bounds_map.insert(bound.column_name.clone(), (lb, ub));
    }

    let mut var_mismatches = 0;
    let mut bound_mismatches = 0;
    let mut obj_mismatches = 0;

    println!("\n=== Variable Value Validation ===");

    for var in &scip_vars {
        let var_name = var.name();

        // Check if we parsed this variable
        if let Some(our_var) = our_var_map.get(&var_name.to_string()) {
            // Check objective coefficient
            let our_obj = our_var.coefficients.get(&obj_row_name).unwrap_or(&0.0);
            let scip_obj = var.obj();

            if (our_obj - scip_obj).abs() > 1e-10 {
                println!("  OBJ MISMATCH {}: {} (ours) vs {} (SCIP)", var_name, our_obj, scip_obj);
                obj_mismatches += 1;
            }

            // Check bounds
            let (our_lb, our_ub) = our_bounds_map.get(&var_name.to_string()).unwrap_or(&(Some(0.0), None));
            let scip_lb = var.lb();
            let scip_ub = var.ub();

            let our_lb_val = our_lb.unwrap_or(0.0);
            let our_ub_val = our_ub.unwrap_or(f64::INFINITY);

            if (our_lb_val - scip_lb).abs() > 1e-10 {
                println!("  LB MISMATCH {}: {} (ours) vs {} (SCIP)", var_name, our_lb_val, scip_lb);
                bound_mismatches += 1;
            }

            // SCIP uses large finite numbers for unbounded variables, we use infinity
            let scip_is_unbounded = scip_ub >= 1e19;
            let our_is_unbounded = our_ub_val == f64::INFINITY;

            if !(scip_is_unbounded && our_is_unbounded) && (our_ub_val - scip_ub).abs() > 1e-10 {
                println!("  UB MISMATCH {}: {} (ours) vs {} (SCIP)", var_name, our_ub_val, scip_ub);
                bound_mismatches += 1;
            }
        } else {
            println!("  MISSING VAR: {} not found in our parser", var_name);
            var_mismatches += 1;
        }
    }

    // Compare counts
    let our_var_count = parse_columns_section(&content).map(|c| c.len()).unwrap_or(0);
    let scip_var_count = scip_model.n_vars();

    let our_row_count = parse_rows_section(&content)
        .map(|r| r.iter().filter(|row| row.row_type != 'N').count())
        .unwrap_or(0);
    let scip_cons_count = scip_model.n_conss();

    let our_indicator_count = parse_indicators_section(&content).map(|i| i.len()).unwrap_or(0);

    println!("\n✓ Variable count: {} (ours) vs {} (SCIP) - {}",
             our_var_count, scip_var_count,
             if our_var_count == scip_var_count { "MATCH" } else { "MISMATCH" });

    println!("✓ Constraint count: {} (ours) vs {} (SCIP) - {}",
             our_row_count, scip_cons_count,
             if our_row_count == scip_cons_count { "MATCH" } else { "MISMATCH" });

    if our_indicator_count > 0 {
        println!("✓ Indicator constraints parsed: {} (SCIP processes these into additional constraints/variables)", our_indicator_count);
        println!("  Note: SCIP expands indicators, so exact counts may differ");
    }

    // Constraint matrix validation
    println!("\n=== Constraint Matrix Validation ===");
    let mut matrix_mismatches = 0;

    // Create constraint name mapping
    let mut constraint_name_map: HashMap<String, usize> = HashMap::new();
    let scip_conss = scip_model.conss();
    for (i, cons) in scip_conss.iter().enumerate() {
        constraint_name_map.insert(cons.name().to_string(), i);
    }

    for var in &scip_vars {
        let var_name = var.name();

        if let Some(our_var) = our_var_map.get(&var_name.to_string()) {
            // Get SCIP's constraint coefficients for this variable
            if let Some(col) = var.col() {
                let scip_rows = col.rows();
                let scip_vals = col.vals();

                // Create map of SCIP constraint -> coefficient
                let mut scip_coeffs: HashMap<String, f64> = HashMap::new();
                for (row, val) in scip_rows.iter().zip(scip_vals.iter()) {
                    scip_coeffs.insert(row.name().to_string(), *val);
                }

                // Compare with our parsed coefficients (excluding objective)
                for (constraint_name, our_coeff) in &our_var.coefficients {
                    if constraint_name == &obj_row_name {
                        continue; // Skip objective, already validated
                    }

                    if let Some(scip_coeff) = scip_coeffs.get(constraint_name) {
                        if (our_coeff - scip_coeff).abs() > 1e-10 {
                            println!("  COEFF MISMATCH {}: constraint {} = {} (ours) vs {} (SCIP)",
                                    var_name, constraint_name, our_coeff, scip_coeff);
                            matrix_mismatches += 1;
                        }
                    } else {
                        println!("  MISSING CONSTRAINT {}: variable {} has coefficient in {} not found in SCIP",
                                var_name, var_name, constraint_name);
                        matrix_mismatches += 1;
                    }
                }

                // Check for SCIP coefficients we might have missed
                for (scip_constraint, scip_coeff) in &scip_coeffs {
                    if !our_var.coefficients.contains_key(scip_constraint) {
                        println!("  MISSING COEFF {}: SCIP has coefficient {} = {} not found in our parser",
                                var_name, scip_constraint, scip_coeff);
                        matrix_mismatches += 1;
                    }
                }
            }
        }
    }

    // Value validation summary
    println!("\n=== Value Validation Summary ===");
    println!("Variables missing: {}", var_mismatches);
    println!("Objective mismatches: {}", obj_mismatches);
    println!("Bound mismatches: {}", bound_mismatches);
    println!("Matrix coefficient mismatches: {}", matrix_mismatches);

    if var_mismatches == 0 && obj_mismatches == 0 && bound_mismatches == 0 && matrix_mismatches == 0 {
        println!("✓ All variable values and constraint coefficients MATCH perfectly!");
    } else {
        println!("⚠ Some mismatches found (see details above)");
    }
}

fn parse_name_section(content: &str) -> Option<String> {
    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }
        if line.trim_start().starts_with("NAME") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() > 1 {
                return Some(parts[1].to_string());
            }
            return Some("".to_string());
        }
    }
    None
}

fn parse_rows_section(content: &str) -> Option<Vec<Row>> {
    let mut in_rows_section = false;
    let mut rows = Vec::new();

    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }

        let trimmed = line.trim();

        if trimmed == "ROWS" {
            in_rows_section = true;
            continue;
        }

        if in_rows_section && (trimmed == "COLUMNS" || trimmed == "ENDATA") {
            break;
        }

        if in_rows_section && !trimmed.is_empty() {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                let row_type = parts[0].chars().next()?;
                let name = parts[1].to_string();
                rows.push(Row { row_type, name });
            }
        }
    }

    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
}

fn parse_bounds_section(content: &str) -> Option<Vec<Bound>> {
    let mut in_bounds_section = false;
    let mut bounds = Vec::new();

    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }

        let trimmed = line.trim();

        if trimmed == "BOUNDS" {
            in_bounds_section = true;
            continue;
        }

        if in_bounds_section && trimmed == "ENDATA" {
            break;
        }

        if in_bounds_section && !trimmed.is_empty() {
            // Format: type bound_name column_name [value]
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            if parts.len() >= 3 {
                let bound_type = parts[0].to_string();
                let column_name = parts[2].to_string();

                // Some bound types don't have values (FR, MI, PL, BV)
                let value = if parts.len() >= 4 {
                    parts[3].parse::<f64>().ok()
                } else {
                    None
                };

                bounds.push(Bound {
                    bound_type,
                    column_name,
                    value,
                });
            }
        }
    }

    if bounds.is_empty() {
        None
    } else {
        Some(bounds)
    }
}

fn parse_rhs_section(content: &str) -> Option<HashMap<String, f64>> {
    let mut in_rhs_section = false;
    let mut rhs_values = HashMap::new();

    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }

        let trimmed = line.trim();

        if trimmed == "RHS" {
            in_rhs_section = true;
            continue;
        }

        if in_rhs_section && (trimmed == "BOUNDS" || trimmed == "RANGES" || trimmed == "ENDATA") {
            break;
        }

        if in_rhs_section && !trimmed.is_empty() {
            // Format: rhs_name row_name value [row_name value]
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            if parts.len() >= 3 {
                // Skip the RHS vector name (first field)
                let row_name = parts[1];
                if let Ok(value) = parts[2].parse::<f64>() {
                    rhs_values.insert(row_name.to_string(), value);
                }

                // Check for second RHS pair
                if parts.len() >= 5 {
                    let row_name2 = parts[3];
                    if let Ok(value2) = parts[4].parse::<f64>() {
                        rhs_values.insert(row_name2.to_string(), value2);
                    }
                }
            }
        }
    }

    if rhs_values.is_empty() {
        None
    } else {
        Some(rhs_values)
    }
}

fn parse_columns_section(content: &str) -> Option<Vec<Column>> {
    let mut in_columns_section = false;
    let mut in_integer_section = false;
    let mut columns_map: HashMap<String, Column> = HashMap::new();

    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }

        let trimmed = line.trim();

        if trimmed == "COLUMNS" {
            in_columns_section = true;
            continue;
        }

        if in_columns_section && (trimmed == "RHS" || trimmed == "BOUNDS" || trimmed == "ENDATA") {
            break;
        }

        if in_columns_section && !trimmed.is_empty() {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            // Check for MARKER lines
            if parts.len() >= 3 && parts[1].contains("MARKER") {
                if parts.len() >= 3 && parts[2].contains("INTORG") {
                    in_integer_section = true;
                    println!("  [Found INTORG marker - starting integer section]");
                    continue;
                } else if parts.len() >= 3 && parts[2].contains("INTEND") {
                    in_integer_section = false;
                    println!("  [Found INTEND marker - ending integer section]");
                    continue;
                }
            }

            // Parse regular column data
            if parts.len() >= 3 {
                let col_name = parts[0];
                let row_name = parts[1];

                // Try to parse coefficient
                if let Ok(coeff) = parts[2].parse::<f64>() {
                    let column = columns_map.entry(col_name.to_string())
                        .or_insert(Column {
                            name: col_name.to_string(),
                            coefficients: HashMap::new(),
                            is_integer: in_integer_section,
                        });
                    column.coefficients.insert(row_name.to_string(), coeff);

                    // Update integer status if we're in integer section
                    if in_integer_section {
                        column.is_integer = true;
                    }
                }

                // Check for second coefficient pair
                if parts.len() >= 5 {
                    let row_name2 = parts[3];
                    if let Ok(coeff2) = parts[4].parse::<f64>() {
                        let column = columns_map.entry(col_name.to_string())
                            .or_insert(Column {
                                name: col_name.to_string(),
                                coefficients: HashMap::new(),
                                is_integer: in_integer_section,
                            });
                        column.coefficients.insert(row_name2.to_string(), coeff2);
                    }
                }
            }
        }
    }

    if columns_map.is_empty() {
        None
    } else {
        let mut columns: Vec<Column> = columns_map.into_values().collect();
        columns.sort_by(|a, b| a.name.cmp(&b.name));
        Some(columns)
    }
}

fn parse_indicators_section(content: &str) -> Option<Vec<Indicator>> {
    let mut in_indicators_section = false;
    let mut indicators = Vec::new();

    for line in content.lines() {
        if line.starts_with('*') {
            continue;
        }

        let trimmed = line.trim();

        if trimmed == "INDICATORS" {
            in_indicators_section = true;
            continue;
        }

        if in_indicators_section && trimmed == "ENDATA" {
            break;
        }

        if in_indicators_section && !trimmed.is_empty() {
            // Format: IF row_name indicator_variable indicator_value
            let parts: Vec<&str> = trimmed.split_whitespace().collect();

            if parts.len() >= 4 && parts[0] == "IF" {
                let row_name = parts[1].to_string();
                let indicator_variable = parts[2].to_string();

                if let Ok(indicator_value) = parts[3].parse::<f64>() {
                    indicators.push(Indicator {
                        row_name,
                        indicator_variable,
                        indicator_value,
                    });
                }
            }
        }
    }

    if indicators.is_empty() {
        None
    } else {
        Some(indicators)
    }
}