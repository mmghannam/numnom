// Simple MPS parser - adding COLUMNS section
use std::fs;
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;
use flate2::read::GzDecoder;
#[cfg(feature = "validate")]
use russcip::prelude::*;
use serde::{Deserialize, Serialize};

use nom::{
    IResult,
    bytes::complete::{tag, take_while1, take_until},
    character::complete::{char, space0, space1, line_ending, multispace0, alphanumeric1, one_of},
    multi::{many0, many1},
    sequence::{tuple, preceded, terminated},
    combinator::{opt, map_res, recognize},
    branch::alt,
};

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

// Zero-copy parsing structures that reference the original input
#[derive(Debug)]
struct ColumnRef<'a> {
    name: &'a str,
    coefficients: HashMap<&'a str, f64>,
    is_integer: bool,
}

#[derive(Debug)]
struct RowRef<'a> {
    row_type: char,
    name: &'a str,
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

// Central MIP (Mixed Integer Programming) model structures
// These can be used for any optimization format: MPS, LP, etc.
#[derive(Debug, Serialize, Deserialize)]
pub struct MipModel {
    pub name: String,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub obj_coeff: f64,
    #[serde(rename = "type")]
    pub var_type: VariableType,
    pub lb: f64,
    pub ub: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VariableType {
    Integer,
    Continuous,
    Binary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub coefficients: Vec<Coefficient>,
    pub lhs: f64,
    pub rhs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coefficient {
    pub var_name: String,
    pub coeff: f64,
}


fn read_gzipped_file(filepath: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = fs::File::open(filepath)?;
    let mut decoder = GzDecoder::new(file);
    let mut content = String::new();
    decoder.read_to_string(&mut content)?;
    Ok(content)
}

// Ultra-fast byte-level COLUMNS parser with aggressive optimizations
fn parse_columns_ultra_fast(content: &str) -> Option<Vec<Column>> {
    let start_time = Instant::now();

    let bytes = content.as_bytes();
    let len = bytes.len();

    // Pre-estimate capacity based on file size (roughly 1 variable per 200 bytes)
    let estimated_variables = (len / 200).max(1024);
    let mut columns_map: HashMap<String, Column> = HashMap::with_capacity(estimated_variables);

    let mut in_columns_section = false;
    let mut in_integer_section = false;

    let mut line_parsing_time = std::time::Duration::ZERO;
    let mut tokenization_time = std::time::Duration::ZERO;
    let mut string_conversion_time = std::time::Duration::ZERO;
    let mut hashmap_time = std::time::Duration::ZERO;
    let mut sort_time = std::time::Duration::ZERO;

    // Ultra-fast line scanning with optimized newline detection
    let mut pos = 0;

    // Optimized line-by-line parsing
    while pos < len {
        let line_start_time = Instant::now();
        let line_start = pos;

        // Simple byte-by-byte newline scanning (fastest for this use case)
        while pos < len && bytes[pos] != b'\n' {
            pos += 1;
        }

        let line_bytes = &bytes[line_start..pos];
        if pos < len { pos += 1; } // Skip newline

        // Skip empty lines and comments (byte-level check)
        if line_bytes.is_empty() || line_bytes[0] == b'*' {
            continue;
        }

        // Fast whitespace trimming
        let line_trimmed = trim_bytes(line_bytes);

        // Section detection (byte comparison)
        if line_trimmed == b"COLUMNS" {
            in_columns_section = true;
            continue;
        }

        if in_columns_section && (line_trimmed == b"RHS" || line_trimmed == b"BOUNDS" || line_trimmed == b"ENDATA") {
            break;
        }

        if !in_columns_section {
            continue;
        }

        line_parsing_time += line_start_time.elapsed();

        // Fast tokenization without allocation
        let tok_start = Instant::now();
        let (parts, part_count) = split_whitespace_bytes_fast(line_trimmed);
        if part_count < 3 {
            continue;
        }
        tokenization_time += tok_start.elapsed();

        // MARKER detection (byte comparison)
        if part_count >= 3 && parts[1] == b"'MARKER'" {
            if part_count >= 3 && parts[2] == b"'INTORG'" {
                in_integer_section = true;
            } else if part_count >= 3 && parts[2] == b"'INTEND'" {
                in_integer_section = false;
            }
            continue;
        }

        // Parse coefficient directly from bytes
        if part_count >= 3 {
            if let Ok(coeff) = parse_f64_bytes(parts[2]) {
                // Convert to strings only when needed
                let str_start = Instant::now();
                let col_name = bytes_to_string(parts[0]);
                let row_name = bytes_to_string(parts[1]);
                string_conversion_time += str_start.elapsed();

                let hash_start = Instant::now();
                // Avoid cloning by using get_mut first
                if let Some(column) = columns_map.get_mut(&col_name) {
                    column.coefficients.insert(row_name, coeff);
                    if in_integer_section {
                        column.is_integer = true;
                    }
                } else {
                    // Only create new entry if it doesn't exist
                    let mut new_column = Column {
                        name: col_name.clone(),
                        coefficients: HashMap::with_capacity(64), // Even larger initial capacity
                        is_integer: in_integer_section,
                    };
                    new_column.coefficients.insert(row_name, coeff);
                    columns_map.insert(col_name, new_column);
                }
                hashmap_time += hash_start.elapsed();
            }

            // Second coefficient pair
            if part_count >= 5 {
                if let Ok(coeff2) = parse_f64_bytes(parts[4]) {
                    let str_start = Instant::now();
                    let col_name = bytes_to_string(parts[0]);
                    let row_name2 = bytes_to_string(parts[3]);
                    string_conversion_time += str_start.elapsed();

                    let hash_start = Instant::now();
                    // Avoid cloning by using get_mut first
                    if let Some(column) = columns_map.get_mut(&col_name) {
                        column.coefficients.insert(row_name2, coeff2);
                    } else {
                        // Only create new entry if it doesn't exist
                        let mut new_column = Column {
                            name: col_name.clone(),
                            coefficients: HashMap::with_capacity(64), // Even larger initial capacity
                            is_integer: in_integer_section,
                        };
                        new_column.coefficients.insert(row_name2, coeff2);
                        columns_map.insert(col_name, new_column);
                    }
                    hashmap_time += hash_start.elapsed();
                }
            }
        }
    }

    if columns_map.is_empty() {
        None
    } else {
        let sort_start = Instant::now();
        let mut columns: Vec<Column> = Vec::with_capacity(columns_map.len());
        columns.extend(columns_map.into_values());
        columns.sort_unstable_by(|a, b| a.name.cmp(&b.name)); // unstable sort is faster
        sort_time += sort_start.elapsed();

        let total_time = start_time.elapsed();

        println!("=== PROFILING RESULTS ===");
        println!("Total parsing time: {:?}", total_time);
        println!("Line parsing:       {:?} ({:.1}%)", line_parsing_time, line_parsing_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("Tokenization:       {:?} ({:.1}%)", tokenization_time, tokenization_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("String conversion:  {:?} ({:.1}%)", string_conversion_time, string_conversion_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("HashMap ops:        {:?} ({:.1}%)", hashmap_time, hashmap_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("Sort:               {:?} ({:.1}%)", sort_time, sort_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("========================");

        Some(columns)
    }
}

// Zero-copy parsing - eliminates string allocations during parsing
fn parse_columns_zero_copy(content: &str) -> Option<Vec<ColumnRef<'_>>> {
    let start_time = Instant::now();

    let bytes = content.as_bytes();
    let len = bytes.len();

    // Pre-estimate capacity based on file size (roughly 1 variable per 200 bytes)
    let estimated_variables = (len / 200).max(1024);
    let mut columns_map: HashMap<&str, ColumnRef<'_>> = HashMap::with_capacity(estimated_variables);

    let mut in_columns_section = false;
    let mut in_integer_section = false;

    let mut line_parsing_time = std::time::Duration::ZERO;
    let mut tokenization_time = std::time::Duration::ZERO;
    let mut string_conversion_time = std::time::Duration::ZERO;
    let mut hashmap_time = std::time::Duration::ZERO;
    let mut sort_time = std::time::Duration::ZERO;

    // Ultra-fast line scanning
    let mut pos = 0;

    // Optimized line-by-line parsing
    while pos < len {
        let line_start_time = Instant::now();
        let line_start = pos;

        // Simple byte-by-byte newline scanning (fastest for this use case)
        while pos < len && bytes[pos] != b'\n' {
            pos += 1;
        }

        let line_bytes = &bytes[line_start..pos];
        if pos < len { pos += 1; } // Skip newline

        // Skip empty lines and comments (byte-level check)
        if line_bytes.is_empty() || line_bytes[0] == b'*' {
            continue;
        }

        // Fast whitespace trimming
        let line_trimmed = trim_bytes(line_bytes);

        // Section detection (byte comparison)
        if line_trimmed == b"COLUMNS" {
            in_columns_section = true;
            continue;
        }

        if in_columns_section && (line_trimmed == b"RHS" || line_trimmed == b"BOUNDS" || line_trimmed == b"ENDATA") {
            break;
        }

        if !in_columns_section {
            continue;
        }

        line_parsing_time += line_start_time.elapsed();

        // Fast tokenization without allocation
        let tok_start = Instant::now();
        let (parts, part_count) = split_whitespace_bytes_fast(line_trimmed);
        if part_count < 3 {
            continue;
        }
        tokenization_time += tok_start.elapsed();

        // MARKER detection (byte comparison)
        if part_count >= 3 && parts[1] == b"'MARKER'" {
            if part_count >= 3 && parts[2] == b"'INTORG'" {
                in_integer_section = true;
            } else if part_count >= 3 && parts[2] == b"'INTEND'" {
                in_integer_section = false;
            }
            continue;
        }

        // Parse coefficient directly from bytes - ZERO-COPY!
        if part_count >= 3 {
            if let Ok(coeff) = parse_f64_bytes(parts[2]) {
                // Zero-copy string conversion - no allocations!
                let str_start = Instant::now();
                let col_name = unsafe { std::str::from_utf8_unchecked(parts[0]) };
                let row_name = unsafe { std::str::from_utf8_unchecked(parts[1]) };
                string_conversion_time += str_start.elapsed();

                let hash_start = Instant::now();
                // Avoid cloning by using get_mut first
                if let Some(column) = columns_map.get_mut(col_name) {
                    column.coefficients.insert(row_name, coeff);
                    if in_integer_section {
                        column.is_integer = true;
                    }
                } else {
                    // Only create new entry if it doesn't exist
                    let mut new_column = ColumnRef {
                        name: col_name,
                        coefficients: HashMap::with_capacity(64), // Even larger initial capacity
                        is_integer: in_integer_section,
                    };
                    new_column.coefficients.insert(row_name, coeff);
                    columns_map.insert(col_name, new_column);
                }
                hashmap_time += hash_start.elapsed();
            }

            // Second coefficient pair - ZERO-COPY!
            if part_count >= 5 {
                if let Ok(coeff2) = parse_f64_bytes(parts[4]) {
                    let str_start = Instant::now();
                    let col_name = unsafe { std::str::from_utf8_unchecked(parts[0]) };
                    let row_name2 = unsafe { std::str::from_utf8_unchecked(parts[3]) };
                    string_conversion_time += str_start.elapsed();

                    let hash_start = Instant::now();
                    // Avoid cloning by using get_mut first
                    if let Some(column) = columns_map.get_mut(col_name) {
                        column.coefficients.insert(row_name2, coeff2);
                    } else {
                        // Only create new entry if it doesn't exist
                        let mut new_column = ColumnRef {
                            name: col_name,
                            coefficients: HashMap::with_capacity(64), // Even larger initial capacity
                            is_integer: in_integer_section,
                        };
                        new_column.coefficients.insert(row_name2, coeff2);
                        columns_map.insert(col_name, new_column);
                    }
                    hashmap_time += hash_start.elapsed();
                }
            }
        }
    }

    if columns_map.is_empty() {
        None
    } else {
        let sort_start = Instant::now();
        let mut columns: Vec<ColumnRef<'_>> = Vec::with_capacity(columns_map.len());
        columns.extend(columns_map.into_values());
        columns.sort_unstable_by(|a, b| a.name.cmp(&b.name)); // unstable sort is faster
        sort_time += sort_start.elapsed();

        let total_time = start_time.elapsed();

        println!("=== ZERO-COPY PROFILING RESULTS ===");
        println!("Total parsing time: {:?}", total_time);
        println!("Line parsing:       {:?} ({:.1}%)", line_parsing_time, line_parsing_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("Tokenization:       {:?} ({:.1}%)", tokenization_time, tokenization_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("String conversion:  {:?} ({:.1}%)", string_conversion_time, string_conversion_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("HashMap ops:        {:?} ({:.1}%)", hashmap_time, hashmap_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("Sort:               {:?} ({:.1}%)", sort_time, sort_time.as_secs_f64() / total_time.as_secs_f64() * 100.0);
        println!("====================================");

        Some(columns)
    }
}

// Convert zero-copy ColumnRef to owned Column for JSON serialization
fn convert_columns_to_owned(column_refs: Vec<ColumnRef<'_>>) -> Vec<Column> {
    column_refs.into_iter()
        .map(|col_ref| Column {
            name: col_ref.name.to_string(),  // Only allocation happens here
            coefficients: col_ref.coefficients.into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            is_integer: col_ref.is_integer,
        })
        .collect()
}

fn parse_rows_zero_copy(content: &str) -> Option<Vec<RowRef<'_>>> {
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut rows = Vec::with_capacity(256);

    let mut in_rows_section = false;
    let mut pos = 0;

    while pos < len {
        let line_start = pos;

        // Find end of line
        while pos < len && bytes[pos] != b'\n' {
            pos += 1;
        }

        let line_bytes = &bytes[line_start..pos];
        if pos < len { pos += 1; } // Skip newline

        // Skip empty lines and comments
        if line_bytes.is_empty() || line_bytes[0] == b'*' {
            continue;
        }

        let line_trimmed = trim_bytes(line_bytes);

        // Section detection
        if line_trimmed == b"ROWS" {
            in_rows_section = true;
            continue;
        }

        if in_rows_section && (line_trimmed == b"COLUMNS" || line_trimmed == b"ENDATA") {
            break;
        }

        if !in_rows_section {
            continue;
        }

        // Fast tokenization
        let (parts, part_count) = split_whitespace_bytes_fast(line_trimmed);
        if part_count < 2 {
            continue;
        }

        // Zero-copy parsing - use string slices that reference original content
        let type_str = unsafe { std::str::from_utf8_unchecked(parts[0]) };
        let name_str = unsafe { std::str::from_utf8_unchecked(parts[1]) };

        if let Some(row_type) = type_str.chars().next() {
            rows.push(RowRef {
                row_type,
                name: name_str,
            });
        }
    }

    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
}

fn convert_rows_to_owned(row_refs: Vec<RowRef<'_>>) -> Vec<Row> {
    row_refs.into_iter()
        .map(|row_ref| Row {
            row_type: row_ref.row_type,
            name: row_ref.name.to_string(),  // Only allocation happens here
        })
        .collect()
}

// Optimized byte-to-string conversion
fn bytes_to_string(bytes: &[u8]) -> String {
    // Safety: MPS format is ASCII, so this is safe
    unsafe { String::from_utf8_unchecked(bytes.to_vec()) }
}

// Helper functions for byte parsing
fn trim_bytes(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();

    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }

    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &bytes[start..end]
}

fn split_whitespace_bytes(line: &[u8]) -> Vec<&[u8]> {
    let mut parts = Vec::with_capacity(8);
    let mut start = 0;
    let mut in_token = false;

    for (i, &byte) in line.iter().enumerate() {
        if byte.is_ascii_whitespace() {
            if in_token {
                parts.push(&line[start..i]);
                in_token = false;
            }
        } else if !in_token {
            start = i;
            in_token = true;
        }
    }

    if in_token {
        parts.push(&line[start..]);
    }

    parts
}

// Ultra-fast tokenization without Vec allocation
fn split_whitespace_bytes_fast(line: &[u8]) -> ([&[u8]; 8], usize) {
    let mut parts = [&line[0..0]; 8]; // Initialize with empty slices
    let mut part_count = 0;
    let mut start = 0;
    let mut in_token = false;

    for (i, &byte) in line.iter().enumerate() {
        if byte.is_ascii_whitespace() {
            if in_token && part_count < 8 {
                parts[part_count] = &line[start..i];
                part_count += 1;
                in_token = false;
            }
        } else if !in_token {
            start = i;
            in_token = true;
        }
    }

    if in_token && part_count < 8 {
        parts[part_count] = &line[start..];
        part_count += 1;
    }

    (parts, part_count)
}

fn parse_f64_bytes(bytes: &[u8]) -> Result<f64, std::num::ParseFloatError> {
    // Safety: We know these are ASCII bytes from MPS format
    let s = unsafe { std::str::from_utf8_unchecked(bytes) };
    s.parse()
}


// Zero-copy nom parsers
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '-')(input)
}

fn parse_float(input: &str) -> IResult<&str, f64> {
    map_res(
        recognize(tuple((
            opt(char('-')),
            take_while1(|c: char| c.is_ascii_digit()),
            opt(tuple((char('.'), take_while1(|c: char| c.is_ascii_digit())))),
            opt(tuple((one_of("eE"), opt(one_of("+-")), take_while1(|c: char| c.is_ascii_digit()))))
        ))),
        |s: &str| s.parse::<f64>()
    )(input)
}

fn skip_comment_lines(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c != '\n')(input)
}

// High-performance nom-based COLUMNS parser with minimal allocations
fn parse_columns_nom_fast(content: &str) -> Option<Vec<Column>> {
    let mut columns_map: HashMap<String, Column> = HashMap::with_capacity(1024);
    let mut in_columns_section = false;
    let mut in_integer_section = false;

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.starts_with('*') || line.is_empty() {
            continue;
        }

        // Section detection
        if line == "COLUMNS" {
            in_columns_section = true;
            continue;
        }

        if in_columns_section && (line == "RHS" || line == "BOUNDS" || line == "ENDATA") {
            break;
        }

        if !in_columns_section {
            continue;
        }

        // Use nom for precise parsing
        // Parse MARKER lines
        if let Ok((_, (_, _, _, _, marker_type))) = tuple((
            parse_identifier,
            space1,
            tag("'MARKER'"),
            space1,
            alt((tag("'INTORG'"), tag("'INTEND'")))
        ))(line) {
            in_integer_section = marker_type == "'INTORG'";
            continue;
        }

        // Parse regular column entries with nom
        if let Ok((rest_line, (col_name, _, row_name, _, coeff))) = tuple((
            parse_identifier,
            space1,
            parse_identifier,
            space1,
            parse_float
        ))(line) {
            // Efficiently handle column creation
            let column = columns_map.entry(col_name.to_string())
                .or_insert_with(|| Column {
                    name: col_name.to_string(),
                    coefficients: HashMap::with_capacity(16),
                    is_integer: in_integer_section,
                });

            column.coefficients.insert(row_name.to_string(), coeff);

            if in_integer_section {
                column.is_integer = true;
            }

            // Parse optional second coefficient pair
            if let Ok((_, (_, row_name2, _, coeff2))) = tuple((
                space1,
                parse_identifier,
                space1,
                parse_float
            ))(rest_line) {
                column.coefficients.insert(row_name2.to_string(), coeff2);
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

fn print_help(program_name: &str) {
    println!("numnom - A Mathematical Programming files parser (currently MPS only)");
    println!();
    println!("USAGE:");
    println!("    {} <file> [OPTIONS]", program_name);
    println!();
    println!("ARGS:");
    println!("    <file>        Path to the optimization file to parse");
    println!();
    println!("OPTIONS:");
    println!("    --json        Output parsed data as JSON5 format with native Infinity support");
    println!("    --test-json   Test JSON5 serialization/deserialization round-trip");
    println!("    --validate    Validate parsing against SCIP solver (requires 'validate' feature)");
    println!("    --help, -h    Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    {} problem.mps              # Parse and display file contents", program_name);
    println!("    {} problem.mps --json       # Output as JSON5 format", program_name);
    println!("    {} problem.mps --validate   # Validate against SCIP solver", program_name);
    println!();
    println!("BUILD OPTIONS:");
    println!("    Default build: cargo build");
    println!("    With validation: cargo build --features validate");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help(&args[0]);
        return;
    }

    // Check for help flag
    if args.len() >= 2 && (args[1] == "--help" || args[1] == "-h") {
        print_help(&args[0]);
        return;
    }

    let filepath = &args[1];

    // Check file extension
    if !filepath.ends_with(".mps") && !filepath.ends_with(".mps.gz") {
        eprintln!("Error: Currently only MPS files are supported (.mps or .mps.gz extensions)");
        eprintln!("Future versions will support LP and other optimization formats");
        std::process::exit(1);
    }

    let json_output = args.len() > 2 && args[2] == "--json";
    let test_json = args.len() > 2 && args[2] == "--test-json";
    let validate = args.len() > 2 && args[2] == "--validate";

    // Check for help as second argument
    if args.len() > 2 && (args[2] == "--help" || args[2] == "-h") {
        print_help(&args[0]);
        return;
    }
    let content = if filepath.ends_with(".mps.gz") {
        read_gzipped_file(filepath).expect("Failed to read gzipped file")
    } else {
        fs::read_to_string(filepath).expect("Failed to read file")
    };

    // Handle JSON output early (skip human-readable output)
    if json_output {
        let name = parse_name_section(&content).unwrap_or_else(|| "unnamed".to_string());
        let columns = parse_columns_section_quiet(&content).unwrap_or_default();
        let rows = parse_rows_zero_copy(&content)
            .map(convert_rows_to_owned)
            .unwrap_or_default();
        let bounds = parse_bounds_section(&content).unwrap_or_default();
        let rhs = parse_rhs_section(&content).unwrap_or_default();

        // Find objective row name
        let obj_row_name = rows.iter()
            .find(|row| row.row_type == 'N')
            .map(|row| row.name.clone())
            .unwrap_or_else(|| "obj".to_string());

        let mip_model = convert_to_mip(&name, &columns, &rows, &bounds, &rhs, &obj_row_name);

        match serde_json5::to_string(&mip_model) {
            Ok(json_str) => println!("{}", json_str),
            Err(e) => eprintln!("Error serializing to JSON5: {}", e),
        }
        return; // Exit early for JSON output
    }

    // Handle JSON deserialization test
    if test_json {
        let name = parse_name_section(&content).unwrap_or_else(|| "unnamed".to_string());
        let columns = parse_columns_section_quiet(&content).unwrap_or_default();
        let rows = parse_rows_zero_copy(&content)
            .map(convert_rows_to_owned)
            .unwrap_or_default();
        let bounds = parse_bounds_section(&content).unwrap_or_default();
        let rhs = parse_rhs_section(&content).unwrap_or_default();

        // Find objective row name
        let obj_row_name = rows.iter()
            .find(|row| row.row_type == 'N')
            .map(|row| row.name.clone())
            .unwrap_or_else(|| "obj".to_string());

        let mip_model = convert_to_mip(&name, &columns, &rows, &bounds, &rhs, &obj_row_name);

        // Serialize to JSON5 string
        let json_str = serde_json5::to_string(&mip_model).expect("Failed to serialize");

        // Deserialize back from JSON5 string
        let deserialized: MipModel = serde_json5::from_str(&json_str).expect("Failed to deserialize");

        println!("✓ JSON serialization/deserialization test successful!");
        println!("Original problem name: {}", mip_model.name);
        println!("Deserialized problem name: {}", deserialized.name);
        println!("Original variables: {}", mip_model.variables.len());
        println!("Deserialized variables: {}", deserialized.variables.len());
        println!("Original constraints: {}", mip_model.constraints.len());
        println!("Deserialized constraints: {}", deserialized.constraints.len());

        // Test a specific variable
        if let Some(var) = deserialized.variables.first() {
            println!("First variable: {} (obj_coeff: {}, type: {:?}, bounds: {:?} to {:?})",
                    var.name, var.obj_coeff, var.var_type, var.lb, var.ub);
        }

        return;
    }

    // Handle validation request
    if validate {
        #[cfg(feature = "validate")]
        {
            run_validation(filepath, &content);
            return;
        }
        #[cfg(not(feature = "validate"))]
        {
            eprintln!("Error: --validate flag requires building with validation feature enabled:");
            eprintln!("  cargo build --features validate");
            eprintln!("  cargo run --features validate -- {} --validate", filepath);
            return;
        }
    }

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
}

#[cfg(feature = "validate")]
fn run_validation(filepath: &str, content: &str) {
    println!("=== Our Parser ===");

    // Time just the parsing, not the output
    let our_start = Instant::now();
    let name = parse_name_section(&content);
    let rows = parse_rows_section(&content);
    let columns = parse_columns_section_quiet(&content);
    let rhs = parse_rhs_section(&content);
    let bounds = parse_bounds_section(&content);
    let indicators = parse_indicators_section(&content);
    let our_duration = our_start.elapsed();

    // Now display results
    if let Some(name) = name {
        println!("Problem name: {}", name);
    }

    // Display ROWS section
    if let Some(ref rows) = rows {
        println!("\nRows found: {}", rows.len());
        for row in rows {
            println!("  {} {}", row.row_type, row.name);
        }
    }

    // Display COLUMNS section
    if let Some(ref columns) = columns {
        println!("\nColumns found: {}", columns.len());
        for col in columns {
            let var_type = if col.is_integer { "INTEGER" } else { "CONTINUOUS" };
            println!("  Variable: {} [{}]", col.name, var_type);
            for (row, coeff) in &col.coefficients {
                println!("    {} = {}", row, coeff);
            }
        }
    }

    // Display RHS section
    if let Some(ref rhs) = rhs {
        println!("\nRHS values:");
        for (row_name, value) in rhs {
            println!("  {} = {}", row_name, value);
        }
    }

    // Display BOUNDS section
    if let Some(ref bounds) = bounds {
        println!("\nBounds:");
        for bound in bounds {
            match &bound.value {
                Some(val) => println!("  {} {} {}", bound.bound_type, bound.column_name, val),
                None => println!("  {} {}", bound.bound_type, bound.column_name),
            }
        }
    }

    // Display INDICATORS section
    if let Some(ref indicators) = indicators {
        println!("\nIndicators:");
        for indicator in indicators {
            println!("  IF {} = {} THEN {}", indicator.indicator_variable, indicator.indicator_value, indicator.row_name);
        }
    }

    println!("\nOur parser took: {:.2?}", our_duration);

    println!("\n=== SCIP Parser ===");
    let scip_start = Instant::now();
    let scip_model = Model::new()
        .hide_output()
        .include_default_plugins()
        .read_prob(filepath)
        .expect("SCIP failed to read MPS file");

    let scip_duration = scip_start.elapsed();
    println!("SCIP parsing took: {:.2?}", scip_duration);
    println!("Constraints: {}", scip_model.n_conss());
    println!("Variables: {}", scip_model.n_vars());

    println!("\n=== Performance Comparison ===");
    println!("Our parser:  {:.2?}", our_duration);
    println!("SCIP parser: {:.2?}", scip_duration);
    if our_duration < scip_duration {
        let speedup = scip_duration.as_secs_f64() / our_duration.as_secs_f64();
        println!("Our parser is {:.1}x faster", speedup);
    } else {
        let slowdown = our_duration.as_secs_f64() / scip_duration.as_secs_f64();
        println!("SCIP parser is {:.1}x faster", slowdown);
    }

    // Detailed comparison
    println!("\n=== Verification ===");

    // Detailed validation of variable values using already-parsed data
    let scip_vars = scip_model.vars();
    let our_columns = columns.unwrap_or_default();
    let our_bounds = bounds.unwrap_or_default();
    let _our_rhs = rhs.unwrap_or_default();
    let our_rows = rows.unwrap_or_default();

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
            "LI" => lb = bound.value,  // Lower Integer bound
            "UI" => ub = bound.value,  // Upper Integer bound
            "FX" => { lb = bound.value; ub = bound.value; },
            "FR" => { lb = None; ub = None; },
            _ => {}
        }
        our_bounds_map.insert(bound.column_name.clone(), (lb, ub));
    }

    let mut var_mismatches = 0;
    let mut bound_mismatches = 0;
    let mut obj_mismatches = 0;
    let mut type_mismatches = 0;

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

            // Check variable type
            let scip_vtype = var.var_type();
            // Check if variable has integer bounds (LI/UI indicate integer variables)
            let has_integer_bounds = our_bounds.iter()
                .any(|bound| bound.column_name == var_name &&
                     (bound.bound_type == "LI" || bound.bound_type == "UI"));
            let our_is_integer = our_var.is_integer || has_integer_bounds;
            let scip_is_integer = matches!(scip_vtype, russcip::VarType::Integer | russcip::VarType::Binary);

            if our_is_integer != scip_is_integer {
                let our_type = if our_is_integer { "INTEGER" } else { "CONTINUOUS" };
                let scip_type = if scip_is_integer { "INTEGER" } else { "CONTINUOUS" };
                println!("  TYPE MISMATCH {}: {} (ours) vs {} (SCIP)", var_name, our_type, scip_type);
                type_mismatches += 1;
            }
        } else {
            println!("  MISSING VAR: {} not found in our parser", var_name);
            var_mismatches += 1;
        }
    }

    // Compare counts
    let our_var_count = parse_columns_section_quiet(&content).map(|c| c.len()).unwrap_or(0);
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
    println!("Variable type mismatches: {}", type_mismatches);
    println!("Matrix coefficient mismatches: {}", matrix_mismatches);

    if var_mismatches == 0 && obj_mismatches == 0 && bound_mismatches == 0 && type_mismatches == 0 && matrix_mismatches == 0 {
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
    let mut rows = Vec::with_capacity(256);

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
            // Optimized: avoid Vec allocation
            let mut parts = trimmed.split_whitespace();
            if let Some(type_str) = parts.next() {
                if let Some(name_str) = parts.next() {
                    if let Some(row_type) = type_str.chars().next() {
                        rows.push(Row {
                            row_type,
                            name: name_str.to_string()
                        });
                    }
                }
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
    let mut bounds = Vec::with_capacity(512);

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
            // Optimized: avoid Vec allocation
            let mut parts = trimmed.split_whitespace();

            if let Some(bound_type_str) = parts.next() {
                if let Some(_bound_name) = parts.next() {  // Skip bound name
                    if let Some(column_name_str) = parts.next() {
                        // Some bound types don't have values (FR, MI, PL, BV)
                        let value = if let Some(value_str) = parts.next() {
                            value_str.parse::<f64>().ok()
                        } else {
                            None
                        };

                        bounds.push(Bound {
                            bound_type: bound_type_str.to_string(),
                            column_name: column_name_str.to_string(),
                            value,
                        });
                    }
                }
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
    parse_columns_ultra_fast(content)
}

fn parse_columns_section_quiet(content: &str) -> Option<Vec<Column>> {
    // Use zero-copy parsing, then convert to owned strings only for output
    match parse_columns_zero_copy(content) {
        Some(column_refs) => Some(convert_columns_to_owned(column_refs)),
        None => None,
    }
}

// New nom-based implementation
fn parse_columns_section_nom(content: &str, _quiet: bool) -> Option<Vec<Column>> {
    // Try the fast nom implementation first
    if let Some(columns) = parse_columns_nom_fast(content) {
        Some(columns)
    } else {
        // Fallback to old implementation if nom parsing fails
        parse_columns_section_impl(content, _quiet)
    }
}

fn parse_columns_section_impl(content: &str, quiet: bool) -> Option<Vec<Column>> {
    let mut in_columns_section = false;
    let mut in_integer_section = false;
    // Pre-allocate HashMap with estimated capacity to reduce rehashing
    let mut columns_map: HashMap<String, Column> = HashMap::with_capacity(1024);

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
            // Optimized: avoid Vec allocation, use iterator directly
            let mut parts = trimmed.split_whitespace();

            if let Some(col_name) = parts.next() {
                if let Some(second) = parts.next() {
                    // Check for MARKER lines - optimize string comparison
                    if second == "'MARKER'" {
                        if let Some(third) = parts.next() {
                            if third == "'INTORG'" {
                                in_integer_section = true;
                                if !quiet {
                                    println!("  [Found INTORG marker - starting integer section]");
                                }
                            } else if third == "'INTEND'" {
                                in_integer_section = false;
                                if !quiet {
                                    println!("  [Found INTEND marker - ending integer section]");
                                }
                            }
                        }
                        continue;
                    }

                    // Parse regular column data - second is row_name
                    if let Some(coeff_str) = parts.next() {
                        if let Ok(coeff) = coeff_str.parse::<f64>() {
                            // Reduce string allocations by using entry with closure
                            let column = columns_map.entry(col_name.to_string())
                                .or_insert_with(|| Column {
                                    name: col_name.to_string(),
                                    coefficients: HashMap::with_capacity(16),
                                    is_integer: in_integer_section,
                                });
                            column.coefficients.insert(second.to_string(), coeff);

                            // Update integer status if we're in integer section
                            if in_integer_section {
                                column.is_integer = true;
                            }
                        }

                        // Check for second coefficient pair
                        if let Some(row_name2) = parts.next() {
                            if let Some(coeff_str2) = parts.next() {
                                if let Ok(coeff2) = coeff_str2.parse::<f64>() {
                                    let column = columns_map.entry(col_name.to_string())
                                        .or_insert_with(|| Column {
                                            name: col_name.to_string(),
                                            coefficients: HashMap::with_capacity(16),
                                            is_integer: in_integer_section,
                                        });
                                    column.coefficients.insert(row_name2.to_string(), coeff2);
                                }
                            }
                        }
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

fn convert_to_mip(
    problem_name: &str,
    columns: &[Column],
    rows: &[Row],
    bounds: &[Bound],
    rhs: &HashMap<String, f64>,
    obj_row_name: &str,
) -> MipModel {
    // Build bounds map for quick lookup
    let mut bounds_map: HashMap<String, (Option<f64>, Option<f64>)> = HashMap::new();
    for bound in bounds {
        let (mut lb, mut ub) = bounds_map.get(&bound.column_name).unwrap_or(&(None, None)).clone();
        match bound.bound_type.as_str() {
            "LO" => lb = bound.value,
            "UP" => ub = bound.value,
            "LI" => lb = bound.value, // Lower integer bound
            "UI" => ub = bound.value, // Upper integer bound
            "FX" => { lb = bound.value; ub = bound.value; },
            "FR" => { lb = None; ub = None; },
            "BV" => { lb = Some(0.0); ub = Some(1.0); }, // Binary variable
            _ => {}
        }
        bounds_map.insert(bound.column_name.clone(), (lb, ub));
    }

    // Convert variables
    let variables: Vec<Variable> = columns.iter().map(|col| {
        let (lb, ub) = bounds_map.get(&col.name).unwrap_or(&(None, None));
        let obj_coeff = col.coefficients.get(obj_row_name).unwrap_or(&0.0);

        // Check if variable has integer bounds (LI/UI indicate integer variables)
        let has_integer_bounds = bounds.iter()
            .any(|bound| bound.column_name == col.name &&
                 (bound.bound_type == "LI" || bound.bound_type == "UI"));

        // Determine variable type
        let var_type = if lb == &Some(0.0) && ub == &Some(1.0) {
            VariableType::Binary
        } else if col.is_integer || has_integer_bounds {
            VariableType::Integer
        } else {
            VariableType::Continuous
        };

        Variable {
            name: col.name.clone(),
            obj_coeff: *obj_coeff,
            var_type,
            lb: lb.unwrap_or(f64::NEG_INFINITY),
            ub: ub.unwrap_or(f64::INFINITY),
        }
    }).collect();

    // Convert constraints
    let mut constraint_coeffs: HashMap<String, Vec<Coefficient>> = HashMap::new();

    // Build constraint coefficients from variables
    for col in columns {
        for (constraint_name, coeff) in &col.coefficients {
            if constraint_name == obj_row_name {
                continue; // Skip objective
            }
            constraint_coeffs.entry(constraint_name.clone())
                .or_insert_with(Vec::new)
                .push(Coefficient {
                    var_name: col.name.clone(),
                    coeff: *coeff,
                });
        }
    }

    let constraints: Vec<Constraint> = rows.iter()
        .filter(|row| row.row_type != 'N') // Skip objective
        .map(|row| {
            let coefficients = constraint_coeffs.get(&row.name).unwrap_or(&Vec::new()).clone();
            let rhs_value = rhs.get(&row.name).copied();

            let (lhs, rhs) = match row.row_type {
                'E' => (rhs_value.unwrap_or(0.0), rhs_value.unwrap_or(0.0)), // Equality: lhs = rhs
                'L' => (f64::NEG_INFINITY, rhs_value.unwrap_or(0.0)),      // Less than: -inf <= x <= rhs
                'G' => (rhs_value.unwrap_or(0.0), f64::INFINITY),      // Greater than: lhs <= x <= +inf
                _ => (f64::NEG_INFINITY, rhs_value.unwrap_or(0.0)),
            };

            Constraint {
                name: row.name.clone(),
                coefficients,
                lhs,
                rhs,
            }
        }).collect();

    MipModel {
        name: problem_name.to_string(),
        variables,
        constraints,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_roundtrip() {
        // Sample JSON data
        let json_str = r#"{
            "name": "test_problem",
            "variables": [
                {
                    "name": "x1",
                    "obj_coeff": 2.5,
                    "type": "continuous",
                    "lb": 0.0,
                    "ub": "inf"
                },
                {
                    "name": "x2",
                    "obj_coeff": 1.0,
                    "type": "integer",
                    "lb": 1.0,
                    "ub": 10.0
                }
            ],
            "constraints": [
                {
                    "name": "c1",
                    "coefficients": [
                        {"var_name": "x1", "coeff": 1.0},
                        {"var_name": "x2", "coeff": 2.0}
                    ],
                    "lhs": "-inf",
                    "rhs": 5.0
                }
            ]
        }"#;

        // Deserialize from JSON5
        let mip_model: MipModel = serde_json5::from_str(json_str).expect("Failed to deserialize JSON5");

        // Verify deserialized data
        assert_eq!(mip_model.name, "test_problem");
        assert_eq!(mip_model.variables.len(), 2);
        assert_eq!(mip_model.constraints.len(), 1);

        // Check first variable
        let var1 = &mip_model.variables[0];
        assert_eq!(var1.name, "x1");
        assert_eq!(var1.obj_coeff, 2.5);
        assert!(matches!(var1.var_type, VariableType::Continuous));
        assert_eq!(var1.lb, 0.0);
        assert_eq!(var1.ub, f64::INFINITY);

        // Check second variable
        let var2 = &mip_model.variables[1];
        assert_eq!(var2.name, "x2");
        assert_eq!(var2.obj_coeff, 1.0);
        assert!(matches!(var2.var_type, VariableType::Integer));

        // Serialize back to JSON5
        let serialized = serde_json5::to_string(&mip_model).expect("Failed to serialize");

        // Deserialize again to ensure round-trip works
        let _roundtrip: MipModel = serde_json5::from_str(&serialized).expect("Failed round-trip");
    }
}