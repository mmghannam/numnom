use std::time::Instant;

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1}KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes}B")
    }
}

fn fmt_time(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let quiet = args.iter().any(|a| a == "--quiet" || a == "-q");
    let help = args.iter().any(|a| a == "--help" || a == "-h");
    let file_args: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with('-')).collect();

    if help || file_args.is_empty() {
        eprintln!("Usage: numnom <file.mps|file.mps.gz> [-q|--quiet]");
        std::process::exit(1);
    }

    let path = file_args[0];

    if !quiet {
        let filename = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(path);
        let is_gz = path.ends_with(".gz");
        eprint!("  Loading {filename}");
        if is_gz {
            eprint!(" (decompressing)");
        }
        eprint!("...");
    }

    let mut timings = Some(numnom::SectionTimings::default());

    let start = Instant::now();
    let model = match numnom::parse_mps_file_timed(path, &mut timings) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("\nError: {e}");
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0;

    if !quiet {
        eprintln!(" done in {}", fmt_time(ms));
    }

    if quiet {
        return;
    }

    let n_int = model
        .col_integrality
        .iter()
        .filter(|&&t| t == numnom::VarType::Integer)
        .count();
    let n_cont = model
        .col_integrality
        .iter()
        .filter(|&&t| t == numnom::VarType::Continuous)
        .count();
    let n_bin = model
        .col_integrality
        .iter()
        .enumerate()
        .filter(|&(i, &t)| {
            t == numnom::VarType::Integer && model.col_lower[i] == 0.0 && model.col_upper[i] == 1.0
        })
        .count();
    let n_int_only = n_int - n_bin;

    let name = if model.name.is_empty() {
        std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .trim_end_matches(".mps")
    } else {
        &model.name
    };
    let sense = if model.obj_sense_minimize {
        "min"
    } else {
        "max"
    };

    println!();
    println!("  {name} ({sense})");
    println!();
    println!("  Statistics");
    println!("  ----------");
    println!();
    let nz = model.a_matrix.value.len();
    let density = if model.num_row > 0 && model.num_col > 0 {
        nz as f64 / (model.num_row as f64 * model.num_col as f64) * 100.0
    } else {
        0.0
    };

    println!(
        "  Rows         {:>10}    Cols       {:>10}",
        fmt_count(model.num_row as usize),
        fmt_count(model.num_col as usize)
    );
    println!(
        "  Nonzeros     {:>10}    Density    {:>9.4}%",
        fmt_count(nz),
        density
    );
    if n_cont > 0 {
        println!("  Continuous   {:>10}", fmt_count(n_cont));
    }
    if n_int_only > 0 {
        println!("  Integer      {:>10}", fmt_count(n_int_only));
    }
    if n_bin > 0 {
        println!("  Binary       {:>10}", fmt_count(n_bin));
    }

    if model.obj_offset != 0.0 {
        println!("  Obj offset   {}", model.obj_offset);
    }

    println!();
    println!("  Timing");
    println!("  ------");
    println!();
    if let Some(t) = timings {
        let compressed = t.file_bytes != t.data_bytes && t.file_bytes > 0;
        let read_ms = t.read.as_secs_f64() * 1000.0;
        let parse_ms = ms - read_ms;

        let sections = [
            ("Rows", t.rows),
            ("Columns", t.cols),
            ("RHS", t.rhs),
            ("Bounds", t.bounds),
            ("Ranges", t.ranges),
            ("Finalize", t.fill_matrix),
        ];

        if compressed {
            println!(
                "  Decompress   {} -> {} in {}",
                fmt_bytes(t.file_bytes),
                fmt_bytes(t.data_bytes),
                fmt_time(read_ms)
            );
        } else if t.data_bytes > 0 {
            println!(
                "  Read         {} in {}",
                fmt_bytes(t.data_bytes),
                fmt_time(read_ms)
            );
        }
        println!(
            "  Parse        {} in {}",
            fmt_bytes(t.data_bytes),
            fmt_time(parse_ms)
        );

        for (name, dur) in &sections {
            let section_ms = dur.as_secs_f64() * 1000.0;
            if section_ms >= 0.01 {
                let pct = section_ms / parse_ms * 100.0;
                println!(
                    "    {:<10} {:>10}  ({:.0}%)",
                    name,
                    fmt_time(section_ms),
                    pct
                );
            }
        }
        println!();
    }
}
