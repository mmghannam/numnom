use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let show_timings = args.iter().any(|a| a == "--timings");
    let help = args.iter().any(|a| a == "--help" || a == "-h");
    let file_args: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with('-')).collect();

    if help || file_args.is_empty() {
        eprintln!("Usage: numnom <file.mps|file.mps.gz> [--timings]");
        std::process::exit(1);
    }

    let path = file_args[0];

    let mut timings = if show_timings {
        Some(numnom::SectionTimings::default())
    } else {
        None
    };

    let start = Instant::now();
    let model = match numnom::parse_mps_file_timed(path, &mut timings) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    println!("Name:        {}", model.name);
    println!("Rows:        {}", model.num_row);
    println!("Cols:        {}", model.num_col);
    println!("Nonzeros:    {}", model.a_matrix.value.len());
    println!(
        "Sense:       {}",
        if model.obj_sense_minimize {
            "minimize"
        } else {
            "maximize"
        }
    );
    println!("Obj offset:  {}", model.obj_offset);
    println!("Parsed in:   {:.3}ms", elapsed.as_secs_f64() * 1000.0);

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
    if n_int > 0 {
        println!("Integer:     {n_int}");
    }
    println!("Continuous:  {n_cont}");

    if let Some(t) = timings {
        println!();
        println!("--- Section timings ---");
        println!("  Read:        {:.3}ms", t.read.as_secs_f64() * 1000.0);
        println!("  Rows:        {:.3}ms", t.rows.as_secs_f64() * 1000.0);
        println!("  Columns:     {:.3}ms", t.cols.as_secs_f64() * 1000.0);
        println!("  RHS:         {:.3}ms", t.rhs.as_secs_f64() * 1000.0);
        println!("  Bounds:      {:.3}ms", t.bounds.as_secs_f64() * 1000.0);
        println!("  Ranges:      {:.3}ms", t.ranges.as_secs_f64() * 1000.0);
        println!("  Fill matrix: {:.3}ms", t.fill_matrix.as_secs_f64() * 1000.0);
    }
}
