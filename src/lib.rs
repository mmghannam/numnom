use rustc_hash::FxHashMap as HashMap;
use std::io::Read;

const INF: f64 = f64::INFINITY;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    Continuous,
    Integer,
    SemiContinuous,
    SemiInteger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowType {
    Le,
    Ge,
    Eq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parsekey {
    None,
    Name,
    Objsense,
    Rows,
    Cols,
    Rhs,
    Bounds,
    Ranges,
    End,
    Skip,
    Comment,
    Fail,
}

/// Timing breakdown for each MPS section.
#[derive(Debug, Default)]
pub struct SectionTimings {
    pub read: std::time::Duration,
    pub file_bytes: u64, // compressed size on disk
    pub data_bytes: u64, // decompressed size in memory
    pub rows: std::time::Duration,
    pub cols: std::time::Duration,
    pub rhs: std::time::Duration,
    pub bounds: std::time::Duration,
    pub ranges: std::time::Duration,
    pub fill_matrix: std::time::Duration,
}

/// Sparse column-wise matrix (CSC format).
#[derive(Debug, Default)]
pub struct SparseMatrix {
    pub start: Vec<u32>,
    pub index: Vec<u32>,
    pub value: Vec<f64>,
}

/// Parsed MPS model.
#[derive(Debug, Default)]
pub struct Model {
    pub name: String,
    pub num_row: u32,
    pub num_col: u32,
    pub obj_sense_minimize: bool,
    pub obj_offset: f64,
    pub objective_name: String,
    pub col_cost: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    pub a_matrix: SparseMatrix,
    pub col_names: Vec<String>,
    pub row_names: Vec<String>,
    pub col_integrality: Vec<VarType>,
}

/// Parser borrows from input string to avoid allocations during hot loops.
struct Parser<'a> {
    num_row: u32,
    num_col: u32,
    num_nz: u32,

    obj_sense_minimize: bool,
    obj_offset: f64,
    mps_name: &'a str,
    objective_name: &'a str,

    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    col_lower: Vec<f64>,
    col_upper: Vec<f64>,

    row_names: Vec<&'a str>,
    col_names: Vec<&'a str>,
    col_integrality: Vec<VarType>,
    col_binary: Vec<bool>,

    row_type: Vec<RowType>,

    // CSC matrix built directly during COLUMNS parsing (no intermediate triplets)
    a_start: Vec<u32>,
    a_index: Vec<u32>,
    a_value: Vec<f64>,
    col_cost: Vec<f64>,

    // Uses i32 because -1 = objective row, -2 = free row
    rowname2idx: HashMap<&'a str, i32>,
    colname2idx: HashMap<&'a str, u32>,

}

impl<'a> Parser<'a> {
    fn new() -> Self {
        Parser {
            num_row: 0,
            num_col: 0,
            num_nz: 0,
            obj_sense_minimize: true,
            obj_offset: 0.0,
            mps_name: "",
            objective_name: "Objective",
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            col_lower: Vec::new(),
            col_upper: Vec::new(),
            row_names: Vec::new(),
            col_names: Vec::new(),
            col_integrality: Vec::new(),
            col_binary: Vec::new(),
            row_type: Vec::new(),
            a_start: Vec::new(),
            a_index: Vec::new(),
            a_value: Vec::new(),
            col_cost: Vec::new(),
            rowname2idx: HashMap::default(),
            colname2idx: HashMap::default(),
        }
    }

    fn get_col_idx(&mut self, name: &'a str) -> u32 {
        if let Some(&idx) = self.colname2idx.get(name) {
            return idx;
        }
        let idx = self.num_col;
        self.num_col += 1;
        self.colname2idx.insert(name, idx);
        self.col_names.push(name);
        self.col_integrality.push(VarType::Continuous);
        self.col_binary.push(false);
        self.col_lower.push(0.0);
        self.col_upper.push(INF);
        idx
    }

    fn check_section(word: &str) -> Parsekey {
        match word {
            "NAME" => Parsekey::Name,
            "OBJSENSE" => Parsekey::Objsense,
            "ROWS" => Parsekey::Rows,
            "COLUMNS" => Parsekey::Cols,
            "RHS" => Parsekey::Rhs,
            "BOUNDS" => Parsekey::Bounds,
            "RANGES" => Parsekey::Ranges,
            "ENDATA" => Parsekey::End,
            "INDICATORS" | "SETS" | "SOS" | "QUADOBJ" | "QMATRIX" | "QSECTION" | "QCMATRIX"
            | "CSECTION" | "DELAYEDROWS" | "MODELCUTS" | "USERCUTS" | "GENCONS" | "PWLOBJ"
            | "PWLNAM" | "PWLCON" => Parsekey::Skip,
            _ => Parsekey::None,
        }
    }

    fn check_first_word(line: &str) -> (Parsekey, &str, &str) {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            return (Parsekey::Comment, "", "");
        }
        let end = trimmed
            .find(|c: char| c.is_ascii_whitespace())
            .unwrap_or(trimmed.len());
        let word = &trimmed[..end];
        let rest = &trimmed[end..];

        let upper = word.to_ascii_uppercase();
        let key = Self::check_section(&upper);

        if key != Parsekey::None
            && key != Parsekey::Name
            && key != Parsekey::Objsense
            && key != Parsekey::Skip
            && !rest.trim().is_empty()
        {
            return (Parsekey::None, word, rest);
        }

        (key, word, rest)
    }

    #[inline]
    fn parse_value(word: &str) -> f64 {
        fast_float(word)
    }

    fn parse_str(
        &mut self,
        input: &'a str,
        timings: &mut Option<SectionTimings>,
    ) -> Result<(), String> {
        // Try to read size hints from comment headers (e.g., *ROWS: 124, *NONZERO: 91028)
        let (hint_rows, hint_cols, hint_nz) = parse_size_hints(input);

        let est_nz = hint_nz.unwrap_or(input.len() / 30);
        let est_cols = hint_cols.unwrap_or(est_nz / 10);
        let est_rows = hint_rows.unwrap_or(est_cols / 4);
        self.a_index.reserve(est_nz);
        self.a_value.reserve(est_nz);
        self.a_start.reserve(est_cols);
        self.col_names.reserve(est_cols);
        self.col_cost.reserve(est_cols);
        self.col_lower.reserve(est_cols);
        self.col_upper.reserve(est_cols);
        self.col_integrality.reserve(est_cols);
        self.col_binary.reserve(est_cols);
        self.row_names.reserve(est_rows);
        self.row_lower.reserve(est_rows);
        self.row_upper.reserve(est_rows);
        self.row_type.reserve(est_rows);

        let mut lines = input.lines();
        let mut keyword = Parsekey::None;

        loop {
            let section_start = std::time::Instant::now();
            let prev = keyword;
            match keyword {
                Parsekey::End | Parsekey::Fail => break,
                Parsekey::Objsense => keyword = self.parse_objsense(&mut lines),
                Parsekey::Rows => keyword = self.parse_rows(&mut lines),
                Parsekey::Cols => {
                    keyword = self.parse_cols(&mut lines);
                    // Build colname2idx after COLUMNS — deferred for speed
                    self.colname2idx.reserve(self.num_col as usize);
                    for (i, &name) in self.col_names.iter().enumerate() {
                        self.colname2idx.insert(name, i as u32);
                    }
                }
                Parsekey::Rhs => keyword = self.parse_rhs(&mut lines),
                Parsekey::Bounds => keyword = self.parse_bounds(&mut lines),
                Parsekey::Ranges => keyword = self.parse_ranges(&mut lines),
                Parsekey::Skip => keyword = self.parse_skip(&mut lines),
                _ => keyword = self.parse_default(&mut lines),
            }
            if let Some(t) = timings {
                let elapsed = section_start.elapsed();
                match prev {
                    Parsekey::Rows => t.rows = elapsed,
                    Parsekey::Cols => t.cols = elapsed,
                    Parsekey::Rhs => t.rhs = elapsed,
                    Parsekey::Bounds => t.bounds = elapsed,
                    Parsekey::Ranges => t.ranges = elapsed,
                    _ => {}
                }
            }
        }

        if keyword == Parsekey::Fail {
            return Err("Failed to parse MPS file".to_string());
        }

        for i in 0..self.num_col as usize {
            if self.col_binary[i] {
                self.col_lower[i] = 0.0;
                self.col_upper[i] = 1.0;
            }
        }

        Ok(())
    }

    #[inline]
    fn is_skip(line: &str) -> bool {
        line.is_empty() || {
            let b = line.as_bytes()[0];
            b == b'*' || b == b'$'
        }
    }

    fn parse_default(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        if let Some(line) = lines.next() {
            if Self::is_skip(line) {
                return Parsekey::Comment;
            }
            let (key, _word, rest) = Self::check_first_word(line);

            if key == Parsekey::Name {
                if let Some(name) = rest.split_whitespace().next() {
                    self.mps_name = name;
                }
                return Parsekey::None;
            }

            if key == Parsekey::Objsense {
                if let Some(sense) = rest.split_whitespace().next() {
                    let upper = sense.to_ascii_uppercase();
                    if upper.starts_with("MAX") {
                        self.obj_sense_minimize = false;
                    }
                }
            }

            key
        } else {
            Parsekey::Fail
        }
    }

    fn parse_skip(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let trimmed = line.trim();
            let (key, _, _) = Self::check_first_word(trimmed);
            if key != Parsekey::None && key != Parsekey::Skip {
                return key;
            }
        }
        Parsekey::Fail
    }

    fn parse_objsense(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let (key, word, _) = Self::check_first_word(line);
            let upper = word.to_ascii_uppercase();
            if upper.starts_with("MAX") {
                self.obj_sense_minimize = false;
                continue;
            }
            if upper.starts_with("MIN") {
                self.obj_sense_minimize = true;
                continue;
            }
            if key != Parsekey::None {
                return key;
            }
        }
        Parsekey::Fail
    }

    fn parse_rows(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        let mut has_obj = false;

        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let trimmed = line.trim();
            let (key, _word, _rest) = Self::check_first_word(trimmed);
            if key != Parsekey::None {
                if !has_obj {
                    self.rowname2idx.insert("artificial_empty_objective", -1);
                }
                return key;
            }

            let mut parts = trimmed.split_ascii_whitespace();
            let row_type_char = match parts.next() {
                Some(s) if !s.is_empty() => s.as_bytes()[0],
                _ => continue,
            };
            let rowname = match parts.next() {
                Some(s) => s,
                None => continue,
            };

            match row_type_char {
                b'N' => {
                    if !has_obj {
                        has_obj = true;
                        self.objective_name = rowname;
                        self.rowname2idx.insert(rowname, -1);
                    } else {
                        self.rowname2idx.insert(rowname, -2);
                    }
                }
                b'L' => {
                    self.rowname2idx.insert(rowname, self.num_row as i32);
                    self.row_names.push(rowname);
                    self.row_lower.push(-INF);
                    self.row_upper.push(0.0);
                    self.row_type.push(RowType::Le);
                    self.num_row += 1;
                }
                b'G' => {
                    self.rowname2idx.insert(rowname, self.num_row as i32);
                    self.row_names.push(rowname);
                    self.row_lower.push(0.0);
                    self.row_upper.push(INF);
                    self.row_type.push(RowType::Ge);
                    self.num_row += 1;
                }
                b'E' => {
                    self.rowname2idx.insert(rowname, self.num_row as i32);
                    self.row_names.push(rowname);
                    self.row_lower.push(0.0);
                    self.row_upper.push(0.0);
                    self.row_type.push(RowType::Eq);
                    self.num_row += 1;
                }
                _ => {
                    return Parsekey::Fail;
                }
            }
        }
        Parsekey::Fail
    }

    #[allow(unused_assignments)]
    fn parse_cols(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        let mut colname: &str = "";
        let mut integral_cols = false;

        let num_row = self.num_row as usize;
        let mut col_value: Vec<f64> = vec![0.0; num_row];
        let mut col_index: Vec<u32> = Vec::with_capacity(num_row);
        let mut col_count: usize = 0;
        let mut col_cost: f64 = 0.0;

        macro_rules! flush_column {
            ($self:expr) => {
                if $self.num_col > 0 {
                    $self.col_cost.push(col_cost);
                    col_cost = 0.0;
                    $self.a_start.push($self.num_nz);
                    for i in 0..col_count {
                        let row = col_index[i];
                        let ru = row as usize;
                        $self.a_index.push(row);
                        $self.a_value.push(col_value[ru]);
                        col_value[ru] = 0.0;
                        $self.num_nz += 1;
                    }
                    col_count = 0;
                    col_index.clear();
                }
            };
        }

        for line in lines {
            if Self::is_skip(line) {
                continue;
            }

            let bytes = line.as_bytes();
            let is_indented = bytes[0] == b' ' || bytes[0] == b'\t';

            let mut words: [&str; 6] = [""; 6];
            let mut nwords = 0;
            let len = bytes.len();
            let mut pos = 0;
            while pos < len && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            while pos < len && nwords < 6 {
                let start = pos;
                while pos < len && !bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                words[nwords] = unsafe { std::str::from_utf8_unchecked(&bytes[start..pos]) };
                nwords += 1;
                while pos < len && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
            }
            if nwords == 0 {
                continue;
            }

            if !is_indented {
                let upper = words[0].to_ascii_uppercase();
                let key = Self::check_section(&upper);
                if key != Parsekey::None {
                    if nwords == 1
                        || key == Parsekey::Name
                        || key == Parsekey::Objsense
                        || key == Parsekey::Skip
                    {
                        flush_column!(self);
                        return key;
                    }
                }
            }

            if nwords >= 3 && words[1] == "'MARKER'" {
                if words[2] == "'INTORG'" {
                    integral_cols = true;
                } else if words[2] == "'INTEND'" {
                    integral_cols = false;
                }
                continue;
            }

            let this_col = words[0];

            if this_col != colname {
                flush_column!(self);
                colname = this_col;
                self.col_names.push(this_col);
                self.col_integrality.push(if integral_cols {
                    VarType::Integer
                } else {
                    VarType::Continuous
                });
                self.col_binary.push(integral_cols);
                self.col_lower.push(0.0);
                self.col_upper.push(INF);
                self.num_col += 1;
            }

            let mut i = 1;
            while i + 1 < nwords {
                let rowname = words[i];
                let valstr = words[i + 1];
                i += 2;

                if let Some(&rowidx) = self.rowname2idx.get(rowname) {
                    let value = Self::parse_value(valstr);
                    if value == 0.0 {
                        continue;
                    }
                    if rowidx >= 0 {
                        let ru = rowidx as usize;
                        if col_value[ru] == 0.0 {
                            col_value[ru] = value;
                            col_index.push(rowidx as u32);
                            col_count += 1;
                        }
                    } else if rowidx == -1 {
                        if col_cost == 0.0 {
                            col_cost = value;
                        }
                    }
                }
            }
        }
        Parsekey::Fail
    }


    fn parse_rhs(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        let mut has_row_entry = vec![false; self.num_row as usize];
        let mut has_obj_entry = false;

        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let is_indented = line.as_bytes()[0] == b' ' || line.as_bytes()[0] == b'\t';
            let trimmed = line.trim();

            let mut words: [&str; 6] = [""; 6];
            let mut nwords = 0;
            for w in trimmed.split_ascii_whitespace() {
                if nwords < 6 {
                    words[nwords] = w;
                    nwords += 1;
                }
            }
            if nwords == 0 {
                continue;
            }

            if !is_indented {
                let upper = words[0].to_ascii_uppercase();
                let key = Self::check_section(&upper);
                if key != Parsekey::None && key != Parsekey::Rhs {
                    if nwords == 1
                        || key == Parsekey::Name
                        || key == Parsekey::Objsense
                        || key == Parsekey::Skip
                    {
                        return key;
                    }
                }
            }

            let start = if self.rowname2idx.contains_key(words[0]) {
                0
            } else {
                1
            };

            let mut i = start;
            while i + 1 < nwords {
                let rowname = words[i];
                let valstr = words[i + 1];
                i += 2;

                if let Some(&rowidx) = self.rowname2idx.get(rowname) {
                    let value = Self::parse_value(valstr);
                    if rowidx >= 0 {
                        let ru = rowidx as usize;
                        if has_row_entry[ru] {
                            continue;
                        }
                        match self.row_type[ru] {
                            RowType::Eq => {
                                self.row_lower[ru] = value;
                                self.row_upper[ru] = value;
                            }
                            RowType::Le => {
                                self.row_upper[ru] = value;
                            }
                            RowType::Ge => {
                                self.row_lower[ru] = value;
                            }
                        }
                        has_row_entry[ru] = true;
                    } else if rowidx == -1 && !has_obj_entry {
                        self.obj_offset = -value;
                        has_obj_entry = true;
                    }
                }
            }
        }
        Parsekey::Fail
    }

    fn parse_bounds(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let is_indented = line.as_bytes()[0] == b' ' || line.as_bytes()[0] == b'\t';
            let trimmed = line.trim();

            let mut words: [&str; 6] = [""; 6];
            let mut nwords = 0;
            for w in trimmed.split_ascii_whitespace() {
                if nwords < 6 {
                    words[nwords] = w;
                    nwords += 1;
                }
            }
            if nwords == 0 {
                continue;
            }

            if !is_indented {
                let upper = words[0].to_ascii_uppercase();
                let key = Self::check_section(&upper);
                if key != Parsekey::None {
                    if nwords == 1
                        || key == Parsekey::Name
                        || key == Parsekey::Objsense
                        || key == Parsekey::Skip
                    {
                        return key;
                    }
                }
            }

            let bound_type = words[0];

            let (col_name, val_idx) = if nwords >= 3 {
                if self.colname2idx.contains_key(words[1]) {
                    (words[1], 2)
                } else {
                    (words[2], 3)
                }
            } else if nwords == 2 {
                (words[1], 2)
            } else {
                continue;
            };

            let colidx = if let Some(&idx) = self.colname2idx.get(col_name) {
                idx
            } else {
                self.get_col_idx(col_name)
            };
            let cu = colidx as usize;

            match bound_type {
                "LO" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_lower[cu] = value;
                    self.col_binary[cu] = false;
                }
                "UP" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_upper[cu] = value;
                    self.col_binary[cu] = false;
                }
                "FX" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_lower[cu] = value;
                    self.col_upper[cu] = value;
                    self.col_binary[cu] = false;
                }
                "FR" => {
                    self.col_lower[cu] = -INF;
                    self.col_upper[cu] = INF;
                    self.col_binary[cu] = false;
                }
                "MI" => {
                    self.col_lower[cu] = -INF;
                    self.col_binary[cu] = false;
                }
                "PL" => {
                    self.col_upper[cu] = INF;
                    self.col_binary[cu] = false;
                }
                "BV" => {
                    self.col_integrality[cu] = VarType::Integer;
                    self.col_binary[cu] = true;
                    self.col_lower[cu] = 0.0;
                    self.col_upper[cu] = 1.0;
                }
                "LI" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_lower[cu] = value;
                    self.col_integrality[cu] = VarType::Integer;
                    self.col_binary[cu] = false;
                }
                "UI" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_upper[cu] = value;
                    self.col_integrality[cu] = VarType::Integer;
                    self.col_binary[cu] = false;
                }
                "SC" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_upper[cu] = value;
                    self.col_integrality[cu] = VarType::SemiContinuous;
                    self.col_binary[cu] = false;
                }
                "SI" => {
                    let value = if val_idx < nwords {
                        Self::parse_value(words[val_idx])
                    } else {
                        0.0
                    };
                    self.col_upper[cu] = value;
                    self.col_integrality[cu] = VarType::SemiInteger;
                    self.col_binary[cu] = false;
                }
                _ => {
                    return Parsekey::Fail;
                }
            }
        }
        Parsekey::Fail
    }

    fn parse_ranges(&mut self, lines: &mut std::str::Lines<'a>) -> Parsekey {
        for line in lines {
            if Self::is_skip(line) {
                continue;
            }
            let is_indented = line.as_bytes()[0] == b' ' || line.as_bytes()[0] == b'\t';
            let trimmed = line.trim();

            let mut words: [&str; 6] = [""; 6];
            let mut nwords = 0;
            for w in trimmed.split_ascii_whitespace() {
                if nwords < 6 {
                    words[nwords] = w;
                    nwords += 1;
                }
            }
            if nwords == 0 {
                continue;
            }

            if !is_indented {
                let upper = words[0].to_ascii_uppercase();
                let key = Self::check_section(&upper);
                if key != Parsekey::None {
                    if nwords == 1
                        || key == Parsekey::Name
                        || key == Parsekey::Objsense
                        || key == Parsekey::Skip
                    {
                        return key;
                    }
                }
            }

            let start = if self.rowname2idx.contains_key(words[0]) {
                0
            } else {
                1
            };

            let mut i = start;
            while i + 1 < nwords {
                let rowname = words[i];
                let valstr = words[i + 1];
                i += 2;

                if let Some(&rowidx) = self.rowname2idx.get(rowname) {
                    if rowidx < 0 {
                        continue;
                    }
                    let ru = rowidx as usize;
                    let val = Self::parse_value(valstr);

                    match self.row_type[ru] {
                        RowType::Le => {
                            self.row_lower[ru] = self.row_upper[ru] - val.abs();
                        }
                        RowType::Ge => {
                            self.row_upper[ru] = self.row_lower[ru] + val.abs();
                        }
                        RowType::Eq => {
                            if val < 0.0 {
                                self.row_lower[ru] = self.row_upper[ru] - val.abs();
                            } else {
                                self.row_upper[ru] = self.row_lower[ru] + val.abs();
                            }
                        }
                    }
                }
            }
        }
        Parsekey::Fail
    }

    fn into_model(mut self) -> Model {
        let num_col = self.num_col as usize;

        // Finalize CSC: add sentinel, pad for columns added by BOUNDS with no entries
        self.a_start.push(self.num_nz);
        while self.a_start.len() < num_col + 1 {
            self.a_start.push(self.num_nz);
        }
        // Pad col_cost for columns added by BOUNDS
        while self.col_cost.len() < num_col {
            self.col_cost.push(0.0);
        }

        Model {
            name: self.mps_name.to_string(),
            num_row: self.num_row,
            num_col: self.num_col,
            obj_sense_minimize: self.obj_sense_minimize,
            obj_offset: self.obj_offset,
            objective_name: self.objective_name.to_string(),
            col_cost: self.col_cost,
            col_lower: self.col_lower,
            col_upper: self.col_upper,
            row_lower: self.row_lower,
            row_upper: self.row_upper,
            a_matrix: SparseMatrix {
                start: self.a_start,
                index: self.a_index,
                value: self.a_value,
            },
            col_names: self.col_names.into_iter().map(|s| s.to_string()).collect(),
            row_names: self.row_names.into_iter().map(|s| s.to_string()).collect(),
            col_integrality: self.col_integrality,
        }
    }
}

/// Parse size hints from MPS comment headers.
/// Some files have lines like: *ROWS: 124  *COLUMNS: 10757  *NONZERO: 91028
fn parse_size_hints(input: &str) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut rows = None;
    let mut cols = None;
    let mut nz = None;

    for line in input.lines().take(20) {
        if !line.starts_with('*') {
            if line.trim_start().starts_with("NAME") {
                break; // past the header comments
            }
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if let Some(pos) = upper.find("ROWS:") {
            if let Some(val) = extract_hint_number(&line[pos + 5..]) {
                rows = Some(val);
            }
        } else if let Some(pos) = upper.find("COLUMNS:") {
            if let Some(val) = extract_hint_number(&line[pos + 8..]) {
                cols = Some(val);
            }
        } else if let Some(pos) = upper.find("NONZERO:") {
            if let Some(val) = extract_hint_number(&line[pos + 8..]) {
                nz = Some(val);
            }
        }
    }
    (rows, cols, nz)
}

fn extract_hint_number(s: &str) -> Option<usize> {
    s.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
        .parse::<usize>()
        .ok()
}

/// Fast f64 parsing for MPS number formats.
/// Uses fast-float2 for bit-exact results matching Rust's str::parse::<f64>().
/// Handles MPS-specific 'd'/'D' exponent notation by replacing with 'e'.
#[inline]
fn fast_float(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    // Check for 'd'/'D' exponent (Fortran-style, used in some MPS files).
    // fast-float2 only handles 'e'/'E', so replace if needed.
    if let Some(pos) = s.as_bytes().iter().position(|&b| b == b'd' || b == b'D') {
        let mut buf = s.to_owned();
        // SAFETY: replacing single ASCII byte with another ASCII byte
        unsafe { buf.as_bytes_mut()[pos] = b'e'; }
        fast_float2::parse(&buf).unwrap_or(0.0)
    } else {
        fast_float2::parse(s).unwrap_or(0.0)
    }
}

/// Parse an MPS file from a string.
pub fn parse_mps_str(input: &str) -> Result<Model, String> {
    parse_mps_str_timed(input, &mut None)
}

/// Parse an MPS file from a string, with optional timing breakdown.
pub fn parse_mps_str_timed(
    input: &str,
    timings: &mut Option<SectionTimings>,
) -> Result<Model, String> {
    let mut parser = Parser::new();
    parser.parse_str(input, timings)?;

    let fm_start = std::time::Instant::now();
    let model = parser.into_model();
    if let Some(t) = timings {
        t.fill_matrix = fm_start.elapsed();
    }
    Ok(model)
}


/// Parse an MPS file from a reader (reads all into memory first).
pub fn parse_mps<R: Read>(mut reader: R) -> Result<Model, String> {
    let mut buf = String::new();
    reader
        .read_to_string(&mut buf)
        .map_err(|e| format!("Read error: {e}"))?;
    parse_mps_str(&buf)
}

/// Read file into a String, skipping UTF-8 validation (MPS is ASCII).
fn read_file_fast(path: &str) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Cannot open file: {e}"))?;

    let mut bytes = Vec::new();
    if path.ends_with(".gz") {
        let mut decoder = flate2::read::GzDecoder::new(file);
        decoder
            .read_to_end(&mut bytes)
            .map_err(|e| format!("Read error: {e}"))?;
    } else {
        let mut file = file;
        file.read_to_end(&mut bytes)
            .map_err(|e| format!("Read error: {e}"))?;
    }

    // SAFETY: MPS files are ASCII, which is valid UTF-8
    Ok(unsafe { String::from_utf8_unchecked(bytes) })
}

/// Parse an MPS file from a file path (supports .mps and .mps.gz), with optional timings.
pub fn parse_mps_file_timed(
    path: &str,
    timings: &mut Option<SectionTimings>,
) -> Result<Model, String> {
    let file_size = std::fs::metadata(path)
        .map(|m| m.len())
        .unwrap_or(0);
    let read_start = std::time::Instant::now();
    let buf = read_file_fast(path)?;
    if let Some(t) = timings {
        t.read = read_start.elapsed();
        t.file_bytes = file_size;
        t.data_bytes = buf.len() as u64;
    }
    parse_mps_str_timed(&buf, timings)
}

/// Parse an MPS file from a file path (supports .mps and .mps.gz).
pub fn parse_mps_file(path: &str) -> Result<Model, String> {
    parse_mps_file_timed(path, &mut None)
}

// ─────────────────────────────────────────────────────────────────────
//  MPS Writer
// ─────────────────────────────────────────────────────────────────────

/// Write a [`Model`] in MPS format to any [`std::io::Write`] sink.
///
/// Produces free-format MPS (space-separated fields) accepted by
/// HiGHS, CPLEX, Gurobi, SCIP, and all other modern solvers. The
/// output is a valid round-trip: `parse_mps_str(write_mps(m))` yields
/// a model equivalent to `m`.
///
/// Handles:
/// * `N` / `L` / `G` / `E` row types inferred from `row_lower` /
///   `row_upper` (±∞ for one-sided, equal for equality).
/// * Ranged rows (`row_lower < row_upper`, both finite) via the
///   `RANGES` section.
/// * Integer markers (`'MARKER' 'INTORG'` / `'INTEND'`) toggled per
///   column as integrality changes.
/// * Objective constant (`obj_offset`) emitted as a negated RHS entry
///   for the objective row.
/// * All standard bound types: `LO`, `UP`, `FX`, `FR`, `MI`, `BV`.
pub fn write_mps<W: std::io::Write>(model: &Model, w: &mut W) -> std::io::Result<()> {
    let ncol = model.num_col as usize;
    let nrow = model.num_row as usize;

    // ── NAME ──────────────────────────────────────────────────────
    writeln!(w, "NAME          {}", model.name)?;

    // ── OBJSENSE ──────────────────────────────────────────────────
    if !model.obj_sense_minimize {
        writeln!(w, "OBJSENSE")?;
        writeln!(w, "    MAX")?;
    }

    // ── Classify rows ─────────────────────────────────────────────
    // (row_type, rhs, optional range value)
    #[derive(Clone, Copy)]
    enum RType {
        N,
        L,
        G,
        E,
    }
    let mut row_info: Vec<(RType, f64, Option<f64>)> = Vec::with_capacity(nrow);
    for i in 0..nrow {
        let lo = model.row_lower[i];
        let hi = model.row_upper[i];
        let lo_fin = lo.is_finite();
        let hi_fin = hi.is_finite();
        if lo_fin && hi_fin {
            if (hi - lo).abs() < 1e-12 {
                // Equality.
                row_info.push((RType::E, lo, None));
            } else {
                // Ranged: emit as L with rhs = hi, range = hi - lo.
                row_info.push((RType::L, hi, Some(hi - lo)));
            }
        } else if hi_fin {
            row_info.push((RType::L, hi, None));
        } else if lo_fin {
            row_info.push((RType::G, lo, None));
        } else {
            // Free row (both sides infinite). Emit as N.
            row_info.push((RType::N, 0.0, None));
        }
    }

    // ── ROWS ──────────────────────────────────────────────────────
    writeln!(w, "ROWS")?;
    let obj_name = if model.objective_name.is_empty() {
        "OBJ"
    } else {
        &model.objective_name
    };
    writeln!(w, " N  {obj_name}")?;
    for i in 0..nrow {
        let letter = match row_info[i].0 {
            RType::N => 'N',
            RType::L => 'L',
            RType::G => 'G',
            RType::E => 'E',
        };
        writeln!(w, " {}  {}", letter, model.row_names[i])?;
    }

    // ── COLUMNS ───────────────────────────────────────────────────
    writeln!(w, "COLUMNS")?;
    let mut in_integer_block = false;
    let mut marker_id = 0u32;
    for j in 0..ncol {
        let is_int = matches!(
            model.col_integrality[j],
            VarType::Integer | VarType::SemiInteger
        );
        // Toggle integer markers.
        if is_int && !in_integer_block {
            marker_id += 1;
            writeln!(
                w,
                "    M{marker_id:07}  'MARKER'                 'INTORG'"
            )?;
            in_integer_block = true;
        } else if !is_int && in_integer_block {
            marker_id += 1;
            writeln!(
                w,
                "    M{marker_id:07}  'MARKER'                 'INTEND'"
            )?;
            in_integer_block = false;
        }

        // Collect all nonzeros for column j: objective + matrix entries.
        let col_name = &model.col_names[j];
        let obj_c = model.col_cost[j];
        let s = model.a_matrix.start[j] as usize;
        let e = model.a_matrix.start[j + 1] as usize;

        // Gather entries: (row_name, value).
        let mut entries: Vec<(&str, f64)> = Vec::new();
        if obj_c != 0.0 {
            entries.push((obj_name, obj_c));
        }
        for k in s..e {
            let ri = model.a_matrix.index[k] as usize;
            let v = model.a_matrix.value[k];
            if v != 0.0 {
                entries.push((&model.row_names[ri], v));
            }
        }

        // Emit in pairs (two entries per line) for compactness.
        let mut idx = 0;
        while idx < entries.len() {
            let (rn, rv) = entries[idx];
            if idx + 1 < entries.len() {
                let (rn2, rv2) = entries[idx + 1];
                writeln!(w, "    {col_name}  {rn}  {rv:.12e}  {rn2}  {rv2:.12e}")?;
                idx += 2;
            } else {
                writeln!(w, "    {col_name}  {rn}  {rv:.12e}")?;
                idx += 1;
            }
        }

        // Columns with no entries still need to appear (they may have
        // bounds). Emit a zero obj entry if the column would otherwise
        // be invisible.
        if entries.is_empty() {
            writeln!(w, "    {col_name}  {obj_name}  0.000000000000e+00")?;
        }
    }
    // Close any open integer block.
    if in_integer_block {
        marker_id += 1;
        writeln!(
            w,
            "    M{marker_id:07}  'MARKER'                 'INTEND'"
        )?;
    }

    // ── RHS ───────────────────────────────────────────────────────
    writeln!(w, "RHS")?;
    // Objective offset (negated back to MPS convention).
    if model.obj_offset != 0.0 {
        writeln!(
            w,
            "    RHS1  {obj_name}  {:.12e}",
            -model.obj_offset
        )?;
    }
    for i in 0..nrow {
        let (_, rhs, _) = row_info[i];
        if rhs != 0.0 {
            writeln!(
                w,
                "    RHS1  {}  {rhs:.12e}",
                model.row_names[i]
            )?;
        }
    }

    // ── RANGES ────────────────────────────────────────────────────
    let has_ranges = row_info.iter().any(|(_, _, r)| r.is_some());
    if has_ranges {
        writeln!(w, "RANGES")?;
        for i in 0..nrow {
            if let (_, _, Some(range)) = row_info[i] {
                writeln!(
                    w,
                    "    RNG1  {}  {range:.12e}",
                    model.row_names[i]
                )?;
            }
        }
    }

    // ── BOUNDS ─────────────────────────────────────────────────────
    writeln!(w, "BOUNDS")?;
    for j in 0..ncol {
        let lo = model.col_lower[j];
        let hi = model.col_upper[j];
        let is_int = matches!(
            model.col_integrality[j],
            VarType::Integer | VarType::SemiInteger
        );
        let col_name = &model.col_names[j];

        // Binary shorthand: integer with [0, 1].
        if is_int && lo == 0.0 && hi == 1.0 {
            writeln!(w, " BV BND1  {col_name}")?;
            continue;
        }

        // Free variable (both infinite).
        if !lo.is_finite() && !hi.is_finite() {
            writeln!(w, " FR BND1  {col_name}")?;
            continue;
        }

        // Fixed variable.
        if lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12 {
            writeln!(w, " FX BND1  {col_name}  {lo:.12e}")?;
            continue;
        }

        // Lower bound (default is 0, so skip if 0).
        if lo.is_finite() {
            if is_int {
                writeln!(w, " LI BND1  {col_name}  {lo:.12e}")?;
            } else if lo != 0.0 {
                writeln!(w, " LO BND1  {col_name}  {lo:.12e}")?;
            }
        } else {
            // -∞ lower bound.
            writeln!(w, " MI BND1  {col_name}")?;
        }

        // Upper bound (default is +∞, so skip if +∞).
        if hi.is_finite() {
            if is_int {
                writeln!(w, " UI BND1  {col_name}  {hi:.12e}")?;
            } else {
                writeln!(w, " UP BND1  {col_name}  {hi:.12e}")?;
            }
        }
    }

    // ── ENDATA ────────────────────────────────────────────────────
    writeln!(w, "ENDATA")?;
    Ok(())
}

/// Write a [`Model`] to a file path in MPS format.
pub fn write_mps_file(model: &Model, path: &str) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);
    write_mps(model, &mut w)
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_simple_mps() {
        let mps = "\
NAME          test
ROWS
 N  OBJ
 L  C1
 G  C2
 E  C3
COLUMNS
    X1  OBJ  1.0  C1  2.0
    X1  C2   3.0
    INT1  'MARKER'  'INTORG'
    X2  OBJ  4.0  C1  5.0
    X2  C3   6.0
    INT1  'MARKER'  'INTEND'
    X3  C2   7.0
RHS
    RHS1  C1  10.0  C2  20.0
    RHS1  C3  30.0
BOUNDS
 UP BND1  X1  100.0
 LO BND1  X3  -5.0
ENDATA
";
        let model = parse_mps_str(mps).unwrap();
        assert_eq!(model.num_col, 3);
        assert_eq!(model.num_row, 3);
        assert_eq!(model.col_names, vec!["X1", "X2", "X3"]);
        assert_eq!(model.col_cost, vec![1.0, 4.0, 0.0]);
        assert_eq!(model.col_integrality[0], VarType::Continuous);
        assert_eq!(model.col_integrality[1], VarType::Integer);
        assert_eq!(model.col_integrality[2], VarType::Continuous);
        assert_eq!(model.col_upper[0], 100.0);
        assert_eq!(model.col_lower[2], -5.0);
        assert_eq!(model.row_upper[0], 10.0);
        assert_eq!(model.row_lower[1], 20.0);
        assert_eq!(model.row_lower[2], 30.0);
        assert_eq!(model.row_upper[2], 30.0);

        assert_eq!(model.a_matrix.start, vec![0, 2, 4, 5]);
        assert_eq!(model.a_matrix.index, vec![0, 1, 0, 2, 1]);
        assert_eq!(model.a_matrix.value, vec![2.0, 3.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_write_roundtrip_complex_mip() {
        // A model exercising every MPS feature the writer handles:
        //   - MIN objective with offset
        //   - L, G, E, and ranged rows
        //   - Continuous, integer, and binary variables
        //   - Free variable, fixed variable, negative lower bound
        //   - Non-trivial matrix with multiple nonzeros per column
        let mps_in = "\
NAME          complex_mip
ROWS
 N  OBJ
 L  LE_ROW
 G  GE_ROW
 E  EQ_ROW
 L  RNG_ROW
COLUMNS
    X1  OBJ  1.0  LE_ROW  2.5
    X1  GE_ROW  -1.0  EQ_ROW  3.0
    INT1  'MARKER'  'INTORG'
    Y1  OBJ  4.0  LE_ROW  1.0
    Y1  GE_ROW  2.0  RNG_ROW  5.0
    INT1  'MARKER'  'INTEND'
    X2  EQ_ROW  7.0  RNG_ROW  -3.0
    INT2  'MARKER'  'INTORG'
    B1  OBJ  -1.0  LE_ROW  6.0
    B1  GE_ROW  1.0
    INT2  'MARKER'  'INTEND'
    X3  OBJ  2.0  GE_ROW  4.0
    X3  RNG_ROW  1.0
RHS
    RHS1  OBJ  -10.0
    RHS1  LE_ROW  20.0  GE_ROW  5.0
    RHS1  EQ_ROW  15.0  RNG_ROW  100.0
RANGES
    RNG1  RNG_ROW  30.0
BOUNDS
 UP BND1  X1  50.0
 LO BND1  X1  -10.0
 LO BND1  Y1  0.0
 UI BND1  Y1  8.0
 BV BND1  B1
 FR BND1  X2
 FX BND1  X3  7.0
ENDATA
";
        let m1 = parse_mps_str(mps_in).unwrap();

        // Verify key properties of the parsed model.
        assert_eq!(m1.num_col, 5);
        assert_eq!(m1.num_row, 4);
        assert_eq!(m1.obj_sense_minimize, true);
        assert!((m1.obj_offset - 10.0).abs() < 1e-12, "obj_offset = {}", m1.obj_offset);
        assert_eq!(m1.col_integrality[0], VarType::Continuous); // X1
        assert_eq!(m1.col_integrality[1], VarType::Integer);    // Y1
        assert_eq!(m1.col_integrality[2], VarType::Continuous); // X2
        assert_eq!(m1.col_integrality[3], VarType::Integer);    // B1
        assert_eq!(m1.col_integrality[4], VarType::Continuous); // X3

        // Bounds.
        assert_eq!(m1.col_lower[0], -10.0);
        assert_eq!(m1.col_upper[0], 50.0);
        assert_eq!(m1.col_lower[1], 0.0);
        assert_eq!(m1.col_upper[1], 8.0);
        assert_eq!(m1.col_lower[2], -INF);
        assert_eq!(m1.col_upper[2], INF);
        assert_eq!(m1.col_lower[3], 0.0);
        assert_eq!(m1.col_upper[3], 1.0);
        assert_eq!(m1.col_lower[4], 7.0);
        assert_eq!(m1.col_upper[4], 7.0);

        // Ranged row: RNG_ROW declared L with rhs 100, range 30.
        // → row_upper = 100, row_lower = 100 - 30 = 70.
        assert_eq!(m1.row_lower[3], 70.0);
        assert_eq!(m1.row_upper[3], 100.0);

        // ── Write → reparse ──────────────────────────────────────
        let mut buf = Vec::new();
        write_mps(&m1, &mut buf).unwrap();
        let mps_out = String::from_utf8(buf).unwrap();
        let m2 = parse_mps_str(&mps_out)
            .unwrap_or_else(|e| panic!("Re-parse failed:\n{mps_out}\nError: {e}"));

        // ── Compare m1 vs m2 ─────────────────────────────────────
        assert_eq!(m2.num_col, m1.num_col);
        assert_eq!(m2.num_row, m1.num_row);
        assert_eq!(m2.name, m1.name);
        assert_eq!(m2.objective_name, m1.objective_name);
        assert_eq!(m2.obj_sense_minimize, m1.obj_sense_minimize);
        assert!(
            (m2.obj_offset - m1.obj_offset).abs() < 1e-9,
            "obj_offset: {} vs {}",
            m2.obj_offset,
            m1.obj_offset
        );
        assert_eq!(m2.col_names, m1.col_names);
        assert_eq!(m2.row_names, m1.row_names);
        assert_eq!(m2.col_integrality, m1.col_integrality);

        for j in 0..m1.num_col as usize {
            assert!(
                (m2.col_cost[j] - m1.col_cost[j]).abs() < 1e-9,
                "col_cost[{j}]: {} vs {}",
                m2.col_cost[j],
                m1.col_cost[j]
            );
            assert!(
                bounds_eq(m2.col_lower[j], m1.col_lower[j]),
                "col_lower[{j}]: {} vs {}",
                m2.col_lower[j],
                m1.col_lower[j]
            );
            assert!(
                bounds_eq(m2.col_upper[j], m1.col_upper[j]),
                "col_upper[{j}]: {} vs {}",
                m2.col_upper[j],
                m1.col_upper[j]
            );
        }
        for i in 0..m1.num_row as usize {
            assert!(
                bounds_eq(m2.row_lower[i], m1.row_lower[i]),
                "row_lower[{i}]: {} vs {}",
                m2.row_lower[i],
                m1.row_lower[i]
            );
            assert!(
                bounds_eq(m2.row_upper[i], m1.row_upper[i]),
                "row_upper[{i}]: {} vs {}",
                m2.row_upper[i],
                m1.row_upper[i]
            );
        }

        // Matrix: compare CSC entry by entry.
        assert_eq!(m2.a_matrix.start, m1.a_matrix.start);
        assert_eq!(m2.a_matrix.index.len(), m1.a_matrix.index.len());
        for k in 0..m1.a_matrix.index.len() {
            assert_eq!(m2.a_matrix.index[k], m1.a_matrix.index[k]);
            assert!(
                (m2.a_matrix.value[k] - m1.a_matrix.value[k]).abs() < 1e-9,
                "a_matrix.value[{k}]: {} vs {}",
                m2.a_matrix.value[k],
                m1.a_matrix.value[k]
            );
        }
    }

    fn bounds_eq(a: f64, b: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            return a.signum() == b.signum();
        }
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_ranges() {
        let mps = "\
NAME          ranges_test
ROWS
 N  OBJ
 L  C1
 G  C2
COLUMNS
    X1  OBJ  1.0  C1  1.0
    X1  C2   1.0
RHS
    RHS1  C1  10.0  C2  5.0
RANGES
    RNG1  C1  3.0  C2  4.0
ENDATA
";
        let model = parse_mps_str(mps).unwrap();
        assert_eq!(model.row_lower[0], 7.0);
        assert_eq!(model.row_upper[0], 10.0);
        assert_eq!(model.row_lower[1], 5.0);
        assert_eq!(model.row_upper[1], 9.0);
    }
}
