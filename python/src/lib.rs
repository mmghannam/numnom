use numnom::{self as nn, VarType};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

const VT_CONTINUOUS: u8 = 0;
const VT_INTEGER: u8 = 1;
const VT_SEMI_CONTINUOUS: u8 = 2;
const VT_SEMI_INTEGER: u8 = 3;

fn vartype_to_u8(v: VarType) -> u8 {
    match v {
        VarType::Continuous => VT_CONTINUOUS,
        VarType::Integer => VT_INTEGER,
        VarType::SemiContinuous => VT_SEMI_CONTINUOUS,
        VarType::SemiInteger => VT_SEMI_INTEGER,
    }
}

fn u8_to_vartype(v: u8) -> PyResult<VarType> {
    match v {
        VT_CONTINUOUS => Ok(VarType::Continuous),
        VT_INTEGER => Ok(VarType::Integer),
        VT_SEMI_CONTINUOUS => Ok(VarType::SemiContinuous),
        VT_SEMI_INTEGER => Ok(VarType::SemiInteger),
        other => Err(PyValueError::new_err(format!(
            "invalid integrality code: {other} (expected 0..=3)"
        ))),
    }
}

#[pyclass(module = "numnom._numnom", name = "SparseMatrix", frozen)]
struct PySparseMatrix {
    #[pyo3(get)]
    start: Py<PyArray1<u32>>,
    #[pyo3(get)]
    index: Py<PyArray1<u32>>,
    #[pyo3(get)]
    value: Py<PyArray1<f64>>,
}

#[pymethods]
impl PySparseMatrix {
    fn __repr__(&self, py: Python<'_>) -> String {
        let nnz = self.value.bind(py).len();
        let ncol_plus_1 = self.start.bind(py).len();
        format!(
            "SparseMatrix(num_col={}, nnz={})",
            ncol_plus_1.saturating_sub(1),
            nnz,
        )
    }
}

#[pyclass(module = "numnom._numnom", name = "Model", frozen)]
struct PyModel {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    num_row: u32,
    #[pyo3(get)]
    num_col: u32,
    #[pyo3(get)]
    obj_sense_minimize: bool,
    #[pyo3(get)]
    obj_offset: f64,
    #[pyo3(get)]
    objective_name: String,
    #[pyo3(get)]
    col_cost: Py<PyArray1<f64>>,
    #[pyo3(get)]
    col_lower: Py<PyArray1<f64>>,
    #[pyo3(get)]
    col_upper: Py<PyArray1<f64>>,
    #[pyo3(get)]
    row_lower: Py<PyArray1<f64>>,
    #[pyo3(get)]
    row_upper: Py<PyArray1<f64>>,
    #[pyo3(get)]
    col_integrality: Py<PyArray1<u8>>,
    #[pyo3(get)]
    a_matrix: Py<PySparseMatrix>,
    #[pyo3(get)]
    col_names: Vec<String>,
    #[pyo3(get)]
    row_names: Vec<String>,
}

#[pymethods]
impl PyModel {
    fn __repr__(&self) -> String {
        format!(
            "Model(name={:?}, num_row={}, num_col={}, sense={})",
            self.name,
            self.num_row,
            self.num_col,
            if self.obj_sense_minimize { "min" } else { "max" },
        )
    }
}

fn build_py_model(py: Python<'_>, m: nn::Model) -> PyResult<PyModel> {
    let integrality_u8: Vec<u8> = m.col_integrality.iter().copied().map(vartype_to_u8).collect();

    let a_matrix = PySparseMatrix {
        start: m.a_matrix.start.into_pyarray(py).unbind(),
        index: m.a_matrix.index.into_pyarray(py).unbind(),
        value: m.a_matrix.value.into_pyarray(py).unbind(),
    };

    Ok(PyModel {
        name: m.name,
        num_row: m.num_row,
        num_col: m.num_col,
        obj_sense_minimize: m.obj_sense_minimize,
        obj_offset: m.obj_offset,
        objective_name: m.objective_name,
        col_cost: m.col_cost.into_pyarray(py).unbind(),
        col_lower: m.col_lower.into_pyarray(py).unbind(),
        col_upper: m.col_upper.into_pyarray(py).unbind(),
        row_lower: m.row_lower.into_pyarray(py).unbind(),
        row_upper: m.row_upper.into_pyarray(py).unbind(),
        col_integrality: integrality_u8.into_pyarray(py).unbind(),
        a_matrix: Py::new(py, a_matrix)?,
        col_names: m.col_names,
        row_names: m.row_names,
    })
}

#[pyfunction]
fn parse_file(py: Python<'_>, path: &str) -> PyResult<PyModel> {
    let model = py
        .allow_threads(|| nn::parse_mps_file(path))
        .map_err(PyValueError::new_err)?;
    build_py_model(py, model)
}

#[pyfunction]
fn parse_str(py: Python<'_>, content: &str) -> PyResult<PyModel> {
    let model = nn::parse_mps_str(content).map_err(PyValueError::new_err)?;
    build_py_model(py, model)
}

fn rebuild_rust_model(py: Python<'_>, m: &PyModel) -> PyResult<nn::Model> {
    let a = m.a_matrix.bind(py);
    let a_ref = a.borrow();

    let col_integrality: Vec<VarType> = m
        .col_integrality
        .bind(py)
        .readonly()
        .as_slice()?
        .iter()
        .copied()
        .map(u8_to_vartype)
        .collect::<PyResult<_>>()?;

    Ok(nn::Model {
        name: m.name.clone(),
        num_row: m.num_row,
        num_col: m.num_col,
        obj_sense_minimize: m.obj_sense_minimize,
        obj_offset: m.obj_offset,
        objective_name: m.objective_name.clone(),
        col_cost: m.col_cost.bind(py).readonly().as_slice()?.to_vec(),
        col_lower: m.col_lower.bind(py).readonly().as_slice()?.to_vec(),
        col_upper: m.col_upper.bind(py).readonly().as_slice()?.to_vec(),
        row_lower: m.row_lower.bind(py).readonly().as_slice()?.to_vec(),
        row_upper: m.row_upper.bind(py).readonly().as_slice()?.to_vec(),
        a_matrix: nn::SparseMatrix {
            start: a_ref.start.bind(py).readonly().as_slice()?.to_vec(),
            index: a_ref.index.bind(py).readonly().as_slice()?.to_vec(),
            value: a_ref.value.bind(py).readonly().as_slice()?.to_vec(),
        },
        col_names: m.col_names.clone(),
        row_names: m.row_names.clone(),
        col_integrality,
    })
}

#[pyfunction]
fn write_file(py: Python<'_>, model: &PyModel, path: &str) -> PyResult<()> {
    let rust_model = rebuild_rust_model(py, model)?;
    py.allow_threads(|| nn::write_mps_file(&rust_model, path))
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn write_str(py: Python<'_>, model: &PyModel) -> PyResult<String> {
    let rust_model = rebuild_rust_model(py, model)?;
    let mut buf: Vec<u8> = Vec::new();
    py.allow_threads(|| nn::write_mps(&rust_model, &mut buf))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    String::from_utf8(buf).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn _numnom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CONTINUOUS", VT_CONTINUOUS)?;
    m.add("INTEGER", VT_INTEGER)?;
    m.add("SEMI_CONTINUOUS", VT_SEMI_CONTINUOUS)?;
    m.add("SEMI_INTEGER", VT_SEMI_INTEGER)?;

    m.add_class::<PyModel>()?;
    m.add_class::<PySparseMatrix>()?;

    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_str, m)?)?;
    m.add_function(wrap_pyfunction!(write_file, m)?)?;
    m.add_function(wrap_pyfunction!(write_str, m)?)?;

    Ok(())
}
