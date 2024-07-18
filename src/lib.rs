mod circle_fit;
mod aux_funcs;
use pyo3::prelude::*;
extern crate blas_src;
use circle_fit::taubin_svd;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn py_taubin_svd(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    taubin_svd(xs, ys)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn circlers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(py_taubin_svd, m)?)?;
    Ok(())
}
