pub mod circle_fit;
mod aux_funcs;
pub mod errors;
use pyo3::prelude::*;
extern crate blas_src;


#[pymodule]
fn shapers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circle_fit::taubin_svd, m)?)?;
    m.add_function(wrap_pyfunction!(circle_fit::fit_geometrical, m)?)?;
    m.add_function(wrap_pyfunction!(circle_fit::fit_lsq, m)?)?;
    
    Ok(())
}
