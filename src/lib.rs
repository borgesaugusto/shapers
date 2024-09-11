#![warn(missing_docs)]
//! ## shapers
//! Routines for operations with shapes.
//!
//! Supported shapes and operations:,
//! | Shape   | Status                       |
//! |---------|------------------------------|
//! | Circle  | Fitting (taubinSVD and LSQ)  |
//! | Ellipse | Planned                      |
//!
//! Part of this funcitons are based in <https://github.com/AlliedToasters/circle-fit>
//! and the algorithms implemented by Nicolai Chernov <https://people.cas.uab.edu/~mosya/cl/MATLABcircle.html>
//! 

/// Module for circle fitting functions
pub mod circle_fit;
mod aux_funcs;
/// Module for native error types
pub mod errors;
use pyo3::prelude::*;
extern crate blas_src;

#[pymodule]
fn shapers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<circle_fit::FitCircleParams>()?;
    m.add_function(wrap_pyfunction!(circle_fit::taubin_svd, m)?)?;
    m.add_function(wrap_pyfunction!(circle_fit::fit_geometrical, m)?)?;
    m.add_function(wrap_pyfunction!(circle_fit::fit_lsq, m)?)?;
    
    Ok(())
}
