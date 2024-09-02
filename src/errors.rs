use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::fmt;
#[derive(Debug)]
pub struct LSQError(argmin::core::Error);

impl From<LSQError> for PyErr {
    fn from(_error: LSQError) -> Self {
        PyValueError::new_err("LSQError")
    }
}


impl fmt::Display for LSQError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<argmin::core::Error> for LSQError {
    fn from(value: argmin::core::Error) -> Self {
        // Self(value)
        LSQError(value)
    }
}
