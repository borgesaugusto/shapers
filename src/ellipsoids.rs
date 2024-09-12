use ndarray::{Array2, arr1, stack, Axis, ArrayBase, OwnedRepr, Dim, array};
use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::{goldensectionsearch::GoldenSectionSearch},
}; 
// use argmin_observer_slog::SlogLogger;
use crate::errors::LSQError;
use pyo3::prelude::*;
use ndarray_linalg::Inverse;

#[derive(Debug, Clone)]
#[pyclass]
/// Ellipsoid struct
pub struct Ellipsoid {
    #[pyo3(get, set)]
    x: f64,
    #[pyo3(get, set)]
    y: f64,
    #[pyo3(get, set)]
    major_axis: f64,
    #[pyo3(get, set)]
    minor_axis: f64,
    #[pyo3(get, set)]
    angle: f64,
}

#[pymethods]
impl Ellipsoid {
    #[new]
    pub fn py_new(x: f64, y: f64, major_axis: f64, minor_axis: f64, angle: f64) -> Self {
        Ellipsoid {
            x,
            y,
            major_axis,
            minor_axis,
            angle
        }
    }

}

impl Ellipsoid {
    /// Build a new Ellipsoid
    pub fn new(x: f64, y: f64, major_axis: f64, minor_axis: f64, angle: f64) -> Self {
    Ellipsoid {
            x,
            y,
            major_axis,
            minor_axis,
            angle
        }
    }
    /// Get eigenvectors of the ellipsoid
    pub fn get_eigenvectors(&self) -> ([f64; 2], [f64; 2]){
        let fist_eigen = [self.angle.cos(), self.angle.sin()];
        let second_eigen = [-self.angle.sin(), self.angle.cos()];
        (fist_eigen, second_eigen)
    }
    

    /// Get eigenvalues of the ellipsoid
    pub fn get_eigenvalues(&self) -> (f64, f64) {
        let first_eigen = 1.0 / self.major_axis.powi(2);
        let second_eigen = 1.0 / self.minor_axis.powi(2);
        (first_eigen, second_eigen)
    }

    /// Get the matrix representation of the ellipsoid
    /// The representation is a 2x2 matrix given by UDU^T, where U is the matrix of eigenvectors and D is a diagonal matrix with the eigenvalues
    pub fn get_matrix_representation(&self) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>{
        let eigenvectors = self.get_eigenvectors();
        let eigen_matrix = stack![Axis(0), arr1(&eigenvectors.0), arr1(&eigenvectors.1)];
        let eigenvalues = self.get_eigenvalues();

        let diag_matrix = Array2::from_diag(&arr1(&[eigenvalues.0, eigenvalues.1]));
        let ellipse_representation = eigen_matrix.t().dot(&diag_matrix).dot(&eigen_matrix); 
        
        ellipse_representation
    }
}
/// Ellipsoid intersection struct to define the cost function
pub struct EllipsoidIntersection {
    ellipse_a: Ellipsoid,
    ellipse_b: Ellipsoid,
}

impl EllipsoidIntersection {
    /// Build a new EllipsoidIntersection
    pub fn new(ellipse_a: Ellipsoid, ellipse_b: Ellipsoid) -> Self {
        EllipsoidIntersection {
            ellipse_a,
            ellipse_b,
        }
    }
    
}

impl CostFunction for EllipsoidIntersection {
    type Param = f64;
    type Output = f64;
    
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let matrix_a = self.ellipse_a.get_matrix_representation();
        let matrix_b = self.ellipse_b.get_matrix_representation();
        
        let subst_centers = array![self.ellipse_b.x - self.ellipse_a.x,
            self.ellipse_b.y - self.ellipse_a.y];
        
        let inv_a = match matrix_a.inv() {
            Ok(inv) => inv / (1.0 - param),
            Err(e) => {
                eprintln!("Error inverting matrix A: {:?}", e);
                return Ok(f64::INFINITY);
            }
            
        };
        let inv_b = match matrix_b.inv() {
            Ok(inv) => inv / (param - 0.0),
            Err(e) => {
                eprint!("Error inverting matrix B: {:?}", e);
                return Ok(f64::INFINITY);
            }
            
        };
        let inv_a_plus_b = (inv_a + inv_b).inv().unwrap();

        
        let k_of_s = 1.0 - subst_centers.t().dot(&inv_a_plus_b).dot(&subst_centers);
        Ok(k_of_s)
    }
}
#[derive(Debug, Clone)]
#[pyclass]
/// Parameters for the ellipsoid EllipsoidIntersection
/// In the future, this struct would be unique for all optimizations
pub struct EllipsoidIntersectionParameters {
    #[pyo3(get, set)]
    /// Tolerance for the optimization
    pub tolerance: f64,
    /// Maximum number of iterations
    #[pyo3(get, set)]
    pub max_iters: u64,
}

#[pymethods]
impl EllipsoidIntersectionParameters {
    #[new]
    pub fn py_new() -> Self {
        EllipsoidIntersectionParameters::new()
    }
}


impl EllipsoidIntersectionParameters {
    /// Build a new EllipsoidIntersectionParameters
    pub fn new() -> Self {
        EllipsoidIntersectionParameters {
            tolerance: 1e-4,
            max_iters: 1000,
        }
    }
    /// Set the tolerance for the optimization
    pub fn with_tolerance(mut self, precision: f64) -> Self {
        self.tolerance = precision;
        self
    }
    /// Set the maximum number of iterations for the optimization
    pub fn with_max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }
}

/// Check the intersection between two ellipses
/// Takes two ellipses and returns the intersection parameter K. If K<0, there is no intersection 
/// This algorithm was based on:
/// *I. Gilitschenski and U. D. Hanebeck, "A robust computational test for overlap of two arbitrary-dimensional ellipsoids in fault-detection of Kalman filters," 2012 15th International Conference on Information Fusion, Singapore, 2012, pp. 396-401.*
/// # Examples
/// ```
/// use shapers::ellipsoids::{Ellipsoid, EllipsoidIntersectionParameters, check_ellipsoid_intersection};
/// let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
/// let ellipse2 = Ellipsoid::new(4.0, 0.0, 2.0, 1.0, 0.0);
/// let parameters = EllipsoidIntersectionParameters::new();
/// let intersection = check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters)); 
/// assert_eq!(intersection.unwrap(), 0.0);
/// ```
#[pyfunction]
#[pyo3(signature = (ellipse_a, ellipse_b, parameters=None))]
pub fn check_ellipsoid_intersection(ellipse_a: Ellipsoid, ellipse_b: Ellipsoid, parameters: Option<EllipsoidIntersectionParameters>) -> Result<f64, LSQError> { 
    let parameters = match parameters {
        Some(p) => p,
        None => EllipsoidIntersectionParameters::new(),
    };
    let ellipse_intersection = EllipsoidIntersection::new(ellipse_a, ellipse_b);
    // let line_search = MoreThuenteLineSearch::new().with_c(parameters.tolerance, 0.9)?;
    let init_param = 0.5;
    let solver = GoldenSectionSearch::new(0.0, 1.0)?.with_tolerance(parameters.tolerance)?;
    let result = Executor::new(ellipse_intersection, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    Ok(result.state.get_best_cost())

}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_not_touching() {
        let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
        let ellipse2 = Ellipsoid::new(10.0, 0.0, 2.0, 1.0, 0.0);
        let parameters = EllipsoidIntersectionParameters::new();
        let intersection = check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
        assert_eq!(intersection.unwrap(), -5.25);
    }
    #[test]
    fn test_not_touching_default_params() {
        let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
        let ellipse2 = Ellipsoid::new(10.0, 0.0, 2.0, 1.0, 0.0);
        let intersection = check_ellipsoid_intersection(ellipse1, ellipse2, None);
        assert_eq!(intersection.unwrap(), -5.25);
    }
    #[test]
    fn test_just_touching() {
        let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
        let ellipse2 = Ellipsoid::new(4.0, 0.0, 2.0, 1.0, 0.0);
        let parameters = EllipsoidIntersectionParameters::new();
        let intersection = check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
        assert_eq!(intersection.unwrap(), 0.0);
    }
    #[test]
    fn test_superposition() {
        let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
        let ellipse2 = Ellipsoid::new(2.0, 0.0, 2.0, 1.0, 0.0);
        let parameters = EllipsoidIntersectionParameters::new();
        let intersection = check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
        assert_eq!(intersection.unwrap(), 0.75);
    }
}
