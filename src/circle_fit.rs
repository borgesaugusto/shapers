use crate::{aux_funcs::{get_distance, get_circle_centroid, self}, errors::LSQError};
use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, State},
    solver::neldermead::NelderMead,
}; 
use ndarray::{s, array, Array1, Array, stack, Axis, concatenate};
use ndarray_linalg::{SVD};
use argmin_observer_slog::SlogLogger;
use pyo3::prelude::*;

pub struct Circle {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,

}

impl Circle {
    fn mean_distance_to_center(&self, center: Vec<f64>) -> f64 {
        let xs = &self.xs;
        let ys = &self.ys;
        let distances: Vec<f64> = xs.iter().zip(ys.iter()).map(|(x, y)| get_distance(vec![*x, *y], center.clone())).collect();
        distances.iter().sum::<f64>() / distances.len() as f64
    }
}

impl CostFunction for Circle {
    type Param = Vec<f64>;
    type Output = f64;
    
    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let center = vec![params[0], params[1]];
        Ok(self.mean_distance_to_center(center))
    }
}

#[pyfunction]
pub fn fit_geometrical(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    aux_funcs::get_circle_centroid(xs, ys)
}

#[pyfunction]
pub fn fit_lsq(xs: Vec<f64>, ys: Vec<f64>) -> Result<Vec<f64>, LSQError> {
    if xs.len() != ys.len() {
        eprint!("The number of x and y points must be the same.");
    }
    
    if xs.len() > 10000 {
        eprint!("The number of points is too high. Might lead to performance issues");
    }


    let circle = Circle { xs: xs.clone(), ys: ys.clone() };
    let mut simplicial_vertices = vec![];
    let ave_distance = circle.mean_distance_to_center(aux_funcs::get_circle_centroid(xs.clone(), ys.clone()));
    let geom_center = aux_funcs::get_circle_centroid(xs.clone(), ys.clone());
    let n_vertices = 20;
    for i in 0..n_vertices {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / n_vertices as f64;
        let x = geom_center[0] + ave_distance * 5.0 * (angle as f64).cos();
        let y = geom_center[1] + ave_distance * 5.0 * (angle as f64).sin();
        simplicial_vertices.push(vec![x, y]);
    }    

    let nm_solver = NelderMead::new(simplicial_vertices)
                                .with_sd_tolerance(1e-15_f64)?;
    let result = Executor::new(circle, nm_solver)
        .configure(|state| state.max_iters(1000))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run();

    let final_result = result.unwrap();
    return Ok(final_result.state.get_best_param().expect("Found params").to_vec());
}

#[pyfunction]
pub fn taubin_svd(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    let c0 = get_circle_centroid(xs.clone(), ys.clone());
    let x_rel_center = xs.iter().map(|x| x - c0[0]).collect::<Vec<f64>>();
    let y_rel_center = ys.iter().map(|y| y - c0[1]).collect::<Vec<f64>>();
    
    let z_value = x_rel_center.iter().zip(y_rel_center.iter()).map(|(x, y)| x * x + y * y).collect::<Vec<f64>>();
    let z_mean = z_value.iter().sum::<f64>() / z_value.len() as f64;
    let z0 = z_value.iter().map(|z_value_i| (z_value_i - z_mean) / (2.0 * z_mean.sqrt())).collect::<Vec<f64>>();
    let x_matrix = Array1::from(x_rel_center.clone());
    let y_matrix = Array1::from(y_rel_center.clone());
    let z_matrix = Array1::from(z0.clone());

    let zxy_matrix = stack![Axis(0), z_matrix, x_matrix, y_matrix].t().to_owned();
    
    let (_, _, vt) = SVD::svd(&zxy_matrix, false, true).unwrap();
    
    let mut v = vt.unwrap().t().to_owned();
    let mut second_column_v = v.column_mut(2);
    second_column_v[0] = second_column_v[0] / 2.0 * z_mean.sqrt();
    let a_matrix = concatenate![Axis(0), second_column_v,  array![- 1.0 * z_mean * second_column_v[0]]];

    let middle_matrix = a_matrix.slice(s![1..3]).mapv(|x| -2.0 * x / a_matrix[0]) + Array::from_vec(c0);

    return vec![middle_matrix[0], middle_matrix[1]];
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fit_lsq() {
        let precision: f64 = 100000000.0;
        let xs = vec![-1.0, 0.0, 1.0, 0.0];
        let ys = vec![0.0, 1.0, 0.0, -1.0];
        let mut res = fit_lsq(xs, ys);
        res = res.map(|x| x.iter().map(|y| ( y.round() * precision ) / precision).collect());
        assert_eq!(res.unwrap(), vec![0.0, 0.0]);
    }
    #[test]
    fn test_fit_taubin() {
        let xs = vec![0.0, 4.0, 2.0, 2.0];
        let ys = vec![0.0, 0.0, 2.0, -2.0];
        assert_eq!(taubin_svd(xs, ys), vec![2.0, 0.0]);
    }
    #[test]
    fn test_fit_geometrical() {
        let xs = vec![-1.0, 0.0, 1.0, 0.0];
        let ys = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(fit_geometrical(xs, ys), vec![0.0, 0.0]);
    }
}
