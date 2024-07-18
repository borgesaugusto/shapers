use crate::circle_fit::Circle;

pub fn get_distance(v0: Vec<f64>, v1: Vec<f64>) -> f64 {
    let x = v0[0] - v1[0];
    let y = v0[1] - v1[1];
    (x * x + y * y).sqrt()
}


pub fn get_circle_centroid(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    let x = xs.iter().sum::<f64>() / xs.len() as f64;
    let y = ys.iter().sum::<f64>() / ys.len() as f64;
    vec![x, y]
}


pub fn mean_distance_to_center(center: Vec<f64>, circle: &Circle) -> f64 {
    let xs = &circle.xs;
    let ys = &circle.ys;
    let distances: Vec<f64> = xs.iter().zip(ys.iter()).map(|(x, y)| get_distance(vec![*x, *y], center.clone())).collect();
    distances.iter().sum::<f64>() / distances.len() as f64
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_distance() {
        let v0 = vec![0.0, 0.0];
        let v1 = vec![3.0, 4.0];
        assert_eq!(get_distance(v0, v1), 5.0);
    }

    #[test]
    fn test_get_circle_centroid() {
        let xs = vec![-1.0, 0.0, 1.0, 0.0];
        let ys = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(get_circle_centroid(xs, ys), vec![0.0, 0.0]);
    }

    #[test] 
    fn test_mean_distance_to_center() {
        let xs = vec![-1.0, 0.0, 1.0];
        let ys = vec![0.0, 1.0, 0.0];
        let circle = Circle { xs: xs.clone(), ys: ys.clone() };
        let center = vec![0.0, 0.0];
        assert_eq!(mean_distance_to_center(center, &circle), 1.0);
    }
}
