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
}
