use shapers::{self, circle_fit::FitCircleParams};
fn main() {
    let n_points = 10000;
    let all_angles = (0..n_points).map(|i| 2.0 * std::f64::consts::PI * (i as f64) / n_points as f64).collect::<Vec<f64>>();
    let circle_radius = 5.0;
    let _std = 0.0;
    // let xs = vec![0.0, 4.0, 2.0, 2.0];
    // let ys = vec![0.0, 0.0, 2.0, -2.0];
    
    let circle_center = vec![80.2, 55.3];
    let xs = all_angles.clone().iter().map(|theta| circle_center[0] + circle_radius * theta.cos()).collect::<Vec<f64>>();
    let ys = all_angles.iter().map(|theta| circle_center[1] + circle_radius * theta.sin()).collect::<Vec<f64>>();
    let xs_test = vec![139.81, 136.206, 132.556, 128.81, 124.997, 130.664, 132.686, 130.75, 125.445, 126.669, 127.873, 129.05, 130.189, 34.576, 139.013, 143.478, 147.965, 149.626, 150.972, 152.002, 152.707, 149.991, 146.86, 143.416];
    let ys_test = vec![184.751, 185.263, 185.53, 185.551, 185.298, 179.939, 172.44, 164.949, 159.338, 158.73, 158.051, 157.299, 156.48, 158.279, 159.905, 161.396, 162.776, 166.452, 170.253, 174.151, 178.12, 180.558, 182.487, 183.883];

    // arrx = circle_center[0] + (circle_radius * np.random.normal(1, std)) * np.cos(theta)
    // arry = circle_center[1] + (circle_radius * np.random.normal(1, std)) * np.sin(theta)
    // let center_taubin = circle_fit::taubin_svd(xs.clone(), ys.clone());
    println!("Executing circular fit...");
    // let center_lsq = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(), Some("lbfgs"));

    // println!("Geometrical center: {:?}", aux_funcs::get_circle_centroid(xs.clone(), ys.clone()));
    // println!("Center: {:?}", center_taubin);
    println!("Real center: {:?}", circle_center);
    // println!("Geometrical center: {:?}", shapers::circle_fit::fit_geometrical(xs, ys));
    // let parameters = FitCircleParams::new().with_n_vertices(50);
    let parameters = FitCircleParams::new()
        .with_precision(1e-6)
        .with_max_iters(100)
        .with_method("lbfgs");

    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(),Some(parameters.clone()));
    let nm_center = shapers::circle_fit::fit_lsq(xs_test.clone(), ys_test.clone(),Some(parameters.clone()));
    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(),Some(parameters.clone()));
    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(), None);
    println!("Geometric center: {:?}", shapers::circle_fit::fit_geometrical(xs_test, ys_test));
    println!("Calculated center: {:?}", nm_center);
    // println!("LSQ Center: {:?}", center_lsq);
}
