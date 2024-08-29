use shapers;
fn main() {
    let n_points = 10000;
    let all_angles = (0..n_points).map(|i| 2.0 * std::f64::consts::PI * (i as f64) / n_points as f64).collect::<Vec<f64>>();
    let circle_radius = 5.0;
    let std = 0.0;
    // let xs = vec![0.0, 4.0, 2.0, 2.0];
    // let ys = vec![0.0, 0.0, 2.0, -2.0];
    
    let circle_center = vec![80.2, 55.3];
    let xs = all_angles.clone().iter().map(|theta| circle_center[0] + circle_radius * theta.cos()).collect::<Vec<f64>>();
    let ys = all_angles.iter().map(|theta| circle_center[1] + circle_radius * theta.sin()).collect::<Vec<f64>>();
    // arrx = circle_center[0] + (circle_radius * np.random.normal(1, std)) * np.cos(theta)
    // arry = circle_center[1] + (circle_radius * np.random.normal(1, std)) * np.sin(theta)
    // let center_taubin = circle_fit::taubin_svd(xs.clone(), ys.clone());
    println!("Executing circular fit...");
    let center_lsq = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(), Some("lbfgs"));

    // println!("Geometrical center: {:?}", aux_funcs::get_circle_centroid(xs.clone(), ys.clone()));
    // println!("Center: {:?}", center_taubin);
    println!("Real center: {:?}", circle_center);
    println!("Geometrical center: {:?}", shapers::circle_fit::fit_geometrical(xs, ys));
    println!("LSQ Center: {:?}", center_lsq);
}
