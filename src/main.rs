use shapers::{self, circle_fit::FitCircleParams};
use shapers::ellipsoids::{Ellipsoid, self};

fn main() {
    // Lets try with intersecting ellipses
    // Intersection must be zero
    let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
    let ellipse2 = Ellipsoid::new(10.0, 0.0, 2.0, 1.0, 0.0);
    // just intersected
    let ellipse3 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
    let ellipse4 = Ellipsoid::new(2.0, 0.0, 2.0, 1.0, 0.0);
    
    dbg!(ellipse1.get_matrix_representation());
    dbg!(ellipse2.get_matrix_representation());

    let parameters = ellipsoids::EllipsoidIntersectionParameters::new()
        .with_tolerance(1e-6)
        .with_max_iters(100);
    let intersection = ellipsoids::check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
    dbg!(intersection);
}


fn test_lsq() {

    println!("Executing circular fit...");
    let parameters = FitCircleParams::new()
        .with_precision(1e-4)
        .with_max_iters(100)
        .with_method("lbfgs");

    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(),Some(parameters.clone()));
    // let nm_center = shapers::circle_fit::fit_lsq(xs_test.clone(), ys_test.clone(),Some(parameters.clone()));
    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(),Some(parameters.clone()));
    // let nm_center = shapers::circle_fit::fit_lsq(xs.clone(), ys.clone(), None);
    // println!("Geometric center: {:?}", shapers::circle_fit::fit_geometrical(xs_test, ys_test));
    // println!("Calculated center: {:?}", nm_center);
    // println!("LSQ Center: {:?}", center_lsq);
}


fn test_example_circle() {
    let n_points = 10000;
    let all_angles = (0..n_points).map(|i| 2.0 * std::f64::consts::PI * (i as f64) / n_points as f64).collect::<Vec<f64>>();
    let circle_radius = 5.0;
    let _std = 0.0;
    
    // arrx = circle_center[0] + (circle_radius * np.random.normal(1, std)) * np.cos(theta)
    // arry = circle_center[1] + (circle_radius * np.random.normal(1, std)) * np.sin(theta)
    // let center_taubin = circle_fit::taubin_svd(xs.clone(), ys.clone());
    let circle_center = vec![80.2, 55.3];
    let xs = all_angles.clone().iter().map(|theta| circle_center[0] + circle_radius * theta.cos()).collect::<Vec<f64>>();
    let ys = all_angles.iter().map(|theta| circle_center[1] + circle_radius * theta.sin()).collect::<Vec<f64>>();

    println!("Real center: {:?}", circle_center);
}

fn test_example_small() {
    let xs_test = vec![139.81, 136.206, 132.556, 128.81, 124.997, 130.664, 132.686, 130.75, 125.445, 126.669, 127.873, 129.05, 130.189, 34.576, 139.013, 143.478, 147.965, 149.626, 150.972, 152.002, 152.707, 149.991, 146.86, 143.416];
    let ys_test = vec![184.751, 185.263, 185.53, 185.551, 185.298, 179.939, 172.44, 164.949, 159.338, 158.73, 158.051, 157.299, 156.48, 158.279, 159.905, 161.396, 162.776, 166.452, 170.253, 174.151, 178.12, 180.558, 182.487, 183.883];


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

}
