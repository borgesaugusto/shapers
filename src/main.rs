mod aux_funcs;
mod circle_fit;
extern crate blas_src;

fn main() {
    println!("Executing circular fit...");
    let xs = vec![0.0, 4.0, 2.0, 2.0];
    let ys = vec![0.0, 0.0, 2.0, -2.0];
    // _ = circle_fit::fit_lsq(xs, ys);
    let center_taubin = circle_fit::taubin_svd(xs.clone(), ys.clone());

    println!("Geometrical center: {:?}", aux_funcs::get_circle_centroid(xs.clone(), ys.clone()));
    println!("Center: {:?}", center_taubin);
}
