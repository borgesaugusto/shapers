[![Crates.io Version](https://img.shields.io/crates/v/shapers)](https://crates.io/crates/shapers)
[![PyPI - Version](https://img.shields.io/pypi/v/shapers)](https://pypi.org/project/shapers/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/shapers)](https://pypi.org/project/shapers/)
# Shapers
Fitting shapes with Rust in python

Currently, only TaubinSVD and geometrical fitting are implemented. 

This package is based on [circle-fit](https://github.com/AlliedToasters/circle-fit/). We are incorporating functions from the original [work](https://people.cas.uab.edu/~mosya/cl/MATLABcircle.html) by Nikolai Chernov.


### How to install
The software is available on [crates.io](https://crates.io/crates/shapers) or in [PyPi](https://pypi.org/project/shapers/).
#### Python
To install in python run
```bash
pip install shapers
```

#### Rust
To install in Rust, add the following to your `Cargo.toml` file
```toml
[dependencies]
shapers = "0.3.0"
```

## Circle fitting 
### How to use 
Currently, to the exposed functions are
| Function | Description |
| --- | --- |
| `fit_geometrical(x_values: Vec<f64>, y_values: Vec<f64>)` | Give the average value of the points provided|
| `taubin_svd(x_values: Vec<f64>, y_values: Vec<f64>)` | Fit a shape to a set of points using Taubin SVD |
| `fit_lsq(x_values: Vec<f64>, y_values: Vec<f64>)` | Fit a shape to a set of points using least squares |


#### Python example:
```python
import shapers as shrs

# create an artificial circle as an example
circle_center = [5, 5]
circle_radius = 5
n_points = 50
theta = np.linspace(0, 2 * np.pi, n_points)
arrx = circle_center[0] + (circle_radius * np.random.normal(1, 0.1)) * np.cos(theta)
arry = circle_center[1] + (circle_radius * np.random.normal(1, 0.1)) * np.sin(theta)
###
x_center, y_center = shrs.taubin_svd(arrx, arry)
# x_center, y_center = shrs.fit_geometrical(arrx, arry)
# x_center, y_center = shrs.fit_lsq(arrx, arry)
```

It is possible to modify the parameters of the algorithm through the `FitCircleParams` class. To do it, you can
```python
import shapers as shrs

# arrx and arry have the x and y coordiantes of the circle's boundary
parameters = shrs.FitCircleParams()
parameters.method = "lbfgs"
parameters.precision = 1e-4
parameters.max_iterations = 1000
x_center, y_center = shrs.fit_lsq(arrx, arry, parameters)
```
Each method might have specific parameters. For more information, please refer to the documentation of the method.

## Ellipsoid superposition
To determine if two ellipsoids are superimposed, you can use
`check_ellipsoid_intersection(ellipse_a: Ellipsoid, ellipse_b: Ellipsoid, parameters: Option<EllipsoidIntersectionParameters>`. 
### Python example
In python, you can use the following code: 
```python 
eli1 = shrs.Ellipsoid(0, 0, 2, 1, 0)
eli2 = shrs.Ellipsoid(4, 0, 2, 1, 0)

parameters = shrs.EllipsoidIntersectionParameters()

intersection = shrs.check_ellipsoid_intersection(eli1, eli2, parameters)
```
To modify the parameters, you can use the `EllipsoidIntersectionParameters` class. For example:
```python
parameters = shrs.EllipsoidIntersectionParameters()
parameters.tolerance = 1e-4
parameters.max_iters = 1000
```
### Rust example
In Rust, you can use the following code:
```rust
let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
let ellipse2 = Ellipsoid::new(4.0, 0.0, 2.0, 1.0, 0.0);

let parameters = ellipsoids::EllipsoidIntersectionParameters::new()
    .with_tolerance(1e-6)
    .with_max_iters(100);
let intersection = ellipsoids::check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
```

If the value of `intersection` is bigger than zero, the ellipsoids are superimposed. If it is zero, they are only touching at one point.

