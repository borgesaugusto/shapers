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
Currently, the exposed functions are:
| Function | Description |
| --- | --- |
| `fit_geometrical(x_values: Vec<f64>, y_values: Vec<f64>)` | Calculate the geometrical mean |
| `taubin_svd(x_values: Vec<f64>, y_values: Vec<f64>)` | Fit a circle using Taubin SVD |
| `fit_lsq(x_values: Vec<f64>, y_values: Vec<f64>)` | Fit a circle using least squares |


### Python example:
```python
import shapers as shs

# create example circle
circle_center = [5, 5]
circle_radius = 5
n_points = 50
theta = np.linspace(0, 2 * np.pi, n_points)
arrx = circle_center[0] + (circle_radius * np.random.normal(1, 0.1)) * np.cos(theta)
arry = circle_center[1] + (circle_radius * np.random.normal(1, 0.1)) * np.sin(theta)

# fit circle
x_center, y_center = shs.taubin_svd(arrx, arry)
# x_center, y_center = shs.fit_geometrical(arrx, arry)  # alternatively
# x_center, y_center = shs.fit_lsq(arrx, arry)  # alternatively
```

It is possible to modify the parameters of the algorithm through the `FitCircleParams` class in the following wway:
```python
# fit circle
parameters = shs.FitCircleParams()
parameters.method = "lbfgs"
parameters.precision = 1e-4
parameters.max_iterations = 1000
x_center, y_center = shs.fit_lsq(arrx, arry, parameters)
```
Each method might have specific parameters. For more information, please refer to the documentation of the method.

## Ellipsoid superposition
To determine if two ellipsoids are superimposed, you can use
`check_ellipsoid_intersection(ellipse_a: Ellipsoid, ellipse_b: Ellipsoid, parameters: Option<EllipsoidIntersectionParameters>`. 
### Python example
```python 
import shapers as shs

# create example Ellipsoids
eli1 = shs.Ellipsoid(0, 0, 2, 1, 0)
eli2 = shs.Ellipsoid(4, 0, 2, 1, 0)

# check intersection
intersection = shs.check_ellipsoid_intersection(eli1, eli2)
```
If the value of `intersection` is positive, the ellipsoids are superimposed. If it is zero, they are only touching at one point. If it is negative, they are not touching at all.

To modify the parameters, you can use the `EllipsoidIntersectionParameters` class. For example:
```python
# check intersection
parameters = shs.EllipsoidIntersectionParameters()
parameters.tolerance = 1e-4
parameters.max_iters = 1000
intersection = shs.check_ellipsoid_intersection(eli1, eli2, parameters)
```
### Rust example
```rust
// create example Ellipsoids
let ellipse1 = Ellipsoid::new(0.0, 0.0, 2.0, 1.0, 0.0);
let ellipse2 = Ellipsoid::new(4.0, 0.0, 2.0, 1.0, 0.0);

// check intersection
let parameters = ellipsoids::EllipsoidIntersectionParameters::new()
    .with_tolerance(1e-6)
    .with_max_iters(100);
let intersection = ellipsoids::check_ellipsoid_intersection(ellipse1, ellipse2, Some(parameters));
```


