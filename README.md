[![Crates.io Version](https://img.shields.io/crates/v/shapers)(https://crates.io/crates/shapers)]
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
shapers = "0.2.0"
```


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

# create an artificial circle as an example
circle_center = [5, 5]
circle_radius = 5
n_points = 50
theta = np.linspace(0, 2 * np.pi, n_points)
arrx = circle_center[0] + (circle_radius * np.random.normal(1, 0.1)) * np.cos(theta)
arry = circle_center[1] + (circle_radius * np.random.normal(1, 0.1)) * np.sin(theta)
### 
parameters = shrs.FitCircleParams()
parameters.method = "lbfgs"
parameters.precision = 1e-4
parameters.max_iterations = 1000
x_center, y_center = shrs.fit_lsq(arrx, arry, parameters)
```
Each method might have specific parameters. For more information, please refer to the documentation of the method.
