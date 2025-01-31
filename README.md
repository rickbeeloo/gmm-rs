# Gaussian Mixture Model

A Rust implementation of Gaussian Mixture Models (GMM) using the EM algorithm. This is a port of the [rusty-machine GMM implementation](https://athemathmo.github.io/rusty-machine/doc/src/rusty_machine/src/learning/gmm.rs.html#1-382), but updated to use [ndarray](https://docs.rs/ndarray/latest/ndarray/) instead of the deprecated rusty-machine matrix operations.

## Overview

This library provides an implementation of Gaussian Mixture Models for unsupervised learning tasks. While not extensively tested, it aims to maintain feature parity with the original rusty-machine implementation.

## Usage

Here's a basic example of how to use the GMM implementation:

```rust
use ndarray::Array2;
use gmm_rs::{GaussianMixtureModel, CovOption};

fn main() {
// Create sample data
let data = Array2::from_shape_vec((4, 2), vec![
    1.0, 2.0,
    -3.0, -3.0,
    0.1, 1.5,
    -5.0, -2.5,
    ]).unwrap();
// Create test data
let test_data = Array2::from_shape_vec((3, 2), vec![
    1.0, 2.0,
    3.0, 2.9,
    -4.4, -2.5,
]).unwrap();

// Create GMM with k(=2) components
let mut model = GaussianMixtureModel::new(2);
model.set_max_iters(10);
model.cov_option = CovOption::Diagonal;

// Train the model
model.train(&data).unwrap();

// Print the means and covariances
println!("Means: {:?}", model.means());
println!("Covariances: {:?}", model.covariances());

// Get posterior probabilities for test data
let post_probs = model.predict(&test_data).unwrap();
println!("Posterior probabilities: {:?}", post_probs);
}
```

## Features

- Configurable number of components (k)
- Multiple covariance options (Full, Diagonal, Regularized)
- Configurable maximum iterations for EM algorithm
- Support for custom mixture weights

## Dependencies

- ndarray: For matrix and vector operations

## License
[MIT](LICENSE)