use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Clone, Copy, Debug)]
pub enum CovOption {
    Full,
    Regularized(f64),
    Diagonal,
}


#[derive(Debug)]
pub struct GaussianMixtureModel {
    comp_count: usize,
    mix_weights: Array1<f64>,
    model_means: Option<Array2<f64>>,
    model_covars: Option<Vec<Array2<f64>>>,
    log_lik: f64,
    max_iters: usize,
    pub cov_option: CovOption,
}

impl GaussianMixtureModel {
    pub fn new(k: usize) -> GaussianMixtureModel {
        GaussianMixtureModel {
            comp_count: k,
            mix_weights: Array1::ones(k) / (k as f64),
            model_means: None,
            model_covars: None,
            log_lik: 0f64,
            max_iters: 100,
            cov_option: CovOption::Full,
        }
    }

    pub fn with_weights(k: usize, mixture_weights: Array1<f64>) -> Result<GaussianMixtureModel, &'static str> {
        if mixture_weights.len() != k {
            return Err("Mixture weights must have length k.");
        } else if mixture_weights.iter().any(|&x| x < 0f64) {
            return Err("Mixture weights must have only non-negative entries.");
        }
        
        let sum = mixture_weights.sum();
        let normalized_weights = &mixture_weights / sum;

        Ok(GaussianMixtureModel {
            comp_count: k,
            mix_weights: normalized_weights,
            model_means: None,
            model_covars: None,
            log_lik: 0f64,
            max_iters: 100,
            cov_option: CovOption::Full,
        })
    }

    pub fn means(&self) -> Option<&Array2<f64>> {
        self.model_means.as_ref()
    }

    pub fn covariances(&self) -> Option<&Vec<Array2<f64>>> {
        self.model_covars.as_ref()
    }

    pub fn mixture_weights(&self) -> &Array1<f64> {
        &self.mix_weights
    }

    pub fn set_max_iters(&mut self, iters: usize) {
        self.max_iters = iters;
    }

    pub fn train(&mut self, inputs: &Array2<f64>) -> Result<(), &'static str> {
        if inputs.nrows() <= 1 {
            return Err("Only one row of data provided.");
        }

        let reg_value = 1f64 / (inputs.nrows() - 1) as f64;
        let k = self.comp_count;

        // Initialize covariances
        self.model_covars = {
            let cov_mat = self.initialize_covariances(inputs, reg_value)?;
            Some(vec![cov_mat; k])
        };

        // Initialize means by randomly selecting k rows from inputs
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..inputs.nrows()).collect();
        indices.shuffle(&mut rng);
        let random_rows = &indices[..k];
        
        self.model_means = Some(Array2::from_shape_fn((k, inputs.ncols()), |(i, j)| {
            inputs[[random_rows[i], j]]
        }));

        // EM Algorithm
        for _ in 0..self.max_iters {
            let log_lik_0 = self.log_lik;
            let (weights, log_lik_1) = self.membership_weights(inputs)?;

            if (log_lik_1 - log_lik_0).abs() < 1e-15 {
                break;
            }

            self.log_lik = log_lik_1;
            self.update_params(inputs, &weights);
        }

        Ok(())
    }

    pub fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
        if self.model_means.is_none() || self.model_covars.is_none() {
            return Err("Model not trained");
        }
        Ok(self.membership_weights(inputs)?.0)
    }

    fn initialize_covariances(&self, inputs: &Array2<f64>, reg_value: f64) -> Result<Array2<f64>, &'static str> {
        match self.cov_option {
            CovOption::Diagonal => {
                let variance = inputs.var_axis(Axis(0), 1.0);
                Ok(Array2::from_diag(&variance) * reg_value.sqrt())
            }
            CovOption::Full | CovOption::Regularized(_) => {
                let means = inputs.mean_axis(Axis(0)).unwrap();
                let d = inputs.ncols();
                let mut cov_mat = Array2::zeros((d, d));

                for i in 0..d {
                    for j in 0..d {
                        let mut sum = 0.0;
                        for row in inputs.rows() {
                            sum += (row[i] - means[i]) * (row[j] - means[j]);
                        }
                        cov_mat[[i, j]] = sum * reg_value;
                    }
                }

                if let CovOption::Regularized(eps) = self.cov_option {
                    for i in 0..d {
                        cov_mat[[i, i]] += eps;
                    }
                }

                Ok(cov_mat)
            }
        }
    }

    fn membership_weights(&self, inputs: &Array2<f64>) -> Result<(Array2<f64>, f64), &'static str> {
        let n = inputs.nrows();
        let mut weights = Array2::zeros((n, self.comp_count));
        let mut log_lik = 0f64;

        if let (Some(means), Some(covars)) = (&self.model_means, &self.model_covars) {
            for i in 0..n {
                let x_i = inputs.row(i);
                let mut pdfs = Vec::with_capacity(self.comp_count);

                for (j, (mu_j, cov_j)) in means.rows().into_iter().zip(covars.iter()).enumerate() {
                    let diff = &x_i - &mu_j;
                    let cov_det = self.compute_determinant(cov_j)?;
                    let cov_inv = self.compute_inverse(cov_j)?;

                    let exponent = -0.5 * diff.dot(&cov_inv.dot(&diff));
                    let pdf = exponent.exp() / (cov_det.sqrt());
                    pdfs.push(pdf);
                }

                let weighted_pdfs: Vec<f64> = pdfs.iter()
                    .zip(self.mix_weights.iter())
                    .map(|(&pdf, &w)| w * pdf)
                    .collect();
                let weighted_sum: f64 = weighted_pdfs.iter().sum();

                for (j, &pdf) in pdfs.iter().enumerate() {
                    weights[[i, j]] = self.mix_weights[j] * pdf / weighted_sum;
                }

                log_lik += weighted_sum.ln();
            }
        }

        Ok((weights, log_lik))
    }

    fn update_params(&mut self, inputs: &Array2<f64>, weights: &Array2<f64>) {
        let n = weights.nrows();
        let d = inputs.ncols();
        
        // Update mixture weights
        self.mix_weights = weights.sum_axis(Axis(0)) / (n as f64);

        // Update means
        let mut new_means = Array2::zeros((self.comp_count, d));
        for k in 0..self.comp_count {
            let w_k = weights.column(k);
            for j in 0..d {
                new_means[[k, j]] = inputs.column(j).dot(&w_k) / w_k.sum();
            }
        }

        // Update covariances
        let mut new_covs = Vec::with_capacity(self.comp_count);
        for k in 0..self.comp_count {
            let mut cov_k = Array2::zeros((d, d));
            let w_k = weights.column(k);
            let mu_k = new_means.row(k);

            for i in 0..n {
                let x_i = inputs.row(i);
                let diff = &x_i - &mu_k;
                let weight = w_k[i];
                cov_k = cov_k + self.compute_cov(diff.view(), weight);
            }

            cov_k = &cov_k / w_k.sum();
            
            if let CovOption::Regularized(eps) = self.cov_option {
                for i in 0..d {
                    cov_k[[i, i]] += eps;
                }
            }

            new_covs.push(cov_k);
        }

        self.model_means = Some(new_means);
        self.model_covars = Some(new_covs);
    }

    fn compute_cov(&self, diff: ArrayView1<f64>, weight: f64) -> Array2<f64> {
        match self.cov_option {
            CovOption::Full | CovOption::Regularized(_) => {
                let d = diff.len();
                Array2::from_shape_fn((d, d), |(i, j)| diff[i] * diff[j] * weight)
            }
            CovOption::Diagonal => Array2::from_diag(&(&diff * &diff * weight)),
        }
    }

    fn compute_determinant(&self, matrix: &Array2<f64>) -> Result<f64, &'static str> {
        // Note: For production use, you might want to use a linear algebra library
        // that provides determinant computation. This is a simple implementation
        // that works for 2x2 matrices
        if matrix.nrows() != matrix.ncols() {
            return Err("Matrix must be square");
        }
        
        match matrix.nrows() {
            1 => Ok(matrix[[0, 0]]),
            2 => Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]),
            _ => Err("Determinant computation not implemented for matrices larger than 2x2")
        }
    }

    fn compute_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
        // Note: For production use, you might want to use a linear algebra library
        // that provides matrix inversion. This is a simple implementation
        // that works for 2x2 matrices
        if matrix.nrows() != matrix.ncols() {
            return Err("Matrix must be square");
        }
        
        match matrix.nrows() {
            1 => Ok(Array2::from_elem((1, 1), 1.0 / matrix[[0, 0]])),
            2 => {
                let det = self.compute_determinant(matrix)?;
                if det == 0.0 {
                    return Err("Matrix is singular");
                }
                let mut result = Array2::zeros((2, 2));
                result[[0, 0]] = matrix[[1, 1]] / det;
                result[[0, 1]] = -matrix[[0, 1]] / det;
                result[[1, 0]] = -matrix[[1, 0]] / det;
                result[[1, 1]] = matrix[[0, 0]] / det;
                Ok(result)
            },
            _ => Err("Matrix inversion not implemented for matrices larger than 2x2")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_means_none() {
        let model = GaussianMixtureModel::new(5);
        assert!(model.means().is_none());
    }

    #[test]
    fn test_covars_none() {
        let model = GaussianMixtureModel::new(5);
        assert!(model.covariances().is_none());
    }

    #[test]
    fn test_negative_mixtures() {
        let mix_weights = arr1(&[-0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }

    #[test]
    fn test_wrong_length_mixtures() {
        let mix_weights = arr1(&[0.1, 0.25, 0.75, 0.5]);
        let gmm_res = GaussianMixtureModel::with_weights(3, mix_weights);
        assert!(gmm_res.is_err());
    }

    #[test]
    fn test_basic_training() {
        use ndarray::arr2;
        let inputs = arr2(&[
            [1.0, 2.0],
            [-3.0, -3.0],
            [0.1, 1.5],
            [-5.0, -2.5]
        ]);
        let mut model = GaussianMixtureModel::new(2);
        model.set_max_iters(10);
        model.cov_option = CovOption::Diagonal;
        
        assert!(model.train(&inputs).is_ok());
        assert!(model.means().is_some());
        assert!(model.covariances().is_some());
    }
}