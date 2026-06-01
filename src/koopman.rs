//! Koopman operator
//!
//! The Koopman operator K acts on observable functions by composition with
//! the dynamics: (Kf)(x) = f(T(x)). It is the adjoint of the transfer operator
//! and provides a linear framework for nonlinear dynamics.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A Koopman operator represented as a matrix over a set of basis observables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoopmanOperator {
    /// Matrix K where (Kf)_j = Σ_i K[j][i] * f_i
    /// K acts on observables by left-multiplication of the row vector.
    matrix: DMatrix<f64>,
}

impl KoopmanOperator {
    /// Create from a matrix.
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self { matrix }
    }

    /// Create the Koopman operator as the transpose of a transfer operator matrix.
    /// If P is the transfer operator, K = P^T.
    pub fn from_transfer_matrix(transfer: &DMatrix<f64>) -> Self {
        Self {
            matrix: transfer.transpose(),
        }
    }

    /// Create from data using Dynamic Mode Decomposition (DMD).
    /// Given snapshots X (states at time t) and Y (states at time t+1),
    /// computes K ≈ Y * X^+ where X^+ is the pseudoinverse.
    pub fn from_dmd(x: &DMatrix<f64>, y: &DMatrix<f64>) -> Result<Self, String> {
        if x.nrows() != y.nrows() || x.ncols() != y.ncols() {
            return Err("X and Y must have the same dimensions".into());
        }
        // K = Y * X^pseudo_inverse via least squares: K^T = X^T \ Y^T
        // We solve K^T * x_cols = y_cols for each column
        let n = x.nrows();
        let xt = x.transpose();
        let yt = y.transpose();
        
        // Simple pseudoinverse via SVD-like approach
        let k_t = match xt.clone().try_inverse() {
            Some(inv) => &inv * &yt,
            None => {
                // Use least squares: K^T = (X*X^T)^-1 * X * Y^T
                let xxt = x * x.transpose();
                match xxt.try_inverse() {
                    Some(inv) => inv * x * y.transpose(),
                    None => return Err("Could not compute Koopman operator from data".into()),
                }
            }
        };
        Ok(Self {
            matrix: k_t.transpose(),
        })
    }

    /// Create from a list of trajectories.
    /// trajectories[t][i] = value of observable i at time t.
    pub fn from_trajectories(trajectories: &DMatrix<f64>) -> Result<Self, String> {
        if trajectories.ncols() < 2 {
            return Err("Need at least 2 time steps".into());
        }
        let x = trajectories.columns(0, trajectories.ncols() - 1).into_owned();
        let y = trajectories.columns(1, trajectories.ncols() - 1).into_owned();
        Self::from_dmd(&x, &y)
    }

    /// Get the underlying matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Apply the Koopman operator to an observable vector (row action).
    pub fn apply(&self, observable: &DVector<f64>) -> DVector<f64> {
        &self.matrix * observable
    }

    /// Compute the n-th iterate K^n.
    pub fn iterate(&self, n: usize) -> KoopmanOperator {
        if n == 0 {
            KoopmanOperator::new(DMatrix::identity(self.dim(), self.dim()))
        } else {
            let mut result = self.matrix.clone();
            for _ in 1..n {
                result = &self.matrix * &result;
            }
            KoopmanOperator::new(result)
        }
    }

    /// Compute eigenvalues of the Koopman operator.
    pub fn eigenvalues(&self) -> Vec<num_complex::Complex<f64>> {
        let n = self.dim();
        if n == 0 {
            return vec![];
        }
        // Use power method and deflation for small matrices
        let mut eigenvalues = Vec::new();
        let mut mat = self.matrix.clone();
        for _ in 0..n {
            let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
            for _ in 0..200 {
                let v_new = &mat * &v;
                let norm = v_new.norm();
                if norm > 1e-15 {
                    v = v_new / norm;
                } else {
                    break;
                }
            }
            let lambda: f64 = v.dot(&(&mat * &v));
            eigenvalues.push(num_complex::Complex::new(lambda, 0.0));
            // Deflate
            let w = &mat * &v;
            if v.norm() > 1e-15 {
                for i in 0..n {
                    for j in 0..n {
                        mat[(i, j)] -= lambda * v[i] * v[j];
                    }
                }
            }
        }
        eigenvalues
    }

    /// Compute the spectral radius.
    pub fn spectral_radius(&self) -> f64 {
        let eigvals = self.eigenvalues();
        eigvals.iter().map(|z| z.norm()).fold(0.0_f64, f64::max)
    }

    /// Check if the Koopman operator is unitary (measure-preserving dynamics).
    pub fn is_unitary(&self, tol: f64) -> bool {
        let kkt = &self.matrix * &self.matrix.transpose();
        let kk_t = &self.matrix.transpose() * &self.matrix;
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (kkt[(i, j)] - expected).abs() > tol || (kk_t[(i, j)] - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_koopman_identity() {
        let k = KoopmanOperator::new(DMatrix::identity(3, 3));
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_relative_eq!(k.apply(&v), v, epsilon = 1e-10);
    }

    #[test]
    fn test_koopman_from_transfer() {
        let p = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let k = KoopmanOperator::from_transfer_matrix(&p);
        // For doubly stochastic, K = P^T = P
        assert_relative_eq!(k.matrix()[(0, 0)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(k.matrix()[(1, 0)], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_koopman_iteration() {
        let k = KoopmanOperator::new(DMatrix::identity(2, 2));
        let k5 = k.iterate(5);
        assert_relative_eq!(k5.matrix()[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(k5.matrix()[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unitary_check() {
        // Rotation by 90 degrees is unitary
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]));
        assert!(k.is_unitary(1e-10));
    }

    #[test]
    fn test_not_unitary() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 0.5]));
        assert!(!k.is_unitary(1e-10));
    }

    #[test]
    fn test_spectral_radius() {
        let k = KoopmanOperator::new(DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]));
        let sr = k.spectral_radius();
        assert_relative_eq!(sr, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_apply_preserves_constant_observable() {
        // For stochastic matrix, K applied to constant should give constant
        let p = DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.4, 0.6]);
        let k = KoopmanOperator::from_transfer_matrix(&p);
        let ones = DVector::from_element(2, 1.0);
        let result = k.apply(&ones);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
    }
}
