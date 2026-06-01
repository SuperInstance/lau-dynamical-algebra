//! Transfer operator (Perron-Frobenius operator)
//!
//! The transfer operator (also called the Perron-Frobenius operator) describes
//! how densities evolve under a dynamical system. For a map T: X → X, it acts
//! on density functions f as: (Pf)(y) = Σ_{T(x)=y} f(x) / |T'(x)|.

use nalgebra::{DMatrix, DVector, ComplexField};
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// A transfer operator represented as a stochastic matrix over a finite partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferOperator {
    /// The stochastic matrix P where P[i][j] = Prob(transition from j to i)
    matrix: DMatrix<f64>,
}

impl TransferOperator {
    /// Create a transfer operator from a stochastic matrix (columns sum to 1).
    pub fn from_stochastic_matrix(matrix: DMatrix<f64>) -> Result<Self, String> {
        let n = matrix.ncols();
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| matrix[(i, j)]).sum();
            if (col_sum - 1.0).abs() > 1e-10 {
                return Err(format!(
                    "Column {} sums to {} (expected 1.0)",
                    j, col_sum
                ));
            }
        }
        Ok(Self { matrix })
    }

    /// Create from a transition matrix (rows sum to 1, will transpose).
    pub fn from_row_stochastic(matrix: DMatrix<f64>) -> Result<Self, String> {
        Self::from_stochastic_matrix(matrix.transpose())
    }

    /// Create from a map on a finite set and a partition size.
    /// map[i] = j means element i maps to element j.
    pub fn from_map(map: &[usize], n: usize) -> Self {
        let mut matrix = DMatrix::zeros(n, n);
        for (i, &j) in map.iter().enumerate() {
            if j < n && i < n {
                matrix[(j, i)] += 1.0;
            }
        }
        // Normalize columns
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| matrix[(i, j)]).sum();
            if col_sum > 0.0 {
                for i in 0..n {
                    matrix[(i, j)] /= col_sum;
                }
            }
        }
        Self { matrix }
    }

    /// Get the underlying matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Dimension of the operator.
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Apply the transfer operator to a density vector.
    pub fn apply(&self, density: &DVector<f64>) -> DVector<f64> {
        &self.matrix * density
    }

    /// Compute the n-th iterate P^n.
    pub fn iterate(&self, n: usize) -> TransferOperator {
        if n == 0 {
            TransferOperator {
                matrix: DMatrix::identity(self.dim(), self.dim()),
            }
        } else {
            let mut result = self.matrix.clone();
            for _ in 1..n {
                result = &self.matrix * &result;
            }
            TransferOperator { matrix: result }
        }
    }

    /// Find the invariant (stationary) measure via power iteration.
    pub fn invariant_measure(&self, max_iter: usize, tol: f64) -> DVector<f64> {
        let n = self.dim();
        let mut v = DVector::from_element(n, 1.0 / n as f64);
        for _ in 0..max_iter {
            let v_new = self.apply(&v);
            let diff = (&v_new - &v).norm();
            v = v_new;
            if diff < tol {
                break;
            }
        }
        // Normalize
        let sum: f64 = v.iter().sum();
        if sum > 0.0 {
            v /= sum;
        }
        v
    }

    /// Check if the operator is irreducible (Perron-Frobenius condition).
    pub fn is_irreducible(&self) -> bool {
        let n = self.dim();
        let pn = self.iterate(n * n);
        pn.matrix.iter().all(|&x| x > 0.0 || x.is_nan() == false) &&
            (0..n).all(|i| (0..n).any(|j| pn.matrix[(i,j)] > 1e-15))
    }

    /// Compute the Perron-Frobenius eigenvalue (spectral radius).
    pub fn perron_frobenius_eigenvalue(&self) -> f64 {
        let n = self.dim();
        let mut v = DVector::from_element(n, 1.0 / n as f64);
        for _ in 0..1000 {
            let v_new = self.apply(&v);
            let lambda = v_new.dot(&v) / v.dot(&v);
            let norm = v_new.norm();
            if norm > 0.0 {
                v = v_new / norm;
            }
        }
        let v_new = self.apply(&v);
        v_new.dot(&v) / v.dot(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_transfer() {
        let p = TransferOperator::from_stochastic_matrix(DMatrix::identity(3, 3)).unwrap();
        let v = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let result = p.apply(&v);
        assert_relative_eq!(v, result, epsilon = 1e-10);
    }

    #[test]
    fn test_from_map() {
        // A 2-cycle on {0, 1}
        let p = TransferOperator::from_map(&[1, 0], 2);
        assert_eq!(p.matrix()[(1, 0)], 1.0);
        assert_eq!(p.matrix()[(0, 1)], 1.0);
    }

    #[test]
    fn test_stochastic_columns() {
        let p = TransferOperator::from_map(&[1, 2, 0], 3);
        let m = p.matrix();
        for j in 0..3 {
            let col_sum: f64 = (0..3).map(|i| m[(i, j)]).sum();
            assert_relative_eq!(col_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_invariant_measure_uniform() {
        // Fully connected with equal weights
        let m = DMatrix::from_element(3, 3, 1.0 / 3.0);
        let p = TransferOperator::from_stochastic_matrix(m).unwrap();
        let mu = p.invariant_measure(1000, 1e-12);
        for i in 0..3 {
            assert_relative_eq!(mu[i], 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_iteration() {
        let p = TransferOperator::from_map(&[1, 0], 2);
        let p2 = p.iterate(2);
        // P^2 should be identity for a 2-cycle
        assert_relative_eq!(p2.matrix()[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(p2.matrix()[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_perron_frobenius_eigenvalue() {
        let m = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7]);
        let p = TransferOperator::from_row_stochastic(m).unwrap();
        let lambda = p.perron_frobenius_eigenvalue();
        assert_relative_eq!(lambda, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_invalid_stochastic() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]);
        assert!(TransferOperator::from_stochastic_matrix(m).is_err());
    }

    #[test]
    fn test_apply_preserves_total() {
        let p = TransferOperator::from_map(&[1, 2, 0], 3);
        let v = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let result = p.apply(&v);
        let total: f64 = result.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }
}
