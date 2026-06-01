//! Spectral decomposition
//!
//! Spectral decomposition of operators arising in dynamical systems.
//! Includes eigenvalue computation, spectral projections, and functional calculus.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Spectral data for an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralData {
    /// Approximate eigenvalues.
    pub eigenvalues: Vec<Complex<f64>>,
    /// Corresponding eigenvectors (columns).
    pub eigenvectors: Vec<DVector<Complex<f64>>>,
    /// Spectral radius.
    pub spectral_radius: f64,
}

/// Compute spectral decomposition using iterative methods.
pub fn spectral_decomposition(matrix: &DMatrix<f64>, max_eigenvalues: usize) -> SpectralData {
    let n = matrix.nrows();
    let n_find = max_eigenvalues.min(n);

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Vec::new();
    let mut residual = matrix.clone();

    for _ in 0..n_find {
        let (lambda, v) = power_iteration(&residual, 500, 1e-12);
        eigenvalues.push(lambda);
        eigenvectors.push(v.clone());

        // Deflation
        for i in 0..n {
            for j in 0..n {
                residual[(i, j)] -= lambda.re * v[i].re * v[j].re;
            }
        }
    }

    let spectral_radius = eigenvalues.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);

    SpectralData {
        eigenvalues,
        eigenvectors,
        spectral_radius,
    }
}

/// Power iteration for dominant eigenvalue.
pub fn power_iteration(
    matrix: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> (Complex<f64>, DVector<Complex<f64>>) {
    let n = matrix.nrows();
    let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());

    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let v_new = matrix * &v;
        let norm = v_new.norm();
        if norm < 1e-20 {
            break;
        }
        v = v_new / norm;
        lambda = v.dot(&(matrix * &v));
    }

    (
        Complex::new(lambda, 0.0),
        v.map(|x| Complex::new(x, 0.0)),
    )
}

/// Spectral projection onto the eigenspace of eigenvalue λ.
pub fn spectral_projection(matrix: &DMatrix<f64>, eigenvalue: f64, tol: f64) -> DMatrix<f64> {
    let n = matrix.nrows();
    let mut projection = DMatrix::zeros(n, n);

    let (_, v) = power_iteration(&(matrix.clone() - DMatrix::from_diagonal(&DVector::from_element(n, eigenvalue))), 200, tol);

    for i in 0..n {
        for j in 0..n {
            projection[(i, j)] += v[i].re * v[j].re;
        }
    }
    projection
}

/// Functional calculus for self-adjoint operators: evaluate f(A) using spectral decomposition.
pub fn functional_calculus<F>(matrix: &DMatrix<f64>, f: F) -> DMatrix<f64>
where
    F: Fn(f64) -> f64,
{
    let n = matrix.nrows();
    let spectral = spectral_decomposition(matrix, n);

    let mut result = DMatrix::zeros(n, n);
    for (k, &lambda) in spectral.eigenvalues.iter().enumerate() {
        let v = &spectral.eigenvectors[k];
        let f_lambda = f(lambda.re);
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] += f_lambda * v[i].re * v[j].re;
            }
        }
    }
    result
}

/// Compute the matrix exponential exp(A) via spectral decomposition.
pub fn matrix_exp(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    functional_calculus(matrix, |x| x.exp())
}

/// Compute the resolvent (zI - A)⁻¹.
pub fn resolvent(matrix: &DMatrix<f64>, z: Complex<f64>) -> Option<DMatrix<Complex<f64>>> {
    let n = matrix.nrows();
    let z_matrix: DMatrix<Complex<f64>> = DMatrix::from_diagonal(&DVector::from_element(n, z))
        - matrix.map(|x| Complex::new(x, 0.0));
    z_matrix.try_inverse()
}

/// Compute the spectral gap (difference between top two eigenvalue magnitudes).
pub fn spectral_gap(matrix: &DMatrix<f64>) -> f64 {
    let spectral = spectral_decomposition(matrix, 2.min(matrix.nrows()));
    if spectral.eigenvalues.len() >= 2 {
        spectral.eigenvalues[0].norm() - spectral.eigenvalues[1].norm()
    } else if spectral.eigenvalues.len() == 1 {
        spectral.eigenvalues[0].norm()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_power_iteration_diagonal() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 1.0]));
        let (lambda, v) = power_iteration(&m, 100, 1e-10);
        assert_relative_eq!(lambda.re, 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_spectral_radius() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, -1.0, 0.5]));
        let data = spectral_decomposition(&m, 3);
        assert_relative_eq!(data.spectral_radius, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_functional_calculus_exp() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 0.0]));
        let exp_m = matrix_exp(&m);
        assert_relative_eq!(exp_m[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_m[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_resolvent() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0]));
        let r = resolvent(&m, Complex::new(0.0, 0.0)).unwrap();
        assert_relative_eq!(r[(0, 0)].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(r[(1, 1)].re, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_resolvent_singular() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0]));
        // z = 1 is an eigenvalue, resolvent should be singular
        assert!(resolvent(&m, Complex::new(1.0, 0.0)).is_none());
    }

    #[test]
    fn test_spectral_gap() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 1.0]));
        let gap = spectral_gap(&m);
        assert_relative_eq!(gap, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_spectral_projection_idempotent() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 1.0]));
        let p = spectral_projection(&m, 2.0, 1e-10);
        let p2 = &p * &p;
        assert_relative_eq!(p[(0, 0)], p2[(0, 0)], epsilon = 1e-8);
    }
}
