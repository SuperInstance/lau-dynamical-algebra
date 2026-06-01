//! Ruelle operator
//!
//! The Ruelle operator (transfer operator with potential) acts on functions as:
//! (L_φ f)(y) = Σ_{T(x)=y} exp(φ(x)) f(x)
//!
//! This is the key operator in thermodynamic formalism, connecting
//! dynamical systems with statistical mechanics.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Ruelle operator with potential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuelleOperator {
    /// Transfer matrix with potential weights.
    matrix: DMatrix<f64>,
    /// The potential function values.
    potential: DVector<f64>,
}

impl RuelleOperator {
    /// Create a Ruelle operator from a transition matrix and potential.
    pub fn new(transition: &DMatrix<f64>, potential: &DVector<f64>) -> Result<Self, String> {
        if transition.nrows() != potential.len() {
            return Err("Dimension mismatch".into());
        }
        let n = potential.len();
        let mut matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                // L_φ[i][j] = exp(φ(j)) * T[i][j]
                matrix[(i, j)] = potential[j].exp() * transition[(i, j)];
            }
        }
        Ok(Self {
            matrix,
            potential: potential.clone(),
        })
    }

    /// Create from a 0-1 transition matrix with a constant potential.
    pub fn with_constant_potential(transition: &DMatrix<f64>, beta: f64) -> Self {
        let n = transition.nrows();
        let potential = DVector::from_element(n, beta);
        let matrix = transition.map(|x| x * beta.exp());
        Self { matrix, potential }
    }

    /// Get the matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Apply the Ruelle operator to a function.
    pub fn apply(&self, f: &DVector<f64>) -> DVector<f64> {
        &self.matrix * f
    }

    /// Iterate n times.
    pub fn iterate(&self, n: usize) -> RuelleOperator {
        if n == 0 {
            RuelleOperator {
                matrix: DMatrix::identity(self.dim(), self.dim()),
                potential: self.potential.clone(),
            }
        } else {
            let mut result = self.matrix.clone();
            for _ in 1..n {
                result = &self.matrix * &result;
            }
            RuelleOperator {
                matrix: result,
                potential: self.potential.clone(),
            }
        }
    }

    /// Compute the leading eigenvalue (spectral radius) via power iteration.
    pub fn leading_eigenvalue(&self) -> f64 {
        let n = self.dim();
        let mut v = DVector::from_element(n, 1.0 / n as f64);
        for _ in 0..1000 {
            let v_new = self.apply(&v);
            let norm = v_new.norm();
            if norm < 1e-20 {
                return 0.0;
            }
            v = v_new / norm;
        }
        v.dot(&self.apply(&v))
    }

    /// Compute the pressure: P(φ) = log(leading eigenvalue of L_φ).
    pub fn pressure(&self) -> f64 {
        self.leading_eigenvalue().ln()
    }

    /// Find the eigenfunction corresponding to the leading eigenvalue.
    pub fn leading_eigenfunction(&self) -> DVector<f64> {
        let n = self.dim();
        let mut v = DVector::from_element(n, 1.0 / n as f64);
        for _ in 0..1000 {
            let v_new = self.apply(&v);
            let norm = v_new.norm();
            if norm < 1e-20 {
                break;
            }
            v = v_new / norm;
        }
        v
    }

    /// Compute the dual (adjoint) operator for computing equilibrium measures.
    pub fn dual(&self) -> DMatrix<f64> {
        self.matrix.transpose()
    }

    /// Compute the equilibrium measure via the dual operator.
    pub fn equilibrium_measure(&self, max_iter: usize, tol: f64) -> DVector<f64> {
        let dual = self.dual();
        let n = self.dim();
        let mut v = DVector::from_element(n, 1.0 / n as f64);
        for _ in 0..max_iter {
            let v_new = &dual * &v;
            let sum: f64 = v_new.iter().sum();
            if sum.abs() < 1e-20 {
                break;
            }
            let v_new = v_new / sum;
            let diff = (&v_new - &v).norm();
            v = v_new;
            if diff < tol {
                break;
            }
        }
        v
    }

    /// Compute correlation: C_n(f, g) = ∫ f · L^n g dμ - (∫ f dμ)(∫ g dμ).
    pub fn correlation(
        &self,
        f: &DVector<f64>,
        g: &DVector<f64>,
        n: usize,
        mu: &DVector<f64>,
    ) -> f64 {
        let ln_g = self.iterate(n).apply(g);
        let integral_f_ln_g: f64 = f.iter().zip(ln_g.iter()).zip(mu.iter()).map(|((fi, li), mi)| fi * li * mi).sum();
        let integral_f: f64 = f.iter().zip(mu.iter()).map(|(fi, mi)| fi * mi).sum();
        let integral_g: f64 = g.iter().zip(mu.iter()).map(|(gi, mi)| gi * mi).sum();
        integral_f_ln_g - integral_f * integral_g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ruelle_identity() {
        let t = DMatrix::identity(2, 2);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let v = DVector::from_vec(vec![1.0, 2.0]);
        assert_relative_eq!(r.apply(&v), v, epsilon = 1e-10);
    }

    #[test]
    fn test_ruelle_with_potential() {
        let t = DMatrix::identity(2, 2);
        let phi = DVector::from_vec(vec![1.0_f64.ln(), 1.0_f64.ln()]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let result = r.apply(&v);
        // exp(ln(1)) * 1 = 1 for each
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_leading_eigenvalue() {
        let t = DMatrix::identity(2, 2);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let lambda = r.leading_eigenvalue();
        assert_relative_eq!(lambda, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_pressure_zero_potential() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let p = r.pressure();
        // P(0) = log(spectral radius of transition)
        assert!(p.abs() < 1.0);
    }

    #[test]
    fn test_equilibrium_measure_sums_to_one() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let mu = r.equilibrium_measure(1000, 1e-12);
        let sum: f64 = mu.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_correlation_decay() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let r = RuelleOperator::new(&t, &phi).unwrap();
        let mu = r.equilibrium_measure(1000, 1e-12);
        let f = DVector::from_vec(vec![1.0, -1.0]);
        let g = DVector::from_vec(vec![1.0, 0.0]);
        let c0 = r.correlation(&f, &g, 0, &mu);
        let c10 = r.correlation(&f, &g, 10, &mu);
        // Correlation should decay
        assert!(c10.abs() < c0.abs() + 0.1);
    }

    #[test]
    fn test_constant_potential() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let r = RuelleOperator::with_constant_potential(&t, 0.0);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let result = r.apply(&v);
        // exp(0) * 0.5 * [1+1] = 1.0 for each
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
    }
}
