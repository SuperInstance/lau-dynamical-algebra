//! Dynamical zeta function det(I − zT)⁻¹
//!
//! The dynamical zeta function encodes periodic orbit information of a dynamical
//! system. For a transfer operator T, ζ(z) = det(I − zT)⁻¹.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Dynamical zeta function associated with an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicalZeta {
    /// The transfer operator matrix.
    operator: DMatrix<f64>,
}

impl DynamicalZeta {
    /// Create from a transfer operator matrix.
    pub fn new(operator: DMatrix<f64>) -> Self {
        Self { operator }
    }

    /// Evaluate ζ(z) = 1 / det(I − zT).
    pub fn evaluate(&self, z: Complex<f64>) -> Complex<f64> {
        let n = self.operator.nrows();
        let i_minus_zt: DMatrix<Complex<f64>> = DMatrix::identity(n, n)
            - self.operator.map(|x| z * Complex::new(x, 0.0));
        let det = complex_determinant(&i_minus_zt);
        if det.norm() < 1e-20 {
            Complex::new(f64::INFINITY, 0.0)
        } else {
            det.inv()
        }
    }

    /// Compute det(I − zT).
    pub fn determinant(&self, z: Complex<f64>) -> Complex<f64> {
        let n = self.operator.nrows();
        let i_minus_zt: DMatrix<Complex<f64>> = DMatrix::identity(n, n)
            - self.operator.map(|x| z * Complex::new(x, 0.0));
        complex_determinant(&i_minus_zt)
    }

    /// Compute log ζ(z).
    pub fn log_zeta(&self, z: Complex<f64>) -> Complex<f64> {
        self.evaluate(z).ln()
    }

    /// Find poles of ζ (zeros of det(I − zT)).
    /// These are reciprocals of eigenvalues of T.
    pub fn poles(&self) -> Vec<Complex<f64>> {
        let n = self.operator.nrows();
        let mut poles = Vec::new();

        // Power iteration + deflation for eigenvalues
        let mut mat = self.operator.clone();
        for _ in 0..n {
            let mut v = DVector::from_element(n, 1.0 / (n as f64).sqrt());
            for _ in 0..300 {
                let v_new = &mat * &v;
                let norm = v_new.norm();
                if norm < 1e-20 {
                    break;
                }
                v = v_new / norm;
            }
            let lambda: f64 = v.dot(&(&mat * &v));
            if lambda.abs() > 1e-15 {
                poles.push(Complex::new(1.0 / lambda, 0.0));
            }
            // Deflate
            for i in 0..n {
                for j in 0..n {
                    mat[(i, j)] -= lambda * v[i] * v[j];
                }
            }
        }
        poles
    }

    /// Compute d/dz log ζ(z) (derivative).
    pub fn derivative_log_zeta(&self, z: Complex<f64>, h: f64) -> Complex<f64> {
        let z_plus = z + Complex::new(h, 0.0);
        let z_minus = z - Complex::new(h, 0.0);
        (self.log_zeta(z_plus) - self.log_zeta(z_minus)) / Complex::new(2.0 * h, 0.0)
    }

    /// Taylor expansion of log ζ(z) around z=0.
    /// log ζ(z) = Σ_{n≥1} c_n z^n where c_n = Tr(T^n)/n.
    pub fn taylor_coefficients(&self, order: usize) -> Vec<f64> {
        let mut coeffs = Vec::new();
        let mut tn = self.operator.clone();
        for n in 1..=order {
            let trace: f64 = tn.trace();
            coeffs.push(trace / n as f64);
            tn = &self.operator * &tn;
        }
        coeffs
    }

    /// Radius of convergence (reciprocal of spectral radius of T).
    pub fn radius_of_convergence(&self) -> f64 {
        let spectral = crate::spectral::spectral_decomposition(&self.operator, self.operator.nrows());
        if spectral.spectral_radius > 0.0 {
            1.0 / spectral.spectral_radius
        } else {
            f64::INFINITY
        }
    }
}

/// Compute the determinant of a complex matrix using cofactor expansion (small matrices).
fn complex_determinant(mat: &DMatrix<Complex<f64>>) -> Complex<f64> {
    let n = mat.nrows();
    if n == 0 {
        return Complex::new(1.0, 0.0);
    }
    if n == 1 {
        return mat[(0, 0)];
    }
    if n == 2 {
        return mat[(0, 0)] * mat[(1, 1)] - mat[(0, 1)] * mat[(1, 0)];
    }
    // LU-style: try to use the real determinant on the real part if imaginary is small
    // Otherwise cofactor expansion
    let mut det = Complex::new(0.0, 0.0);
    for j in 0..n {
        let cofactor = mat[(0, j)] * complex_determinant(&minor(mat, 0, j));
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det = det + cofactor * Complex::new(sign, 0.0);
    }
    det
}

/// Get the minor matrix excluding row i and column j.
fn minor(mat: &DMatrix<Complex<f64>>, i: usize, j: usize) -> DMatrix<Complex<f64>> {
    let n = mat.nrows();
    let mut result = DMatrix::zeros(n - 1, n - 1);
    let mut ri = 0;
    for r in 0..n {
        if r == i {
            continue;
        }
        let mut ci = 0;
        for c in 0..n {
            if c == j {
                continue;
            }
            result[(ri, ci)] = mat[(r, c)];
            ci += 1;
        }
        ri += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zeta_at_zero() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let zeta = DynamicalZeta::new(t);
        // ζ(0) = 1/det(I) = 1
        let val = zeta.evaluate(Complex::new(0.0, 0.0));
        assert_relative_eq!(val.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_determinant_at_zero() {
        let t = DMatrix::identity(2, 2);
        let zeta = DynamicalZeta::new(t);
        let det = zeta.determinant(Complex::new(0.0, 0.0));
        assert_relative_eq!(det.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_taylor_coefficients_identity() {
        let t = DMatrix::identity(2, 2);
        let zeta = DynamicalZeta::new(t);
        let coeffs = zeta.taylor_coefficients(3);
        // Tr(I^n) = 2 for all n, so c_n = 2/n
        assert_relative_eq!(coeffs[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs[2], 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poles_identity() {
        let t = DMatrix::identity(2, 2);
        let zeta = DynamicalZeta::new(t);
        let poles = zeta.poles();
        // Eigenvalues of I are 1, so poles should be at z=1
        assert!(poles.iter().any(|p| (p.re - 1.0).abs() < 0.1));
    }

    #[test]
    fn test_complex_det_2x2() {
        let m = DMatrix::from_row_slice(2, 2, &[
            Complex::new(1.0, 0.0), Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0), Complex::new(4.0, 0.0),
        ]);
        let det = complex_determinant(&m);
        assert_relative_eq!(det.re, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radius_of_convergence() {
        let t = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, 0.3]));
        let zeta = DynamicalZeta::new(t);
        let r = zeta.radius_of_convergence();
        assert_relative_eq!(r, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_evaluate_real_positive() {
        let t = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5]));
        let zeta = DynamicalZeta::new(t);
        // ζ(z) = 1/(1 - 0.5z)
        let val = zeta.evaluate(Complex::new(1.0, 0.0));
        assert_relative_eq!(val.re, 2.0, epsilon = 1e-10);
    }
}
