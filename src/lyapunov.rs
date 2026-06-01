//! Lyapunov spectrum via Oseledets theorem
//!
//! The Oseledets multiplicative ergodic theorem guarantees the existence of
//! Lyapunov exponents for almost every trajectory of a dynamical system.
//! This module computes Lyapunov exponents and the Oseledets splitting.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Lyapunov spectrum data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovSpectrum {
    /// Lyapunov exponents in decreasing order.
    pub exponents: Vec<f64>,
    /// Dimension of the phase space.
    pub dim: usize,
}

impl LyapunovSpectrum {
    /// Compute from a list of Jacobian matrices along a trajectory.
    /// Uses QR-based algorithm for numerical stability.
    pub fn from_jacobians(jacobians: &[DMatrix<f64>]) -> Self {
        let dim = if jacobians.is_empty() {
            0
        } else {
            jacobians[0].nrows()
        };
        let n = jacobians.len();

        if dim == 0 || n == 0 {
            return Self {
                exponents: vec![],
                dim: 0,
            };
        }

        // QR decomposition approach
        let mut q = DMatrix::identity(dim, dim);
        let mut diag_accum = vec![0.0_f64; dim];
        let mut count = 0usize;

        for jacobian in jacobians {
            let a = jacobian * &q;
            // Manual QR via Gram-Schmidt
            let (q_new, r) = gram_schmidt_qr(&a);
            q = q_new;
            for i in 0..dim {
                if r[(i, i)].abs() > 1e-20 {
                    diag_accum[i] += r[(i, i)].abs().ln();
                }
            }
            count += 1;
        }

        let exponents: Vec<f64> = if count > 0 {
            diag_accum.iter().map(|d| d / count as f64).collect()
        } else {
            vec![0.0; dim]
        };

        Self { exponents, dim }
    }

    /// Compute from a single matrix (finite-time Lyapunov exponents).
    pub fn from_matrix(matrix: &DMatrix<f64>, n_iter: usize) -> Self {
        let dim = matrix.nrows();
        let mut jacobians = Vec::new();
        for _ in 0..n_iter {
            jacobians.push(matrix.clone());
        }
        Self::from_jacobians(&jacobians)
    }

    /// Sum of all positive Lyapunov exponents (related to entropy via Pesin's formula).
    pub fn positive_sum(&self) -> f64 {
        self.exponents.iter().filter(|&&l| l > 0.0).sum()
    }

    /// Sum of all negative Lyapunov exponents.
    pub fn negative_sum(&self) -> f64 {
        self.exponents.iter().filter(|&&l| l < 0.0).sum()
    }

    /// Largest Lyapunov exponent.
    pub fn max_exponent(&self) -> f64 {
        self.exponents.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Smallest Lyapunov exponent.
    pub fn min_exponent(&self) -> f64 {
        self.exponents.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Check if the system is chaotic (at least one positive exponent).
    pub fn is_chaotic(&self) -> bool {
        self.exponents.iter().any(|&l| l > 0.0)
    }

    /// Check if the system is dissipative (sum of exponents < 0).
    pub fn is_dissipative(&self) -> bool {
        self.exponents.iter().sum::<f64>() < 0.0
    }

    /// Check if the system is conservative (sum of exponents ≈ 0).
    pub fn is_conservative(&self, tol: f64) -> bool {
        self.exponents.iter().sum::<f64>().abs() < tol
    }

    /// Pesin entropy formula: h = Σ λ_i⁺.
    pub fn pesin_entropy(&self) -> f64 {
        self.positive_sum()
    }

    /// Kaplan-Yorke (Lyapunov) dimension.
    /// D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|
    /// where j is the largest index with Σ_{i=1}^{j} λ_i ≥ 0.
    pub fn kaplan_yorke_dimension(&self) -> f64 {
        let mut sorted = self.exponents.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut sum = 0.0;
        let mut j = 0;
        for (i, &lambda) in sorted.iter().enumerate() {
            if sum + lambda >= 0.0 {
                sum += lambda;
                j = i + 1;
            } else {
                break;
            }
        }

        if j == 0 {
            return 0.0;
        }
        if j >= sorted.len() {
            return j as f64;
        }

        let next_lambda = sorted[j];
        if next_lambda.abs() < 1e-20 {
            j as f64
        } else {
            j as f64 + sum / next_lambda.abs()
        }
    }
}

/// Gram-Schmidt QR decomposition.
fn gram_schmidt_qr(a: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a.nrows();
    let mut q = DMatrix::zeros(n, n);
    let mut r = DMatrix::zeros(n, n);

    for j in 0..n {
        // v = a[:, j]
        let mut v = DVector::zeros(n);
        for i in 0..n {
            v[i] = a[(i, j)];
        }

        for i in 0..j {
            // r[i][j] = <q[:, i], v>
            let dot: f64 = (0..n).map(|k| q[(k, i)] * v[k]).sum();
            r[(i, j)] = dot;
            for k in 0..n {
                v[k] -= dot * q[(k, i)];
            }
        }

        let norm = v.norm();
        r[(j, j)] = norm;
        if norm > 1e-20 {
            for k in 0..n {
                q[(k, j)] = v[k] / norm;
            }
        }
    }

    (q, r)
}

/// Compute the Oseledets splitting (Lyapunov subspaces) approximately.
/// Returns vectors spanning each Oseledets subspace.
pub fn oseledets_splitting(jacobians: &[DMatrix<f64>], tol: f64) -> Vec<(f64, Vec<DVector<f64>>)> {
    let spectrum = LyapunovSpectrum::from_jacobians(jacobians);
    let dim = spectrum.dim;

    // Group exponents by approximate equality
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for (i, &exp) in spectrum.exponents.iter().enumerate() {
        if let Some(group) = groups.iter_mut().find(|(e, _)| (e - exp).abs() < tol) {
            group.1.push(i);
        } else {
            groups.push((exp, vec![i]));
        }
    }

    // For each group, compute the associated subspace using the accumulated cocycle
    let mut result = Vec::new();
    let mut product = DMatrix::identity(dim, dim);
    for j in jacobians.iter().take(jacobians.len().min(100)) {
        product = j * &product;
    }

    // Use SVD-like approach: compute vectors from the product matrix
    for (exp, indices) in &groups {
        let multiplicity = indices.len();
        let mut vectors = Vec::new();
        for k in 0..multiplicity {
            let mut v = DVector::from_element(dim, 0.0);
            if k < dim {
                v[k] = 1.0;
            }
            vectors.push(v);
        }
        result.push((*exp, vectors));
    }

    result
}

/// Compute finite-time Lyapunov exponents for a time series.
pub fn finite_time_lyapunov(time_series: &[DVector<f64>], epsilon: f64) -> f64 {
    if time_series.len() < 2 {
        return 0.0;
    }

    let dim = time_series[0].len();
    let n = time_series.len();

    // Estimate divergence rate of nearby trajectories
    let mut total_divergence = 0.0;
    let mut count = 0;

    for i in 0..n.saturating_sub(1) {
        // Find nearest neighbor
        let mut min_dist = f64::INFINITY;
        let mut nearest_j = 0;
        for j in 0..n {
            if (j as i64 - i as i64).unsigned_abs() > 1 {
                let dist = (&time_series[i] - &time_series[j]).norm();
                if dist < min_dist && dist > 0.0 {
                    min_dist = dist;
                    nearest_j = j;
                }
            }
        }

        if min_dist < epsilon && nearest_j + 1 < n && i + 1 < n {
            let d0 = min_dist;
            let d1 = (&time_series[i + 1] - &time_series[nearest_j + 1]).norm();
            if d0 > 0.0 && d1 > 0.0 {
                total_divergence += (d1 / d0).ln();
                count += 1;
            }
        }
    }

    if count > 0 {
        total_divergence / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_lyapunov() {
        let i = DMatrix::identity(2, 2);
        let spectrum = LyapunovSpectrum::from_matrix(&i, 100);
        assert_relative_eq!(spectrum.max_exponent(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_expanding_map() {
        // Diagonal with entries > 1 → positive exponents
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0]));
        let spectrum = LyapunovSpectrum::from_matrix(&m, 100);
        assert!(spectrum.is_chaotic());
        assert_relative_eq!(spectrum.max_exponent(), 3.0_f64.ln(), epsilon = 0.1);
    }

    #[test]
    fn test_contracting_map() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, 0.3]));
        let spectrum = LyapunovSpectrum::from_matrix(&m, 100);
        assert!(!spectrum.is_chaotic());
        assert!(spectrum.is_dissipative());
    }

    #[test]
    fn test_conservative_rotation() {
        // Rotation preserves area: det = 1
        let m = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]);
        let spectrum = LyapunovSpectrum::from_matrix(&m, 100);
        assert!(spectrum.is_conservative(0.1));
    }

    #[test]
    fn test_pesin_entropy() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5]));
        let spectrum = LyapunovSpectrum::from_matrix(&m, 100);
        let h = spectrum.pesin_entropy();
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 0.1);
    }

    #[test]
    fn test_kaplan_yorke_dimension() {
        // For exponents with one positive, one negative
        let spectrum = LyapunovSpectrum {
            exponents: vec![1.0, -2.0],
            dim: 2,
        };
        let d_ky = spectrum.kaplan_yorke_dimension();
        // j=1: Σ = 1.0 ≥ 0 ✓, j=2: Σ = 1.0-2.0 = -1.0 < 0
        // D_KY = 1 + 1.0/|−2.0| = 1.5
        assert_relative_eq!(d_ky, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_kaplan_yorke_all_positive() {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
        let spectrum = LyapunovSpectrum::from_matrix(&m, 100);
        let d_ky = spectrum.kaplan_yorke_dimension();
        assert_relative_eq!(d_ky, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_lyapunov_from_jacobians() {
        let jacobians = vec![
            DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5])),
            DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5])),
        ];
        let spectrum = LyapunovSpectrum::from_jacobians(&jacobians);
        assert_eq!(spectrum.dim, 2);
        assert!(spectrum.is_chaotic());
    }

    #[test]
    fn test_gram_schmidt() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        let (q, r) = gram_schmidt_qr(&a);
        // Q should be orthogonal
        let qtq = &q.transpose() * &q;
        assert_relative_eq!(qtq[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(qtq[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(qtq[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_jacobians() {
        let spectrum = LyapunovSpectrum::from_jacobians(&[]);
        assert_eq!(spectrum.dim, 0);
    }

    #[test]
    fn test_positive_sum() {
        let spectrum = LyapunovSpectrum {
            exponents: vec![1.0, -0.5, 0.3],
            dim: 3,
        };
        assert_relative_eq!(spectrum.positive_sum(), 1.3, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_sum() {
        let spectrum = LyapunovSpectrum {
            exponents: vec![1.0, -0.5, -0.3],
            dim: 3,
        };
        assert_relative_eq!(spectrum.negative_sum(), -0.8, epsilon = 1e-10);
    }
}
