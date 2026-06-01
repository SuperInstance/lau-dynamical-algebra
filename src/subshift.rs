//! Subshift of finite type
//!
//! A subshift of finite type (SFT) is defined by a transition matrix A:
//! Σ_A = {x ∈ {0,...,n-1}^ℕ : A[x_i][x_{i+1}] = 1 for all i}.
//! SFTs are fundamental objects in symbolic dynamics.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A subshift of finite type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubshiftFiniteType {
    /// Adjacency/transition matrix (0-1 entries).
    adjacency: DMatrix<f64>,
    /// Number of symbols.
    n_symbols: usize,
}

impl SubshiftFiniteType {
    /// Create an SFT from a 0-1 adjacency matrix.
    pub fn new(adjacency: DMatrix<f64>) -> Result<Self, String> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err("Adjacency matrix must be square".into());
        }
        // Validate entries are 0 or 1
        for i in 0..n {
            for j in 0..n {
                let v = adjacency[(i, j)];
                if v != 0.0 && v != 1.0 {
                    return Err(format!("Entry [{},{}] = {} is not 0 or 1", i, j, v));
                }
            }
        }
        Ok(Self {
            adjacency,
            n_symbols: n,
        })
    }

    /// Create a full shift on n symbols.
    pub fn full_shift(n: usize) -> Self {
        Self {
            adjacency: DMatrix::from_element(n, n, 1.0),
            n_symbols: n,
        }
    }

    /// Create a golden mean shift (no consecutive 1s on {0, 1}).
    pub fn golden_mean() -> Self {
        Self {
            adjacency: DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]),
            n_symbols: 2,
        }
    }

    /// Get the adjacency matrix.
    pub fn adjacency(&self) -> &DMatrix<f64> {
        &self.adjacency
    }

    /// Number of symbols.
    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// Count the number of allowed transitions.
    pub fn n_transitions(&self) -> usize {
        self.adjacency.iter().filter(|&&x| x == 1.0).count()
    }

    /// Count the number of words of length n.
    pub fn count_words(&self, n: usize) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let an = self.adjacency.pow(n as u32 - 1);
        an.iter().sum()
    }

    /// Compute the topological entropy h_top = lim (1/n) log(|B_n|).
    /// This equals log(λ) where λ is the spectral radius of the adjacency matrix.
    pub fn topological_entropy(&self) -> f64 {
        let spectral = crate::spectral::spectral_decomposition(&self.adjacency, self.n_symbols);
        spectral.spectral_radius.ln()
    }

    /// Get the transition matrix as a stochastic matrix (Parry measure).
    pub fn parry_measure_matrix(&self) -> DMatrix<f64> {
        let n = self.n_symbols;
        let mut v_left = DVector::from_element(n, 1.0 / n as f64);
        let mut v_right = DVector::from_element(n, 1.0 / n as f64);

        // Left eigenvector
        let at = self.adjacency.transpose();
        for _ in 0..1000 {
            let v_new = &at * &v_left;
            let norm = v_new.norm();
            if norm > 1e-20 {
                v_left = v_new / norm;
            }
        }

        // Right eigenvector
        for _ in 0..1000 {
            let v_new = &self.adjacency * &v_right;
            let norm = v_new.norm();
            if norm > 1e-20 {
                v_right = v_new / norm;
            }
        }

        // Normalize so that <v_left, v_right> = 1
        let dot: f64 = v_left.dot(&v_right);
        if dot.abs() > 1e-20 {
            v_right /= dot;
        }

        // Parry measure stochastic matrix: P[i][j] = (1/λ) v_right[i] / v_left[i] * A[i][j] * v_left[j]
        let lambda = v_left.dot(&(&self.adjacency * &v_right));
        let mut p = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.adjacency[(i, j)] > 0.0 && lambda.abs() > 1e-20 && v_left[i].abs() > 1e-20 {
                    p[(i, j)] = (1.0 / lambda) * v_right[i] / v_left[i]
                        * self.adjacency[(i, j)]
                        * v_left[j];
                }
            }
        }
        p
    }

    /// Compute the Parry measure (measure of maximal entropy).
    pub fn parry_measure(&self) -> DVector<f64> {
        let n = self.n_symbols;
        let mut v_left = DVector::from_element(n, 1.0 / n as f64);

        let at = self.adjacency.transpose();
        for _ in 0..1000 {
            let v_new = &at * &v_left;
            let sum: f64 = v_new.iter().sum();
            if sum.abs() > 1e-20 {
                v_left = v_new / sum;
            }
        }
        let sum: f64 = v_left.iter().sum();
        if sum > 0.0 {
            v_left / sum
        } else {
            v_left
        }
    }

    /// Check if a word is allowed.
    pub fn is_allowed_word(&self, word: &[usize]) -> bool {
        for i in 0..word.len().saturating_sub(1) {
            let a = word[i];
            let b = word[i + 1];
            if a >= self.n_symbols || b >= self.n_symbols {
                return false;
            }
            if self.adjacency[(a, b)] == 0.0 {
                return false;
            }
        }
        true
    }

    /// Enumerate all allowed words of a given length.
    pub fn enumerate_words(&self, length: usize) -> Vec<Vec<usize>> {
        if length == 0 {
            return vec![vec![]];
        }
        if length == 1 {
            return (0..self.n_symbols).map(|i| vec![i]).collect();
        }

        let shorter = self.enumerate_words(length - 1);
        let mut result = Vec::new();
        for word in &shorter {
            let last = word[word.len() - 1];
            for s in 0..self.n_symbols {
                if self.adjacency[(last, s)] == 1.0 {
                    let mut new_word = word.clone();
                    new_word.push(s);
                    result.push(new_word);
                }
            }
        }
        result
    }

    /// Compute the zeta function of the SFT: ζ(z) = 1/det(I - zA).
    pub fn zeta_function(&self, z: f64) -> f64 {
        let n = self.n_symbols;
        let i_minus_za = DMatrix::identity(n, n) - self.adjacency.scale(z);
        i_minus_za.determinant()
    }

    /// Check if irreducible (strongly connected).
    pub fn is_irreducible(&self) -> bool {
        let n = self.n_symbols;
        let a_n = self.adjacency.pow((n * n) as u32);
        a_n.iter().all(|&x| x > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_full_shift() {
        let sft = SubshiftFiniteType::full_shift(2);
        assert_eq!(sft.n_transitions(), 4);
    }

    #[test]
    fn test_golden_mean() {
        let sft = SubshiftFiniteType::golden_mean();
        assert_eq!(sft.n_transitions(), 3);
    }

    #[test]
    fn test_golden_mean_entropy() {
        let sft = SubshiftFiniteType::golden_mean();
        // h_top = log((1+√5)/2) = log(golden ratio)
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let h = sft.topological_entropy();
        assert_relative_eq!(h, golden_ratio.ln(), epsilon = 0.05);
    }

    #[test]
    fn test_count_words_full_shift() {
        let sft = SubshiftFiniteType::full_shift(2);
        assert_relative_eq!(sft.count_words(3), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_count_words_golden_mean() {
        let sft = SubshiftFiniteType::golden_mean();
        // Words of length 3: 000, 001, 010, 100, 101 = 5
        assert_relative_eq!(sft.count_words(3), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_allowed_word() {
        let sft = SubshiftFiniteType::golden_mean();
        assert!(sft.is_allowed_word(&[0, 0]));
        assert!(sft.is_allowed_word(&[0, 1]));
        assert!(sft.is_allowed_word(&[1, 0]));
        assert!(!sft.is_allowed_word(&[1, 1]));
    }

    #[test]
    fn test_enumerate_words() {
        let sft = SubshiftFiniteType::golden_mean();
        let words = sft.enumerate_words(2);
        assert_eq!(words.len(), 3); // 00, 01, 10
    }

    #[test]
    fn test_parry_measure_sums_to_one() {
        let sft = SubshiftFiniteType::golden_mean();
        let mu = sft.parry_measure();
        let sum: f64 = mu.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_zeta_at_zero() {
        let sft = SubshiftFiniteType::golden_mean();
        // det(I - 0*A) = det(I) = 1
        assert_relative_eq!(sft.zeta_function(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_full_shift_irreducible() {
        let sft = SubshiftFiniteType::full_shift(2);
        assert!(sft.is_irreducible());
    }

    #[test]
    fn test_invalid_adjacency() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.0, 1.0]);
        assert!(SubshiftFiniteType::new(m).is_err());
    }
}
