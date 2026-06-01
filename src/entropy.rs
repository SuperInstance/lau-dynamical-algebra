//! Entropy of dynamical systems
//!
//! Provides various entropy notions: topological entropy, measure-theoretic
//! (Kolmogorov-Sinai) entropy, conditional entropy, and mutual information.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Compute the Shannon entropy of a probability distribution.
/// H(p) = -Σ p_i log(p_i).
pub fn shannon_entropy(prob: &[f64]) -> f64 {
    prob.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Compute the Shannon entropy of a DVector probability distribution.
pub fn shannon_entropy_vec(prob: &DVector<f64>) -> f64 {
    shannon_entropy(prob.as_slice())
}

/// Compute joint entropy H(X, Y) from a joint distribution matrix.
pub fn joint_entropy(joint: &DMatrix<f64>) -> f64 {
    let mut h = 0.0;
    for val in joint.iter() {
        if *val > 0.0 {
            h -= val * val.ln();
        }
    }
    h
}

/// Compute conditional entropy H(X|Y) from a joint distribution.
pub fn conditional_entropy(joint: &DMatrix<f64>) -> f64 {
    let n_y = joint.ncols();
    let mut h = 0.0;
    for j in 0..n_y {
        let p_y: f64 = (0..joint.nrows()).map(|i| joint[(i, j)]).sum();
        if p_y > 0.0 {
            for i in 0..joint.nrows() {
                let p_xy = joint[(i, j)];
                if p_xy > 0.0 {
                    h -= p_xy * (p_xy / p_y).ln();
                }
            }
        }
    }
    h
}

/// Compute mutual information I(X; Y) = H(X) + H(Y) - H(X,Y).
pub fn mutual_information(joint: &DMatrix<f64>) -> f64 {
    let h_x: f64 = {
        let mut marg = vec![0.0; joint.nrows()];
        for i in 0..joint.nrows() {
            for j in 0..joint.ncols() {
                marg[i] += joint[(i, j)];
            }
        }
        shannon_entropy(&marg)
    };
    let h_y: f64 = {
        let mut marg = vec![0.0; joint.ncols()];
        for i in 0..joint.nrows() {
            for j in 0..joint.ncols() {
                marg[j] += joint[(i, j)];
            }
        }
        shannon_entropy(&marg)
    };
    let h_xy = joint_entropy(joint);
    h_x + h_y - h_xy
}

/// Compute topological entropy of a subshift from its adjacency matrix.
pub fn topological_entropy_adjacency(adjacency: &DMatrix<f64>) -> f64 {
    let n = adjacency.nrows();
    let spectral = crate::spectral::spectral_decomposition(adjacency, n);
    spectral.spectral_radius.ln()
}

/// Kolmogorov-Sinai entropy of a Markov chain.
/// h(μ) = -Σ_{i,j} μ[i] P[i][j] log(P[i][j]).
pub fn kolmogorov_sinai_entropy(measure: &DVector<f64>, transition: &DMatrix<f64>) -> f64 {
    let n = measure.len();
    let mut h = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p = transition[(i, j)];
            if p > 0.0 && measure[i] > 0.0 {
                h -= measure[i] * p * p.ln();
            }
        }
    }
    h
}

/// Compute the entropy rate of a Markov chain.
pub fn entropy_rate(measure: &DVector<f64>, transition: &DMatrix<f64>) -> f64 {
    kolmogorov_sinai_entropy(measure, transition)
}

/// Relative entropy (KL divergence) D(P || Q).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 0.0 && qi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Topological entropy via (1/n) log(|B_n|) for word counts.
pub fn topological_entropy_from_counts(word_counts: &[f64]) -> f64 {
    if word_counts.len() < 2 {
        return 0.0;
    }
    // Linear regression on log(|B_n|) vs n
    let n = word_counts.len();
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (i, &count) in word_counts.iter().enumerate() {
        let x = (i + 1) as f64;
        let y = if count > 0.0 { count.ln() } else { 0.0 };
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let n_f = n as f64;
    let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
    slope
}

/// Information dimension: lim_{ε→0} H(ε) / log(1/ε).
pub fn information_dimension(entropies: &[f64], scales: &[f64]) -> f64 {
    if entropies.len() != scales.len() || entropies.len() < 2 {
        return 0.0;
    }
    // Linear regression: H(ε) = -D * log(ε) + const
    let n = entropies.len();
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    for (i, &h) in entropies.iter().enumerate() {
        let x = -scales[i].ln();
        let y = h;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    let n_f = n as f64;
    (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_shannon_entropy_uniform() {
        let p = vec![0.5, 0.5];
        assert_relative_eq!(shannon_entropy(&p), 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let p = vec![1.0, 0.0];
        assert_relative_eq!(shannon_entropy(&p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_shannon_entropy_vec() {
        let v = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        assert_relative_eq!(shannon_entropy_vec(&v), 4.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_joint_entropy() {
        // Independent uniform on {0,1} x {0,1}
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let h = joint_entropy(&joint);
        assert_relative_eq!(h, (4.0_f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_conditional_entropy_independent() {
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let h = conditional_entropy(&joint);
        // H(X|Y) = H(X) for independent
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        let joint = DMatrix::from_row_slice(2, 2, &[0.25, 0.25, 0.25, 0.25]);
        let mi = mutual_information(&joint);
        assert_relative_eq!(mi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_dependent() {
        // Perfect correlation
        let joint = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let mi = mutual_information(&joint);
        assert_relative_eq!(mi, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_kolmogorov_sinai() {
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let p = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let h = kolmogorov_sinai_entropy(&mu, &p);
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = vec![0.5, 0.5];
        assert_relative_eq!(kl_divergence(&p, &p), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = vec![1.0, 0.0];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_topological_entropy_from_counts() {
        // Full shift on 2 symbols: |B_n| = 2^n
        let counts: Vec<f64> = (1..=5).map(|n| 2.0_f64.powi(n)).collect();
        let h = topological_entropy_from_counts(&counts);
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 0.01);
    }

    #[test]
    fn test_information_dimension() {
        // H(ε) = 2 * (-log ε), so D = 2
        let scales: Vec<f64> = vec![0.1, 0.01, 0.001];
        let entropies: Vec<f64> = scales.iter().map(|&s: &f64| -2.0_f64 * s.ln()).collect();
        let dim = information_dimension(&entropies, &scales);
        assert_relative_eq!(dim, 2.0, epsilon = 0.01);
    }
}
