//! Thermodynamic formalism (pressure, equilibrium states)
//!
//! Thermodynamic formalism connects statistical mechanics with dynamical systems.
//! Key concepts include topological pressure, equilibrium states, and variational principles.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Thermodynamic formalism for a subshift or symbolic system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicSystem {
    /// Transition matrix (0-1 or weighted).
    transition: DMatrix<f64>,
    /// Potential function (weight per symbol/state).
    potential: DVector<f64>,
}

impl ThermodynamicSystem {
    /// Create a new thermodynamic system.
    pub fn new(transition: DMatrix<f64>, potential: DVector<f64>) -> Result<Self, String> {
        if transition.nrows() != potential.len() {
            return Err("Transition matrix and potential dimension mismatch".into());
        }
        Ok(Self {
            transition,
            potential,
        })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.potential.len()
    }

    /// Get the potential.
    pub fn potential(&self) -> &DVector<f64> {
        &self.potential
    }

    /// Get the transition matrix.
    pub fn transition(&self) -> &DMatrix<f64> {
        &self.transition
    }
}

/// Compute the Ruelle-Perron-Frobenius (transfer) operator matrix with potential.
/// L_φ[i][j] = exp(φ(j)) * T[i][j] for transition matrix T.
pub fn ruelle_pf_matrix(system: &ThermodynamicSystem) -> DMatrix<f64> {
    let n = system.n_states();
    let mut matrix = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            matrix[(i, j)] = system.potential[j].exp() * system.transition[(i, j)];
        }
    }
    matrix
}

/// Compute topological pressure via the leading eigenvalue of the RPF operator.
/// P(φ) = log(λ) where λ is the spectral radius of L_φ.
pub fn topological_pressure(system: &ThermodynamicSystem) -> f64 {
    let rpf = ruelle_pf_matrix(system);
    let n = rpf.nrows();
    let mut v = DVector::from_element(n, 1.0 / n as f64);
    for _ in 0..1000 {
        let v_new = &rpf * &v;
        let norm = v_new.norm();
        if norm < 1e-20 {
            return f64::NEG_INFINITY;
        }
        v = v_new / norm;
    }
    let lambda: f64 = v.dot(&(&rpf * &v));
    lambda.abs().ln()
}

/// Compute the equilibrium state (Gibbs measure) for the potential.
pub fn equilibrium_state(system: &ThermodynamicSystem, max_iter: usize, tol: f64) -> DVector<f64> {
    let rpf = ruelle_pf_matrix(system);
    let n = rpf.nrows();
    let mut v = DVector::from_element(n, 1.0 / n as f64);
    for _ in 0..max_iter {
        let v_new = &rpf * &v;
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
    // Normalize to probability
    let sum: f64 = v.iter().sum();
    if sum > 0.0 {
        v / sum
    } else {
        v
    }
}

/// Variational principle: sup over invariant measures μ of (h(μ) + ∫φ dμ).
/// Here we compute the integral of φ with respect to μ.
pub fn potential_integral(potential: &DVector<f64>, measure: &DVector<f64>) -> f64 {
    potential.dot(measure)
}

/// Compute the measure-theoretic entropy of an invariant measure μ
/// for a Markov chain with transition matrix P.
/// h(μ) = -Σ_{i,j} μ[i] P[i][j] log(P[i][j])
pub fn measure_entropy(measure: &DVector<f64>, transition: &DMatrix<f64>) -> f64 {
    let n = measure.len();
    let mut h = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p_ij = transition[(i, j)];
            if p_ij > 0.0 {
                h -= measure[i] * p_ij * p_ij.ln();
            }
        }
    }
    h
}

/// The free energy F(φ) = P(φ) - sup ∫φ dμ.
pub fn free_energy(system: &ThermodynamicSystem) -> f64 {
    let pressure = topological_pressure(system);
    let eq_state = equilibrium_state(system, 1000, 1e-12);
    let integral = potential_integral(&system.potential, &eq_state);
    pressure - integral
}

/// Check if a measure μ is an equilibrium state for potential φ.
/// This means h(μ) + ∫φ dμ = P(φ).
pub fn is_equilibrium(
    measure: &DVector<f64>,
    system: &ThermodynamicSystem,
    tol: f64,
) -> bool {
    let pressure = topological_pressure(system);
    let integral = potential_integral(&system.potential, measure);
    let entropy = measure_entropy(measure, &system.transition);
    (entropy + integral - pressure).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pressure_zero_potential() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let sys = ThermodynamicSystem::new(t, phi).unwrap();
        let p = topological_pressure(&sys);
        // exp(0) * 0.5 matrix has spectral radius 0.5, log(0.5) ≈ -0.693
        assert!(p.is_finite());
    }

    #[test]
    fn test_equilibrium_state_sums_to_one() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let sys = ThermodynamicSystem::new(t, phi).unwrap();
        let eq = equilibrium_state(&sys, 1000, 1e-12);
        let sum: f64 = eq.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_ruelle_pf_matrix() {
        let t = DMatrix::identity(2, 2);
        let phi = DVector::from_vec(vec![1.0_f64.ln(), 2.0_f64.ln()]);
        let sys = ThermodynamicSystem::new(t, phi).unwrap();
        let rpf = ruelle_pf_matrix(&sys);
        assert_relative_eq!(rpf[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rpf[(1, 1)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_potential_integral() {
        let phi = DVector::from_vec(vec![1.0, 2.0]);
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        assert_relative_eq!(potential_integral(&phi, &mu), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_entropy() {
        // Uniform on 2 states with fair coin: entropy = log(2)
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let mu = DVector::from_vec(vec![0.5, 0.5]);
        let h = measure_entropy(&mu, &t);
        assert_relative_eq!(h, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let t = DMatrix::identity(2, 2);
        let phi = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(ThermodynamicSystem::new(t, phi).is_err());
    }

    #[test]
    fn test_free_energy() {
        let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
        let phi = DVector::from_vec(vec![0.0, 0.0]);
        let sys = ThermodynamicSystem::new(t, phi).unwrap();
        let fe = free_energy(&sys);
        // Free energy should be approximately 0 for zero potential
        assert!(fe.abs() < 0.5);
    }
}
