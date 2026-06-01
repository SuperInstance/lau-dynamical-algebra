# lau-dynamical-algebra

**The algebra of dynamical systems** — operator algebras from evolution. Transfer operators, Koopman operators, C\*-algebras, spectral theory, thermodynamic formalism, and Lyapunov exponents, all in pure Rust.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-87-green.svg)](#testing)

---

## Overview

This crate studies dynamical systems through the lens of **operator algebras**. The key insight: a map T: X → X generates a family of linear operators that form algebraic structures revealing the system's deep properties.

| Operator | Acts On | Action | Module |
|---|---|---|---|
| Transfer (Perron-Frobenius) | Density functions | Push densities forward | `transfer` |
| Koopman | Observable functions | Pull observables back | `koopman` |
| C\*-algebra | Matrix elements | Algebra of compositions | `cstar` |
| Ruelle | Functions with potential | Weighted push-forward | `ruelle` |

These are complemented by: spectral decomposition, dynamical zeta functions, thermodynamic formalism, subshifts of finite type, entropy theory, and Lyapunov spectrum computation.

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
lau-dynamical-algebra = "0.1"
```

```rust
use lau_dynamical_algebra::*;

// --- Transfer operator for a 2-cycle ---
let p = TransferOperator::from_map(&[1, 0], 2);
let invariant = p.invariant_measure(1000, 1e-12);
assert!((invariant[0] - 0.5).abs() < 1e-8);

// --- Koopman operator via DMD from data ---
let snapshots = DMatrix::from_row_slice(2, 4, &[
    1.0, 0.5, 0.25, 0.125,
    0.0, 0.5, 0.75, 0.875,
]);
let k = KoopmanOperator::from_trajectories(&snapshots).unwrap();
let radius = k.spectral_radius();

// --- Lyapunov exponents ---
let expanding = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5]));
let spectrum = LyapunovSpectrum::from_matrix(&expanding, 100);
assert!(spectrum.is_chaotic());           // λ₁ > 0
assert!(spectrum.is_dissipative());       // Σ λᵢ < 0
let d_ky = spectrum.kaplan_yorke_dimension(); // fractal dimension
```

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│  Dynamical System T: X → X                         │
│                                                    │
│  ┌─────────────┐     ┌─────────────┐              │
│  │ Transfer Op  │ ←→  │ Koopman Op  │  (adjoints)  │
│  │ (on densities)│    │ (on observ.) │              │
│  └──────┬───────┘     └──────┬───────┘              │
│         │                    │                      │
│  ┌──────▼──────────────────────▼──────┐            │
│  │     Spectral Decomposition          │            │
│  │  eigenvalues, projections, f(A)     │            │
│  └──────────────┬─────────────────────┘            │
│                 │                                  │
│  ┌──────────────▼─────────────────────┐            │
│  │     C*-Algebra of Evolution         │            │
│  │  polynomials, functional calculus    │            │
│  └────────────────────────────────────┘            │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐               │
│  │ Ruelle Op     │  │ Subshift SFT  │               │
│  │ (w/ potential) │  │ (symbolic dyn)│               │
│  └──────┬────────┘  └──────┬────────┘               │
│         │                  │                        │
│  ┌──────▼──────────────────▼────────┐              │
│  │   Thermodynamic Formalism         │              │
│  │  pressure, equilibrium, free energy│              │
│  └──────────────────────────────────┘              │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐               │
│  │ Entropy       │  │ Lyapunov      │               │
│  │ Shannon, KS,  │  │ Spectrum,     │               │
│  │ KL, mutual    │  │ Kaplan-Yorke  │               │
│  │ information   │  │ dimension     │               │
│  └──────────────┘  └──────────────┘               │
│                                                    │
│  ┌────────────────────────────────────┐            │
│  │  Dynamical Zeta Function ζ(z)       │            │
│  │  det(I − zT)⁻¹                     │            │
│  └────────────────────────────────────┘            │
└────────────────────────────────────────────────────┘
```

---

## Modules in Detail

### Transfer Operator (`transfer`)

The Perron-Frobenius operator describes how probability densities evolve under the dynamics. For T: X → X:

> (Pf)(y) = Σ_{T(x)=y} f(x) / |T'(x)|

```rust
// From a stochastic matrix (columns sum to 1)
let p = TransferOperator::from_stochastic_matrix(matrix)?;

// From a deterministic map: map[i] = j means i → j
let p = TransferOperator::from_map(&[1, 2, 0], 3);

// Evolve a density
let new_density = p.apply(&density);

// Find the invariant (stationary) measure via power iteration
let mu = p.invariant_measure(1000, 1e-12);

// Compute iterates P^n
let p10 = p.iterate(10);

// Perron-Frobenius eigenvalue (= 1 for stochastic matrices)
let lambda = p.perron_frobenius_eigenvalue();
```

**Key properties:**
- Stochasticity is validated on construction
- Apply preserves total mass (density sums to 1)
- Invariant measure converges via power iteration

### Koopman Operator (`koopman`)

The adjoint of the transfer operator. Instead of pushing densities forward, it pulls observables back:

> (Kf)(x) = f(T(x))

This linearizes nonlinear dynamics — a nonlinear map on state space becomes a linear operator on function space.

```rust
// From the transfer matrix (K = P^T)
let k = KoopmanOperator::from_transfer_matrix(&transfer_matrix);

// From data via Dynamic Mode Decomposition (DMD)
let k = KoopmanOperator::from_dmd(&X_snapshots, &Y_snapshots)?;

// From a full trajectory matrix
let k = KoopmanOperator::from_trajectories(&trajectory_data)?;

// Apply to an observable
let evolved_observable = k.apply(&f);

// Check if dynamics are measure-preserving (unitary Koopman)
let unitary = k.is_unitary(1e-10);

// Compute eigenvalues and spectral radius
let eigvals = k.eigenvalues();
let sr = k.spectral_radius();
```

### C\*-Algebra (`cstar`)

The algebraic structure generated by repeatedly applying the evolution operator. C\*-algebras capture spectral information through Gelfand duality.

```rust
// Create an algebra element from a matrix
let a = AlgebraElement::from_real(matrix);
let id = AlgebraElement::identity(3);

// C*-norm (operator norm via spectral radius of A*A)
let norm = a.cstar_norm();

// Adjoint (conjugate transpose)
let a_star = a.adjoint();

// Check properties
a.is_self_adjoint(1e-10);  // Hermitian?
a.is_positive(1e-10);       // Positive semi-definite?

// Build the algebra generated by an operator
let alg = EvolutionAlgebra::new(a);

// Compute polynomials in the generator
let p_x = alg.polynomial(&[c0, c1, c2]); // c0*I + c1*A + c2*A²

// Functional calculus
let f_a = alg.functional_calculus_poly(&coefficients);

// Commutator [A, A*] — if zero, algebra is commutative
let is_commutative = alg.is_commutative(1e-10);
```

### Spectral Decomposition (`spectral`)

Eigenvalue computation, spectral projections, and functional calculus for dynamical operators.

```rust
// Full spectral decomposition
let data = spectral_decomposition(&matrix, max_eigenvalues);
// data.eigenvalues, data.eigenvectors, data.spectral_radius

// Power iteration for dominant eigenvalue
let (eigenvalue, eigenvector) = power_iteration(&matrix, 500, 1e-12);

// Spectral projection onto eigenspace
let proj = spectral_projection(&matrix, eigenvalue, 1e-10);

// Functional calculus: compute f(A) for self-adjoint A
let f_a = functional_calculus(&matrix, |lambda| lambda.exp());
```

### Dynamical Zeta Function (`zeta`)

The zeta function ζ(z) = det(I − zT)⁻¹ encodes the spectrum of periodic orbits.

```rust
let zeta = DynamicalZeta::new(transfer_matrix);

// Evaluate at a complex point
let val = zeta.evaluate(Complex::new(0.5, 0.0));

// Determinant det(I − zT)
let det = zeta.determinant(Complex::new(0.3, 0.0));

// Taylor coefficients: ζ(z) = exp(Σ Tr(T^n)/n · z^n)
let coeffs = zeta.taylor_coefficients(10);

// Poles of the zeta function (reciprocals of eigenvalues)
let poles = zeta.poles();

// Radius of convergence (= 1/spectral_radius(T))
let r = zeta.radius_of_convergence();
```

### Thermodynamic Formalism (`thermo`)

Connects statistical mechanics with dynamics: topological pressure, equilibrium states, and the variational principle.

```rust
let sys = ThermodynamicSystem::new(transition_matrix, potential)?;

// Topological pressure P(φ) = log(leading eigenvalue of L_φ)
let pressure = topological_pressure(&sys);

// Equilibrium (Gibbs) state — the measure maximizing h(μ) + ∫φ dμ
let eq = equilibrium_state(&sys, 1000, 1e-12);

// Free energy
let f = free_energy(&sys);

// Check if a measure is an equilibrium state
let is_eq = is_equilibrium(&measure, &sys, 1e-6);

// Measure-theoretic entropy h(μ) = -Σ μ[i] P[i][j] log P[i][j]
let h = measure_entropy(&mu, &transition);

// Potential integral ∫φ dμ
let integral = potential_integral(&potential, &measure);
```

### Ruelle Operator (`ruelle`)

The Ruelle operator is the transfer operator weighted by a potential, central to thermodynamic formalism:

> (L_φ f)(y) = Σ_{T(x)=y} exp(φ(x)) f(x)

```rust
let r = RuelleOperator::new(&transition, &potential)?;

// Leading eigenvalue and pressure
let lambda = r.leading_eigenvalue();
let pressure = r.pressure(); // = log(λ)

// Leading eigenfunction
let h = r.leading_eigenfunction();

// Equilibrium measure via dual operator
let mu = r.equilibrium_measure(1000, 1e-12);

// Correlation function C_n(f,g) = ∫ f · L^n g dμ - (∫f dμ)(∫g dμ)
let c = r.correlation(&f, &g, n, &mu);
```

**Key fact:** The pressure P(φ) equals log of the spectral radius of L_φ. The equilibrium measure is the eigenmeasure of the dual operator.

### Subshift of Finite Type (`subshift`)

Symbolic dynamics defined by a 0-1 transition matrix. The shift space Σ_A consists of all infinite sequences where adjacent symbols are allowed transitions.

```rust
// Built-in examples
let full = SubshiftFiniteType::full_shift(2);   // All binary sequences
let golden = SubshiftFiniteType::golden_mean(); // No consecutive 1s

// Topological entropy h = log(λ_max)
let h = golden.topological_entropy(); // ≈ log((1+√5)/2)

// Count allowed words of length n
let count = golden.count_words(3); // = 5

// Check if a specific word is allowed
assert!(golden.is_allowed_word(&[0, 1]));
assert!(!golden.is_allowed_word(&[1, 1])); // golden mean forbids "11"

// Enumerate all words of a given length
let words = golden.enumerate_words(2); // [[0,0], [0,1], [1,0]]

// Parry measure (measure of maximal entropy)
let mu = golden.parry_measure();

// Zeta function ζ(z) = 1/det(I - zA)
let z = golden.zeta_function(0.5);
```

### Entropy (`entropy`)

Comprehensive entropy toolkit for dynamical systems and information theory.

```rust
// Shannon entropy H(p) = -Σ pᵢ log pᵢ
let h = shannon_entropy(&[0.5, 0.5]); // = ln(2)

// Joint entropy H(X,Y)
let h_xy = joint_entropy(&joint_distribution);

// Conditional entropy H(X|Y)
let h_x_given_y = conditional_entropy(&joint);

// Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
let mi = mutual_information(&joint);

// Kolmogorov-Sinai entropy of a Markov chain
let h_ks = kolmogorov_sinai_entropy(&measure, &transition);

// KL divergence D(P || Q)
let kl = kl_divergence(&p, &q);

// Topological entropy from word counts
let h_top = topological_entropy_from_counts(&counts);

// Information dimension (fractal dimension from entropy scaling)
let dim = information_dimension(&entropies, &scales);
```

### Lyapunov Spectrum (`lyapunov`)

Lyapunov exponents via the Oseledets multiplicative ergodic theorem, using QR-decomposition for numerical stability.

```rust
// From a trajectory of Jacobian matrices
let spectrum = LyapunovSpectrum::from_jacobians(&jacobians);

// From a single matrix (repeated application)
let spectrum = LyapunovSpectrum::from_matrix(&matrix, 1000);

// Key diagnostics
spectrum.is_chaotic();        // Any λ > 0?
spectrum.is_dissipative();    // Σ λᵢ < 0?
spectrum.is_conservative(tol);// Σ λᵢ ≈ 0?

// Pesin entropy: h = Σ λᵢ⁺
let h = spectrum.pesin_entropy();

// Kaplan-Yorke dimension (fractal attractor dimension)
let d = spectrum.kaplan_yorke_dimension();

// Oseledets splitting (Lyapunov subspaces)
let splitting = oseledets_splitting(&jacobians, 0.01);

// Finite-time Lyapunov exponent from time series
let ftle = finite_time_lyapunov(&time_series, epsilon);
```

**Algorithm:** Uses Gram-Schmidt QR decomposition on the accumulated cocycle for numerically stable exponent extraction, following the standard algorithm from Eckmann & Ruelle.

---

## Testing

87 tests covering all modules:

```bash
cargo test
```

Test categories:
- **Transfer operator** — stochasticity, invariant measures, iteration, Perron-Frobenius eigenvalue
- **Koopman operator** — DMD construction, unitarity checks, spectral radius
- **C\*-algebra** — norms, self-adjointness, positivity, commutativity, functional calculus
- **Spectral decomposition** — power iteration, spectral projections, functional calculus
- **Zeta function** — evaluation, poles, Taylor coefficients, radius of convergence
- **Thermodynamic formalism** — pressure, equilibrium states, free energy, entropy
- **Ruelle operator** — leading eigenvalue, pressure, correlations, equilibrium measure
- **Subshifts** — word counting, entropy, Parry measure, word enumeration
- **Entropy** — Shannon, joint, conditional, mutual information, KL divergence, information dimension
- **Lyapunov spectrum** — chaotic/dissipative/conservative detection, Pesin entropy, Kaplan-Yorke dimension

---

## Mathematical Background

### Duality: Transfer vs Koopman

For a map T: X → X preserving measure μ:
- **Transfer operator** P acts on densities: P pushes probability measures forward
- **Koopman operator** K acts on observables: K pulls functions back
- They are **adjoints**: ⟨Kf, g⟩ = ⟨f, Pg⟩

### Thermodynamic Formalism

For a potential φ: X → ℝ:
- **Ruelle operator** L_φ weights transitions by exp(φ)
- **Topological pressure** P(φ) = log(spectral radius of L_φ)
- **Equilibrium state**: the measure μ maximizing h(μ) + ∫φ dμ
- **Variational principle**: P(φ) = sup_μ [h(μ) + ∫φ dμ]

### Lyapunov Exponents and Pesin's Formula

The **Oseledets theorem** guarantees that for almost every trajectory, the limit:

> λᵢ = lim (1/n) log σᵢ(DT^n)

exists and is constant, where σᵢ are singular values. **Pesin's formula** relates these to entropy:

> h(μ) = Σ λᵢ⁺ (sum of positive Lyapunov exponents)

The **Kaplan-Yorke dimension** estimates the fractal dimension of the attractor:

> D_KY = j + (Σᵢ₌₁ʲ λᵢ) / |λ_{j+1}|

where j is the largest index with cumulative sum ≥ 0.

---

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Linear algebra with serde support |
| `num-complex` | Complex number arithmetic |
| `num-traits` | Numeric trait abstractions |
| `serde` / `serde_json` | Serialization |
| `approx` (dev) | Floating-point assertions in tests |

---

## License

MIT
