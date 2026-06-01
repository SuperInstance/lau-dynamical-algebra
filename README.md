# lau-dynamical-algebra

**Operator algebras and ergodic theory for dynamical systems — in pure Rust.**

This crate provides a unified computational framework for the operator-theoretic study of dynamical systems. It brings together transfer operators (Perron-Frobenius), Koopman operators, C\*-algebra elements, spectral analysis, dynamical zeta functions, thermodynamic formalism, Ruelle operators, subshifts of finite type, entropy theory, and Lyapunov exponents — all with first-class matrix operations and 87 property-tested unit tests.

---

## What This Does

| Module | Core Abstraction | What You Can Compute |
|---|---|---|
| `transfer` | Perron-Frobenius transfer operator | Invariant measures, finite-rank approximations |
| `koopman` | Koopman operator (composition operator) | Dynamic Mode Decomposition (DMD), finite approximations |
| `cstar` | C\*-algebra elements | Operator norms, positivity, adjoints, spectrum bounds |
| `spectral` | Spectral decomposition | Eigenvalues, power iteration, spectral radius |
| `zeta` | Dynamical zeta function ζ(z) = det(I − zT)⁻¹ | Poles, Taylor coefficients, radius of convergence |
| `thermo` | Thermodynamic formalism | Topological pressure, equilibrium states, free energy |
| `ruelle` | Ruelle operator with potential | Leading eigenvalue, pressure, equilibrium measure, correlations |
| `subshift` | Subshifts of finite type (SFT) | Topological entropy, Parry measure, word enumeration, zeta function |
| `entropy` | Entropy theory | Shannon, Kolmogorov-Sinai, conditional, KL divergence, information dimension |
| `lyapunov` | Lyapunov spectrum via Oseledets theorem | QR-based exponents, Pesin entropy, Kaplan-Yorke dimension |

---

## Key Idea

In ergodic theory, a dynamical system *T : X → X* is studied not by tracking individual orbits, but through the **operators** those systems induce on function spaces:

- The **Perron-Frobenius (transfer) operator** pushes densities forward: it tells you how probability distributions evolve.
- The **Koopman operator** pulls observables back: it tells you how measurements transform.
- The **Ruelle operator** weights the transfer operator by a potential, connecting dynamics to statistical mechanics.

These operators live in **C\*-algebras** — Banach \*-algebras with a norm satisfying ‖a\*a‖ = ‖a‖². Their spectral properties encode everything: mixing rates, entropy, equilibrium states, and the distribution of periodic orbits (via zeta functions).

This crate makes all of that computational. You build matrices, compose operators, and extract spectral invariants — with the mathematical structures enforced by the type system.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-dynamical-algebra = { git = "https://github.com/SuperInstance/lau-dynamical-algebra" }
```

Or publish to [crates.io](https://crates.io) and use:

```toml
[dependencies]
lau-dynamical-algebra = "0.1"
```

Requires **Rust 2021 edition** (MSRV 1.56+).

### Dependencies

| Crate | Why |
|---|---|
| `nalgebra` 0.33 | Dense linear algebra (matrices, vectors, eigenvalues) |
| `num-complex` 0.4 | Complex number arithmetic for zeta functions |
| `serde` 1 | Serialization of all data structures |
| `approx` 0.5 (dev) | Floating-point tolerance assertions in tests |

---

## Quick Start

### Transfer operator and invariant measure

```rust
use nalgebra::{DMatrix, DVector};
use lau_dynamical_algebra::transfer::TransferOperator;

// Build a stochastic transition matrix (2-state Markov chain)
let t = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
let op = TransferOperator::new(t);

// Compute the invariant measure (stationary distribution)
let mu = op.invariant_measure(1000, 1e-12);
println!("Invariant measure: {:.4}, {:.4}", mu[0], mu[1]);
// → 0.5714, 0.4286
```

### Dynamical zeta function

```rust
use num_complex::Complex;
use lau_dynamical_algebra::zeta::DynamicalZeta;

let t = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, 0.3]));
let zeta = DynamicalZeta::new(t);

// Evaluate ζ(z) at a point
let val = zeta.evaluate(Complex::new(0.5, 0.0));

// Poles are reciprocals of eigenvalues of T
let poles = zeta.poles();

// Taylor coefficients: c_n = Tr(T^n) / n
let coeffs = zeta.taylor_coefficients(5);
```

### Lyapunov exponents and chaos detection

```rust
use lau_dynamical_algebra::lyapunov::LyapunovSpectrum;

// Diagonal map: one expanding, one contracting direction
let m = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5]));
let spectrum = LyapunovSpectrum::from_matrix(&m, 100);

assert!(spectrum.is_chaotic());         // has positive exponent
assert!(spectrum.is_dissipative());     // sum < 0

// Pesin entropy: h = Σ λ⁺
let h = spectrum.pesin_entropy();

// Kaplan-Yorke (fractal) dimension
let d_ky = spectrum.kaplan_yorke_dimension();
```

### Subshift of finite type

```rust
use lau_dynamical_algebra::subshift::SubshiftFiniteType;

// Golden mean shift: no consecutive 1s
let sft = SubshiftFiniteType::golden_mean();

// Topological entropy = log(golden ratio)
let h = sft.topological_entropy();

// Enumerate all allowed words of length 3
let words = sft.enumerate_words(3);
// → [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,0,1]]

// Parry measure (measure of maximal entropy)
let mu = sft.parry_measure();
```

### Thermodynamic formalism

```rust
use lau_dynamical_algebra::thermo::ThermodynamicSystem;

let t = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 0.5]);
let phi = DVector::from_vec(vec![0.0, 0.0]);
let sys = ThermodynamicSystem::new(t, phi).unwrap();

// Topological pressure = log(leading eigenvalue of RPF operator)
let pressure = topological_pressure(&sys);

// Equilibrium (Gibbs) state
let eq = equilibrium_state(&sys, 1000, 1e-12);
```

---

## API Reference

### `transfer` — Perron-Frobenius Transfer Operator

| Signature | Description |
|---|---|
| `TransferOperator::new(matrix)` | Build from a stochastic matrix |
| `op.apply(f)` | Apply operator to a density vector |
| `op.iterate(n)` | Compose operator *n* times |
| `op.invariant_measure(max_iter, tol)` | Fixed-point stationary distribution |
| `op.leading_eigenvalue()` | Spectral radius via power iteration |
| `op.adjoint()` | Dual (Koopman-like) operator |

### `koopman` — Koopman Operator & DMD

| Signature | Description |
|---|---|
| `KoopmanOperator::from_data(X, Y, basis)` | Construct from snapshot matrices |
| `koop.apply(f)` | Pull back an observable |
| `koop.dmd_eigenvalues()` | Dynamic Mode Decomposition eigenvalues |
| `koop.dmd_modes()` | DMD mode vectors |
| `koop.finite_approx(basis)` | Galerkin projection onto basis functions |

### `cstar` — C\*-Algebra Elements

| Signature | Description |
|---|---|
| `CStarElement::new(matrix)` | Wrap a matrix as a C\*-element |
| `el.norm()` | Operator norm (largest singular value) |
| `el.is_positive(tol)` | Check positive semi-definiteness |
| `el.is_self_adjoint(tol)` | Check Hermiticity |
| `el.adjoint()` | Conjugate transpose |
| `el.spectrum_bounds()` | Gershgorin circle bounds |

### `spectral` — Spectral Decomposition

| Signature | Description |
|---|---|
| `spectral_decomposition(mat, n)` | Eigenvalues + spectral radius |
| `power_iteration(mat, n_iter)` | Leading eigenvector/value pair |

### `zeta` — Dynamical Zeta Function

| Signature | Description |
|---|---|
| `DynamicalZeta::new(operator)` | Build ζ(z) from transfer operator |
| `zeta.evaluate(z)` | Evaluate ζ(z) = 1/det(I − zT) |
| `zeta.determinant(z)` | Compute det(I − zT) |
| `zeta.poles()` | Poles = 1/λᵢ (reciprocals of eigenvalues) |
| `zeta.taylor_coefficients(order)` | cₙ = Tr(Tⁿ)/n |
| `zeta.radius_of_convergence()` | 1/spectral_radius(T) |
| `zeta.derivative_log_zeta(z, h)` | d/dz log ζ(z) via finite differences |

### `thermo` — Thermodynamic Formalism

| Signature | Description |
|---|---|
| `ThermodynamicSystem::new(transition, potential)` | Build system with transition matrix + potential |
| `ruelle_pf_matrix(&sys)` | Ruelle-Perron-Frobenius operator matrix |
| `topological_pressure(&sys)` | log(leading eigenvalue of RPF operator) |
| `equilibrium_state(&sys, max_iter, tol)` | Gibbs/equilibrium measure |
| `measure_entropy(mu, transition)` | Kolmogorov-Sinai entropy of μ |
| `potential_integral(phi, mu)` | ∫φ dμ |
| `free_energy(&sys)` | P(φ) − sup ∫φ dμ |
| `is_equilibrium(mu, &sys, tol)` | Check variational principle equality |

### `ruelle` — Ruelle Operator

| Signature | Description |
|---|---|
| `RuelleOperator::new(transition, potential)` | Weighted transfer operator |
| `r.apply(f)` | Apply L\_φ to function f |
| `r.pressure()` | log(leading eigenvalue) |
| `r.leading_eigenfunction()` | Eigenfunction of leading eigenvalue |
| `r.equilibrium_measure(max_iter, tol)` | Equilibrium state via dual operator |
| `r.correlation(f, g, n, mu)` | Cₙ(f,g) = ∫f·Lⁿg dμ − (∫f dμ)(∫g dμ) |
| `r.dual()` | Adjoint operator |

### `subshift` — Subshifts of Finite Type

| Signature | Description |
|---|---|
| `SubshiftFiniteType::new(adjacency)` | From 0-1 adjacency matrix |
| `SubshiftFiniteType::full_shift(n)` | Full shift on *n* symbols |
| `SubshiftFiniteType::golden_mean()` | Classic golden mean shift |
| `sft.topological_entropy()` | log(λ\_max) of adjacency matrix |
| `sft.count_words(n)` | Number of allowed words of length *n* |
| `sft.enumerate_words(length)` | List all allowed words |
| `sft.parry_measure()` | Measure of maximal entropy |
| `sft.zeta_function(z)` | det(I − zA) |
| `sft.is_irreducible()` | Check strong connectivity |

### `entropy` — Entropy Theory

| Signature | Description |
|---|---|
| `shannon_entropy(prob)` | H(p) = −Σ pᵢ log pᵢ |
| `joint_entropy(joint)` | H(X,Y) from joint distribution |
| `conditional_entropy(joint)` | H(X\|Y) |
| `mutual_information(joint)` | I(X;Y) = H(X) + H(Y) − H(X,Y) |
| `kolmogorov_sinai_entropy(mu, P)` | Markov chain KS entropy |
| `kl_divergence(p, q)` | D(P‖Q) |
| `topological_entropy_from_counts(counts)` | From word count growth rates |
| `information_dimension(entropies, scales)` | Fractal information dimension |

### `lyapunov` — Lyapunov Spectrum & Oseledets

| Signature | Description |
|---|---|
| `LyapunovSpectrum::from_jacobians(jacobians)` | QR-based from trajectory Jacobians |
| `LyapunovSpectrum::from_matrix(M, n_iter)` | From repeated matrix application |
| `spectrum.is_chaotic()` | Any positive exponent? |
| `spectrum.is_dissipative()` | Sum of exponents < 0? |
| `spectrum.pesin_entropy()` | h = Σ λ⁺ |
| `spectrum.kaplan_yorke_dimension()` | Fractal dimension estimate |
| `oseledets_splitting(jacobians, tol)` | Lyapunov subspaces |
| `finite_time_lyapunov(series, ε)` | From time series via nearest-neighbor |

---

## How It Works

### Operator-theoretic pipeline

```
Dynamical System T: X → X
        │
        ├─→ Transfer Operator (Perron-Frobenius)  ──→ Invariant measures
        │                                            mixing rates
        │                                            ergodicity
        │
        ├─→ Koopman Operator (composition)      ──→ Eigenfunctions
        │                                            DMD modes
        │                                            Spectral decomposition
        │
        ├─→ Ruelle Operator (weighted transfer)  ──→ Pressure P(φ)
        │                                            Equilibrium states
        │                                            Correlation decay
        │
        └─→ Linearization (Jacobian)            ──→ Lyapunov exponents
                                                 Pesin entropy
                                                 Kaplan-Yorke dimension
```

Each operator is represented as a finite-dimensional matrix (Galerkin projection onto a basis). The spectral properties of these matrices approximate the infinite-dimensional truth to whatever resolution your basis provides.

### Numerical methods

- **Power iteration** for leading eigenvalues/eigenvectors (transfer, Koopman, Ruelle, spectral)
- **QR decomposition** (Gram-Schmidt) for Lyapunov exponents — the standard Benettin algorithm
- **Cofactor expansion** for determinants of complex matrices (zeta function)
- **Gershgorin circles** for spectral bounds (C\*-algebra)
- **Deflation** for multi-eigenvalue extraction (zeta poles)
- **Linear regression** for entropy from word counts and information dimension

---

## The Math

### Perron-Frobenius operator

For a non-singular transformation *T* on a measure space *(X, μ)*, the **Perron-Frobenius operator** *P* acts on *L¹* functions by:

> (Pf)(y) = Σ_{T(x)=y} f(x) / |det DT(x)|

In matrix form: *P = T^⊤* (the transpose of the stochastic transition matrix). The **invariant measure** is the fixed point *Pμ = μ*, computed via power iteration.

### Koopman operator

The **Koopman operator** *K* is the adjoint of the transfer operator:

> (Kg)(x) = g(T(x))

It acts on observables (functions) rather than densities. **Dynamic Mode Decomposition (DMD)** approximates the eigenfunctions of *K* from data: given snapshot pairs *(X, Y = TX)*, compute *A = YX⁺* and find its eigen-decomposition.

### C\*-algebra

A **C\*-algebra** is a Banach \*-algebra *A* satisfying ‖a\*a‖ = ‖a‖² for all *a ∈ A*. In the matrix setting, elements are self-adjoint (Hermitian) operators with the operator norm. The **spectral radius** ρ(a) = sup|σ(a)| satisfies ρ(a) ≤ ‖a‖.

### Dynamical zeta function

For a transfer operator *T*:

> ζ(z) = det(I − zT)⁻¹ = exp(Σ_{n≥1} Tr(Tⁿ) zⁿ / n)

The **poles** of ζ are at *z = 1/λᵢ* where *λᵢ* are eigenvalues of *T*. The **radius of convergence** is *1/ρ(T)*. These encode the periodic orbit structure: the zeta function "knows" about every periodic point of the system.

### Thermodynamic formalism

Given a potential *φ: X → ℝ*:

- The **Ruelle operator** *L\_φ* weights the transfer operator by *exp(φ)*
- **Topological pressure**: P(φ) = log(λ\_max(L\_φ))
- **Equilibrium state**: the unique invariant measure *μ* maximizing *h(μ) + ∫φ dμ*
- **Variational principle**: P(φ) = sup\_μ {h(μ) + ∫φ dμ}

This bridges dynamics and statistical mechanics: pressure plays the role of free energy, entropy is the thermodynamic entropy, and the equilibrium state is the Gibbs measure.

### Subshifts of finite type

An SFT is defined by a 0-1 adjacency matrix *A*. The allowed sequences are *Σ\_A = {x : A[x\_i][x\_{i+1}] = 1 ∀i}*. Key invariants:

- **Topological entropy**: h\_top = log(ρ(A)) — exponential growth rate of allowed words
- **Parry measure**: the measure of maximal entropy, constructed from left/right Perron eigenvectors
- **Zeta function**: ζ\_A(z) = 1/det(I − zA)

The **golden mean shift** (forbidden word: "11") has h\_top = log(φ) where φ = (1+√5)/2.

### Entropy theory

| Concept | Formula |
|---|---|
| Shannon entropy | H(p) = −Σ pᵢ log pᵢ |
| Kolmogorov-Sinai entropy | h\_KS(μ) = −Σ μᵢ Pᵢⱼ log Pᵢⱼ |
| Conditional entropy | H(X\|Y) = −Σ p(x,y) log p(x\|y) |
| Mutual information | I(X;Y) = H(X) + H(Y) − H(X,Y) |
| KL divergence | D(P‖Q) = Σ pᵢ log(pᵢ/qᵢ) |
| Information dimension | D = lim\_{ε→0} H(ε)/log(1/ε) |

### Lyapunov exponents

By the **Oseledets multiplicative ergodic theorem**, for a.e. trajectory of a smooth dynamical system, the limit:

> λᵢ = lim\_{n→∞} (1/n) log σᵢ(DTⁿ)

exists and gives the **Lyapunov spectrum** *λ₁ ≥ λ₂ ≥ … ≥ λ\_d*. These measure exponential expansion/contraction rates:

- **Chaotic** ↔ at least one λᵢ > 0
- **Pesin's formula**: h = Σ λᵢ⁺ (entropy = sum of positive exponents)
- **Kaplan-Yorke dimension**: D\_KY = j + (Σᵢ₌₁ʲ λᵢ)/|λ\_{j+1}| where j maximizes the partial sum ≥ 0

The QR-based algorithm decomposes the product *Jₙ · Jₙ₋₁ · … · J₁ = QₙRₙ · … · Q₁R₁* and reads the Lyapunov exponents from the diagonal of the accumulated *R* matrices.

---

## Test Coverage

87 unit tests covering all 10 modules:

| Module | Tests | Key Assertions |
|---|---|---|
| `transfer` | 9 | Invariant measures, adjoints, leading eigenvalues |
| `koopman` | 9 | DMD eigenvalues, finite approximations, basis projections |
| `cstar` | 9 | Norms, positivity, self-adjointness, Gershgorin bounds |
| `spectral` | 6 | Eigenvalues, spectral radius, power iteration |
| `zeta` | 8 | Evaluation, poles, Taylor coefficients, convergence radius |
| `thermo` | 7 | Pressure, equilibrium states, entropy, dimension validation |
| `ruelle` | 7 | Pressure, equilibrium measures, correlation decay |
| `subshift` | 11 | Golden mean entropy, word counting, Parry measure, irreducibility |
| `entropy` | 12 | Shannon, conditional, mutual info, KL, topological, info dimension |
| `lyapunov` | 9 | Chaotic/dissipative/conservative checks, Pesin, Kaplan-Yorke |

Run with:

```bash
cargo test
```

---

## Project Structure

```
src/
├── lib.rs            # Module re-exports
├── transfer.rs       # Perron-Frobenius transfer operator
├── koopman.rs        # Koopman operator & DMD
├── cstar.rs          # C*-algebra elements
├── spectral.rs       # Spectral decomposition
├── zeta.rs           # Dynamical zeta function
├── thermo.rs         # Thermodynamic formalism
├── ruelle.rs         # Ruelle operator
├── subshift.rs       # Subshifts of finite type
├── entropy.rs        # Entropy theory
└── lyapunov.rs       # Lyapunov spectrum & Oseledets
```

---

## License

MIT
