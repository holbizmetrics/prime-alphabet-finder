# Binary Index Structure of Primes: Findings

**Date:** January 2026
**Authors:** PROMETHEUS v4.1.1 + Human
**Status:** Concluded - see final assessment

---

## Core Discovery

The binary representation of prime indices (n where p_n is the nth prime) encodes non-trivial information about:
1. The arithmetic properties of p_n (mod 4 residue class)
2. The error term π(x) - Li(x)

This suggests a connection between combinatorial structure (binary) and analytic structure (zeta zeros).

---

## Confirmed Findings (Updated after 5M analysis)

### 1. Chebyshev Bias Splits by Binary Parity - OSCILLATORY

**Observation:** The bias difference OSCILLATES, not grows or decays.

| n_primes | Bias difference | Correlation |
|----------|-----------------|-------------|
| 100,000  | 0.0104 | -0.003 |
| 200,000  | 0.0169 | -0.007 |
| 500,000  | 0.0086 | +0.021 |
| 1,000,000 | 0.0019 | +0.027 |
| 1,200,000 | 0.0026 | -0.008 |
| 1,500,000 | 0.0024 | -0.017 |
| 2,000,000 | 0.0041 | +0.035 |
| 2,400,000 | 0.0049 | +0.077 |

**Key:** Both metrics OSCILLATE around zero with sign changes. This is consistent with zeta-zero-controlled oscillations in π(x) - Li(x).

**Interpretation:** The parity of popcount(n) correlates with p_n mod 4. This is unexpected—there's no obvious reason why the binary structure of the counting index should relate to the residue class of the prime itself.

### 2. RH Error Correlation

**Observation:** Negative correlation between binary density of index and normalized π-error.

| n_primes | Correlation |
|----------|-------------|
| 100,000  | -0.244 |
| 150,000  | -0.254 |
| 160,000  | -0.260 |

**Definition:**
- Binary density = popcount(n) / bit_length(n)
- Normalized error = (π(p_n) - Li(p_n)) / (√p_n × log(p_n))

**Interpretation:** Higher density indices (more 1-bits) correlate with smaller magnitude errors. The error term is controlled by zeta zeros. This suggests binary structure "sees" something about the zeros.

### 3. Path Balance

**Observation:** Consistently ~52% of primes have indices that are "right-heavy" (more 1s than 0s in binary).

| n_primes | Left-heavy | Right-heavy | Balanced |
|----------|------------|-------------|----------|
| 100,000  | 39.4% | 51.8% | 8.8% |
| 150,000  | 39.8% | 52.0% | 8.1% |
| 160,000  | 39.8% | 51.1% | 9.1% |

---

## Critical Discovery: Oscillatory Behavior (5M+ analysis)

The correlation between binary structure and π-error is NOT converging to zero or growing indefinitely. Instead, it **OSCILLATES**:

```
n=500k:  corr = +0.021
n=1.0M:  corr = +0.027
n=1.2M:  corr = -0.008  ← sign change!
n=1.5M:  corr = -0.017
n=2.0M:  corr = +0.035  ← sign change!
n=2.4M:  corr = +0.077
```

This is **exactly what zeta zeros produce** - the error π(x) - Li(x) oscillates due to the explicit formula:

```
π(x) - Li(x) ≈ -Σ_ρ Li(x^ρ)
```

Each zero ρ = 1/2 + iγ contributes an oscillation with frequency γ/2π in log(x).

**Interpretation:** The binary structure of prime indices is detecting these oscillations. The sign changes in correlation correspond to the superposition of oscillations from multiple zeros crossing through zero.

---

## Theoretical Framework

### Connection to Riemann Hypothesis

The prime counting function π(x) satisfies:

```
π(x) = Li(x) + O(√x log x)  [assuming RH]
```

The error term oscillates due to the zeros of ζ(s) on the critical line. The explicit formula:

```
π(x) = Li(x) - Σ_ρ Li(x^ρ) + smaller terms
```

where ρ runs over the non-trivial zeros of ζ(s).

**Our finding:** The binary structure of n = π(p_n) correlates with properties controlled by these zeros.

### Hypothesis

The vertical distribution of zeta zeros creates oscillations in π(x) - Li(x). These oscillations have a "binary shadow"—they manifest as correlations in the binary representation of π.

If this is true:
1. The binary structure might encode partial information about zero locations
2. Patterns in binary index space might reveal structure in the zeros
3. This could provide a new lens for studying RH

---

## Graph Structures Explored

### 1. Binary Tree
- Each prime index traces a path (LSB to MSB)
- Primes at depth d = indices with d bits
- Subtrees partition primes by index prefix

### 2. Hamming Graph
- Connect indices differing by 1 bit
- Single connected component (all primes reachable)
- High degree regularity (most nodes degree 9-10)

### 3. Bit-Position Co-occurrence
- Which bit positions activate together
- Lower bits have higher centrality
- Associations decay with distance

### 4. Hypercube Projection
- Project binary index to 2D/3D
- Density clustering visible
- Gray code ordering reveals structure

---

## Data Files

- `prime_binary_lite.py` - O(n) analysis, scales to millions
- `prime_binary_graphs.py` - Full graph analysis (O(n²) for Hamming)
- `prime_binary_pure.py` - Original embedding framework

---

## Next Steps

1. **Scale to 10^6+** - Run overnight analysis
2. **Zero location test** - Correlate binary structure with actual zeta zero positions
3. **Spectral analysis** - Eigenvalues of Hamming graph Laplacian vs zeta zeros
4. **Prime gaps** - Binary structure of gap sequences

---

## Open Questions

1. **Why does bit-count parity correlate with mod 4 residue?**
   - Is there a number-theoretic explanation?
   - Does this relate to quadratic reciprocity?

2. **What determines the correlation strength?**
   - Why -0.26 and not some other value?
   - Does it converge to a limit?

3. **Can we predict zero locations from binary patterns?**
   - The zeros control π(x) - Li(x)
   - If binary structure correlates, can we reverse-engineer?

4. **Is this related to the Thue-Morse sequence?**
   - Bit-count parity is exactly the Thue-Morse sequence
   - Known connections to number theory

---

## References

- Chebyshev's bias: Rubinstein & Sarnak (1994)
- Explicit formula: Edwards, "Riemann's Zeta Function"
- Thue-Morse and primes: Mauduit & Rivat (2010)
- de Bruijn-Newman constant: Rodgers & Tao (2020)

---

## Final Assessment (January 2026)

### Rigorous Testing with ChatGPT Verification

All findings were subjected to rigorous control tests recommended by ChatGPT. Results:

#### 1. Binary Index Correlations - INCONCLUSIVE

| Test | Result |
|------|--------|
| Block permutation | Mixed - partial collapse |
| Spectral power at zeta zeros | Decays with N - not stable |
| Oscillation in correlation | Real but just sampling different phases |

**Verdict:** The oscillatory correlation is real but doesn't constitute a "new law" - it's consistent with known zeta-zero-controlled oscillations being sampled at different phases.

#### 2. Topological Analysis (Prime Gap Point Clouds) - METHODOLOGICAL FLAW

Initial finding: Real prime gaps showed ~2-7× more ε-edges than Cramér/Markov surrogates.

**Critical test (pair-cloud-preserving surrogate):**
```
Control          | Edges | vs Real
---------------------------------
Real             |  8886 | 1.00x
Bootstrap (same) |  4092 | 2.17x  <- PROBLEM
Shuffled         |  4272 | 2.08x
```

**Root cause identified:** Consecutive triples (g_i, g_{i+1}, g_{i+2}) share two coordinates with adjacent triples. This overlap creates artificial clustering that ANY independent sampling destroys—even from identical distributions.

**Verdict:** The ε-edge metric on consecutive triples measures sequential overlap geometry, not prime-specific structure. The ~2× excess is a property of sequential data with overlap, not a new prime law.

### What We Learned

1. **Primes ≠ Poisson** - Real, but not new (Hardy-Littlewood k-tuple conjecture predicts this)
2. **Binary correlations oscillate** - Real, consistent with known zeta behavior
3. **Topology shows clustering** - Methodological artifact from overlapping windows

### Lessons in Rigorous Hypothesis Testing

1. Always test against proper nulls (Cramér is too weak)
2. High variance across seeds = surrogate mismatch, not discovery
3. Bootstrap from same distribution should give ratio ~1x
4. "Beats shuffled by 2×" can be sequential overlap, not structure

### Files Created

- `spectral_zero_test.py` - Spectral power at zeta zeros
- `block_permutation_test.py` - ChatGPT's recommended control
- `prime_topology.py` - Persistent homology analysis
- `overnight_5M.txt` - 5 million prime analysis results

### Conclusion

No new mathematical discoveries. The investigation was valuable as an exercise in:
- Scaling numerical experiments (5M primes)
- Implementing proper statistical controls
- Collaborative verification with multiple AI systems
- Recognizing methodological artifacts before claiming discovery
