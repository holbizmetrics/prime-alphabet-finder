# Binary Index & Fractal Analysis of Prime Numbers

**Date:** January 2026
**Authors:** PROMETHEUS v4.1.1 + Human
**Verified by:** Multi-AI review (Claude + ChatGPT)

---

## Overview

This directory contains research exploring prime numbers through:
1. **Binary index structure** - Properties of n in binary where p_n is the nth prime
2. **Topological data analysis** - Persistent homology on prime gap point clouds
3. **Fractal dynamics** - Mandelbrot/Mandelbulb escape times with prime-encoded parameters

All findings were rigorously tested with proper null surrogates and multi-AI verification.

---

## Key Results

### 1. Binary Index Correlations
**Status:** Explored, not novel

- Chebyshev bias splits by bit-count parity (oscillates, doesn't converge)
- Correlation with π(x) - Li(x) error term oscillates (consistent with zeta zeros)
- No new structure beyond known phenomena

### 2. Topology on Prime Gaps
**Status:** Methodological artifact identified

- Initial finding: Real gaps showed 2-7× more clustering than controls
- **Fatal flaw:** Bootstrap test revealed overlap artifact (consecutive triples share points)
- Conclusion: ε-edge metric measures sequential geometry, not prime structure

### 3. Fractal Escape Dynamics
**Status:** Valid measurement lens (not new discovery)

- Mapped gap triplets (g_n, g_{n+1}, g_{n+2}) → Mandelbulb coordinates
- Real gaps escape 15% faster than shuffled
- Bootstrap ≈ Real confirms triplet distribution determines behavior
- **Finding:** Detects Hardy-Littlewood k-tuple constraints through nonlinear dynamics
- Robust across multiple fractals (Mandelbulb, Burning Ship 3D)

---

## Files

### Analysis Scripts

| File | Description |
|------|-------------|
| `prime_binary_lite.py` | O(n) binary index analysis, scales to millions |
| `prime_binary_graphs.py` | Graph structures (Hamming, binary tree) |
| `prime_topology.py` | Persistent homology on gap point clouds |
| `prime_mandelbrot.py` | 2D/3D fractal escape time analysis |
| `prime_zero_connection.py` | Correlation with zeta zeros |
| `spectral_zero_test.py` | Spectral power at zeta zero frequencies |
| `block_permutation_test.py` | Control test for binary correlations |
| `overnight_run.py` | Large-scale runner (10M+ primes) |

### Documentation

| File | Description |
|------|-------------|
| `FINDINGS.md` | Complete findings from binary/topology investigation |
| `MANDELBROT_FINDINGS.md` | Fractal dynamics results and final verdict |

---

## Methodology

### Null Surrogates Used
1. **Shuffled** - Destroys all correlations, preserves marginals
2. **Bootstrap** - Samples from same distribution (tests for overlap artifacts)
3. **Pair-exact edge-walk** - Preserves transition frequencies (tests triplet structure)
4. **Markov chain** - Preserves pair transitions (with caveats)

### Multi-Fractal Robustness
Tested across:
- Mandelbulb (3D, power=8)
- Burning Ship 3D variant
- Exponential Julia 3D

---

## Correct Claims

What CAN be claimed:

> "We mapped prime gap triplets into several fractal dynamical systems and measured
> escape times. After eliminating overlap and distributional artifacts via triplet-exact
> bootstrap and pair-exact surrogates, we find that real primes differ from shuffled data
> but match the empirical triplet distribution across multiple fractals. The effect is
> robust across polynomial fractals and reflects known Hardy-Littlewood k-tuple
> constraints, revealed through a nonlinear dynamical lens."

What CANNOT be claimed:
- New prime structure discovered
- Connection to zeta zeros beyond known correlations
- Breakthrough in understanding primes

---

## Running the Code

```bash
# Binary analysis (scales to millions)
python prime_binary_lite.py 1000000

# Topology analysis
python prime_topology.py 50000

# Mandelbrot/Mandelbulb analysis
python prime_mandelbrot.py 30000

# Spectral test at zeta zeros
python spectral_zero_test.py 50000
```

---

## Lessons Learned

1. **Strong signals need rigorous nulls** - "Real beats shuffled" is just the start
2. **Bootstrap reveals overlap artifacts** - If Bootstrap ≠ Real, you have a problem
3. **Multi-AI verification works** - Independent review caught methodological issues
4. **Valid tool ≠ New discovery** - A good measurement lens is still valuable

---

## References

- Chebyshev's bias: Rubinstein & Sarnak (1994)
- Hardy-Littlewood k-tuple conjecture
- Explicit formula: Edwards, "Riemann's Zeta Function"
- 3Blue1Brown: "Why do prime numbers make these spirals?"
