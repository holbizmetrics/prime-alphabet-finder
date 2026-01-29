# Prime Gaps in Mandelbrot/Mandelbulb Dynamics

**Date:** January 2026
**Status:** COMPLETE - Validated measurement lens for k-tuple constraints

---

## Core Idea

Map prime gap sequences to fractal parameters:
- **2D Mandelbrot:** c = scale × (g_n + i·g_{n+1})
- **3D Mandelbulb:** c = scale × (g_n, g_{n+1}, g_{n+2})

Study escape times and compare to shuffled controls.

---

## Results

### 2D Mandelbrot (20k primes, scale=0.1)

| Control | Mean Escape | vs Real |
|---------|-------------|---------|
| Real | 7.66 | 1.00 |
| Shuffled | 8.65 | 0.89 |

Real gaps escape **11% faster**.

### 3D Mandelbulb (20k primes, scale=0.1, power=8)

| Control | Mean Escape | Ratio |
|---------|-------------|-------|
| Real | 5.06 | 1.00 |
| Bootstrap | 4.89 | 1.03 |
| Shuffled | 5.86 | 0.86 |

Real gaps escape **14% faster** than shuffled.

### Scale Dependence

| Scale | Real | Shuffled | Ratio |
|-------|------|----------|-------|
| 0.05 | 26.0 | 26.0 | 1.00 |
| 0.10 | 5.06 | 6.01 | 0.84 |
| 0.15 | 1.71 | 2.20 | 0.78 |

Effect strengthens at larger scales (where more triplets escape quickly).

---

## Interpretation

### Bootstrap Test
- Bootstrap (sampling from same triplet distribution) ≈ Real
- This means the **triplet distribution itself** determines escape behavior
- NOT a sequential overlap artifact

### What It Detects
- Shuffling destroys pair correlations between consecutive gaps
- Real prime gaps have Hardy-Littlewood correlations
- The Mandelbulb dynamics provide a **novel measure** of these correlations

### Hierarchy of Surrogates

| Surrogate | Escape | Interpretation |
|-----------|--------|----------------|
| Pair-preserving (pair + random 3rd) | 3.4 | Escapes too fast |
| **Real** | 5.1 | Actual prime structure |
| Bootstrap (same triplets) | 4.9 | ≈ Real |
| Shuffled | 5.9 | No correlations |

Real is "in between" - it has specific triplet constraints that neither extreme captures.

---

## Novelty Assessment

**Novel aspect:** Using Mandelbulb escape dynamics to probe prime gap structure. No prior work found on this specific approach.

**Not novel:** The underlying phenomenon (pair correlations in prime gaps) is well-known from Hardy-Littlewood.

**Question for verification:** Does the Mandelbulb dynamics reveal anything *beyond* pair correlations, or is it just a fancy way to measure them?

---

## Technical Details

### Mandelbulb Iteration
```
z_{n+1} = z_n^8 + c (in spherical coordinates)

Spherical power:
- r → r^8
- θ → 8θ
- φ → 8φ
```

### Code
See `prime_mandelbrot.py` for full implementation.

---

## Files

- `prime_mandelbrot.py` - Full 2D/3D analysis
- `run_mandelbrot_test.py` - Quick test script

---

## Final Verdict (Verified by Multi-AI Review)

### What Was Established

| Claim | Status | Evidence |
|-------|--------|----------|
| Signal is not artifact | ✅ Confirmed | Bootstrap ≈ Real (triplet distribution matches) |
| Not overlap artifact | ✅ Confirmed | Unlike topology test, this passed |
| Measures triplet structure | ✅ Confirmed | Pair-exact surrogate diverges |
| Robust across fractals | ✅ Confirmed | Mandelbulb, Burn3D show same pattern |
| New prime structure | ❌ Not established | Effect explained by Hardy-Littlewood |

### Correct Characterization

> "A nonlinear dynamical observable that measures Hardy-Littlewood k-tuple
> constraints via escape-time statistics."

**Not:** A new law, hidden structure, or zeta-zero encoding.

**Is:** A clean detector, rigorously null-tested, artifact-controlled.

### Key Finding: Forbidden Triplets

Real primes have triplet-level constraints beyond pair frequencies:
- (2,6,2): Real=0, EdgeWalk=45
- (4,6,4): Real=0, EdgeWalk=87

These are textbook consequences of mod-6 residue constraints, predicted by Hardy-Littlewood.

### Multi-Fractal Robustness

| Fractal | Boot/Real | Real/Shuf | Interpretation |
|---------|-----------|-----------|----------------|
| Mandelbulb | 1.02 | 0.86 | Robust |
| Burn3D | ~1.0 | ~0.85 | Robust |
| Exp3D | ~1.0 | 1.18 | Different weighting |

### Safe Public Claim

> "We mapped prime gap triplets into several fractal dynamical systems and
> measured escape times. After eliminating overlap and distributional artifacts
> via triplet-exact bootstrap and pair-exact surrogates, we find that real
> primes differ from shuffled data but match the empirical triplet distribution
> across multiple fractals. The effect is robust across polynomial fractals and
> reflects known Hardy-Littlewood k-tuple constraints, revealed through a
> nonlinear dynamical lens."

---

## Investigation Complete

This investigation demonstrated:
1. Strong signal found
2. Destroyed as "new discovery"
3. Preserved as valid measurement lens

The difference between "Real beats null by 60%" and "why does it beat THIS null but not THAT one" is research maturity.
