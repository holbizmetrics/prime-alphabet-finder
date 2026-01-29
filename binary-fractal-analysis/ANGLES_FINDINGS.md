# Prime Gaps as Rotation Angles - Findings

**Date:** January 2026
**Status:** VERIFIED - Novel visualization of known structure

---

## Core Discovery

Map prime gaps to rotation angles in a 2D walk:
- Each step: rotate by g_n × mult degrees, move forward 1 unit
- At **mult=12°**, Real walk is **25-200× longer** than shuffled

## Why It Works

The cumulative angle at step n:
```
θ_n = (Σg_i × 12) mod 360
    = ((p_n - 2) × 12) mod 360
    = (p_n mod 30) × 12°
```

Since 360/12 = 30, the walk direction is determined by **p_n mod 30**.

### The 8 Residue Classes

Primes > 5 occupy only 8 classes mod 30:

| Class | Angle | Unit Vector |
|-------|-------|-------------|
| 1 | 12° | (0.98, 0.21) |
| 7 | 84° | (0.10, 0.99) |
| 11 | 132° | (−0.67, 0.74) |
| 13 | 156° | (−0.91, 0.41) |
| 17 | 204° | (−0.91, −0.41) |
| 19 | 228° | (−0.67, −0.74) |
| 23 | 276° | (0.10, −0.99) |
| 29 | 348° | (0.98, −0.21) |

**Sum: (−1, 0)** → consistent leftward drift

### Walk Efficiency

- Expected drift per step: (−0.125, 0)
- Walk efficiency: **12.5%** of theoretical maximum
- Linear in N (verified up to 50k primes)

---

## Residual Analysis

After subtracting predicted drift (N = 50,000 primes):

| Metric | Value |
|--------|-------|
| Residual as % of actual | 0.44% |
| Residual vs √N | 0.12× |
| Walk explanation | 98%+ by mod 30 |

**Null model:** Randomized-order sequence with same residue class distribution
(i.e., bootstrap with shuffled ordering while preserving marginal frequencies).

**Key finding:** Residual is **~10× smaller** than the randomized-order null.

The observed residue sequence has smaller fluctuation than shuffled controls,
because shuffling breaks the telescoping link p_n = 2 + Σg_i at step n.

---

## Chebyshev Bias in Residual

Mod 4 grouping of residue classes:
- 1 (mod 4): {1, 13, 17, 29} → vector sum V₁ = (0.13, 0)
- 3 (mod 4): {7, 11, 19, 23} → vector sum V₃ = (−1.13, 0)

Difference: **Δ = V₃ − V₁ = (−1.26, 0)** (leftward)

### Derivation of Chebyshev Contribution

If 3 (mod 4) primes lead by B primes (typical B ≈ 30-80 for N = 50k):
```
Chebyshev x-drift ≈ B × (Δx / 4) = B × (−1.26 / 4) ≈ −0.315 × B
```

For B = 30: x-drift ≈ −9.4

Observed residual x-component: ~10-40 (varies with N).

**Conclusion:** Chebyshev bias contributes ~30-40% of residual x-component
when the mod 4 race is active.

---

## Other Resonant Multipliers

| Mult | Ratio (Real/Shuf) | What it reveals |
|------|-------------------|-----------------|
| 12° | 215× | Mod 30 structure |
| 24° | 31× | Mod 15 |
| 36° | 46× | Mod 10 |
| 60° | 48× | Mod 6 |
| 72° | 24× | Mod 5 |

All resonances occur at divisors of 360 related to prime admissibility.

---

## Verification

**Bootstrap test (N = 20k primes, mult = 12°):**
```
Real:      2541
Bootstrap: ~150 (same distribution, random order)
Shuffled:  ~100
```

Effect is about **sequential ordering**, not just distribution.

**Why shuffling fails:** Shuffling breaks the telescoping identity
`p_n = 2 + Σg_i` at step n. The cumulative sum of shuffled gaps no longer
equals an actual prime, so the mod 30 residue structure is destroyed.

---

## What This Visualizes

1. **Dirichlet equidistribution** - main drift (98%+)
2. **Chebyshev bias** - tiny residual correction
3. **Prime regularity** - sub-random residual

---

## Correct Claims

**Is:** Novel visualization technique for prime residue structure.

**Is not:** New prime law or hidden structure.

**Safe statement:**
> "Angle walks with gap-derived rotations visualize prime residue class
> distribution. At mult=12°, the walk direction equals (p_n mod 30) × 12°,
> revealing Dirichlet equidistribution geometrically. The residual after
> subtracting predicted drift is sub-random, with Chebyshev bias contributing
> ~34% of the small remaining structure."

---

## Files

- `prime_angles.py` - Full implementation with residual analysis
