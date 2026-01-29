"""
Prime Binary Index Analysis - LITE VERSION
==========================================
Optimized for scale: O(n) operations only, no O(n²) graphs.

Focus on the interesting findings:
1. Chebyshev bias split by binary parity
2. Path balance (left/right heavy)
3. Bit density distribution
4. RH error correlation

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
from collections import defaultdict

def sieve_primes(n):
    """Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def first_n_primes(n):
    """Get first n primes."""
    if n < 6:
        limit = 15
    else:
        ln_n = math.log(n)
        limit = int(n * (ln_n + math.log(ln_n) + 2))
    primes = sieve_primes(limit)
    while len(primes) < n:
        limit = int(limit * 1.5)
        primes = sieve_primes(limit)
    return primes[:n]

def li(x):
    """Logarithmic integral approximation."""
    if x <= 2:
        return 0
    result = 0
    term = 1
    ln_x = math.log(x)
    for k in range(1, 100):
        term *= ln_x / k
        result += term / k
        if abs(term) < 1e-10:
            break
    return result + math.log(ln_x) + 0.5772156649

def analyze(n_primes):
    """Main analysis - all O(n) operations."""

    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]:,}")
    print()

    # ===== 1. CHEBYSHEV BIAS BY BINARY PARITY =====
    # This was the most interesting finding

    cheb = {
        'even_bits': {'mod1': 0, 'mod3': 0, 'total': 0},
        'odd_bits': {'mod1': 0, 'mod3': 0, 'total': 0}
    }

    # Also track by specific bit counts
    by_bitcount = defaultdict(lambda: {'mod1': 0, 'mod3': 0})

    # ===== 2. PATH BALANCE =====
    path_stats = {'left': 0, 'right': 0, 'balanced': 0}

    # ===== 3. BIT DENSITY =====
    density_buckets = defaultdict(int)  # density rounded to 0.1

    # ===== 4. RH ERROR TRACKING =====
    rh_data = []  # (n, density, normalized_error)

    # ===== 5. DEPTH DISTRIBUTION =====
    depth_counts = defaultdict(int)

    # Single pass through all primes
    for i, p in enumerate(primes):
        n = i + 1

        # Binary properties of index
        bits = bin(n).count('1')
        bit_len = n.bit_length()
        density = bits / bit_len if bit_len > 0 else 0

        # Depth
        depth_counts[bit_len] += 1

        # Density bucket
        bucket = round(density, 1)
        density_buckets[bucket] += 1

        # Path balance
        zeros = bit_len - bits
        if bits > zeros:
            path_stats['right'] += 1
        elif zeros > bits:
            path_stats['left'] += 1
        else:
            path_stats['balanced'] += 1

        # Chebyshev (skip p=2)
        if p > 2:
            mod4 = p % 4
            key = 'mod1' if mod4 == 1 else 'mod3'

            if bits % 2 == 0:
                cheb['even_bits'][key] += 1
                cheb['even_bits']['total'] += 1
            else:
                cheb['odd_bits'][key] += 1
                cheb['odd_bits']['total'] += 1

            by_bitcount[bits][key] += 1

        # RH error (sample every 100th to save memory at scale)
        if n % 100 == 0 or n <= 1000:
            if p > 2:
                li_p = li(p)
                error = n - li_p
                norm_error = error / (math.sqrt(p) * math.log(p))
                rh_data.append((n, density, norm_error))

    # ===== OUTPUT =====

    print("="*60)
    print("1. CHEBYSHEV BIAS BY BINARY PARITY OF INDEX")
    print("="*60)
    print()
    print("If primes were randomly distributed w.r.t. binary structure,")
    print("the mod4 split should be ~same for even/odd bit-count indices.")
    print()

    e = cheb['even_bits']
    o = cheb['odd_bits']

    e_ratio = e['mod3'] / e['mod1'] if e['mod1'] > 0 else 0
    o_ratio = o['mod3'] / o['mod1'] if o['mod1'] > 0 else 0

    print(f"Even bit-count indices ({e['total']:,} primes):")
    print(f"  p ≡ 1 (mod 4): {e['mod1']:,}")
    print(f"  p ≡ 3 (mod 4): {e['mod3']:,}")
    print(f"  Ratio (3/1):   {e_ratio:.4f}")
    print()
    print(f"Odd bit-count indices ({o['total']:,} primes):")
    print(f"  p ≡ 1 (mod 4): {o['mod1']:,}")
    print(f"  p ≡ 3 (mod 4): {o['mod3']:,}")
    print(f"  Ratio (3/1):   {o_ratio:.4f}")
    print()
    print(f"BIAS DIFFERENCE: {abs(e_ratio - o_ratio):.4f}")
    print()

    # Detailed by bit count
    print("By specific bit count:")
    for bc in sorted(by_bitcount.keys()):
        d = by_bitcount[bc]
        total = d['mod1'] + d['mod3']
        if total > 100:  # Only show significant buckets
            ratio = d['mod3'] / d['mod1'] if d['mod1'] > 0 else 0
            bias = "←1" if ratio < 1 else "→3" if ratio > 1 else "="
            print(f"  {bc:2d} bits: {total:6,} primes, ratio={ratio:.3f} {bias}")

    print()
    print("="*60)
    print("2. PATH BALANCE (1s vs 0s in binary index)")
    print("="*60)
    print()
    print(f"Left-heavy (more 0s):  {path_stats['left']:,} ({100*path_stats['left']/n_primes:.1f}%)")
    print(f"Right-heavy (more 1s): {path_stats['right']:,} ({100*path_stats['right']/n_primes:.1f}%)")
    print(f"Balanced:              {path_stats['balanced']:,} ({100*path_stats['balanced']/n_primes:.1f}%)")
    print()

    # Expected from random: slightly right-heavy because density → 0.5 as n→∞
    # but the split should follow a specific distribution

    print("="*60)
    print("3. BIT DENSITY DISTRIBUTION")
    print("="*60)
    print()
    print("Density = (# of 1-bits) / (bit length of index)")
    print()
    for d in sorted(density_buckets.keys()):
        count = density_buckets[d]
        bar = "#" * min(50, count // max(1, n_primes // 500))
        print(f"  {d:.1f}: {count:6,} {bar}")

    print()
    print("="*60)
    print("4. DEPTH (BIT LENGTH) DISTRIBUTION")
    print("="*60)
    print()
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        # Expected: 2^(d-1) primes at depth d (roughly)
        expected = 2 ** (depth - 1) if depth > 0 else 1
        ratio = count / expected if expected > 0 else 0
        bar = "#" * min(40, count // max(1, n_primes // 400))
        print(f"  Depth {depth:2d}: {count:6,} (expected ~{expected:6,}, ratio={ratio:.2f}) {bar}")

    print()
    print("="*60)
    print("5. RH ERROR vs BINARY DENSITY CORRELATION")
    print("="*60)
    print()

    if len(rh_data) > 10:
        # Compute correlation
        densities = [d[1] for d in rh_data]
        errors = [d[2] for d in rh_data]

        n_samples = len(rh_data)
        mean_d = sum(densities) / n_samples
        mean_e = sum(errors) / n_samples

        cov = sum((d - mean_d) * (e - mean_e) for d, e in zip(densities, errors)) / n_samples
        var_d = sum((d - mean_d)**2 for d in densities) / n_samples
        var_e = sum((e - mean_e)**2 for e in errors) / n_samples

        std_d = math.sqrt(var_d) if var_d > 0 else 1
        std_e = math.sqrt(var_e) if var_e > 0 else 1

        corr = cov / (std_d * std_e)

        print(f"Samples: {n_samples:,}")
        print(f"Correlation(density, normalized_error): {corr:.6f}")
        print()

        # Show some sample points
        print("Sample points (n, density, norm_error):")
        samples = [100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        for s in samples:
            matches = [d for d in rh_data if d[0] == s]
            if matches:
                n, dens, err = matches[0]
                print(f"  n={n:>7,}: density={dens:.3f}, error={err:+.6f}")

    print()
    print("="*60)
    print("6. KEY OBSERVATIONS")
    print("="*60)
    print()

    bias_diff = abs(e_ratio - o_ratio)
    if bias_diff > 0.01:
        print(f"★ CHEBYSHEV BIAS SPLITS: {bias_diff:.4f} difference between")
        print(f"  even and odd bit-count indices. This is UNEXPECTED.")
        print(f"  Suggests: binary structure of n correlates with p_n mod 4.")
        print()

    if path_stats['right'] > path_stats['left'] * 1.5:
        print(f"★ RIGHT-HEAVY DOMINANCE: {path_stats['right']/path_stats['left']:.2f}x more")
        print(f"  primes have indices with more 1s than 0s.")
        print()

    if len(rh_data) > 10:
        if abs(corr) > 0.01:
            direction = "positive" if corr > 0 else "negative"
            print(f"★ WEAK {direction.upper()} CORRELATION ({corr:.4f}) between")
            print(f"  binary density and π(x)-Li(x) error.")
            print()

    print("="*60)
    print("INTERPRETATION FOR RH")
    print("="*60)
    print("""
The binary structure of prime indices encodes information about
the prime counting function π(x).

If the Chebyshev bias genuinely splits by binary parity, this
suggests a deep connection between:
  - Arithmetic properties of primes (mod 4 residue)
  - Combinatorial structure (binary representation of π)

The zeta zeros control the oscillation of π(x) - Li(x).
If binary structure correlates with these oscillations,
the zeros might have a "binary shadow" we can detect.

NEXT: Test at 10^6+ to see if patterns strengthen or wash out.
""")

    return {
        'chebyshev': cheb,
        'by_bitcount': dict(by_bitcount),
        'path_stats': path_stats,
        'density_buckets': dict(density_buckets),
        'rh_correlation': corr if len(rh_data) > 10 else None
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    analyze(n)
