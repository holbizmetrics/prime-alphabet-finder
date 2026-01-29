"""
Block Permutation Control Test
==============================

ChatGPT's recommended verification:
"Partition indices into blocks, randomly permute density(n) within each block;
recompute correlations. If oscillation survives → artifact. If collapses → real."

This tests whether the binary structure is doing real work or if correlations
are just artifacts of windowing/drift.

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
import random
from collections import defaultdict

def sieve_primes(n):
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

def correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return 0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    var_x = sum((x - mean_x)**2 for x in xs) / n
    var_y = sum((y - mean_y)**2 for y in ys) / n
    if var_x <= 0 or var_y <= 0:
        return 0
    return cov / (math.sqrt(var_x) * math.sqrt(var_y))

def run_test(n_primes, block_size=10000, num_trials=10):
    """
    Run block permutation test.

    1. Compute REAL correlation (density vs normalized error)
    2. Permute densities within blocks, recompute correlation
    3. Compare: if permuted correlations are similar to real → artifact
                if permuted correlations collapse to ~0 → real signal
    """

    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]:,}")
    print(f"Block size: {block_size:,}")
    print(f"Permutation trials: {num_trials}")
    print()

    # Compute densities and errors for all primes
    data = []  # (n, density, normalized_error)

    for i, p in enumerate(primes):
        n = i + 1
        if p <= 2:
            continue

        bits = bin(n).count('1')
        bit_len = n.bit_length()
        density = bits / bit_len

        li_p = li(p)
        error = n - li_p
        norm_error = error / (math.sqrt(p) * math.log(p))

        data.append((n, density, norm_error))

    densities = [d[1] for d in data]
    errors = [d[2] for d in data]

    # REAL correlation
    real_corr = correlation(densities, errors)
    print(f"REAL correlation (density, error): {real_corr:+.6f}")
    print()

    # Block permutation test
    print("="*50)
    print("BLOCK PERMUTATION TEST")
    print("="*50)
    print()
    print("Permuting densities within blocks, preserving error structure...")
    print()

    permuted_corrs = []

    for trial in range(num_trials):
        # Create permuted densities
        permuted_densities = densities.copy()

        # Permute within blocks
        num_blocks = (len(permuted_densities) + block_size - 1) // block_size
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, len(permuted_densities))
            block = permuted_densities[start:end]
            random.shuffle(block)
            permuted_densities[start:end] = block

        # Compute correlation with permuted densities
        perm_corr = correlation(permuted_densities, errors)
        permuted_corrs.append(perm_corr)
        print(f"  Trial {trial+1:2d}: permuted corr = {perm_corr:+.6f}")

    # Statistics
    mean_perm = sum(permuted_corrs) / len(permuted_corrs)
    std_perm = math.sqrt(sum((c - mean_perm)**2 for c in permuted_corrs) / len(permuted_corrs))

    print()
    print("-"*50)
    print(f"REAL correlation:      {real_corr:+.6f}")
    print(f"Permuted mean:         {mean_perm:+.6f}")
    print(f"Permuted std:          {std_perm:.6f}")
    print(f"Z-score of real:       {(real_corr - mean_perm) / std_perm if std_perm > 0 else 0:.2f}")
    print("-"*50)

    # Interpretation
    print()
    print("="*50)
    print("INTERPRETATION")
    print("="*50)

    collapse_ratio = abs(mean_perm / real_corr) if real_corr != 0 else float('inf')

    if abs(mean_perm) < abs(real_corr) * 0.3:
        print("""
RESULT: Permuted correlations COLLAPSED.

The block permutation destroyed the correlation, meaning:
→ The binary structure of indices IS doing real work
→ The correlation is NOT an artifact of windowing/drift
→ There is genuine alignment between bit patterns and error

This SUPPORTS the hypothesis that binary index structure
encodes information related to π(x) - Li(x) oscillations.
""")
        verdict = "SIGNAL_REAL"
    elif abs(mean_perm) > abs(real_corr) * 0.7:
        print("""
RESULT: Permuted correlations SURVIVED.

The block permutation did NOT destroy the correlation, meaning:
→ The effect may be an artifact of slow drift in errors
→ The specific bit pattern alignment may not be essential
→ Need further investigation with different controls

This casts DOUBT on the binary structure hypothesis.
""")
        verdict = "LIKELY_ARTIFACT"
    else:
        print(f"""
RESULT: Partial collapse (permuted/real ratio = {collapse_ratio:.2f})

The permutation partially reduced the correlation.
→ Some signal may be real, some may be drift artifact
→ Need finer block sizes or additional controls
""")
        verdict = "MIXED"

    # Also test at different checkpoints
    print()
    print("="*50)
    print("CHECKPOINT ANALYSIS")
    print("="*50)
    print()

    checkpoints = [50000, 100000, 200000, 500000]
    checkpoints = [c for c in checkpoints if c <= n_primes]

    print(f"{'N':>10} | {'Real':>10} | {'Perm Mean':>10} | {'Collapse?':>10}")
    print("-"*50)

    for cp in checkpoints:
        cp_densities = densities[:cp]
        cp_errors = errors[:cp]

        real_cp = correlation(cp_densities, cp_errors)

        # Quick permutation test
        perm_corrs_cp = []
        for _ in range(5):
            perm_d = cp_densities.copy()
            bs = min(block_size, cp // 10)
            num_b = (len(perm_d) + bs - 1) // bs
            for b in range(num_b):
                s, e = b * bs, min((b+1) * bs, len(perm_d))
                block = perm_d[s:e]
                random.shuffle(block)
                perm_d[s:e] = block
            perm_corrs_cp.append(correlation(perm_d, cp_errors))

        mean_perm_cp = sum(perm_corrs_cp) / len(perm_corrs_cp)
        collapse = "YES" if abs(mean_perm_cp) < abs(real_cp) * 0.3 else "NO"

        print(f"{cp:>10,} | {real_cp:>+10.4f} | {mean_perm_cp:>+10.4f} | {collapse:>10}")

    return {
        'real_corr': real_corr,
        'permuted_mean': mean_perm,
        'permuted_std': std_perm,
        'verdict': verdict
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    block = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    results = run_test(n, block, trials)
