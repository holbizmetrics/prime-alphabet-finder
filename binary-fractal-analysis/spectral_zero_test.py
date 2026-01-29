"""
Spectral Test: Does binary weighting produce peaks at zeta zero frequencies?

ChatGPT's litmus test #2:
"The oscillation frequency lines up cleanly with known zeta zeros"

We compute: S(γ) = Σ w(n) * exp(i * γ * log(p_n))

where w(n) = popcount(n) or density(n)

If peaks appear at known γ values, that's the smoking gun.

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math

# First 50 zeta zeros (imaginary parts)
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756509, 138.116042, 139.736209, 141.123707, 143.111846,
]

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

def spectral_power(primes, gamma, weight_func):
    """
    Compute |S(γ)|² where S(γ) = Σ w(n) * exp(i * γ * log(p_n))
    """
    real_sum = 0
    imag_sum = 0

    for i, p in enumerate(primes):
        n = i + 1
        if p < 3:
            continue

        w = weight_func(n)
        log_p = math.log(p)

        real_sum += w * math.cos(gamma * log_p)
        imag_sum += w * math.sin(gamma * log_p)

    return real_sum**2 + imag_sum**2

def run_spectral_test(n_primes):
    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]:,}")
    print()

    # Weight functions to test
    def popcount(n):
        return bin(n).count('1')

    def density(n):
        bits = bin(n).count('1')
        return bits / n.bit_length() if n > 0 else 0

    def thue_morse(n):
        return 1 if bin(n).count('1') % 2 == 0 else -1

    def centered_popcount(n):
        # Center around expected value
        bits = bin(n).count('1')
        expected = n.bit_length() / 2
        return bits - expected

    weights = [
        ("popcount", popcount),
        ("density", density),
        ("thue_morse", thue_morse),
        ("centered_popcount", centered_popcount),
    ]

    # Also test random frequencies (control)
    import random
    random.seed(42)
    random_freqs = [random.uniform(10, 150) for _ in range(10)]

    print("="*70)
    print("SPECTRAL POWER AT ZETA ZEROS vs RANDOM FREQUENCIES")
    print("="*70)
    print()
    print("If binary weighting 'sees' zeta zeros, power at γ_k should be")
    print("systematically higher than at random frequencies.")
    print()

    for weight_name, weight_func in weights:
        print(f"\n{'='*70}")
        print(f"Weight: {weight_name}")
        print(f"{'='*70}")

        # Power at zeta zeros
        zero_powers = []
        for gamma in ZETA_ZEROS[:20]:
            power = spectral_power(primes, gamma, weight_func)
            zero_powers.append((gamma, power))

        # Power at random frequencies
        random_powers = []
        for gamma in random_freqs:
            power = spectral_power(primes, gamma, weight_func)
            random_powers.append((gamma, power))

        # Normalize by N for comparison
        N = len(primes)

        avg_zero_power = sum(p for _, p in zero_powers) / len(zero_powers)
        avg_random_power = sum(p for _, p in random_powers) / len(random_powers)

        print(f"\nAverage power at zeta zeros:     {avg_zero_power/N:.4f}")
        print(f"Average power at random freqs:   {avg_random_power/N:.4f}")
        print(f"Ratio (zeros/random):            {avg_zero_power/avg_random_power:.4f}")

        # Top zeros by power
        zero_powers.sort(key=lambda x: -x[1])
        print(f"\nTop 5 zeta zeros by power:")
        for gamma, power in zero_powers[:5]:
            # Find which zero this is
            idx = ZETA_ZEROS.index(gamma) + 1
            print(f"  γ_{idx} = {gamma:.3f}: power = {power/N:.4f}")

        # Statistical test: how many zeros beat median random?
        median_random = sorted([p for _, p in random_powers])[len(random_powers)//2]
        zeros_above_median = sum(1 for _, p in zero_powers if p > median_random)
        print(f"\nZeros with power > median random: {zeros_above_median}/20")

        # Is this significant?
        # Under null, expect 10/20. Binomial test.
        from math import comb
        # P(X >= k) under binomial(20, 0.5)
        if zeros_above_median > 10:
            p_value = sum(comb(20, k) * 0.5**20 for k in range(zeros_above_median, 21))
            print(f"One-sided p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  → SIGNIFICANT: zeros have higher power than random!")

    # Detailed frequency scan
    print("\n" + "="*70)
    print("FREQUENCY SCAN: Looking for peaks")
    print("="*70)

    # Scan frequencies from 10 to 50 in small steps
    # Look for peaks that align with zeros

    best_weight = centered_popcount  # Often most sensitive

    print("\nScanning γ = 10 to 50, step 0.5...")
    print("Marking known zeros with *")
    print()

    scan_results = []
    for gamma in [10 + 0.5*i for i in range(80)]:
        power = spectral_power(primes, gamma, best_weight)
        scan_results.append((gamma, power / N))

    # Find local maxima
    peaks = []
    for i in range(1, len(scan_results) - 1):
        if scan_results[i][1] > scan_results[i-1][1] and scan_results[i][1] > scan_results[i+1][1]:
            peaks.append(scan_results[i])

    print("Local maxima in power spectrum:")
    print(f"{'γ':>8} | {'Power':>10} | {'Near zero?':>15}")
    print("-"*40)

    for gamma, power in sorted(peaks, key=lambda x: -x[1])[:15]:
        # Check if near a known zero
        nearest_zero = min(ZETA_ZEROS, key=lambda z: abs(z - gamma))
        dist = abs(gamma - nearest_zero)
        near = f"γ={nearest_zero:.2f} (d={dist:.2f})" if dist < 1.5 else ""
        marker = " ***" if dist < 1.0 else ""
        print(f"{gamma:>8.2f} | {power:>10.4f} | {near:>15}{marker}")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    # Check if peaks align with zeros better than chance
    peak_gammas = [g for g, p in peaks]
    aligned = 0
    for pg in peak_gammas:
        if any(abs(pg - z) < 1.0 for z in ZETA_ZEROS):
            aligned += 1

    print(f"\nPeaks within 1.0 of a zeta zero: {aligned}/{len(peaks)}")
    expected = len(peaks) * len(ZETA_ZEROS) * 2.0 / 40  # rough expected under uniform
    print(f"Expected by chance: ~{expected:.1f}")

    if aligned > expected * 1.5:
        print("\n→ PEAKS ALIGN WITH ZEROS better than chance!")
        print("  This would pass ChatGPT's litmus test #2")
    else:
        print("\n→ No clear alignment between peaks and zeros")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    run_spectral_test(n)
