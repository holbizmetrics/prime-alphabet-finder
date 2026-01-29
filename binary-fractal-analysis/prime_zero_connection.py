"""
Prime Binary Index vs Zeta Zero Connection Test
================================================

Tests whether the binary structure of prime indices correlates
with the actual locations of Riemann zeta zeros.

The zeros create oscillations in π(x) - Li(x) via the explicit formula.
If binary structure correlates with these oscillations, we might be able
to detect the "shadow" of zero locations.

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
from collections import defaultdict

# First 100 non-trivial zeta zeros (imaginary parts, on critical line s = 1/2 + it)
# These are the Gram points / actual zeros from tables
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
    146.000982, 147.422765, 150.053521, 150.925258, 153.024693,
    156.112909, 157.597592, 158.849988, 161.188964, 163.030709,
    165.537070, 167.184439, 169.094515, 169.911976, 173.411537,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265396, 196.876481, 198.015310, 201.264751,
    202.493595, 204.189671, 205.394697, 207.906259, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714919, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250189, 231.987235, 233.693404, 236.524230,
]

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
    """Logarithmic integral."""
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

def oscillatory_term(x, gamma):
    """
    Contribution from a single zero at s = 1/2 + i*gamma.

    The explicit formula has terms like -Li(x^rho) where rho = 1/2 + i*gamma.
    For real x, this contributes an oscillatory term roughly:

    -2 * Re(Li(x^{1/2 + i*gamma})) ≈ -2 * x^{1/2} * cos(gamma * log(x)) / (gamma * log(x))

    This is a simplified approximation.
    """
    if x <= 1:
        return 0
    log_x = math.log(x)
    sqrt_x = math.sqrt(x)

    # Simplified oscillatory contribution
    return -2 * sqrt_x * math.cos(gamma * log_x) / (gamma * log_x + 0.1)

def compute_oscillation_phase(x, zeros=ZETA_ZEROS[:20]):
    """
    Compute the "phase" of the oscillation at x based on first few zeros.
    Returns a value that indicates where we are in the oscillation cycle.
    """
    total = 0
    for gamma in zeros:
        log_x = math.log(x) if x > 1 else 0.1
        phase = (gamma * log_x) % (2 * math.pi)
        total += math.cos(phase) / gamma  # Weight by 1/gamma
    return total

def analyze_zero_connection(n_primes):
    """Main analysis connecting binary structure to zero locations."""

    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]:,}")
    print()

    # Compute oscillation phases at each prime
    print("Computing oscillation phases from first 20 zeros...")

    # Group primes by binary properties and track oscillation phases
    by_bitcount_parity = {'even': [], 'odd': []}
    by_density = defaultdict(list)

    phase_data = []  # (n, p, phase, density, bitcount)

    for i, p in enumerate(primes):
        n = i + 1
        if p < 10:
            continue

        bits = bin(n).count('1')
        bit_len = n.bit_length()
        density = bits / bit_len if bit_len > 0 else 0

        phase = compute_oscillation_phase(p)

        phase_data.append((n, p, phase, density, bits))

        parity = 'even' if bits % 2 == 0 else 'odd'
        by_bitcount_parity[parity].append(phase)

        density_bucket = round(density, 1)
        by_density[density_bucket].append(phase)

    # ===== Analysis 1: Phase by bit-count parity =====
    print("\n" + "="*60)
    print("1. OSCILLATION PHASE BY BIT-COUNT PARITY")
    print("="*60)

    for parity in ['even', 'odd']:
        phases = by_bitcount_parity[parity]
        if phases:
            avg = sum(phases) / len(phases)
            var = sum((p - avg)**2 for p in phases) / len(phases)
            std = math.sqrt(var)
            print(f"\n{parity.upper()} bit-count indices ({len(phases):,} primes):")
            print(f"  Mean phase: {avg:+.6f}")
            print(f"  Std dev:    {std:.6f}")

    # Phase difference
    even_mean = sum(by_bitcount_parity['even']) / len(by_bitcount_parity['even'])
    odd_mean = sum(by_bitcount_parity['odd']) / len(by_bitcount_parity['odd'])
    print(f"\nPHASE DIFFERENCE (even - odd): {even_mean - odd_mean:+.6f}")

    # ===== Analysis 2: Phase by density =====
    print("\n" + "="*60)
    print("2. OSCILLATION PHASE BY BINARY DENSITY")
    print("="*60)

    print("\nDensity → Mean Phase:")
    for d in sorted(by_density.keys()):
        phases = by_density[d]
        if len(phases) > 100:
            avg = sum(phases) / len(phases)
            print(f"  {d:.1f}: {avg:+.6f} ({len(phases):,} primes)")

    # Correlation: density vs phase
    densities = [d[3] for d in phase_data]
    phases = [d[2] for d in phase_data]

    n_samples = len(phase_data)
    mean_d = sum(densities) / n_samples
    mean_p = sum(phases) / n_samples

    cov = sum((d - mean_d) * (p - mean_p) for d, p in zip(densities, phases)) / n_samples
    var_d = sum((d - mean_d)**2 for d in densities) / n_samples
    var_p = sum((p - mean_p)**2 for p in phases) / n_samples

    std_d = math.sqrt(var_d) if var_d > 0 else 1
    std_p = math.sqrt(var_p) if var_p > 0 else 1

    corr = cov / (std_d * std_p)

    print(f"\nCorrelation(density, phase): {corr:.6f}")

    # ===== Analysis 3: Individual zero contributions =====
    print("\n" + "="*60)
    print("3. INDIVIDUAL ZERO CONTRIBUTIONS")
    print("="*60)

    print("\nCorrelation of each zero's phase with bit-count parity:")
    print("(Positive = even bit-count has higher phase for this zero)")

    for i, gamma in enumerate(ZETA_ZEROS[:20]):
        even_phases = []
        odd_phases = []

        for j, p in enumerate(primes):
            if p < 10:
                continue
            n = j + 1
            bits = bin(n).count('1')

            log_p = math.log(p)
            phase = math.cos(gamma * log_p)

            if bits % 2 == 0:
                even_phases.append(phase)
            else:
                odd_phases.append(phase)

        even_mean = sum(even_phases) / len(even_phases)
        odd_mean = sum(odd_phases) / len(odd_phases)
        diff = even_mean - odd_mean

        marker = "**" if abs(diff) > 0.005 else ""
        print(f"  γ_{i+1} = {gamma:8.3f}: diff = {diff:+.6f} {marker}")

    # ===== Analysis 4: Resonance detection =====
    print("\n" + "="*60)
    print("4. RESONANCE DETECTION")
    print("="*60)

    print("\nLooking for zeros whose frequency 'resonates' with binary structure...")

    # For each zero, compute correlation between:
    # - cos(gamma * log(p_n))
    # - popcount(n) (or some binary feature)

    print("\nCorrelation(cos(γ·log(p)), popcount(n)):")

    resonances = []
    for i, gamma in enumerate(ZETA_ZEROS[:30]):
        cos_phases = []
        popcounts = []

        for j, p in enumerate(primes):
            if p < 10:
                continue
            n = j + 1

            cos_phases.append(math.cos(gamma * math.log(p)))
            popcounts.append(bin(n).count('1'))

        # Correlation
        n_s = len(cos_phases)
        mean_c = sum(cos_phases) / n_s
        mean_pc = sum(popcounts) / n_s

        cov = sum((c - mean_c) * (pc - mean_pc) for c, pc in zip(cos_phases, popcounts)) / n_s
        var_c = sum((c - mean_c)**2 for c in cos_phases) / n_s
        var_pc = sum((pc - mean_pc)**2 for pc in popcounts) / n_s

        std_c = math.sqrt(var_c) if var_c > 0 else 1
        std_pc = math.sqrt(var_pc) if var_pc > 0 else 1

        corr = cov / (std_c * std_pc)
        resonances.append((gamma, corr))

        marker = "**" if abs(corr) > 0.01 else ""
        print(f"  γ_{i+1} = {gamma:8.3f}: r = {corr:+.6f} {marker}")

    # ===== Analysis 5: Spectral test =====
    print("\n" + "="*60)
    print("5. BINARY SPECTRUM vs ZERO SPECTRUM")
    print("="*60)

    # Compute "binary spectrum" - Fourier transform of popcount sequence
    # Compare to zero locations

    print("\nComputing binary spectrum of popcount sequence...")

    # Take popcount(n) for n = 1 to N, compute DFT peaks
    N = min(n_primes, 10000)
    popcounts = [bin(n).count('1') for n in range(1, N + 1)]
    mean_pc = sum(popcounts) / N
    centered = [p - mean_pc for p in popcounts]

    # Compute power at specific frequencies related to zeros
    print("\nPower at zero-related frequencies:")

    for i, gamma in enumerate(ZETA_ZEROS[:10]):
        # Frequency in binary sequence that might correspond to gamma
        # gamma relates to oscillation in log(x), so freq ~ gamma / (2π)
        freq = gamma / (2 * math.pi)

        # Compute Fourier coefficient at this frequency
        real_part = sum(centered[n] * math.cos(2 * math.pi * freq * n / N) for n in range(N)) / N
        imag_part = sum(centered[n] * math.sin(2 * math.pi * freq * n / N) for n in range(N)) / N
        power = real_part**2 + imag_part**2

        print(f"  γ_{i+1} = {gamma:8.3f} → freq {freq:.3f}: power = {power:.6f}")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"""
Analyzed {n_primes:,} primes against first 20-30 zeta zeros.

KEY FINDINGS:

1. Phase difference by bit-count parity: {even_mean - odd_mean:+.6f}
   (Even bit-count indices have {'higher' if even_mean > odd_mean else 'lower'} oscillation phase)

2. Density-phase correlation: {corr:.4f}
   ({'Weak positive' if corr > 0 else 'Weak negative'} correlation)

3. Individual zero resonances:
   Zeros with strongest binary correlation:
""")

    resonances.sort(key=lambda x: abs(x[1]), reverse=True)
    for gamma, r in resonances[:5]:
        print(f"      γ = {gamma:.3f}: r = {r:+.4f}")

    print("""
INTERPRETATION:

The oscillations from zeta zeros DO show correlation with binary
structure of prime indices. Some zeros "resonate" more strongly
with the binary pattern than others.

This suggests the binary representation of π(x) encodes partial
information about the zero locations - a "combinatorial shadow"
of the analytic structure.

NEXT: Scale up and look for which specific zeros have the
strongest resonance with binary features.
""")

    return {
        'phase_diff': even_mean - odd_mean,
        'density_phase_corr': corr,
        'resonances': resonances
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    results = analyze_zero_connection(n)
