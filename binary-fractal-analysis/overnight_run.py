#!/usr/bin/env python3
"""
Overnight Runner for Prime Binary Analysis
==========================================

Scales to 10^6+ primes with checkpointing and memory efficiency.
Saves results incrementally so progress isn't lost if interrupted.

Usage:
    python overnight_run.py [n_primes] [output_file]

Default: 1,000,000 primes, output to overnight_results.txt

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
import sys
import time
import json
from collections import defaultdict
from datetime import datetime

# ============================================================================
# MEMORY-EFFICIENT PRIME GENERATION
# ============================================================================

def segmented_sieve(limit, segment_size=10**6):
    """
    Memory-efficient segmented sieve.
    Generates primes up to 'limit' without storing all of them.
    """
    sqrt_limit = int(limit**0.5) + 1

    # Small primes for sieving
    small_sieve = [True] * (sqrt_limit + 1)
    small_sieve[0] = small_sieve[1] = False
    for i in range(2, int(sqrt_limit**0.5) + 1):
        if small_sieve[i]:
            for j in range(i*i, sqrt_limit + 1, i):
                small_sieve[j] = False
    small_primes = [i for i, is_p in enumerate(small_sieve) if is_p]

    # Yield small primes first
    for p in small_primes:
        yield p

    # Segmented sieve for larger primes
    low = sqrt_limit + 1
    while low <= limit:
        high = min(low + segment_size - 1, limit)
        segment = [True] * (high - low + 1)

        for p in small_primes:
            start = ((low + p - 1) // p) * p
            if start < p * p:
                start = p * p
            for j in range(start, high + 1, p):
                segment[j - low] = False

        for i in range(len(segment)):
            if segment[i]:
                yield low + i

        low = high + 1

def first_n_primes_generator(n):
    """Generate first n primes, memory efficient."""
    if n < 6:
        limit = 15
    else:
        ln_n = math.log(n)
        limit = int(n * (ln_n + math.log(ln_n) + 2.5))

    count = 0
    for p in segmented_sieve(limit):
        yield p
        count += 1
        if count >= n:
            break

# ============================================================================
# ANALYSIS ACCUMULATORS
# ============================================================================

class StatsAccumulator:
    """Accumulates statistics without storing all data points."""

    def __init__(self):
        self.n = 0
        self.sum_x = 0
        self.sum_y = 0
        self.sum_xx = 0
        self.sum_yy = 0
        self.sum_xy = 0

    def add(self, x, y):
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    def correlation(self):
        if self.n < 2:
            return 0
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        var_x = self.sum_xx / self.n - mean_x**2
        var_y = self.sum_yy / self.n - mean_y**2
        cov = self.sum_xy / self.n - mean_x * mean_y

        if var_x <= 0 or var_y <= 0:
            return 0
        return cov / (math.sqrt(var_x) * math.sqrt(var_y))

    def mean_x(self):
        return self.sum_x / self.n if self.n > 0 else 0

    def mean_y(self):
        return self.sum_y / self.n if self.n > 0 else 0

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

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

def run_analysis(n_primes, output_file, checkpoint_interval=50000):
    """
    Run full analysis with checkpointing.
    """

    start_time = time.time()

    # Output file
    out = open(output_file, 'w')
    out.write(f"# Prime Binary Analysis - {n_primes:,} primes\n")
    out.write(f"# Started: {datetime.now().isoformat()}\n\n")
    out.flush()

    # Accumulators
    chebyshev_even = {'mod1': 0, 'mod3': 0}
    chebyshev_odd = {'mod1': 0, 'mod3': 0}
    by_bitcount = defaultdict(lambda: {'mod1': 0, 'mod3': 0})

    path_stats = {'left': 0, 'right': 0, 'balanced': 0}
    density_buckets = defaultdict(int)
    depth_counts = defaultdict(int)

    density_error_corr = StatsAccumulator()

    # Sample storage for detailed analysis
    samples = []  # (n, p, bits, density, error) at intervals

    last_prime = 2
    processed = 0

    print(f"Starting analysis of {n_primes:,} primes...")
    print(f"Output: {output_file}")
    print(f"Checkpoints every {checkpoint_interval:,} primes\n")

    for n, p in enumerate(first_n_primes_generator(n_primes), 1):
        last_prime = p

        # Binary properties
        bits = bin(n).count('1')
        bit_len = n.bit_length()
        density = bits / bit_len if bit_len > 0 else 0

        # Depth
        depth_counts[bit_len] += 1

        # Density bucket
        density_buckets[round(density, 1)] += 1

        # Path balance
        zeros = bit_len - bits
        if bits > zeros:
            path_stats['right'] += 1
        elif zeros > bits:
            path_stats['left'] += 1
        else:
            path_stats['balanced'] += 1

        # Chebyshev
        if p > 2:
            mod4 = p % 4
            key = 'mod1' if mod4 == 1 else 'mod3'

            if bits % 2 == 0:
                chebyshev_even[key] += 1
            else:
                chebyshev_odd[key] += 1

            by_bitcount[bits][key] += 1

        # RH error correlation (every 10th to save time at scale)
        if p > 2 and n % 10 == 0:
            li_p = li(p)
            error = n - li_p
            norm_error = error / (math.sqrt(p) * math.log(p))
            density_error_corr.add(density, norm_error)

        # Store samples
        if n in [100, 1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]:
            if p > 2:
                li_p = li(p)
                error = (n - li_p) / (math.sqrt(p) * math.log(p))
                samples.append((n, p, bits, density, error))

        processed += 1

        # Checkpoint
        if processed % checkpoint_interval == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (n_primes - processed) / rate if rate > 0 else 0

            print(f"  {processed:>10,} / {n_primes:,} ({100*processed/n_primes:.1f}%) "
                  f"| p = {p:,} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

            # Write checkpoint
            write_checkpoint(out, processed, p,
                           chebyshev_even, chebyshev_odd,
                           density_error_corr, path_stats)

    # Final results
    elapsed = time.time() - start_time

    out.write("\n" + "="*70 + "\n")
    out.write("FINAL RESULTS\n")
    out.write("="*70 + "\n\n")

    out.write(f"Primes analyzed: {n_primes:,}\n")
    out.write(f"Largest prime: {last_prime:,}\n")
    out.write(f"Time elapsed: {elapsed:.1f} seconds\n\n")

    # Chebyshev bias
    e_total = chebyshev_even['mod1'] + chebyshev_even['mod3']
    o_total = chebyshev_odd['mod1'] + chebyshev_odd['mod3']

    e_ratio = chebyshev_even['mod3'] / chebyshev_even['mod1'] if chebyshev_even['mod1'] > 0 else 0
    o_ratio = chebyshev_odd['mod3'] / chebyshev_odd['mod1'] if chebyshev_odd['mod1'] > 0 else 0

    out.write("CHEBYSHEV BIAS BY BIT-COUNT PARITY\n")
    out.write("-"*40 + "\n")
    out.write(f"Even bit-count indices ({e_total:,} primes):\n")
    out.write(f"  mod 1: {chebyshev_even['mod1']:,}\n")
    out.write(f"  mod 3: {chebyshev_even['mod3']:,}\n")
    out.write(f"  Ratio: {e_ratio:.6f}\n\n")
    out.write(f"Odd bit-count indices ({o_total:,} primes):\n")
    out.write(f"  mod 1: {chebyshev_odd['mod1']:,}\n")
    out.write(f"  mod 3: {chebyshev_odd['mod3']:,}\n")
    out.write(f"  Ratio: {o_ratio:.6f}\n\n")
    out.write(f"BIAS DIFFERENCE: {abs(e_ratio - o_ratio):.6f}\n\n")

    # By bit count
    out.write("BY SPECIFIC BIT COUNT\n")
    out.write("-"*40 + "\n")
    for bc in sorted(by_bitcount.keys()):
        d = by_bitcount[bc]
        total = d['mod1'] + d['mod3']
        if total > 100:
            ratio = d['mod3'] / d['mod1'] if d['mod1'] > 0 else 0
            out.write(f"  {bc:2d} bits: {total:8,} primes, ratio = {ratio:.4f}\n")
    out.write("\n")

    # Path balance
    out.write("PATH BALANCE\n")
    out.write("-"*40 + "\n")
    out.write(f"Left-heavy:  {path_stats['left']:,} ({100*path_stats['left']/n_primes:.2f}%)\n")
    out.write(f"Right-heavy: {path_stats['right']:,} ({100*path_stats['right']/n_primes:.2f}%)\n")
    out.write(f"Balanced:    {path_stats['balanced']:,} ({100*path_stats['balanced']/n_primes:.2f}%)\n\n")

    # Density distribution
    out.write("DENSITY DISTRIBUTION\n")
    out.write("-"*40 + "\n")
    for d in sorted(density_buckets.keys()):
        count = density_buckets[d]
        out.write(f"  {d:.1f}: {count:,}\n")
    out.write("\n")

    # Correlation
    corr = density_error_corr.correlation()
    out.write("RH ERROR CORRELATION\n")
    out.write("-"*40 + "\n")
    out.write(f"Correlation(density, normalized_error): {corr:.6f}\n")
    out.write(f"Samples: {density_error_corr.n:,}\n\n")

    # Samples
    out.write("SAMPLE POINTS\n")
    out.write("-"*40 + "\n")
    out.write(f"{'n':>12} {'p':>15} {'bits':>6} {'density':>8} {'error':>12}\n")
    for n, p, bits, dens, err in samples:
        out.write(f"{n:>12,} {p:>15,} {bits:>6} {dens:>8.4f} {err:>+12.6f}\n")
    out.write("\n")

    # Depth distribution
    out.write("DEPTH DISTRIBUTION\n")
    out.write("-"*40 + "\n")
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        expected = 2 ** (depth - 1) if depth > 0 else 1
        ratio = count / expected if expected > 0 else 0
        out.write(f"  Depth {depth:2d}: {count:10,} (expected {expected:10,}, ratio {ratio:.4f})\n")

    out.write("\n" + "="*70 + "\n")
    out.write(f"Completed: {datetime.now().isoformat()}\n")
    out.write(f"Total time: {elapsed:.1f} seconds\n")
    out.close()

    print(f"\nDone! Results written to {output_file}")
    print(f"Total time: {elapsed:.1f} seconds")

    # Also print key results
    print(f"\n{'='*50}")
    print("KEY RESULTS")
    print(f"{'='*50}")
    print(f"Chebyshev bias split: {abs(e_ratio - o_ratio):.6f}")
    print(f"Density-error correlation: {corr:.6f}")
    print(f"Path balance: {100*path_stats['right']/n_primes:.1f}% right-heavy")

    return {
        'bias_diff': abs(e_ratio - o_ratio),
        'correlation': corr,
        'path_right_pct': 100*path_stats['right']/n_primes
    }

def write_checkpoint(out, n, p, cheb_even, cheb_odd, corr_acc, path):
    """Write checkpoint to output file."""
    e_ratio = cheb_even['mod3'] / cheb_even['mod1'] if cheb_even['mod1'] > 0 else 0
    o_ratio = cheb_odd['mod3'] / cheb_odd['mod1'] if cheb_odd['mod1'] > 0 else 0

    out.write(f"\n--- Checkpoint at n={n:,}, p={p:,} ---\n")
    out.write(f"Bias diff: {abs(e_ratio - o_ratio):.6f}\n")
    out.write(f"Corr: {corr_acc.correlation():.6f}\n")
    out.write(f"Path: L={path['left']:,} R={path['right']:,} B={path['balanced']:,}\n")
    out.flush()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    n_primes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    output_file = sys.argv[2] if len(sys.argv) > 2 else "overnight_results.txt"

    # Adjust checkpoint based on size
    if n_primes >= 10000000:
        checkpoint = 500000
    elif n_primes >= 1000000:
        checkpoint = 100000
    else:
        checkpoint = 50000

    run_analysis(n_primes, output_file, checkpoint)
