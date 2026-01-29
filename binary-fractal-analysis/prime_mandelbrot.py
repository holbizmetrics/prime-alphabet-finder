"""
Prime-Encoded Mandelbrot Escape Times
======================================

NOVEL EXPLORATION: Map prime gaps to complex parameters c, study escape behavior.

Hypothesis: If primes have hidden structure, the Mandelbrot dynamics might reveal it
through patterns in escape times that differ from random gap sequences.

Encodings to try:
1. c = g_n + i*g_{n+1}  (consecutive gaps as complex number)
2. c = (g_n/log(p_n)) * exp(i * 2π * n/φ)  (golden spiral with normalized gaps)
3. c = on Mandelbrot boundary, indexed by prime

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
import random
from collections import defaultdict

# ============================================================================
# PRIME GENERATION
# ============================================================================

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

# ============================================================================
# MANDELBROT ITERATION
# ============================================================================

def mandelbrot_escape(c_real, c_imag, max_iter=1000):
    """
    Compute escape time for c = c_real + i*c_imag.
    Returns iterations until |z| > 2, or max_iter if bounded.
    """
    zr, zi = 0.0, 0.0
    for n in range(max_iter):
        zr2, zi2 = zr*zr, zi*zi
        if zr2 + zi2 > 4.0:
            return n
        zi = 2*zr*zi + c_imag
        zr = zr2 - zi2 + c_real
    return max_iter

def smooth_escape(c_real, c_imag, max_iter=1000):
    """
    Smooth escape time using continuous potential.
    """
    zr, zi = 0.0, 0.0
    for n in range(max_iter):
        zr2, zi2 = zr*zr, zi*zi
        mag2 = zr2 + zi2
        if mag2 > 4.0:
            # Smooth coloring
            log_zn = math.log(mag2) / 2
            nu = math.log(log_zn / math.log(2)) / math.log(2)
            return n + 1 - nu
        zi = 2*zr*zi + c_imag
        zr = zr2 - zi2 + c_real
    return max_iter

# ============================================================================
# MANDELBULB (3D MANDELBROT)
# ============================================================================

def mandelbulb_escape(cx, cy, cz, power=8, max_iter=100):
    """
    Compute escape time for 3D Mandelbulb.

    The Mandelbulb uses spherical coordinates:
    - (x,y,z) → (r, θ, φ) where r=|v|, θ=arctan(y/x), φ=arccos(z/r)
    - v^n: r^n, θ*n, φ*n
    - Convert back and add c

    Power 8 gives the classic Mandelbulb shape.
    """
    x, y, z = 0.0, 0.0, 0.0

    for n in range(max_iter):
        r = math.sqrt(x*x + y*y + z*z)

        if r > 2.0:
            return n

        if r < 1e-10:
            # At origin, stay at origin + c
            x, y, z = cx, cy, cz
            continue

        # Spherical coordinates
        theta = math.atan2(y, x)
        phi = math.acos(z / r)

        # Raise to power
        r_n = r ** power
        theta_n = theta * power
        phi_n = phi * power

        # Back to Cartesian + add c
        x = r_n * math.sin(phi_n) * math.cos(theta_n) + cx
        y = r_n * math.sin(phi_n) * math.sin(theta_n) + cy
        z = r_n * math.cos(phi_n) + cz

    return max_iter

def triplet_to_3d(gaps, scale=0.1):
    """
    Encoding for 3D: Three consecutive gaps → (x, y, z)
    c = scale * (g_n, g_{n+1}, g_{n+2})
    """
    points = []
    for i in range(len(gaps) - 2):
        cx = gaps[i] * scale
        cy = gaps[i+1] * scale
        cz = gaps[i+2] * scale
        points.append((cx, cy, cz, i))
    return points

def analyze_mandelbulb(gaps, scale=0.1, power=8, max_iter=100):
    """Analyze escape times in 3D Mandelbulb."""
    points = triplet_to_3d(gaps, scale)

    escapes = []
    for cx, cy, cz, idx in points:
        esc = mandelbulb_escape(cx, cy, cz, power, max_iter)
        escapes.append((idx, cx, cy, cz, esc))

    return escapes

def compare_3d_to_random(gaps, scale=0.1, power=8, max_iter=100, num_trials=10):
    """Compare real prime gap triplets to shuffled in 3D Mandelbulb."""

    # Real gaps
    real_escapes = []
    for i in range(len(gaps) - 2):
        cx, cy, cz = gaps[i]*scale, gaps[i+1]*scale, gaps[i+2]*scale
        esc = mandelbulb_escape(cx, cy, cz, power, max_iter)
        real_escapes.append(esc)

    real_mean = sum(real_escapes) / len(real_escapes)
    real_in_set = sum(1 for e in real_escapes if e == max_iter)
    real_fast = sum(1 for e in real_escapes if e < 5)

    # Shuffled trials
    shuf_means = []
    shuf_in_sets = []
    shuf_fasts = []

    for _ in range(num_trials):
        shuf = gaps.copy()
        random.shuffle(shuf)

        shuf_esc = []
        for i in range(len(shuf) - 2):
            cx, cy, cz = shuf[i]*scale, shuf[i+1]*scale, shuf[i+2]*scale
            esc = mandelbulb_escape(cx, cy, cz, power, max_iter)
            shuf_esc.append(esc)

        shuf_means.append(sum(shuf_esc) / len(shuf_esc))
        shuf_in_sets.append(sum(1 for e in shuf_esc if e == max_iter))
        shuf_fasts.append(sum(1 for e in shuf_esc if e < 5))

    return {
        'real_mean': real_mean,
        'real_in_set': real_in_set,
        'real_fast': real_fast,
        'shuffled_mean': sum(shuf_means) / len(shuf_means),
        'shuffled_in_set': sum(shuf_in_sets) / len(shuf_in_sets),
        'shuffled_fast': sum(shuf_fasts) / len(shuf_fasts),
        'n_points': len(real_escapes)
    }

# ============================================================================
# PRIME GAP ENCODINGS
# ============================================================================

def encoding_consecutive_gaps(gaps, scale=0.1):
    """
    Encoding 1: c = scale * (g_n + i*g_{n+1})
    Consecutive gaps become complex number.
    """
    points = []
    for i in range(len(gaps) - 1):
        c_real = gaps[i] * scale
        c_imag = gaps[i+1] * scale
        points.append((c_real, c_imag, i))
    return points

def encoding_golden_spiral(gaps, primes, scale=0.5):
    """
    Encoding 2: Normalized gap on golden spiral.
    c = (g_n / log(p_n)) * exp(i * 2π * n * φ)
    """
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
    points = []
    for i, (g, p) in enumerate(zip(gaps, primes[1:])):
        if p < 3:
            continue
        # Normalize gap by expected size
        normalized = g / math.log(p)
        # Spiral angle
        theta = 2 * math.pi * i / PHI
        # Radius based on normalized gap
        r = normalized * scale
        c_real = r * math.cos(theta)
        c_imag = r * math.sin(theta)
        points.append((c_real, c_imag, i))
    return points

def encoding_cardioid_param(gaps, primes):
    """
    Encoding 3: Map to main cardioid of Mandelbrot.
    The main cardioid is parametrized by: c = (e^(iθ)/2) - (e^(2iθ)/4)
    Use θ = 2π * (g_n / max_gap) to map gaps to boundary.
    """
    max_gap = max(gaps)
    points = []
    for i, g in enumerate(gaps):
        theta = 2 * math.pi * g / max_gap
        # Cardioid parametrization
        c_real = 0.5 * math.cos(theta) - 0.25 * math.cos(2*theta)
        c_imag = 0.5 * math.sin(theta) - 0.25 * math.sin(2*theta)
        points.append((c_real, c_imag, i))
    return points

def encoding_triple_to_complex(gaps, scale=0.05):
    """
    Encoding 4: Three consecutive gaps → position + radius.
    c = scale * (g_n + i*g_{n+1}) with magnitude modulated by g_{n+2}
    """
    points = []
    for i in range(len(gaps) - 2):
        base_r = gaps[i] * scale
        base_i = gaps[i+1] * scale
        # Modulate by third gap
        mod = gaps[i+2] / (gaps[i] + gaps[i+1] + 1)
        c_real = base_r * mod
        c_imag = base_i * mod
        points.append((c_real, c_imag, i))
    return points

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_escape_distribution(points, max_iter=500):
    """Compute escape times and analyze distribution."""
    escapes = []
    for c_real, c_imag, idx in points:
        esc = mandelbrot_escape(c_real, c_imag, max_iter)
        escapes.append((idx, c_real, c_imag, esc))
    return escapes

def compare_to_random(gaps, encoding_func, num_trials=10, max_iter=500, **kwargs):
    """
    Compare real prime gaps to shuffled gaps.
    If primes have structure, escape patterns should differ.
    """
    # Real gaps
    real_points = encoding_func(gaps, **kwargs)
    real_escapes = [mandelbrot_escape(cr, ci, max_iter) for cr, ci, _ in real_points]

    # Statistics for real
    real_mean = sum(real_escapes) / len(real_escapes)
    real_in_set = sum(1 for e in real_escapes if e == max_iter)
    real_fast = sum(1 for e in real_escapes if e < 10)

    # Shuffled trials
    shuffled_means = []
    shuffled_in_sets = []
    shuffled_fasts = []

    for _ in range(num_trials):
        shuf_gaps = gaps.copy()
        random.shuffle(shuf_gaps)
        shuf_points = encoding_func(shuf_gaps, **kwargs)
        shuf_escapes = [mandelbrot_escape(cr, ci, max_iter) for cr, ci, _ in shuf_points]

        shuffled_means.append(sum(shuf_escapes) / len(shuf_escapes))
        shuffled_in_sets.append(sum(1 for e in shuf_escapes if e == max_iter))
        shuffled_fasts.append(sum(1 for e in shuf_escapes if e < 10))

    return {
        'real_mean': real_mean,
        'real_in_set': real_in_set,
        'real_fast_escape': real_fast,
        'shuffled_mean': sum(shuffled_means) / len(shuffled_means),
        'shuffled_in_set': sum(shuffled_in_sets) / len(shuffled_in_sets),
        'shuffled_fast_escape': sum(shuffled_fasts) / len(shuffled_fasts),
        'n_points': len(real_escapes)
    }

def escape_by_gap_value(escapes, gaps):
    """Group escape times by gap value."""
    gap_escapes = defaultdict(list)
    for idx, _, _, esc in escapes:
        if idx < len(gaps):
            gap_escapes[gaps[idx]].append(esc)
    return gap_escapes

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(n_primes):
    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]

    print(f"Primes: 2 to {primes[-1]:,}")
    print(f"Gaps: {len(gaps):,}, range [{min(gaps)}, {max(gaps)}]")
    print(f"Mean gap: {sum(gaps)/len(gaps):.2f}")
    print()

    encodings = [
        ("Consecutive gaps (c = g_n + i*g_{n+1})",
         encoding_consecutive_gaps, {'scale': 0.1}),
        ("Golden spiral",
         encoding_golden_spiral, {'primes': primes, 'scale': 0.5}),
        ("Cardioid boundary",
         encoding_cardioid_param, {'primes': primes}),
        ("Triple modulated",
         encoding_triple_to_complex, {'scale': 0.05}),
    ]

    print("=" * 70)
    print("MANDELBROT ESCAPE ANALYSIS: REAL vs SHUFFLED GAPS")
    print("=" * 70)

    for name, func, kwargs in encodings:
        print(f"\n{'-'*70}")
        print(f"Encoding: {name}")
        print(f"{'-'*70}")

        try:
            results = compare_to_random(gaps, func, num_trials=10, max_iter=500, **kwargs)

            print(f"\n  {'Metric':<25} {'Real':>12} {'Shuffled':>12} {'Ratio':>10}")
            print(f"  {'-'*60}")

            ratio_mean = results['real_mean'] / results['shuffled_mean'] if results['shuffled_mean'] > 0 else float('inf')
            print(f"  {'Mean escape time':<25} {results['real_mean']:>12.2f} {results['shuffled_mean']:>12.2f} {ratio_mean:>10.3f}")

            ratio_in = results['real_in_set'] / results['shuffled_in_set'] if results['shuffled_in_set'] > 0 else float('inf')
            print(f"  {'Points in M-set':<25} {results['real_in_set']:>12} {results['shuffled_in_set']:>12.1f} {ratio_in:>10.3f}")

            ratio_fast = results['real_fast_escape'] / results['shuffled_fast_escape'] if results['shuffled_fast_escape'] > 0 else float('inf')
            print(f"  {'Fast escape (<10)':<25} {results['real_fast_escape']:>12} {results['shuffled_fast_escape']:>12.1f} {ratio_fast:>10.3f}")

            # Significance check
            if abs(ratio_mean - 1.0) > 0.1:
                print(f"\n  ★ Mean escape differs by {abs(ratio_mean-1)*100:.1f}%")
            if abs(ratio_in - 1.0) > 0.2:
                print(f"  ★ M-set membership differs by {abs(ratio_in-1)*100:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")

    # Detailed analysis of best encoding
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS: Consecutive gaps encoding")
    print("=" * 70)

    points = encoding_consecutive_gaps(gaps, scale=0.1)
    escapes = analyze_escape_distribution(points, max_iter=500)

    # Escape time distribution
    esc_hist = defaultdict(int)
    for _, _, _, e in escapes:
        if e < 10:
            esc_hist['<10'] += 1
        elif e < 50:
            esc_hist['10-50'] += 1
        elif e < 100:
            esc_hist['50-100'] += 1
        elif e < 500:
            esc_hist['100-500'] += 1
        else:
            esc_hist['in_set'] += 1

    print("\nEscape time distribution:")
    for bucket in ['<10', '10-50', '50-100', '100-500', 'in_set']:
        count = esc_hist[bucket]
        pct = 100 * count / len(escapes)
        bar = '#' * int(pct / 2)
        print(f"  {bucket:>10}: {count:>6} ({pct:>5.1f}%) {bar}")

    # Escape by gap value
    print("\nEscape time by gap value:")
    gap_esc = escape_by_gap_value(escapes, gaps)

    print(f"  {'Gap':>6} | {'Count':>6} | {'Mean Esc':>10} | {'In M-set':>8}")
    print(f"  {'-'*45}")

    for g in sorted(gap_esc.keys())[:15]:
        escs = gap_esc[g]
        mean_e = sum(escs) / len(escs)
        in_set = sum(1 for e in escs if e >= 500)
        print(f"  {g:>6} | {len(escs):>6} | {mean_e:>10.1f} | {in_set:>8}")

    # Look for patterns: do certain gaps lead to M-set membership?
    print("\n" + "=" * 70)
    print("PATTERN SEARCH: Which gap pairs land in Mandelbrot set?")
    print("=" * 70)

    in_set_pairs = []
    out_set_pairs = []

    for i in range(len(gaps) - 1):
        c_real = gaps[i] * 0.1
        c_imag = gaps[i+1] * 0.1
        esc = mandelbrot_escape(c_real, c_imag, 500)

        if esc >= 500:
            in_set_pairs.append((gaps[i], gaps[i+1]))
        else:
            out_set_pairs.append((gaps[i], gaps[i+1]))

    print(f"\nIn Mandelbrot set: {len(in_set_pairs)} pairs")
    print(f"Outside: {len(out_set_pairs)} pairs")

    if in_set_pairs:
        # Analyze in-set pairs
        in_gap1 = [p[0] for p in in_set_pairs]
        in_gap2 = [p[1] for p in in_set_pairs]

        print(f"\nIn-set pairs statistics:")
        print(f"  g_n mean:   {sum(in_gap1)/len(in_gap1):.2f}")
        print(f"  g_n+1 mean: {sum(in_gap2)/len(in_gap2):.2f}")
        print(f"  Most common g_n: {max(set(in_gap1), key=in_gap1.count)}")
        print(f"  Most common g_n+1: {max(set(in_gap2), key=in_gap2.count)}")

        # Ratio of consecutive gaps for in-set
        ratios_in = [g2/g1 if g1 > 0 else 0 for g1, g2 in in_set_pairs]
        ratios_out = [g2/g1 if g1 > 0 else 0 for g1, g2 in out_set_pairs[:len(in_set_pairs)*10]]

        print(f"\n  Mean ratio g_{'{n+1}'}/g_n (in-set):  {sum(ratios_in)/len(ratios_in):.3f}")
        print(f"  Mean ratio g_{'{n+1}'}/g_n (out-set): {sum(ratios_out)/len(ratios_out):.3f}")

    # 3D MANDELBULB ANALYSIS
    print("\n" + "=" * 70)
    print("3D MANDELBULB ANALYSIS: Triple gaps → (x, y, z)")
    print("=" * 70)

    print("\nComputing Mandelbulb escape times for gap triplets...")
    print("(This uses power=8, the classic Mandelbulb)")

    for scale in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n  Scale = {scale}:")
        try:
            res3d = compare_3d_to_random(gaps, scale=scale, power=8, max_iter=100, num_trials=5)

            ratio_mean = res3d['real_mean'] / res3d['shuffled_mean'] if res3d['shuffled_mean'] > 0 else 0
            ratio_in = res3d['real_in_set'] / res3d['shuffled_in_set'] if res3d['shuffled_in_set'] > 0 else 0

            print(f"    Mean escape:  Real={res3d['real_mean']:.2f}, Shuf={res3d['shuffled_mean']:.2f}, Ratio={ratio_mean:.3f}")
            print(f"    In Bulb:      Real={res3d['real_in_set']}, Shuf={res3d['shuffled_in_set']:.1f}, Ratio={ratio_in:.3f}")
            print(f"    Fast (<5):    Real={res3d['real_fast']}, Shuf={res3d['shuffled_fast']:.1f}")

            if abs(ratio_mean - 1.0) > 0.05:
                print(f"    ★ Escape time differs by {abs(ratio_mean-1)*100:.1f}%!")

        except Exception as e:
            print(f"    Error: {e}")

    # Analyze which triplets land inside Mandelbulb
    print("\n" + "-" * 70)
    print("Which gap triplets land INSIDE the Mandelbulb?")
    print("-" * 70)

    inside = []
    outside = []
    scale = 0.1

    for i in range(len(gaps) - 2):
        cx, cy, cz = gaps[i]*scale, gaps[i+1]*scale, gaps[i+2]*scale
        esc = mandelbulb_escape(cx, cy, cz, power=8, max_iter=100)
        if esc >= 100:
            inside.append((gaps[i], gaps[i+1], gaps[i+2]))
        else:
            outside.append((gaps[i], gaps[i+1], gaps[i+2]))

    print(f"\nInside Mandelbulb: {len(inside)} triplets")
    print(f"Outside: {len(outside)} triplets")

    if inside:
        # Statistics of inside triplets
        g1_in = [t[0] for t in inside]
        g2_in = [t[1] for t in inside]
        g3_in = [t[2] for t in inside]

        print(f"\nInside triplet statistics:")
        print(f"  g_n mean:   {sum(g1_in)/len(g1_in):.2f}")
        print(f"  g_n+1 mean: {sum(g2_in)/len(g2_in):.2f}")
        print(f"  g_n+2 mean: {sum(g3_in)/len(g3_in):.2f}")
        print(f"  Sum mean:   {sum(g1+g2+g3 for g1,g2,g3 in inside)/len(inside):.2f}")

        # Most common triplet patterns
        from collections import Counter
        triplet_counts = Counter(inside)
        print(f"\n  Most common inside triplets:")
        for triplet, count in triplet_counts.most_common(10):
            print(f"    {triplet}: {count} times")

    return {
        'primes': primes,
        'gaps': gaps,
        'escapes': escapes
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    results = run_analysis(n)
