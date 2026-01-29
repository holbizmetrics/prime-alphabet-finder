"""
Prime Gaps as Angles
====================

NOVEL EXPLORATION: Interpret prime gaps as rotation angles.

Approaches:
1. 2D angle walk: Each gap rotates direction by g_n degrees
2. 3D sphere: (θ, φ) from consecutive gaps mapped to sphere
3. Cumulative angle: θ_n = Σg_i mod 360, track clustering

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
import random
from collections import defaultdict, Counter

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
# 2D ANGLE WALK
# ============================================================================

def angle_walk_2d(gaps, scale=1.0, angle_mult=1.0):
    """
    Walk in 2D where each gap rotates direction by g_n * angle_mult degrees.

    Returns list of (x, y) positions.
    """
    x, y = 0.0, 0.0
    angle = 0.0  # degrees
    path = [(x, y)]

    for g in gaps:
        # Rotate by gap degrees
        angle = (angle + g * angle_mult) % 360
        # Move forward
        rad = math.radians(angle)
        x += scale * math.cos(rad)
        y += scale * math.sin(rad)
        path.append((x, y))

    return path

def analyze_walk(path):
    """Analyze properties of a 2D walk."""
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # Final position
    final_dist = math.sqrt(path[-1][0]**2 + path[-1][1]**2)

    # Bounding box
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # Center of mass
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # Spread (std from center)
    spread = math.sqrt(sum((x-cx)**2 + (y-cy)**2 for x, y in path) / len(path))

    # Self-intersections (approximate - grid-based)
    grid = defaultdict(int)
    cell_size = max(width, height) / 100 if max(width, height) > 0 else 1
    for x, y in path:
        cell = (int(x / cell_size), int(y / cell_size))
        grid[cell] += 1
    revisits = sum(1 for v in grid.values() if v > 1)

    return {
        'final_dist': final_dist,
        'width': width,
        'height': height,
        'spread': spread,
        'revisits': revisits,
        'n_points': len(path)
    }

# ============================================================================
# 3D SPHERE MAPPING
# ============================================================================

def gaps_to_sphere(gaps, theta_mult=1.0, phi_mult=1.0):
    """
    Map consecutive gap pairs to points on unit sphere.
    θ = g_n * theta_mult (mod 360) -> azimuthal
    φ = g_{n+1} * phi_mult (mod 180) -> polar

    Returns list of (x, y, z) on unit sphere.
    """
    points = []
    for i in range(len(gaps) - 1):
        theta = math.radians((gaps[i] * theta_mult) % 360)
        phi = math.radians((gaps[i+1] * phi_mult) % 180)

        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        points.append((x, y, z))

    return points

def sphere_clustering(points, n_bins=12):
    """
    Analyze clustering on sphere by dividing into bins.
    Returns entropy-like measure (uniform = high, clustered = low).
    """
    # Bin by (theta_bin, phi_bin)
    bins = defaultdict(int)
    for x, y, z in points:
        theta = math.atan2(y, x)  # -π to π
        phi = math.acos(max(-1, min(1, z)))  # 0 to π

        t_bin = int((theta + math.pi) / (2 * math.pi) * n_bins) % n_bins
        p_bin = int(phi / math.pi * n_bins) % n_bins
        bins[(t_bin, p_bin)] += 1

    # Entropy
    total = len(points)
    entropy = 0
    for count in bins.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    max_entropy = math.log(n_bins * n_bins)  # Uniform distribution

    return {
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': entropy / max_entropy if max_entropy > 0 else 0,
        'n_occupied_bins': len(bins),
        'max_bins': n_bins * n_bins
    }

# ============================================================================
# CUMULATIVE ANGLE
# ============================================================================

def cumulative_angles(gaps, mult=1.0):
    """
    θ_n = (Σ_{i=1}^{n} g_i * mult) mod 360

    Track where cumulative angle lands.
    """
    angles = []
    cumsum = 0
    for g in gaps:
        cumsum = (cumsum + g * mult) % 360
        angles.append(cumsum)
    return angles

def angle_distribution(angles, n_bins=36):
    """Analyze distribution of angles (10-degree bins)."""
    bin_size = 360 / n_bins
    bins = [0] * n_bins
    for a in angles:
        b = int(a / bin_size) % n_bins
        bins[b] += 1

    # Chi-square vs uniform
    expected = len(angles) / n_bins
    chi_sq = sum((obs - expected)**2 / expected for obs in bins)

    return {
        'bins': bins,
        'chi_square': chi_sq,
        'expected_uniform': expected,
        'max_bin': max(bins),
        'min_bin': min(bins)
    }

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

    # ========== 2D ANGLE WALK ==========
    print("=" * 60)
    print("2D ANGLE WALK")
    print("=" * 60)
    print("Each step: rotate by g_n degrees, move forward 1 unit")
    print()

    for mult in [1.0, 6.0, 30.0, 60.0]:
        print(f"Angle multiplier: {mult}")

        # Real
        real_path = angle_walk_2d(gaps, angle_mult=mult)
        real_stats = analyze_walk(real_path)

        # Shuffled
        shuf_gaps = gaps.copy()
        random.shuffle(shuf_gaps)
        shuf_path = angle_walk_2d(shuf_gaps, angle_mult=mult)
        shuf_stats = analyze_walk(shuf_path)

        print(f"  Real:     final_dist={real_stats['final_dist']:.1f}, spread={real_stats['spread']:.1f}, revisits={real_stats['revisits']}")
        print(f"  Shuffled: final_dist={shuf_stats['final_dist']:.1f}, spread={shuf_stats['spread']:.1f}, revisits={shuf_stats['revisits']}")

        # Ratio
        if shuf_stats['final_dist'] > 0:
            print(f"  Ratio (final_dist): {real_stats['final_dist']/shuf_stats['final_dist']:.3f}")
        print()

    # ========== 3D SPHERE ==========
    print("=" * 60)
    print("3D SPHERE MAPPING")
    print("=" * 60)
    print("θ = g_n mod 360, φ = g_{n+1} mod 180 -> point on sphere")
    print()

    for mult in [1.0, 10.0, 30.0]:
        print(f"Angle multiplier: {mult}")

        # Real
        real_pts = gaps_to_sphere(gaps, theta_mult=mult, phi_mult=mult)
        real_clust = sphere_clustering(real_pts)

        # Shuffled
        shuf_pts = gaps_to_sphere(shuf_gaps, theta_mult=mult, phi_mult=mult)
        shuf_clust = sphere_clustering(shuf_pts)

        print(f"  Real:     entropy={real_clust['normalized_entropy']:.3f}, occupied={real_clust['n_occupied_bins']}/{real_clust['max_bins']}")
        print(f"  Shuffled: entropy={shuf_clust['normalized_entropy']:.3f}, occupied={shuf_clust['n_occupied_bins']}/{shuf_clust['max_bins']}")
        print()

    # ========== CUMULATIVE ANGLE ==========
    print("=" * 60)
    print("CUMULATIVE ANGLE DISTRIBUTION")
    print("=" * 60)
    print("θ_n = (Σg_i) mod 360 - where does cumulative angle cluster?")
    print()

    for mult in [1.0, 6.0, 10.0]:
        print(f"Multiplier: {mult}")

        real_angles = cumulative_angles(gaps, mult=mult)
        real_dist = angle_distribution(real_angles)

        shuf_angles = cumulative_angles(shuf_gaps, mult=mult)
        shuf_dist = angle_distribution(shuf_angles)

        print(f"  Real:     χ²={real_dist['chi_square']:.1f}, range=[{real_dist['min_bin']}, {real_dist['max_bin']}]")
        print(f"  Shuffled: χ²={shuf_dist['chi_square']:.1f}, range=[{shuf_dist['min_bin']}, {shuf_dist['max_bin']}]")

        # Show top bins for real
        top_bins = sorted(enumerate(real_dist['bins']), key=lambda x: -x[1])[:5]
        print(f"  Top angles: {[(b*10, c) for b, c in top_bins]}")
        print()

    # ========== MOD 360 DIRECT ==========
    print("=" * 60)
    print("GAP MOD 360 DISTRIBUTION")
    print("=" * 60)
    print("Direct: which angles do gaps occupy?")
    print()

    gap_angles = [(g % 360) for g in gaps]
    angle_counts = Counter(gap_angles)

    print(f"Unique angles: {len(angle_counts)}")
    print(f"Top 10 gap angles:")
    for angle, count in angle_counts.most_common(10):
        print(f"  {angle:>3}°: {count:>5} ({100*count/len(gaps):.1f}%)")

    # Gaps mod 6 (known structure)
    print(f"\nGaps mod 6:")
    mod6 = Counter(g % 6 for g in gaps)
    for m in sorted(mod6.keys()):
        print(f"  {m}: {mod6[m]:>5} ({100*mod6[m]/len(gaps):.1f}%)")

    return {
        'primes': primes,
        'gaps': gaps
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    results = run_analysis(n)
