"""
Persistent Homology on Prime Gap Point Clouds
==============================================

GENUINELY UNEXPLORED: TDA (Topological Data Analysis) on prime sequences.

We build point clouds from consecutive prime gaps and compute:
1. Betti numbers (connected components, loops, voids)
2. Persistence diagrams (birth-death of features)
3. Compare to random/shuffled controls

If primes have hidden structure, topology might see it.

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
# POINT CLOUD CONSTRUCTION
# ============================================================================

def gaps_to_point_cloud(gaps, dim=3):
    """
    Convert gap sequence to point cloud.
    Each point is (gap_i, gap_{i+1}, ..., gap_{i+dim-1})
    """
    points = []
    for i in range(len(gaps) - dim + 1):
        point = tuple(gaps[i:i+dim])
        points.append(point)
    return points

def normalize_points(points):
    """Normalize to [0,1] range per dimension."""
    if not points:
        return points

    dim = len(points[0])
    mins = [min(p[d] for p in points) for d in range(dim)]
    maxs = [max(p[d] for p in points) for d in range(dim)]
    ranges = [maxs[d] - mins[d] if maxs[d] > mins[d] else 1 for d in range(dim)]

    normalized = []
    for p in points:
        np = tuple((p[d] - mins[d]) / ranges[d] for d in range(dim))
        normalized.append(np)
    return normalized

# ============================================================================
# DISTANCE FUNCTIONS
# ============================================================================

def euclidean_dist(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

def distance_matrix(points):
    """Compute pairwise distance matrix."""
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_dist(points[i], points[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist

# ============================================================================
# UNION-FIND FOR CONNECTED COMPONENTS
# ============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def num_components(self):
        return len(set(self.find(i) for i in range(len(self.parent))))

# ============================================================================
# SIMPLIFIED PERSISTENT HOMOLOGY (Betti-0 and approximate Betti-1)
# ============================================================================

def compute_persistence_beta0(dist_matrix, max_epsilon=1.0, steps=50):
    """
    Compute β₀ (connected components) as function of epsilon.
    Returns list of (epsilon, num_components).
    """
    n = len(dist_matrix)

    # Get all pairwise distances sorted
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist_matrix[i][j], i, j))
    edges.sort()

    # Track component merges (persistence diagram for H0)
    persistence = []  # (birth, death) pairs
    uf = UnionFind(n)

    # All points born at epsilon=0
    births = {i: 0.0 for i in range(n)}

    beta0_curve = []
    edge_idx = 0

    for step in range(steps + 1):
        epsilon = max_epsilon * step / steps

        # Add all edges with distance <= epsilon
        while edge_idx < len(edges) and edges[edge_idx][0] <= epsilon:
            d, i, j = edges[edge_idx]
            ci, cj = uf.find(i), uf.find(j)
            if ci != cj:
                # Merge: younger component dies
                if births[ci] < births[cj]:
                    dying = cj
                else:
                    dying = ci
                persistence.append((births[dying], d))
                uf.union(i, j)
            edge_idx += 1

        beta0_curve.append((epsilon, uf.num_components()))

    # Remaining components persist to infinity
    final_components = set(uf.find(i) for i in range(n))
    for c in final_components:
        persistence.append((births[c], float('inf')))

    return beta0_curve, persistence

def estimate_beta1(points, dist_matrix, epsilon):
    """
    Estimate β₁ (loops) at given epsilon using Euler characteristic.

    For Vietoris-Rips complex:
    χ = V - E + F - ...
    β₀ - β₁ + β₂ - ... = χ

    For low dimensions, approximate β₁ ≈ E - V + β₀ (ignoring higher)
    """
    n = len(points)

    # Count vertices
    V = n

    # Count edges (pairs within epsilon)
    E = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] <= epsilon:
                E += 1

    # Count triangles (triples all within epsilon)
    F = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] > epsilon:
                continue
            for k in range(j + 1, n):
                if dist_matrix[i][k] <= epsilon and dist_matrix[j][k] <= epsilon:
                    F += 1

    # Compute β₀
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] <= epsilon:
                uf.union(i, j)
    beta0 = uf.num_components()

    # Euler characteristic: χ = V - E + F
    chi = V - E + F

    # β₀ - β₁ + β₂ = χ, assume β₂ ≈ 0 for sparse data
    # β₁ ≈ β₀ - χ = β₀ - V + E - F
    beta1_estimate = beta0 - chi

    return {
        'V': V, 'E': E, 'F': F,
        'chi': chi,
        'beta0': beta0,
        'beta1_est': max(0, beta1_estimate)  # Can't be negative
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_topology(n_primes, dim=3, max_points=2000):
    """Full topological analysis of prime gaps."""

    print(f"Generating {n_primes:,} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]:,}")

    # Compute gaps
    gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
    print(f"Gaps: {len(gaps):,}")
    print(f"Gap range: {min(gaps)} to {max(gaps)}")
    print(f"Mean gap: {sum(gaps)/len(gaps):.2f}")
    print()

    # Build point cloud
    print(f"Building {dim}D point cloud from consecutive gaps...")
    points = gaps_to_point_cloud(gaps, dim=dim)

    # Subsample if too many points (for computational tractability)
    if len(points) > max_points:
        print(f"Subsampling from {len(points)} to {max_points} points...")
        step = len(points) // max_points
        points = points[::step][:max_points]

    print(f"Point cloud size: {len(points)}")

    # Normalize
    points = normalize_points(points)

    # Distance matrix
    print("Computing distance matrix...")
    dist = distance_matrix(points)

    # ========== PERSISTENCE FOR REAL DATA ==========
    print()
    print("="*60)
    print("PERSISTENT HOMOLOGY: REAL PRIME GAPS")
    print("="*60)

    beta0_curve, persistence_h0 = compute_persistence_beta0(dist, max_epsilon=1.0, steps=50)

    print("\nβ₀ (connected components) vs epsilon:")
    for eps, b0 in beta0_curve[::10]:
        bar = "#" * min(50, b0 // 10)
        print(f"  ε={eps:.2f}: β₀={b0:4d} {bar}")

    # Persistence statistics
    lifetimes = [d - b for b, d in persistence_h0 if d != float('inf')]
    if lifetimes:
        print(f"\nH₀ persistence lifetimes:")
        print(f"  Mean: {sum(lifetimes)/len(lifetimes):.4f}")
        print(f"  Max:  {max(lifetimes):.4f}")
        print(f"  Long-lived (>0.1): {sum(1 for l in lifetimes if l > 0.1)}")

    # β₁ at various epsilon
    print("\nβ₁ (loops) estimates:")
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        result = estimate_beta1(points, dist, eps)
        print(f"  ε={eps:.1f}: V={result['V']}, E={result['E']}, F={result['F']}, "
              f"β₀={result['beta0']}, β₁≈{result['beta1_est']}")

    # ========== CONTROL: SHUFFLED GAPS ==========
    print()
    print("="*60)
    print("CONTROL: SHUFFLED GAPS (destroy sequential structure)")
    print("="*60)

    shuffled_gaps = gaps.copy()
    random.shuffle(shuffled_gaps)

    shuffled_points = gaps_to_point_cloud(shuffled_gaps, dim=dim)
    if len(shuffled_points) > max_points:
        step = len(shuffled_points) // max_points
        shuffled_points = shuffled_points[::step][:max_points]
    shuffled_points = normalize_points(shuffled_points)

    shuffled_dist = distance_matrix(shuffled_points)
    shuffled_beta0, shuffled_persistence = compute_persistence_beta0(shuffled_dist, max_epsilon=1.0, steps=50)

    print("\nβ₀ vs epsilon (shuffled):")
    for eps, b0 in shuffled_beta0[::10]:
        bar = "#" * min(50, b0 // 10)
        print(f"  ε={eps:.2f}: β₀={b0:4d} {bar}")

    shuffled_lifetimes = [d - b for b, d in shuffled_persistence if d != float('inf')]
    if shuffled_lifetimes:
        print(f"\nH₀ persistence lifetimes (shuffled):")
        print(f"  Mean: {sum(shuffled_lifetimes)/len(shuffled_lifetimes):.4f}")
        print(f"  Max:  {max(shuffled_lifetimes):.4f}")
        print(f"  Long-lived (>0.1): {sum(1 for l in shuffled_lifetimes if l > 0.1)}")

    print("\nβ₁ estimates (shuffled):")
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        result = estimate_beta1(shuffled_points, shuffled_dist, eps)
        print(f"  ε={eps:.1f}: β₀={result['beta0']}, β₁≈{result['beta1_est']}")

    # ========== CONTROL: RANDOM GAPS ==========
    print()
    print("="*60)
    print("CONTROL: RANDOM GAPS (Cramér model)")
    print("="*60)

    # Cramér model: gaps ~ exponential with mean log(p)
    mean_gap = sum(gaps) / len(gaps)
    random_gaps = [max(2, int(random.expovariate(1/mean_gap))) for _ in range(len(gaps))]
    # Round to even (prime gaps > 2 are even)
    random_gaps = [g if g == 2 else g + (g % 2) for g in random_gaps]

    random_points = gaps_to_point_cloud(random_gaps, dim=dim)
    if len(random_points) > max_points:
        step = len(random_points) // max_points
        random_points = random_points[::step][:max_points]
    random_points = normalize_points(random_points)

    random_dist = distance_matrix(random_points)
    random_beta0, random_persistence = compute_persistence_beta0(random_dist, max_epsilon=1.0, steps=50)

    print("\nβ₀ vs epsilon (random Cramér):")
    for eps, b0 in random_beta0[::10]:
        bar = "#" * min(50, b0 // 10)
        print(f"  ε={eps:.2f}: β₀={b0:4d} {bar}")

    random_lifetimes = [d - b for b, d in random_persistence if d != float('inf')]
    if random_lifetimes:
        print(f"\nH₀ persistence lifetimes (random):")
        print(f"  Mean: {sum(random_lifetimes)/len(random_lifetimes):.4f}")
        print(f"  Max:  {max(random_lifetimes):.4f}")

    print("\nβ₁ estimates (random):")
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        result = estimate_beta1(random_points, random_dist, eps)
        print(f"  ε={eps:.1f}: β₀={result['beta0']}, β₁≈{result['beta1_est']}")

    # ========== COMPARISON ==========
    print()
    print("="*60)
    print("COMPARISON: REAL vs CONTROLS")
    print("="*60)

    # Compare at epsilon = 0.3 (typical interesting scale)
    eps = 0.3
    real_stats = estimate_beta1(points, dist, eps)
    shuf_stats = estimate_beta1(shuffled_points, shuffled_dist, eps)
    rand_stats = estimate_beta1(random_points, random_dist, eps)

    print(f"\nAt ε = {eps}:")
    print(f"              β₀    β₁    Edges   Triangles")
    print(f"  Real:      {real_stats['beta0']:4d}  {real_stats['beta1_est']:4d}  {real_stats['E']:6d}  {real_stats['F']:6d}")
    print(f"  Shuffled:  {shuf_stats['beta0']:4d}  {shuf_stats['beta1_est']:4d}  {shuf_stats['E']:6d}  {shuf_stats['F']:6d}")
    print(f"  Random:    {rand_stats['beta0']:4d}  {rand_stats['beta1_est']:4d}  {rand_stats['E']:6d}  {rand_stats['F']:6d}")

    # Verdict
    print()
    print("="*60)
    print("VERDICT")
    print("="*60)

    real_mean_life = sum(lifetimes)/len(lifetimes) if lifetimes else 0
    shuf_mean_life = sum(shuffled_lifetimes)/len(shuffled_lifetimes) if shuffled_lifetimes else 0
    rand_mean_life = sum(random_lifetimes)/len(random_lifetimes) if random_lifetimes else 0

    print(f"\nMean H₀ persistence lifetime:")
    print(f"  Real:     {real_mean_life:.4f}")
    print(f"  Shuffled: {shuf_mean_life:.4f}")
    print(f"  Random:   {rand_mean_life:.4f}")

    if real_stats['beta1_est'] > shuf_stats['beta1_est'] * 1.5:
        print("\n★ REAL has MORE LOOPS than shuffled!")
        print("  This suggests genuine topological structure in prime gaps.")
    elif real_stats['beta1_est'] < shuf_stats['beta1_est'] * 0.67:
        print("\n★ REAL has FEWER LOOPS than shuffled!")
        print("  Prime gaps may be more 'spread out' / less clustered.")
    else:
        print("\n→ No significant difference in loop structure.")

    if abs(real_mean_life - shuf_mean_life) > 0.02:
        print(f"\n★ Persistence differs: real={real_mean_life:.4f}, shuffled={shuf_mean_life:.4f}")

    return {
        'real': {'beta0_curve': beta0_curve, 'persistence': persistence_h0},
        'shuffled': {'beta0_curve': shuffled_beta0, 'persistence': shuffled_persistence},
        'random': {'beta0_curve': random_beta0, 'persistence': random_persistence}
    }

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    results = analyze_topology(n)
