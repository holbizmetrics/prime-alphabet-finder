"""
Prime-to-Word Embedding Framework (Pure Python - No Dependencies)
Includes: Binary Index Mapping (Approach 5)

Transformation Algebra: P(NLP) ⊗ P(binary) ⊗ T_mrg ⊗ P(primes) → Analysis Space

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations

# ============================================================================
# VECTOR OPERATIONS (Pure Python)
# ============================================================================

def dot(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def cosine_sim(v1: List[float], v2: List[float]) -> float:
    n1, n2 = norm(v1), norm(v2)
    if n1 == 0 or n2 == 0:
        return 0
    return dot(v1, v2) / (n1 * n2)

def euclidean_dist(v1: List[float], v2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    return [a + b for a, b in zip(v1, v2)]

def vec_sub(v1: List[float], v2: List[float]) -> List[float]:
    return [a - b for a, b in zip(v1, v2)]

def vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]

def normalize(v: List[float]) -> List[float]:
    n = norm(v)
    return [x / n for x in v] if n > 0 else v

# ============================================================================
# PRIME GENERATION
# ============================================================================

def sieve_primes(n: int) -> List[int]:
    """Generate primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def first_n_primes(n: int) -> List[int]:
    """Get first n primes."""
    if n <= 0:
        return []
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)) + 2))
    primes = sieve_primes(limit)
    while len(primes) < n:
        limit *= 2
        primes = sieve_primes(limit)
    return primes[:n]

# ============================================================================
# APPROACH 1: PROPERTY VECTORS
# ============================================================================

class PropertyEmbedding:
    """Property-based embeddings."""

    def __init__(self, primes: List[int]):
        self.primes = primes
        self.prime_set = set(primes)
        self.vectors = self._compute()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _compute(self) -> List[List[float]]:
        vectors = []
        for i, p in enumerate(self.primes):
            gap_before = p - self.primes[i-1] if i > 0 else 0
            gap_after = self.primes[i+1] - p if i < len(self.primes) - 1 else 0

            vec = [
                math.log(p + 1),
                i / 1000,
                gap_before / 100,
                gap_after / 100,
                sum(int(d) for d in str(p)) / 50,
                len(str(p)) / 10,
                (p % 6) / 6,
                (p % 30) / 30,
                (p % 210) / 210,
                (p % 10) / 10,
                1.0 if (p + 2) in self.prime_set else 0.0,
                1.0 if (p - 2) in self.prime_set else 0.0,
                1.0 if (2 * p + 1) in self.prime_set else 0.0,
                1.0 if p > 2 and ((p - 1) // 2) in self.prime_set else 0.0,
            ]
            vectors.append(vec)
        return vectors

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

# ============================================================================
# APPROACH 2: SENTENCE EMBEDDING (Gap Patterns)
# ============================================================================

class SentenceEmbedding:
    """Gap-pattern 'sentence' embeddings."""

    def __init__(self, primes: List[int], window: int = 5):
        self.primes = primes
        self.window = window
        self.gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        self.vectors = self._compute()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _compute(self) -> List[List[float]]:
        vectors = []
        half = self.window // 2

        for i in range(len(self.primes)):
            start = max(0, i - half)
            end = min(len(self.gaps), i + half + 1)
            local = self.gaps[start:end]

            # Pad
            while len(local) < self.window:
                local.append(0)
            local = local[:self.window]

            # Features
            vec = [g / 100 for g in local]
            if local:
                vec.extend([
                    sum(local) / len(local) / 100,
                    (max(local) - min(local)) / 100,
                    max(local) / 100,
                    min(local) / 100
                ])
            else:
                vec.extend([0, 0, 0, 0])

            vectors.append(vec)
        return vectors

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

# ============================================================================
# APPROACH 3: FACTORIZATION EMBEDDING (Simplified)
# ============================================================================

class FactorizationEmbedding:
    """Co-occurrence based factorization embeddings."""

    def __init__(self, primes: List[int], max_primes: int = 200):
        self.primes = primes
        self.max_primes = min(max_primes, len(primes))
        self.prime_to_idx = {p: i for i, p in enumerate(primes[:self.max_primes])}
        self.vectors = self._compute()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _factorize(self, n: int) -> List[int]:
        factors = []
        for p in self.primes[:self.max_primes]:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1 and n in self.prime_to_idx:
            factors.append(n)
        return factors

    def _compute(self) -> List[List[float]]:
        # Build co-occurrence
        cooc = [[0.0] * self.max_primes for _ in range(self.max_primes)]
        max_n = min(self.primes[self.max_primes - 1] * 5, 50000)

        for n in range(2, max_n):
            factors = self._factorize(n)
            idxs = [self.prime_to_idx[f] for f in factors if f in self.prime_to_idx]
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    cooc[idxs[a]][idxs[b]] += 1
                    cooc[idxs[b]][idxs[a]] += 1

        # Log transform
        vectors = []
        for i in range(len(self.primes)):
            if i < self.max_primes:
                vec = [math.log1p(c) for c in cooc[i]]
            else:
                vec = [0.0] * self.max_primes
            vectors.append(vec)

        return vectors

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

# ============================================================================
# APPROACH 4: SPECTRAL EMBEDDING
# ============================================================================

class SpectralEmbedding:
    """Frequency/harmonic embeddings."""

    def __init__(self, primes: List[int], harmonics: int = 8):
        self.primes = primes
        self.harmonics = harmonics
        self.vectors = self._compute()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _compute(self) -> List[List[float]]:
        vectors = []
        for i, p in enumerate(self.primes):
            freq = 20 * (1 + math.log(p))

            # Harmonic series
            harms = [math.sin(2 * math.pi * freq * k / 1000) for k in range(1, self.harmonics + 1)]

            # Phase diffs
            phase_prev = (p - self.primes[i-1]) / p if i > 0 else 0
            phase_next = (self.primes[i+1] - p) / p if i < len(self.primes) - 1 else 0

            vec = [
                math.log(freq),
                *harms,
                phase_prev,
                phase_next,
                freq * (1 + 0.1 * sum(harms[:4])) / 1000,
                math.sin(p),
                math.cos(p)
            ]
            vectors.append(vec)
        return vectors

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

# ============================================================================
# APPROACH 5: BINARY INDEX EMBEDDING (NEW - User's Idea)
# ============================================================================

class BinaryIndexEmbedding:
    """
    Map prime p_n → n → binary(n) → features

    This looks at the structure of the prime counting function π(x)
    through the lens of binary representation.
    """

    def __init__(self, primes: List[int], max_bits: int = 16):
        self.primes = primes
        self.max_bits = max_bits
        self.vectors = self._compute()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _to_binary_features(self, n: int) -> List[float]:
        """Convert index n to binary feature vector."""
        # Raw bits (padded to max_bits)
        bits = [(n >> i) & 1 for i in range(self.max_bits)]

        # Bit statistics
        num_ones = sum(bits)
        num_zeros = self.max_bits - num_ones
        leading_zeros = 0
        for b in reversed(bits):
            if b == 0:
                leading_zeros += 1
            else:
                break

        # Run patterns (consecutive 1s or 0s)
        runs = []
        current = bits[0]
        run_len = 1
        for b in bits[1:]:
            if b == current:
                run_len += 1
            else:
                runs.append(run_len)
                current = b
                run_len = 1
        runs.append(run_len)

        avg_run = sum(runs) / len(runs) if runs else 0
        max_run = max(runs) if runs else 0

        # Bit transitions
        transitions = sum(1 for i in range(len(bits)-1) if bits[i] != bits[i+1])

        # Position of highest set bit
        highest_bit = n.bit_length() if n > 0 else 0

        # Binary "density" - ones per bit length
        density = num_ones / highest_bit if highest_bit > 0 else 0

        # Parity features
        even_pos_ones = sum(bits[i] for i in range(0, len(bits), 2))
        odd_pos_ones = sum(bits[i] for i in range(1, len(bits), 2))

        return [
            *[float(b) for b in bits],  # Raw bits
            num_ones / self.max_bits,
            num_zeros / self.max_bits,
            leading_zeros / self.max_bits,
            avg_run / self.max_bits,
            max_run / self.max_bits,
            transitions / self.max_bits,
            highest_bit / self.max_bits,
            density,
            even_pos_ones / (self.max_bits // 2),
            odd_pos_ones / (self.max_bits // 2),
            (even_pos_ones - odd_pos_ones) / self.max_bits,  # Parity imbalance
        ]

    def _compute(self) -> List[List[float]]:
        vectors = []
        for i, p in enumerate(self.primes):
            # Index is i+1 (1-indexed: 2 is the 1st prime)
            idx = i + 1
            vec = self._to_binary_features(idx)

            # Also add features relating binary index to prime value
            # Ratio: how does binary structure of index relate to prime?
            p_bits = p.bit_length()
            idx_bits = idx.bit_length() if idx > 0 else 0

            vec.extend([
                p_bits / 32,  # Prime's bit length
                idx_bits / self.max_bits,
                (p_bits - idx_bits) / 32 if p_bits > 0 else 0,  # Bit length gap
                math.log(p) / math.log(idx + 1) if idx > 0 else 0,  # log ratio (relates to prime number theorem)
            ])

            vectors.append(vec)
        return vectors

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

    def find_binary_patterns(self) -> Dict:
        """Analyze patterns in binary index space."""
        patterns = {
            'power_of_2_indices': [],  # Primes at index 2^k
            'mersenne_indices': [],     # Primes at index 2^k - 1
            'high_density_indices': [], # Indices with many 1-bits
            'low_density_indices': [],  # Indices with few 1-bits
        }

        for i, p in enumerate(self.primes):
            idx = i + 1
            bits = bin(idx).count('1')
            bit_len = idx.bit_length()
            density = bits / bit_len if bit_len > 0 else 0

            # Power of 2
            if idx > 0 and (idx & (idx - 1)) == 0:
                patterns['power_of_2_indices'].append((idx, p))

            # Mersenne (all 1s)
            if idx > 0 and (idx & (idx + 1)) == 0:
                patterns['mersenne_indices'].append((idx, p))

            # Density thresholds
            if density > 0.7:
                patterns['high_density_indices'].append((idx, p, density))
            elif density < 0.3 and bit_len > 3:
                patterns['low_density_indices'].append((idx, p, density))

        return patterns

# ============================================================================
# STACKED EMBEDDING
# ============================================================================

class StackedEmbedding:
    """Combine multiple embeddings."""

    def __init__(self, embeddings: List, weights: Optional[List[float]] = None):
        self.embeddings = embeddings
        self.weights = weights or [1.0] * len(embeddings)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self.vectors = self._stack()
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def _stack(self) -> List[List[float]]:
        if not self.embeddings:
            return []
        n = len(self.embeddings[0].vectors)
        stacked = []
        for i in range(n):
            combined = []
            for emb, w in zip(self.embeddings, self.weights):
                combined.extend(vec_scale(emb.vectors[i], w))
            stacked.append(combined)
        return stacked

    def embed(self, i: int) -> List[float]:
        return self.vectors[i]

    def similarity(self, i: int, j: int) -> float:
        return cosine_sim(self.vectors[i], self.vectors[j])

    def distance(self, i: int, j: int) -> float:
        return euclidean_dist(self.vectors[i], self.vectors[j])

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze(emb, primes: List[int], name: str):
    """Quick analysis of embedding."""
    print(f"\n{'='*50}")
    print(f"{name} (dim={emb.dim})")
    print(f"{'='*50}")

    # Most similar pairs for first few primes
    print("Similar primes:")
    for i in range(min(5, len(primes))):
        sims = [(j, emb.similarity(i, j)) for j in range(len(primes)) if j != i]
        sims.sort(key=lambda x: -x[1])
        top = sims[:3]
        print(f"  {primes[i]:5d} ~ {[(primes[j], f'{s:.2f}') for j, s in top]}")

def find_analogy(emb, primes, a, b, c):
    """a:b :: c:? using vector arithmetic."""
    try:
        ia, ib, ic = primes.index(a), primes.index(b), primes.index(c)
    except ValueError:
        return []

    # target = c + (b - a)
    target = vec_add(emb.embed(ic), vec_sub(emb.embed(ib), emb.embed(ia)))

    results = []
    for i, p in enumerate(primes):
        if p not in [a, b, c]:
            d = euclidean_dist(emb.embed(i), target)
            results.append((p, d))
    results.sort(key=lambda x: x[1])
    return results[:5]

# ============================================================================
# MAIN
# ============================================================================

def main(n_primes: int = 500):
    print(f"Generating {n_primes} primes...")
    primes = first_n_primes(n_primes)
    print(f"Range: 2 to {primes[-1]}")

    # Build all 5 individual embeddings
    print("\nBuilding embeddings...")

    print("  1. Property...")
    prop = PropertyEmbedding(primes)

    print("  2. Sentence (gaps)...")
    sent = SentenceEmbedding(primes)

    print("  3. Factorization...")
    fact = FactorizationEmbedding(primes)

    print("  4. Spectral...")
    spec = SpectralEmbedding(primes)

    print("  5. Binary Index (NEW)...")
    binary = BinaryIndexEmbedding(primes)

    individuals = [
        ('Property', prop),
        ('Sentence', sent),
        ('Factorization', fact),
        ('Spectral', spec),
        ('Binary Index', binary),
    ]

    # Analyze individuals
    for name, emb in individuals:
        analyze(emb, primes, name)

    # Binary patterns analysis
    print("\n" + "="*50)
    print("BINARY INDEX PATTERNS")
    print("="*50)
    patterns = binary.find_binary_patterns()

    print(f"\nPrimes at power-of-2 indices (p_{{2^k}}):")
    for idx, p in patterns['power_of_2_indices'][:10]:
        print(f"  p_{idx} = {p}")

    print(f"\nPrimes at Mersenne indices (p_{{2^k-1}}):")
    for idx, p in patterns['mersenne_indices'][:10]:
        print(f"  p_{idx} = {p}")

    # Build stacks
    print("\n" + "="*50)
    print("STACKED EMBEDDINGS")
    print("="*50)

    emb_list = [prop, sent, fact, spec, binary]
    names = ['prop', 'sent', 'fact', 'spec', 'bin']

    # Full stack
    full = StackedEmbedding(emb_list)
    analyze(full, primes, "FULL STACK (all 5)")

    # Analogy tests
    print("\n" + "="*50)
    print("ANALOGY TESTS (full stack)")
    print("="*50)

    print("\n3:5 (twin) :: 11:?")
    print(f"  Results: {find_analogy(full, primes, 3, 5, 11)}")

    print("\n7:11 (gap 4) :: 13:?")
    print(f"  Results: {find_analogy(full, primes, 7, 11, 13)}")

    print("\n2:3 :: 5:?")
    print(f"  Results: {find_analogy(full, primes, 2, 3, 5)}")

    # Binary-specific analogies
    print("\n" + "="*50)
    print("BINARY-SPECIFIC ANALYSIS")
    print("="*50)

    # Find primes with similar binary index structure
    print("\nPrimes at indices with same bit count:")
    bit_groups = {}
    for i, p in enumerate(primes):
        bits = bin(i + 1).count('1')
        if bits not in bit_groups:
            bit_groups[bits] = []
        bit_groups[bits].append((i + 1, p))

    for bits in sorted(bit_groups.keys())[:6]:
        group = bit_groups[bits][:8]
        print(f"  {bits} bits: {[(idx, p) for idx, p in group]}")

    return primes, individuals, full

if __name__ == "__main__":
    main(500)
