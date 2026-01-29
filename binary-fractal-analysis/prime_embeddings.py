"""
Prime-to-Word Embedding Framework
Transformation Algebra Application: P(NLP) ⊗ T_mrg ⊗ P(primes) → Analysis Space

Author: PROMETHEUS v4.1.1 + Human
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import math

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
    # Estimate upper bound using prime number theorem
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

@dataclass
class PrimeProperties:
    """Feature vector for a single prime."""
    value: int
    index: int  # nth prime
    gap_before: int
    gap_after: int
    digit_sum: int
    num_digits: int
    mod_6: int
    mod_30: int
    mod_210: int  # primorial
    last_digit: int
    is_twin_lower: bool  # p where p+2 is prime
    is_twin_upper: bool  # p where p-2 is prime
    is_sophie_germain: bool  # p where 2p+1 is prime
    is_safe_prime: bool  # p where (p-1)/2 is prime

def compute_prime_properties(primes: List[int]) -> List[PrimeProperties]:
    """Compute property vectors for all primes."""
    prime_set = set(primes)
    properties = []

    for i, p in enumerate(primes):
        gap_before = p - primes[i-1] if i > 0 else 0
        gap_after = primes[i+1] - p if i < len(primes) - 1 else 0

        props = PrimeProperties(
            value=p,
            index=i,
            gap_before=gap_before,
            gap_after=gap_after,
            digit_sum=sum(int(d) for d in str(p)),
            num_digits=len(str(p)),
            mod_6=p % 6,
            mod_30=p % 30,
            mod_210=p % 210,
            last_digit=p % 10,
            is_twin_lower=(p + 2) in prime_set,
            is_twin_upper=(p - 2) in prime_set,
            is_sophie_germain=(2 * p + 1) in prime_set,
            is_safe_prime=((p - 1) // 2) in prime_set and p > 2
        )
        properties.append(props)

    return properties

def properties_to_vector(props: PrimeProperties) -> np.ndarray:
    """Convert PrimeProperties to numpy vector."""
    return np.array([
        np.log(props.value + 1),  # log scale for value
        props.index / 1000,  # normalized index
        props.gap_before / 100,  # normalized gaps
        props.gap_after / 100,
        props.digit_sum / 50,
        props.num_digits / 10,
        props.mod_6 / 6,
        props.mod_30 / 30,
        props.mod_210 / 210,
        props.last_digit / 10,
        float(props.is_twin_lower),
        float(props.is_twin_upper),
        float(props.is_sophie_germain),
        float(props.is_safe_prime),
    ], dtype=np.float32)

class PropertyEmbedding:
    """Approach 1: Property-based embeddings."""

    def __init__(self, primes: List[int]):
        self.primes = primes
        self.properties = compute_prime_properties(primes)
        self.vectors = np.array([properties_to_vector(p) for p in self.properties])
        self.dim = self.vectors.shape[1]

    def embed(self, prime_index: int) -> np.ndarray:
        return self.vectors[prime_index]

    def similarity(self, i: int, j: int) -> float:
        """Cosine similarity between two primes."""
        v1, v2 = self.vectors[i], self.vectors[j]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between two primes."""
        return np.linalg.norm(self.vectors[i] - self.vectors[j])

    def find_similar(self, prime_index: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar primes by cosine similarity."""
        sims = []
        for i in range(len(self.primes)):
            if i != prime_index:
                sims.append((i, self.similarity(prime_index, i)))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

# ============================================================================
# APPROACH 2: PRIME "SENTENCES" (N-gram style)
# ============================================================================

class PrimeSentenceEmbedding:
    """Approach 2: Treat prime sequences as sentences."""

    def __init__(self, primes: List[int], window_size: int = 5):
        self.primes = primes
        self.window_size = window_size
        self.gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        # Create "sentences" - sliding windows of gap patterns
        self.sentences = self._create_sentences()

        # Simple co-occurrence based embedding
        self.vectors = self._compute_embeddings()
        self.dim = self.vectors.shape[1] if len(self.vectors) > 0 else 0

    def _create_sentences(self) -> List[List[int]]:
        """Create sliding window 'sentences' of gaps."""
        sentences = []
        for i in range(len(self.gaps) - self.window_size + 1):
            sentences.append(self.gaps[i:i + self.window_size])
        return sentences

    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings based on gap context patterns."""
        # Each prime gets embedded based on its surrounding gap pattern
        embeddings = []
        half_w = self.window_size // 2

        for i in range(len(self.primes)):
            # Get surrounding gaps
            start = max(0, i - half_w)
            end = min(len(self.gaps), i + half_w)
            local_gaps = self.gaps[start:end]

            # Pad if necessary
            while len(local_gaps) < self.window_size:
                local_gaps.append(0)

            # Create feature vector from gap pattern
            vec = np.array(local_gaps[:self.window_size], dtype=np.float32) / 100.0

            # Add gap statistics
            gap_mean = np.mean(local_gaps) if local_gaps else 0
            gap_std = np.std(local_gaps) if local_gaps else 0
            gap_max = max(local_gaps) if local_gaps else 0
            gap_min = min(local_gaps) if local_gaps else 0

            extended = np.concatenate([
                vec,
                [gap_mean/100, gap_std/50, gap_max/100, gap_min/100]
            ])
            embeddings.append(extended)

        return np.array(embeddings)

    def embed(self, prime_index: int) -> np.ndarray:
        return self.vectors[prime_index]

    def similarity(self, i: int, j: int) -> float:
        v1, v2 = self.vectors[i], self.vectors[j]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    def distance(self, i: int, j: int) -> float:
        return np.linalg.norm(self.vectors[i] - self.vectors[j])

# ============================================================================
# APPROACH 3: FACTORIZATION VOCABULARY
# ============================================================================

class FactorizationEmbedding:
    """Approach 3: Primes as atomic tokens, factorizations as words."""

    def __init__(self, primes: List[int], context_range: int = 50):
        self.primes = primes
        self.prime_set = set(primes)
        self.prime_to_idx = {p: i for i, p in enumerate(primes)}
        self.context_range = context_range

        # Build co-occurrence matrix (which primes appear "near" each other)
        self.cooccurrence = self._build_cooccurrence()

        # SVD for dimensionality reduction
        self.vectors = self._compute_embeddings()
        self.dim = self.vectors.shape[1] if len(self.vectors) > 0 else 0

    def _factorize(self, n: int) -> List[int]:
        """Return prime factors of n."""
        factors = []
        for p in self.primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1 and n in self.prime_set:
            factors.append(n)
        return factors

    def _build_cooccurrence(self) -> np.ndarray:
        """Build prime co-occurrence matrix from factorizations."""
        n_primes = min(len(self.primes), 1000)  # Limit for memory
        cooc = np.zeros((n_primes, n_primes), dtype=np.float32)

        # For numbers in range, count co-occurrences in factorizations
        max_n = self.primes[n_primes - 1] * 10 if n_primes > 0 else 1000
        max_n = min(max_n, 100000)  # Cap for performance

        for n in range(2, max_n):
            factors = self._factorize(n)
            factor_indices = [self.prime_to_idx[f] for f in factors if f in self.prime_to_idx and self.prime_to_idx[f] < n_primes]

            # Update co-occurrence
            for i, idx1 in enumerate(factor_indices):
                for idx2 in factor_indices[i+1:]:
                    cooc[idx1, idx2] += 1
                    cooc[idx2, idx1] += 1

        return cooc

    def _compute_embeddings(self, dim: int = 32) -> np.ndarray:
        """Compute embeddings via SVD of co-occurrence matrix."""
        if self.cooccurrence.shape[0] == 0:
            return np.array([])

        # Log transform (like GloVe)
        log_cooc = np.log1p(self.cooccurrence)

        # SVD
        try:
            U, S, Vt = np.linalg.svd(log_cooc, full_matrices=False)
            # Use top dimensions
            dim = min(dim, len(S))
            embeddings = U[:, :dim] * np.sqrt(S[:dim])
        except:
            # Fallback: use raw matrix rows
            embeddings = log_cooc[:, :dim] if log_cooc.shape[1] >= dim else log_cooc

        # Pad to full prime list if needed
        if len(embeddings) < len(self.primes):
            padding = np.zeros((len(self.primes) - len(embeddings), embeddings.shape[1]))
            embeddings = np.vstack([embeddings, padding])

        return embeddings

    def embed(self, prime_index: int) -> np.ndarray:
        if prime_index < len(self.vectors):
            return self.vectors[prime_index]
        return np.zeros(self.dim)

    def similarity(self, i: int, j: int) -> float:
        v1, v2 = self.embed(i), self.embed(j)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm + 1e-10) if norm > 0 else 0

    def distance(self, i: int, j: int) -> float:
        return np.linalg.norm(self.embed(i) - self.embed(j))

# ============================================================================
# APPROACH 4: SPECTRAL EMBEDDING
# ============================================================================

class SpectralEmbedding:
    """Approach 4: Prime → frequency → spectral features."""

    def __init__(self, primes: List[int], n_harmonics: int = 8):
        self.primes = primes
        self.n_harmonics = n_harmonics
        self.vectors = self._compute_embeddings()
        self.dim = self.vectors.shape[1] if len(self.vectors) > 0 else 0

    def _prime_to_frequency(self, p: int, base_freq: float = 20.0) -> float:
        """Map prime to frequency (log scale for musical perception)."""
        return base_freq * (1 + np.log(p))

    def _compute_embeddings(self) -> np.ndarray:
        """Compute spectral embeddings."""
        embeddings = []

        for i, p in enumerate(self.primes):
            freq = self._prime_to_frequency(p)

            # Harmonic series features
            harmonics = [np.sin(2 * np.pi * freq * k / 1000) for k in range(1, self.n_harmonics + 1)]

            # Phase features (relative to neighbors)
            if i > 0:
                phase_diff_prev = (p - self.primes[i-1]) / p
            else:
                phase_diff_prev = 0

            if i < len(self.primes) - 1:
                phase_diff_next = (self.primes[i+1] - p) / p
            else:
                phase_diff_next = 0

            # Spectral centroid approximation
            spectral_centroid = freq * (1 + 0.1 * sum(harmonics[:4]))

            # Combine features
            vec = np.array([
                np.log(freq),
                *harmonics,
                phase_diff_prev,
                phase_diff_next,
                spectral_centroid / 1000,
                np.sin(p),  # Cyclic features
                np.cos(p),
            ], dtype=np.float32)

            embeddings.append(vec)

        return np.array(embeddings)

    def embed(self, prime_index: int) -> np.ndarray:
        return self.vectors[prime_index]

    def similarity(self, i: int, j: int) -> float:
        v1, v2 = self.vectors[i], self.vectors[j]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    def distance(self, i: int, j: int) -> float:
        return np.linalg.norm(self.vectors[i] - self.vectors[j])

# ============================================================================
# STACKED EMBEDDINGS
# ============================================================================

class StackedEmbedding:
    """Combine multiple embedding approaches."""

    def __init__(self, embeddings: List, weights: Optional[List[float]] = None):
        self.embeddings = embeddings
        self.weights = weights or [1.0] * len(embeddings)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Concatenate vectors
        self.vectors = self._stack_vectors()
        self.dim = self.vectors.shape[1] if len(self.vectors) > 0 else 0

    def _stack_vectors(self) -> np.ndarray:
        """Concatenate all embedding vectors with weights."""
        if not self.embeddings:
            return np.array([])

        n_items = len(self.embeddings[0].vectors)
        stacked = []

        for i in range(n_items):
            vecs = []
            for emb, w in zip(self.embeddings, self.weights):
                if i < len(emb.vectors):
                    vecs.append(emb.vectors[i] * w)
                else:
                    vecs.append(np.zeros(emb.dim) * w)
            stacked.append(np.concatenate(vecs))

        return np.array(stacked)

    def embed(self, prime_index: int) -> np.ndarray:
        return self.vectors[prime_index]

    def similarity(self, i: int, j: int) -> float:
        v1, v2 = self.vectors[i], self.vectors[j]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    def distance(self, i: int, j: int) -> float:
        return np.linalg.norm(self.vectors[i] - self.vectors[j])

# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_embedding(emb, primes: List[int], name: str = "Embedding"):
    """Analyze an embedding space."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*60}")
    print(f"Dimension: {emb.dim}")
    print(f"Primes covered: {len(primes)}")

    # Sample similarities
    print(f"\nSample similarities (first 10 primes):")
    for i in range(min(5, len(primes))):
        similar = []
        for j in range(len(primes)):
            if i != j:
                similar.append((primes[j], emb.similarity(i, j)))
        similar.sort(key=lambda x: -x[1])
        top3 = similar[:3]
        print(f"  {primes[i]} most similar to: {[(p, f'{s:.3f}') for p, s in top3]}")

    # Distance statistics
    if len(primes) >= 10:
        sample_dists = []
        for _ in range(100):
            i, j = np.random.randint(0, min(len(primes), 100), 2)
            if i != j:
                sample_dists.append(emb.distance(i, j))
        print(f"\nDistance stats (sample): mean={np.mean(sample_dists):.3f}, std={np.std(sample_dists):.3f}")

    return emb

def find_analogies(emb, primes: List[int], a: int, b: int, c: int, top_k: int = 5):
    """
    Find d such that a:b :: c:d
    Using vector arithmetic: d = c + (b - a)
    """
    idx_a = primes.index(a) if a in primes else None
    idx_b = primes.index(b) if b in primes else None
    idx_c = primes.index(c) if c in primes else None

    if None in [idx_a, idx_b, idx_c]:
        print("Prime not found in list")
        return []

    # Compute target vector
    target = emb.embed(idx_c) + (emb.embed(idx_b) - emb.embed(idx_a))

    # Find closest primes
    results = []
    for i, p in enumerate(primes):
        if p not in [a, b, c]:
            dist = np.linalg.norm(emb.embed(i) - target)
            results.append((p, dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

# ============================================================================
# MAIN: BUILD ALL 15 ANALYSIS PATHS
# ============================================================================

def build_all_embeddings(n_primes: int = 1000):
    """Build all 15 embedding approaches."""
    print(f"Generating first {n_primes} primes...")
    primes = first_n_primes(n_primes)
    print(f"Generated {len(primes)} primes (2 to {primes[-1]})")

    # Individual approaches
    print("\nBuilding individual embeddings...")

    print("  1. Property embedding...")
    prop_emb = PropertyEmbedding(primes)

    print("  2. Sentence embedding...")
    sent_emb = PrimeSentenceEmbedding(primes)

    print("  3. Factorization embedding...")
    fact_emb = FactorizationEmbedding(primes)

    print("  4. Spectral embedding...")
    spec_emb = SpectralEmbedding(primes)

    individuals = {
        'property': prop_emb,
        'sentence': sent_emb,
        'factorization': fact_emb,
        'spectral': spec_emb
    }

    # All stacks
    print("\nBuilding stacked embeddings...")
    emb_list = [prop_emb, sent_emb, fact_emb, spec_emb]
    names = ['property', 'sentence', 'factorization', 'spectral']

    stacks = {}

    # Pairs (6)
    for i, j in combinations(range(4), 2):
        name = f"{names[i]}+{names[j]}"
        print(f"  {name}...")
        stacks[name] = StackedEmbedding([emb_list[i], emb_list[j]])

    # Triples (4)
    for combo in combinations(range(4), 3):
        name = "+".join(names[i] for i in combo)
        print(f"  {name}...")
        stacks[name] = StackedEmbedding([emb_list[i] for i in combo])

    # Full (1)
    print("  full stack (all 4)...")
    stacks['full'] = StackedEmbedding(emb_list)

    return primes, individuals, stacks

if __name__ == "__main__":
    # Build everything
    primes, individuals, stacks = build_all_embeddings(1000)

    # Analyze each
    print("\n" + "="*60)
    print("INDIVIDUAL EMBEDDINGS")
    print("="*60)

    for name, emb in individuals.items():
        analyze_embedding(emb, primes, name.upper())

    print("\n" + "="*60)
    print("STACKED EMBEDDINGS")
    print("="*60)

    for name, emb in stacks.items():
        analyze_embedding(emb, primes, name.upper())

    # Test analogies on full stack
    print("\n" + "="*60)
    print("ANALOGY TEST (full stack)")
    print("="*60)

    full = stacks['full']

    # Twin prime analogy: 3:5 :: 11:?
    print("\nAnalogy: 3:5 (twin) :: 11:? (expecting 13)")
    results = find_analogies(full, primes, 3, 5, 11)
    print(f"Results: {results}")

    # Gap pattern: 7:11 :: 13:?
    print("\nAnalogy: 7:11 (gap 4) :: 13:? (expecting 17)")
    results = find_analogies(full, primes, 7, 11, 13)
    print(f"Results: {results}")
