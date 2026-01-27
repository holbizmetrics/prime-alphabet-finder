#!/usr/bin/env python3
"""
Dimensional Analysis of Prime Patterns
Projects primes into various representations and dimensional spaces.
"""

import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from prime_encoder_extended import *

# =============================================================================
# NUMBER REPRESENTATIONS
# =============================================================================

def to_roman(n: int) -> str:
    """Convert number to Roman numerals."""
    if n <= 0 or n > 3999:
        return str(n)

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']

    result = ''
    for i, v in enumerate(val):
        while n >= v:
            result += syms[i]
            n -= v
    return result

def to_binary(n: int) -> str:
    """Convert to binary string."""
    return bin(n)[2:]

def to_ternary(n: int) -> str:
    """Convert to base 3."""
    if n == 0: return '0'
    digits = []
    while n:
        digits.append(str(n % 3))
        n //= 3
    return ''.join(reversed(digits))

def to_octal(n: int) -> str:
    """Convert to octal."""
    return oct(n)[2:]

def to_hexadecimal(n: int) -> str:
    """Convert to hexadecimal."""
    return hex(n)[2:].upper()

def to_base(n: int, base: int) -> str:
    """Convert to any base (2-36)."""
    if n == 0: return '0'
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result = ''
    while n:
        result = digits[n % base] + result
        n //= base
    return result

def to_balanced_ternary(n: int) -> str:
    """Convert to balanced ternary (digits: -, 0, +)."""
    if n == 0: return '0'
    result = ''
    while n != 0:
        rem = n % 3
        if rem == 0:
            result = '0' + result
            n //= 3
        elif rem == 1:
            result = '+' + result
            n //= 3
        else:  # rem == 2
            result = '-' + result
            n = (n + 1) // 3
    return result

def to_factorial_base(n: int) -> str:
    """Convert to factorial number system."""
    if n == 0: return '0'
    result = []
    i = 2
    while n > 0:
        result.append(str(n % i))
        n //= i
        i += 1
    return ''.join(reversed(result))

def to_primorial_base(n: int) -> str:
    """Convert to primorial number system."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if n == 0: return '0'
    result = []
    for p in primes:
        result.append(str(n % p))
        n //= p
        if n == 0: break
    return ''.join(reversed(result)) if result else '0'

def to_egyptian_fractions(n: int, d: int = 1) -> str:
    """Represent as sum of unit fractions (for ratios)."""
    # For a prime p, represent p/1 as sum of unit fractions
    if d == 1:
        return f"1/{n}" * 0 + str(n)  # Just return the number for integers
    fractions = []
    while n > 0:
        # Find smallest unit fraction <= n/d
        unit = (d + n - 1) // n
        fractions.append(f"1/{unit}")
        n = n * unit - d
        d = d * unit
        # Simplify
        from math import gcd
        g = gcd(n, d)
        n //= g
        d //= g
    return ' + '.join(fractions)

# =============================================================================
# REPRESENTATION ENCODINGS TO ALPHABET
# =============================================================================

def roman_letter_count(n: int) -> int:
    """Count letters in Roman numeral representation."""
    return len(to_roman(n))

def binary_length(n: int) -> int:
    """Length of binary representation."""
    return len(to_binary(n))

def binary_ones(n: int) -> int:
    """Count of 1s in binary."""
    return to_binary(n).count('1')

def binary_zeros(n: int) -> int:
    """Count of 0s in binary."""
    return to_binary(n).count('0')

def ternary_length(n: int) -> int:
    """Length of ternary representation."""
    return len(to_ternary(n))

def hex_letter_count(n: int) -> int:
    """Count of letters (A-F) in hex representation."""
    return sum(1 for c in to_hexadecimal(n) if c.isalpha())

def representation_to_letter(n: int, method: str, alphabet_size: int = 26) -> str:
    """Convert number to letter using various representation methods."""
    if method == 'roman_length':
        val = roman_letter_count(n)
    elif method == 'binary_length':
        val = binary_length(n)
    elif method == 'binary_ones':
        val = binary_ones(n)
    elif method == 'binary_zeros':
        val = binary_zeros(n)
    elif method == 'ternary_length':
        val = ternary_length(n)
    elif method == 'hex_letters':
        val = hex_letter_count(n)
    elif method == 'digit_count':
        val = len(str(n))
    elif method == 'factorial_length':
        val = len(to_factorial_base(n))
    else:
        val = n % alphabet_size + 1

    return chr(ord('A') + ((val - 1) % alphabet_size))

REPRESENTATION_METHODS = {
    'roman_length': roman_letter_count,
    'binary_length': binary_length,
    'binary_ones': binary_ones,
    'binary_zeros': binary_zeros,
    'ternary_length': ternary_length,
    'hex_letters': hex_letter_count,
    'digit_count': lambda n: len(str(n)),
    'factorial_length': lambda n: len(to_factorial_base(n)),
}

# =============================================================================
# DIMENSIONAL PROJECTIONS
# =============================================================================

def project_1d(primes: List[int]) -> List[int]:
    """1D projection: just the sequence of prime indices."""
    return [prime_index(p) for p in primes]

def project_2d(primes: List[int], width: int = 10) -> List[Tuple[int, int]]:
    """2D projection: wrap primes into a grid."""
    return [(i % width, i // width) for i in range(len(primes))]

def project_2d_ulam(primes: List[int], limit: int) -> List[Tuple[int, int]]:
    """Project primes onto Ulam spiral coordinates."""
    # Generate Ulam spiral positions for numbers 1 to limit
    positions = {}
    x, y = 0, 0
    dx, dy = 1, 0  # Start moving right
    positions[1] = (x, y)

    for n in range(2, limit + 1):
        x, y = x + dx, y + dy
        positions[n] = (x, y)

        # Turn left if the cell to the left is empty
        new_dx, new_dy = -dy, dx  # Turn left
        if (x + new_dx, y + new_dy) not in [(positions[i]) for i in range(1, n)]:
            dx, dy = new_dx, new_dy

    # Return positions of primes
    return [positions.get(p, (0, 0)) for p in primes if p in positions]

def project_3d_helix(primes: List[int], radius: float = 1.0, pitch: float = 0.5) -> List[Tuple[float, float, float]]:
    """Project primes onto a 3D helix."""
    coords = []
    for i, p in enumerate(primes):
        theta = i * 0.5  # Angle increases with index
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = pitch * i
        coords.append((x, y, z))
    return coords

def project_3d_sphere(primes: List[int]) -> List[Tuple[float, float, float]]:
    """Project primes onto a sphere using Fibonacci lattice."""
    coords = []
    n = len(primes)
    golden_ratio = (1 + math.sqrt(5)) / 2

    for i in range(n):
        theta = 2 * math.pi * i / golden_ratio
        phi = math.acos(1 - 2 * (i + 0.5) / n)

        x = math.cos(theta) * math.sin(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(phi)

        # Scale by prime value
        scale = primes[i] / max(primes)
        coords.append((x * scale, y * scale, z * scale))

    return coords

def project_4d_tesseract(primes: List[int]) -> List[Tuple[float, float, float, float]]:
    """Project primes into 4D space."""
    coords = []
    for i, p in enumerate(primes):
        # Use prime properties for different dimensions
        x = p % 26  # mod 26
        y = digit_sum(p) % 10  # digit sum
        z = binary_ones(p)  # binary 1s
        w = prime_index(p) % 10 if is_prime(p) else 0  # prime index
        coords.append((x, y, z, w))
    return coords

def project_nd(primes: List[int], dimensions: int) -> List[Tuple]:
    """Project primes into n-dimensional space using various properties."""
    property_funcs = [
        lambda p: p % 26,  # mod 26
        lambda p: digit_sum(p),  # digit sum
        lambda p: binary_ones(p),  # binary 1s
        lambda p: len(str(p)),  # digit count
        lambda p: roman_letter_count(p),  # roman length
        lambda p: p % 10,  # last digit
        lambda p: p // 10 % 10 if p >= 10 else 0,  # second to last digit
        lambda p: digital_root(p),  # digital root
        lambda p: prime_index(p) % 26 if is_prime(p) else 0,  # prime index mod
        lambda p: len(to_binary(p)),  # binary length
    ]

    coords = []
    for p in primes:
        point = tuple(property_funcs[d % len(property_funcs)](p) for d in range(dimensions))
        coords.append(point)
    return coords

# =============================================================================
# PATTERN ANALYSIS IN DIFFERENT SPACES
# =============================================================================

def find_collinear_triples_2d(points: List[Tuple[int, int]], primes: List[int]) -> List[Tuple]:
    """Find collinear triples in 2D projection."""
    collinear = []
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Check collinearity using cross product
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]

                cross = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
                if cross == 0:
                    collinear.append((primes[i], primes[j], primes[k]))

    return collinear[:100]  # Limit results

def find_equidistant_triples_nd(coords: List[Tuple], primes: List[int]) -> List[Tuple]:
    """Find equidistant triples in n-dimensional space."""
    equidistant = []
    n = len(coords)

    def distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

    for i in range(min(n, 50)):  # Limit for performance
        for j in range(i + 1, min(n, 50)):
            d1 = distance(coords[i], coords[j])
            for k in range(j + 1, min(n, 50)):
                d2 = distance(coords[j], coords[k])
                d3 = distance(coords[i], coords[k])

                if abs(d1 - d2) < 0.001 and abs(d2 - d3) < 0.001:
                    equidistant.append((primes[i], primes[j], primes[k]))

    return equidistant

def cluster_primes_nd(coords: List[Tuple], primes: List[int], n_clusters: int = 5) -> Dict[int, List[int]]:
    """Simple k-means clustering of primes in n-dimensional space."""
    import random

    if len(coords) < n_clusters:
        return {0: primes}

    # Initialize centroids randomly
    centroids = random.sample(coords, n_clusters)

    def distance(p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

    # Iterate
    for _ in range(10):
        # Assign points to nearest centroid
        clusters = defaultdict(list)
        for i, coord in enumerate(coords):
            distances = [distance(coord, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(i)

        # Update centroids
        for k in range(n_clusters):
            if clusters[k]:
                cluster_coords = [coords[i] for i in clusters[k]]
                dim = len(coords[0])
                new_centroid = tuple(
                    sum(c[d] for c in cluster_coords) / len(cluster_coords)
                    for d in range(dim)
                )
                centroids[k] = new_centroid

    # Return clusters with actual prime values
    return {k: [primes[i] for i in indices] for k, indices in clusters.items()}

# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def analyze_triplet_all_representations(triplet: Tuple[int, int, int]) -> Dict:
    """Analyze a triplet using all representations and methods."""
    results = {
        'triplet': triplet,
        'sum': sum(triplet),
        'representations': {},
        'encodings': {},
        'dimensional': {},
    }

    # Various representations
    for p in triplet:
        results['representations'][p] = {
            'decimal': str(p),
            'binary': to_binary(p),
            'ternary': to_ternary(p),
            'octal': to_octal(p),
            'hex': to_hexadecimal(p),
            'roman': to_roman(p),
            'balanced_ternary': to_balanced_ternary(p),
            'factorial': to_factorial_base(p),
        }

    # Encodings using representation methods
    for method_name, method_func in REPRESENTATION_METHODS.items():
        letters = ''.join(representation_to_letter(p, method_name) for p in triplet)
        is_triple = len(set(letters)) == 1
        results['encodings'][method_name] = {
            'letters': letters,
            'is_triple': is_triple,
            'values': [method_func(p) for p in triplet]
        }

    # Dimensional projections
    coords_4d = project_nd(list(triplet), 4)
    results['dimensional']['4d_coords'] = coords_4d

    return results

def comprehensive_prime_analysis(limit: int = 1000):
    """Run comprehensive analysis across all dimensions and representations."""

    print("=" * 80)
    print("COMPREHENSIVE PRIME PATTERN ANALYSIS")
    print("=" * 80)

    primes = primes_up_to(limit)
    triplets = prime_triplets(limit)

    print(f"\nPrimes up to {limit}: {len(primes)}")
    print(f"Prime triplets (p, p+6, p+8): {len(triplets)}")

    # The special triplet
    him_triplet = (23, 29, 31)

    print("\n" + "=" * 80)
    print("ANALYSIS OF THE HIM TRIPLET (23, 29, 31)")
    print("=" * 80)

    # Representations
    print("\n1. NUMBER REPRESENTATIONS:")
    print("-" * 60)
    print(f"{'Prime':>8} {'Binary':>12} {'Ternary':>10} {'Roman':>8} {'Hex':>6} {'Octal':>6}")
    print("-" * 60)

    for p in him_triplet:
        print(f"{p:>8} {to_binary(p):>12} {to_ternary(p):>10} {to_roman(p):>8} {to_hexadecimal(p):>6} {to_octal(p):>6}")

    # Representation-based encodings
    print("\n2. REPRESENTATION-BASED LETTER ENCODINGS:")
    print("-" * 60)

    for method_name in REPRESENTATION_METHODS.keys():
        letters = ''.join(representation_to_letter(p, method_name) for p in him_triplet)
        values = [REPRESENTATION_METHODS[method_name](p) for p in him_triplet]
        triple = " ← TRIPLE!" if len(set(letters)) == 1 else ""
        print(f"  {method_name:>20}: {letters} (values: {values}){triple}")

    # Cross all triplets with representation methods
    print("\n3. TRIPLE-LETTER PATTERNS BY REPRESENTATION METHOD:")
    print("-" * 60)

    for method_name in REPRESENTATION_METHODS.keys():
        triple_count = 0
        triple_examples = []
        for t in triplets:
            letters = ''.join(representation_to_letter(p, method_name) for p in t)
            if len(set(letters)) == 1:
                triple_count += 1
                if len(triple_examples) < 3:
                    triple_examples.append((t, letters))

        print(f"  {method_name:>20}: {triple_count:>3} triple-letter triplets")
        for t, letters in triple_examples:
            print(f"                        {t} → {letters}")

    # Dimensional analysis
    print("\n4. DIMENSIONAL PROJECTIONS:")
    print("-" * 60)

    # 2D grid projection
    coords_2d = project_2d(primes, 10)
    print(f"  2D grid (10 wide): First 10 primes at positions {coords_2d[:10]}")

    # 3D helix
    coords_3d = project_3d_helix(primes[:10])
    print(f"  3D helix: Prime 2 at {tuple(round(c, 2) for c in coords_3d[0])}")

    # 4D projection of HIM triplet
    coords_4d = project_nd(list(him_triplet), 4)
    print(f"\n  HIM triplet in 4D space:")
    for i, (p, coord) in enumerate(zip(him_triplet, coords_4d)):
        print(f"    {p}: {coord}")

    # Check for special properties in 4D
    def dist_4d(c1, c2):
        return sum((a-b)**2 for a, b in zip(c1, c2)) ** 0.5

    d01 = dist_4d(coords_4d[0], coords_4d[1])
    d12 = dist_4d(coords_4d[1], coords_4d[2])
    d02 = dist_4d(coords_4d[0], coords_4d[2])
    print(f"\n  4D distances: d(23,29)={d01:.2f}, d(29,31)={d12:.2f}, d(23,31)={d02:.2f}")

    # Find patterns across all dimensions
    print("\n5. N-DIMENSIONAL PATTERN SEARCH:")
    print("-" * 60)

    for dim in [2, 3, 4, 5, 6]:
        coords = project_nd(list(him_triplet), dim)

        # Calculate pairwise distances
        distances = []
        for i in range(3):
            for j in range(i+1, 3):
                d = sum((a-b)**2 for a, b in zip(coords[i], coords[j])) ** 0.5
                distances.append(d)

        # Check for equilateral (all distances equal)
        is_equilateral = max(distances) - min(distances) < 0.001
        equi_marker = " ← EQUILATERAL!" if is_equilateral else ""

        # Check for isoceles (at least two equal)
        sorted_d = sorted(distances)
        is_isoceles = abs(sorted_d[0] - sorted_d[1]) < 0.001 or abs(sorted_d[1] - sorted_d[2]) < 0.001
        iso_marker = " (isoceles)" if is_isoceles and not is_equilateral else ""

        print(f"  {dim}D: distances {[round(d, 2) for d in distances]}{equi_marker}{iso_marker}")

    # Language + representation combination
    print("\n6. LANGUAGE × REPRESENTATION MATRIX (HIM triplet):")
    print("-" * 60)

    print(f"{'':>15}", end='')
    for method in ['letter_count', 'roman_length', 'binary_ones']:
        print(f"{method[:12]:>14}", end='')
    print()

    for lang in ['english', 'chinese', 'hebrew', 'japanese', 'irish']:
        print(f"{lang:>15}", end='')

        # Letter count encoding
        letters_lc = encode_sequence(list(him_triplet), lang, 'letter_count')
        triple_lc = "*" if len(set(letters_lc)) == 1 else ""
        print(f"{letters_lc + triple_lc:>14}", end='')

        # Roman length encoding (language-independent)
        letters_rl = ''.join(representation_to_letter(p, 'roman_length') for p in him_triplet)
        triple_rl = "*" if len(set(letters_rl)) == 1 else ""
        print(f"{letters_rl + triple_rl:>14}", end='')

        # Binary ones encoding (language-independent)
        letters_bo = ''.join(representation_to_letter(p, 'binary_ones') for p in him_triplet)
        triple_bo = "*" if len(set(letters_bo)) == 1 else ""
        print(f"{letters_bo + triple_bo:>14}", end='')

        print()

    print("\n  (* = triple-letter pattern)")

    return primes, triplets

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    primes, triplets = comprehensive_prime_analysis(1000)

    print("\n" + "=" * 80)
    print("SEARCHING FOR SPECIAL PATTERNS ACROSS ALL METHODS")
    print("=" * 80)

    # Find triplets with the most triple-letter encodings
    triplet_scores = {}

    for t in triplets:
        score = 0
        methods_matched = []

        # Check representation methods
        for method_name in REPRESENTATION_METHODS.keys():
            letters = ''.join(representation_to_letter(p, method_name) for p in t)
            if len(set(letters)) == 1:
                score += 1
                methods_matched.append(f"rep:{method_name}")

        # Check language encodings
        for lang in LANGUAGES.keys():
            letters = encode_sequence(list(t), lang, 'letter_count')
            if len(set(letters)) == 1:
                score += 1
                methods_matched.append(f"lang:{lang}")

        if score > 0:
            triplet_scores[t] = (score, methods_matched)

    # Sort by score
    sorted_triplets = sorted(triplet_scores.items(), key=lambda x: x[1][0], reverse=True)

    print(f"\nTop 10 triplets by number of triple-letter encodings:")
    print("-" * 60)

    for t, (score, methods) in sorted_triplets[:10]:
        indices = [prime_index(p) for p in t]
        print(f"\n  {t} (indices {indices}): {score} triple-letter encodings")
        print(f"    Methods: {', '.join(methods[:8])}")
        if len(methods) > 8:
            print(f"             ... and {len(methods) - 8} more")
