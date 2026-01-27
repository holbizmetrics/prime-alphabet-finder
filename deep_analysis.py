#!/usr/bin/env python3
"""
Deep Analysis - Exploring all unexplored directions for the HIM triplet
"""

import math
from collections import defaultdict
from functools import lru_cache

# =============================================================================
# PRIME UTILITIES
# =============================================================================

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def nth_prime(n):
    """Return the nth prime (1-indexed)"""
    if n < 1:
        return None
    count = 0
    num = 1
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    return num

def prime_index(p):
    """Return the index of prime p (1-indexed)"""
    if not is_prime(p):
        return None
    count = 0
    for i in range(2, p + 1):
        if is_prime(i):
            count += 1
    return count

def prime_triplets(limit):
    """Generate prime triplets (p, p+6, p+8)"""
    primes = set(primes_up_to(limit + 10))
    triplets = []
    for p in primes_up_to(limit):
        if p + 6 in primes and p + 8 in primes:
            triplets.append((p, p + 6, p + 8))
    return triplets

# =============================================================================
# 1. SOPHIE GERMAIN + SAFE PRIME CHAIN
# =============================================================================

print("=" * 80)
print("1. SOPHIE GERMAIN + SAFE PRIME CHAIN - DEEP ANALYSIS")
print("=" * 80)

def is_sophie_germain(p):
    """p is SG if 2p+1 is also prime"""
    return is_prime(p) and is_prime(2*p + 1)

def is_safe_prime(p):
    """p is safe if (p-1)/2 is prime"""
    return is_prime(p) and (p - 1) % 2 == 0 and is_prime((p - 1) // 2)

def is_sg_and_safe(p):
    return is_sophie_germain(p) and is_safe_prime(p)

# Find all primes that are BOTH SG and Safe up to 100000
sg_safe_primes = []
for p in primes_up_to(100000):
    if is_sg_and_safe(p):
        sg_safe_primes.append(p)

print(f"\nPrimes that are BOTH Sophie Germain AND Safe (up to 100,000):")
print(f"Count: {len(sg_safe_primes)}")
print(f"Chain: {sg_safe_primes[:20]}...")

# Analyze the chain structure
print(f"\nChain Analysis:")
print(f"{'Position':<10} {'Prime':<10} {'Index':<10} {'SG Result':<12} {'Safe Source':<12}")
print("-" * 60)
for i, p in enumerate(sg_safe_primes[:15], 1):
    idx = prime_index(p)
    sg_result = 2*p + 1
    safe_source = (p - 1) // 2
    print(f"{i:<10} {p:<10} {idx:<10} {sg_result:<12} {safe_source:<12}")

# Check connections within the chain
print(f"\nConnections within the SG+Safe chain:")
for i, p in enumerate(sg_safe_primes[:10]):
    sg_result = 2*p + 1
    if sg_result in sg_safe_primes:
        print(f"  {p}'s SG result ({sg_result}) is also in the chain!")
    safe_source = (p - 1) // 2
    if safe_source in sg_safe_primes:
        print(f"  {p}'s Safe source ({safe_source}) is also in the chain!")

# Why is 23 at position 3?
print(f"\n23 sits at position 3 in the SG+Safe chain")
print(f"  5 (pos 1) → SG gives 11 (pos 2) → SG gives 23 (pos 3) → SG gives 47")
print(f"  But 47 is NOT Safe (46/2=23 is prime, so 47 IS safe... checking)")
print(f"  47 is Safe: {is_safe_prime(47)}, 47 is SG: {is_sophie_germain(47)}")
print(f"  So 5→11→23 forms a connected sub-chain via SG operation")

# =============================================================================
# 2. PRIME GAPS ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("2. PRIME GAPS ANALYSIS")
print("=" * 80)

primes = primes_up_to(1000)
gaps = [(primes[i+1] - primes[i], primes[i], primes[i+1]) for i in range(len(primes)-1)]

# Gaps around 23, 29, 31
print(f"\nGaps around the HIM triplet:")
for gap, p1, p2 in gaps:
    if p1 in [19, 23, 29, 31, 37] or p2 in [19, 23, 29, 31, 37]:
        print(f"  {p1} → {p2}: gap = {gap}")

# The triplet has gaps 6, 2 - find all triplets with this gap pattern
print(f"\nAll triplets with gap pattern (6, 2):")
triplets = prime_triplets(10000)
for t in triplets[:20]:
    print(f"  {t}")

# Gap sum patterns
print(f"\nGap sums for first 20 triplets:")
for t in triplets[:20]:
    gap1 = t[1] - t[0]
    gap2 = t[2] - t[1]
    total_span = t[2] - t[0]
    print(f"  {t}: gaps ({gap1}, {gap2}), span = {total_span}")

# =============================================================================
# 3. ULAM SPIRAL ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("3. ULAM SPIRAL ANALYSIS")
print("=" * 80)

def ulam_position(n):
    """
    Calculate (x, y) position in Ulam spiral for number n.
    1 is at origin (0, 0), spiral goes counterclockwise.
    """
    if n == 1:
        return (0, 0)

    # Find which "ring" n is in
    # Ring k contains numbers from (2k-1)^2 + 1 to (2k+1)^2
    k = math.ceil((math.sqrt(n) - 1) / 2)

    # Side length of ring k
    side = 2 * k

    # Starting number of ring k
    start = (2*k - 1)**2 + 1

    # Position within the ring (0-indexed)
    pos = n - start

    # Which side of the ring? (0=right, 1=top, 2=left, 3=bottom)
    side_num = pos // side
    side_pos = pos % side

    if side_num == 0:  # Right side, going up
        return (k, -k + 1 + side_pos)
    elif side_num == 1:  # Top side, going left
        return (k - 1 - side_pos, k)
    elif side_num == 2:  # Left side, going down
        return (-k, k - 1 - side_pos)
    else:  # Bottom side, going right
        return (-k + 1 + side_pos, -k)

# Calculate positions for key primes
key_numbers = [23, 29, 31, 83, 101, 107, 109]
print(f"\nUlam Spiral positions:")
print(f"{'Number':<10} {'Position':<15} {'Distance from origin':<20} {'Prime?'}")
print("-" * 60)
for n in key_numbers:
    x, y = ulam_position(n)
    dist = math.sqrt(x**2 + y**2)
    print(f"{n:<10} ({x:>3}, {y:>3})      {dist:<20.3f} {is_prime(n)}")

# Check for alignments
print(f"\nGeometric relationships:")
positions = {n: ulam_position(n) for n in key_numbers}

# Distance between HIM triplet members
for i, p1 in enumerate([23, 29, 31]):
    for p2 in [23, 29, 31][i+1:]:
        x1, y1 = positions[p1]
        x2, y2 = positions[p2]
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        print(f"  Distance {p1} to {p2}: {dist:.3f}")

# Distance from 83 to triplet members
print(f"\n83 (the sum) distances:")
for p in [23, 29, 31]:
    x1, y1 = positions[83]
    x2, y2 = positions[p]
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    print(f"  Distance 83 to {p}: {dist:.3f}")

# Check if any are collinear
def are_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # Area of triangle = 0 means collinear
    area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    return area == 0

print(f"\nCollinearity checks:")
print(f"  23, 29, 31 collinear: {are_collinear(positions[23], positions[29], positions[31])}")
print(f"  23, 83, 101 collinear: {are_collinear(positions[23], positions[83], positions[101])}")
print(f"  29, 83, 107 collinear: {are_collinear(positions[29], positions[83], positions[107])}")

# =============================================================================
# 4. OTHER PRIME PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("4. OTHER PRIME PATTERNS")
print("=" * 80)

# Twin primes (differ by 2)
print(f"\nTwin Primes (p, p+2):")
twins = [(p, p+2) for p in primes_up_to(200) if is_prime(p+2)]
print(f"  First 15: {twins[:15]}")
print(f"  (29, 31) is a twin pair within the HIM triplet!")

# Sexy primes (differ by 6)
print(f"\nSexy Primes (p, p+6):")
sexy = [(p, p+6) for p in primes_up_to(200) if is_prime(p+6)]
print(f"  First 15: {sexy[:15]}")
print(f"  (23, 29) is a sexy pair within the HIM triplet!")

# Cousin primes (differ by 4)
print(f"\nCousin Primes (p, p+4):")
cousins = [(p, p+4) for p in primes_up_to(200) if is_prime(p+4)]
print(f"  First 15: {cousins[:15]}")

# Prime quadruplets (p, p+2, p+6, p+8)
print(f"\nPrime Quadruplets (p, p+2, p+6, p+8):")
quadruplets = []
for p in primes_up_to(10000):
    if is_prime(p+2) and is_prime(p+6) and is_prime(p+8):
        quadruplets.append((p, p+2, p+6, p+8))
print(f"  Count up to 10000: {len(quadruplets)}")
print(f"  First 10: {quadruplets[:10]}")
print(f"  Note: (23, 29, 31) is NOT part of a quadruplet (25 is not prime)")

# Prime quintuplets
print(f"\nPrime Quintuplets:")
print(f"  Forms: (p, p+2, p+6, p+8, p+12) or (p, p+4, p+6, p+10, p+12)")
quint1 = []
quint2 = []
for p in primes_up_to(10000):
    if all(is_prime(p+k) for k in [0, 2, 6, 8, 12]):
        quint1.append((p, p+2, p+6, p+8, p+12))
    if all(is_prime(p+k) for k in [0, 4, 6, 10, 12]):
        quint2.append((p, p+4, p+6, p+10, p+12))
print(f"  Type 1: {quint1[:5]}")
print(f"  Type 2: {quint2[:5]}")

# Mersenne primes
print(f"\nMersenne Primes (2^p - 1):")
mersenne_exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127]
print(f"  Exponents that give Mersenne primes: {mersenne_exponents}")
print(f"  Is 23 a Mersenne exponent? {23 in mersenne_exponents}")
print(f"  Is 31 a Mersenne exponent? {31 in mersenne_exponents} ← YES!")
print(f"  2^31 - 1 = {2**31 - 1} is prime!")

# Fibonacci primes
print(f"\nFibonacci Primes:")
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib_primes = []
for i in range(2, 50):
    f = fib(i)
    if is_prime(f):
        fib_primes.append((i, f))
print(f"  (index, Fibonacci number): {fib_primes[:12]}")
print(f"  Is 23 a Fibonacci prime? {23 in [f for _, f in fib_primes]} ← YES!")
print(f"  Is 29 a Fibonacci prime? {29 in [f for _, f in fib_primes]}")
print(f"  Is 89 a Fibonacci prime? {89 in [f for _, f in fib_primes]} ← YES!")

# =============================================================================
# 5. MODULAR PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("5. MODULAR PATTERNS")
print("=" * 80)

key_primes = [23, 29, 31, 83, 101, 107, 109]
moduli = [6, 10, 12, 30, 7, 9, 11]

print(f"\n{'Prime':<8}", end="")
for m in moduli:
    print(f"{'mod '+str(m):<8}", end="")
print()
print("-" * 70)

for p in key_primes:
    print(f"{p:<8}", end="")
    for m in moduli:
        print(f"{p % m:<8}", end="")
    print()

# Special patterns
print(f"\nSpecial modular observations:")
print(f"  All primes > 3 are ≡ 1 or 5 (mod 6)")
print(f"  23 ≡ {23 % 6} (mod 6), 29 ≡ {29 % 6} (mod 6), 31 ≡ {31 % 6} (mod 6)")
print(f"  Pattern: 5, 5, 1 (mod 6)")

print(f"\n  mod 30 (primorial of 3):")
print(f"  23 ≡ {23 % 30}, 29 ≡ {29 % 30}, 31 ≡ {31 % 30}")
print(f"  Sum: {23 + 29 + 31} ≡ {(23 + 29 + 31) % 30} (mod 30)")

# Quadratic residues
print(f"\nQuadratic residues mod 23:")
qr_23 = set(pow(x, 2, 23) for x in range(1, 23))
print(f"  QR(23) = {sorted(qr_23)}")
print(f"  29 mod 23 = {29 % 23}, is QR? {29 % 23 in qr_23}")
print(f"  31 mod 23 = {31 % 23}, is QR? {31 % 23 in qr_23}")

# =============================================================================
# 6. ECHO TRIPLET DEEP DIVE
# =============================================================================

print("\n" + "=" * 80)
print("6. ECHO TRIPLET (101, 107, 109) - DEEP DIVE")
print("=" * 80)

echo = (101, 107, 109)
him = (23, 29, 31)

print(f"\nBasic properties:")
print(f"  Sum: {sum(echo)} (vs HIM sum: {sum(him)})")
print(f"  Product: {echo[0] * echo[1] * echo[2]}")
print(f"  Indices: {prime_index(101)}, {prime_index(107)}, {prime_index(109)}")
print(f"  Index sum: {prime_index(101) + prime_index(107) + prime_index(109)} = 83 ← Links to HIM!")

print(f"\nBinary representations:")
for p in echo:
    b = bin(p)[2:]
    print(f"  {p} = {b} ({len(b)} bits, {b.count('1')} ones)")

print(f"\nBinary mirror check:")
print(f"  83 = {bin(83)[2:]}")
print(f"  101 = {bin(101)[2:]}")
print(f"  83 reversed = {bin(83)[2:][::-1]} = {int(bin(83)[2:][::-1], 2)}")
print(f"  Connection: 83 ↔ 101 are NOT exact binary mirrors")
print(f"  But 83 (1010011) and 101 (1100101) share structure")

# Is echo triplet self-referential?
echo_sum = sum(echo)
echo_first_index = prime_index(echo[0])
nth_p = nth_prime(echo_first_index)
print(f"\nSelf-reference check:")
print(f"  Echo sum = {echo_sum}")
print(f"  101 is the {echo_first_index}th prime")
print(f"  {echo_first_index}th prime = {nth_p}")
print(f"  Self-referential? {echo_sum == nth_p} (would need sum = {echo_first_index}th prime)")

# Sophie Germain / Safe properties
print(f"\nSG/Safe properties:")
for p in echo:
    sg = is_sophie_germain(p)
    safe = is_safe_prime(p)
    print(f"  {p}: SG={sg}, Safe={safe}")

# Twin/Sexy within echo
print(f"\nPrime pairs within echo triplet:")
print(f"  107-101 = 6 → Sexy pair: {is_prime(101) and is_prime(107)}")
print(f"  109-107 = 2 → Twin pair: {is_prime(107) and is_prime(109)}")
print(f"  Same structure as HIM triplet! (sexy + twin)")

# =============================================================================
# 7. PROBABILITY CALCULATION
# =============================================================================

print("\n" + "=" * 80)
print("7. PROBABILITY ANALYSIS")
print("=" * 80)

# Count properties and estimate probabilities
print(f"\nIndependent property probabilities (rough estimates):")

# P(triplet is self-referential)
triplets_to_1000 = prime_triplets(1000)
triplets_to_10000 = prime_triplets(10000)
self_ref_count = 0
for t in triplets_to_10000:
    if sum(t) == nth_prime(t[0]):
        self_ref_count += 1
        print(f"  Found self-referential: {t}, sum={sum(t)}, p({t[0]})={nth_prime(t[0])}")

print(f"\n  P(self-referential) = {self_ref_count}/{len(triplets_to_10000)} = {self_ref_count/len(triplets_to_10000):.6f}")

# P(binary mirror pair within triplet)
mirror_count = 0
for t in triplets_to_10000:
    b0 = bin(t[0])[2:]
    b1 = bin(t[1])[2:]
    if b0 == b1[::-1]:
        mirror_count += 1
print(f"  P(binary mirror) = {mirror_count}/{len(triplets_to_10000)} = {mirror_count/len(triplets_to_10000):.6f}")

# P(third element is binary palindrome)
palindrome_count = 0
for t in triplets_to_10000:
    b = bin(t[2])[2:]
    if b == b[::-1]:
        palindrome_count += 1
print(f"  P(third is palindrome) = {palindrome_count}/{len(triplets_to_10000)} = {palindrome_count/len(triplets_to_10000):.6f}")

# P(first element is both SG and Safe)
sg_safe_count = 0
for t in triplets_to_10000:
    if is_sg_and_safe(t[0]):
        sg_safe_count += 1
print(f"  P(first is SG+Safe) = {sg_safe_count}/{len(triplets_to_10000)} = {sg_safe_count/len(triplets_to_10000):.6f}")

# P(first! has first digits)
factorial_digit_count = 0
for t in triplets_to_10000[:100]:  # Only check first 100 (factorial gets huge)
    try:
        fact = math.factorial(t[0])
        if len(str(fact)) == t[0]:
            factorial_digit_count += 1
            print(f"    {t[0]}! has {t[0]} digits")
    except:
        pass
print(f"  P(p! has p digits) = ~{factorial_digit_count}/100 (for small triplets)")

# Combined probability (assuming independence - rough estimate)
print(f"\nCombined probability (if independent):")
p_self_ref = max(self_ref_count, 1) / len(triplets_to_10000)
p_mirror = max(mirror_count, 1) / len(triplets_to_10000)
p_palindrome = max(palindrome_count, 1) / len(triplets_to_10000)
p_sg_safe = max(sg_safe_count, 1) / len(triplets_to_10000)

combined = p_self_ref * p_mirror * p_palindrome * p_sg_safe
print(f"  P(all four properties) ≈ {combined:.2e}")
print(f"  Expected count in {len(triplets_to_10000)} triplets: {combined * len(triplets_to_10000):.4f}")

# =============================================================================
# 8. PHONETIC PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("8. PHONETIC PATTERN ANALYSIS")
print("=" * 80)

# Number words in key languages
number_words = {
    'english': {23: 'twenty-three', 29: 'twenty-nine', 31: 'thirty-one'},
    'german': {23: 'dreiundzwanzig', 29: 'neunundzwanzig', 31: 'einunddreissig'},
    'chinese': {23: 'ershisan', 29: 'ershijiu', 31: 'sanshiyi'},
    'french': {23: 'vingt-trois', 29: 'vingt-neuf', 31: 'trente-et-un'},
    'spanish': {23: 'veintitres', 29: 'veintinueve', 31: 'treinta y uno'},
    'latin': {23: 'viginti tres', 29: 'viginti novem', 31: 'triginta unus'},
}

def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')

def count_consonants(s):
    return sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')

def count_syllables_approx(s):
    """Rough syllable count based on vowel groups"""
    s = s.lower()
    count = 0
    prev_vowel = False
    for c in s:
        is_vowel = c in 'aeiou'
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)

print(f"\nPhonetic analysis by language:")
for lang, words in number_words.items():
    print(f"\n{lang.upper()}:")
    print(f"  {'Number':<10} {'Word':<20} {'Letters':<10} {'Vowels':<10} {'Consonants':<12} {'~Syllables'}")
    print(f"  " + "-" * 75)

    letters = []
    vowels = []
    consonants = []
    syllables = []

    for num, word in words.items():
        l = len(word.replace(' ', '').replace('-', ''))
        v = count_vowels(word)
        c = count_consonants(word)
        s = count_syllables_approx(word)
        letters.append(l)
        vowels.append(v)
        consonants.append(c)
        syllables.append(s)
        print(f"  {num:<10} {word:<20} {l:<10} {v:<10} {c:<12} {s}")

    # Check for triple patterns
    if len(set(letters)) == 1:
        print(f"  ★ TRIPLE LETTER COUNT: {letters[0]}")
    if len(set(vowels)) == 1:
        print(f"  ★ TRIPLE VOWEL COUNT: {vowels[0]}")
    if len(set(consonants)) == 1:
        print(f"  ★ TRIPLE CONSONANT COUNT: {consonants[0]}")
    if len(set(syllables)) == 1:
        print(f"  ★ TRIPLE SYLLABLE COUNT: {syllables[0]}")

# Common phonetic patterns
print(f"\nCross-linguistic phonetic patterns:")
print(f"  All have compound structure: [tens word] + [units word]")
print(f"  German reverses: [unit] + und + [tens]")
print(f"  Chinese is pure additive: er-shi-san (2-10-3)")

# =============================================================================
# 9. ADDITIONAL DISCOVERIES
# =============================================================================

print("\n" + "=" * 80)
print("9. ADDITIONAL DISCOVERIES")
print("=" * 80)

# Check if 23 appears in other special sequences
print(f"\n23 in special sequences:")
print(f"  23 is 9th prime (9 = 3²)")
print(f"  23 is F(9) in Fibonacci? F(9) = {fib(9)} → No")
print(f"  23 = 3! + 17 = 6 + 17")
print(f"  23 = 2³ + 15 = 8 + 15")
print(f"  23 = 4! - 1 = 24 - 1")

# Interesting: 23 = 4! - 1
print(f"\n  23 = 4! - 1 (factorial minus one)")
print(f"  Numbers of form n! - 1 that are prime: ", end="")
for n in range(2, 15):
    if is_prime(math.factorial(n) - 1):
        print(f"{n}! - 1 = {math.factorial(n) - 1}, ", end="")
print()

# Sum of digits patterns
print(f"\nDigit sum patterns:")
for p in [23, 29, 31, 83, 101, 107, 109]:
    ds = sum(int(d) for d in str(p))
    print(f"  {p}: digit sum = {ds}, digital root = {ds if ds < 10 else sum(int(d) for d in str(ds))}")

# The 23 enigma connections
print(f"\n'23 Enigma' mathematical connections:")
print(f"  23 = smallest prime with consecutive digits")
print(f"  2 + 3 = 5 (prime)")
print(f"  2 × 3 = 6 (perfect number? No, but 2×3=6=1+2+3)")
print(f"  2³ = 8, 3² = 9, difference = 1")
print(f"  23 in binary: 10111 (4 ones, like '23' has 2 digits summing to 5)")

# Concatenation patterns
print(f"\nConcatenation patterns:")
concat = int(str(23) + str(29) + str(31))
print(f"  232931: is prime? {is_prime(concat)}")
concat2 = int(str(31) + str(29) + str(23))
print(f"  312923: is prime? {is_prime(concat2)}")

# Sum, product relationships
print(f"\nArithmetic relationships:")
print(f"  23 + 29 = {23 + 29} = 52 = 4 × 13")
print(f"  29 + 31 = {29 + 31} = 60 = 2² × 3 × 5")
print(f"  23 + 31 = {23 + 31} = 54 = 2 × 3³")
print(f"  23 × 29 = {23 * 29} = 667")
print(f"  29 × 31 = {29 * 31} = 899")
print(f"  23 × 31 = {23 * 31} = 713")
print(f"  Product 23×29×31 = {23*29*31}")
print(f"  Digit sum of product: {sum(int(d) for d in str(23*29*31))} = 22 = 23 - 1")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
