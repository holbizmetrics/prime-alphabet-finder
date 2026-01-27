#!/usr/bin/env python3
"""
Systematic Pattern Search
Searches for meaningful patterns across all language/encoding combinations.
"""

from prime_encoder import *
import json
from collections import defaultdict

# =============================================================================
# COMMON ENGLISH WORDS (3-4 letters) FOR PATTERN MATCHING
# =============================================================================

COMMON_WORDS = [
    # 3-letter words
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS",
    "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID",
    "GOD", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "SUN",
    "SON", "GOT", "MAY", "OWN", "SAW", "MEN", "RUN", "END", "RED", "TEN",
    "YES", "YET", "BIG", "ACE", "AGE", "AID", "AIM", "AIR", "ADD", "ARM",
    "ART", "ASK", "BAD", "BAG", "BAR", "BAT", "BED", "BET", "BIT", "BOX",
    "BUS", "BUY", "CAR", "CAT", "COP", "CRY", "CUP", "CUT", "DIE", "DOG",
    "DOT", "DRY", "EAR", "EAT", "EGG", "EYE", "FAR", "FAT", "FEW", "FIT",
    "FLY", "FOX", "FUN", "GAS", "GUN", "GUY", "HAT", "HIT", "HOT", "ICE",
    "JOB", "JOY", "KEY", "KID", "LAW", "LAY", "LEG", "LIE", "LIP", "LOT",
    "LOW", "MAP", "MAT", "MIX", "MUD", "NET", "NOR", "NUT", "ODD", "OIL",
    "PAY", "PEN", "PET", "PIE", "PIG", "PIN", "PIT", "POT", "RAT", "RAW",
    "RIB", "RID", "ROD", "ROW", "RUB", "SAD", "SAT", "SET", "SIT", "SIX",
    "SKY", "TAX", "TEA", "TIE", "TIP", "TOP", "TOY", "TRY", "TWO", "VAN",
    "WAR", "WEB", "WET", "WIN", "WON", "ZOO",
    # 4-letter words
    "THAT", "WITH", "HAVE", "THIS", "WILL", "YOUR", "FROM", "THEY", "BEEN",
    "HAVE", "MANY", "SOME", "THEM", "THAN", "WORD", "SAID", "EACH", "TIME",
    "VERY", "WHEN", "COME", "MADE", "FIND", "LONG", "LOOK", "MORE", "OVER",
    "SUCH", "TAKE", "YEAR", "BACK", "GOOD", "GIVE", "MOST", "ONLY", "THEN",
    "NAME", "KNOW", "LIFE", "LIVE", "LOVE", "MAKE", "MUCH", "MUST", "NEED",
    "PART", "SAME", "SEEM", "SHOW", "SIDE", "TELL", "WANT", "WELL", "WORK",
    "BEST", "BOOK", "CALL", "CASE", "CITY", "DOES", "DONE", "DOOR", "DOWN",
    "EVEN", "EVER", "EYES", "FACE", "FACT", "FEEL", "FELT", "FIVE", "FOUR",
    "FREE", "GIRL", "GOES", "GONE", "HAND", "HEAD", "HELP", "HERE", "HIGH",
    "HOME", "HOPE", "IDEA", "INTO", "JUST", "KEEP", "KIND", "LAST", "LEFT",
    "LESS", "LINE", "LOST", "MIND", "MISS", "MOVE", "NEXT", "ONCE", "OPEN",
    "PLAY", "REAL", "REST", "ROOM", "SEEN", "SELF", "SORT", "STOP", "SURE",
    "TALK", "TERM", "THEE", "THOU", "TRUE", "TURN", "UPON", "WAIT", "WALK",
    "WEEK", "WENT", "WHAT", "WHOM", "WIFE", "WISH", "ZERO", "FIRE", "HOLY",
    "LORD", "KING", "SOUL", "PRAY", "AMEN", "WORD", "ADAM", "EDEN",
    # Mystical/religious words
    "GOD", "SON", "SIN", "ARK", "EVE", "OHM", "ZEN", "TAO", "CHI", "YIN",
    "YANG", "KALI", "SHIV", "RAMA", "DEVA", "GURU", "YOGA", "MANA", "VEDA",
    "AURA", "SOUL", "MIND", "BODY", "LIFE", "DEATH",
    # Names
    "ADAM", "ABEL", "CAIN", "NOAH", "MARY", "JOHN", "PAUL", "LUKE", "MARK",
    "SETH", "ISAAC", "JACOB", "MOSES", "DAVID", "PETER",
]

# Hebrew-related patterns
HEBREW_PATTERNS = ["HIM", "HER", "GOD", "ONE", "YAH", "ELI", "YOD", "HEH", "VAV"]

# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================

def find_all_words_in_sequence(letters: str, min_length: int = 3) -> List[Tuple[int, str]]:
    """Find all dictionary words in the letter sequence."""
    return find_words(letters, COMMON_WORDS, min_length)


def find_repeated_patterns(letters: str, min_length: int = 2, min_count: int = 2) -> Dict[str, List[int]]:
    """Find patterns that repeat in the sequence."""
    patterns = defaultdict(list)

    for length in range(min_length, min(6, len(letters) // 2) + 1):
        for i in range(len(letters) - length + 1):
            pattern = letters[i:i+length]
            patterns[pattern].append(i)

    # Filter to only patterns that appear multiple times
    return {p: positions for p, positions in patterns.items()
            if len(positions) >= min_count}


def calculate_statistics(encoded_values: List[int]) -> Dict:
    """Calculate various statistics on the encoded values."""
    if not encoded_values:
        return {}

    n = len(encoded_values)
    total = sum(encoded_values)
    mean = total / n

    # Check if total has interesting factorization
    factors = factorize(total)

    # Check if total is prime
    total_is_prime = is_prime(total)

    # Check various mod patterns
    mod_patterns = {
        f"mod_{m}": [v % m for v in encoded_values]
        for m in [2, 3, 5, 7, 10, 26]
    }

    # Count how many encoded values are themselves prime
    prime_count = sum(1 for v in encoded_values if is_prime(v))

    return {
        'total': total,
        'mean': mean,
        'factors': factors,
        'total_is_prime': total_is_prime,
        'prime_count': prime_count,
        'prime_ratio': prime_count / n if n > 0 else 0,
        'mod_patterns': mod_patterns
    }


def factorize(n: int) -> List[Tuple[int, int]]:
    """Return prime factorization as list of (prime, exponent) tuples."""
    if n < 2:
        return []

    factors = []
    d = 2
    while d * d <= n:
        exp = 0
        while n % d == 0:
            exp += 1
            n //= d
        if exp > 0:
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


def xor_sequences(seq1: List[int], seq2: List[int], alphabet_size: int = 26) -> List[int]:
    """XOR two sequences (treating as positions in alphabet)."""
    min_len = min(len(seq1), len(seq2))
    return [((a - 1) ^ (b - 1)) % alphabet_size + 1 for a, b in zip(seq1[:min_len], seq2[:min_len])]


def difference_sequences(seq1: List[int], seq2: List[int], alphabet_size: int = 26) -> List[int]:
    """Compute difference between two sequences mod alphabet size."""
    min_len = min(len(seq1), len(seq2))
    return [((a - b) % alphabet_size) + 1 for a, b in zip(seq1[:min_len], seq2[:min_len])]


# =============================================================================
# COMPREHENSIVE SEARCH
# =============================================================================

def comprehensive_search(limit: int = 500):
    """Run comprehensive pattern search."""

    print("=" * 70)
    print("COMPREHENSIVE PRIME-ALPHABET PATTERN SEARCH")
    print(f"Analyzing primes up to {limit}")
    print("=" * 70)

    primes = primes_up_to(limit)
    triplets = prime_triplets(limit)

    print(f"\nFound {len(primes)} primes")
    print(f"Found {len(triplets)} prime triplets (p, p+6, p+8)")

    languages = list(LANGUAGES.keys())
    methods = ['letter_count', 'digit_sum', 'direct_mod', 'prime_index']

    results = {
        'words_found': [],
        'triple_letter_triplets': [],
        'interesting_totals': [],
        'high_correlations': [],
        'repeated_patterns': []
    }

    # =============================================================================
    # 1. SEARCH FOR WORDS
    # =============================================================================
    print("\n" + "=" * 70)
    print("1. SEARCHING FOR WORDS IN ENCODINGS")
    print("=" * 70)

    for lang in languages:
        for method in methods:
            enc = encode_sequence(primes, lang, method)
            words = find_all_words_in_sequence(enc.letters)
            if words:
                for pos, word in words:
                    entry = {
                        'language': lang,
                        'method': method,
                        'word': word,
                        'position': pos,
                        'context': enc.letters[max(0, pos-2):pos+len(word)+2]
                    }
                    results['words_found'].append(entry)
                    print(f"  [{lang:>10}/{method:>12}] Found '{word}' at position {pos}: ...{entry['context']}...")

    # =============================================================================
    # 2. TRIPLE-LETTER TRIPLETS
    # =============================================================================
    print("\n" + "=" * 70)
    print("2. TRIPLE-LETTER TRIPLETS")
    print("=" * 70)

    for lang in languages:
        tlt = find_triple_letter_triplets(triplets, lang, 'letter_count')
        if tlt:
            print(f"\n  {lang.upper()}:")
            for triplet, letters in tlt:
                indices = [prime_index(p) for p in triplet]
                index_sum = sum(indices)
                entry = {
                    'language': lang,
                    'triplet': triplet,
                    'letters': letters,
                    'indices': indices,
                    'index_sum': index_sum
                }
                results['triple_letter_triplets'].append(entry)
                print(f"    {triplet} → {letters}  (indices {indices}, sum={index_sum})")

    # =============================================================================
    # 3. INTERESTING TOTALS
    # =============================================================================
    print("\n" + "=" * 70)
    print("3. INTERESTING TOTALS (letter count encoding)")
    print("=" * 70)

    for lang in languages:
        enc = encode_sequence(primes, lang, 'letter_count')
        stats = calculate_statistics(enc.encoded_values)

        factors_str = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in stats['factors'])
        print(f"  {lang:>10}: total = {stats['total']:>4} = {factors_str}")

        # Check if total contains any of our primes as factors
        prime_factors = [p for p, _ in stats['factors']]
        matching_primes = [p for p in prime_factors if p in primes]
        if matching_primes:
            results['interesting_totals'].append({
                'language': lang,
                'total': stats['total'],
                'factors': stats['factors'],
                'matching_primes': matching_primes
            })
            print(f"           → Contains primes from our set: {matching_primes}")

    # =============================================================================
    # 4. CROSS-LANGUAGE CORRELATIONS
    # =============================================================================
    print("\n" + "=" * 70)
    print("4. CROSS-LANGUAGE CORRELATIONS (letter count)")
    print("=" * 70)

    correlations = []
    for l1, l2 in itertools.combinations(languages, 2):
        corr = cross_language_correlation(primes, l1, l2, 'letter_count')
        correlations.append((l1, l2, corr))

    # Sort by correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\n  Highest correlations:")
    for l1, l2, corr in correlations[:10]:
        print(f"    {l1:>10} <-> {l2:<10}: {corr:+.4f}")
        if abs(corr) > 0.7:
            results['high_correlations'].append({
                'lang1': l1,
                'lang2': l2,
                'correlation': corr
            })

    print("\n  Lowest correlations:")
    for l1, l2, corr in correlations[-5:]:
        print(f"    {l1:>10} <-> {l2:<10}: {corr:+.4f}")

    # =============================================================================
    # 5. CROSS-LANGUAGE DIFFERENCES
    # =============================================================================
    print("\n" + "=" * 70)
    print("5. CROSS-LANGUAGE DIFFERENCE PATTERNS")
    print("=" * 70)

    # Check XOR and difference patterns between select language pairs
    pairs_to_check = [
        ('english', 'hebrew'),
        ('english', 'greek'),
        ('hebrew', 'greek'),
        ('english', 'latin'),
        ('latin', 'greek'),
    ]

    for l1, l2 in pairs_to_check:
        enc1 = encode_sequence(primes, l1, 'letter_count')
        enc2 = encode_sequence(primes, l2, 'letter_count')

        diff = difference_sequences(enc1.encoded_values, enc2.encoded_values)
        diff_letters = ''.join(num_to_letter(v) for v in diff[:30])

        xored = xor_sequences(enc1.encoded_values, enc2.encoded_values)
        xor_letters = ''.join(num_to_letter(v) for v in xored[:30])

        print(f"\n  {l1} - {l2}:")
        print(f"    Diff: {diff_letters}...")
        print(f"    XOR:  {xor_letters}...")

        # Check for words in difference
        diff_full = ''.join(num_to_letter(v) for v in diff)
        words_in_diff = find_all_words_in_sequence(diff_full)
        if words_in_diff:
            print(f"    Words in diff: {[w for _, w in words_in_diff[:5]]}")

    # =============================================================================
    # 6. SPECIAL TRIPLET ANALYSIS
    # =============================================================================
    print("\n" + "=" * 70)
    print("6. SPECIAL TRIPLET ANALYSIS")
    print("=" * 70)

    special_triplets = [
        (23, 29, 31),    # "HIM" triplet
        (53, 59, 61),    # HIM + 30
        (101, 107, 109), # Index sum = 83
        (11, 17, 19),    # Small triplet
    ]

    for triplet in special_triplets:
        if all(is_prime(p) for p in triplet):
            print_triplet_analysis(triplet)

    # =============================================================================
    # 7. SUMMARY
    # =============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Words found: {len(results['words_found'])}")
    if results['words_found']:
        unique_words = set(e['word'] for e in results['words_found'])
        print(f"    Unique words: {sorted(unique_words)}")

    print(f"\n  Triple-letter triplets: {len(results['triple_letter_triplets'])}")

    print(f"\n  Interesting totals: {len(results['interesting_totals'])}")

    print(f"\n  High correlations (|r| > 0.7): {len(results['high_correlations'])}")

    return results


# =============================================================================
# SPECIFIC PATTERN INVESTIGATIONS
# =============================================================================

def investigate_him_triplet():
    """Deep investigation of the (23, 29, 31) triplet and its connections."""

    print("\n" + "=" * 70)
    print("DEEP INVESTIGATION: THE HIM TRIPLET (23, 29, 31)")
    print("=" * 70)

    triplet = (23, 29, 31)

    # Basic properties
    print("\n1. BASIC PROPERTIES")
    print("-" * 40)
    print(f"   Triplet: {triplet}")
    print(f"   Sum: {sum(triplet)} = 83")
    print(f"   83 is the {prime_index(83)}th prime")
    print(f"   23rd prime is: {nth_prime(23)}")
    print(f"   SELF-REFERENCE: 23 → sum 83 → 23rd prime → 83")

    # Special prime properties
    print("\n2. SPECIAL PRIME PROPERTIES")
    print("-" * 40)

    # Sophie Germain primes: p where 2p+1 is also prime
    for p in triplet:
        is_sg = is_prime(2 * p + 1)
        print(f"   {p}: Sophie Germain prime? {is_sg} (2×{p}+1 = {2*p+1})")

    # Twin primes
    print(f"   29 and 31 are twin primes: {is_prime(29) and is_prime(31) and 31-29==2}")

    # Mersenne prime
    print(f"   31 = 2^5 - 1, Mersenne prime: {31 == 2**5 - 1}")

    # Multi-language encoding
    print("\n3. ENCODING ACROSS LANGUAGES")
    print("-" * 40)

    results = analyze_triplet_across_languages(triplet)
    for lang, letters in sorted(results.items()):
        word = LANGUAGES[lang](triplet[0])
        counts = [letter_count(LANGUAGES[lang](p)) for p in triplet]
        print(f"   {lang:>10}: {letters}  (counts: {counts})")

    # The +30 transformation
    print("\n4. THE +30 TRANSFORMATION")
    print("-" * 40)

    for k in range(5):
        new_triplet = tuple(p + 30*k for p in triplet)
        all_prime = all(is_prime(p) for p in new_triplet)
        print(f"   k={k}: {new_triplet} - all prime? {all_prime}")

        if all_prime:
            results = analyze_triplet_across_languages(new_triplet)
            print(f"        Encodings: ", end="")
            for lang in ['english', 'hebrew', 'greek']:
                print(f"{lang[:3]}={results[lang]} ", end="")
            print()

    # Index sum = 83 triplet
    print("\n5. THE INDEX SUM = 83 CONNECTION")
    print("-" * 40)

    # Find triplets where index sum = 83
    triplets = prime_triplets(1000)
    for t in triplets:
        indices = [prime_index(p) for p in t]
        if sum(indices) == 83:
            print(f"   Triplet {t}: indices {indices}, sum = 83")
            results = analyze_triplet_across_languages(t)
            for lang in ['english', 'hebrew', 'greek', 'latin']:
                print(f"      {lang}: {results[lang]}")


def investigate_self_references():
    """Find self-referential structures in prime encodings."""

    print("\n" + "=" * 70)
    print("INVESTIGATING SELF-REFERENTIAL STRUCTURES")
    print("=" * 70)

    # Pattern: prime p → encoding → points back to p somehow

    primes = primes_up_to(200)

    print("\n1. ENCODING SUM EQUALS PRIME INDEX")
    print("-" * 40)

    for lang in ['english', 'hebrew', 'greek']:
        print(f"\n   {lang.upper()}:")
        for p in primes:
            enc = encode_sequence([p], lang, 'letter_count')
            idx = prime_index(p)
            if enc.encoded_values[0] == idx:
                word = LANGUAGES[lang](p)
                print(f"      {p} (index {idx}): letter_count({word}) = {letter_count(word)} = index!")


if __name__ == "__main__":
    # Run comprehensive search
    results = comprehensive_search(500)

    # Deep investigations
    investigate_him_triplet()
    investigate_self_references()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
