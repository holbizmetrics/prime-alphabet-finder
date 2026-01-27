#!/usr/bin/env python3
"""
Complete Matrix Analysis - ALL combinations of languages × encodings × alphabets
"""

import sys
from prime_encoder_extended import *
from languages_extended import EXTENDED_LANGUAGES
from collections import defaultdict

# Merge all languages
ALL_LANGUAGES = {**LANGUAGES, **EXTENDED_LANGUAGES}

print("=" * 80)
print("  COMPLETE MATRIX ANALYSIS")
print(f"  {len(ALL_LANGUAGES)} Languages × {len(ENCODING_METHODS)} Encodings × {len(ALPHABETS)} Alphabets")
print("=" * 80)

# The HIM triplet
him = (23, 29, 31)
triplets = prime_triplets(1000)

print(f"\nTotal languages: {len(ALL_LANGUAGES)}")
print(f"Total encoding methods: {len(ENCODING_METHODS)}")
print(f"Total alphabets: {len(ALPHABETS)}")
print(f"Total combinations: {len(ALL_LANGUAGES) * len(ENCODING_METHODS) * len(ALPHABETS)}")

# =============================================================================
# 1. TEST ALL LANGUAGES ON HIM TRIPLET
# =============================================================================

print("\n" + "=" * 80)
print("1. ALL LANGUAGES - LETTER COUNT ENCODING ON (23, 29, 31)")
print("=" * 80)

triple_letter_languages = []
all_encodings = []

for lang in sorted(ALL_LANGUAGES.keys()):
    try:
        func = ALL_LANGUAGES[lang]
        words = [func(p) for p in him]
        counts = [letter_count(w) for w in words]
        letters = ''.join(chr(ord('A') + (c - 1) % 26) for c in counts)
        is_triple = len(set(letters)) == 1

        all_encodings.append((lang, letters, counts, is_triple))

        if is_triple:
            triple_letter_languages.append((lang, letters, counts))
    except Exception as e:
        pass

print(f"\nLanguages with TRIPLE-LETTER encoding for (23,29,31): {len(triple_letter_languages)}")
print("-" * 60)
for lang, letters, counts in sorted(triple_letter_languages):
    print(f"  {lang:>20}: {letters} (counts: {counts})")

print(f"\nAll other encodings:")
print("-" * 60)
for lang, letters, counts, is_triple in sorted(all_encodings):
    if not is_triple:
        print(f"  {lang:>20}: {letters} (counts: {counts})")

# =============================================================================
# 2. FIND ALL TRIPLE-LETTER TRIPLETS ACROSS ALL LANGUAGES
# =============================================================================

print("\n" + "=" * 80)
print("2. ALL TRIPLE-LETTER TRIPLETS BY LANGUAGE")
print("=" * 80)

language_triples = defaultdict(list)

for lang in ALL_LANGUAGES.keys():
    func = ALL_LANGUAGES[lang]
    for t in triplets:
        try:
            words = [func(p) for p in t]
            counts = [letter_count(w) for w in words]
            letters = ''.join(chr(ord('A') + (c - 1) % 26) for c in counts)
            if len(set(letters)) == 1:
                language_triples[lang].append((t, letters))
        except:
            pass

# Rank by count
ranked_langs = sorted(language_triples.items(), key=lambda x: len(x[1]), reverse=True)

print(f"\nRanking by number of triple-letter triplets:")
print("-" * 60)
for lang, triples_list in ranked_langs[:30]:
    examples = ', '.join(f"{t}→{l}" for t, l in triples_list[:2])
    print(f"  {lang:>20}: {len(triples_list):>3} triplets  (e.g., {examples})")

# =============================================================================
# 3. MOST UNIVERSAL TRIPLETS
# =============================================================================

print("\n" + "=" * 80)
print("3. TRIPLETS WITH TRIPLE-LETTERS IN MOST LANGUAGES")
print("=" * 80)

triplet_universality = defaultdict(list)

for lang, triples_list in language_triples.items():
    for t, letters in triples_list:
        triplet_universality[t].append((lang, letters))

# Rank triplets by universality
ranked_triplets = sorted(triplet_universality.items(), key=lambda x: len(x[1]), reverse=True)

print(f"\nMost universal triplets:")
print("-" * 60)
for t, lang_list in ranked_triplets[:15]:
    primes_str = f"{t}"
    langs = [f"{l}:{lt}" for l, lt in lang_list[:5]]
    more = f" +{len(lang_list)-5} more" if len(lang_list) > 5 else ""
    print(f"  {primes_str:<18}: {len(lang_list):>2} languages - {', '.join(langs)}{more}")

# =============================================================================
# 4. ALL ENCODING METHODS ON HIM TRIPLET
# =============================================================================

print("\n" + "=" * 80)
print("4. ALL ENCODING METHODS ON (23, 29, 31)")
print("=" * 80)

print(f"\n{'Method':<20} {'Values':<15} {'Letters':<10} {'Triple?'}")
print("-" * 55)

for method_name in ENCODING_METHODS.keys():
    try:
        # Use English for language-dependent methods
        vals = [ENCODING_METHODS[method_name](p, 'english', 26) for p in him]
        letters = ''.join(chr(ord('A') + (v - 1) % 26) for v in vals)
        is_triple = "YES!" if len(set(letters)) == 1 else ""
        print(f"  {method_name:<20} {str(vals):<15} {letters:<10} {is_triple}")
    except Exception as e:
        print(f"  {method_name:<20} Error: {e}")

# =============================================================================
# 5. REPRESENTATION-BASED ENCODINGS
# =============================================================================

print("\n" + "=" * 80)
print("5. REPRESENTATION-BASED ENCODINGS ON (23, 29, 31)")
print("=" * 80)

rep_methods = {
    'roman_length': lambda n: len(to_roman(n)),
    'binary_length': lambda n: len(to_binary(n)),
    'binary_ones': lambda n: to_binary(n).count('1'),
    'binary_zeros': lambda n: to_binary(n).count('0'),
    'ternary_length': lambda n: len(to_ternary(n)),
    'octal_length': lambda n: len(to_octal(n)),
    'hex_length': lambda n: len(to_hexadecimal(n)),
    'decimal_digits': lambda n: len(str(n)),
    'digit_sum': lambda n: sum(int(d) for d in str(n)),
    'digital_root': lambda n: digital_root(n),
    'balanced_ternary_len': lambda n: len(to_balanced_ternary(n)),
}

print(f"\n{'Method':<25} {'Values':<15} {'Letters':<10} {'Triple?'}")
print("-" * 60)

rep_triples = []
for method_name, method_func in rep_methods.items():
    vals = [method_func(p) for p in him]
    letters = ''.join(chr(ord('A') + (v - 1) % 26) for v in vals)
    is_triple = len(set(letters)) == 1
    marker = "YES!" if is_triple else ""
    print(f"  {method_name:<25} {str(vals):<15} {letters:<10} {marker}")
    if is_triple:
        rep_triples.append((method_name, letters))

# =============================================================================
# 6. CROSS-ALPHABET ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("6. CROSS-ALPHABET ANALYSIS (Letter Count Encoding)")
print("=" * 80)

print(f"\nHIM triplet letter counts: 23→{letter_count(ALL_LANGUAGES['chinese'](23))}, etc.")
print(f"\nUsing Chinese (counts 8,8,8):")

for alpha_name, alpha in ALPHABETS.items():
    size = len(alpha)
    # Chinese gives 8,8,8 - map to this alphabet
    pos = ((8 - 1) % size) + 1
    letter = alpha[pos - 1] if pos <= len(alpha) else '?'
    print(f"  {alpha_name:>12} ({size:>2} letters): position {pos} = '{letter}'")

# =============================================================================
# 7. COMPLETE STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("7. COMPLETE STATISTICS SUMMARY")
print("=" * 80)

# Count unique triple-letter patterns
all_triple_patterns = set()
for lang, triples_list in language_triples.items():
    for t, letters in triples_list:
        all_triple_patterns.add((t, letters))

print(f"""
MATRIX COVERAGE:
  Languages tested:           {len(ALL_LANGUAGES)}
  Encoding methods:           {len(ENCODING_METHODS)}
  Representation methods:     {len(rep_methods)}
  Alphabets:                  {len(ALPHABETS)}
  Prime triplets analyzed:    {len(triplets)}

RESULTS FOR (23, 29, 31):
  Languages with triple-letter: {len(triple_letter_languages)}
  Representation triples:       {len(rep_triples)}

OVERALL RESULTS:
  Unique (triplet, encoding) pairs with triple-letters: {len(all_triple_patterns)}
  Languages with at least one triple: {len(language_triples)}
  Most universal triplet: {ranked_triplets[0][0]} ({len(ranked_triplets[0][1])} languages)
""")

# =============================================================================
# 8. THE HIM TRIPLET SPECIAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("8. SPECIAL REPORT: (23, 29, 31) ACROSS ALL DIMENSIONS")
print("=" * 80)

him_report = {
    'triple_letter_languages': triple_letter_languages,
    'representation_triples': rep_triples,
}

print(f"""
THE HIM TRIPLET (23, 29, 31) EXHIBITS:

LANGUAGE TRIPLE-LETTERS ({len(triple_letter_languages)} languages):
""")
for lang, letters, counts in triple_letter_languages:
    word_23 = ALL_LANGUAGES[lang](23)
    print(f"  {lang:>20}: {letters} - '{word_23}' has {counts[0]} letters")

print(f"""
REPRESENTATION TRIPLE-LETTERS ({len(rep_triples)} methods):
""")
for method, letters in rep_triples:
    print(f"  {method:>25}: {letters}")

print(f"""
TOTAL TRIPLE-LETTER DIMENSIONS: {len(triple_letter_languages) + len(rep_triples)}
""")

# =============================================================================
# 9. SAVE COMPLETE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("9. ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
Files contain:
  - {len(ALL_LANGUAGES)} language number word functions
  - {len(ENCODING_METHODS)} encoding methods
  - {len(rep_methods)} representation-based encodings
  - {len(ALPHABETS)} alphabet systems

Total theoretical combinations: {len(ALL_LANGUAGES) * (len(ENCODING_METHODS) + len(rep_methods)) * len(ALPHABETS):,}
""")
