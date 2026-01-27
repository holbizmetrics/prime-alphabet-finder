#!/usr/bin/env python3
"""
Extended Pattern Search - Up to 10,000
Focuses on triple-letter triplets and statistical anomalies.
"""

from prime_encoder import *
from collections import defaultdict
import time

def extended_triple_letter_search(limit: int = 10000):
    """Search for triple-letter triplets across all languages up to limit."""

    print("=" * 70)
    print(f"EXTENDED TRIPLE-LETTER TRIPLET SEARCH (up to {limit})")
    print("=" * 70)

    start = time.time()

    # Get all prime triplets
    triplets = prime_triplets(limit)
    print(f"\nFound {len(triplets)} prime triplets (p, p+6, p+8) under {limit}")

    # Track results by language
    results = {lang: [] for lang in LANGUAGES.keys()}

    # Search each language
    for lang in LANGUAGES.keys():
        print(f"\nSearching {lang}...", end=" ", flush=True)
        tlt = find_triple_letter_triplets(triplets, lang, 'letter_count')
        results[lang] = tlt
        print(f"found {len(tlt)} triple-letter triplets")

    elapsed = time.time() - start
    print(f"\nSearch completed in {elapsed:.1f} seconds")

    return results, triplets


def analyze_triple_letter_distribution(results: dict, triplets: list):
    """Analyze the distribution of triple-letter triplets."""

    print("\n" + "=" * 70)
    print("TRIPLE-LETTER TRIPLET ANALYSIS")
    print("=" * 70)

    # Count by language
    print("\n1. COUNT BY LANGUAGE")
    print("-" * 40)

    counts = [(lang, len(tlt)) for lang, tlt in results.items()]
    counts.sort(key=lambda x: x[1], reverse=True)

    for lang, count in counts:
        pct = 100 * count / len(triplets) if triplets else 0
        bar = "█" * (count // 2) + "░" * ((50 - count) // 2)
        print(f"  {lang:>10}: {count:>3} ({pct:>5.1f}%) {bar[:30]}")

    # Which letters appear most often?
    print("\n2. LETTER FREQUENCY IN TRIPLE-LETTER TRIPLETS")
    print("-" * 40)

    letter_counts = defaultdict(lambda: defaultdict(int))
    for lang, tlt_list in results.items():
        for triplet, letters in tlt_list:
            letter = letters[0]  # They're all the same
            letter_counts[lang][letter] += 1

    for lang in ['hebrew', 'chinese', 'japanese', 'english', 'greek', 'latin']:
        if results[lang]:
            lc = letter_counts[lang]
            sorted_letters = sorted(lc.items(), key=lambda x: x[1], reverse=True)
            print(f"  {lang:>10}: {dict(sorted_letters[:10])}")

    # Find triplets that are triple-letter in MULTIPLE languages
    print("\n3. TRIPLETS WITH TRIPLE-LETTERS IN MULTIPLE LANGUAGES")
    print("-" * 40)

    triplet_langs = defaultdict(list)
    for lang, tlt_list in results.items():
        for triplet, letters in tlt_list:
            triplet_langs[triplet].append((lang, letters))

    multi_lang = [(t, langs) for t, langs in triplet_langs.items() if len(langs) >= 2]
    multi_lang.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"\n  Found {len(multi_lang)} triplets with triple-letters in 2+ languages:\n")

    for triplet, langs in multi_lang[:20]:
        indices = [prime_index(p) for p in triplet]
        idx_sum = sum(indices)
        print(f"  {triplet} (idx sum={idx_sum}):")
        for lang, letters in sorted(langs):
            print(f"      {lang:>10}: {letters}")

    # Special index sums
    print("\n4. TRIPLETS WITH SPECIAL INDEX SUMS")
    print("-" * 40)

    special_sums = [30, 83, 100, 137]  # 30 = HIM index sum, 83 = HIM prime sum

    for target_sum in special_sums:
        print(f"\n  Index sum = {target_sum}:")
        for triplet, langs in triplet_langs.items():
            indices = [prime_index(p) for p in triplet]
            if sum(indices) == target_sum:
                print(f"    {triplet}: ", end="")
                for lang, letters in langs:
                    print(f"{lang[:3]}={letters} ", end="")
                print()

    return triplet_langs, multi_lang


def find_letter_sequences(results: dict):
    """Look for interesting letter sequences across triplets."""

    print("\n" + "=" * 70)
    print("LETTER SEQUENCE ANALYSIS")
    print("=" * 70)

    for lang in ['hebrew', 'chinese', 'japanese', 'greek']:
        if not results[lang]:
            continue

        print(f"\n{lang.upper()} triple-letter sequence:")

        # Sort by first prime in triplet
        sorted_tlt = sorted(results[lang], key=lambda x: x[0][0])

        # Extract just the letters
        letters = [tlt[1][0] for tlt in sorted_tlt]
        sequence = ''.join(letters)

        print(f"  Triplets: {len(sorted_tlt)}")
        print(f"  Letter sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")

        # Check for patterns
        letter_positions = defaultdict(list)
        for i, (triplet, ltrs) in enumerate(sorted_tlt):
            letter_positions[ltrs[0]].append(i)

        # Find most common letters
        common = sorted(letter_positions.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        print(f"  Most common: {[(l, len(pos)) for l, pos in common]}")


def analyze_why_languages_differ(triplets: list):
    """Investigate why some languages have more triple-letter triplets."""

    print("\n" + "=" * 70)
    print("WHY DO LANGUAGES DIFFER?")
    print("=" * 70)

    # For each triplet, show the letter counts in different languages
    print("\n Sample of triplets with their letter counts:\n")

    sample_triplets = triplets[:10]

    # Header
    print(f"{'Triplet':<20}", end="")
    for lang in ['english', 'german', 'hebrew', 'chinese', 'japanese']:
        print(f"{lang[:7]:>10}", end="")
    print()
    print("-" * 70)

    for triplet in sample_triplets:
        print(f"{str(triplet):<20}", end="")
        for lang in ['english', 'german', 'hebrew', 'chinese', 'japanese']:
            counts = [letter_count(LANGUAGES[lang](p)) for p in triplet]
            # Check if all same
            if len(set(counts)) == 1:
                print(f"  [{counts[0]:>2}]*  ", end="")
            else:
                print(f"{counts[0]:>2},{counts[1]:>2},{counts[2]:>2}", end="")
        print()

    # Analyze variance in letter counts
    print("\n\nLetter count variance by language:")
    print("-" * 40)

    for lang in LANGUAGES.keys():
        all_counts = []
        for triplet in triplets:
            counts = [letter_count(LANGUAGES[lang](p)) for p in triplet]
            all_counts.extend(counts)

        if all_counts:
            mean = sum(all_counts) / len(all_counts)
            variance = sum((c - mean) ** 2 for c in all_counts) / len(all_counts)
            unique = len(set(all_counts))
            print(f"  {lang:>10}: mean={mean:.1f}, var={variance:.1f}, unique values={unique}")


def investigate_chinese_hhh(triplets: list):
    """Deep dive into why Chinese produces so many HHH triplets."""

    print("\n" + "=" * 70)
    print("CHINESE HHH INVESTIGATION")
    print("=" * 70)

    hhh_triplets = []

    for triplet in triplets:
        counts = [letter_count(LANGUAGES['chinese'](p)) for p in triplet]
        if counts[0] == counts[1] == counts[2] == 8:  # H is 8th letter
            hhh_triplets.append((triplet, counts))

    print(f"\nFound {len(hhh_triplets)} HHH triplets in Chinese:")

    for triplet, counts in hhh_triplets[:20]:
        words = [LANGUAGES['chinese'](p) for p in triplet]
        indices = [prime_index(p) for p in triplet]
        print(f"\n  {triplet} (indices {indices}):")
        for p, word in zip(triplet, words):
            print(f"    {p}: {word} ({letter_count(word)} letters)")

    # Why 8 letters?
    print("\n\nWhy do these numbers have 8-letter words in Chinese?")
    print("(Looking at the structure of Chinese number words...)")

    # Sample various primes and their Chinese representations
    print("\n  Prime  |  Chinese (pinyin)  |  Letters")
    print("  " + "-" * 45)

    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
        word = LANGUAGES['chinese'](p)
        count = letter_count(word)
        marker = " ← 8" if count == 8 else ""
        print(f"  {p:>5}  |  {word:<17}  |  {count}{marker}")


def comprehensive_extended_analysis():
    """Run the full extended analysis."""

    # Run the search
    results, triplets = extended_triple_letter_search(10000)

    # Analyze distribution
    triplet_langs, multi_lang = analyze_triple_letter_distribution(results, triplets)

    # Find letter sequences
    find_letter_sequences(results)

    # Analyze why languages differ
    analyze_why_languages_differ(triplets)

    # Special investigation: Chinese HHH
    investigate_chinese_hhh(triplets)

    # Summary statistics
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_triplets = len(triplets)

    print(f"\nTotal prime triplets analyzed: {total_triplets}")
    print(f"\nTriple-letter triplets by language:")

    for lang, tlt in sorted(results.items(), key=lambda x: len(x[1]), reverse=True):
        if tlt:
            pct = 100 * len(tlt) / total_triplets
            print(f"  {lang:>10}: {len(tlt):>4} ({pct:>5.1f}%)")

    print(f"\nTriplets with triple-letters in 2+ languages: {len(multi_lang)}")

    # Most "universal" triplets
    if multi_lang:
        most_universal = max(multi_lang, key=lambda x: len(x[1]))
        print(f"\nMost universal triplet: {most_universal[0]}")
        print(f"  Triple-letter in {len(most_universal[1])} languages:")
        for lang, letters in most_universal[1]:
            print(f"    {lang}: {letters}")

    return results, triplets, triplet_langs


if __name__ == "__main__":
    results, triplets, triplet_langs = comprehensive_extended_analysis()
