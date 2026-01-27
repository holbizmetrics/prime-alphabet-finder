#!/usr/bin/env python3
"""
Prime-to-Alphabet Pattern Finder
Systematically encodes primes across multiple languages and encoding methods
to discover patterns, words, and statistical anomalies.
"""

import math
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from collections import Counter
import itertools

# =============================================================================
# NUMBER WORDS BY LANGUAGE
# =============================================================================

# English number words
def english_number_word(n: int) -> str:
    """Convert number to English words."""
    if n == 0:
        return "zero"

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n < 20:
        return ones[n]
    elif n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    elif n < 1000:
        return ones[n // 100] + " hundred" + (" " + english_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        return english_number_word(n // 1000) + " thousand" + (" " + english_number_word(n % 1000) if n % 1000 else "")
    else:
        return english_number_word(n // 1000000) + " million" + (" " + english_number_word(n % 1000000) if n % 1000000 else "")

# German number words
def german_number_word(n: int) -> str:
    """Convert number to German words."""
    if n == 0:
        return "null"

    ones = ["", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
            "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
            "siebzehn", "achtzehn", "neunzehn"]
    tens = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        elif n % 10 == 1:
            return "einund" + tens[n // 10]
        else:
            return ones[n % 10] + "und" + tens[n // 10]
    elif n < 1000:
        prefix = "ein" if n // 100 == 1 else ones[n // 100]
        return prefix + "hundert" + (german_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "ein"
        else:
            prefix = german_number_word(n // 1000)
        return prefix + "tausend" + (german_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# French number words
def french_number_word(n: int) -> str:
    """Convert number to French words."""
    if n == 0:
        return "zéro"

    ones = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
            "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
            "dix-sept", "dix-huit", "dix-neuf"]
    tens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante", "soixante", "quatre-vingt", "quatre-vingt"]

    if n < 20:
        return ones[n]
    elif n < 70:
        if n % 10 == 0:
            return tens[n // 10]
        elif n % 10 == 1 and n // 10 in [2, 3, 4, 5, 6]:
            return tens[n // 10] + " et un"
        else:
            return tens[n // 10] + "-" + ones[n % 10]
    elif n < 80:
        return "soixante-" + french_number_word(n - 60)
    elif n < 100:
        if n == 80:
            return "quatre-vingts"
        return "quatre-vingt-" + french_number_word(n - 80)
    elif n < 1000:
        if n // 100 == 1:
            prefix = "cent"
        else:
            prefix = ones[n // 100] + " cent"
        if n % 100 == 0:
            return prefix + ("s" if n // 100 > 1 else "")
        return prefix + " " + french_number_word(n % 100)
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "mille"
        else:
            prefix = french_number_word(n // 1000) + " mille"
        return prefix + (" " + french_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Spanish number words
def spanish_number_word(n: int) -> str:
    """Convert number to Spanish words."""
    if n == 0:
        return "cero"

    ones = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
            "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
            "diecisiete", "dieciocho", "diecinueve"]
    tens = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]

    if n < 20:
        return ones[n]
    elif n < 30:
        if n == 20:
            return "veinte"
        return "veinti" + ones[n - 20]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return tens[n // 10] + " y " + ones[n % 10]
    elif n < 1000:
        if n == 100:
            return "cien"
        if n // 100 == 1:
            prefix = "ciento"
        elif n // 100 == 5:
            prefix = "quinientos"
        elif n // 100 == 7:
            prefix = "setecientos"
        elif n // 100 == 9:
            prefix = "novecientos"
        else:
            prefix = ones[n // 100] + "cientos"
        return prefix + (" " + spanish_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "mil"
        else:
            prefix = spanish_number_word(n // 1000) + " mil"
        return prefix + (" " + spanish_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Italian number words
def italian_number_word(n: int) -> str:
    """Convert number to Italian words."""
    if n == 0:
        return "zero"

    ones = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
            "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
            "diciassette", "diciotto", "diciannove"]
    tens = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]

    if n < 20:
        return ones[n]
    elif n < 100:
        ten = tens[n // 10]
        one = ones[n % 10]
        if n % 10 in [1, 8]:
            ten = ten[:-1]  # Drop last vowel before uno/otto
        return ten + one
    elif n < 1000:
        if n // 100 == 1:
            prefix = "cento"
        else:
            prefix = ones[n // 100] + "cento"
        return prefix + (italian_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "mille"
        else:
            prefix = italian_number_word(n // 1000) + "mila"
        return prefix + (italian_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Dutch number words
def dutch_number_word(n: int) -> str:
    """Convert number to Dutch words."""
    if n == 0:
        return "nul"

    ones = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen",
            "tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien",
            "zeventien", "achttien", "negentien"]
    tens = ["", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        connector = "ën" if n % 10 in [2, 3] else "en"
        return ones[n % 10] + connector + tens[n // 10]
    elif n < 1000:
        prefix = "" if n // 100 == 1 else ones[n // 100]
        return prefix + "honderd" + (dutch_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "" if n // 1000 == 1 else dutch_number_word(n // 1000)
        return prefix + "duizend" + (dutch_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Latin number words
def latin_number_word(n: int) -> str:
    """Convert number to Latin words."""
    if n == 0:
        return "nihil"

    ones = ["", "unus", "duo", "tres", "quattuor", "quinque", "sex", "septem", "octo", "novem",
            "decem", "undecim", "duodecim", "tredecim", "quattuordecim", "quindecim", "sedecim",
            "septendecim", "duodeviginti", "undeviginti"]
    tens = ["", "", "viginti", "triginta", "quadraginta", "quinquaginta", "sexaginta", "septuaginta", "octoginta", "nonaginta"]
    hundreds = ["", "centum", "ducenti", "trecenti", "quadringenti", "quingenti", "sescenti", "septingenti", "octingenti", "nongenti"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        return hundreds[n // 100] + (" " + latin_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "mille"
        else:
            prefix = latin_number_word(n // 1000) + " milia"
        return prefix + (" " + latin_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Hebrew number words (transliterated)
def hebrew_number_word(n: int) -> str:
    """Convert number to Hebrew words (transliterated)."""
    if n == 0:
        return "efes"

    ones_m = ["", "echad", "shnayim", "shlosha", "arba'a", "chamisha", "shisha", "shiv'a", "shmona", "tish'a",
              "asara", "achad-asar", "shnem-asar", "shlosha-asar", "arba'a-asar", "chamisha-asar",
              "shisha-asar", "shiv'a-asar", "shmona-asar", "tish'a-asar"]
    tens = ["", "", "esrim", "shloshim", "arba'im", "chamishim", "shishim", "shiv'im", "shmonim", "tish'im"]

    if n < 20:
        return ones_m[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return tens[n // 10] + " ve" + ones_m[n % 10]
    elif n < 1000:
        if n // 100 == 1:
            prefix = "me'a"
        elif n // 100 == 2:
            prefix = "matayim"
        else:
            prefix = ones_m[n // 100] + " me'ot"
        return prefix + (" ve" + hebrew_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "elef"
        elif n // 1000 == 2:
            prefix = "alpayim"
        else:
            prefix = hebrew_number_word(n // 1000) + " alafim"
        return prefix + (" ve" + hebrew_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Greek number words (transliterated)
def greek_number_word(n: int) -> str:
    """Convert number to Greek words (transliterated)."""
    if n == 0:
        return "miden"

    ones = ["", "ena", "dyo", "tria", "tessera", "pente", "exi", "epta", "okto", "ennea",
            "deka", "endeka", "dodeka", "dekatria", "dekatessera", "dekapente", "dekaexi",
            "dekaepta", "dekaokto", "dekaennea"]
    tens = ["", "", "eikosi", "trianta", "saranta", "peninta", "exinta", "evdominta", "ogdonta", "eneninta"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1:
            prefix = "ekato"
        elif n // 100 == 2:
            prefix = "diakosia"
        else:
            prefix = ones[n // 100] + "kosia"
        return prefix + (" " + greek_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "chilia"
        else:
            prefix = greek_number_word(n // 1000) + " chiliades"
        return prefix + (" " + greek_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Arabic number words (transliterated)
def arabic_number_word(n: int) -> str:
    """Convert number to Arabic words (transliterated)."""
    if n == 0:
        return "sifr"

    ones = ["", "wahid", "ithnan", "thalatha", "arba'a", "khamsa", "sitta", "sab'a", "thamaniya", "tis'a",
            "ashara", "ahad ashar", "ithna ashar", "thalatha ashar", "arba'a ashar", "khamsa ashar",
            "sitta ashar", "sab'a ashar", "thamaniya ashar", "tis'a ashar"]
    tens = ["", "", "ishrun", "thalathun", "arba'un", "khamsun", "sittun", "sab'un", "thamanun", "tis'un"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return ones[n % 10] + " wa " + tens[n // 10]
    elif n < 1000:
        if n // 100 == 1:
            prefix = "mi'a"
        elif n // 100 == 2:
            prefix = "mi'atan"
        else:
            prefix = ones[n // 100] + " mi'a"
        return prefix + (" wa " + arabic_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "alf"
        elif n // 1000 == 2:
            prefix = "alfan"
        else:
            prefix = arabic_number_word(n // 1000) + " alaf"
        return prefix + (" wa " + arabic_number_word(n % 1000) if n % 1000 else "")
    return str(n)

# Japanese number words (romaji)
def japanese_number_word(n: int) -> str:
    """Convert number to Japanese words (romaji)."""
    if n == 0:
        return "zero"

    ones = ["", "ichi", "ni", "san", "yon", "go", "roku", "nana", "hachi", "kyuu", "juu"]

    if n <= 10:
        return ones[n]
    elif n < 20:
        return "juu" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ones[n // 10] + "juu"
        return ones[n // 10] + "juu" + ones[n % 10]
    elif n < 1000:
        prefix = "" if n // 100 == 1 else ones[n // 100]
        return prefix + "hyaku" + (japanese_number_word(n % 100) if n % 100 else "")
    elif n < 10000:
        prefix = "" if n // 1000 == 1 else ones[n // 1000]
        return prefix + "sen" + (japanese_number_word(n % 1000) if n % 1000 else "")
    elif n < 100000000:
        prefix = japanese_number_word(n // 10000)
        return prefix + "man" + (japanese_number_word(n % 10000) if n % 10000 else "")
    return str(n)

# Chinese number words (pinyin)
def chinese_number_word(n: int) -> str:
    """Convert number to Chinese words (pinyin)."""
    if n == 0:
        return "ling"

    ones = ["", "yi", "er", "san", "si", "wu", "liu", "qi", "ba", "jiu", "shi"]

    if n <= 10:
        return ones[n]
    elif n < 20:
        return "shi" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ones[n // 10] + "shi"
        return ones[n // 10] + "shi" + ones[n % 10]
    elif n < 1000:
        prefix = ones[n // 100] + "bai"
        if n % 100 == 0:
            return prefix
        elif n % 100 < 10:
            return prefix + "ling" + ones[n % 100]
        return prefix + chinese_number_word(n % 100)
    elif n < 10000:
        prefix = ones[n // 1000] + "qian"
        if n % 1000 == 0:
            return prefix
        elif n % 1000 < 100:
            return prefix + "ling" + chinese_number_word(n % 1000)
        return prefix + chinese_number_word(n % 1000)
    elif n < 100000000:
        prefix = chinese_number_word(n // 10000) + "wan"
        if n % 10000 == 0:
            return prefix
        elif n % 10000 < 1000:
            return prefix + "ling" + chinese_number_word(n % 10000)
        return prefix + chinese_number_word(n % 10000)
    return str(n)

# Sanskrit number words (transliterated)
def sanskrit_number_word(n: int) -> str:
    """Convert number to Sanskrit words (transliterated)."""
    if n == 0:
        return "shunya"

    ones = ["", "eka", "dvi", "tri", "chatur", "pancha", "shat", "sapta", "ashta", "nava",
            "dasha", "ekadasha", "dvadasha", "trayodasha", "chaturdasha", "panchadasha",
            "shodasha", "saptadasha", "ashtadasha", "navadasha"]
    tens = ["", "", "vimshati", "trimshati", "chatvarimshat", "panchashat", "shashti", "saptati", "ashiti", "navati"]

    if n < 20:
        return ones[n]
    elif n < 100:
        if n % 10 == 0:
            return tens[n // 10]
        return ones[n % 10] + " " + tens[n // 10]
    elif n < 1000:
        if n // 100 == 1:
            prefix = "shatam"
        else:
            prefix = ones[n // 100] + " shatam"
        return prefix + (" " + sanskrit_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1:
            prefix = "sahasram"
        else:
            prefix = sanskrit_number_word(n // 1000) + " sahasram"
        return prefix + (" " + sanskrit_number_word(n % 1000) if n % 1000 else "")
    return str(n)


# =============================================================================
# LANGUAGE REGISTRY
# =============================================================================

LANGUAGES = {
    'english': english_number_word,
    'german': german_number_word,
    'french': french_number_word,
    'spanish': spanish_number_word,
    'italian': italian_number_word,
    'dutch': dutch_number_word,
    'latin': latin_number_word,
    'hebrew': hebrew_number_word,
    'greek': greek_number_word,
    'arabic': arabic_number_word,
    'japanese': japanese_number_word,
    'chinese': chinese_number_word,
    'sanskrit': sanskrit_number_word,
}


# =============================================================================
# PRIME GENERATION
# =============================================================================

def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def primes_up_to(limit: int) -> List[int]:
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def nth_prime(n: int) -> int:
    """Get the nth prime (1-indexed)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate):
            count += 1
    return candidate

def prime_index(p: int) -> int:
    """Get the index of prime p (1-indexed). Returns -1 if not prime."""
    if not is_prime(p):
        return -1
    count = 0
    for i in range(2, p + 1):
        if is_prime(i):
            count += 1
    return count

def prime_triplets(limit: int, gap: Tuple[int, int] = (6, 8)) -> List[Tuple[int, int, int]]:
    """Find prime triplets (p, p+gap[0], p+gap[1]) up to limit."""
    triplets = []
    primes_set = set(primes_up_to(limit + gap[1]))
    for p in primes_up_to(limit):
        if (p + gap[0]) in primes_set and (p + gap[1]) in primes_set:
            triplets.append((p, p + gap[0], p + gap[1]))
    return triplets


# =============================================================================
# ENCODING METHODS
# =============================================================================

def letter_count(word: str) -> int:
    """Count letters only (no spaces, hyphens, apostrophes)."""
    return sum(1 for c in word if c.isalpha())

def encode_letter_count(n: int, lang_func: Callable, alphabet_size: int = 26) -> int:
    """Encode number by letter count mod alphabet size."""
    word = lang_func(n)
    count = letter_count(word)
    return ((count - 1) % alphabet_size) + 1  # 1-indexed

def encode_digit_sum(n: int, alphabet_size: int = 26) -> int:
    """Encode by digit sum mod alphabet size."""
    digit_sum = sum(int(d) for d in str(n))
    return ((digit_sum - 1) % alphabet_size) + 1

def encode_direct_mod(n: int, alphabet_size: int = 26) -> int:
    """Encode by n mod alphabet size."""
    return ((n - 1) % alphabet_size) + 1

def encode_prime_index_mod(n: int, alphabet_size: int = 26) -> int:
    """Encode by prime index mod alphabet size."""
    idx = prime_index(n)
    if idx == -1:
        return 0
    return ((idx - 1) % alphabet_size) + 1


# =============================================================================
# ALPHABETS
# =============================================================================

ALPHABETS = {
    'english': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'hebrew': 'אבגדהוזחטיכלמנסעפצקרשת',  # 22 letters
    'greek': 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',  # 24 letters
    'cyrillic': 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',  # 33 letters
}

def num_to_letter(n: int, alphabet: str = 'english') -> str:
    """Convert number (1-indexed) to letter in alphabet."""
    if alphabet in ALPHABETS:
        alpha = ALPHABETS[alphabet]
    else:
        alpha = ALPHABETS['english']

    if n < 1 or n > len(alpha):
        n = ((n - 1) % len(alpha)) + 1
    return alpha[n - 1]


# =============================================================================
# PATTERN ANALYSIS
# =============================================================================

@dataclass
class EncodingResult:
    """Result of encoding a sequence of numbers."""
    language: str
    encoding_method: str
    alphabet: str
    numbers: List[int]
    encoded_values: List[int]
    letters: str

    def letter_counts(self) -> Dict[str, int]:
        return dict(Counter(self.letters))

    def total(self) -> int:
        return sum(self.encoded_values)

    def has_repeated_triple(self) -> List[Tuple[int, str]]:
        """Find positions where same letter appears 3+ times consecutively."""
        results = []
        i = 0
        while i < len(self.letters) - 2:
            if self.letters[i] == self.letters[i+1] == self.letters[i+2]:
                j = i + 3
                while j < len(self.letters) and self.letters[j] == self.letters[i]:
                    j += 1
                results.append((i, self.letters[i] * (j - i)))
                i = j
            else:
                i += 1
        return results


def encode_sequence(
    numbers: List[int],
    language: str,
    encoding_method: str = 'letter_count',
    alphabet: str = 'english'
) -> EncodingResult:
    """Encode a sequence of numbers."""
    lang_func = LANGUAGES.get(language, english_number_word)
    alpha = ALPHABETS.get(alphabet, ALPHABETS['english'])
    alpha_size = len(alpha)

    encoded = []
    for n in numbers:
        if encoding_method == 'letter_count':
            val = encode_letter_count(n, lang_func, alpha_size)
        elif encoding_method == 'digit_sum':
            val = encode_digit_sum(n, alpha_size)
        elif encoding_method == 'direct_mod':
            val = encode_direct_mod(n, alpha_size)
        elif encoding_method == 'prime_index':
            val = encode_prime_index_mod(n, alpha_size)
        else:
            val = encode_letter_count(n, lang_func, alpha_size)
        encoded.append(val)

    letters = ''.join(num_to_letter(v, alphabet) for v in encoded)

    return EncodingResult(
        language=language,
        encoding_method=encoding_method,
        alphabet=alphabet,
        numbers=numbers,
        encoded_values=encoded,
        letters=letters
    )


def find_words(letters: str, wordlist: List[str], min_length: int = 3) -> List[Tuple[int, str]]:
    """Find words from wordlist that appear in the letter sequence."""
    found = []
    letters_upper = letters.upper()
    for word in wordlist:
        if len(word) >= min_length:
            pos = letters_upper.find(word.upper())
            if pos != -1:
                found.append((pos, word))
    return sorted(found)


# =============================================================================
# CROSS-LANGUAGE ANALYSIS
# =============================================================================

def analyze_triplet_across_languages(
    triplet: Tuple[int, int, int],
    languages: List[str] = None,
    encoding_method: str = 'letter_count'
) -> Dict[str, str]:
    """Encode a triplet in all languages and return the letter codes."""
    if languages is None:
        languages = list(LANGUAGES.keys())

    results = {}
    for lang in languages:
        result = encode_sequence(list(triplet), lang, encoding_method)
        results[lang] = result.letters
    return results


def find_triple_letter_triplets(
    triplets: List[Tuple[int, int, int]],
    language: str,
    encoding_method: str = 'letter_count'
) -> List[Tuple[Tuple[int, int, int], str]]:
    """Find triplets where all three primes encode to the same letter."""
    matches = []
    for triplet in triplets:
        result = encode_sequence(list(triplet), language, encoding_method)
        if len(set(result.letters)) == 1:
            matches.append((triplet, result.letters[0] * 3))
    return matches


def cross_language_correlation(
    numbers: List[int],
    lang1: str,
    lang2: str,
    encoding_method: str = 'letter_count'
) -> float:
    """Calculate correlation between two language encodings."""
    r1 = encode_sequence(numbers, lang1, encoding_method)
    r2 = encode_sequence(numbers, lang2, encoding_method)

    n = len(numbers)
    if n == 0:
        return 0.0

    mean1 = sum(r1.encoded_values) / n
    mean2 = sum(r2.encoded_values) / n

    cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1.encoded_values, r2.encoded_values)) / n
    std1 = (sum((a - mean1) ** 2 for a in r1.encoded_values) / n) ** 0.5
    std2 = (sum((b - mean2) ** 2 for b in r2.encoded_values) / n) ** 0.5

    if std1 == 0 or std2 == 0:
        return 0.0

    return cov / (std1 * std2)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def full_analysis(
    limit: int = 100,
    languages: List[str] = None,
    encoding_methods: List[str] = None,
    find_patterns: bool = True
) -> Dict:
    """Run full analysis across all languages and methods."""

    if languages is None:
        languages = list(LANGUAGES.keys())

    if encoding_methods is None:
        encoding_methods = ['letter_count', 'digit_sum', 'direct_mod', 'prime_index']

    primes = primes_up_to(limit)
    triplets = prime_triplets(limit)

    results = {
        'primes': primes,
        'triplets': triplets,
        'encodings': {},
        'triple_letter_triplets': {},
        'correlations': {},
        'totals': {},
        'patterns': {}
    }

    # Encode in all language/method combinations
    for lang in languages:
        results['encodings'][lang] = {}
        results['triple_letter_triplets'][lang] = {}
        results['totals'][lang] = {}

        for method in encoding_methods:
            enc = encode_sequence(primes, lang, method)
            results['encodings'][lang][method] = {
                'letters': enc.letters,
                'values': enc.encoded_values,
                'total': enc.total(),
                'repeated_triples': enc.has_repeated_triple()
            }
            results['totals'][lang][method] = enc.total()

            # Find triple-letter triplets
            if triplets:
                tlt = find_triple_letter_triplets(triplets, lang, method)
                results['triple_letter_triplets'][lang][method] = tlt

    # Cross-language correlations for letter_count method
    if 'letter_count' in encoding_methods:
        for l1, l2 in itertools.combinations(languages, 2):
            corr = cross_language_correlation(primes, l1, l2, 'letter_count')
            results['correlations'][(l1, l2)] = corr

    return results


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_encoding_table(numbers: List[int], languages: List[str] = None):
    """Print a table of encodings across languages."""
    if languages is None:
        languages = ['english', 'german', 'french', 'spanish', 'italian', 'hebrew', 'greek']

    # Header
    print(f"{'Prime':>6} |", end='')
    for lang in languages:
        print(f" {lang[:7]:>7} |", end='')
    print()
    print("-" * (8 + 10 * len(languages)))

    for n in numbers:
        print(f"{n:>6} |", end='')
        for lang in languages:
            word = LANGUAGES[lang](n)
            count = letter_count(word)
            letter = num_to_letter(count)
            print(f" {count:>2}={letter:>1}    |", end='')
        print()


def print_triplet_analysis(triplet: Tuple[int, int, int]):
    """Print detailed analysis of a single triplet."""
    print(f"\n{'='*60}")
    print(f"TRIPLET: {triplet}")
    print(f"{'='*60}")

    # Get indices
    indices = [prime_index(p) for p in triplet]
    print(f"Prime indices: {indices}")
    print(f"Index sum: {sum(indices)}")
    print(f"Prime sum: {sum(triplet)}")
    print(f"Prime sum is prime: {is_prime(sum(triplet))}")

    print(f"\nEncodings by language (letter_count):")
    print("-" * 40)

    results = analyze_triplet_across_languages(triplet)
    for lang, letters in results.items():
        print(f"  {lang:>10}: {letters}")


if __name__ == "__main__":
    # Demo
    print("Prime-to-Alphabet Pattern Finder")
    print("=" * 50)

    # First 25 primes
    primes = primes_up_to(100)
    print(f"\nFirst {len(primes)} primes: {primes}")

    # Show encoding table
    print("\n" + "=" * 50)
    print("ENCODING TABLE (letter count)")
    print("=" * 50)
    print_encoding_table(primes[:12])

    # Find triplets
    triplets = prime_triplets(1000)
    print(f"\n\nFound {len(triplets)} prime triplets (p, p+6, p+8) under 1000")

    # Analyze HIM triplet
    print_triplet_analysis((23, 29, 31))
    print_triplet_analysis((101, 107, 109))

    # Find triple-letter triplets
    print("\n" + "=" * 50)
    print("TRIPLE-LETTER TRIPLETS (under 1000)")
    print("=" * 50)

    for lang in ['english', 'hebrew', 'greek', 'latin']:
        tlt = find_triple_letter_triplets(triplets, lang)
        if tlt:
            print(f"\n{lang.upper()}:")
            for triplet, letters in tlt:
                print(f"  {triplet} → {letters}")
