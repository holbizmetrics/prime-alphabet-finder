#!/usr/bin/env python3
"""
Extended Prime-to-Alphabet Pattern Finder
Expanded with 30+ languages, multiple encoding methods, and diverse prime patterns.
"""

import math
from typing import Dict, List, Tuple, Callable, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import itertools

# =============================================================================
# NUMBER WORDS - EXTENDED LANGUAGE COLLECTION (30+ LANGUAGES)
# =============================================================================

# --- GERMANIC LANGUAGES ---

def english_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if n < 20: return ones[n]
    elif n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    elif n < 1000:
        return ones[n // 100] + " hundred" + (" " + english_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        return english_number_word(n // 1000) + " thousand" + (" " + english_number_word(n % 1000) if n % 1000 else "")
    return english_number_word(n // 1000000) + " million" + (" " + english_number_word(n % 1000000) if n % 1000000 else "")

def german_number_word(n: int) -> str:
    if n == 0: return "null"
    ones = ["", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun",
            "zehn", "elf", "zwoelf", "dreizehn", "vierzehn", "fuenfzehn", "sechzehn",
            "siebzehn", "achtzehn", "neunzehn"]
    tens = ["", "", "zwanzig", "dreissig", "vierzig", "fuenfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        elif n % 10 == 1: return "einund" + tens[n // 10]
        else: return ones[n % 10] + "und" + tens[n // 10]
    elif n < 1000:
        prefix = "ein" if n // 100 == 1 else ones[n // 100]
        return prefix + "hundert" + (german_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "ein" if n // 1000 == 1 else german_number_word(n // 1000)
        return prefix + "tausend" + (german_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def dutch_number_word(n: int) -> str:
    if n == 0: return "nul"
    ones = ["", "een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen",
            "tien", "elf", "twaalf", "dertien", "veertien", "vijftien", "zestien",
            "zeventien", "achttien", "negentien"]
    tens = ["", "", "twintig", "dertig", "veertig", "vijftig", "zestig", "zeventig", "tachtig", "negentig"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + "en" + tens[n // 10]
    elif n < 1000:
        prefix = "" if n // 100 == 1 else ones[n // 100]
        return prefix + "honderd" + (dutch_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "" if n // 1000 == 1 else dutch_number_word(n // 1000)
        return prefix + "duizend" + (dutch_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def swedish_number_word(n: int) -> str:
    if n == 0: return "noll"
    ones = ["", "ett", "tva", "tre", "fyra", "fem", "sex", "sju", "atta", "nio",
            "tio", "elva", "tolv", "tretton", "fjorton", "femton", "sexton",
            "sjutton", "arton", "nitton"]
    tens = ["", "", "tjugo", "trettio", "fyrtio", "femtio", "sextio", "sjuttio", "attio", "nittio"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + ones[n % 10]
    elif n < 1000:
        prefix = "ett" if n // 100 == 1 else ones[n // 100]
        return prefix + "hundra" + (swedish_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "ett" if n // 1000 == 1 else swedish_number_word(n // 1000)
        return prefix + "tusen" + (swedish_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def norwegian_number_word(n: int) -> str:
    if n == 0: return "null"
    ones = ["", "en", "to", "tre", "fire", "fem", "seks", "sju", "atte", "ni",
            "ti", "elleve", "tolv", "tretten", "fjorten", "femten", "seksten",
            "sytten", "atten", "nitten"]
    tens = ["", "", "tjue", "tretti", "forti", "femti", "seksti", "sytti", "atti", "nitti"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + ones[n % 10]
    elif n < 1000:
        prefix = "ett" if n // 100 == 1 else ones[n // 100]
        return prefix + "hundre" + (norwegian_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "ett" if n // 1000 == 1 else norwegian_number_word(n // 1000)
        return prefix + "tusen" + (norwegian_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def danish_number_word(n: int) -> str:
    if n == 0: return "nul"
    ones = ["", "en", "to", "tre", "fire", "fem", "seks", "syv", "otte", "ni",
            "ti", "elleve", "tolv", "tretten", "fjorten", "femten", "seksten",
            "sytten", "atten", "nitten"]
    tens = ["", "", "tyve", "tredive", "fyrre", "halvtreds", "tres", "halvfjerds", "firs", "halvfems"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + "og" + tens[n // 10]
    elif n < 1000:
        prefix = "et" if n // 100 == 1 else ones[n // 100]
        return prefix + "hundrede" + (danish_number_word(n % 100) if n % 100 else "")
    return str(n)

def icelandic_number_word(n: int) -> str:
    if n == 0: return "null"
    ones = ["", "einn", "tveir", "thrir", "fjorir", "fimm", "sex", "sjo", "atta", "niu",
            "tiu", "ellefu", "tolf", "threttan", "fjortan", "fimmtan", "sextan",
            "sautjan", "atjan", "nitjan"]
    tens = ["", "", "tuttugu", "thrjatiu", "fjorutil", "fimmtiu", "sextiu", "sjotiu", "attatiu", "niutiu"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " og " + ones[n % 10]
    elif n < 1000:
        prefix = "eitt" if n // 100 == 1 else ones[n // 100]
        return prefix + " hundrad" + (" " + icelandic_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- ROMANCE LANGUAGES ---

def french_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
            "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
            "dix-sept", "dix-huit", "dix-neuf"]
    tens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante", "soixante", "quatre-vingt", "quatre-vingt"]
    if n < 20: return ones[n]
    elif n < 70:
        if n % 10 == 0: return tens[n // 10]
        elif n % 10 == 1 and n // 10 in [2,3,4,5,6]: return tens[n // 10] + " et un"
        else: return tens[n // 10] + "-" + ones[n % 10]
    elif n < 80: return "soixante-" + french_number_word(n - 60)
    elif n < 100:
        if n == 80: return "quatre-vingts"
        return "quatre-vingt-" + french_number_word(n - 80)
    elif n < 1000:
        prefix = "cent" if n // 100 == 1 else ones[n // 100] + " cent"
        if n % 100 == 0: return prefix + ("s" if n // 100 > 1 else "")
        return prefix + " " + french_number_word(n % 100)
    elif n < 1000000:
        prefix = "mille" if n // 1000 == 1 else french_number_word(n // 1000) + " mille"
        return prefix + (" " + french_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def spanish_number_word(n: int) -> str:
    if n == 0: return "cero"
    ones = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
            "diez", "once", "doce", "trece", "catorce", "quince", "dieciseis",
            "diecisiete", "dieciocho", "diecinueve"]
    tens = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
    if n < 20: return ones[n]
    elif n < 30:
        if n == 20: return "veinte"
        return "veinti" + ones[n - 20]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " y " + ones[n % 10]
    elif n < 1000:
        if n == 100: return "cien"
        prefix = "ciento" if n // 100 == 1 else ones[n // 100] + "cientos"
        return prefix + (" " + spanish_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "mil" if n // 1000 == 1 else spanish_number_word(n // 1000) + " mil"
        return prefix + (" " + spanish_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def italian_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove",
            "dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici",
            "diciassette", "diciotto", "diciannove"]
    tens = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]
    if n < 20: return ones[n]
    elif n < 100:
        ten = tens[n // 10]
        one = ones[n % 10]
        if n % 10 in [1, 8]: ten = ten[:-1]
        return ten + one
    elif n < 1000:
        prefix = "cento" if n // 100 == 1 else ones[n // 100] + "cento"
        return prefix + (italian_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "mille" if n // 1000 == 1 else italian_number_word(n // 1000) + "mila"
        return prefix + (italian_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def portuguese_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove",
            "dez", "onze", "doze", "treze", "catorze", "quinze", "dezasseis",
            "dezassete", "dezoito", "dezanove"]
    tens = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " e " + ones[n % 10]
    elif n < 1000:
        if n == 100: return "cem"
        prefix = "cento" if n // 100 == 1 else ones[n // 100] + "centos"
        return prefix + (" e " + portuguese_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        prefix = "mil" if n // 1000 == 1 else portuguese_number_word(n // 1000) + " mil"
        return prefix + (" e " + portuguese_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def romanian_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "unu", "doi", "trei", "patru", "cinci", "sase", "sapte", "opt", "noua",
            "zece", "unsprezece", "doisprezece", "treisprezece", "paisprezece", "cincisprezece",
            "sasesprezece", "saptesprezece", "optsprezece", "nouasprezece"]
    tens = ["", "", "douazeci", "treizeci", "patruzeci", "cincizeci", "saizeci", "saptezeci", "optzeci", "nouazeci"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " si " + ones[n % 10]
    elif n < 1000:
        if n == 100: return "o suta"
        if n // 100 == 1: prefix = "o suta"
        elif n // 100 == 2: prefix = "doua sute"
        else: prefix = ones[n // 100] + " sute"
        return prefix + (" " + romanian_number_word(n % 100) if n % 100 else "")
    return str(n)

def latin_number_word(n: int) -> str:
    if n == 0: return "nihil"
    ones = ["", "unus", "duo", "tres", "quattuor", "quinque", "sex", "septem", "octo", "novem",
            "decem", "undecim", "duodecim", "tredecim", "quattuordecim", "quindecim", "sedecim",
            "septendecim", "duodeviginti", "undeviginti"]
    tens = ["", "", "viginti", "triginta", "quadraginta", "quinquaginta", "sexaginta", "septuaginta", "octoginta", "nonaginta"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        hundreds = ["", "centum", "ducenti", "trecenti", "quadringenti", "quingenti", "sescenti", "septingenti", "octingenti", "nongenti"]
        return hundreds[n // 100] + (" " + latin_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- SLAVIC LANGUAGES ---

def russian_number_word(n: int) -> str:
    if n == 0: return "nol"
    ones = ["", "odin", "dva", "tri", "chetyre", "pyat", "shest", "sem", "vosem", "devyat",
            "desyat", "odinnadtsat", "dvenadtsat", "trinadtsat", "chetyrnadtsat", "pyatnadtsat",
            "shestnadtsat", "semnadtsat", "vosemnadtsat", "devyatnadtsat"]
    tens = ["", "", "dvadtsat", "tridtsat", "sorok", "pyatdesyat", "shestdesyat", "semdesyat", "vosemdesyat", "devyanosto"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "sto"
        elif n // 100 == 2: prefix = "dvesti"
        elif n // 100 in [3,4]: prefix = ones[n // 100] + "sta"
        else: prefix = ones[n // 100] + "sot"
        return prefix + (" " + russian_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1: prefix = "tysyacha"
        elif n // 1000 in [2,3,4]: prefix = russian_number_word(n // 1000) + " tysyachi"
        else: prefix = russian_number_word(n // 1000) + " tysyach"
        return prefix + (" " + russian_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def polish_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "jeden", "dwa", "trzy", "cztery", "piec", "szesc", "siedem", "osiem", "dziewiec",
            "dziesiec", "jedenascie", "dwanascie", "trzynascie", "czternascie", "pietnascie",
            "szesnascie", "siedemnascie", "osiemnascie", "dziewietnascie"]
    tens = ["", "", "dwadziescia", "trzydziesci", "czterdziesci", "piecdziesiat", "szescdziesiat",
            "siedemdziesiat", "osiemdziesiat", "dziewiecdziesiat"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "sto"
        elif n // 100 == 2: prefix = "dwiescie"
        elif n // 100 in [3,4]: prefix = ones[n // 100] + "sta"
        else: prefix = ones[n // 100] + "set"
        return prefix + (" " + polish_number_word(n % 100) if n % 100 else "")
    return str(n)

def czech_number_word(n: int) -> str:
    if n == 0: return "nula"
    ones = ["", "jedna", "dva", "tri", "ctyri", "pet", "sest", "sedm", "osm", "devet",
            "deset", "jedenact", "dvanact", "trinact", "ctrnact", "patnact",
            "sestnact", "sedmnact", "osmnact", "devatenact"]
    tens = ["", "", "dvacet", "tricet", "ctyricet", "padesat", "sedesat", "sedmdesat", "osmdesat", "devadesat"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "sto"
        elif n // 100 == 2: prefix = "dveste"
        else: prefix = ones[n // 100] + " set"
        return prefix + (" " + czech_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- CELTIC LANGUAGES ---

def irish_number_word(n: int) -> str:
    """Irish (Gaeilge) number words."""
    if n == 0: return "naid"
    ones = ["", "aon", "do", "tri", "ceathair", "cuig", "se", "seacht", "ocht", "naoi", "deich"]
    teens = ["", "aon deag", "do dheag", "tri deag", "ceathair deag", "cuig deag",
             "se deag", "seacht deag", "ocht deag", "naoi deag"]
    tens = ["", "", "fiche", "triochad", "daichead", "caoga", "seasca", "seachto", "ochto", "nocha"]
    if n <= 10: return ones[n]
    elif n < 20: return teens[n - 10]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " a " + ones[n % 10]
    elif n < 1000:
        prefix = "cead" if n // 100 == 1 else ones[n // 100] + " chead"
        return prefix + (" " + irish_number_word(n % 100) if n % 100 else "")
    return str(n)

def welsh_number_word(n: int) -> str:
    """Welsh (Cymraeg) number words."""
    if n == 0: return "dim"
    ones = ["", "un", "dau", "tri", "pedwar", "pump", "chwech", "saith", "wyth", "naw", "deg"]
    teens = ["", "un ar ddeg", "deuddeg", "tri ar ddeg", "pedwar ar ddeg", "pymtheg",
             "un ar bymtheg", "dau ar bymtheg", "deunaw", "pedwar ar bymtheg"]
    tens = ["", "", "ugain", "deg ar hugain", "deugain", "hanner cant", "trigain", "deg a thrigain", "pedwar ugain", "deg a phedwar ugain"]
    if n <= 10: return ones[n]
    elif n < 20: return teens[n - 10]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " ar " + tens[n // 10]
    elif n < 1000:
        prefix = "cant" if n // 100 == 1 else ones[n // 100] + " cant"
        return prefix + (" " + welsh_number_word(n % 100) if n % 100 else "")
    return str(n)

def scottish_gaelic_number_word(n: int) -> str:
    """Scottish Gaelic (Gaidhlig) number words."""
    if n == 0: return "neoni"
    ones = ["", "aon", "dha", "tri", "ceithir", "coig", "sia", "seachd", "ochd", "naoi", "deich"]
    teens = ["", "aon deug", "dha dheug", "tri deug", "ceithir deug", "coig deug",
             "sia deug", "seachd deug", "ochd deug", "naoi deug"]
    tens = ["", "", "fichead", "trithead", "ceathrad", "caogad", "seasgad", "seachdad", "ochdad", "naochad"]
    if n <= 10: return ones[n]
    elif n < 20: return teens[n - 10]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " sa " + ones[n % 10]
    return str(n)

# --- SEMITIC LANGUAGES ---

def hebrew_number_word(n: int) -> str:
    if n == 0: return "efes"
    ones = ["", "echad", "shnayim", "shlosha", "arba", "chamisha", "shisha", "shiva", "shmona", "tisha",
            "asara", "achad-asar", "shnem-asar", "shlosha-asar", "arba-asar", "chamisha-asar",
            "shisha-asar", "shiva-asar", "shmona-asar", "tisha-asar"]
    tens = ["", "", "esrim", "shloshim", "arbaim", "chamishim", "shishim", "shivim", "shmonim", "tishim"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " ve" + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "mea"
        elif n // 100 == 2: prefix = "matayim"
        else: prefix = ones[n // 100] + " meot"
        return prefix + (" ve" + hebrew_number_word(n % 100) if n % 100 else "")
    elif n < 1000000:
        if n // 1000 == 1: prefix = "elef"
        elif n // 1000 == 2: prefix = "alpayim"
        else: prefix = hebrew_number_word(n // 1000) + " alafim"
        return prefix + (" ve" + hebrew_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def arabic_number_word(n: int) -> str:
    if n == 0: return "sifr"
    ones = ["", "wahid", "ithnan", "thalatha", "arbaa", "khamsa", "sitta", "saba", "thamaniya", "tisa",
            "ashara", "ahad ashar", "ithna ashar", "thalatha ashar", "arbaa ashar", "khamsa ashar",
            "sitta ashar", "saba ashar", "thamaniya ashar", "tisa ashar"]
    tens = ["", "", "ishrun", "thalathun", "arbaun", "khamsun", "sittun", "sabun", "thamanun", "tisun"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " wa " + tens[n // 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "mia"
        elif n // 100 == 2: prefix = "miatan"
        else: prefix = ones[n // 100] + " mia"
        return prefix + (" wa " + arabic_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- HELLENIC ---

def greek_number_word(n: int) -> str:
    if n == 0: return "miden"
    ones = ["", "ena", "dyo", "tria", "tessera", "pente", "exi", "epta", "okto", "ennea",
            "deka", "endeka", "dodeka", "dekatria", "dekatessera", "dekapente", "dekaexi",
            "dekaepta", "dekaokto", "dekaennea"]
    tens = ["", "", "eikosi", "trianta", "saranta", "peninta", "exinta", "evdominta", "ogdonta", "eneninta"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "ekato"
        elif n // 100 == 2: prefix = "diakosia"
        else: prefix = ones[n // 100] + "kosia"
        return prefix + (" " + greek_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- INDO-ARYAN ---

def hindi_number_word(n: int) -> str:
    """Hindi number words (transliterated)."""
    if n == 0: return "shunya"
    # Hindi has unique words for 1-99
    words = {
        1: "ek", 2: "do", 3: "teen", 4: "char", 5: "paanch",
        6: "chhah", 7: "saat", 8: "aath", 9: "nau", 10: "das",
        11: "gyarah", 12: "barah", 13: "terah", 14: "chaudah", 15: "pandrah",
        16: "solah", 17: "satrah", 18: "athaarah", 19: "unnis", 20: "bees",
        21: "ikkis", 22: "baais", 23: "teis", 24: "chaubis", 25: "pachchis",
        26: "chhabbis", 27: "sattais", 28: "atthaais", 29: "untis", 30: "tees",
        31: "ikatees", 32: "battis", 33: "tentis", 34: "chautis", 35: "paintis",
        36: "chhattis", 37: "saintis", 38: "adtis", 39: "untalis", 40: "chalis",
        41: "iktalis", 42: "bayalis", 43: "tentalis", 44: "chavalis", 45: "paintalis",
        46: "chhiyalis", 47: "saintalis", 48: "adtalis", 49: "unchaas", 50: "pachaas",
        51: "ikyavan", 52: "baavan", 53: "tirpan", 54: "chauvan", 55: "pachpan",
        56: "chhappan", 57: "sattavan", 58: "athavan", 59: "unsath", 60: "saath",
        61: "iksath", 62: "basath", 63: "tirsath", 64: "chausath", 65: "painsath",
        66: "chhiyasath", 67: "sarsath", 68: "adsath", 69: "unhattar", 70: "sattar",
        71: "ikattar", 72: "bahattar", 73: "tihattar", 74: "chauhattar", 75: "pachattar",
        76: "chhihattar", 77: "satattar", 78: "athhattar", 79: "unasi", 80: "assi",
        81: "ikasi", 82: "bayasi", 83: "tirasi", 84: "chaurasi", 85: "pachaasi",
        86: "chhiyasi", 87: "sattasi", 88: "athasi", 89: "navasi", 90: "nabbe",
        91: "ikyaanave", 92: "baanave", 93: "tiranave", 94: "chauranave", 95: "pachanave",
        96: "chhiyanave", 97: "sattanave", 98: "atthanave", 99: "ninyanave"
    }
    if n in words: return words[n]
    elif n < 1000:
        if n == 100: return "sau"
        prefix = words.get(n // 100, str(n // 100)) + " sau"
        return prefix + (" " + hindi_number_word(n % 100) if n % 100 else "")
    elif n < 100000:
        if n // 1000 == 1: prefix = "hazaar"
        else: prefix = hindi_number_word(n // 1000) + " hazaar"
        return prefix + (" " + hindi_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def sanskrit_number_word(n: int) -> str:
    if n == 0: return "shunya"
    ones = ["", "eka", "dvi", "tri", "chatur", "pancha", "shat", "sapta", "ashta", "nava",
            "dasha", "ekadasha", "dvadasha", "trayodasha", "chaturdasha", "panchadasha",
            "shodasha", "saptadasha", "ashtadasha", "navadasha"]
    tens = ["", "", "vimshati", "trimshati", "chatvarimshat", "panchashat", "shashti", "saptati", "ashiti", "navati"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " " + tens[n // 10]
    elif n < 1000:
        prefix = "shatam" if n // 100 == 1 else ones[n // 100] + " shatam"
        return prefix + (" " + sanskrit_number_word(n % 100) if n % 100 else "")
    return str(n)

def bengali_number_word(n: int) -> str:
    """Bengali number words (transliterated)."""
    if n == 0: return "shunno"
    ones = ["", "ek", "dui", "tin", "char", "panch", "chhoy", "shat", "at", "noy",
            "dosh", "egaro", "baro", "tero", "choddo", "ponero", "sholo",
            "shotero", "atharo", "unish"]
    tens = ["", "", "kuri", "trish", "chollish", "ponchash", "shaath", "shottor", "ashi", "nobboi"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        prefix = "eksho" if n // 100 == 1 else ones[n // 100] + "sho"
        return prefix + (" " + bengali_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- EAST ASIAN ---

def japanese_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "ichi", "ni", "san", "yon", "go", "roku", "nana", "hachi", "kyuu", "juu"]
    if n <= 10: return ones[n]
    elif n < 20: return "juu" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0: return ones[n // 10] + "juu"
        return ones[n // 10] + "juu" + ones[n % 10]
    elif n < 1000:
        prefix = "" if n // 100 == 1 else ones[n // 100]
        return prefix + "hyaku" + (japanese_number_word(n % 100) if n % 100 else "")
    elif n < 10000:
        prefix = "" if n // 1000 == 1 else ones[n // 1000]
        return prefix + "sen" + (japanese_number_word(n % 1000) if n % 1000 else "")
    return str(n)

def chinese_number_word(n: int) -> str:
    if n == 0: return "ling"
    ones = ["", "yi", "er", "san", "si", "wu", "liu", "qi", "ba", "jiu", "shi"]
    if n <= 10: return ones[n]
    elif n < 20: return "shi" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0: return ones[n // 10] + "shi"
        return ones[n // 10] + "shi" + ones[n % 10]
    elif n < 1000:
        prefix = ones[n // 100] + "bai"
        if n % 100 == 0: return prefix
        elif n % 100 < 10: return prefix + "ling" + ones[n % 100]
        return prefix + chinese_number_word(n % 100)
    elif n < 10000:
        prefix = ones[n // 1000] + "qian"
        if n % 1000 == 0: return prefix
        elif n % 1000 < 100: return prefix + "ling" + chinese_number_word(n % 1000)
        return prefix + chinese_number_word(n % 1000)
    return str(n)

def korean_number_word(n: int) -> str:
    """Korean number words (native system, transliterated)."""
    if n == 0: return "yeong"
    ones = ["", "hana", "dul", "set", "net", "daseot", "yeoseot", "ilgop", "yeodeol", "ahop", "yeol"]
    tens = ["", "", "seumul", "seoreun", "maheun", "swin", "yesun", "ilheun", "yeodeun", "aheun"]
    if n <= 10: return ones[n]
    elif n < 20: return "yeol" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        prefix = "baek" if n // 100 == 1 else korean_number_word(n // 100) + "baek"
        return prefix + (" " + korean_number_word(n % 100) if n % 100 else "")
    return str(n)

def vietnamese_number_word(n: int) -> str:
    """Vietnamese number words."""
    if n == 0: return "khong"
    ones = ["", "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "muoi" + (" " + ones[n % 10] if n % 10 else "")
    elif n < 100:
        prefix = ones[n // 10] + " muoi"
        if n % 10 == 0: return prefix
        elif n % 10 == 5: return prefix + " lam"
        elif n % 10 == 1: return prefix + " mot"
        return prefix + " " + ones[n % 10]
    elif n < 1000:
        prefix = ones[n // 100] + " tram"
        if n % 100 == 0: return prefix
        elif n % 100 < 10: return prefix + " le " + ones[n % 100]
        return prefix + " " + vietnamese_number_word(n % 100)
    return str(n)

def thai_number_word(n: int) -> str:
    """Thai number words (transliterated)."""
    if n == 0: return "sun"
    ones = ["", "nueng", "song", "sam", "si", "ha", "hok", "chet", "paet", "kao", "sip"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "sip" + (" " + ones[n % 10] if n % 10 else "")
    elif n < 100:
        if n % 10 == 0: return ones[n // 10] + " sip"
        elif n % 10 == 1: return ones[n // 10] + " sip et"
        return ones[n // 10] + " sip " + ones[n % 10]
    elif n < 1000:
        prefix = "nueng roi" if n // 100 == 1 else ones[n // 100] + " roi"
        return prefix + (" " + thai_number_word(n % 100) if n % 100 else "")
    return str(n)

# --- OTHER LANGUAGES ---

def finnish_number_word(n: int) -> str:
    """Finnish number words."""
    if n == 0: return "nolla"
    ones = ["", "yksi", "kaksi", "kolme", "nelja", "viisi", "kuusi", "seitseman", "kahdeksan", "yhdeksan",
            "kymmenen", "yksitoista", "kaksitoista", "kolmetoista", "neljätoista", "viisitoista",
            "kuusitoista", "seitsemäntoista", "kahdeksantoista", "yhdeksäntoista"]
    tens = ["", "", "kaksikymmentä", "kolmekymmentä", "neljakymmentä", "viisikymmentä",
            "kuusikymmentä", "seitsemänkymmentä", "kahdeksankymmentä", "yhdeksänkymmentä"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + ones[n % 10]
    elif n < 1000:
        prefix = "sata" if n // 100 == 1 else ones[n // 100] + "sataa"
        return prefix + (finnish_number_word(n % 100) if n % 100 else "")
    return str(n)

def hungarian_number_word(n: int) -> str:
    """Hungarian number words."""
    if n == 0: return "nulla"
    ones = ["", "egy", "ketto", "harom", "negy", "ot", "hat", "het", "nyolc", "kilenc",
            "tiz", "tizenegy", "tizenketto", "tizenharom", "tizennegy", "tizeot",
            "tizenhat", "tizenhet", "tizennyolc", "tizenkilenc"]
    tens = ["", "", "husz", "harminc", "negyven", "otven", "hatvan", "hetven", "nyolcvan", "kilencven"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + ones[n % 10]
    elif n < 1000:
        prefix = "szaz" if n // 100 == 1 else ones[n // 100] + "szaz"
        return prefix + (hungarian_number_word(n % 100) if n % 100 else "")
    return str(n)

def turkish_number_word(n: int) -> str:
    """Turkish number words."""
    if n == 0: return "sifir"
    ones = ["", "bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz", "dokuz"]
    tens = ["", "on", "yirmi", "otuz", "kirk", "elli", "altmis", "yetmis", "seksen", "doksan"]
    if n < 10: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        prefix = "yuz" if n // 100 == 1 else ones[n // 100] + " yuz"
        return prefix + (" " + turkish_number_word(n % 100) if n % 100 else "")
    return str(n)

def indonesian_number_word(n: int) -> str:
    """Indonesian/Malay number words."""
    if n == 0: return "nol"
    ones = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
            "sepuluh", "sebelas", "dua belas", "tiga belas", "empat belas", "lima belas",
            "enam belas", "tujuh belas", "delapan belas", "sembilan belas"]
    tens = ["", "", "dua puluh", "tiga puluh", "empat puluh", "lima puluh",
            "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    elif n < 1000:
        prefix = "seratus" if n // 100 == 1 else ones[n // 100] + " ratus"
        return prefix + (" " + indonesian_number_word(n % 100) if n % 100 else "")
    return str(n)

def swahili_number_word(n: int) -> str:
    """Swahili number words."""
    if n == 0: return "sifuri"
    ones = ["", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "kumi na " + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ones[n // 10] + "ini" if n // 10 in [2,3,4,5] else ones[n // 10] + "i"
        tens = ["", "", "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini", "themanini", "tisini"]
        return tens[n // 10] + " na " + ones[n % 10]
    elif n < 1000:
        prefix = "mia moja" if n // 100 == 1 else "mia " + ones[n // 100]
        return prefix + (" na " + swahili_number_word(n % 100) if n % 100 else "")
    return str(n)

def basque_number_word(n: int) -> str:
    """Basque (Euskara) number words."""
    if n == 0: return "zero"
    ones = ["", "bat", "bi", "hiru", "lau", "bost", "sei", "zazpi", "zortzi", "bederatzi",
            "hamar", "hamaika", "hamabi", "hamahiru", "hamalau", "hamabost",
            "hamasei", "hamazazpi", "hemezortzi", "hemeretzi"]
    tens = ["", "", "hogei", "hogeita hamar", "berrogei", "berrogeita hamar",
            "hirurogei", "hirurogeita hamar", "laurogei", "laurogeita hamar"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        # Basque uses vigesimal system
        if n < 40:
            return "hogeita " + ones[n - 20]
        elif n < 60:
            return "berrogeita " + ones[n - 40]
        elif n < 80:
            return "hirurogeita " + ones[n - 60]
        else:
            return "laurogeita " + ones[n - 80]
    elif n < 1000:
        prefix = "ehun" if n // 100 == 1 else ones[n // 100] + "ehun"
        return prefix + (" " + basque_number_word(n % 100) if n % 100 else "")
    return str(n)

def esperanto_number_word(n: int) -> str:
    """Esperanto number words."""
    if n == 0: return "nul"
    ones = ["", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "nau"]
    if n < 10: return ones[n]
    elif n < 100:
        if n < 20:
            return "dek" + (" " + ones[n % 10] if n % 10 else "")
        if n % 10 == 0:
            return ones[n // 10] + "dek"
        return ones[n // 10] + "dek " + ones[n % 10]
    elif n < 1000:
        if n // 100 == 1: prefix = "cent"
        else: prefix = ones[n // 100] + "cent"
        return prefix + (" " + esperanto_number_word(n % 100) if n % 100 else "")
    return str(n)

# =============================================================================
# LANGUAGE REGISTRY (35 LANGUAGES)
# =============================================================================

LANGUAGES = {
    # Germanic
    'english': english_number_word,
    'german': german_number_word,
    'dutch': dutch_number_word,
    'swedish': swedish_number_word,
    'norwegian': norwegian_number_word,
    'danish': danish_number_word,
    'icelandic': icelandic_number_word,

    # Romance
    'french': french_number_word,
    'spanish': spanish_number_word,
    'italian': italian_number_word,
    'portuguese': portuguese_number_word,
    'romanian': romanian_number_word,
    'latin': latin_number_word,

    # Slavic
    'russian': russian_number_word,
    'polish': polish_number_word,
    'czech': czech_number_word,

    # Celtic
    'irish': irish_number_word,
    'welsh': welsh_number_word,
    'scottish_gaelic': scottish_gaelic_number_word,

    # Semitic
    'hebrew': hebrew_number_word,
    'arabic': arabic_number_word,

    # Hellenic
    'greek': greek_number_word,

    # Indo-Aryan
    'hindi': hindi_number_word,
    'sanskrit': sanskrit_number_word,
    'bengali': bengali_number_word,

    # East Asian
    'japanese': japanese_number_word,
    'chinese': chinese_number_word,
    'korean': korean_number_word,
    'vietnamese': vietnamese_number_word,
    'thai': thai_number_word,

    # Uralic
    'finnish': finnish_number_word,
    'hungarian': hungarian_number_word,

    # Turkic
    'turkish': turkish_number_word,

    # Austronesian
    'indonesian': indonesian_number_word,

    # Niger-Congo
    'swahili': swahili_number_word,

    # Language Isolate
    'basque': basque_number_word,

    # Constructed
    'esperanto': esperanto_number_word,
}

# =============================================================================
# ALPHABETS (EXTENDED)
# =============================================================================

ALPHABETS = {
    'english': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # 26
    'hebrew': 'אבגדהוזחטיכלמנסעפצקרשת',  # 22
    'greek': 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',  # 24
    'cyrillic': 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',  # 33
    'arabic': 'ابتثجحخدذرزسشصضطظعغفقكلمنهوي',  # 28
    'devanagari': 'अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह',  # 46
    'hiragana': 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん',  # 46
    'hangul': 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅓㅗㅜㅡㅣ',  # 21 (basic)
    'thai': 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ',  # 44
    'runic': 'ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ',  # 24 (Elder Futhark)
    'ogham': 'ᚁᚂᚃᚄᚅᚆᚇᚈᚉᚊᚋᚌᚍᚎᚏᚐᚑᚒᚓᚔ',  # 20
}

# =============================================================================
# PRIME GENERATION AND PATTERNS
# =============================================================================

def is_prime(n: int) -> bool:
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def primes_up_to(limit: int) -> List[int]:
    if limit < 2: return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def nth_prime(n: int) -> int:
    if n < 1: raise ValueError("n must be >= 1")
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if is_prime(candidate): count += 1
    return candidate

def prime_index(p: int) -> int:
    if not is_prime(p): return -1
    count = 0
    for i in range(2, p + 1):
        if is_prime(i): count += 1
    return count

# --- PRIME PATTERNS ---

def twin_primes(limit: int) -> List[Tuple[int, int]]:
    """Find twin prime pairs (p, p+2) up to limit."""
    pairs = []
    primes_set = set(primes_up_to(limit + 2))
    for p in primes_up_to(limit):
        if (p + 2) in primes_set:
            pairs.append((p, p + 2))
    return pairs

def cousin_primes(limit: int) -> List[Tuple[int, int]]:
    """Find cousin prime pairs (p, p+4) up to limit."""
    pairs = []
    primes_set = set(primes_up_to(limit + 4))
    for p in primes_up_to(limit):
        if (p + 4) in primes_set:
            pairs.append((p, p + 4))
    return pairs

def sexy_primes(limit: int) -> List[Tuple[int, int]]:
    """Find sexy prime pairs (p, p+6) up to limit."""
    pairs = []
    primes_set = set(primes_up_to(limit + 6))
    for p in primes_up_to(limit):
        if (p + 6) in primes_set:
            pairs.append((p, p + 6))
    return pairs

def prime_triplets(limit: int, gap: Tuple[int, int] = (6, 8)) -> List[Tuple[int, int, int]]:
    """Find prime triplets (p, p+gap[0], p+gap[1]) up to limit."""
    triplets = []
    primes_set = set(primes_up_to(limit + gap[1]))
    for p in primes_up_to(limit):
        if (p + gap[0]) in primes_set and (p + gap[1]) in primes_set:
            triplets.append((p, p + gap[0], p + gap[1]))
    return triplets

def prime_quadruplets(limit: int) -> List[Tuple[int, int, int, int]]:
    """Find prime quadruplets (p, p+2, p+6, p+8) up to limit."""
    quads = []
    primes_set = set(primes_up_to(limit + 8))
    for p in primes_up_to(limit):
        if all((p + d) in primes_set for d in [2, 6, 8]):
            quads.append((p, p+2, p+6, p+8))
    return quads

def prime_quintuplets(limit: int) -> List[Tuple[int, ...]]:
    """Find prime quintuplets (p, p+2, p+6, p+8, p+12) or (p, p+4, p+6, p+10, p+12)."""
    quints = []
    primes_set = set(primes_up_to(limit + 12))
    for p in primes_up_to(limit):
        # Type 1: (p, p+2, p+6, p+8, p+12)
        if all((p + d) in primes_set for d in [2, 6, 8, 12]):
            quints.append((p, p+2, p+6, p+8, p+12))
        # Type 2: (p, p+4, p+6, p+10, p+12)
        elif all((p + d) in primes_set for d in [4, 6, 10, 12]):
            quints.append((p, p+4, p+6, p+10, p+12))
    return quints

def sophie_germain_primes(limit: int) -> List[Tuple[int, int]]:
    """Find Sophie Germain primes: p where both p and 2p+1 are prime."""
    pairs = []
    primes_set = set(primes_up_to(2 * limit + 1))
    for p in primes_up_to(limit):
        if (2*p + 1) in primes_set:
            pairs.append((p, 2*p + 1))
    return pairs

def balanced_primes(limit: int) -> List[Tuple[int, int, int]]:
    """Find balanced primes: primes equal to average of neighbors."""
    primes = primes_up_to(limit)
    balanced = []
    for i in range(1, len(primes) - 1):
        if primes[i] == (primes[i-1] + primes[i+1]) // 2:
            balanced.append((primes[i-1], primes[i], primes[i+1]))
    return balanced

def emirps(limit: int) -> List[Tuple[int, int]]:
    """Find emirps: primes that become different primes when reversed."""
    pairs = []
    primes_set = set(primes_up_to(limit))
    for p in primes_up_to(limit):
        rev = int(str(p)[::-1])
        if rev != p and rev in primes_set:
            pairs.append((p, rev))
    return pairs

def palindromic_primes(limit: int) -> List[int]:
    """Find palindromic primes."""
    return [p for p in primes_up_to(limit) if str(p) == str(p)[::-1]]

def mersenne_primes(limit: int) -> List[int]:
    """Find Mersenne primes 2^p - 1 up to limit."""
    mersennes = []
    p = 2
    while True:
        m = (1 << p) - 1  # 2^p - 1
        if m > limit: break
        if is_prime(m):
            mersennes.append(m)
        p += 1
    return mersennes

def fibonacci_primes(limit: int) -> List[int]:
    """Find Fibonacci numbers that are prime, up to limit."""
    fibs = [1, 1]
    while fibs[-1] < limit:
        fibs.append(fibs[-1] + fibs[-2])
    return [f for f in fibs if f > 1 and is_prime(f) and f <= limit]

def arithmetic_prime_progressions(limit: int, length: int = 3) -> List[Tuple[int, ...]]:
    """Find arithmetic progressions of primes."""
    primes = primes_up_to(limit)
    primes_set = set(primes)
    progressions = []

    for i, start in enumerate(primes):
        for j in range(i + 1, len(primes)):
            diff = primes[j] - start
            progression = [start]
            current = start + diff
            while current in primes_set and len(progression) < length:
                progression.append(current)
                current += diff
            if len(progression) >= length:
                progressions.append(tuple(progression[:length]))

    return list(set(progressions))

# =============================================================================
# ENCODING METHODS (EXTENDED)
# =============================================================================

def letter_count(word: str) -> int:
    """Count letters only (no spaces, hyphens, apostrophes)."""
    return sum(1 for c in word if c.isalpha())

def vowel_count(word: str) -> int:
    """Count vowels in word."""
    vowels = set('aeiouAEIOUàáâãäåæèéêëìíîïòóôõöùúûüýÿ')
    return sum(1 for c in word if c in vowels)

def consonant_count(word: str) -> int:
    """Count consonants in word."""
    return letter_count(word) - vowel_count(word)

def syllable_count_estimate(word: str) -> int:
    """Estimate syllable count (simple heuristic)."""
    vowels = 'aeiouAEIOUàáâãäåæèéêëìíîïòóôõöùúûüýÿ'
    count = 0
    prev_vowel = False
    for c in word:
        is_vowel = c in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count) if word else 0

def digital_root(n: int) -> int:
    """Calculate digital root (repeated digit sum until single digit)."""
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n

def digit_sum(n: int) -> int:
    """Sum of digits."""
    return sum(int(d) for d in str(n))

def digit_product(n: int) -> int:
    """Product of digits."""
    result = 1
    for d in str(n):
        result *= int(d)
    return result

def prime_digit_count(n: int) -> int:
    """Count of prime digits (2, 3, 5, 7) in n."""
    return sum(1 for d in str(n) if d in '2357')

def binary_ones_count(n: int) -> int:
    """Count of 1s in binary representation."""
    return bin(n).count('1')

# Encoding functions
def encode_letter_count(n: int, lang_func: Callable, alphabet_size: int = 26) -> int:
    word = lang_func(n)
    count = letter_count(word)
    return ((count - 1) % alphabet_size) + 1

def encode_vowel_count(n: int, lang_func: Callable, alphabet_size: int = 26) -> int:
    word = lang_func(n)
    count = vowel_count(word)
    return ((count - 1) % alphabet_size) + 1 if count > 0 else 1

def encode_consonant_count(n: int, lang_func: Callable, alphabet_size: int = 26) -> int:
    word = lang_func(n)
    count = consonant_count(word)
    return ((count - 1) % alphabet_size) + 1 if count > 0 else 1

def encode_syllable_count(n: int, lang_func: Callable, alphabet_size: int = 26) -> int:
    word = lang_func(n)
    count = syllable_count_estimate(word)
    return ((count - 1) % alphabet_size) + 1

def encode_digit_sum(n: int, alphabet_size: int = 26) -> int:
    return ((digit_sum(n) - 1) % alphabet_size) + 1

def encode_digital_root(n: int, alphabet_size: int = 26) -> int:
    return ((digital_root(n) - 1) % alphabet_size) + 1

def encode_direct_mod(n: int, alphabet_size: int = 26) -> int:
    return ((n - 1) % alphabet_size) + 1

def encode_prime_index_mod(n: int, alphabet_size: int = 26) -> int:
    idx = prime_index(n)
    if idx == -1: return 0
    return ((idx - 1) % alphabet_size) + 1

def encode_binary_ones(n: int, alphabet_size: int = 26) -> int:
    return ((binary_ones_count(n) - 1) % alphabet_size) + 1

# =============================================================================
# ENCODING REGISTRY
# =============================================================================

ENCODING_METHODS = {
    'letter_count': lambda n, lang, alpha_size: encode_letter_count(n, LANGUAGES[lang], alpha_size),
    'vowel_count': lambda n, lang, alpha_size: encode_vowel_count(n, LANGUAGES[lang], alpha_size),
    'consonant_count': lambda n, lang, alpha_size: encode_consonant_count(n, LANGUAGES[lang], alpha_size),
    'syllable_count': lambda n, lang, alpha_size: encode_syllable_count(n, LANGUAGES[lang], alpha_size),
    'digit_sum': lambda n, lang, alpha_size: encode_digit_sum(n, alpha_size),
    'digital_root': lambda n, lang, alpha_size: encode_digital_root(n, alpha_size),
    'direct_mod': lambda n, lang, alpha_size: encode_direct_mod(n, alpha_size),
    'prime_index': lambda n, lang, alpha_size: encode_prime_index_mod(n, alpha_size),
    'binary_ones': lambda n, lang, alpha_size: encode_binary_ones(n, alpha_size),
}

PRIME_PATTERNS = {
    'twin_primes': twin_primes,
    'cousin_primes': cousin_primes,
    'sexy_primes': sexy_primes,
    'triplets_6_8': lambda lim: prime_triplets(lim, (6, 8)),
    'triplets_2_6': lambda lim: prime_triplets(lim, (2, 6)),
    'quadruplets': prime_quadruplets,
    'quintuplets': prime_quintuplets,
    'sophie_germain': sophie_germain_primes,
    'balanced': balanced_primes,
    'emirps': emirps,
    'palindromic': palindromic_primes,
    'mersenne': mersenne_primes,
    'fibonacci': fibonacci_primes,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def num_to_letter(n: int, alphabet: str = 'english') -> str:
    if alphabet in ALPHABETS:
        alpha = ALPHABETS[alphabet]
    else:
        alpha = ALPHABETS['english']
    if n < 1 or n > len(alpha):
        n = ((n - 1) % len(alpha)) + 1
    return alpha[n - 1]

def encode_number(n: int, language: str, method: str = 'letter_count', alphabet: str = 'english') -> str:
    """Encode a single number to a letter."""
    alpha_size = len(ALPHABETS.get(alphabet, ALPHABETS['english']))
    if method in ENCODING_METHODS:
        val = ENCODING_METHODS[method](n, language, alpha_size)
    else:
        val = encode_letter_count(n, LANGUAGES[language], alpha_size)
    return num_to_letter(val, alphabet)

def encode_sequence(numbers: List[int], language: str, method: str = 'letter_count', alphabet: str = 'english') -> str:
    """Encode a sequence of numbers to letters."""
    return ''.join(encode_number(n, language, method, alphabet) for n in numbers)

# =============================================================================
# PATTERN FINDING
# =============================================================================

def find_triple_letter_patterns(pattern: List[Tuple], language: str, method: str = 'letter_count') -> List[Tuple]:
    """Find patterns where all elements encode to the same letter."""
    matches = []
    for item in pattern:
        if isinstance(item, int):
            continue  # Skip single numbers
        letters = encode_sequence(list(item), language, method)
        if len(set(letters)) == 1:
            matches.append((item, letters[0] * len(item)))
    return matches

def analyze_pattern_across_languages(pattern: Tuple, method: str = 'letter_count') -> Dict[str, str]:
    """Encode a pattern in all languages."""
    results = {}
    for lang in LANGUAGES.keys():
        results[lang] = encode_sequence(list(pattern), lang, method)
    return results

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("Extended Prime-to-Alphabet Pattern Finder")
    print("=" * 60)
    print(f"\nLanguages available: {len(LANGUAGES)}")
    print(f"Alphabets available: {len(ALPHABETS)}")
    print(f"Encoding methods: {len(ENCODING_METHODS)}")
    print(f"Prime patterns: {len(PRIME_PATTERNS)}")

    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS: The HIM triplet (23, 29, 31)")
    print("=" * 60)

    triplet = (23, 29, 31)
    print(f"\nTriplet: {triplet}")
    print(f"Sum: {sum(triplet)} = 83 = p(23)")

    print("\nLetter-count encodings across all {len(LANGUAGES)} languages:")
    for lang in sorted(LANGUAGES.keys()):
        letters = encode_sequence(list(triplet), lang, 'letter_count')
        triple = " ← TRIPLE" if len(set(letters)) == 1 else ""
        print(f"  {lang:>16}: {letters}{triple}")
