#!/usr/bin/env python3
"""
Extended Languages Module - 60+ Languages
"""

# =============================================================================
# ADDITIONAL LANGUAGES (Adding 30+ more)
# =============================================================================

# --- ROMANCE ---

def catalan_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "un", "dos", "tres", "quatre", "cinc", "sis", "set", "vuit", "nou",
            "deu", "onze", "dotze", "tretze", "catorze", "quinze", "setze",
            "disset", "divuit", "dinou"]
    tens = ["", "", "vint", "trenta", "quaranta", "cinquanta", "seixanta", "setanta", "vuitanta", "noranta"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        if n < 30: return "vint-i-" + ones[n % 10]
        return tens[n // 10] + "-" + ones[n % 10]
    elif n < 1000:
        if n == 100: return "cent"
        prefix = "cent" if n // 100 == 1 else ones[n // 100] + "-cents"
        return prefix + (" " + catalan_number_word(n % 100) if n % 100 else "")
    return str(n)

def galician_number_word(n: int) -> str:
    if n == 0: return "cero"
    ones = ["", "un", "dous", "tres", "catro", "cinco", "seis", "sete", "oito", "nove",
            "dez", "once", "doce", "trece", "catorce", "quince", "dezaseis",
            "dezasete", "dezaoito", "dezanove"]
    tens = ["", "", "vinte", "trinta", "corenta", "cincuenta", "sesenta", "setenta", "oitenta", "noventa"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " e " + ones[n % 10]
    return str(n)

# --- SLAVIC ---

def ukrainian_number_word(n: int) -> str:
    if n == 0: return "nol"
    ones = ["", "odyn", "dva", "try", "chotyry", "pyat", "shist", "sim", "visim", "devyat",
            "desyat", "odynadtsyat", "dvanadtsyat", "trynadtsyat", "chotyrnadtsyat", "pyatnadtsyat",
            "shistnadtsyat", "simnadtsyat", "visimnadtsyat", "devyatnadtsyat"]
    tens = ["", "", "dvadtsyat", "trydtsyat", "sorok", "pyatdesyat", "shistdesyat", "simdesyat", "visimdesyat", "devyanosto"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def bulgarian_number_word(n: int) -> str:
    if n == 0: return "nula"
    ones = ["", "edno", "dve", "tri", "chetiri", "pet", "shest", "sedem", "osem", "devet",
            "deset", "edinadeset", "dvanadeset", "trinadeset", "chetirinadeset", "petnadeset",
            "shestnadeset", "sedemnadeset", "osemnadeset", "devetnadeset"]
    tens = ["", "", "dvadeset", "trideset", "chetirideset", "petdeset", "shestdeset", "sedemdeset", "osemdeset", "devetdeset"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " i " + ones[n % 10]
    return str(n)

def serbian_number_word(n: int) -> str:
    if n == 0: return "nula"
    ones = ["", "jedan", "dva", "tri", "cetiri", "pet", "sest", "sedam", "osam", "devet",
            "deset", "jedanaest", "dvanaest", "trinaest", "cetrnaest", "petnaest",
            "sesnaest", "sedamnaest", "osamnaest", "devetnaest"]
    tens = ["", "", "dvadeset", "trideset", "cetrdeset", "pedeset", "sezdeset", "sedamdeset", "osamdeset", "devedeset"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def croatian_number_word(n: int) -> str:
    return serbian_number_word(n)  # Very similar

def slovak_number_word(n: int) -> str:
    if n == 0: return "nula"
    ones = ["", "jeden", "dva", "tri", "styri", "pat", "sest", "sedem", "osem", "devat",
            "desat", "jedenast", "dvanast", "trinast", "strnast", "patnast",
            "sestnast", "sedemnast", "osemnast", "devatnast"]
    tens = ["", "", "dvadsat", "tridsat", "styridsat", "patdesiat", "sestdesiat", "sedemdesiat", "osemdesiat", "devatdesiat"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def slovenian_number_word(n: int) -> str:
    if n == 0: return "nic"
    ones = ["", "ena", "dva", "tri", "stiri", "pet", "sest", "sedem", "osem", "devet",
            "deset", "enajst", "dvanajst", "trinajst", "stirinajst", "petnajst",
            "sestnajst", "sedemnajst", "osemnajst", "devetnajst"]
    tens = ["", "", "dvajset", "trideset", "stirideset", "petdeset", "sestdeset", "sedemdeset", "osemdeset", "devetdeset"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def macedonian_number_word(n: int) -> str:
    if n == 0: return "nula"
    ones = ["", "eden", "dva", "tri", "cetiri", "pet", "sest", "sedum", "osum", "devet",
            "deset", "edinaeset", "dvanaeset", "trinaeset", "cetirinaeset", "petnaeset",
            "sesnaeset", "sedumnaeset", "osumnaeset", "devetnaeset"]
    tens = ["", "", "dvaeset", "trieset", "cetirieset", "pedeset", "sezdeset", "sedumeset", "osumeset", "devedeset"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " i " + ones[n % 10]
    return str(n)

# --- BALTIC ---

def lithuanian_number_word(n: int) -> str:
    if n == 0: return "nulis"
    ones = ["", "vienas", "du", "trys", "keturi", "penki", "sesi", "septyni", "astuoni", "devyni",
            "desimt", "vienuolika", "dvylika", "trylika", "keturiolika", "penkiolika",
            "sesiolika", "septyniolika", "astuoniolika", "devyniolika"]
    tens = ["", "", "dvidesimt", "trisdesimt", "keturiasdesimt", "penkiasdesimt", "sesiasdesimt", "septyniasdesimt", "astuoniasdesimt", "devyniasdesimt"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def latvian_number_word(n: int) -> str:
    if n == 0: return "nulle"
    ones = ["", "viens", "divi", "tris", "cetri", "pieci", "sesi", "septini", "astoni", "devini",
            "desmit", "vienpadsmit", "divpadsmit", "trispadsmit", "cetrpadsmit", "piecpadsmit",
            "sespadsmit", "septinpadsmit", "astonpadsmit", "devinpadsmit"]
    tens = ["", "", "divdesmit", "trisdesmit", "cetrdesmit", "piecdesmit", "sesdesmit", "septindesmit", "astondesmit", "devindesmit"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def estonian_number_word(n: int) -> str:
    if n == 0: return "null"
    ones = ["", "uks", "kaks", "kolm", "neli", "viis", "kuus", "seitse", "kaheksa", "uheksa",
            "kumme", "uksteist", "kaksteist", "kolmteist", "neliteist", "viisteist",
            "kuusteist", "seitseteist", "kaheksateist", "uheksateist"]
    tens = ["", "", "kakskummend", "kolmkummend", "nelikummend", "viiskummend", "kuuskummend", "seitsekummend", "kaheksakummend", "uheksakummend"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

# --- OTHER EUROPEAN ---

def albanian_number_word(n: int) -> str:
    if n == 0: return "zero"
    ones = ["", "nje", "dy", "tre", "kater", "pese", "gjashte", "shtate", "tete", "nente",
            "dhjete", "njembedhjete", "dymbedhjete", "trembedhjete", "katermbedhjete", "pesembedhjete",
            "gjashtembedhjete", "shtatembedhjete", "tetembedhjete", "nentembedhjete"]
    tens = ["", "", "njezet", "tridhjete", "dyzet", "pesedhjete", "gjashtedhjete", "shtatedhjete", "tetedhjete", "nentedhjete"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " e " + ones[n % 10]
    return str(n)

def maltese_number_word(n: int) -> str:
    if n == 0: return "xejn"
    ones = ["", "wiehed", "tnejn", "tlieta", "erbgha", "hamsa", "sitta", "sebgha", "tmienja", "disgha",
            "ghaxra", "hdax", "tnax", "tlettax", "erbatax", "hmistax",
            "sittax", "sbatax", "tmintax", "dsatax"]
    tens = ["", "", "ghoxrin", "tletin", "erbghin", "hamsin", "sittin", "sebghin", "tmenin", "disghin"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " u " + tens[n // 10]
    return str(n)

# --- MIDDLE EASTERN / CAUCASIAN ---

def persian_number_word(n: int) -> str:
    """Farsi number words (transliterated)"""
    if n == 0: return "sefr"
    ones = ["", "yek", "do", "se", "chahar", "panj", "shesh", "haft", "hasht", "noh",
            "dah", "yazdah", "davazdah", "sizdah", "chahardah", "panzdah",
            "shanzdah", "hefdah", "hejdah", "nuzdah"]
    tens = ["", "", "bist", "si", "chehel", "panjah", "shast", "haftad", "hashtad", "navad"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " o " + ones[n % 10]
    return str(n)

def georgian_number_word(n: int) -> str:
    """Georgian number words (transliterated)"""
    if n == 0: return "nuli"
    ones = ["", "erti", "ori", "sami", "otkhi", "khuti", "ekvsi", "shvidi", "rva", "tskhra",
            "ati", "tertmeti", "tormeti", "tsameti", "totkhmet", "khutmeti",
            "tekvsmeti", "chvidmeti", "tvrameti", "tskhrament"]
    tens = ["", "", "otsi", "ocdaati", "ormotsi", "ormotsdaati", "samotsi", "samotsdaati", "otkhrmotsi", "otkhrmotsdaati"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        # Georgian uses vigesimal
        if n < 40:
            return "otsdaati" if n == 30 else "otsi da " + ones[n - 20]
        return tens[n // 10] + " da " + ones[n % 10]
    return str(n)

def armenian_number_word(n: int) -> str:
    """Armenian number words (transliterated)"""
    if n == 0: return "zero"
    ones = ["", "mek", "yerku", "yerek", "chors", "hing", "vec", "yot", "ut", "inna",
            "tas", "tasnemek", "tasnerku", "tasnerek", "tasnchors", "tasnhing",
            "tasnvec", "tasnyot", "tasnut", "tasninna"]
    tens = ["", "", "ksan", "yeresun", "karasun", "hisun", "vatsun", "yotanasun", "utanasun", "innsun"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def azerbaijani_number_word(n: int) -> str:
    """Azerbaijani number words"""
    if n == 0: return "sifir"
    ones = ["", "bir", "iki", "uc", "dord", "bes", "alti", "yeddi", "sekkiz", "doqquz",
            "on", "on bir", "on iki", "on uc", "on dord", "on bes",
            "on alti", "on yeddi", "on sekkiz", "on doqquz"]
    tens = ["", "", "iyirmi", "otuz", "qirx", "elli", "altmis", "yetmis", "seksen", "doxsan"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def kazakh_number_word(n: int) -> str:
    """Kazakh number words (transliterated)"""
    if n == 0: return "nol"
    ones = ["", "bir", "yeki", "ush", "tort", "bes", "alty", "zheti", "segiz", "togyz",
            "on", "on bir", "on yeki", "on ush", "on tort", "on bes",
            "on alty", "on zheti", "on segiz", "on togyz"]
    tens = ["", "", "zhiyrma", "otyz", "qyryq", "yelu", "alpys", "zhetpis", "seksen", "toqsan"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

# --- AFRICAN ---

def yoruba_number_word(n: int) -> str:
    """Yoruba number words"""
    if n == 0: return "odo"
    ones = ["", "okan", "eji", "eta", "erin", "arun", "efa", "eje", "ejo", "esan", "ewa"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "okan din " + ones[20 - n]  # Subtractive system
    elif n < 100:
        if n % 10 == 0:
            return ["", "", "ogun", "ogbon", "ogoji", "aadota", "ogota", "aadotin", "ogerin", "aadorun"][n // 10]
        return yoruba_number_word((n // 10) * 10) + " le " + ones[n % 10]
    return str(n)

def hausa_number_word(n: int) -> str:
    """Hausa number words"""
    if n == 0: return "sifili"
    ones = ["", "daya", "biyu", "uku", "hudu", "biyar", "shida", "bakwai", "takwas", "tara", "goma"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "goma sha " + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ["", "", "ashirin", "talatin", "arbahin", "hamsin", "sittin", "sabahin", "tamanin", "tisahin"][n // 10]
        return hausa_number_word((n // 10) * 10) + " da " + ones[n % 10]
    return str(n)

def amharic_number_word(n: int) -> str:
    """Amharic number words (transliterated)"""
    if n == 0: return "zero"
    ones = ["", "and", "hulet", "sost", "arat", "amist", "sidist", "sebat", "simint", "zetegn", "asir"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "asra " + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ["", "", "haya", "selasa", "arba", "hamsa", "silsa", "seba", "semanya", "zetena"][n // 10]
        return amharic_number_word((n // 10) * 10) + " " + ones[n % 10]
    return str(n)

def zulu_number_word(n: int) -> str:
    """Zulu number words"""
    if n == 0: return "iqanda"
    ones = ["", "kunye", "kubili", "kuthathu", "kune", "kuhlanu", "isithupha", "isikhombisa", "isishiyagalombili", "isishiyagalolunye", "ishumi"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "ishumi nan" + ones[n - 10]
    elif n < 100:
        if n % 10 == 0:
            return ["", "", "amashumi amabili", "amashumi amathathu", "amashumi amane", "amashumi amahlanu",
                    "amashumi ayisithupha", "amashumi ayisikhombisa", "amashumi ayisishiyagalombili", "amashumi ayisishiyagalolunye"][n // 10]
    return str(n)

def xhosa_number_word(n: int) -> str:
    """Xhosa number words"""
    if n == 0: return "zero"
    ones = ["", "nye", "mbini", "ntathu", "ne", "ntlanu", "ntandathu", "sixhenxe", "sibhozo", "lithoba", "lishumi"]
    if n <= 10: return ones[n]
    return str(n)

def afrikaans_number_word(n: int) -> str:
    """Afrikaans number words"""
    if n == 0: return "nul"
    ones = ["", "een", "twee", "drie", "vier", "vyf", "ses", "sewe", "agt", "nege",
            "tien", "elf", "twaalf", "dertien", "veertien", "vyftien",
            "sestien", "sewentien", "agtien", "negentien"]
    tens = ["", "", "twintig", "dertig", "veertig", "vyftig", "sestig", "sewentig", "tagtig", "negentig"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " en " + tens[n // 10]
    return str(n)

# --- SOUTH/SOUTHEAST ASIAN ---

def urdu_number_word(n: int) -> str:
    """Urdu number words (similar to Hindi, transliterated)"""
    if n == 0: return "sifar"
    # Urdu uses same number words as Hindi with slight pronunciation differences
    words = {
        1: "ek", 2: "do", 3: "teen", 4: "char", 5: "panch",
        6: "chhe", 7: "saat", 8: "aath", 9: "nau", 10: "das",
        11: "gyarah", 12: "barah", 13: "terah", 14: "chaudah", 15: "pandrah",
        16: "solah", 17: "satrah", 18: "atharah", 19: "unnis", 20: "bees",
        21: "ikkis", 22: "bais", 23: "teis", 24: "chaubis", 25: "pachchis",
        26: "chhabis", 27: "sattais", 28: "atthais", 29: "untis", 30: "tees",
    }
    if n in words: return words[n]
    elif n < 100:
        tens = ["", "", "bees", "tees", "chalis", "pachas", "saath", "sattar", "assi", "nabbe"]
        return tens[n // 10] + " " + words.get(n % 10, "")
    return str(n)

def punjabi_number_word(n: int) -> str:
    """Punjabi number words (transliterated)"""
    if n == 0: return "sifar"
    ones = ["", "ik", "do", "tin", "char", "panj", "chhe", "sat", "ath", "nau", "das"]
    if n <= 10: return ones[n]
    elif n < 20:
        teens = ["", "gyara", "bara", "tera", "chauda", "pandra", "sola", "satara", "athara", "unni"]
        return teens[n - 10]
    elif n < 100:
        tens = ["", "", "vih", "tih", "chali", "panjah", "sath", "sattar", "assi", "nabbe"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def tamil_number_word(n: int) -> str:
    """Tamil number words (transliterated)"""
    if n == 0: return "poojyam"
    ones = ["", "onru", "irandu", "moondru", "naangu", "ainthu", "aaru", "yezhu", "ettu", "onpathu", "pathu"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "pathin" + ones[n - 10]
    elif n < 100:
        tens = ["", "", "irupathu", "muppathu", "naarpathu", "aimpathu", "arupathu", "ezhupathu", "enpathu", "thonnuuru"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def telugu_number_word(n: int) -> str:
    """Telugu number words (transliterated)"""
    if n == 0: return "sunna"
    ones = ["", "okati", "rendu", "mudu", "nalugu", "aidu", "aru", "edu", "enimidi", "tommidi", "padi"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "padak" + ones[n - 10]
    elif n < 100:
        tens = ["", "", "iruvai", "muppai", "nalubai", "yabhai", "aravai", "debbai", "enabai", "tombai"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

def tagalog_number_word(n: int) -> str:
    """Tagalog/Filipino number words"""
    if n == 0: return "wala"
    ones = ["", "isa", "dalawa", "tatlo", "apat", "lima", "anim", "pito", "walo", "siyam", "sampu"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "labing" + ones[n - 10]
    elif n < 100:
        tens = ["", "", "dalawampu", "tatlumpu", "apatnapu", "limampu", "animnapu", "pitumpu", "walumpu", "siyamnapu"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " at " + ones[n % 10]
    return str(n)

def malay_number_word(n: int) -> str:
    """Malay number words"""
    if n == 0: return "kosong"
    ones = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan", "sepuluh"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "se" + "belas" if n == 11 else ones[n - 10] + " belas"
    elif n < 100:
        tens = ["", "", "dua puluh", "tiga puluh", "empat puluh", "lima puluh", "enam puluh", "tujuh puluh", "lapan puluh", "sembilan puluh"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " " + ones[n % 10]
    return str(n)

# --- PACIFIC ---

def hawaiian_number_word(n: int) -> str:
    """Hawaiian number words"""
    if n == 0: return "ole"
    ones = ["", "ekahi", "elua", "ekolu", "eha", "elima", "eono", "ehiku", "ewalu", "eiwa", "umi"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "umi kuma" + ones[n - 10]
    elif n < 100:
        tens = ["", "", "iwakalua", "kanakolu", "kanaha", "kanalima", "kanaono", "kanahiku", "kanawalu", "kanaiva"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " kuma" + ones[n % 10]
    return str(n)

def maori_number_word(n: int) -> str:
    """Maori number words"""
    if n == 0: return "kore"
    ones = ["", "tahi", "rua", "toru", "wha", "rima", "ono", "whitu", "waru", "iwa", "tekau"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "tekau ma " + ones[n - 10]
    elif n < 100:
        tens = ["", "", "rua tekau", "toru tekau", "wha tekau", "rima tekau", "ono tekau", "whitu tekau", "waru tekau", "iwa tekau"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " ma " + ones[n % 10]
    return str(n)

def samoan_number_word(n: int) -> str:
    """Samoan number words"""
    if n == 0: return "zero"
    ones = ["", "tasi", "lua", "tolu", "fa", "lima", "ono", "fitu", "valu", "iva", "sefulu"]
    if n <= 10: return ones[n]
    elif n < 20:
        return "sefulu ma le " + ones[n - 10]
    elif n < 100:
        tens = ["", "", "luasefulu", "tolusefulu", "fasefulu", "limasefulu", "onosefulu", "fitusefulu", "valusefulu", "ivasefulu"]
        if n % 10 == 0: return tens[n // 10]
        return tens[n // 10] + " ma le " + ones[n % 10]
    return str(n)

# --- OTHER ---

def yiddish_number_word(n: int) -> str:
    """Yiddish number words (transliterated)"""
    if n == 0: return "nul"
    ones = ["", "eyns", "tsvey", "dray", "fir", "finf", "zeks", "zibn", "akht", "nayn",
            "tsen", "elf", "tsvelf", "draytsen", "fertsen", "fuftsn",
            "zekhtsn", "zibetsn", "akhtsn", "nayntsn"]
    tens = ["", "", "tsvantsik", "draysik", "fertsik", "fuftsik", "zekhtsik", "zibetsik", "akhtsik", "nayntsik"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + " un " + tens[n // 10]
    return str(n)

def luxembourgish_number_word(n: int) -> str:
    """Luxembourgish number words"""
    if n == 0: return "null"
    ones = ["", "eent", "zwee", "drai", "veier", "fonnef", "sechs", "siwen", "aacht", "neng",
            "zeng", "eelef", "zwielef", "draizen", "vierzeng", "fofzeng",
            "siechzeng", "siwwenzeng", "uechtzeng", "nonzeng"]
    tens = ["", "", "zwanzeg", "draiseg", "verzeg", "fofzeg", "sechzeg", "siwwenzeg", "achtzeg", "nonzeg"]
    if n < 20: return ones[n]
    elif n < 100:
        if n % 10 == 0: return tens[n // 10]
        return ones[n % 10] + "an" + tens[n // 10]
    return str(n)


# =============================================================================
# EXTENDED LANGUAGE REGISTRY (60+ languages)
# =============================================================================

EXTENDED_LANGUAGES = {
    # New Romance
    'catalan': catalan_number_word,
    'galician': galician_number_word,

    # New Slavic
    'ukrainian': ukrainian_number_word,
    'bulgarian': bulgarian_number_word,
    'serbian': serbian_number_word,
    'croatian': croatian_number_word,
    'slovak': slovak_number_word,
    'slovenian': slovenian_number_word,
    'macedonian': macedonian_number_word,

    # Baltic
    'lithuanian': lithuanian_number_word,
    'latvian': latvian_number_word,
    'estonian': estonian_number_word,

    # Other European
    'albanian': albanian_number_word,
    'maltese': maltese_number_word,
    'yiddish': yiddish_number_word,
    'luxembourgish': luxembourgish_number_word,
    'afrikaans': afrikaans_number_word,

    # Middle Eastern / Caucasian
    'persian': persian_number_word,
    'georgian': georgian_number_word,
    'armenian': armenian_number_word,
    'azerbaijani': azerbaijani_number_word,
    'kazakh': kazakh_number_word,

    # African
    'yoruba': yoruba_number_word,
    'hausa': hausa_number_word,
    'amharic': amharic_number_word,
    'zulu': zulu_number_word,
    'xhosa': xhosa_number_word,

    # South/Southeast Asian
    'urdu': urdu_number_word,
    'punjabi': punjabi_number_word,
    'tamil': tamil_number_word,
    'telugu': telugu_number_word,
    'tagalog': tagalog_number_word,
    'malay': malay_number_word,

    # Pacific
    'hawaiian': hawaiian_number_word,
    'maori': maori_number_word,
    'samoan': samoan_number_word,
}

if __name__ == "__main__":
    # Test all new languages
    print("Testing extended languages with number 23:")
    print("=" * 60)
    for lang, func in sorted(EXTENDED_LANGUAGES.items()):
        word = func(23)
        letters = sum(1 for c in word if c.isalpha())
        print(f"  {lang:>15}: {word:<25} ({letters} letters)")
