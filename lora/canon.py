"""Deterministic canonicalizer v2 — conservative, script-aware.

v1 bugs fixed:
  * decimal decoder dropped the final codepoint (regex demanded a trailing separator)
  * homoglyph map rewrote whole-script Cyrillic text (mangled ~10% of corpus)
  * ZWJ/ZWNJ removal broke Devanagari/Bengali conjuncts
  * de-leet mangled technical identifiers (SHA256 -> SHA2s6); now an appended view
"""
import re, unicodedata, base64

# --- zero-width: always-unsafe controls vs. Indic-legitimate joiners -------
ALWAYS_STRIP = dict.fromkeys(map(ord, '​⁠﻿­‪‫‬‭‮'), None)
JOINERS = '‌‍'   # ZWNJ, ZWJ - required in Devanagari/Bengali conjuncts
_LATIN_JOIN = re.compile(f'(?<=[A-Za-z])[{JOINERS}](?=[A-Za-z])')

_HOMO_BACK = {'а':'a','с':'c','е':'e','о':'o','р':'p','х':'x','у':'y','і':'i','ѕ':'s',
              'А':'A','В':'B','С':'C','Е':'E','Н':'H','К':'K','М':'M','О':'O','Р':'P','Т':'T','Х':'X'}
_TOKEN = re.compile(r'\w{2,}', re.UNICODE)

def _is_mixed_script_token(w):
    """True only if a single token mixes Latin with Cyrillic/Greek (= spoofing)."""
    has_lat = has_cyr = False
    for ch in w:
        if not ch.isalpha():
            continue
        try:
            fam = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        if fam == 'LATIN':
            has_lat = True
        elif fam in ('CYRILLIC', 'GREEK'):
            has_cyr = True
    return has_lat and has_cyr

def _fix_homoglyph_token(m):
    w = m.group()
    if not _is_mixed_script_token(w):
        return w                      # genuine Russian / Greek text: leave alone
    return ''.join(_HOMO_BACK.get(c, c) for c in w)

def _strip_combining_on_latin(s):
    out = []
    for ch in unicodedata.normalize('NFD', s):
        if unicodedata.combining(ch) and out:
            try:
                if unicodedata.name(out[-1]).split()[0] == 'LATIN':
                    continue          # drop only marks stacked on Latin bases
            except ValueError:
                pass
        out.append(ch)
    return unicodedata.normalize('NFC', ''.join(out))

_LATIN_SPACING = re.compile(r'(?<=\b[A-Za-z]) (?=[A-Za-z]\b)')

def canonicalize(s):
    s = s.translate(ALWAYS_STRIP)
    s = _LATIN_JOIN.sub('', s)
    s = unicodedata.normalize('NFKC', s)
    s = _TOKEN.sub(_fix_homoglyph_token, s)
    s = _strip_combining_on_latin(s)
    s = _LATIN_SPACING.sub('', s)
    return s

# ----------------------------------------------------------- decoded view
# FIX: final element no longer requires a trailing separator.
_NUM = r'(?:3[2-9]|[4-9]\d|1[0-1]\d|12[0-6])'
DEC  = re.compile(r'\b(?:' + _NUM + r'\b[ ,]+){5,}' + _NUM + r'\b')
HEXE = re.compile(r'(?:\\x[0-9a-fA-F]{2}){3,}')
UESC = re.compile(r'(?:\\u[0-9a-fA-F]{4}){3,}')
B64  = re.compile(r'\b[A-Za-z0-9+/]{8,}={0,2}\b')
_LEET_BACK = {'4':'a','3':'e','1':'i','0':'o','5':'s','7':'t','@':'a','$':'s'}
_LEETTOK = re.compile(r'[A-Za-z0-9@$]{3,}')

def _unleet(s):
    def f(m):
        w = m.group()
        letters = sum(c.isalpha() for c in w)
        digits  = sum(c in _LEET_BACK for c in w)
        if letters >= 2 and 1 <= digits <= letters:
            return ''.join(_LEET_BACK.get(c, c) for c in w)
        return w
    return _LEETTOK.sub(f, s)

def decoded_view(s):
    """Extra evidence appended after the original. Never substitutes it."""
    found = []
    for m in DEC.finditer(s):
        try:
            found.append(''.join(chr(int(x)) for x in re.findall(r'\d+', m.group())))
        except Exception:
            pass
    for m in HEXE.finditer(s):
        try:
            found.append(bytes.fromhex(m.group().replace('\\x', '')).decode('utf-8', 'ignore'))
        except Exception:
            pass
    for m in UESC.finditer(s):
        try:
            found.append(m.group().encode().decode('unicode_escape'))
        except Exception:
            pass
    for m in B64.finditer(s):
        t = m.group()
        if len(t) % 4:
            continue
        try:
            d = base64.b64decode(t, validate=True).decode('utf-8')
            if d.isprintable() and sum(c.isalpha() for c in d) >= 3:
                found.append(d)
        except Exception:
            pass
    ul = _unleet(s)
    if ul != s:
        found.append(ul[:400])
    return ' '.join(found)
