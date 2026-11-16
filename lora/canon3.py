"""Preprocessing v3 = canonicalize -> decode in place -> targeted unleet (appended).

Pipeline order matters: decode BEFORE any truncation, so a window holds
decoded content rather than expanded digits.
"""
import re
from canon import canonicalize
from decode_inplace import decode_inplace

_LEET = {'4':'a','3':'e','1':'i','0':'o','5':'s','7':'t','@':'a','$':'s'}
_TOK = re.compile(r'(?<![A-Za-z0-9])[A-Za-z0-9@$]{3,}(?![A-Za-z0-9])')
# real identifiers that must never be "unleeted"
_KEEP = re.compile(r'^(?:x86|x64|ipv[46]|md5|sha\d+|base64|utf-?8|win\d+|kernel32|python3|log4j|aes\d+|rc4|v\d+)$', re.I)

def _leet_tokens(s):
    out = []
    for m in _TOK.finditer(s):
        w = m.group()
        if _KEEP.match(w):
            continue
        letters = sum(c.isalpha() for c in w)
        leet = sum(c in _LEET for c in w)
        # leet = digits/symbols interleaved INSIDE a letter word
        if letters >= 1 and leet >= 1 and len(w) >= 4 and not w.isdigit() and re.search(r'[A-Za-z][0-9@$]|[0-9@$][A-Za-z]', w):
            out.append(''.join(_LEET.get(c, c) for c in w).lower())
    return out

def preprocess(s):
    s = decode_inplace(canonicalize(s))
    toks = _leet_tokens(s)
    return s + (' | LEET | ' + ' '.join(toks) if toks else '')
