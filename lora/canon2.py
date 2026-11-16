"""Decoder v2 — fixes found 2026-11-12.

v1 bugs:
  * base64 regex required {8,} chars BEFORE padding, so every short word
    (<=5 source chars) silently failed to decode.
  * the _unleet fallback appended up to 400 chars of the WHOLE string
    whenever any digit looked leet-ish, injecting noise into every
    base64/decimal prompt.
  * callers capped the decoded view at 500 chars while decimal encoding
    expands text 2.8x, so the recovered payload was truncated away.
"""
import re, base64
from canon import canonicalize   # unchanged

_NUM = r'(?:3[2-9]|[4-9]\d|1[0-1]\d|12[0-6])'
DEC  = re.compile(r'\b(?:' + _NUM + r'\b[ ,]+){1,}' + _NUM + r'\b')
HEXE = re.compile(r'(?:\\x[0-9a-fA-F]{2}){3,}')
UESC = re.compile(r'(?:\\u[0-9a-fA-F]{4}){3,}')
# FIX: allow 4+ chars (base64 of a 3-char word is 4 chars) and validate by decoding.
B64  = re.compile(r'\b[A-Za-z0-9+/]{4,}={0,2}')

def _try_b64(t):
    if len(t) % 4 or len(t) < 4:
        return None
    try:
        d = base64.b64decode(t, validate=True).decode('utf-8')
    except Exception:
        return None
    if not d.isprintable():
        return None
    alpha = sum(c.isalpha() for c in d)
    # require it to look like text, not a coincidental decode
    if alpha < 2 or alpha / max(len(d), 1) < 0.6:
        return None
    return d

def decoded_view2(s):
    found = []
    for m in DEC.finditer(s):
        try:
            found.append(''.join(chr(int(x)) for x in re.findall(r'\d+', m.group())))
        except Exception:
            pass
    for m in HEXE.finditer(s):
        try:
            found.append(bytes.fromhex(m.group().replace('\\x','')).decode('utf-8','ignore'))
        except Exception:
            pass
    for m in UESC.finditer(s):
        try:
            found.append(m.group().encode().decode('unicode_escape'))
        except Exception:
            pass
    for m in B64.finditer(s):
        d = _try_b64(m.group())
        if d:
            found.append(d)
    return ' '.join(found)
