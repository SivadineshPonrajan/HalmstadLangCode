"""In-place decoding: substitute encoded spans with their plaintext, in position.
This is the 'just convert it normally' approach."""
import re, base64
from canon2 import DEC, HEXE, UESC, B64, _try_b64

def _dec_sub(m):
    try: return ''.join(chr(int(x)) for x in re.findall(r'\d+', m.group()))
    except Exception: return m.group()
def _hex_sub(m):
    try: return bytes.fromhex(m.group().replace('\\x','')).decode('utf-8','ignore')
    except Exception: return m.group()
def _uesc_sub(m):
    try: return m.group().encode().decode('unicode_escape')
    except Exception: return m.group()
def _b64_sub(m):
    d = _try_b64(m.group()); return d if d else m.group()

def decode_inplace(s):
    s = DEC.sub(_dec_sub, s)
    s = HEXE.sub(_hex_sub, s)
    s = UESC.sub(_uesc_sub, s)
    s = B64.sub(_b64_sub, s)
    return s
