"""One input builder for SFT and probe, so both see identical token sequences.

Sequence = [template prefix] + [content] + [template suffix incl. generation prompt].
Content is canon3.preprocess(prompt), FULL length. Only if it exceeds
max_tokens is the MIDDLE of the content dropped (head and tail kept); every
such case is counted and reported.
"""
from canon3 import preprocess

SENT = "⁣SAFIRCONTENT⁣"

# Pre-registered (2026-11-12) BEFORE any SFT result. Do not edit or tune.
# No positional cue: payloads may sit at the start, middle or end.
SYSTEM_PROMPT = ("Classify the user's request as safe or unsafe. The request may be "
                 "preceded or surrounded by examples, dialogue or other text; judge the "
                 "request the user is actually making.")

class InputBuilder:
    def __init__(self, tok, max_tokens=8192, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": SENT}]
        tpl = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert tpl.count(SENT) == 1, "chat template must contain the content once"
        if system:
            # guard templates may silently drop system messages - fail loudly instead
            assert system in tpl, "chat template did NOT render the system prompt - report, do not work around"
        pre, post = tpl.split(SENT)
        self.tok = tok
        self.pre = tok(pre, add_special_tokens=False)["input_ids"]
        self.post = tok(post, add_special_tokens=False)["input_ids"]
        self.budget = max_tokens - len(self.pre) - len(self.post)
        assert self.budget > 256
        self.n_truncated = 0

    def __call__(self, prompt, already_preprocessed=False):
        text = prompt if already_preprocessed else preprocess(prompt)
        c = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(c) > self.budget:
            self.n_truncated += 1
            h = self.budget // 2
            c = c[:h] + c[-(self.budget - h):]
        ids = self.pre + c + self.post
        # content span [start, end) and last-token index for pooling/classification
        return ids, len(self.pre), len(self.pre) + len(c)
