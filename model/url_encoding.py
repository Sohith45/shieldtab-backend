"""
Character-level URL encoding for the CNN -- kept dependency-free (no torch)
so the serving path (ensemble.py) can use it without triggering torch's
~460MB import cost. cnn_model.py (used only for training) imports from here
too, so there's a single source of truth for the vocabulary/encoding.
"""
import re

VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 reserved for padding/unknown
VOCAB_SIZE = len(VOCAB) + 1
MAX_LEN = 200  # URLs longer than this are truncated

SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')


def encode_url(url: str, max_len: int = MAX_LEN):
    # Strip any leading scheme first: this dataset records schemes far more
    # often for malicious URLs than benign ones (collection artifact, not a
    # real signal) -- see features.py for the same fix on the XGBoost side.
    url = SCHEME_RE.sub('', url.strip())
    ids = [CHAR_TO_IDX.get(c, 0) for c in url[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return ids
