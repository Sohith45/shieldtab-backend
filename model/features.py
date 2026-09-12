"""
Lexical + structural feature extraction for the URL.
Used by the XGBoost fast-path classifier.
"""
import re
import math
from urllib.parse import urlparse
from collections import Counter

SUSPICIOUS_WORDS = [
    'login', 'signin', 'verify', 'account', 'update', 'secure', 'banking',
    'confirm', 'password', 'billing', 'suspend', 'urgent', 'click', 'webscr',
    'paypal', 'ebay', 'amazon', 'security', 'alert', 'validate', 'wallet'
]

SHORTENERS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'shorte.st', 'rebrand.ly', 'cutt.ly', 'tiny.cc'
}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def has_ip_address(host: str) -> bool:
    return bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host or ''))


SCHEME_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')


def extract_features(raw_url: str) -> dict:
    """Return a flat dict of numeric features for one URL string.

    IMPORTANT: this dataset's benign URLs are recorded almost always without
    an explicit http(s):// prefix, while malicious ones almost always include
    one (a data-collection artifact, not a real signal) — that leak inflates
    slash/length/entropy counts and a trivial "has_https" feature to near-100%
    "accuracy" that doesn't generalize. To avoid learning that artifact, every
    character-count-based feature below is computed on the URL with any
    leading scheme stripped, and no explicit "has_https" feature is used.
    """
    url = raw_url.strip()
    stripped = SCHEME_RE.sub('', url)  # used for all count/length/token features

    # normalize so urlparse works even without a scheme
    if not SCHEME_RE.match(url):
        parse_target = 'http://' + url
    else:
        parse_target = url

    try:
        parsed = urlparse(parse_target)
    except Exception:
        parsed = urlparse('http://invalid')

    host = parsed.netloc.split(':')[0].lower()
    path = parsed.path or ''
    query = parsed.query or ''
    full = stripped.lower()

    tokens = re.split(r'[/\-_.?=&]', stripped)
    tokens = [t for t in tokens if t]

    features = {
        'url_length': len(stripped),
        'host_length': len(host),
        'path_length': len(path),
        'query_length': len(query),
        'num_dots': stripped.count('.'),
        'num_hyphens': stripped.count('-'),
        'num_underscores': stripped.count('_'),
        'num_slashes': stripped.count('/'),
        'num_question_marks': stripped.count('?'),
        'num_equal_signs': stripped.count('='),
        'num_at_signs': stripped.count('@'),
        'num_ampersands': stripped.count('&'),
        'num_digits': sum(c.isdigit() for c in stripped),
        'digit_ratio': sum(c.isdigit() for c in stripped) / max(len(stripped), 1),
        'num_params': len(query.split('&')) if query else 0,
        'num_subdomains': max(host.count('.') - 1, 0),
        'has_ip': int(has_ip_address(host)),
        'has_port': int(':' in parsed.netloc),
        'is_shortener': int(host in SHORTENERS),
        'suspicious_word_count': sum(w in full for w in SUSPICIOUS_WORDS),
        'entropy_url': shannon_entropy(stripped),
        'entropy_host': shannon_entropy(host),
        'avg_token_length': sum(len(t) for t in tokens) / max(len(tokens), 1),
        'max_token_length': max((len(t) for t in tokens), default=0),
        'num_tokens': len(tokens),
        'tld_length': len(host.split('.')[-1]) if '.' in host else 0,
        'has_double_slash_redirect': int('//' in path),
        'has_at_symbol': int('@' in stripped),
        'has_hex_encoding': int('%' in stripped),
    }
    return features


FEATURE_NAMES = list(extract_features('http://example.com/test?x=1').keys())
