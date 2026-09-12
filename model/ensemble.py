"""
Combines the XGBoost (lexical/structural) and CNN (char-sequence) models
into a single verdict. Weighted average of probabilities, tunable.

Also applies a trusted-domain allowlist pre-filter (top ~100K sites by
Chrome UX Report traffic) before running the ML models. Pure lexical/
structural features can't reliably tell a real login page (e.g.
accounts.google.com/signin) from a phishing page mimicking that same
structure -- both look "login-like" in the feature space. A small,
well-known-domain allowlist closes that gap cheaply without needing a
full reputation/RAG system.

The CNN runs via ONNX Runtime rather than PyTorch at inference time --
importing torch alone costs ~460MB of RAM (vs ~30MB for onnxruntime),
which is the difference between fitting in a free-tier host's 512MB
limit or not. Training still uses PyTorch (see train_cnn.py); only the
served model is exported to ONNX (see export_to_onnx.py).
"""
import os
import re
import sys
import numpy as np
import xgboost as xgb
import joblib
import onnxruntime as ort
from urllib.parse import urlparse

sys.path.append(os.path.dirname(__file__))
from features import extract_features, FEATURE_NAMES, SCHEME_RE
from url_encoding import encode_url

MODEL_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(MODEL_DIR, '..', 'data')

XGB_WEIGHT = 0.6
CNN_WEIGHT = 0.4
MALICIOUS_THRESHOLD = 0.5

MULTI_PART_TLDS = {'co.uk', 'com.au', 'co.in', 'co.jp', 'com.br', 'co.za',
                    'com.mx', 'co.kr', 'com.tr', 'co.nz', 'co.id'}


def registrable_domain(host: str) -> str:
    parts = host.split('.')
    if len(parts) <= 2:
        return host
    last_two = '.'.join(parts[-2:])
    last_three = '.'.join(parts[-3:])
    if last_two in MULTI_PART_TLDS:
        return last_three
    return last_two


def _load_domain_set(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


class URLEnsemble:
    def __init__(self):
        self.xgb_binary = xgb.XGBClassifier()
        self.xgb_binary.load_model(os.path.join(MODEL_DIR, 'xgb_binary.json'))

        self.xgb_multi = xgb.XGBClassifier()
        self.xgb_multi.load_model(os.path.join(MODEL_DIR, 'xgb_multiclass.json'))

        self.label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))

        self.cnn_session = None
        onnx_path = os.path.join(MODEL_DIR, 'cnn_model.onnx')
        if os.path.exists(onnx_path):
            self.cnn_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.cnn_available = True
        else:
            self.cnn_available = False

        self.trusted_domains = _load_domain_set('trusted_domains.txt')
        self.trusted_hosts = _load_domain_set('trusted_hosts.txt')
        print(f'Loaded {len(self.trusted_domains)} trusted domains, '
              f'{len(self.trusted_hosts)} trusted hosts')

    def _is_trusted(self, url: str) -> bool:
        if not self.trusted_domains:
            return False
        stripped = SCHEME_RE.sub('', url.strip())
        if not SCHEME_RE.match(url):
            parse_target = 'http://' + url
        else:
            parse_target = url
        try:
            host = urlparse(parse_target).netloc.split(':')[0].lower()
        except Exception:
            return False
        if not host:
            return False
        if host in self.trusted_hosts:
            return True
        return registrable_domain(host) in self.trusted_domains

    def _xgb_proba(self, url: str):
        feats = extract_features(url)
        row = np.array([[feats[f] for f in FEATURE_NAMES]])
        return float(self.xgb_binary.predict_proba(row)[0][1])

    def _xgb_threat_type(self, url: str):
        feats = extract_features(url)
        row = np.array([[feats[f] for f in FEATURE_NAMES]])
        pred = self.xgb_multi.predict(row)[0]
        return self.label_encoder.inverse_transform([pred])[0]

    def _cnn_proba(self, url: str):
        if not self.cnn_available:
            return None
        ids = np.array([encode_url(url)], dtype=np.int64)
        logits = self.cnn_session.run(['logits'], {'input_ids': ids})[0][0]
        exp = np.exp(logits - logits.max())  # numerically stable softmax
        proba = exp / exp.sum()
        return float(proba[1])

    def predict(self, url: str) -> dict:
        if self._is_trusted(url):
            return {
                'url': url,
                'is_malicious': False,
                'confidence': 0.0,
                'threat_type': 'benign',
                'xgb_score': None,
                'cnn_score': None,
                'trusted_domain': True,
            }

        xgb_p = self._xgb_proba(url)
        cnn_p = self._cnn_proba(url)

        if cnn_p is not None:
            final_score = XGB_WEIGHT * xgb_p + CNN_WEIGHT * cnn_p
        else:
            final_score = xgb_p  # fall back to XGBoost alone if CNN not loaded yet

        is_malicious = final_score >= MALICIOUS_THRESHOLD
        threat_type = self._xgb_threat_type(url) if is_malicious else 'benign'

        return {
            'url': url,
            'is_malicious': bool(is_malicious),
            'confidence': round(final_score, 4),
            'threat_type': threat_type,
            'xgb_score': round(xgb_p, 4),
            'cnn_score': round(cnn_p, 4) if cnn_p is not None else None,
            'trusted_domain': False,
        }


# Module-level singleton so FastAPI doesn't reload models per request
_ensemble_instance = None


def get_ensemble() -> URLEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = URLEnsemble()
    return _ensemble_instance
