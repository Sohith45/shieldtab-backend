"""
RAG explanation layer: for a URL already flagged malicious by the ensemble,
retrieves the most similar known-malicious URLs (via the CNN's own learned
embedding space) and combines that with a rule-based read of which lexical
signals fired, to produce a grounded, human-readable explanation.

Deliberately template-based rather than open-ended LLM generation: it's
faster (no generation latency), can't hallucinate a reason that isn't
actually in the data, and every claim it makes is traceable to either a
retrieved neighbor or a concrete feature value. See ensemble.py's docstring
for the same "explainability without a heavyweight model" philosophy.
"""
import os
import csv
import numpy as np
import onnxruntime as ort

from features import extract_features
from url_encoding import encode_url

MODEL_DIR = os.path.dirname(__file__)
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5  # below this, don't claim a match -- just use feature signals


class RAGExplainer:
    def __init__(self):
        emb_path = os.path.join(MODEL_DIR, 'rag_embeddings.npy')
        meta_path = os.path.join(MODEL_DIR, 'rag_metadata.csv')
        onnx_path = os.path.join(MODEL_DIR, 'cnn_model.onnx')

        self.available = all(os.path.exists(p) for p in [emb_path, meta_path, onnx_path])
        if not self.available:
            return

        self.embeddings = np.load(emb_path)  # (N, 64), already L2-normalized
        with open(meta_path, newline='') as f:
            self.metadata = list(csv.DictReader(f))  # list of {'url':..., 'type':...}
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Grab the CNN's intermediate embedding output too -- need a session
        # that exposes the pooled layer, not just final logits. The exported
        # ONNX graph only has 'logits' as output, so instead we recompute the
        # embedding by re-running the same conv stack via onnxruntime isn't
        # possible without re-exporting. Simplest fix: re-export with both
        # outputs (see export_to_onnx.py) -- assume that's been done.
        self._has_embed_output = 'embedding' in [o.name for o in self.session.get_outputs()]

    def _embed(self, url: str) -> np.ndarray:
        ids = np.array([encode_url(url)], dtype=np.int64)
        outputs = self.session.run(None, {'input_ids': ids})
        # embedding is the second output if the model was exported with it
        emb = outputs[1][0] if len(outputs) > 1 else outputs[0][0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _retrieve_similar(self, url: str, k: int = TOP_K):
        if not self.available or not self._has_embed_output:
            return []
        query_emb = self._embed(url)
        sims = self.embeddings @ query_emb  # cosine similarity (both L2-normalized)
        # Over-fetch then dedupe by URL -- guards against any near-duplicate
        # rows slipping past the index-build-time dedup.
        top_idx = np.argsort(-sims)[:k * 3]
        results = []
        seen_urls = set()
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < SIMILARITY_THRESHOLD:
                break
            row = self.metadata[idx]
            if row['url'] in seen_urls:
                continue
            seen_urls.add(row['url'])
            results.append({'url': row['url'], 'type': row['type'], 'similarity': round(sim, 3)})
            if len(results) >= k:
                break
        return results

    def _feature_reasons(self, url: str) -> list:
        """Rule-based, human-readable reasons grounded directly in feature
        values -- no generation, so nothing here can be fabricated."""
        f = extract_features(url)
        reasons = []
        if f['has_ip']:
            reasons.append("uses a raw IP address instead of a domain name")
        if f['is_shortener']:
            reasons.append("uses a URL-shortening service, which can hide the real destination")
        if f['suspicious_word_count'] >= 2:
            reasons.append("contains multiple account/login-related keywords "
                            "commonly used in phishing (e.g. 'verify', 'secure', 'login')")
        elif f['suspicious_word_count'] == 1:
            reasons.append("contains an account/login-related keyword often used in phishing")
        if f['num_subdomains'] >= 3:
            reasons.append(f"has an unusually high number of subdomains ({f['num_subdomains']})")
        if f['has_hex_encoding']:
            reasons.append("contains hex-encoded characters, sometimes used to obscure the real URL")
        if f['num_params'] >= 4:
            reasons.append(f"has an unusually large number of query parameters ({f['num_params']})")
        if f['entropy_url'] > 4.5:
            reasons.append("has unusually random-looking characters in the URL")
        if f['host_length'] > 40:
            reasons.append("has an unusually long domain name")
        return reasons

    def explain(self, url: str, threat_type: str) -> dict:
        similar = self._retrieve_similar(url)
        reasons = self._feature_reasons(url)

        if similar:
            best = similar[0]
            summary = (f"{int(best['similarity']*100)}% similar to a known {best['type']} URL pattern"
                       + (f". Also, this URL {'; '.join(reasons)}." if reasons else "."))
        elif reasons:
            summary = f"Flagged as {threat_type}: this URL {'; '.join(reasons)}."
        else:
            summary = (f"Flagged as {threat_type} by the detection model, though no single "
                       "dominant signal stood out -- the classification is based on a "
                       "combination of subtle lexical and structural patterns.")

        return {
            'summary': summary,
            'similar_threats': similar,
            'signals': reasons,
        }


_explainer_instance = None


def get_explainer() -> RAGExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = RAGExplainer()
    return _explainer_instance
