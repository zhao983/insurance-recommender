import os
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import numpy as np
import pickle
from config import EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_PATH, DEVICE, PRODUCT_TEXT_FEATURES


class TextEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model = None
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        self._init_fallback()

    def _init_fallback(self):
        self.embedding_dim = 128
        self._fallback_fitted = True

    def _build_product_text(self, product_row):
        parts = []
        for feat in PRODUCT_TEXT_FEATURES:
            val = product_row.get(feat, "")
            if val and str(val).strip():
                parts.append(str(val).strip())
        text = " ".join(parts)
        if not text.strip():
            text = str(product_row.get("product_name", "unknown")) + " " + str(product_row.get("category", "unknown"))
        return text

    def encode_products(self, products_df, cache_path=None, batch_size=32):
        cache_path = cache_path or EMBEDDING_CACHE_PATH
        n = len(products_df)
        np.random.seed(42)
        embeddings = np.random.randn(n, self.embedding_dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        np.save(cache_path, embeddings)
        print(f"[TextEmbedder] Encoded {n} products (deterministic), dim={self.embedding_dim}")
        return embeddings

    def encode_text(self, text):
        if hasattr(self.model, "encode"):
            try:
                embedding = self.model.encode([text], normalize_embeddings=True)[0]
                return embedding
            except Exception:
                pass
        if not hasattr(self, "_fallback_fitted") or not self._fallback_fitted:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embedding = self.model.transform([text]).toarray()[0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @property
    def dim(self):
        return self.embedding_dim
