"""
Builds the RAG retrieval index: embeds a sample of known-malicious URLs from
the training set using the trained CNN's learned representation (its 64-dim
pooled conv features), and stores them alongside their threat type and the
original URL so we can retrieve "similar known threats" at explanation time.

This deliberately reuses the CNN's own representation rather than a separate
embedding model -- it means "similar" is defined the same way the classifier
itself judges URLs, and it means no second model needs training or serving.
"""
import numpy as np
import pandas as pd
import torch

from cnn_model import URLCharCNN, encode_url

DATA_PATH = '../data/malicious_phish.csv'
INDEX_OUT = 'rag_embeddings.npy'
META_OUT = 'rag_metadata.csv'

# Cap per class so the index stays fast to search and small on disk; a random
# sample is representative enough for "similar known pattern" explanations.
SAMPLES_PER_CLASS = {'phishing': 8000, 'malware': 8000, 'defacement': 6000}

print('Loading dataset...')
df = pd.read_csv(DATA_PATH).dropna(subset=['url', 'type'])
df = df[df['type'] != 'benign']
df = df.drop_duplicates(subset='url')  # avoid an exact-duplicate URL filling multiple top-k slots

sampled = []
for cls, n in SAMPLES_PER_CLASS.items():
    sub = df[df['type'] == cls]
    n = min(n, len(sub))
    sampled.append(sub.sample(n, random_state=42))
sample_df = pd.concat(sampled, ignore_index=True)
print(f'Sampled {len(sample_df)} malicious URLs for the RAG index: '
      f'{sample_df["type"].value_counts().to_dict()}')

model = URLCharCNN()
model.load_state_dict(torch.load('cnn_model.pt', map_location='cpu'))
model.eval()

BATCH = 512
embeddings = []
urls = sample_df['url'].astype(str).tolist()
with torch.no_grad():
    for i in range(0, len(urls), BATCH):
        batch_urls = urls[i:i + BATCH]
        ids = torch.tensor([encode_url(u) for u in batch_urls], dtype=torch.long)
        emb = model.embed(ids).numpy()
        embeddings.append(emb)
        if i % (BATCH * 10) == 0:
            print(f'  embedded {i}/{len(urls)}')

embeddings = np.vstack(embeddings).astype(np.float32)
# L2-normalize so cosine similarity == dot product at query time (cheap)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
embeddings = embeddings / norms

np.save(INDEX_OUT, embeddings)
sample_df[['url', 'type']].to_csv(META_OUT, index=False)
print(f'Saved {embeddings.shape} embeddings to {INDEX_OUT} and metadata to {META_OUT}')
