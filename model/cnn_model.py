"""
Lightweight character-level CNN for malicious URL detection.
Operates directly on the raw URL string (catches obfuscation/typosquatting
patterns that lexical features can miss), kept small for real-time inference.

Only used for training/export -- serving imports url_encoding.py directly
to avoid pulling torch's ~460MB import cost into the deployed backend.
"""
import torch
import torch.nn as nn

from url_encoding import VOCAB, CHAR_TO_IDX, VOCAB_SIZE, MAX_LEN, SCHEME_RE, encode_url  # noqa: F401


class URLCharCNN(nn.Module):
    """Small char-CNN: embedding -> 2 conv blocks -> global max pool -> FC.
    Kept lightweight (~200K params) so CPU inference stays under ~5ms/URL.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_classes=2, max_len=MAX_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, max_len) long tensor of char ids
        e = self.embedding(x)              # (batch, max_len, embed_dim)
        e = e.permute(0, 2, 1)              # (batch, embed_dim, max_len)
        c = self.relu(self.bn1(self.conv1(e)))
        c = self.relu(self.bn2(self.conv2(c)))
        pooled = self.pool(c).squeeze(-1)   # (batch, 64)
        h = self.relu(self.fc1(pooled))
        h = self.dropout(h)
        out = self.fc2(h)                   # (batch, num_classes) logits
        return out

    def embed(self, x):
        """Return the 64-dim pooled conv representation (pre-classification-head).
        Used as the retrieval vector for the RAG similar-threats index -- it's
        the same learned URL representation the classifier itself relies on,
        so "similar" here means "similar in the way the model actually sees
        URLs", not just superficial string similarity.
        """
        e = self.embedding(x)
        e = e.permute(0, 2, 1)
        c = self.relu(self.bn1(self.conv1(e)))
        c = self.relu(self.bn2(self.conv2(c)))
        pooled = self.pool(c).squeeze(-1)
        return pooled
