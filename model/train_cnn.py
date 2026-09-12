"""
Trains the lightweight char-CNN on raw URL strings.
Resumable: saves a checkpoint every N batches and at each epoch end, and
picks up where it left off if re-run. This sandbox doesn't reliably keep
long background processes alive between tool calls, so this script is
designed to be invoked repeatedly (each run does a bounded slice of work)
until training is complete.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from cnn_model import URLCharCNN, encode_url, MAX_LEN

DATA_PATH = '../data/malicious_phish.csv'
CHECKPOINT_PATH = 'cnn_checkpoint.pt'
FINAL_MODEL_PATH = 'cnn_model.pt'
DEVICE = 'cpu'
EPOCHS = 3
BATCH_SIZE = 512
CHECKPOINT_EVERY_N_BATCHES = 50  # frequent, since this tool's exec limit can kill us anytime

print('Using device:', DEVICE)
print('Loading dataset...')
df = pd.read_csv(DATA_PATH).dropna(subset=['url', 'type'])

try:
    aug = pd.read_csv('../data/synthetic_benign.csv')
    aug_oversampled = pd.concat([aug] * 60, ignore_index=True)
    df = pd.concat([df, aug_oversampled], ignore_index=True)
    print(f'Added {len(aug_oversampled)} oversampled benign-with-params rows')
except FileNotFoundError:
    print('No synthetic_benign.csv found, skipping augmentation')

df['is_malicious'] = (df['type'] != 'benign').astype(int)
urls = df['url'].astype(str).values
labels = df['is_malicious'].values

train_urls, test_urls, y_train, y_test = train_test_split(
    urls, labels, test_size=0.15, random_state=42, stratify=labels
)


class URLDataset(Dataset):
    def __init__(self, urls, labels):
        self.urls = urls
        self.labels = labels

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        ids = encode_url(self.urls[idx])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


train_ds = URLDataset(train_urls, y_train)
test_ds = URLDataset(test_urls, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
batches_per_epoch = len(train_loader)

model = URLCharCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
start_batch = 0

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch']
    start_batch = ckpt['batch']
    print(f'Resuming from checkpoint: epoch {start_epoch+1}, batch {start_batch}/{batches_per_epoch}')
else:
    print('No checkpoint found, starting fresh')

if start_epoch >= EPOCHS:
    print('Training already complete per checkpoint. Skipping to final save/eval.')
else:
    t_start = time.time()
    done_early = False
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        skip_to = start_batch if epoch == start_epoch else 0
        for i, (x, y) in enumerate(train_loader):
            if i < skip_to:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'  epoch {epoch+1} batch {i}/{batches_per_epoch} loss={loss.item():.4f} '
                      f'elapsed={time.time()-t_start:.0f}s', flush=True)

            if i % CHECKPOINT_EVERY_N_BATCHES == 0 and i > 0:
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'batch': i + 1,
                }, CHECKPOINT_PATH)
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'batch': 0,
        }, CHECKPOINT_PATH)
        print(f'Epoch {epoch+1}/{EPOCHS} complete. Checkpoint saved.')

    if done_early:
        exit(0)

print('\nAll epochs complete. Running evaluation...')
model.eval()
all_preds, all_probs, all_true = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_true.extend(y.numpy())

print('\n--- CNN classification report ---')
print(classification_report(all_true, all_preds, target_names=['benign', 'malicious']))
print('ROC-AUC:', roc_auc_score(all_true, all_probs))

torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f'Saved {FINAL_MODEL_PATH}')
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print('Removed checkpoint (training complete)')
