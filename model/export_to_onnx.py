"""
Exports the trained PyTorch CNN (cnn_model.pt) to ONNX format (cnn_model.onnx)
for serving. Run this once after training/retraining the CNN.

Why: importing torch costs ~460MB of RAM before a single model is even
loaded, which alone exceeds free-tier hosting limits (e.g. Render's 512MB).
onnxruntime costs ~30MB. Training still needs PyTorch (autograd, DataLoader,
etc.) so train_cnn.py is unchanged -- only the served artifact changes.

Exports BOTH the classification logits AND the 64-dim pooled embedding from
a single forward pass, so one ONNX model serves both the malicious/benign
prediction (ensemble.py) and the RAG similar-threats retrieval (rag_explain.py)
without needing two separate exports or a second inference pass.
"""
import torch
import torch.nn as nn
from cnn_model import URLCharCNN, MAX_LEN


class ExportWrapper(nn.Module):
    """Thin wrapper so a single forward() call returns both outputs --
    ONNX export traces exactly what forward() returns."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding = self.model.embed(x)
        h = self.model.relu(self.model.fc1(embedding))
        h = self.model.dropout(h)
        logits = self.model.fc2(h)
        return logits, embedding


model = URLCharCNN()
model.load_state_dict(torch.load('cnn_model.pt', map_location='cpu'))
model.eval()

wrapper = ExportWrapper(model)
wrapper.eval()

dummy_input = torch.zeros((1, MAX_LEN), dtype=torch.long)

torch.onnx.export(
    wrapper, dummy_input, 'cnn_model.onnx',
    input_names=['input_ids'],
    output_names=['logits', 'embedding'],
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'logits': {0: 'batch_size'},
                  'embedding': {0: 'batch_size'}},
    opset_version=17,
    dynamo=False,
)
print('Exported cnn_model.onnx with logits + embedding outputs')
