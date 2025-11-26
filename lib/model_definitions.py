import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# # # classes
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
# # define neural network
class DeeperNeuNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeeperNeuNet, self).__init__()
        self.layer1 = nn.Linear(input_size, int(input_size / 2))
        self.layer2 = nn.Linear(int(input_size / 2), 16)
        self.output_layer = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# # # # EXPLAINER
# # # (MLP)    
class ModelWrapper:
    def __init__(self, model):
        self.model = model.eval()  # eval mode for SHAP
        self.device = next(model.parameters()).device

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)

        return probs.cpu().numpy()