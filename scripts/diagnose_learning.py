import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fedids_bench.config import load_config
from fedids_bench.data.loaders import get_dataset_loader
from fedids_bench.models.mlp import SmallMLP
from fedids_bench.federated.trainer import NumPyDataset
from fedids_bench.evaluation.classification import compute_classification_metrics

def diagnose():
    cfg = load_config("configs/smoke.yaml")
    loader = get_dataset_loader(cfg.dataset.name)
    dataset = loader.load(cfg.dataset, seed=cfg.seed)

    train_mask = (dataset.split == 0)
    test_mask = (dataset.split == 2)

    X_train, y_train = dataset.X[train_mask], dataset.y[train_mask]
    X_test, y_test = dataset.X[test_mask], dataset.y[test_mask]

    print("Y train class distribution:", {int(c): int(np.sum(y_train == c)) for c in np.unique(y_train)})
    print("Y test class distribution:", {int(c): int(np.sum(y_test == c)) for c in np.unique(y_test)})

    model = SmallMLP(n_features=20, n_classes=4, hidden_dims=[32, 16], seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Test Adam vs SGD
    criterion = nn.CrossEntropyLoss()

    train_dataset = NumPyDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("\n--- CENTRALIZED ADAM TRAINING LOGS ---")
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test, dtype=torch.float32))
            test_preds = torch.argmax(test_logits, dim=1).numpy()
        
        metrics = compute_classification_metrics(y_test, test_preds)
        print(f"Epoch {epoch:02d} | Train Loss: {total_loss:.4f} | Test Acc: {metrics['accuracy']:.4f} | Binary F1: {metrics['f1']:.4f} | Macro F1: {metrics['multiclass']['macro_f1']:.4f}")

if __name__ == "__main__":
    diagnose()
