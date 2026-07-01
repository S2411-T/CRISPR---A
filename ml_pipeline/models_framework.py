import os
import abc
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================================
# 1. UNIFIED ABSTRACT MODEL INTERFACE
# =====================================================================
class CRISPRModelInterface(abc.ABC):
    """
    Polymorphic interface enforcing standard training, evaluation,
    and prediction pipelines across both tabular and deep-learning models.
    """
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save(self, directory: str, filename: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, directory: str, filename: str) -> None:
        pass


# =====================================================================
# 2. CLASSICAL TABULAR IMPLEMENTATIONS
# =====================================================================
class CRISPRRandomForest(CRISPRModelInterface):
    def __init__(self, n_estimators=100, max_depth=12, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state, 
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, directory: str, filename: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, filename))

    def load(self, directory: str, filename: str) -> None:
        self.model = joblib.load(os.path.join(directory, filename))


class CRISPRXGBoost(CRISPRModelInterface):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, directory: str, filename: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, filename))

    def load(self, directory: str, filename: str) -> None:
        self.model = joblib.load(os.path.join(directory, filename))


# =====================================================================
# 3. DEEP LEARNING MODEL TOPOLOGIES (PYTORCH)
# =====================================================================
class PyTorchCNNNet(nn.Module):
    """
    1D Convolutional network processing concatenated one-hot 
    and mismatch spatial sequence blocks to detect structural motifs.
    """
    def __init__(self, input_dim=240):
        super().__init__()
        # Input tensor structure: [Batch, Channels=1, Dimension=240]
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # Linear projections post down-sampling (240 / 2 = 120 features)
        self.fc1 = nn.Linear(64 * 120, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PyTorchTransformerNet(nn.Module):
    """
    Self-Attention Bio-Transformer capturing long-range mismatch spatial dependencies 
    across the active gRNA-DNA cross-examination field.
    """
    def __init__(self, input_dim=240, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        # Project 12 spatial nucleotide dimensions to transformer hidden space
        self.input_projection = nn.Linear(12, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        # Reshape flat array back into token blocks: [Batch, 20 sequence items, 12 features]
        x = x.view(x.size(0), 20, 12)
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(2)
        return self.fc(x)


class DeepLearningWrapper(CRISPRModelInterface):
    """
    Standard Scikit-Learn compliance wrapper handling PyTorch network topologies.
    """
    def __init__(self, network_type="CNN", epochs=5, lr=0.001, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_type = network_type
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.network = PyTorchCNNNet() if network_type == "CNN" else PyTorchTransformerNet()
        self.network.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.network.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.network(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, directory: str, filename: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(directory, filename))

    def load(self, directory: str, filename: str) -> None:
        self.network.load_state_dict(torch.load(os.path.join(directory, filename), map_location=self.device))
        self.network.eval()


# =====================================================================
# 4. ARCHITECTURAL MODEL REGISTRY
# =====================================================================
class CRISPRModelRegistry:
    """
    Central lookup hub managing architectural deployment configurations.
    """
    @staticmethod
    def get_model(architecture_name: str) -> CRISPRModelInterface:
        registry = {
            "Random Forest": CRISPRRandomForest(),
            "XGBoost": CRISPRXGBoost(),
            "CNN": DeepLearningWrapper(network_type="CNN"),
            "Transformer": DeepLearningWrapper(network_type="Transformer")
        }
        if architecture_name not in registry:
            raise ValueError(f"Architecture configuration '{architecture_name}' is not supported.")
        return registry[architecture_name]