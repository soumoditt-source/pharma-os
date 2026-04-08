import os
import urllib.request
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Constant Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
DATA_PATH = os.path.join(DATA_DIR, "esol_delaney.csv")
MODEL_PATH = os.path.join(DATA_DIR, "pharma_solubility_net.pt")
RF_MODEL_PATH = os.path.join(DATA_DIR, "pharma_rf.pkl")
FP_SIZE = 1024
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def _verbose_logging_enabled() -> bool:
    """Allow inference.py to suppress non-structured stdout during grading."""
    raw_value = os.getenv("PHARMAOS_VERBOSE_LOGS", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _log(message):
    """Emit ASCII-safe console logs across Windows and container shells."""
    if not _verbose_logging_enabled():
        return
    print(str(message).encode("ascii", errors="replace").decode("ascii"))

class PharmaBioactivityNet(nn.Module):
    """
    High-End PyTorch Deep Neural Network for predicting molecular bioactivity/solubility.
    """
    def __init__(self, input_dim=FP_SIZE):
        super(PharmaBioactivityNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

def smiles_to_fp(smiles, n_bits=FP_SIZE):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((n_bits,), dtype=np.float32)
        generator = MORGAN_GENERATOR if n_bits == FP_SIZE else rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((0,), dtype=np.int8)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    except Exception:
        return np.zeros((n_bits,), dtype=np.float32)

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels_list):
        self.smiles = smiles_list
        self.labels = labels_list
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        fp = smiles_to_fp(self.smiles[idx])
        label = self.labels[idx]
        return torch.tensor(fp, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

def prepare_real_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATA_PATH):
        _log("[PharmaOS Engine] Downloading real ESOL ADMET Dataset from DeepChem...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        _log("[PharmaOS Engine] Download complete.")

    smiles_list = []
    labels_list = []
    _log("[PharmaOS Engine] Parsing real clinical data...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles")
            logS_str = row.get("measured log solubility in mols per litre")
            if smi and logS_str:
                try:
                    smiles_list.append(smi)
                    labels_list.append(float(logS_str))
                except ValueError:
                    continue
    _log(f"[PharmaOS Engine] Validated {len(smiles_list)} real molecules.")
    return smiles_list, labels_list

def train_ml_engine():
    """Trains BOTH PyTorch Deep Learning & Scikit-Learn .pkl Ensemble Models"""
    smiles_list, labels_list = prepare_real_data()
    split_idx = int(len(smiles_list) * 0.8)
    
    train_smiles, train_labels = smiles_list[:split_idx], labels_list[:split_idx]
    val_smiles, val_labels = smiles_list[split_idx:], labels_list[split_idx:]
    
    # 1. Train Scikit-Learn Random Forest (.pkl)
    _log("[PharmaOS Engine] Training Scikit-Learn Random Forest Ensemble (.pkl)...")
    X_train = np.array([smiles_to_fp(s) for s in train_smiles])
    y_train = np.array(train_labels)
    # Upgraded Random Forest parameters
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    _log(f"[PharmaOS Engine] Random Forest model saved to {RF_MODEL_PATH}")

    # 2. Train PyTorch Deep Neural Network (.pt)
    train_dataset = MoleculeDataset(train_smiles, train_labels)
    val_dataset = MoleculeDataset(val_smiles, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = PharmaBioactivityNet()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    _log(f"[PharmaOS Engine] Initiating PyTorch training on {device.type.upper()}...")
    EPOCHS = 20
    best_val_loss = float('inf')
    
    for _ in range(EPOCHS):
        model.train()
        for fps, labels in train_loader:
            fps, labels = fps.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(fps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fps, labels in val_loader:
                fps, labels = fps.to(device), labels.to(device)
                outputs = model(fps)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * fps.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    _log(f"[PharmaOS Engine] Training complete. PyTorch weights saved to {MODEL_PATH}")

class MLPredictor:
    """Enterprise-grade Ensemble AI (PyTorch + RF PKL) inference layer."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLPredictor, cls).__new__(cls)
            cls._instance.dl_model = None
            cls._instance.rf_model = None
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls._instance._load_or_train()
        return cls._instance
        
    def _load_or_train(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(RF_MODEL_PATH):
            train_ml_engine()
            
        self.dl_model = PharmaBioactivityNet()
        self.dl_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        self.dl_model.to(self.device)
        self.dl_model.eval()
        
        with open(RF_MODEL_PATH, "rb") as f:
            self.rf_model = pickle.load(f)
            
        _log("[PharmaOS ML] Multi-architecture ensemble (PyTorch + RF) is ready.")
        
    def _mc_dropout_predict(self, tensor_fp, num_passes=5):
        self.dl_model.eval()
        dropout_layers = []
        for module in self.dl_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                dropout_layers.append(module)

        predictions = []
        with torch.no_grad():
            for _ in range(num_passes):
                pred = self.dl_model(tensor_fp).item()
                predictions.append(pred)

        for module in dropout_layers:
            module.eval()

        return float(np.mean(predictions)), float(np.std(predictions))

    def predict_with_uncertainty(self, smiles):
        fp = smiles_to_fp(smiles)
        tensor_fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # DL MC-Dropout Prediction
        dl_mean, dl_std = self._mc_dropout_predict(tensor_fp, num_passes=5)
        
        # RF Individual Tree Predictions
        rf_tree_preds = [tree.predict([fp])[0] for tree in self.rf_model.estimators_]
        rf_mean = float(np.mean(rf_tree_preds))
        rf_std = float(np.std(rf_tree_preds))
        
        # Enterprise-level Ensemble Output
        ensemble_pred = (dl_mean * 0.6) + (rf_mean * 0.4)
        # Combined uncertainty (simplified pooling)
        ensemble_std = (dl_std * 0.6) + (rf_std * 0.4)
        
        # Calculate 95% Confidence Interval (approx 1.96 * std)
        ci_lower = ensemble_pred - (1.96 * ensemble_std)
        ci_upper = ensemble_pred + (1.96 * ensemble_std)
        
        return {
            "prediction": ensemble_pred,
            "uncertainty": ensemble_std,
            "confidence_interval": (ci_lower, ci_upper),
            "dl_components": {"mean": dl_mean, "std": dl_std},
            "rf_components": {"mean": rf_mean, "std": rf_std}
        }

    def predict_solubility(self, smiles):
        return self.predict_with_uncertainty(smiles)["prediction"]
        
    def batch_predict(self, smiles_list):
        predictions = []
        for smi in smiles_list:
            predictions.append(self.predict_solubility(smi))
        return predictions

engine = None

def get_ml_engine():
    global engine
    if engine is None:
        engine = MLPredictor()
    return engine

if __name__ == "__main__":
    train_ml_engine()
