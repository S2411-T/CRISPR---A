import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import GenomicDataLoader
from models_framework import CRISPRModelRegistry

def train_and_benchmarking_suite() -> dict:
    print("Executing Central Multi-Architecture Benchmarking Suite...")
    loader = GenomicDataLoader()
    df = loader.generate_synthetic_benchmark(n_samples=3000)
    X, y = loader.prepare_features(df)
    
    architectures = ["Random Forest", "XGBoost", "CNN", "Transformer"]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    global_report = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for arch in architectures:
        print(f"\nEvaluating System Topology: {arch}")
        auroc_folds, auprc_folds = [], []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = CRISPRModelRegistry.get_model(arch)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            
            auroc_folds.append(float(roc_auc_score(y_val, probs)))
            auprc_folds.append(float(average_precision_score(y_val, probs)))
            
        mean_auroc = float(np.mean(auroc_folds))
        mean_auprc = float(np.mean(auprc_folds))
        print(f"[{arch}] Validation Complete. Mean AUROC: {mean_auroc:.4f}")
        
        global_report[arch] = {
            "auroc": round(mean_auroc * 100, 2),
            "auprc": round(mean_auprc * 100, 2)
        }
        
        # Train final production instance
        production_model = CRISPRModelRegistry.get_model(arch)
        production_model.fit(X, y)
        safe_filename = f"{arch.lower().replace(' ', '_')}_crispr.pkl"
        production_model.save(models_dir, safe_filename)
        
    # Save the metrics to a performance log file for the app to ingest
    metrics_path = os.path.join(models_dir, 'benchmark_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(global_report, f, indent=4)
        
    print(f"\nRegistry Verification Complete. Benchmark log written to: {metrics_path}")
    return global_report

if __name__ == "__main__":
    train_and_benchmarking_suite()