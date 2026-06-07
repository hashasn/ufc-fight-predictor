import os
import json
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from preprocessor import run_preprocessing


# =========================================================
# Config
# =========================================================

DATA_PATH = "./mayberealdata_enriched.csv"
ARTIFACT_DIR = "artifacts"

BASE_ELO = 1500
ELO_K = 16
ROLLING_WINDOW = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 60
PATIENCE = 8
EARLY_STOP_EPS = 1e-4
RANDOM_SEED = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    """
    Make results more reproducible.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Train/test split
# =========================================================

def chronological_split_3way(paired, feature_cols, test_size=0.2, val_size=0.2, output_dir="."):
    """
    Split by fight_id chronologically into:
        train -> oldest fights
        val   -> middle fights
        test  -> newest fights

    val_size is applied to the pre-test portion only.
    """
    sorted_fights = sorted(paired["fight_id"].unique())
    n_fights = len(sorted_fights)

    test_start = int(n_fights * (1 - test_size))
    pretest_fights = sorted_fights[:test_start]
    test_fights = sorted_fights[test_start:]

    val_start = int(len(pretest_fights) * (1 - val_size))
    train_fights = pretest_fights[:val_start]
    val_fights = pretest_fights[val_start:]

    train_df = paired[paired["fight_id"].isin(train_fights)].copy()
    val_df = paired[paired["fight_id"].isin(val_fights)].copy()
    test_df = paired[paired["fight_id"].isin(test_fights)].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df["result_f"].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df["result_f"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["result_f"].copy()
    # print(test_df.columns)
    test_export = test_df[[
        "fight_id",
        "fighter_f",
        "fighter_o",
        "event_date_f",
        "event_f"
        
    ]].copy()

    test_export = test_export.rename(columns={
        "fighter_f": "Fighter1",
        "fighter_o": "Fighter2"
    })

    test_export = test_export.drop_duplicates(subset=["fight_id"])
    test_fights_path = os.path.join(output_dir, "test_fights_to_scrape.csv")
    test_export.to_csv(test_fights_path, index=False)

    print(f"Saved {len(test_export)} unique test fights to {test_fights_path}")
    

    return (
        train_df, val_df, test_df,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


# =========================================================
# Numeric preprocessor
# =========================================================

def make_numeric_preprocessor():
    """
    Fit this on train only, then reuse on test/future data.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def augment_with_mirrored_fights(X_train, y_train):
    """
    Add mirrored versions of all training rows.

    Since features are directional differences:
        fighter_A - fighter_B

    the mirrored fight is:
        fighter_B - fighter_A = -(fighter_A - fighter_B)

    Labels also flip:
        1 -> 0
        0 -> 1

    Draw / NaN labels should not be present here because preprocessing
    already removed them, but we guard anyway.
    """
    valid_mask = y_train.notna()

    X_base = X_train.loc[valid_mask].copy()
    y_base = y_train.loc[valid_mask].copy()

    X_mirror = -X_base
    y_mirror = 1 - y_base

    X_aug = pd.concat([X_base, X_mirror], ignore_index=True)
    y_aug = pd.concat([y_base, y_mirror], ignore_index=True)

    return X_aug, y_aug
# =========================================================
# Neural network
# =========================================================

class UFCNet(nn.Module):
    """
    Simple feedforward network for binary classification.
    Output is a single logit.
    """
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Linear(input_dim, 64),
            # nn.ReLU(),
            # nn.Dropout(0.30),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(0.20),

            # nn.Linear(32, 1)

            # nn.Linear(input_dim, 32),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Dropout(0.3),

            # nn.Linear(16, 1)
            
            nn.Linear(input_dim, 64),
            
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(model, X_np, y_np, device):
    """
    Evaluate the model on a given dataset.
    Returns accuracy, AUC, log loss, probabilities, predictions.
    """
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        logits = model(X_t).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_np, preds)
    auc = roc_auc_score(y_np, probs)
    ll = log_loss(y_np, probs)

    return acc, auc, ll, probs, preds


# =========================================================
# Training loop
# =========================================================

def train_nn(
    X_train_np,
    y_train_np,
    X_val_np,
    y_val_np,
    X_test_np,
    y_test_np,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-3,
    epochs=60,
    patience=8,
    early_stop_eps=1e-4,
):
    """
    Train a neural network with early stopping based on validation AUC.
    Test set is kept untouched until the very end.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Torch datasets
    Xtr_t = torch.tensor(X_train_np, dtype=torch.float32)
    ytr_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = UFCNet(input_dim=X_train_np.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_auc = -1.0
    best_state = None
    pat = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_ds)

        val_acc, val_auc, val_ll, _, _ = evaluate_model(
            model, X_val_np, y_val_np, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_logloss": val_ll,
        })

        improved = val_auc > best_val_auc + early_stop_eps
        if improved:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            pat = 0
        else:
            pat += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_logloss={val_ll:.4f} | "
            f"pat={pat}/{patience}"
        )

        if pat >= patience:
            print("Early stopping triggered.")
            break

    # Restore best validation model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation
    final_val_acc, final_val_auc, final_val_ll, _, _ = evaluate_model(
        model, X_val_np, y_val_np, device
    )

    # Final evaluation on untouched test set
    final_test_acc, final_test_auc, final_test_ll, final_test_probs, final_test_preds = evaluate_model(
        model, X_test_np, y_test_np, device
    )

    print("\nBest validation metrics:")
    print("Accuracy:", final_val_acc)
    print("AUC:", final_val_auc)
    print("LogLoss:", final_val_ll)

    print("\nFinal test metrics:")
    print("Accuracy:", final_test_acc)
    print("AUC:", final_test_auc)
    print("LogLoss:", final_test_ll)

    return model, history, {
        "val_accuracy": final_val_acc,
        "val_auc": final_val_auc,
        "val_logloss": final_val_ll,
        "accuracy": final_test_acc,
        "auc": final_test_auc,
        "logloss": final_test_ll,
        "probs": final_test_probs,
        "preds": final_test_preds,
        "device": str(device),
    }


# =========================================================
# Save artifacts
# =========================================================

def save_artifacts(
    model,
    preprocessor,
    feature_cols,
    history,
    metrics,
    config,
    artifact_dir="artifacts",
):
    """
    Save everything needed for prediction later.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    # Save torch model weights
    model_path = os.path.join(artifact_dir, "ufc_nn_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save sklearn preprocessor
    preprocessor_path = os.path.join(artifact_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    # Save feature column order
    feature_cols_path = os.path.join(artifact_dir, "feature_cols.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save training history
    history_path = os.path.join(artifact_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save metrics
    # metrics_clean = {
    #     "accuracy": float(metrics["accuracy"]),
    #     "auc": float(metrics["auc"]),
    #     "logloss": float(metrics["logloss"]),
    #     "device": metrics["device"],
    # }
    metrics_clean = {
    "val_accuracy": float(metrics["val_accuracy"]),
    "val_auc": float(metrics["val_auc"]),
    "val_logloss": float(metrics["val_logloss"]),
    "accuracy": float(metrics["accuracy"]),
    "auc": float(metrics["auc"]),
    "logloss": float(metrics["logloss"]),
    "device": metrics["device"],
    }   
    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)

    # Save config metadata
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved artifacts to:", artifact_dir)
    print(" -", model_path)
    print(" -", preprocessor_path)
    print(" -", feature_cols_path)
    print(" -", history_path)
    print(" -", metrics_path)
    print(" -", config_path)


# =========================================================
# Main
# =========================================================

def main(data_path=DATA_PATH, artifact_dir=ARTIFACT_DIR):
    set_seed(RANDOM_SEED)

    # ---------------------------------
    # Load raw data
    # ---------------------------------
    df_raw = pd.read_csv(data_path)
    print("Loaded rows:", len(df_raw))
    print("Loaded columns:", len(df_raw.columns))

    # ---------------------------------
    # Run reusable preprocessing
    # ---------------------------------
    bundle = run_preprocessing(
        df_raw=df_raw,
        base_elo=BASE_ELO,
        elo_k=ELO_K,
        rolling_window=ROLLING_WINDOW,
    )

    paired = bundle["paired"].copy()
    feature_cols = bundle["feature_cols"]

    print("Paired rows:", len(paired))
    print("Feature count:", len(feature_cols))

    # ---------------------------------
    # Train/test split
    # ---------------------------------
    # train_df, test_df, X_train, y_train, X_test, y_test = chronological_split(
    #     paired=paired,
    #     feature_cols=feature_cols,
    #     test_size=TEST_SIZE,
    # )
    (
    train_df, val_df, test_df,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test) = chronological_split_3way(
        paired=paired,
        feature_cols=feature_cols,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        output_dir=artifact_dir,
    )

    # print("Before symmetry augmentation:")
    # print("Train rows:", X_train.shape)
    # print("Test rows:", X_test.shape)
    # print("Unique train fights:", train_df["fight_id"].nunique())
    # print("Unique test fights:", test_df["fight_id"].nunique())

    # overlap = set(train_df["fight_id"]) & set(test_df["fight_id"])
    # print("Fight overlap:", len(overlap))
    print("Before symmetry augmentation:")
    print("Train rows:", X_train.shape)
    print("Val rows:", X_val.shape)
    print("Test rows:", X_test.shape)

    print("Unique train fights:", train_df["fight_id"].nunique())
    print("Unique val fights:", val_df["fight_id"].nunique())
    print("Unique test fights:", test_df["fight_id"].nunique())

    train_val_overlap = set(train_df["fight_id"]) & set(val_df["fight_id"])
    train_test_overlap = set(train_df["fight_id"]) & set(test_df["fight_id"])
    val_test_overlap = set(val_df["fight_id"]) & set(test_df["fight_id"])

    print("Train/Val overlap:", len(train_val_overlap))
    print("Train/Test overlap:", len(train_test_overlap))
    print("Val/Test overlap:", len(val_test_overlap))

    # ---------------------------------
    # Symmetry augmentation on TRAIN only
    # ---------------------------------
    # X_train, y_train = augment_with_mirrored_fights(X_train, y_train)

    # print("\nAfter symmetry augmentation:")
    # print("Augmented train rows:", X_train.shape)
    # print("Augmented train label distribution:")
    # print(y_train.value_counts(dropna=False))

    # ---------------------------------
    # Preprocessor
    # ---------------------------------
    preprocessor = make_numeric_preprocessor()

    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_val_np = y_val.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    # ---------------------------------
    # Train NN
    # ---------------------------------
    # model, history, metrics = train_nn(
    #     X_train_np=X_train_np,
    #     y_train_np=y_train_np,
    #     X_test_np=X_test_np,
    #     y_test_np=y_test_np,
    #     batch_size=BATCH_SIZE,
    #     learning_rate=LEARNING_RATE,
    #     weight_decay=WEIGHT_DECAY,
    #     epochs=EPOCHS,
    #     patience=PATIENCE,
    #     early_stop_eps=EARLY_STOP_EPS,
    # )
    model, history, metrics = train_nn(
        X_train_np=X_train_np,
        y_train_np=y_train_np,
        X_val_np=X_val_np,
        y_val_np=y_val_np,
        X_test_np=X_test_np,
        y_test_np=y_test_np,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        patience=PATIENCE,
        early_stop_eps=EARLY_STOP_EPS,
)

    # ---------------------------------
    # Save artifacts
    # ---------------------------------
    config = {
        "data_path": data_path,
        "base_elo": BASE_ELO,
        "elo_k": ELO_K,
        "rolling_window": ROLLING_WINDOW,
        
        "test_size": TEST_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "random_seed": RANDOM_SEED,
        "model_class": "UFCNet",
        "input_dim": len(feature_cols),
    }

    save_artifacts(
        model=model,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        history=history,
        metrics=metrics,
        config=config,
        artifact_dir=artifact_dir,
    )

    # Save test predictions for analysis
    pred_df = pd.DataFrame({
        "fight_id": test_df["fight_id"].values,
        "Fighter1": test_df["fighter_f"].values,
        "Fighter2": test_df["fighter_o"].values,
        "p_model": metrics["probs"],
        "actual": y_test_np
    })

    test_predictions_path = os.path.join(artifact_dir, "test_predictions.csv")
    pred_df.to_csv(test_predictions_path, index=False)
    print("Saved test predictions to", test_predictions_path)


    
if __name__ == "__main__":
    main()