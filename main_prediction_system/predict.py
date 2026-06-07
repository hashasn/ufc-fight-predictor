import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re

from preprocessor import run_preprocessing, clean_stance


# =========================================================
# Config
# =========================================================

DATA_PATH = "./mayberealdata_enriched.csv"
ARTIFACT_DIR = "artifacts"

MODEL_PATH = f"{ARTIFACT_DIR}/ufc_nn_model.pt"
PREPROCESSOR_PATH = f"{ARTIFACT_DIR}/preprocessor.joblib"
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_cols.json"
CONFIG_PATH = f"{ARTIFACT_DIR}/config.json"

INPUT_CSV = "../future-card-prediction/upcoming_new.csv"      # can be compact OR full fight-card format
OUTPUT_CSV = "predictions.csv"


# =========================================================
# Same NN architecture as train.py
# =========================================================

class UFCNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
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
# Artifact loading
# =========================================================

def load_artifacts():
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = UFCNet(input_dim=config["input_dim"])
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, preprocessor, feature_cols, config


# =========================================================
# Helper parsers
# =========================================================

def height_to_inches(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    match = re.search(r"(\d+)'\s*(\d+)", value)
    if not match:
        return np.nan

    feet = int(match.group(1))
    inches = int(match.group(2))
    return feet * 12 + inches


def reach_to_inches(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().replace('"', "")
    if value in ["", "--", "nan", "None"]:
        return np.nan

    digits = "".join(c for c in value if c.isdigit())
    return float(digits) if digits else np.nan


# =========================================================
# CSV format detection + conversion
# =========================================================

def detect_csv_format(df):
    """
    Return:
        'compact'   -> already in prediction-ready format
        'fightcard' -> full UFCStats-style card format
    """
    compact_required = {"fighter_a", "fighter_b", "event_date"}
    fightcard_required = {
        "Fighter1", "Fighter2", "event_date",
        "f1_dob", "f1_height", "f1_reach", "f1_stance",
        "f2_dob", "f2_height", "f2_reach", "f2_stance"
    }

    cols = set(df.columns)

    if compact_required.issubset(cols):
        return "compact"

    if fightcard_required.issubset(cols):
        return "fightcard"

    raise ValueError(
        "Input CSV format not recognized. "
        "Expected either compact columns "
        "['fighter_a', 'fighter_b', 'event_date'] "
        "or fight-card columns like "
        "['Fighter1', 'Fighter2', 'event_date', 'f1_dob', 'f1_height', ...]."
    )


def convert_fightcard_df_to_compact(df):
    """
    Convert full fight-card dataframe into compact prediction format.
    """
    out_df = pd.DataFrame({
        "fighter_a": df["Fighter1"].astype(str).str.strip(),
        "fighter_b": df["Fighter2"].astype(str).str.strip(),
        "event_date": df["event_date"],

        "fighter_a_dob": df["f1_dob"],
        "fighter_a_height_in": df["f1_height"].apply(height_to_inches),
        "fighter_a_reach_in": df["f1_reach"].apply(reach_to_inches),
        "fighter_a_stance": df["f1_stance"].apply(clean_stance),

        "fighter_b_dob": df["f2_dob"],
        "fighter_b_height_in": df["f2_height"].apply(height_to_inches),
        "fighter_b_reach_in": df["f2_reach"].apply(reach_to_inches),
        "fighter_b_stance": df["f2_stance"].apply(clean_stance),
    })

    return out_df


def load_prediction_input(input_csv):
    """
    Load input CSV and return compact prediction dataframe.
    """
    df = pd.read_csv(input_csv)
    fmt = detect_csv_format(df)

    if fmt == "compact":
        print("Detected compact prediction CSV format.")
        return df

    if fmt == "fightcard":
        print("Detected full fight-card CSV format. Converting internally.")
        return convert_fightcard_df_to_compact(df)

    raise ValueError("Unsupported CSV format.")


# =========================================================
# Historical snapshots
# =========================================================

def latest_fighter_history_row(fighter_hist_rolled, fighter_name):
    d = fighter_hist_rolled[
        fighter_hist_rolled["fighter"].astype(str).str.strip() == str(fighter_name).strip()
    ].copy()

    if d.empty:
        return None

    d = d.sort_values("fight_index")
    return d.iloc[-1]


# Maps each rolling *_avg feature to the raw per-fight column it averages.
# Mirrors add_rolling_features() in preprocessor.py.
ROLLING_AVG_MAP = {
    "sig_per_sec_avg": "sig_per_sec",
    "sig_rate_avg": "sig_rate",
    "ctrl_pct_avg": "ctrl_pct",
    "td_rate_avg": "td_rate",
    "win_rate_avg": "result",
    "kd_per_sec_avg": "kd_per_sec",
    "ko_win_rate_avg": "win_ko",
    "sub_win_rate_avg": "win_sub",
}


def build_fighter_snapshot(fighter_hist_rolled, fighter_name, window=5):
    """
    Build a fighter's PRE-FIGHT feature snapshot for an upcoming (future) fight.

    The rolling features stored on each historical row use shift(1), so they
    describe the state *going into* that row's fight -- which means the last
    historical row excludes the fighter's most recent fight. A future fight
    needs the state *after* the latest fight (i.e. including it), so we recompute
    the snapshot here from the raw per-fight columns instead of reading the
    shifted columns off the last row.

    Returns a dict of the BASE_FEATURE_COLS values, or None if the fighter has
    no history.
    """
    d = fighter_hist_rolled[
        fighter_hist_rolled["fighter"].astype(str).str.strip() == str(fighter_name).strip()
    ].copy()

    if d.empty:
        return None

    d = d.sort_values("fight_index")
    last = d.iloc[-1]

    snap = {}

    # Rolling averages over the most recent `window` fights, INCLUDING the
    # latest one. Keep training's min_periods=window semantics: require at
    # least `window` valid observations, otherwise leave NaN for the imputer.
    for avg_col, raw_col in ROLLING_AVG_MAP.items():
        recent = pd.to_numeric(d[raw_col], errors="coerce").tail(window)
        if recent.notna().sum() < window:
            snap[avg_col] = np.nan
        else:
            snap[avg_col] = recent.mean()

    # Experience going into the next fight = total fights so far.
    snap["exp_prior"] = len(d)

    # Divisional experience: total prior fights in the fighter's latest weightclass.
    latest_wc = last["weightclass"]
    snap["exp_div_prior"] = int((d["weightclass"] == latest_wc).sum())

    # Streaks going into the next fight = streak state AFTER the latest fight,
    # i.e. the unshifted win_streak/loss_streak from the most recent row.
    snap["win_streak_prior"] = last["win_streak"]
    snap["loss_streak_prior"] = last["loss_streak"]

    return snap


def latest_fight_metadata(df_fight, fighter_name):
    fighter_name = str(fighter_name).strip()
    rows = []

    f1 = df_fight[df_fight["Fighter1"].astype(str).str.strip() == fighter_name].copy()
    for _, r in f1.iterrows():
        rows.append({
            "fight_index": r["fight_index"],
            "dob": r["f1_dob"],
            "height_in": r["f1_height_in"],
            "reach_in": r["f1_reach_in"],
            "stance": r["f1_stance_clean"],
            "elo_post": r["elo_post_f1"],
        })

    f2 = df_fight[df_fight["Fighter2"].astype(str).str.strip() == fighter_name].copy()
    for _, r in f2.iterrows():
        rows.append({
            "fight_index": r["fight_index"],
            "dob": r["f2_dob"],
            "height_in": r["f2_height_in"],
            "reach_in": r["f2_reach_in"],
            "stance": r["f2_stance_clean"],
            "elo_post": r["elo_post_f2"],
        })

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x["fight_index"])
    return rows[-1]


# =========================================================
# Metadata helpers
# =========================================================

def optional_meta_from_row(row, prefix):
    meta = {}

    dob_col = f"{prefix}_dob"
    height_col = f"{prefix}_height_in"
    reach_col = f"{prefix}_reach_in"
    stance_col = f"{prefix}_stance"

    if dob_col in row and pd.notna(row[dob_col]) and str(row[dob_col]).strip() != "":
        meta["dob"] = row[dob_col]

    if height_col in row and pd.notna(row[height_col]) and str(row[height_col]).strip() != "":
        meta["height_in"] = float(row[height_col])

    if reach_col in row and pd.notna(row[reach_col]) and str(row[reach_col]).strip() != "":
        meta["reach_in"] = float(row[reach_col])

    if stance_col in row and pd.notna(row[stance_col]) and str(row[stance_col]).strip() != "":
        meta["stance"] = row[stance_col]

    return meta if meta else None


# =========================================================
# Build future feature row
# =========================================================

def build_future_feature_row(bundle, fighter_a, fighter_b, event_date,
                             fighter_a_meta=None, fighter_b_meta=None, window=5):
    df_fight = bundle["df_fight"]
    fighter_hist_rolled = bundle["fighter_hist_rolled"]

    event_date = pd.to_datetime(event_date)

    # Pre-fight snapshots computed as of AFTER each fighter's most recent bout,
    # so they include that latest fight (see build_fighter_snapshot).
    snap_a = build_fighter_snapshot(fighter_hist_rolled, fighter_a, window=window)
    snap_b = build_fighter_snapshot(fighter_hist_rolled, fighter_b, window=window)

    if snap_a is None:
        raise ValueError(f"No fighter history found for '{fighter_a}'")
    if snap_b is None:
        raise ValueError(f"No fighter history found for '{fighter_b}'")

    meta_a = latest_fight_metadata(df_fight, fighter_a)
    meta_b = latest_fight_metadata(df_fight, fighter_b)

    if meta_a is None:
        meta_a = {
            "dob": pd.NaT,
            "height_in": np.nan,
            "reach_in": np.nan,
            "stance": "Unknown",
            "elo_post": 1500.0,
        }

    if meta_b is None:
        meta_b = {
            "dob": pd.NaT,
            "height_in": np.nan,
            "reach_in": np.nan,
            "stance": "Unknown",
            "elo_post": 1500.0,
        }

    if fighter_a_meta is not None:
        if "dob" in fighter_a_meta:
            meta_a["dob"] = pd.to_datetime(fighter_a_meta["dob"], errors="coerce")
        if "height_in" in fighter_a_meta:
            meta_a["height_in"] = fighter_a_meta["height_in"]
        if "reach_in" in fighter_a_meta:
            meta_a["reach_in"] = fighter_a_meta["reach_in"]
        if "stance" in fighter_a_meta:
            meta_a["stance"] = clean_stance(fighter_a_meta["stance"])

    if fighter_b_meta is not None:
        if "dob" in fighter_b_meta:
            meta_b["dob"] = pd.to_datetime(fighter_b_meta["dob"], errors="coerce")
        if "height_in" in fighter_b_meta:
            meta_b["height_in"] = fighter_b_meta["height_in"]
        if "reach_in" in fighter_b_meta:
            meta_b["reach_in"] = fighter_b_meta["reach_in"]
        if "stance" in fighter_b_meta:
            meta_b["stance"] = clean_stance(fighter_b_meta["stance"])

    # Must match current trained feature set
    base_feature_cols = [
        "sig_per_sec_avg",
        "sig_rate_avg",
        "ctrl_pct_avg",
        "td_rate_avg",
        "win_rate_avg",
        "kd_per_sec_avg",
        "exp_prior",
        "exp_div_prior",
        "ko_win_rate_avg",
        "sub_win_rate_avg",
        "win_streak_prior",
        "loss_streak_prior",
    ]

    feature_row = {}

    for col in base_feature_cols:
        feature_row[f"{col}_diff"] = snap_a.get(col, np.nan) - snap_b.get(col, np.nan)

    feature_row["elo_diff_pre"] = meta_a["elo_post"] - meta_b["elo_post"]

    age_a = (event_date - pd.to_datetime(meta_a["dob"])).days / 365.25 if pd.notna(meta_a["dob"]) else np.nan
    age_b = (event_date - pd.to_datetime(meta_b["dob"])).days / 365.25 if pd.notna(meta_b["dob"]) else np.nan
    feature_row["age_diff"] = age_a - age_b

    feature_row["height_diff"] = meta_a["height_in"] - meta_b["height_in"]
    feature_row["reach_diff"] = meta_a["reach_in"] - meta_b["reach_in"]

        # ---------------------------------
    # Style profile features for each fighter
    # ---------------------------------
    striker_index_a = (
        (0 if pd.isna(snap_a.get("sig_per_sec_avg")) else snap_a.get("sig_per_sec_avg"))
        + (0 if pd.isna(snap_a.get("sig_rate_avg")) else snap_a.get("sig_rate_avg"))
    )

    striker_index_b = (
        (0 if pd.isna(snap_b.get("sig_per_sec_avg")) else snap_b.get("sig_per_sec_avg"))
        + (0 if pd.isna(snap_b.get("sig_rate_avg")) else snap_b.get("sig_rate_avg"))
    )

    grappler_index_a = (
        (0 if pd.isna(snap_a.get("td_rate_avg")) else snap_a.get("td_rate_avg"))
        + (0 if pd.isna(snap_a.get("ctrl_pct_avg")) else snap_a.get("ctrl_pct_avg"))
        + (0 if pd.isna(snap_a.get("sub_win_rate_avg")) else snap_a.get("sub_win_rate_avg"))
    )

    grappler_index_b = (
        (0 if pd.isna(snap_b.get("td_rate_avg")) else snap_b.get("td_rate_avg"))
        + (0 if pd.isna(snap_b.get("ctrl_pct_avg")) else snap_b.get("ctrl_pct_avg"))
        + (0 if pd.isna(snap_b.get("sub_win_rate_avg")) else snap_b.get("sub_win_rate_avg"))
    )

    # feature_row["striker_vs_grappler_adv"] = striker_index_a - grappler_index_b
    # feature_row["grappler_vs_striker_adv"] = grappler_index_a - striker_index_b
    # feature_row["sub_threat_vs_opp"] = (
    #     (0 if pd.isna(snap_a.get("sub_win_rate_avg")) else snap_a.get("sub_win_rate_avg"))
    #     - (0 if pd.isna(snap_b.get("sig_rate_avg")) else snap_b.get("sig_rate_avg"))
    # )

    return pd.DataFrame([feature_row])


# =========================================================
# Single prediction
# =========================================================

def predict_future_fight(bundle, model, preprocessor, feature_cols,
                         fighter_a, fighter_b, event_date,
                         fighter_a_meta=None, fighter_b_meta=None, window=5):
    X_future = build_future_feature_row(
        bundle=bundle,
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        event_date=event_date,
        fighter_a_meta=fighter_a_meta,
        fighter_b_meta=fighter_b_meta,
        window=window,
    )

    raw_cols = list(X_future.columns)

    missing_before_reindex = [c for c in feature_cols if c not in raw_cols]
    extra_before_reindex = [c for c in raw_cols if c not in feature_cols]

    if missing_before_reindex:
        print(f"Warning: constructed future row is missing trained features: {missing_before_reindex}")

    if extra_before_reindex:
        print(f"Warning: extra constructed features ignored: {extra_before_reindex}")

    X_future = X_future.reindex(columns=feature_cols)

    nan_cols = X_future.columns[X_future.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: NaNs present in future row for columns: {nan_cols}")

    # preprocess original orientation: A vs B
    X_future_proc = preprocessor.transform(X_future)
    X_future_t = torch.tensor(X_future_proc, dtype=torch.float32)

    # preprocess mirrored orientation: B vs A
    X_future_mirror = -X_future
    X_future_mirror_proc = preprocessor.transform(X_future_mirror)
    X_future_mirror_t = torch.tensor(X_future_mirror_proc, dtype=torch.float32)

    with torch.no_grad():
        logit_ab = model(X_future_t).squeeze(1)
        p_ab = torch.sigmoid(logit_ab).item()

        logit_ba = model(X_future_mirror_t).squeeze(1)
        p_ba = torch.sigmoid(logit_ba).item()

    # symmetric average
    prob_a = (p_ab + (1.0 - p_ba)) / 2.0
    prob_b = 1.0 - prob_a

    predicted_winner = fighter_a if prob_a >= 0.5 else fighter_b

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "event_date": str(pd.to_datetime(event_date).date()),
        "prob_fighter_a_wins": prob_a,
        "prob_fighter_b_wins": prob_b,
        "predicted_winner": predicted_winner,
        "X_future": X_future,
    }

# =========================================================
# Batch prediction
# =========================================================

def predict_batch_from_csv(bundle, model, preprocessor, feature_cols,
                           input_csv, output_csv, window=5):
    fights_df = load_prediction_input(input_csv)

    required_cols = ["fighter_a", "fighter_b", "event_date"]
    missing_cols = [c for c in required_cols if c not in fights_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    results = []

    for _, row in fights_df.iterrows():
        fighter_a = row["fighter_a"]
        fighter_b = row["fighter_b"]
        event_date = row["event_date"]

        fighter_a_meta = optional_meta_from_row(row, "fighter_a")
        fighter_b_meta = optional_meta_from_row(row, "fighter_b")

        try:
            pred = predict_future_fight(
                bundle=bundle,
                model=model,
                preprocessor=preprocessor,
                feature_cols=feature_cols,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                event_date=event_date,
                fighter_a_meta=fighter_a_meta,
                fighter_b_meta=fighter_b_meta,
                window=window,
            )

            prob_a = pred["prob_fighter_a_wins"]
            prob_b = pred["prob_fighter_b_wins"]
            confidence = max(prob_a, prob_b)

            results.append({
                "event_date": pred["event_date"],
                "fight": f"{pred['fighter_a']} vs {pred['fighter_b']}",
                "predicted_winner": pred["predicted_winner"],
                "confidence": f"{confidence * 100:.1f}%"
            })

        except Exception as e:
            results.append({
                "event_date": str(event_date),
                "fight": f"{fighter_a} vs {fighter_b}",
                "predicted_winner": "ERROR",
                "confidence": "",
            })
            print(f"Prediction failed for {fighter_a} vs {fighter_b}: {e}")

    out_df = pd.DataFrame(results)
    out_df = out_df[["event_date", "fight", "predicted_winner", "confidence"]]
    out_df.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")
    print(out_df)

    return out_df


# =========================================================
# Main
# =========================================================

def main():
    model, preprocessor, feature_cols, config = load_artifacts()

    df_raw = pd.read_csv(DATA_PATH)

    bundle = run_preprocessing(
        df_raw=df_raw,
        base_elo=config["base_elo"],
        elo_k=config["elo_k"],
        rolling_window=config["rolling_window"],
    )

    predict_batch_from_csv(
        bundle=bundle,
        model=model,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        window=config["rolling_window"],
    )


if __name__ == "__main__":
    main()