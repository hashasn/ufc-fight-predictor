"""
Orchestration layer shared by the CLI.

Holds the loaded model + preprocessed history bundle in memory (built once per
session) and exposes the three high-level workflows: predict a single matchup,
predict scraped upcoming cards, and refresh data + retrain.
"""

import io
import shutil
import contextlib
from datetime import datetime

import pandas as pd

import config
import predict
import train
import scrape_results
import scrape_upcoming
from preprocessor import run_preprocessing

# In-memory caches (built lazily, rebuilt after a retrain).
_BUNDLE = None
_MODEL_ARTIFACTS = None  # (model, preprocessor, feature_cols, model_config)


# =========================================================
# Lazy loaders
# =========================================================

def get_bundle(progress=print):
    global _BUNDLE
    if _BUNDLE is None:
        progress("Loading fight history and building features (one-time)...")
        df_raw = pd.read_csv(config.DATA_PATH)
        cfg = config.load_training_config()
        _BUNDLE = run_preprocessing(
            df_raw=df_raw,
            base_elo=cfg["base_elo"],
            elo_k=cfg["elo_k"],
            rolling_window=cfg["rolling_window"],
        )
    return _BUNDLE


def get_model():
    global _MODEL_ARTIFACTS
    if _MODEL_ARTIFACTS is None:
        _MODEL_ARTIFACTS = predict.load_artifacts(config.ARTIFACT_DIR)
    return _MODEL_ARTIFACTS


def reload_all(progress=print):
    """Drop cached bundle/model so the next call rebuilds from disk."""
    global _BUNDLE, _MODEL_ARTIFACTS
    _BUNDLE = None
    _MODEL_ARTIFACTS = None
    get_bundle(progress=progress)
    get_model()


def known_fighters(bundle=None):
    bundle = bundle or get_bundle()
    return set(
        bundle["fighter_hist_rolled"]["fighter"].astype(str).str.strip().unique()
    )


# =========================================================
# Option 1: predict a single matchup (exact name match)
# =========================================================

def predict_two(name_a, name_b, event_date=None, progress=print):
    bundle = get_bundle(progress=progress)
    model, preprocessor, feature_cols, model_config = get_model()

    names = known_fighters(bundle)
    missing = [n for n in (name_a, name_b) if str(n).strip() not in names]
    if missing:
        raise ValueError(
            "Fighter(s) not found in history (exact match required): "
            + ", ".join(repr(m) for m in missing)
            + ". Check spelling/capitalization."
        )

    if event_date is None:
        event_date = datetime.today().strftime("%Y-%m-%d")

    return predict.predict_future_fight(
        bundle=bundle,
        model=model,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        fighter_a=str(name_a).strip(),
        fighter_b=str(name_b).strip(),
        event_date=event_date,
        window=model_config.get("rolling_window", 5),
    )


# =========================================================
# Option 2: predict scraped upcoming cards
# =========================================================

def _row_meta(row, prefix):
    """Build a predict_future_fight meta dict from a fightcard row."""
    meta = {}
    dob = row.get(f"{prefix}_dob")
    height = row.get(f"{prefix}_height")
    reach = row.get(f"{prefix}_reach")
    stance = row.get(f"{prefix}_stance")
    if pd.notna(dob):
        meta["dob"] = dob
    if pd.notna(height):
        meta["height_in"] = predict.height_to_inches(height)
    if pd.notna(reach):
        meta["reach_in"] = predict.reach_to_inches(reach)
    if pd.notna(stance):
        meta["stance"] = stance
    return meta or None


def predict_card_df(df_card, progress=print):
    """
    Predict every fight in a fightcard-schema DataFrame. Returns a results
    DataFrame with event/fight/predicted_winner/confidence (+ raw probs).
    Fighters without history are flagged rather than crashing the batch.
    """
    bundle = get_bundle(progress=progress)
    model, preprocessor, feature_cols, model_config = get_model()
    window = model_config.get("rolling_window", 5)
    names = known_fighters(bundle)

    results = []
    for _, row in df_card.iterrows():
        fa, fb = str(row["Fighter1"]).strip(), str(row["Fighter2"]).strip()
        base = {
            "event": row.get("event", ""),
            "event_date": row.get("event_date", ""),
            "fight": f"{fa} vs {fb}",
        }

        missing = [n for n in (fa, fb) if n not in names]
        if missing:
            results.append({**base, "predicted_winner": "N/A (no history)",
                            "confidence": "", "prob_a": None, "prob_b": None})
            continue

        try:
            # predict_future_fight prints per-fight NaN notices; silence them
            # in the batch view (sparse-history fighters are expected here).
            with contextlib.redirect_stdout(io.StringIO()):
                pred = predict.predict_future_fight(
                    bundle=bundle, model=model, preprocessor=preprocessor,
                    feature_cols=feature_cols,
                    fighter_a=fa, fighter_b=fb, event_date=row["event_date"],
                    fighter_a_meta=_row_meta(row, "f1"),
                    fighter_b_meta=_row_meta(row, "f2"),
                    window=window,
                )
            prob_a = pred["prob_fighter_a_wins"]
            prob_b = pred["prob_fighter_b_wins"]
            results.append({
                **base,
                "predicted_winner": pred["predicted_winner"],
                "confidence": f"{max(prob_a, prob_b) * 100:.1f}%",
                "prob_a": prob_a, "prob_b": prob_b,
            })
        except Exception as e:  # noqa: BLE001 - fail soft per fight
            progress(f"  prediction failed for {fa} vs {fb}: {e}")
            results.append({**base, "predicted_winner": "ERROR",
                            "confidence": "", "prob_a": None, "prob_b": None})

    return pd.DataFrame(results)


def predict_upcoming(n=2, force_refresh=False, progress=print):
    df_card = scrape_upcoming.get_next_cards(
        n=n, force_refresh=force_refresh, progress=progress
    )
    if df_card.empty:
        return df_card
    return predict_card_df(df_card, progress=progress)


# =========================================================
# Option 3: refresh data + retrain
# =========================================================

def _backup_before_retrain(progress=print):
    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_dir = f"{config.BACKUP_DIR}/{stamp}"
    import os
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(config.DATA_PATH, f"{dst_dir}/mayberealdata_enriched.csv")
    if os.path.isdir(config.ARTIFACT_DIR):
        shutil.copytree(config.ARTIFACT_DIR, f"{dst_dir}/artifacts", dirs_exist_ok=True)
    progress(f"Backed up dataset + artifacts to {dst_dir}")
    return dst_dir


def refresh_and_retrain(max_new_events=3, progress=print):
    """
    Scrape recently completed events, append to the master CSV (deduped), and
    retrain if anything new was added. Returns a summary dict.
    """
    _backup_before_retrain(progress=progress)

    master = pd.read_csv(config.DATA_PATH)
    rows_before = len(master)

    progress("Fetching newly completed fights...")
    new_rows = scrape_results.fetch_new(
        master, max_new_events=max_new_events, progress=progress
    )
    added = scrape_results.append_dedup(config.DATA_PATH, new_rows)
    progress(f"Added {added} new fight rows (dataset: {rows_before} -> {rows_before + added}).")

    if added == 0:
        progress("No new data; skipping retrain.")
        return {"added": 0, "retrained": False, "rows": rows_before}

    progress("Retraining model...")
    train.main(data_path=config.DATA_PATH, artifact_dir=config.ARTIFACT_DIR)

    progress("Reloading model + features...")
    reload_all(progress=progress)

    return {"added": added, "retrained": True, "rows": rows_before + added}
