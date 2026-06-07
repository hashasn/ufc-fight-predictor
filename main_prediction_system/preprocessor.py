import re
import numpy as np
import pandas as pd


# =========================================================
# Basic parsing helpers
# =========================================================

def mmss_to_seconds(value):
    """
    Convert a time string like '4:32' into total seconds.
    Returns np.nan if the value is missing or invalid.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    match = re.match(r"^(\d+):(\d{2})$", value)
    if not match:
        return np.nan

    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return minutes * 60 + seconds


def height_to_inches(h):
    """
    Convert a UFC height string like 5' 11" into inches.
    Returns np.nan if missing or invalid.
    """
    if pd.isna(h):
        return np.nan

    h = str(h).strip()
    match = re.search(r"(\d+)'\s*(\d+)", h)
    if not match:
        return np.nan

    feet = int(match.group(1))
    inches = int(match.group(2))
    return feet * 12 + inches


def reach_to_inches(r):
    """
    Convert a UFC reach string like 72" into numeric inches.
    Returns np.nan if missing or invalid.
    """
    if pd.isna(r):
        return np.nan

    r = str(r).strip().replace('"', '')
    if r in ["--", "", "nan", "None"]:
        return np.nan

    digits = "".join(c for c in r if c.isdigit())
    return float(digits) if digits else np.nan


def clean_stance(s):
    """
    Normalize stance values into a small set of categories.
    """
    if pd.isna(s):
        return "Unknown"

    s = str(s).strip().lower()

    if "southpaw" in s:
        return "Southpaw"
    if "orthodox" in s:
        return "Orthodox"
    if "switch" in s:
        return "Switch"

    return "Unknown"


def parse_of(value):
    """
    Convert a stat string like '45 of 102' into:
    landed, attempted, rate

    Returns:
        (landed, attempted, rate)
    or:
        (np.nan, np.nan, np.nan)
    """
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    value = str(value).strip()
    if "of" not in value:
        return np.nan, np.nan, np.nan

    try:
        landed, attempted = value.split("of")
        landed = int(landed.strip())
        attempted = int(attempted.strip())
        rate = landed / attempted if attempted > 0 else 0.0
        return landed, attempted, rate
    except Exception:
        return np.nan, np.nan, np.nan


def get_result(winner, fighter_name):
    """
    Return result from one fighter's perspective.

    1   -> fighter won
    0   -> fighter lost
    NaN -> draw / no contest / unknown
    """
    if pd.isna(winner):
        return np.nan

    w = str(winner).strip().upper()
    if w in ["DRAW", "NO CONTEST", "NC"]:
        return np.nan

    return 1 if str(winner).strip() == str(fighter_name).strip() else 0


def method_group(m):
    """
    Collapse detailed method labels into broad groups.
    """
    m = "" if pd.isna(m) else str(m).strip().lower()

    if "ko/tko" in m:
        return "ko"
    if "submission" in m:
        return "sub"
    if "decision" in m:
        return "dec"

    return "other"


# =========================================================
# Elo helpers
# =========================================================

def elo_expected(r_a, r_b, scale=400):
    """
    Standard Elo expected score formula.
    """
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / scale))


def build_elo_features(df, base_elo=1500, k=16, scale=400):
    """
    Compute Elo ratings over time and add them to the fight-level dataframe.

    Adds:
        elo_pre_f1
        elo_pre_f2
        elo_post_f1
        elo_post_f2
        elo_diff_pre
        s1
    """
    d = df.sort_values("fight_index").reset_index(drop=True).copy()

    elo = {}
    pre1, pre2, post1, post2, s1_list = [], [], [], [], []

    for _, row in d.iterrows():
        f1 = str(row["Fighter1"]).strip()
        f2 = str(row["Fighter2"]).strip()
        winner = "" if pd.isna(row["Winner"]) else str(row["Winner"]).strip()

        r1 = elo.get(f1, base_elo)
        r2 = elo.get(f2, base_elo)

        pre1.append(r1)
        pre2.append(r2)

        w_up = winner.upper()

        if w_up in ["NO CONTEST", "NC", ""]:
            s1 = np.nan
            s1_list.append(s1)
            post1.append(r1)
            post2.append(r2)
            continue

        if w_up == "DRAW":
            s1 = 0.5
        elif winner == f1:
            s1 = 1.0
        elif winner == f2:
            s1 = 0.0
        else:
            s1 = np.nan

        s1_list.append(s1)

        if np.isnan(s1):
            post1.append(r1)
            post2.append(r2)
            continue

        e1 = elo_expected(r1, r2, scale=scale)

        r1_new = r1 + k * (s1 - e1)
        r2_new = r2 + k * ((1.0 - s1) - (1.0 - e1))

        elo[f1] = r1_new
        elo[f2] = r2_new

        post1.append(r1_new)
        post2.append(r2_new)

    d["elo_pre_f1"] = pre1
    d["elo_pre_f2"] = pre2
    d["elo_post_f1"] = post1
    d["elo_post_f2"] = post2
    d["elo_diff_pre"] = d["elo_pre_f1"] - d["elo_pre_f2"]
    d["s1"] = s1_list

    return d


# =========================================================
# Fight-level cleaning
# =========================================================

def clean_fight_level(df_raw):
    """
    Clean the raw enriched fight-level dataframe.

    Expected columns include:
        Event, Fighter1, Fighter2, Winner, Weightclass, Method,
        Round, Time, event_date,
        f1_dob, f2_dob,
        f1_height, f2_height,
        f1_reach, f2_reach,
        f1_stance, f2_stance

    Returns:
        Cleaned fight-level dataframe
    """
    df = df_raw.copy()

    # Convert time and round into total fight duration
    df["round_time_seconds"] = df["Time"].apply(mmss_to_seconds)
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["total_fight_seconds"] = (df["Round"] - 1) * 5 * 60 + df["round_time_seconds"]

    # Convert date columns
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["f1_dob"] = pd.to_datetime(df["f1_dob"], errors="coerce")
    df["f2_dob"] = pd.to_datetime(df["f2_dob"], errors="coerce")

    # Age features
    df["f1_age"] = (df["event_date"] - df["f1_dob"]).dt.days / 365.25
    df["f2_age"] = (df["event_date"] - df["f2_dob"]).dt.days / 365.25
    df["age_diff"] = df["f1_age"] - df["f2_age"]

    # Height features
    df["f1_height_in"] = df["f1_height"].apply(height_to_inches)
    df["f2_height_in"] = df["f2_height"].apply(height_to_inches)
    df["height_diff"] = df["f1_height_in"] - df["f2_height_in"]

    # Reach features
    df["f1_reach_in"] = df["f1_reach"].apply(reach_to_inches)
    df["f2_reach_in"] = df["f2_reach"].apply(reach_to_inches)
    df["reach_diff"] = df["f1_reach_in"] - df["f2_reach_in"]

    # Stance cleanup
    df["f1_stance_clean"] = df["f1_stance"].apply(clean_stance)
    df["f2_stance_clean"] = df["f2_stance"].apply(clean_stance)

    # Sort from oldest to newest
    df = df.sort_values("event_date").reset_index(drop=True)

    # Chronological fight index
    df["fight_index"] = np.arange(len(df))

    return df


# =========================================================
# Fight-level -> fighter-history
# =========================================================

def build_fighter_history(df):
    """
    Convert each fight into two fighter-perspective rows.

    Returns a dataframe where each row represents:
        one fighter, one fight, one opponent
    """
    rows = []

    for _, r in df.iterrows():
        f1 = r["Fighter1"]
        f2 = r["Fighter2"]
        m = r["Method"]

        rows.append({
            "fight_index": r["fight_index"],
            "event_date": r["event_date"],
            "event": r["Event"],
            "weightclass": r["Weightclass"],
            "method": m,
            "fighter": f1,
            "opponent": f2,
            "result": get_result(r["Winner"], f1),
            "total_fight_seconds": r["total_fight_seconds"],
            "kd": r["KD1"],
            "sig_str": r["SIG_STR1"],
            "td": r["TD1"],
            "sub_att": r["SUB_ATT1"],
            "rev": r["REV1"],
            "ctrl": r["CTRL1"],
            "stance": r["f1_stance_clean"],
            "opp_stance": r["f2_stance_clean"],
            "opp_elo_pre": r["elo_pre_f2"],
        })

        rows.append({
            "fight_index": r["fight_index"],
            "event_date": r["event_date"],
            "event": r["Event"],
            "weightclass": r["Weightclass"],
            "method": m,
            "fighter": f2,
            "opponent": f1,
            "result": get_result(r["Winner"], f2),
            "total_fight_seconds": r["total_fight_seconds"],
            "kd": r["KD2"],
            "sig_str": r["SIG_STR2"],
            "td": r["TD2"],
            "sub_att": r["SUB_ATT2"],
            "rev": r["REV2"],
            "ctrl": r["CTRL2"],
            "stance": r["f2_stance_clean"],
            "opp_stance": r["f1_stance_clean"],
            "opp_elo_pre": r["elo_pre_f1"],
        })

    fighter_hist = pd.DataFrame(rows)
    fighter_hist = fighter_hist.sort_values(["fighter", "fight_index"]).reset_index(drop=True)

    # Parse significant strikes
    fighter_hist[["sig_landed", "sig_attempted", "sig_rate"]] = (
        fighter_hist["sig_str"].apply(parse_of).apply(pd.Series)
    )

    # Parse takedowns
    fighter_hist[["td_landed", "td_attempted", "td_rate"]] = (
        fighter_hist["td"].apply(parse_of).apply(pd.Series)
    )

    # Other derived numeric features
    fighter_hist["ctrl_seconds"] = fighter_hist["ctrl"].apply(mmss_to_seconds)
    fighter_hist["kd"] = pd.to_numeric(fighter_hist["kd"], errors="coerce")

    # Per-second / proportion stats
    fighter_hist["sig_per_sec"] = fighter_hist["sig_landed"] / fighter_hist["total_fight_seconds"]
    fighter_hist["ctrl_pct"] = fighter_hist["ctrl_seconds"] / fighter_hist["total_fight_seconds"]
    fighter_hist["kd_per_sec"] = fighter_hist["kd"] / fighter_hist["total_fight_seconds"]

    # Method group features
    fighter_hist["method_group"] = fighter_hist["method"].apply(method_group)
    fighter_hist["win_ko"] = (
        (fighter_hist["result"] == 1) & (fighter_hist["method_group"] == "ko")
    ).astype(int)
    fighter_hist["win_sub"] = (
        (fighter_hist["result"] == 1) & (fighter_hist["method_group"] == "sub")
    ).astype(int)

    return fighter_hist


# =========================================================
# Rolling pre-fight features
# =========================================================

def add_rolling_features(fighter_hist, window=5):
    """
    Add rolling pre-fight features using only PRIOR fights.

    shift(1) is critical because it prevents the current fight
    from leaking into its own features.
    """
    d = fighter_hist.copy()
    d = d.sort_values(["fighter", "fight_index"]).reset_index(drop=True)

    rolling_cols = ["sig_per_sec", "sig_rate", "ctrl_pct", "td_rate"]

    for col in rolling_cols:
        d[f"{col}_avg"] = (
            d.groupby("fighter")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
        )

    d["win_rate_avg"] = (
        d.groupby("fighter")["result"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    d["kd_per_sec_avg"] = (
        d.groupby("fighter")["kd_per_sec"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    d["opp_elo_avg_5"] = (
        d.groupby("fighter")["opp_elo_pre"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=5).mean())
    )

    d["opp_elo_avg_3"] = (
        d.groupby("fighter")["opp_elo_pre"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=3).mean())
    )

    d["ko_win_rate_avg"] = (
        d.groupby("fighter")["win_ko"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    d["sub_win_rate_avg"] = (
        d.groupby("fighter")["win_sub"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    )

    # Experience features
    d["exp_prior"] = d.groupby("fighter").cumcount()
    d["exp_div_prior"] = d.groupby(["fighter", "weightclass"]).cumcount()

    # Layoff features using actual date gaps
    d["prev_event_date"] = d.groupby("fighter")["event_date"].shift(1)
    d["layoff_days"] = (d["event_date"] - d["prev_event_date"]).dt.days
    d["layoff_log"] = np.log1p(d["layoff_days"])
    d["layoff_big"] = (d["layoff_days"] >= 365).astype(int)
    # =========================================================
    
    # Win streak / loss streak features
    # =========================================================

    # Convert results to numeric for streak logic
    d["result_clean"] = d["result"].fillna(0)

    # Win streak
    d["win_streak"] = (
        d.groupby("fighter")["result_clean"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    )

    d.loc[d["result_clean"] != 1, "win_streak"] = 0

    # Loss streak
    d["loss_flag"] = (d["result_clean"] == 0).astype(int)

    d["loss_streak"] = (
        d.groupby("fighter")["loss_flag"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    )

    d.loc[d["loss_flag"] != 1, "loss_streak"] = 0

    # Only use PRIOR fights
    d["win_streak_prior"] = d.groupby("fighter")["win_streak"].shift(1)
    d["loss_streak_prior"] = d.groupby("fighter")["loss_streak"].shift(1)

    

    # =========================================================
    # Performance trajectory (trend over last fights)
    # =========================================================

    def rolling_trend(series, window=5):
        """
        Compute linear trend slope over a rolling window.
        Positive slope = improving.
        Negative slope = declining.
        """
        if series.isna().all():
            return np.nan

        x = np.arange(len(series))
        y = series.values

        if np.isnan(y).any():
            return np.nan

        slope = np.polyfit(x, y, 1)[0]
        return slope


    d["sig_rate_trend"] = (
        d.groupby("fighter")["sig_rate"]
        .transform(lambda s: s.shift(1).rolling(window=5).apply(rolling_trend, raw=False))
    )

    d["td_rate_trend"] = (
        d.groupby("fighter")["td_rate"]
        .transform(lambda s: s.shift(1).rolling(window=5).apply(rolling_trend, raw=False))
    )

    d["win_rate_trend"] = (
        d.groupby("fighter")["result"]
        .transform(lambda s: s.shift(1).rolling(window=5).apply(rolling_trend, raw=False))
    )


    # Activity features: number of prior fights in trailing time windows
    def count_recent_fights(group, days):
        event_dates = group["event_date"]
        counts = []

        for current_date in event_dates:
            if pd.isna(current_date):
                counts.append(np.nan)
                continue

            window_start = current_date - pd.Timedelta(days=days)

            # count only PRIOR fights, not current one
            prior_mask = (event_dates < current_date) & (event_dates >= window_start)
            counts.append(prior_mask.sum())

        return pd.Series(counts, index=group.index)

    d["fights_last_365_days"] = (
        d.groupby("fighter", group_keys=False)
        .apply(lambda g: count_recent_fights(g, 365), include_groups=False)
    )

    d["fights_last_730_days"] = (
        d.groupby("fighter", group_keys=False)
        .apply(lambda g: count_recent_fights(g, 730), include_groups=False)
    )


    d["total_fights_prior"] = (
        d.groupby("fighter").cumcount()
    )

        # ---------------------------------
    # Style profile features
    # ---------------------------------
    # Striking profile
    d["striker_index"] = (
        d["sig_per_sec_avg"].fillna(0)
        + d["sig_rate_avg"].fillna(0)
    )

    # Grappling profile
    d["grappler_index"] = (
        d["td_rate_avg"].fillna(0)
        + d["ctrl_pct_avg"].fillna(0)
        + d["sub_win_rate_avg"].fillna(0)
    )

    return d


# =========================================================
# Matchup dataset creation
# =========================================================

BASE_FEATURE_COLS = [
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
    # "sig_rate_trend",
    # "td_rate_trend",
    # "win_rate_trend",
    # "fights_last_365_days",
    # "total_fights_prior",
   
 
]

FINAL_FEATURE_COLS = [f"{c}_diff" for c in BASE_FEATURE_COLS] + [
    "elo_diff_pre",
    "age_diff",
    "height_diff",
    "reach_diff",
    # "striker_vs_grappler_adv",
    # "grappler_vs_striker_adv",
    # "sub_threat_vs_opp",
 
  
    # "same_stance",
    # "southpaw_vs_ortho",
    # "ortho_vs_southpaw",
    # "switch_diff",
]


def build_matchup_dataset(df_fight_level, fighter_hist_rolled):
    """
    Build the final matchup-level dataset for modeling.

    Returns:
        paired, X, y, feature_cols
    """
    model_df = fighter_hist_rolled.dropna(subset=BASE_FEATURE_COLS + ["result"]).copy()
    model_df["fight_id"] = model_df["fight_index"]

    # Keep only fights where both fighters have valid rows
    counts = model_df.groupby("fight_id").size()
    valid_fights = counts[counts == 2].index
    model_df = model_df[model_df["fight_id"].isin(valid_fights)].copy()

    # Self-merge to pair each fighter with the other side of the same fight
    paired = model_df.merge(model_df, on="fight_id", suffixes=("_f", "_o"))
    paired = paired[paired["fighter_f"] != paired["fighter_o"]].reset_index(drop=True)
    # ---------------------------------
    # Style matchup interaction features
    # ---------------------------------
    paired["striker_vs_grappler_adv"] = (
        paired["striker_index_f"] - paired["grappler_index_o"]
    )

    paired["grappler_vs_striker_adv"] = (
        paired["grappler_index_f"] - paired["striker_index_o"]
    )

    paired["sub_threat_vs_opp"] = (
        paired["sub_win_rate_avg_f"] - paired["sig_rate_avg_o"]
    )

    # Directional fight-level differences
    fight_level_long = pd.concat([
        df_fight_level[
            ["fight_index", "Fighter1", "Fighter2", "age_diff", "height_diff", "reach_diff", "elo_diff_pre"]
        ].rename(columns={
            "fight_index": "fight_id",
            "Fighter1": "fighter_f",
            "Fighter2": "fighter_o"
        }),

        df_fight_level[
            ["fight_index", "Fighter2", "Fighter1", "age_diff", "height_diff", "reach_diff", "elo_diff_pre"]
        ].rename(columns={
            "fight_index": "fight_id",
            "Fighter2": "fighter_f",
            "Fighter1": "fighter_o"
        }).assign(
            age_diff=lambda x: -x["age_diff"],
            height_diff=lambda x: -x["height_diff"],
            reach_diff=lambda x: -x["reach_diff"],
            elo_diff_pre=lambda x: -x["elo_diff_pre"]
        )
    ], ignore_index=True)

    paired = paired.merge(
        fight_level_long,
        on=["fight_id", "fighter_f", "fighter_o"],
        how="left"
    )

    # Stance matchup features
    paired["same_stance"] = (paired["stance_f"] == paired["stance_o"]).astype(int)
    paired["southpaw_vs_ortho"] = (
        ((paired["stance_f"] == "Southpaw") & (paired["stance_o"] == "Orthodox"))
    ).astype(int)
    paired["ortho_vs_southpaw"] = (
        ((paired["stance_f"] == "Orthodox") & (paired["stance_o"] == "Southpaw"))
    ).astype(int)
    paired["switch_f"] = (paired["stance_f"] == "Switch").astype(int)
    paired["switch_o"] = (paired["stance_o"] == "Switch").astype(int)
    paired["switch_diff"] = paired["switch_f"] - paired["switch_o"]

    # Fighter minus opponent feature differences
    for col in BASE_FEATURE_COLS:
        paired[f"{col}_diff"] = paired[f"{col}_f"] - paired[f"{col}_o"]

    X = paired[FINAL_FEATURE_COLS].copy()
    y = paired["result_f"].copy()

    return paired, X, y, FINAL_FEATURE_COLS


# =========================================================
# Main preprocessing entry point
# =========================================================

def run_preprocessing(df_raw, base_elo=1500, elo_k=16, rolling_window=5):
    """
    Full preprocessing pipeline.

    Steps:
        1. Clean fight-level data
        2. Add Elo features
        3. Build fighter history
        4. Add rolling pre-fight features
        5. Build matchup dataset

    Returns a dictionary of all useful intermediate outputs.
    """
    df_fight = clean_fight_level(df_raw)
    df_fight = build_elo_features(df_fight, base_elo=base_elo, k=elo_k)
    fighter_hist = build_fighter_history(df_fight)
    fighter_hist_rolled = add_rolling_features(fighter_hist, window=rolling_window)
    paired, X, y, feature_cols = build_matchup_dataset(df_fight, fighter_hist_rolled)

    return {
        "df_fight": df_fight,
        "fighter_hist": fighter_hist,
        "fighter_hist_rolled": fighter_hist_rolled,
        "paired": paired,
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
    }